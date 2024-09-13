import sys
sys.path.append('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
plt.rcParams["animation.html"] = "jshtml"
from torch.optim import Adam
from functools import partial
from tqdm import trange, tqdm

################################
##### Data Preprosessing #######
################################
# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

################################
######### SDE setup ############
################################

# Set up VE SDE for diffusion process
def marginal_prob_std(t, sigma, device_):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
      device_: The device to use.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=device_)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device_):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
      device_: The device to use.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=device_)

################################
######## SGD Model setup #######
################################

# Diffusion process time step encoding
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# Dense layer for encoding time steps
class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None, None]

# 3D Fourier layer
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, modes3, width, embed_dim=256):
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 14  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        self.fc0_w = nn.Linear(4, self.width)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)
        # self.trans_conv1 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        # self.trans_conv2 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        # self.trans_conv3 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        # self.trans_conv4 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))

        self.dense0 = Dense(embed_dim, self.width)
        self.dense1 = Dense(embed_dim, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, time_frame, x, w):
        grid = self.get_grid(x.shape, x.device, time_frame, sparse=False)

        x = torch.cat((x, grid), dim=-1)
        w = torch.cat((w, grid), dim=-1)

        x = x.float()

        x = self.fc0(x)
        w = self.fc0_w(w)

        x = x.permute(0, 4, 1, 2, 3)
        w = w.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        w = F.pad(w, [0, self.padding])  # pad the domain if input is non-periodic


        embed = self.embed(t)
        t_embed = self.dense0(embed)

        x1 = self.conv0(x)
        # w11 = self.conv1(w)
        x2 = self.w0(x)
        w11= self.w1(w)
        # x3 = self.trans_conv1(x_sparse)
        x = x1 + x2 + t_embed + w11
        # x = self.bn0(x)
        x = F.gelu(x)

        x1 = self.conv1(x)
        # w11 = self.conv2(w)
        x2 = self.w1(x)
        w11 = self.w3(w)
        # x3 = self.trans_conv2(x_sparse)
        x = x1 + x2 + t_embed + w11
        # x = F.gelu(x)
        #
        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x3 = self.dense2(embed)
        # x = x1 + x2 + x3
        # x = F.gelu(x)
        #
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x3 = self.dense3(embed)
        # x = x1 + x2 + x3
        #
        # w1 = self.conv2(w)
        # w2 = self.w2(w)
        # w = w1+ w2 + t_embed
        # w = F.gelu(w)
        #
        # w1 = self.conv3(w)
        # w2 = self.w3(w)
        # w = w1 + w2 + t_embed


        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        score = x / self.marginal_prob_std(t)[:, None, None, None, None]
        return score

    def get_grid(self, shape, device, time_frame, sparse=False):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        if not sparse:
            # print(time_frame.size())
            # print(size_z)
            # print(time_frame[0].cpu().numpy(), time_frame[-1].cpu().numpy())
            # gridz = torch.tensor(np.linspace(time_frame[0].cpu().numpy(), time_frame[-1].cpu().numpy(), size_z), dtype=torch.float)
            gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        elif sparse:
            gridz = torch.tensor(time_frame[-1].cpu().numpy())
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# Loss function
def loss_fn(model, x, w, time_frame, marginal_prob_std, eps=1e-5):
  """
  The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data, diffusion values here
    w: A mini-batch of conditions, velocity values here
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_target = x + z * std[:, None, None, None, None]
  # combined = torch.cat([w, perturbed_target], dim=-1)
  score = model(random_t, time_frame, perturbed_target, w)
  real_score = -z/std[:, None, None, None, None]

  loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1, 2, 3, 4)))

  return loss, score, real_score


################################
########### Sampling ###########
################################

#Naive sampling: Euler-Maruyama sampler
def Euler_Maruyama_sampler(data,
                           condition,
                           score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size,
                           spatial_dim,
                           sub_dim,
                           num_steps,
                           time_frame,
                           device,
                           eps):

    t = torch.ones(batch_size, device=device)

    # Start with a random initial condition (noise).
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, 1, 1, device=device) * marginal_prob_std(t)[:, None, None, None, None]
    # init_x = data[:batch_size, :, :, :, :] * marginal_prob_std(t)[:, None, None, None, None]

    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x_target = init_x
    # combined = torch.cat([condition, x_target], dim=-1)

    mean_x_target = x_target
    # errors, errors_2, real_score_ls, modeled_score_ls = [], [], [], []

    with (torch.no_grad()):
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            perturbation = (x_target-data[:batch_size, :, :, :, :]) / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            # real_score = -perturbation / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            modeled_score = score_model(batch_time_step, time_frame, x_target, condition)

            # rel_err = torch.mean(torch.norm(modeled_score - real_score, 2, dim=(1, 2))
            #                        / torch.norm(real_score, 2, dim=(1, 2)))

            # loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z) ** 2, dim=(1, 2, 3, 4)))
            # rel_err_2 = torch.mean(torch.sum((modeled_score * marginal_prob_std(batch_time_step)[:, None, None, None, None] + perturbation) ** 2, dim=(1, 2, 3, 4)))
            # rel_err_2 = torch.mean(torch.sum(abs(modeled_score - real_score) / abs(real_score), dim=(1, 2, 3, 4)))

            # errors.append(rel_err.item())
            # errors_2.append(rel_err_2.item())

            mean_x_target = x_target + (g ** 2)[:, None, None, None, None] * modeled_score * step_size
            x_target = mean_x_target + torch.sqrt(step_size) * g[:, None, None, None, None] * torch.randn_like(x_target)
            # x_target = mean_x_target

            # combined = torch.cat([condition, x_target], dim=-1)
            print('time_step:', time_step)

    # Do not include any noise in the last sampling step.
    # full_x = torch.cat([x_cond, mean_x_target], dim=-1)
    return mean_x_target

##############################
#######  Data Loading ########
##############################

# Load data
device = torch.device('cuda')
data = scipy.io.loadmat('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Data_Generation\\2d_ns_diffusion.mat')

sol_t, sol_u, sol_v, sol, forcing, diffusion, nonlinear = data['t'], data['sol_u'], data['sol_v'], data['sol'], data['forcing'], data['diffusion'], data['nonlinear']
sol_t = torch.from_numpy(sol_t).to(device)
sol_u = torch.from_numpy(sol_u).to(device)
sol_v = torch.from_numpy(sol_v).to(device)
sol = torch.from_numpy(sol).to(device)
forcing = torch.from_numpy(forcing).to(device)
diffusion = torch.from_numpy(diffusion).to(device)
nonlinear = torch.from_numpy(nonlinear).to(device)

# norm of forcing and diffusion
forcing_norm = torch.mean(torch.abs(forcing))
diffusion_norm = torch.mean(torch.abs(diffusion))

print('forcing norm:', forcing_norm)
print('diffusion norm:', diffusion_norm)

source = diffusion + 0.01 * forcing

batch_indices = torch.randint(0, 1100, (1000,))
test_batch_indices = torch.randint(0, 1100, (100,))

time_step_indices = torch.randint(0, 100, (1000,))
test_time_step_indices = torch.randint(0, 100, (100,))

train_source = torch.stack([source[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)])
test_source = torch.stack([source[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(test_batch_indices, test_time_step_indices)])
train_vorticity = torch.stack([sol[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)])
test_vorticity = torch.stack([sol[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(test_batch_indices, test_time_step_indices)])
train_time_frame = torch.stack([sol_t[0][time_step_idx] for time_step_idx in time_step_indices])
test_time_frame = torch.stack([sol_t[0][time_step_idx] for time_step_idx in test_time_step_indices])

# Spatial Resolution
s = 64
sub = 1

shifter = 5

# Train/Test
Ntrain = 1000
Ntest = 100
TCond = 1
#
# train_source = source[:Ntrain, ::sub, ::sub, shifter:TCond+shifter]
# test_source = source[-Ntest:, ::sub, ::sub, shifter:TCond+shifter]
# train_vorticity = sol[:Ntrain, ::sub, ::sub, shifter:TCond+shifter]
# test_vorticity = sol[-Ntest:, ::sub, ::sub, shifter:TCond+shifter]



# normalization, gaussian
# train_source_normalizer = GaussianNormalizer(train_source)
# test_source_normalizer = GaussianNormalizer(test_source)
# train_source = train_source_normalizer.encode(train_source)
# test_source = test_source_normalizer.encode(test_source)
#
# train_vorticity_normalizer = GaussianNormalizer(train_vorticity)
# test_vorticity_normalizer = GaussianNormalizer(test_vorticity)
# train_vorticity = train_vorticity_normalizer.encode(train_vorticity)
# test_vorticity = test_vorticity_normalizer.encode(test_vorticity)

train_source = train_source.reshape(Ntrain, s, s, 1, train_source.size()[-1]).repeat([1, 1, 1, 1, 1])
test_source = test_source.reshape(Ntest, s, s, 1, test_source.size()[-1]).repeat([1, 1, 1, 1, 1])
train_vorticity = train_vorticity.reshape(Ntrain, s, s, 1, train_vorticity.size()[-1]).repeat([1, 1, 1, 1, 1])
test_vorticity = test_vorticity.reshape(Ntest, s, s, 1, test_vorticity.size()[-1]).repeat([1, 1, 1, 1, 1])
train_time_frame = train_time_frame.reshape(Ntrain, 1)
test_time_frame = test_time_frame.reshape(Ntest, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_source, train_vorticity, train_time_frame), batch_size=10, shuffle=True)


################################
######## Model Training ########
################################
sigma = 1000
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

model = FNO3d(marginal_prob_std_fn, modes, modes, modes, width).cuda()

optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)

loss_history = []
rel_err_history = []

for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    rel_err = []
    for x, w, t in train_loader:
        # x = x[0].cuda()
        x, w, t = x.cuda(), w.cuda(), t.cuda()
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, w, t, marginal_prob_std_fn)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        relative_loss = torch.mean(torch.norm(score - real_score, 2, dim=(1, 2))
                                   / torch.norm(real_score, 2, dim=(1, 2)))
        rel_err.append(relative_loss.item())
    scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    relative_loss_epoch = np.mean(rel_err)
    loss_history.append(avg_loss_epoch)
    rel_err_history.append(relative_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
torch.save(model.state_dict(), 'DiffusionModel_unnorm.pth')

################################
##########  Sampling ###########
################################
# define and load model
sigma = 1000
modes = 8
width = 20

marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)
model = FNO3d(marginal_prob_std_fn, modes, modes, modes, width).cuda()

ckpt = torch.load('C:\\UWMadisonResearch\\ConditionalScoreFNO\\DiffusionModel_unnorm.pth', map_location=device)
model.load_state_dict(ckpt)

num_steps = 1000
sample_batch_size = 100
sampler = Euler_Maruyama_sampler

samples = sampler(test_source,
                test_vorticity,
                model,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                sample_batch_size,
                s,
                sub,
                num_steps,
                test_time_frame,
                device,
                1e-3)


fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    axs[0, i].contourf(test_source[i, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
    axs[0, i].set_title('Truth {}'.format(i))
    axs[1, i].contourf(samples[i, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
    axs[1, i].set_title('Generated {}'.format(i))

plt.savefig('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Plots\\ModelWithoutSparse.png', dpi=300, bbox_inches='tight')

# plot one batch of samples and the corresponding target
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
axs[0].contourf(samples[5, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
axs[0].set_title('Sample')
axs[1].contourf(test_source[5, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
axs[1].set_title('Target')
plt.show()

mse = torch.mean((samples - test_source[:sample_batch_size, :, :, :, :])**2)
rel_mse = (torch.mean(torch.norm(samples - test_source[:sample_batch_size, :, :, :, :], 2, dim=(1, 2))
                    / torch.norm(test_source[:sample_batch_size, :, :, :, :], 2, dim=(1, 2))))

mse = torch.mean((samples - train_source[:sample_batch_size, :, :, :, :])**2)
rel_mse = (torch.mean(torch.norm(samples - train_source[:sample_batch_size, :, :, :, :], 2, dim=(1, 2))
                    / torch.norm(train_source[:sample_batch_size, :, :, :, :], 2, dim=(1, 2))))




# Euler scheme update on vorticity
import math

TCond = 10
shifter = 1
vorticity_truth = sol[:100, :, :, shifter:TCond+shifter]
vorticity_truth = vorticity_truth.reshape(100, s, s, 1, TCond)

nonlinear_truth = nonlinear[:100, :, :, shifter:TCond+shifter]
nonlinear_truth = nonlinear_truth.reshape(100, s, s, 1, TCond)

source_truth = source[:100, :, :, shifter:TCond+shifter]
source_truth = source_truth.reshape(100, s, s, 1, TCond)

time_frame_truth = sol_t[0][shifter:TCond+shifter].repeat(100, 1)

s = 64
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y))).unsqueeze(-1).unsqueeze(-1).repeat(100, 1, 1, 1, TCond)

print(vorticity_truth.size(), nonlinear_truth.size(), source_truth.size(), f.size())

generated_vorticity = torch.zeros(100, s, s, 1, TCond, device=device)
generated_vorticity[..., 0:1] = vorticity_truth[..., 0:1]

sampler1 = Euler_Maruyama_sampler

time_frame = sol_t[0][shifter:TCond+shifter]

for i in range(1, TCond):
    # source_generated = sampler(source_truth[..., i-1:i],
    #                             generated_vorticity[..., i - 1:i],
    #                             model,
    #                             marginal_prob_std_fn,
    #                             diffusion_coeff_fn,
    #                             100,
    #                             s,
    #                             sub,
    #                             num_steps,
    #                             time_frame,
    #                             device,
    #                             1e-3)

    dynamic = source_truth[..., i-1:i] - nonlinear_truth[..., i-1:i] + f[..., i-1:i]
    generated_vorticity[..., i:i+1] = generated_vorticity[..., i-1:i] + 0.1 * dynamic

# Pick 1 batch of generated_vorticity and vorticity_truth, plot all time steps
batch1 = generated_vorticity[8, :, :, 0, :]  # Shape (64, 64, 10)
batch2 = vorticity_truth[8, :, :, 0, :]  # Shape (64, 64, 10)

fig, axs = plt.subplots(2, 10, figsize=(20, 4))  # 2 rows, 10 columns

for i in range(10):  # Loop through the 10 time steps
    # Plot using contourf instead of imshow
    contour_set1 = axs[0, i].contourf(batch1[:, :, i].cpu().numpy(), 25, cmap=plt.cm.jet)
    axs[0, i].axis('off')  # Turn off axis

    contour_set2 = axs[1, i].contourf(batch2[:, :, i].cpu().numpy(), 25, cmap=plt.cm.jet)
    axs[1, i].axis('off')  # Turn off axis

plt.tight_layout()
plt.show()









































import math
train_nonlinear = torch.stack([nonlinear[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)]).unsqueeze(-1)
train_vorticity = torch.stack([sol[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)]).unsqueeze(-1)
# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
s = 64
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y))).unsqueeze(-1).unsqueeze(-1).repeat(100, 1, 1, 1, 1)

dynamic = samples + train_nonlinear[:100, :, :, :, :] + f

train_vorticity_next = torch.stack([sol[batch_idx, :, :, time_step_idx+1:time_step_idx+2] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)]).unsqueeze(-1)
generated_vorticity = train_vorticity[:100, :, :, :, :] + dynamic * 0.05

# plot 10 batches of samples
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(10):
    axs[i // 5, i % 5].contourf(generated_vorticity[i, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
    axs[i // 5, i % 5].set_title('Sample {}'.format(i))
plt.show()

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(10):
    axs[i // 5, i % 5].contourf(train_vorticity_next[i, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
    axs[i // 5, i % 5].set_title('Sample {}'.format(i))
plt.show()

to_plot = source[10, :, :, :11]

# Number of subplots in each row
n_cols = 5
n_rows = 2

# Create a figure with subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))

for i in range(n_rows):
    for j in range(n_cols):
        time_step = i * n_cols + j
        field = to_plot[:, :, time_step+1].cpu().numpy()  # Convert to numpy array for plotting
        c = axs[i, j].contourf(field, levels=25, cmap=plt.cm.jet)
        fig.colorbar(c, ax=axs[i, j])
        time = sol_t[0][time_step+1]
        axs[i, j].set_title(f'Time Step: {time:.3f}')
        axs[i, j].axis('off')  # Optional: Remove axes for visual clarity

plt.tight_layout()

plt.savefig('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Plots\\laplacian_truth.png', dpi=300)


