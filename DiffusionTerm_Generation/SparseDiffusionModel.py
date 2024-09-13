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
        self.fc0_sparse = nn.Linear(4, self.width)
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
        self.trans_conv1 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        self.trans_conv2 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        # self.trans_conv3 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        # self.trans_conv4 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))

        self.dense0 = Dense(embed_dim, self.width)
        # self.dense1 = Dense(embed_dim, self.width)
        # self.dense2 = Dense(embed_dim, self.width)
        # self.dense3 = Dense(embed_dim, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, time_frame, x, w, x_sparse):
        grid = self.get_grid(x.shape, x.device, time_frame, sparse=False)
        sparse_grid = self.get_grid(x_sparse.shape, x_sparse.device, time_frame, sparse=True)

        x = torch.cat((x, grid), dim=-1)
        x_sparse = torch.cat((x_sparse, sparse_grid), dim=-1)
        w = torch.cat((w, grid), dim=-1)

        x = x.float()

        x = self.fc0(x)
        x_sparse = self.fc0_sparse(x_sparse)
        w = self.fc0_w(w)

        x = x.permute(0, 4, 1, 2, 3)
        x_sparse = x_sparse.permute(0, 4, 1, 2, 3)
        w = w.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        x_sparse = F.pad(x_sparse, [0, self.padding])  # pad the domain if input is non-periodic
        w = F.pad(w, [0, self.padding])  # pad the domain if input is non-periodic

        embed = self.embed(t)
        t_embed = self.dense0(embed)


        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.trans_conv1(x_sparse)
        w11 = self.w1(w)
        x = x1 + x2 + x3 + t_embed + w11
        # x = self.bn0(x)
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w2(x)
        x3 = self.trans_conv2(x_sparse)
        w11 = self.w3(w)
        x = x1 + x2 + x3 + t_embed + w11

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
            # gridz = torch.tensor(np.linspace(time_frame[0].cpu().numpy(), time_frame[-1].cpu().numpy(), size_z), dtype=torch.float)
            gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        elif sparse:
            # gridz = torch.tensor(time_frame[-1].cpu().numpy())
            gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
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
  x_sparse = x[:, ::8, ::8, :, :]
  perturbed_target = x + z * std[:, None, None, None, None]
  # perturbed_x = torch.cat([w, perturbed_target], dim=-1)
  score = model(random_t, time_frame, perturbed_target, w, x_sparse)
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

    # # Start with a random initial condition (noise).
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, 1, 1, device=device) * marginal_prob_std(t)[:, None, None, None, None]
    # init_x = data[:batch_size, :, :, :, :] * marginal_prob_std(t)[:, None, None, None, None]

    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x_target = init_x
    x_sparse = data[:, ::8, ::8, :, :]
    # x = torch.cat([condition, x_target], dim=-1)

    mean_x_target = x_target
    # errors, errors_2, real_score_ls, modeled_score_ls = [], [], [], []

    with (torch.no_grad()):
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            # perturbation = (mean_x_target-data[:batch_size, :, :, :, :]) / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            # real_score = -perturbation / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            modeled_score = score_model(batch_time_step, time_frame, mean_x_target, condition, x_sparse)

            mean_x_target = mean_x_target + (g ** 2)[:, None, None, None, None] * modeled_score * step_size
            # x_target = mean_x_target + 0.01 * torch.sqrt(step_size) * g[:, None, None, None, None] * torch.randn_like(x_target)

            # x = torch.cat([condition, x_target], dim=-1)
            print('time_step:', time_step)

    # Do not include any noise in the last sampling step.
    # full_x = torch.cat([x_cond, mean_x_target], dim=-1)
    return mean_x_target

##############################
#######  Data Loading ########
##############################

# Load data
device = torch.device('cuda')
data = scipy.io.loadmat('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Data_Generation\\2d_ns_diffusion_test_50s_MidV.mat')

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

source = diffusion

batch_indices = torch.randint(0, 20, (1000,))
test_batch_indices = torch.randint(0, 20, (100,))

time_step_indices = torch.randint(1000, 5001, (1000,))
test_time_step_indices = torch.randint(1000, 5001, (100,))

train_source = torch.stack([source[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)])
test_source = torch.stack([source[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(test_batch_indices, test_time_step_indices)])
train_vorticity = torch.stack([sol[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(batch_indices, time_step_indices)])
test_vorticity = torch.stack([sol[batch_idx, :, :, time_step_idx:time_step_idx+1] for batch_idx, time_step_idx in zip(test_batch_indices, test_time_step_indices)])

# Spatial Resolution
s = 64
sub = 1

shifter = 5

# Train/Test
Ntrain = 1000
Ntest = 100
TCond = 1

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
cbar_axs = []  # List to hold colorbar axes

# Generate subplots and store the last ContourSet for colorbars
for i in range(10):
    CS = axs[i // 5, i % 5].contourf(train_source[i * 9, ..., 0].cpu(), 25, cmap=plt.cm.jet)
    axs[i // 5, i % 5].set_title(f'Example {i+1}')
    if i == 4:  # Last plot of the first row
        cbar_axs.append(fig.add_axes([0.91, 0.53, 0.01, 0.35]))  # Adjust these values as needed
    elif i == 9:  # Last plot of the second row
        cbar_axs.append(fig.add_axes([0.91, 0.11, 0.01, 0.35]))  # Adjust these values as needed

# Create color bars
fig.colorbar(CS, cax=cbar_axs[0])
fig.colorbar(CS, cax=cbar_axs[1])

plt.subplots_adjust(right=0.9)  # Adjust space on the right for the colorbars
# plt.savefig('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Plots\\example.png', dpi=300, bbox_inches='tight')

plt.show()
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
time_frame = sol_t[0][shifter:TCond+shifter]

train_source = train_source.reshape(Ntrain, s, s, 1, train_source.size()[-1]).repeat([1, 1, 1, 1, 1])
test_source = test_source.reshape(Ntest, s, s, 1, test_source.size()[-1]).repeat([1, 1, 1, 1, 1])
train_vorticity = train_vorticity.reshape(Ntrain, s, s, 1, train_vorticity.size()[-1]).repeat([1, 1, 1, 1, 1])
test_vorticity = test_vorticity.reshape(Ntest, s, s, 1, test_vorticity.size()[-1]).repeat([1, 1, 1, 1, 1])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_source, train_vorticity), batch_size=10, shuffle=True)


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
    for x, w in train_loader:
        # x = x[0].cuda()
        x, w = x.cuda(), w.cuda()
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, w, time_frame, marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'SparseDiffusionModelSmallV.pth')

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

ckpt = torch.load('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\SparseDiffusionModelSmallV.pth', map_location=device)
model.load_state_dict(ckpt)

num_steps = 1000
sample_batch_size = 100
sampler = Euler_Maruyama_sampler

samples = sampler(train_source[:sample_batch_size, :, :, :, :],
                train_vorticity[:sample_batch_size, :, :, :, :],
                model,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                sample_batch_size,
                s,
                sub,
                num_steps,
                time_frame,
                device,
                1e-3)

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 8))

# Initialize a variable to store the contour collections for adding colorbars later
contours = []

# Generate the contour plots
for i in range(0, 5):
    j = i % 5  # Adjust the index for columns
    contour1 = axs[0, j].contourf(train_source[i*4, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
    axs[0, j].set_title('Truth {}'.format(i+1))
    contour2 = axs[1, j].contourf(samples[i*4, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
    axs[1, j].set_title('Generated {}'.format(i+1))
    if i == 4:  # Store the last contour of each row for the colorbars
        contours.append(contour1)
        contours.append(contour2)

# Adjust the layout to make room for the colorbars
plt.subplots_adjust(right=0.85)

# Create colorbars
cbar_ax1 = fig.add_axes([0.88, 0.52, 0.01, 0.35])  # Adjust these values as needed for the first row
fig.colorbar(contours[0], cax=cbar_ax1)

cbar_ax2 = fig.add_axes([0.88, 0.1, 0.01, 0.35])  # Adjust these values as needed for the second row
fig.colorbar(contours[1], cax=cbar_ax2)

plt.show()

plt.savefig('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Plots\\ModelWithSparseMidV2.png', dpi=300, bbox_inches='tight')


# plot one batch of samples and the corresponding target
i = 10
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
axs[0].contourf(samples[i, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
axs[0].set_title('Sample')
axs[1].contourf(test_source[i, ..., 0, 0].cpu(), 25, cmap=plt.cm.jet)
axs[1].set_title('Target')
plt.show()

mse = torch.mean((samples - test_source[:sample_batch_size, :, :, :, :])**2)
rel_mse = (torch.mean(torch.norm(samples - test_source[:sample_batch_size, :, :, :, :], 2, dim=(1, 2))
                    / torch.norm(test_source[:sample_batch_size, :, :, :, :], 2, dim=(1, 2))))

# Function to calculate pattern correlation for PyTorch tensors
def calculate_pattern_correlation(tensor_a, tensor_b):
    pattern_correlations = []
    for i in range(tensor_a.shape[0]):  # Iterate over the batch dimension
        a = tensor_a[i].flatten()  # Flatten spatial dimensions
        b = tensor_b[i].flatten()
        mean_a = a.mean()
        mean_b = b.mean()
        a_centered = a - mean_a
        b_centered = b - mean_b
        covariance = (a_centered * b_centered).sum()
        std_a = torch.sqrt((a_centered**2).sum())
        std_b = torch.sqrt((b_centered**2).sum())
        correlation = covariance / (std_a * std_b)
        pattern_correlations.append(correlation)
    return torch.tensor(pattern_correlations)

pattern_correlations = torch.mean(calculate_pattern_correlation(samples, test_source[:sample_batch_size, :, :, :, :]))




## Vorticity Generation
import math
delta_t = 1e-3
nu = 5 * 1e-3
shifter = 500
sample_size = 5
vorticity = sol[:sample_size, :, :, shifter]
vorticity_series = torch.zeros(sample_size, s, s, 100)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.5 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

for i in range(100):
    print(i)
    # Do one Cranck-Nicolson step
    N1, N2 = vorticity.size()[-2], vorticity.size()[-1]

    # Maximum frequency
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(vorticity, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

    # forcing
    f_h = torch.fft.fftn(f, dim=[-2, -1])
    f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=vorticity.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=vorticity.device)), 0).repeat(N1, 1).transpose(0, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=vorticity.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=vorticity.device)), 0).repeat(N2, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    # lap_ = lap.clone()
    lap[0, 0] = 1.0
    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2, torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)

    F_h = torch.fft.fftn(nonlinear[:sample_size, :, :, shifter+i], dim=[1, 2])
    F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

    # Dealias
    F_h[..., 0] = dealias * F_h[..., 0]
    F_h[..., 1] = dealias * F_h[..., 1]

    diffusion_target = diffusion[:sample_size, :, :, shifter+i].reshape(sample_size, s, s, 1, 1)
    vorticity_condition = sol[:sample_size, :, :, shifter+i].reshape(sample_size, s, s, 1, 1)

    diffusion_sample = sampler(diffusion_target,
                                vorticity_condition,
                                model,
                                marginal_prob_std_fn,
                                diffusion_coeff_fn,
                                sample_size,
                                s,
                                sub,
                                num_steps,
                                time_frame,
                                device,
                                1e-3)

    diffusion_sample = diffusion_sample[:, :, :, 0, 0]
    # laplacian term
    diffusion_h = torch.fft.fftn(diffusion_sample, dim=[1, 2])
    diffusion_h = torch.stack([diffusion_h.real, diffusion_h.imag], dim=-1)

    # gudWh = torch.fft.fftn(sigma * forcing[:sample_size, :, :, shifter+i], dim=[1, 2])
    # gudWh = torch.stack([gudWh.real, gudWh.imag], dim=-1)

    w_h[..., 0] = ((w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + 0.5 * delta_t * diffusion_h[..., 0])
                   / (1.0 + 0.5 * delta_t * nu * lap))
    w_h[..., 1] = ((w_h[..., 1] - delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + 0.5 * delta_t * diffusion_h[..., 1])
                   / (1.0 + 0.5 * delta_t * nu * lap))

    # w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + (-1 / (nu * lap) + 0.5 * delta_t) * diffusion_h[
    #     ..., 0]) / (1.0 + 0.5 * delta_t * nu * lap)
    #
    # w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + (-1 / (nu * lap) + 0.5 * delta_t) * diffusion_h[
    #     ..., 1]) / (1.0 + 0.5 * delta_t * nu * lap)

    # w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + gudWh[..., 0] + w_h[..., 0] + 0.5 * delta_t *
    #                diffusion_h[..., 0]) / (1.0 + 0.5 * delta_t * nu * lap)
    # w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + gudWh[..., 1] + w_h[..., 1] + 0.5 * delta_t *
    #                diffusion_h[..., 1]) / (1.0 + 0.5 * delta_t * nu * lap)

    # w_h[..., 0] = ((-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] +
    #                 w_h[..., 0] - 0.5 * delta_t * laplacian_h[..., 0])
    #                / (1.0 + 0.5 * delta_t * nu * lap))
    # w_h[..., 1] = ((-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] +
    #                 w_h[..., 1] - 0.5 * delta_t * laplacian_h[..., 1])
    #                / (1.0 + 0.5 * delta_t * nu * lap))
    vorticity_series[..., i] = vorticity

    vorticity = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real
    # vorticity += sigma * 0.1 * lap_forcing[:sample_size, :, :, shifter+i]

known = w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0]
check = (delta_t * known + delta_t**2 / 2 * diffusion_h[..., 0]) * diffusion_h[..., 0] / (2 * (known + delta_t /2 * diffusion_h[..., 0])**2)
# check = 2 / (w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + delta_t/2 * diffusion_h[..., 0])
# norm of check
check_norm = torch.mean(check)

known = sol[..., 0] - delta_t * nonlinear[..., 0] + delta_t * f.repeat(600, 1, 1)
check = (delta_t * known + delta_t**2 / 2 * diffusion[..., 0]) * diffusion[..., 0] / (2 * (known + delta_t /2 * diffusion[..., 0])**2)
check.min()
check_norm = torch.mean(check)

all_values = check.flatten().cpu()
median = torch.median(all_values)
q1, q3 = np.percentile(all_values, [5, 95])
q1
q3
iqr = q3 - q1

plt.boxplot(all_values, vert=False)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(all_values.cpu(), bins=50, density=True, alpha=0.6, color='g')
plt.title('PDF of All Values in the Tensor')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

def relative_mse(tensor1, tensor2):
    """Calculate the Relative Mean Squared Error between two tensors."""
    rel_mse = torch.mean(torch.norm(tensor1 - tensor2, 2, dim=(0, 1)) / torch.norm(tensor2, 2, dim=(0, 1)))
    # return torch.mean((tensor1 - tensor2) ** 2) / torch.mean(tensor2 ** 2)
    return rel_mse

shifter = 500
k = 4

fig, axs = plt.subplots(2, 5, figsize=(28, 12))
cbar_axs = []  # List to hold colorbar axes

for i in range(5):
    j =  i  * 24
    generated = vorticity_series[k, :, :,j].cpu()
    truth = sol[k, :, :, shifter+j].cpu()
    rmse = relative_mse(generated, truth).item()
    pc = torch.mean(calculate_pattern_correlation(generated, truth)).item()

    contour1 = axs[0, i].contourf(generated, 25, cmap=plt.cm.jet)
    axs[0, i].set_title('Generated {}'.format(i+1), fontsize = 20)
    contour2 = axs[1, i].contourf(truth, 25, cmap=plt.cm.jet)
    axs[1, i].set_title('Truth {}'.format(i+1), fontsize = 20)

    # set tick size
    axs[0, i].tick_params(axis='both', which='major', labelsize=16)
    axs[1, i].tick_params(axis='both', which='major', labelsize=16)

    time_step = sol_t[0][shifter+j]
    text_x = 0.5
    text_y = -0.3
    axs[1, i].text(text_x, text_y, 'Time: {:.2f}\nRMSE: {:.2e}\nPC: {:.4f}'.format(time_step, rmse, pc),
                   ha='center', transform=axs[1, i].transAxes, fontsize=20)

    if i == 4:  # Store the last contour of each row for the colorbars
        cbar_axs.append(fig.add_axes([0.92, 0.52, 0.01, 0.35]))  # For the first row
        cbar_axs.append(fig.add_axes([0.92, 0.1, 0.01, 0.35]))  # For the second row

# Create colorbars
fig.colorbar(contour1, cax=cbar_axs[0])
fig.colorbar(contour2, cax=cbar_axs[1])

plt.subplots_adjust(right=0.9)  # Adjust space to accommodate colorbars
# plt.show()
plt.savefig('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Plots\\VorticityGenerationMid2V1.png', dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
cbar_axs = []  # List to hold colorbar axes

for i in range(5):
    j = 100 * i
    # i = h-5
    generated = vorticity_series[k, :, :, j].cpu()
    truth = sol[k, :, :, shifter+j].cpu()
    rmse = relative_mse(generated, truth).item()

    contour1 = axs[0, i].contourf(generated, 25, cmap=plt.cm.jet)
    axs[0, i].set_title('Generated {}'.format(i+1))
    contour2 = axs[1, i].contourf(truth, 25, cmap=plt.cm.jet)
    axs[1, i].set_title('Truth {}'.format(i+1))

    time_step = sol_t[0][shifter+j]
    text_x = 0.5
    text_y = -0.3
    axs[1, i].text(text_x, text_y, 'Time: {:.2f}\nRMSE: {:.2e}'.format(time_step, rmse),
                   ha='center', transform=axs[1, i].transAxes, fontsize=16)

    if i == 4:  # Store the last contour of each row for the colorbars
        cbar_axs.append(fig.add_axes([0.92, 0.55, 0.01, 0.35]))  # For the first row
        cbar_axs.append(fig.add_axes([0.92, 0.1, 0.01, 0.35]))  # For the second row

# Create colorbars
fig.colorbar(contour1, cax=cbar_axs[0])
fig.colorbar(contour2, cax=cbar_axs[1])

plt.subplots_adjust(right=0.9)  # Adjust space to accommodate colorbars
plt.savefig('C:\\UWMadisonResearch\\ConditionalScoreFNO\\ConditionalScoreFNO\\Plots\\VorticityGeneration.png', dpi=300, bbox_inches='tight')


# #visualize all 10 batches of vorticity
# fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# for i in range(10):
#     axs[i // 5, i % 5].contourf(vorticity[i, :, :].cpu(), 25, cmap=plt.cm.jet)
#     axs[i // 5, i % 5].set_title('Vorticity {}'.format(i))
# plt.show()
#
# # visualize first 10 batches of sol
# fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# for i in range(10):
#     axs[i // 5, i % 5].contourf(sol[i, :, :, shifter+1].cpu(), 25, cmap=plt.cm.jet)
#     axs[i // 5, i % 5].set_title('Sol {}'.format(i))
# plt.show()
