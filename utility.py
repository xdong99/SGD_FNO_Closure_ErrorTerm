import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os

################################
##### Data Preprosessing #######
################################
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

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

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
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

# 2d Fourier layer
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

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

# FNO3d model
class FNO3d(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, modes3, width, embed_dim=256):
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 14  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        self.fc0_sparse = nn.Linear(4, self.width)

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
        self.trans_conv3 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))
        self.trans_conv4 = nn.ConvTranspose3d(self.width, self.width, kernel_size=(8, 8, 1), stride=(8, 8, 1))

        self.dense0 = Dense(embed_dim, self.width)
        self.dense1 = Dense(embed_dim, self.width)
        self.dense2 = Dense(embed_dim, self.width)
        self.dense3 = Dense(embed_dim, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, time_frame, x, x_sparse):
        grid = self.get_grid(x.shape, x.device, time_frame, sparse=False)
        sparse_grid = self.get_grid(x_sparse.shape, x.device, time_frame, sparse=True)

        x = torch.cat((x, grid), dim=-1)
        x_sparse = torch.cat((x_sparse, sparse_grid), dim=-1)

        x = x.float()

        x = self.fc0(x)
        x_sparse = self.fc0_sparse(x_sparse)

        x = x.permute(0, 4, 1, 2, 3)
        x_sparse = x_sparse.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        x_sparse = F.pad(x_sparse, [0, self.padding])  # pad the domain if input is non-periodic

        embed = self.embed(t)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.trans_conv1(x_sparse)
        x4 = self.dense0(embed)
        x = x1 + x2 + x3 + x4
        # x = self.bn0(x)
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.trans_conv2(x_sparse)
        x4 = self.dense1(embed)
        x = x1 + x2 + x3 + x4
        # x = self.bn1(x)
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x3 = self.trans_conv3(x_sparse)
        # x4 = self.dense2(embed)
        # x = x1 + x2 + x3 + x4
        # # x = self.bn2(x)
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x3 = self.dense3(embed)
        # x = x1 + x2
        # # x = self.bn3(x)

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
            gridz = torch.tensor(np.linspace(time_frame[0].cpu().numpy(), time_frame[-1].cpu().numpy(), size_z), dtype=torch.float)
        elif sparse:
            gridz = torch.tensor(time_frame[-1].cpu().numpy())
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# Loss function
def loss_fn(model, x, time_frame, marginal_prob_std, eps=1e-5):
  """
  The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

  # Target (10th step)
  x_target = x[..., 9:10]

  # Conditions (history 9 steps & sparse 10th step)
  x_cond = x[..., :9]
  x_sparse = x_target[:, ::8, ::8, :, :]

  z = torch.randn_like(x_target)
  std = marginal_prob_std(random_t)
  perturbed_target = x_target + z * std[:, None, None, None, None]
  perturbed_x = torch.cat([x_cond, perturbed_target], dim=-1)
  score = model(random_t, time_frame, perturbed_x, x_sparse)
  real_score = -z/std[:, None, None, None, None]

  loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1, 2, 3, 4)))

  return loss, score, real_score


################################
########### Sampling ###########
################################

#Naive sampling: Euler-Maruyama sampler
def Euler_Maruyama_sampler(data,
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

    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x_target = init_x
    x_cond = data[:batch_size, :, :, :, :9]
    x_sparse = data[:batch_size, ::sub_dim, ::sub_dim, :, 9:10]
    print(x_sparse.size())
    x = torch.cat([x_cond, x_target], dim=-1)

    mean_x_target = x_target

    errors, errors_2, real_score_ls, modeled_score_ls = [], [], [], []

    with (torch.no_grad()):
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            perturbation = (mean_x_target-data[:batch_size, :, :, :, 9:10]) / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            real_score = -perturbation / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            modeled_score = score_model(batch_time_step, time_frame, x, x_sparse)

            # rel_err = torch.mean(torch.norm(modeled_score - real_score, 2, dim=(1, 2))
            #                        / torch.norm(real_score, 2, dim=(1, 2)))

            # loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z) ** 2, dim=(1, 2, 3, 4)))
            # rel_err_2 = torch.mean(torch.sum((modeled_score * marginal_prob_std(batch_time_step)[:, None, None, None, None] + perturbation) ** 2, dim=(1, 2, 3, 4)))
            # rel_err_2 = torch.mean(torch.sum(abs(modeled_score - real_score) / abs(real_score), dim=(1, 2, 3, 4)))

            # errors.append(rel_err.item())
            # errors_2.append(rel_err_2.item())

            mean_x_target = x_target + (g ** 2)[:, None, None, None, None] * modeled_score * step_size
            x_target = mean_x_target + torch.sqrt(step_size) * g[:, None, None, None, None] * torch.randn_like(x_target)

            x = torch.cat([x_cond, x_target], dim=-1)
            print('time_step:', time_step)

    # Do not include any noise in the last sampling step.
    full_x = torch.cat([x_cond, mean_x_target], dim=-1)
    return mean_x_target, full_x, errors, errors_2, real_score_ls, modeled_score_ls


