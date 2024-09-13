"""
@author: Xinghao Dong

This file contains the code for the FNO3d models which can be used for 2d field generation.
For the Navier-Stokes equation discussed in the paper, this model can generate the score function
for the vorticity or the velocity fields conditioned on historical time steps and sparse observations
of the current time step. This score model can then be used for conditional generation of the vorticity
or the velocity fields.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


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

# 3D Fourier layer - 3d convolution in Fourier space
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

### Time dependent FNO 3d conditioned on historical and sparse information (interpolation)
class FNO3d_Conditional(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, modes3, width, embed_dim=256):
        super(FNO3d_Conditional, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        # self.padding = 0  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        self.fc0_sparse = nn.Linear(4, self.width)

        # self.conv_smooth = nn.Sequential(
        #     nn.Conv2d(width, width, 3, padding=1),
        #     nn.InstanceNorm2d(width),
        #     nn.GELU(),
        #     nn.Conv2d(width, width, 3, padding=1),
        #     nn.InstanceNorm2d(width),
        #     nn.GELU(),
        # )

        self.conv_smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.InstanceNorm2d(width),
            nn.GELU(),
        )

        # self.embedding = nn.Conv2d(4, width, kernel_size=1)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        #self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.conv0_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        #self.conv2_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        #self.conv3_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0_sparse = nn.Conv3d(self.width, self.width, 1)
        self.w1_sparse = nn.Conv3d(self.width, self.width, 1)
        #self.w2_sparse = nn.Conv3d(self.width, self.width, 1)
        #self.w3_sparse = nn.Conv3d(self.width, self.width, 1)

        self.dense_time = Dense(embed_dim, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, x, x_sparse):

        grid = self.get_grid(x.shape, x.device)
        sparse_grid = self.get_grid(x_sparse.shape, x_sparse.device)

        x = torch.cat((x, grid), dim=-1)
        x_sparse = torch.cat((x_sparse, sparse_grid), dim=-1)

        x_sparse = self.fc0_sparse(x_sparse)
        x_sparse = x_sparse.permute(0, 4, 1, 2, 3)
        x_sparse = x_sparse.squeeze(-1)

        # x_sparse = self.conv_smooth(F.interpolate(x_sparse, scale_factor=2))
        x_sparse = self.conv_smooth(F.interpolate(x_sparse, size=(x.shape[1], x.shape[2]))) # (N, C, X, Y)
        x_sparse = x_sparse.unsqueeze(-1)

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        # x_sparse = F.pad(x_sparse, [0, self.padding])  # pad the domain if input is non-periodic

        embed = self.act(self.embed(t))
        t_embed = self.dense_time(embed)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2 + t_embed
        # x = F.gelu(x)
        #
        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2 + t_embed
        # x = F.gelu(x)
        #
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2 + t_embed

        x_sparse1 = self.conv0_sparse(x_sparse)
        x_sparse2 = self.w0_sparse(x_sparse)
        x_sparse = x_sparse1 + x_sparse2
        x_sparse = F.gelu(x_sparse)

        x_sparse1 = self.conv1_sparse(x_sparse)
        x_sparse2 = self.w1_sparse(x_sparse)
        x_sparse = x_sparse1 + x_sparse2
        # x_sparse = F.gelu(x_sparse)
        #
        # x_sparse1 = self.conv2_sparse(x_sparse)
        # x_sparse2 = self.w2_sparse(x_sparse)
        # x_sparse = x_sparse1 + x_sparse2
        # x_sparse = F.gelu(x_sparse)
        #
        # x_sparse1 = self.conv3_sparse(x_sparse)
        # x_sparse2 = self.w3_sparse(x_sparse)
        # x_sparse = x_sparse1 + x_sparse2



        # x = x[..., :-self.padding]
        # x_sparse = x_sparse[..., :-self.padding]
        x = x + x_sparse
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        score = x / self.marginal_prob_std(t)[:, None, None, None, None]
        return score

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    # def get_grid(self, shape, device, time_frame, sparse=False):
    #     batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    #     if not sparse:
    #         gridz = torch.tensor(np.linspace(time_frame[0].cpu().numpy(), time_frame[-1].cpu().numpy(), size_z), dtype=torch.float)
    #     elif sparse:
    #         gridz = torch.tensor(time_frame[-1].cpu().numpy())
    #     gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    #     return torch.cat((gridx, gridy, gridz), dim=-1).to(device)



# Loss function
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
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
  x_target = x[..., -1:]

  # Conditions (history 9 steps & sparse 10th step)
  x_cond = x[..., :-1]
  x_sparse = x_target[:, ::4, ::4, :, :]

  z = torch.randn_like(x_target)
  std = marginal_prob_std(random_t)
  perturbed_target = x_target + z * std[:, None, None, None, None]
  perturbed_x = torch.cat([x_cond, perturbed_target], dim=-1)
  score = model(random_t, perturbed_x, x_sparse)
  real_score = -z/std[:, None, None, None, None]

  loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1, 2, 3, 4)))

  return loss, score, real_score




### Time dependent FNO 3d conditioned on historical and sparse information (mask)
class FNO3d_Conditional_Mask(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, modes3, width, embed_dim=256):
        super(FNO3d_Conditional_Mask, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        # self.padding = 0  # pad the domain if input is non-periodic
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

        self.conv0_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3_sparse = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0_sparse = nn.Conv3d(self.width, self.width, 1)
        self.w1_sparse = nn.Conv3d(self.width, self.width, 1)
        self.w2_sparse = nn.Conv3d(self.width, self.width, 1)
        self.w3_sparse = nn.Conv3d(self.width, self.width, 1)

        self.dense_time = Dense(embed_dim, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, x, x_sparse):

        grid = self.get_grid(x.shape, x.device)
        sparse_grid = self.get_grid(x_sparse.shape, x_sparse.device)

        x = torch.cat((x, grid), dim=-1)
        x_sparse = torch.cat((x_sparse, sparse_grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x_sparse = self.fc0_sparse(x_sparse)
        x_sparse = x_sparse.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        # x_sparse = F.pad(x_sparse, [0, self.padding])  # pad the domain if input is non-periodic

        embed = self.act(self.embed(t))
        t_embed = self.dense_time(embed)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2 + t_embed

        x_sparse1 = self.conv0_sparse(x_sparse)
        x_sparse2 = self.w0_sparse(x_sparse)
        x_sparse = x_sparse1 + x_sparse2
        x_sparse = F.gelu(x_sparse)

        x_sparse1 = self.conv1_sparse(x_sparse)
        x_sparse2 = self.w1_sparse(x_sparse)
        x_sparse = x_sparse1 + x_sparse2

        x = x + x_sparse
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        score = x / self.marginal_prob_std(t)[:, None, None, None, None]
        return score

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    # def get_grid(self, shape, device, time_frame, sparse=False):
    #     batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    #     if not sparse:
    #         gridz = torch.tensor(np.linspace(time_frame[0].cpu().numpy(), time_frame[-1].cpu().numpy(), size_z), dtype=torch.float)
    #     elif sparse:
    #         gridz = torch.tensor(time_frame[-1].cpu().numpy())
    #     gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    #     return torch.cat((gridx, gridy, gridz), dim=-1).to(device)



# Loss function
def loss_fn_mask(model, x, x_sparse, marginal_prob_std, eps=1e-5):
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
  x_target = x[..., -1:]

  # Conditions (history 9 steps & sparse 10th step)
  x_cond = x[..., :-1]

  z = torch.randn_like(x_target)
  std = marginal_prob_std(random_t)
  perturbed_target = x_target + z * std[:, None, None, None, None]
  perturbed_x = torch.cat([x_cond, perturbed_target], dim=-1)
  score = model(random_t, perturbed_x, x_sparse)
  real_score = -z/std[:, None, None, None, None]

  loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1, 2, 3, 4)))

  return loss, score, real_score

#Naive sampling: Euler-Maruyama sampler
def Euler_Maruyama_sampler(data,
                           x_sparse,
                           score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size,
                           spatial_dim,
                           sub_dim,
                           num_steps,
                           device,
                           eps):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, 1, 1, device=device) * marginal_prob_std(t)[:, None, None, None, None]

    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x_target = init_x
    x_cond = data[:batch_size, :, :, :, :-1]
    x = torch.cat([x_cond, x_target], dim=-1)

    with (torch.no_grad()):
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            # perturbation = (mean_x_target-data[:batch_size, :, :, :, 9:10]) / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            # real_score = -perturbation / marginal_prob_std(batch_time_step)[:, None, None, None, None]
            modeled_score = score_model(batch_time_step, x, x_sparse)
            print(modeled_score.shape)
            mean_x_target = x_target + (g ** 2)[:, None, None, None, None] * modeled_score * step_size
            print(mean_x_target.shape)
            x_target = mean_x_target + torch.sqrt(step_size) * g[:, None, None, None, None] * torch.randn_like(x_target)
            print(x_target.shape)
            x = torch.cat([x_cond, x_target], dim=-1)

    # Do not include any noise in the last sampling step.
    full_x = torch.cat([x_cond, mean_x_target], dim=-1)
    return mean_x_target, full_x