import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\DiffusionTerm_Generation')
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy as cp
from tqdm import tqdm
import h5py
plt.rcParams["animation.html"] = "jshtml"
import time
from torch.optim import Adam
from functools import partial
from tqdm import trange
import gc
import seaborn as sns
from cupyx.scipy.signal import convolve2d
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utility import (set_seed, marginal_prob_std, diffusion_coeff, GaussianFourierProjection, Dense,
                    SpectralConv2d)
from Data_Generation.generator_sns import (navier_stokes_2d_orig,
                                           navier_stokes_2d_filtered,
                                           navier_stokes_2d_smag,
                                           navier_stokes_2d_model)


# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

################################
######## SGD Model setup #######
################################


class FNO2d_New(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, width, embed_dim = 128, l2_reg=1e-5):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(5, self.width)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        # self.dense0 = Dense(embed_dim, self.width)
        self.dense0 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32 * 32)
        )

        self.conv0_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0_x = nn.Conv2d(self.width, self.width, 1)
        self.w1_x = nn.Conv2d(self.width, self.width, 1)
        self.w2_x = nn.Conv2d(self.width, self.width, 1)
        self.w3_x = nn.Conv2d(self.width, self.width, 1)
        self.w4_x = nn.Conv2d(self.width, self.width, 1)
        self.w5_x = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

        self.l2_reg = l2_reg

    def forward(self, t, x, x_prev, w):
        embed = self.act(self.embed(t))
        t_embed = self.dense0(embed).view(-1, 32, 32, 1)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        x = x + t_embed
        x_prev = x_prev.reshape(x_prev.shape[0], x_prev.shape[1], x_prev.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)

        x = torch.cat((x, x_prev, w), dim=-1) # (N, X, Y, 1) --> (N, X, Y, 3)

        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1) # (N, X, Y, 3) --> (N, X, Y, 5)

        x = self.fc0(x) # (N, X, Y, 5) --> (N, X, Y, 20)
        x = x.permute(0, 3, 1, 2) # (N, X, Y, 20) --> (N, 20, X, Y)

        x1 = self.conv0_x(x)
        x2 = self.w0_x(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1_x(x)
        x2 = self.w1_x(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2_x(x)
        x2 = self.w2_x(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3_x(x)
        x2 = self.w3_x(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv4_x(x)
        x2 = self.w4_x(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv5_x(x)
        x2 = self.w5_x(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        # Add L2 regularization
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)

        return x / self.marginal_prob_std(t)[:, None, None], self.l2_reg * l2_loss

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_Prev(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, width, embed_dim = 128):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(3, self.width)
        self.fc0_prev = nn.Linear(3, self.width)
        self.fc0_w = nn.Linear(3, self.width)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv0_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0_x_prev = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_x_prev = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_x_prev = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_x_prev = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0_x = nn.Conv2d(self.width, self.width, 1)
        self.w1_x = nn.Conv2d(self.width, self.width, 1)
        self.w2_x = nn.Conv2d(self.width, self.width, 1)
        self.w3_x = nn.Conv2d(self.width, self.width, 1)

        self.w0_x_prev = nn.Conv2d(self.width, self.width, 1)
        self.w1_x_prev = nn.Conv2d(self.width, self.width, 1)
        self.w2_x_prev = nn.Conv2d(self.width, self.width, 1)
        self.w3_x_prev = nn.Conv2d(self.width, self.width, 1)

        self.w0_w = nn.Conv2d(self.width, self.width, 1)
        self.w1_w = nn.Conv2d(self.width, self.width, 1)
        self.w2_w = nn.Conv2d(self.width, self.width, 1)
        self.w3_w = nn.Conv2d(self.width, self.width, 1)

        self.dense0 = Dense(embed_dim, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)

        self.attention = nn.Sequential(
            nn.Conv2d(width*3,  width, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(width, width*3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, x, x_prev, w):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        x_prev = x_prev.reshape(x_prev.shape[0], x_prev.shape[1], x_prev.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)

        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        x_prev = torch.cat((x_prev, grid), dim=-1)
        w = torch.cat((w, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x_prev = self.fc0_prev(x_prev)
        x_prev = x_prev.permute(0, 3, 1, 2)

        w = self.fc0_w(w)
        w = w.permute(0, 3, 1, 2)

        embed = self.act(self.embed(t))
        t_embed = self.dense0(embed).squeeze(-1)

        x1 = self.conv0_x(x)
        x2 = self.w0_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv1_x(x)
        x2 = self.w1_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv2_x(x)
        x2 = self.w2_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv3_x(x)
        x2 = self.w3_x(x)
        x = x1 + x2 + t_embed

        x_prev1 = self.conv0_x_prev(x_prev)
        x_prev2 = self.w0_x_prev(x_prev)
        x_prev = x_prev1 + x_prev2
        x_prev = F.gelu(x_prev)

        x_prev1 = self.conv1_x_prev(x_prev)
        x_prev2 = self.w1_x_prev(x_prev)
        x_prev = x_prev1 + x_prev2
        x_prev = F.gelu(x_prev)

        x_prev1 = self.conv2_x_prev(x_prev)
        x_prev2 = self.w2_x_prev(x_prev)
        x_prev = x_prev1 + x_prev2
        x_prev = F.gelu(x_prev)

        x_prev1 = self.conv3_x_prev(x_prev)
        x_prev2 = self.w3_x_prev(x_prev)
        x_prev = x_prev1 + x_prev2

        w1 = self.conv0_w(w)
        w2 = self.w0_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv1_w(w)
        w2 = self.w1_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv2_w(w)
        w2 = self.w2_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv3_w(w)
        w2 = self.w3_w(w)
        w = w1 + w2

        combined = torch.cat((x, x_prev, w), dim=1)
        weights = self.attention(combined)

        weights1, weights2, weights3 = torch.split(weights, self.width, dim=1)

        # print(weights1)
        # print(weights2)
        # print(weights3)
        x = x * weights1
        x_prev = x_prev * weights2
        w = w * weights3

        x = x + x_prev + w
        # x = self.transformation_net(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        return x / self.marginal_prob_std(t)[:, None, None] # (N, X, Y, 1) --> (N, X, Y)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def loss_fn_prev(model, x, x_prev, w, marginal_prob_std, eps=1e-5):
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_target = x + z * std[:, None, None]
  score = model(random_t, perturbed_target, x_prev, w)
  real_score = -z / std[:, None, None]

  loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1, 2)))

  return loss, score, real_score

class FNO2d(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, width, embed_dim = 32):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(3, self.width)
        self.fc0_w = nn.Linear(3, self.width)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv0_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0_x = nn.Conv2d(self.width, self.width, 1)
        self.w1_x = nn.Conv2d(self.width, self.width, 1)
        self.w2_x = nn.Conv2d(self.width, self.width, 1)
        self.w3_x = nn.Conv2d(self.width, self.width, 1)

        self.w0_w = nn.Conv2d(self.width, self.width, 1)
        self.w1_w = nn.Conv2d(self.width, self.width, 1)
        self.w2_w = nn.Conv2d(self.width, self.width, 1)
        self.w3_w = nn.Conv2d(self.width, self.width, 1)

        self.dense0 = Dense(embed_dim, self.width)

        # Define a transformation network for the concatenated output
        self.transformation_net = nn.Sequential(
            # nn.Conv2d(width*2, width*2, 1),  # Reduce dimensionality while combining information
            # nn.GELU(),
            nn.Conv2d(width*2, width, 1),  # Further compression to original width channels
            nn.GELU(),
            nn.Conv2d(width, width, 1),  # Optional: another layer to refine features
            nn.GELU()
        )

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, x, w):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)

        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        w = torch.cat((w, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        w = self.fc0_w(w)
        w = w.permute(0, 3, 1, 2)

        embed = self.act(self.embed(t))
        t_embed = self.dense0(embed).squeeze(-1)

        x1 = self.conv0_x(x)
        x2 = self.w0_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv1_x(x)
        x2 = self.w1_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv2_x(x)
        x2 = self.w2_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv3_x(x)
        x2 = self.w3_x(x)
        x = x1 + x2 + t_embed

        w1 = self.conv0_w(w)
        w2 = self.w0_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv1_w(w)
        w2 = self.w1_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv2_w(w)
        w2 = self.w2_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv3_w(w)
        w2 = self.w3_w(w)
        w = w1 + w2

        x = torch.cat((x, w), dim=1)
        x = self.transformation_net(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        return x / self.marginal_prob_std(t)[:, None, None] # (N, X, Y, 1) --> (N, X, Y)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def loss_fn(model, x, w, marginal_prob_std, eps=1e-5):

  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_target = x + z * std[:, None, None]
  score = model(random_t, perturbed_target, w)
  real_score = -z / std[:, None, None]

  loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1, 2)))

  return loss, score, real_score


################################
########### Sampling ###########
################################
def relative_mse(tensor1, tensor2):
    """Calculate the Relative Mean Squared Error between two tensors."""
    rel_mse = torch.mean(torch.norm(tensor1 - tensor2, 2, dim=(-2, -1)) / torch.norm(tensor2, 2, dim=(-2, -1)))
    return rel_mse

def cal_mse(tensor1, tensor2):
    """Calculate the Mean Squared Error between two tensors."""
    mse = torch.mean((tensor1 - tensor2)**2)
    return mse


##############################
#######  Data Loading ########
##############################
# Load data
device = torch.device('cuda')
#
filename_1 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_data_res32x32_T30s_nu1e-4.h5'
filename_2 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_data_res32x32_T30s_nu1e-4_smag.h5'
# Open the HDF5 file
with h5py.File(filename_1, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device=device)
    # sol = torch.tensor(file['sol'][()], device=device)
    filtered_sol = torch.tensor(file['filtered_sol'][()], device=device)
    nonlinear_diff = torch.tensor(file['nonlinear_diff'][()], device=device)
    diffusion_diff = torch.tensor(file['diffusion_diff'][()], device=device)

with h5py.File(filename_2, 'r') as file:
    nonlinear_diff_smag = torch.tensor(file['nonlinear_diff_smag'][()], device=device)


##############################
######  Data Preprocess ######
##############################
s = 32
train_batch_size = 22
test_batch_size = 4

# target
nonlinear_train = nonlinear_diff[:train_batch_size, :, :, 2000:]
nonlinear_test = nonlinear_diff[-test_batch_size:, :, :, 2000:]

# conditions
vorticity_train = filtered_sol[:train_batch_size, :, :, 2000:]
vorticity_test = filtered_sol[-test_batch_size:, :, :, 2000:]
nonlinear_prev_train = nonlinear_diff[:train_batch_size, :, :, 1995:2295]
nonlinear_prev_test = nonlinear_diff[-test_batch_size:, :, :, 1995:2295]
nonlinear_smag_train = nonlinear_diff_smag[:train_batch_size, :, :, :]
nonlinear_smag_test = nonlinear_diff_smag[-test_batch_size:, :, :, :]


nonlinear_train = nonlinear_train.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_test = nonlinear_test.permute(0,3,1,2).reshape(-1, s, s)

vorticity_train = vorticity_train.permute(0,3,1,2).reshape(-1, s, s)
vorticity_test = vorticity_test.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_smag_train = nonlinear_smag_train.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_smag_test = nonlinear_smag_test.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_prev_train = nonlinear_prev_train.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_prev_test = nonlinear_prev_test.permute(0,3,1,2).reshape(-1, s, s)

# # Calculate the correlation between the two tensors
# from scipy.stats import pearsonr
#
# # Reshape tensors to 2D for correlation calculation
# nonlinear_train_mean = nonlinear_train.mean(dim=0)
# nonlinear_smag_train_mean = nonlinear_smag_train.mean(dim=0)
# reshaped_smag = nonlinear_smag_train_mean.permute(2, 0, 1).reshape(1000, -1)
# reshaped = nonlinear_train_mean.permute(2, 0, 1).reshape(1000, -1)
#
# # Calculate correlation coefficients
# correlations = np.array([pearsonr(reshaped_smag[i].cpu(), reshaped[i].cpu())[0] for i in range(1000)])
#
# #set plot size
# plt.figure(figsize=(10, 6))
# plt.plot(correlations)
# plt.title('Mean Correlation')
# plt.xlabel('Temporal Index')
# plt.ylabel('Correlation')
# plt.show()
# correlations.mean()

set_seed(42)
indices = torch.randperm(nonlinear_train.shape[0])
nonlinear_train = nonlinear_train[indices]
vorticity_train = vorticity_train[indices]
nonlinear_smag_train = nonlinear_smag_train[indices]
nonlinear_prev_train = nonlinear_prev_train[indices]

set_seed(42)
indiced_test = torch.randperm(nonlinear_test.shape[0])
nonlinear_test = nonlinear_test[indiced_test]
vorticity_test = vorticity_test[indiced_test]
nonlinear_smag_test = nonlinear_smag_test[indiced_test]
nonlinear_prev_test = nonlinear_prev_test[indiced_test]

# Train/Test
Ntrain = 11000
Ntest = 4000

train_nonlinear = nonlinear_train[:Ntrain, :, :]
train_nonlinear_smag = nonlinear_smag_train[:Ntrain, :, :]
train_nonlinear_prev = nonlinear_prev_train[:Ntrain, :, :]
train_vorticity = vorticity_train[:Ntrain, :, :]

test_nonlinear = nonlinear_test[:Ntest, :, :]
test_nonlinear_smag = nonlinear_smag_test[:Ntest, :, :]
test_nonlinear_prev = nonlinear_prev_test[:Ntest, :, :]
test_vorticity = vorticity_test[:Ntest, :, :]

del nonlinear_train, nonlinear_test, vorticity_train, vorticity_test, nonlinear_smag_train, nonlinear_smag_test

gc.collect()
torch.cuda.empty_cache()


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear,
                                                                          train_nonlinear_smag,
                                                                          train_vorticity),
                                                                          batch_size=300, shuffle=True)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear,
                                                                          train_nonlinear_smag),
                                                                          batch_size=100, shuffle=True)

with torch.no_grad():
    current_max_dist = 0
    lam = 1e-6
    for i, (x, w) in enumerate(train_loader):
        x = x.to(device)
        x_ = x.view(x.shape[0], -1)
        max_dist = torch.cdist(x_, x_).max().item()

        if current_max_dist < max_dist:
            current_max_dist = max_dist
        print(current_max_dist)
    print('Final, max eucledian distance: {}'.format(current_max_dist))


################################
######## Model Training ########
################################
sigma = 11
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 12
width = 20
epochs = 1000
learning_rate = 0.001
scheduler_step = 200
scheduler_gamma = 0.5

model = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
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
    for x, x_smag in train_loader:
        x, x_smag  = x.cuda(), x_smag.cuda()
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, x_smag, marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'NonlinearModelonlySmag_32_V1.pth')


################################
##########  Sampling ###########
################################

# define and load model
sigma = 11
modes = 12
width = 20

marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

model = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()

ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\NonlinearModelwithoutSmag_32_V3.pth', map_location=device)
model.load_state_dict(ckpt)


sde_time_data: float = 0.5
sde_time_min = 1e-3
sde_time_max = 0.1
steps = 10

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])
def get_sigmas_karras(n, time_min, time_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = time_min ** (1 / rho)
    max_inv_rho = time_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)
time_noises = append_zero(torch.linspace(sde_time_max, sde_time_min, steps, device=device))



def sampler(prev_condition,
            vorticity_condition,
           score_model,
           marginal_prob_std,
           diffusion_coeff,
           batch_size,
           spatial_dim,
           num_steps,
           device):
    t = torch.ones(batch_size, device=device) * 0.1
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device) * marginal_prob_std(t)[:, None, None]
    x = init_x

    with (torch.no_grad()):
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            # grad = score_model(batch_time_step, x, condition, sparse_data)
            grad = score_model(batch_time_step, x, prev_condition)
            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)

    return mean_x

sample_batch_size = 1200
sample_spatial_dim = 32
sample_device = torch.device('cuda')
num_steps = 10

sampler = partial(sampler,
                  score_model = model,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_batch_size,
                    spatial_dim = sample_spatial_dim,
                    num_steps = num_steps,
                    device = sample_device)



sample = sampler(test_nonlinear_smag[:sample_batch_size], test_vorticity[:sample_batch_size])




nan_batches = torch.isnan(sample).any(dim=1).any(dim=1)
valid_indices = torch.where(~nan_batches)[0]
mse = torch.mean((sample[valid_indices] - test_nonlinear[valid_indices, :, :])**2)
rel_mse = (torch.mean( torch.norm(sample[valid_indices] - test_nonlinear[valid_indices, :, :], 2, dim=(1, 2))
                    / torch.norm(test_nonlinear[valid_indices, :, :], 2, dim=(1, 2))) )



set_seed(12)

fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
plt.rcParams.update({'font.size': 16})

# Ticks setting
ticks = np.arange(0, sample_spatial_dim, 10 * sample_spatial_dim / 64)
tick_labels = [str(int(tick)) for tick in ticks]

# Assuming torch.manual_seed or equivalent has been set as needed
indices = [torch.randint(0, sample_batch_size, (1,)).item() for _ in range(4)]

# Variables to store the min and max values for the first two rows, and separate for error
min_val, max_val = np.inf, -np.inf
min_error, max_error = np.inf, -np.inf

# Determine the global min and max from the data to be plotted for consistent coloring
for idx in indices:
    data1 = test_nonlinear[idx, ...].cpu().numpy()
    data2 = sample[idx, ...].cpu().numpy()
    min_val = min(min_val, data1.min(), data2.min())
    max_val = max(max_val, data1.max(), data2.max())
    error_data = np.abs(data2 - data1)
    min_error = min(min_error, error_data.min())
    max_error = max(max_error, error_data.max())

# Plotting
for i, idx in enumerate(indices):
    j = i % 4

    # Truth plot
    data1 = test_nonlinear[idx, ...].cpu().numpy()
    sns.heatmap(data1, ax=axs[0, j], cmap='rocket', cbar=(i % 4 == 3), vmin=min_error, vmax=max_val)
    axs[0, j].set_title(r"$G(\overline{\omega})$ " + str(j+1))
    axs[0, j].set_xticks(ticks)
    axs[0, j].set_yticks(ticks)
    axs[0, j].set_xticklabels(tick_labels, rotation=0)
    axs[0, j].set_yticklabels(tick_labels, rotation=0)

    # Generated plot
    data2 = sample[idx, ...].cpu().numpy()
    sns.heatmap(data2, ax=axs[1, j], cmap='rocket', cbar=(i % 4 == 3), vmin=min_error, vmax=max_val)
    axs[1, j].set_title(r"$Gen G(\overline{\omega})$ " + str(j+1))
    axs[1, j].set_xticks(ticks)
    axs[1, j].set_yticks(ticks)
    axs[1, j].set_xticklabels(tick_labels, rotation=0)
    axs[1, j].set_yticklabels(tick_labels)

    # Error plot
    error_data = np.abs(data2 - data1)
    sns.heatmap(error_data, ax=axs[2, j], cmap='rocket', cbar=(i % 4 == 3), vmin=0, vmax=0.1)
    axs[2, j].set_title(f"Error {j+1}")
    axs[2, j].set_xticks(ticks)
    axs[2, j].set_yticks(ticks)
    axs[2, j].set_xticklabels(tick_labels, rotation=0)
    axs[2, j].set_yticklabels(tick_labels)

for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust label size as needed
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
# plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\draft_plots\\generated32_onlyw.png',
#             dpi=300, bbox_inches='tight')
plt.show()


### Do reverse SDE sampling every time step

filename = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\test_data_res32x32_T30s_nu1e-4.h5'

with h5py.File(filename, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device=device)
    filtered_sol = torch.tensor(file['filtered_sol'][()], device=device)
    nonlinear_diff = torch.tensor(file['nonlinear_diff'][()], device=device)
    diffusion_diff = torch.tensor(file['diffusion_diff'][()], device=device)

def local_avg(tensor, kernel):
    B, H, W = tensor.shape
    filtered_tensor = cp.zeros((B, H, W))
    for b in range(B):
        filtered_slice = convolve2d(tensor[b, :, :], kernel, mode='same', boundary='wrap')
        filtered_tensor[b, :, :] = filtered_slice
    return filtered_tensor

kernel = torch.ones(16, 16) / 256.0
kernel = cp.from_dlpack(kernel.to('cuda'))
downscale = 8

t = torch.linspace(0, 1, 256 + 1, device=device)
t = t[0:-1]
X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
f_filtered = local_avg(cp.from_dlpack(f.unsqueeze(0)) , kernel)
f_filtered = torch.tensor(f.squeeze(0)).to(device)
f_filtered = f[::downscale, ::downscale]

## Vorticity Generation
delta_t = 1e-3
T = 10
nu = 1e-4
shifter = 20000
sample_size = 1
sampling_steps = 10
total_steps = 10000
record_steps = 10000
s = 32

sol_truth = filtered_sol[0:1, :, :, shifter:shifter+total_steps].to(device)
nonlinear_diff_truth = nonlinear_diff[0:1, :, :, shifter:shifter+total_steps].to(device)
sol_start = filtered_sol[0:1, :, :, shifter].to(device)
# sol_start = sol_start.float()

smag_condition = torch.zeros(sample_size, s, s, total_steps).to(device)
sol_smag, smag_condition = navier_stokes_2d_smag([1, 1], filtered_sol[0:1, :, :, shifter], f_filtered, nu, T,
                                       delta_t=delta_t, record_steps=record_steps)

sampler = partial(sampler,
                  score_model = model,
                  marginal_prob_std = marginal_prob_std_fn,
                  diffusion_coeff = diffusion_coeff_fn,
                  batch_size = sample_size,
                  spatial_dim = s,
                  num_steps = sampling_steps,
                  device = sample_device)


start_time = time.time()

sol_model = torch.zeros(sample_size, s, s, total_steps).to(device)
sol_model = navier_stokes_2d_model([1, 1], sol_start, f_filtered, nu, T,
                                    smag_condition, sampler, delta_t=delta_t, record_steps=record_steps, eva_steps=10)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")

sol_nocorr, _, _, _ = navier_stokes_2d_orig([1, 1], sol_start, f_filtered, nu, T, delta_t=delta_t, record_steps=record_steps)



# Assuming 'vorticity_series', 'vorticity_NoG', and 'sol' are preloaded tensors
shifter = 20000
k = 0

# Create a figure and a grid of subplots
fig, axs = plt.subplots(5, 5, figsize=(25, 27), gridspec_kw={'width_ratios': [1]*4 + [1.073]})

# Plot each row using seaborn heatmap
for row in range(5):
    for i in range(5):  # Loop through all ten columns
        ax = axs[row, i]

        j = i * 2499
        generated =sol_model[k, :, :, j].cpu()
        generated_nog = sol_model[k, :, :, j].cpu()
        truth = filtered_sol[k, :, :, shifter + j].cpu()
        error_field = abs(generated - truth)
        error_field_nog = abs(generated_nog - truth)

        rmse = relative_mse(torch.tensor(generated), torch.tensor(truth)).item()
        mse = cal_mse(torch.tensor(generated), torch.tensor(truth)).item()

        rmse_nog = relative_mse(torch.tensor(generated_nog), torch.tensor(truth)).item()
        mse_nog = cal_mse(torch.tensor(generated_nog), torch.tensor(truth)).item()

        print(f"Time: {sol_t[shifter + j]:.2f}s")
        print(f"RMSE: {rmse:.4f}")
        print(f"MSE: {mse:.8f}")
        print(f"RMSE NoG: {rmse_nog:.4f}")
        print(f"MSE NoG: {mse_nog:.8f}")

        # Set individual vmin and vmax based on the row
        if row == 0:
            data = truth
            vmin, vmax = truth.min(), truth.max()  # Limits for Truth and Generated rows
            ax.set_title(f't = {sol_t[shifter + j]:.2f}s', fontsize=22)
        elif row == 1:
            data = generated
            vmin, vmax = truth.min(), truth.max()  # Limits for Truth and Generated rows
        elif row == 2:
            data = generated_nog
            vmin, vmax = generated_nog.min(), generated_nog.max()
        elif row == 3:
            data = error_field
            vmin, vmax = 0, 3.0
        else:
            data = error_field_nog
            vmin, vmax = 0, 3.0
        # Plot heatmap
        sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)

        ax.axis('off')  # Turn off axis for cleaner look

        if i == 4:
            # Create a new axis for the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(ax.collections[0], cax=cax, ticks=np.linspace(vmin, vmax, 5))
            cax.tick_params(labelsize=22)

            # Format tick labels based on the row
            if row < 3:  # For the first two rows
                cb.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in np.linspace(vmin, vmax, 5)])
            else:  # For the last row
                cb.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.linspace(vmin, vmax, 5)])

# Add row titles on the side
# row_titles = ['Truth', 'Generated', 'Generated NoG', 'Error', 'Error NoG']
row_titles = ['Truth', 'Generated', 'No Correction', 'Error', 'Error No Cor']
for ax, row_title in zip(axs[:, 0], row_titles):
    ax.annotate(row_title, xy=(0.1, 0.5), xytext=(-50, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=22)

plt.tight_layout()  # Adjust the subplots to fit into the figure area
# plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\draft_plots\\surrogate32', dpi=300,
#                                                                     bbox_inches='tight')
plt.show()



import matplotlib.gridspec as gridspec

# Time values in seconds for the x-axis
time_values = [20, 22.5, 25, 27.5, 30]

# MSE and RMSE data for simulations
sim_vort_mse_I = [0, 7.9055e-04, 1.4662e-03, 2.6284e-03, 5.9687e-03]
sim_vort_rmse_I = [0, 0.0187, 0.0241, 0.0308, 0.0457]
sim_vort_mse_II = [0, 8.1041e-04, 1.5531e-03, 2.7812e-03, 1.0517e-02]
sim_vort_rmse_II = [0, 0.0192, 0.0248, 0.0317, 0.0606]
sim_vort_mse_III = [0, 8.0125e-04, 1.5344e-03, 2.7294e-03, 6.3684e-03]
sim_vort_rmse_III = [0, 0.0188, 0.0246, 0.0314, 0.0472]
sim_vort_mse_IV = [0, 3.2736e-02, 6.2798e-02, 1.2587e-01, 2.3230e-01]
sim_vort_rmse_IV = [0, 0.1202, 0.1577, 0.2135, 0.2849]
# Create a figure with a custom gridspec layout
# Enable LaTeX rendering

# Create a figure with a custom gridspec layout
fig = plt.figure(figsize=(28, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
fs = 35

# MSE Plot
ax0 = plt.subplot(gs[0])
ax0.plot(time_values, sim_vort_mse_I, marker='o', linestyle="-.", markersize=10, linewidth=4, label="With true G")
ax0.plot(time_values, sim_vort_mse_II, marker='o', linestyle=":", markersize=10, linewidth=4, label="With G($\\bar{\\omega}$, G$_{smag}$)")
ax0.plot(time_values, sim_vort_mse_III, marker='o', linestyle="--", markersize=10, linewidth=4, label="With G($\\bar{\\omega}$)")
ax0.plot(time_values, sim_vort_mse_IV, marker='o', linestyle="-", markersize=10, linewidth=4, label="Without G")

ax0.set_title("D$_{MSE}$ Comparison", fontsize=fs)
ax0.set_xlabel("Time (s)", fontsize=fs)
ax0.set_ylabel("D$_{MSE}$", fontsize=fs)
ax0.tick_params(axis='both', which='major', labelsize=fs)

# RMSE Plot
ax1 = plt.subplot(gs[1])
ax1.plot(time_values, sim_vort_rmse_I, marker='o', linestyle="-.", markersize=10, linewidth=4, label="With true G")
ax1.plot(time_values, sim_vort_rmse_II, marker='o', linestyle=":", markersize=10, linewidth=4, label="With G($\\bar{\\omega}$, G$_{smag}$)")
ax1.plot(time_values, sim_vort_rmse_III, marker='o', linestyle="--", markersize=10, linewidth=4, label="With G($\\bar{\\omega}$)")
ax1.plot(time_values, sim_vort_rmse_IV, marker='o', linestyle="-", markersize=10, linewidth=4, label="Without G")

ax1.set_title("D$_{RE}$ Comparison", fontsize=fs)
ax1.set_xlabel("Time (s)", fontsize=fs)
ax1.set_ylabel("D$_{RE}$", fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs)

# Create a shared legend
handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs)
plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust this value as needed
plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\draft_plots\\MSE_RE_Comparison', dpi=300)
plt.show()





