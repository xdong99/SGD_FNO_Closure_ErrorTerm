import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\DiffusionTerm_Generation')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
plt.rcParams["animation.html"] = "jshtml"
from torch.optim import Adam
from functools import partial
from tqdm import trange

from utility import (set_seed, marginal_prob_std, diffusion_coeff, GaussianFourierProjection, Dense,
                    SpectralConv2d)


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

# filename_1 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_test_180000_32_32.h5'
# with h5py.File(filename_1, 'r') as file:
#     sol_t = torch.tensor(file['t'][()], device=device)
#     train_nonlinear = torch.tensor(file['train_nonlinear'][()], device=device)
#     test_nonlinear = torch.tensor(file['test_nonlinear'][()], device=device)
#     train_nonlinear_smag = torch.tensor(file['train_nonlinear_smag'][()], device=device)
#     test_nonlinear_smag = torch.tensor(file['test_nonlinear_smag'][()], device=device)
#     train_vorticity = torch.tensor(file['train_vorticity'][()], device=device)
#     test_vorticity = torch.tensor(file['test_vorticity'][()], device=device)

filename_1 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_test_full.h5'
with h5py.File(filename_1, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device=device)
    train_nonlinear = torch.tensor(file['train_nonlinear'][()], device=device)
    test_nonlinear = torch.tensor(file['test_nonlinear'][()], device=device)
    train_nonlinear_smag = torch.tensor(file['train_nonlinear_smag'][()], device=device)
    test_nonlinear_smag = torch.tensor(file['test_nonlinear_smag'][()], device=device)
    train_vorticity = torch.tensor(file['train_vorticity'][()], device=device)
    test_vorticity = torch.tensor(file['test_vorticity'][()], device=device)

# filename_2 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_test_short.h5'
# with h5py.File(filename_2, 'r') as file:
#     sol_t = torch.tensor(file['t'][()], device=device)
#     train_nonlinear_short = torch.tensor(file['train_nonlinear'][()], device=device)
#     test_nonlinear_short = torch.tensor(file['test_nonlinear'][()], device=device)
#     train_nonlinear_smag_short = torch.tensor(file['train_nonlinear_smag'][()], device=device)
#     test_nonlinear_smag_short = torch.tensor(file['test_nonlinear_smag'][()], device=device)
#     train_vorticity_short = torch.tensor(file['train_vorticity'][()], device=device)
#     test_vorticity_short = torch.tensor(file['test_vorticity'][()], device=device)

################################
######## Model Training ########
################################
sigma = 12
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 12
width = 20
epochs = 1500
learning_rate = 0.001
scheduler_step = 500
scheduler_gamma = 0.2

##################################
# 1. G(w), 20-30s,  ##############
# half amount of data  ###########
##################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear[:15000],
                                                                          train_vorticity[:15000]),
                                                                          batch_size=150, shuffle=True)



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
    for x, w in train_loader:
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, w, marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'full_only_w_v2.pth')




##################################
# 2. G(w, smag), 20-30s,  ########
# full amount of data  ###########
##################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear,
                                                                          train_nonlinear_smag,
                                                                          train_vorticity),
                                                                          batch_size=220, shuffle=True)

model = FNO2d_Prev(marginal_prob_std_fn, modes, modes, width).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)
loss_history_2 = []
rel_err_history_2 = []

for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    rel_err = []
    for x, x_smag, w in train_loader:
        optimizer.zero_grad()
        loss, score, real_score = loss_fn_prev(model, x, x_smag, w, marginal_prob_std_fn)
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
    loss_history_2.append(avg_loss_epoch)
    rel_err_history_2.append(relative_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
torch.save(model.state_dict(), 'full_w_smag_v2.pth')


##################################
# 3. G(smag), 20-30s,  ###########
# half amount of data  ###########
##################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear[:11000],
                                                                          train_nonlinear_smag[:11000]),
                                                                          batch_size=100, shuffle=True)



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
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, x_smag,marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'full_only_smag_v1.pth')


##################################
# 4. G(w), 20-23s,  ##############
# half amount of data  ###########
##################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear_short[:3300],
                                                                          train_nonlinear_smag_short[:3300]),
                                                                          batch_size=50, shuffle=True)



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
    for x, w in train_loader:
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, w,marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'short_only_w_v1.pth')



##################################
# 5. G(w, smag), 20-23s,  ########
# full amount of data  ###########
##################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear_short,
                                                                          train_nonlinear_smag_short,
                                                                          train_vorticity_short),
                                                                          batch_size=100, shuffle=True)



model = FNO2d_Prev(marginal_prob_std_fn, modes, modes, width).cuda()
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
    for x, x_smag, w in train_loader:
        optimizer.zero_grad()
        loss, score, real_score = loss_fn_prev(model, x, x_smag, w, marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'short_w_smag_v1.pth')


##################################
# 6. G(smag), 20-23s,  ###########
# half amount of data  ###########
##################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear_short[:3300],
                                                                          train_nonlinear_smag_short[:3300]),
                                                                          batch_size=50, shuffle=True)



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
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, x_smag,marginal_prob_std_fn)
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
torch.save(model.state_dict(), 'short_only_smag_v1.pth')


##################################

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

def sampler(smag_condition,
           vorticity_condition,
           score_model,
           marginal_prob_std,
           diffusion_coeff,
           batch_size,
           spatial_dim,
           num_steps,
           time_noises,
           device):
    t = torch.ones(batch_size, device=device) * 0.1
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device) * marginal_prob_std(t)[:, None, None]
    x = init_x

    with (torch.no_grad()):
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            if smag_condition == None:
                grad = score_model(batch_time_step, x, vorticity_condition)
            elif vorticity_condition == None:
                grad = score_model(batch_time_step, x, smag_condition)
            else:
                grad = score_model(batch_time_step, x, smag_condition, vorticity_condition)
            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)

    return mean_x

sample_batch_size = 4000
sample_spatial_dim = 32
sample_device = torch.device('cuda')
num_steps = 10

sampler = partial(sampler,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_batch_size,
                    spatial_dim = sample_spatial_dim,
                    num_steps = num_steps,
                    time_noises = time_noises,
                    device = sample_device)



model1 = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\full_only_w_v1.pth', map_location=device)
model1.load_state_dict(ckpt)

model2 = FNO2d_Prev(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\full_w_smag_v1.pth', map_location=device)
model2.load_state_dict(ckpt)

model3 = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\full_only_smag_v1.pth', map_location=device)
model3.load_state_dict(ckpt)

model4 = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\short_only_w_v1.pth', map_location=device)
model4.load_state_dict(ckpt)

model5 = FNO2d_Prev(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\short_w_smag_v1.pth', map_location=device)
model5.load_state_dict(ckpt)

model6 = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\short_only_smag_v1.pth', map_location=device)
model6.load_state_dict(ckpt)

sample1_test = sampler(None, test_vorticity_long[:sample_batch_size], model1)
sample1_train = sampler(None, train_vorticity_long[:sample_batch_size], model1)
print('Model 1')
rmse1_test = relative_mse(sample1_test, test_nonlinear_long[:sample_batch_size])
rmse1_train = relative_mse(sample1_train, train_nonlinear_long[:sample_batch_size])
print('RMSE Test:', rmse1_test)
print('RMSE Train:', rmse1_train)

sample2_test = sampler(test_nonlinear_smag_long[:sample_batch_size], test_vorticity_long[:sample_batch_size], model2)
sample2_train = sampler(train_nonlinear_smag_long[:sample_batch_size], train_vorticity_long[:sample_batch_size], model2)
print('Model 2')
rmse2_test = relative_mse(sample2_test, test_nonlinear_long[:sample_batch_size])
rmse2_train = relative_mse(sample2_train, train_nonlinear_long[:sample_batch_size])
print('RMSE Test:', rmse2_test)
print('RMSE Train:', rmse2_train)

sample3_test = sampler(test_nonlinear_smag[:sample_batch_size], None, model3)
sample3_train = sampler(train_nonlinear_smag[:sample_batch_size], None, model3)
print('Model 3')
rmse3_test = relative_mse(sample3_test, test_nonlinear[:sample_batch_size])
rmse3_train = relative_mse(sample3_train, train_nonlinear[:sample_batch_size])
print('RMSE Test:', rmse3_test)
print('RMSE Train:', rmse3_train)

sample4_test = sampler(None, test_vorticity_short[:sample_batch_size], model4)
sample4_train = sampler(None, train_vorticity_short[:sample_batch_size], model4)
print('Model 4')
rmse4_test = relative_mse(sample4_test, test_nonlinear_short[:sample_batch_size])
rmse4_train = relative_mse(sample4_train, train_nonlinear_short[:sample_batch_size])
print('RMSE Test:', rmse4_test)
print('RMSE Train:', rmse4_train)

sample5_test = sampler(test_nonlinear_smag_short[:sample_batch_size], test_vorticity_short[:sample_batch_size], model5)
sample5_train = sampler(train_nonlinear_smag_short[:sample_batch_size], train_vorticity_short[:sample_batch_size], model5)
print('Model 5')
rmse5_test = relative_mse(sample5_test, test_nonlinear_short[:sample_batch_size])
rmse5_train = relative_mse(sample5_train, train_nonlinear_short[:sample_batch_size])
print('RMSE Test:', rmse5_test)
print('RMSE Train:', rmse5_train)

sample6_test = sampler(test_nonlinear_smag_short[:sample_batch_size], None, model6)
sample6_train = sampler(train_nonlinear_smag_short[:sample_batch_size], None, model6)
print('Model 6')
rmse6_test = relative_mse(sample6_test, test_nonlinear_short[:sample_batch_size])
rmse6_train = relative_mse(sample6_train, train_nonlinear_short[:sample_batch_size])
print('RMSE Test:', rmse6_test)
print('RMSE Train:', rmse6_train)