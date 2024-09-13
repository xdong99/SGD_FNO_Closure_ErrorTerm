import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm')

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
plt.rcParams["animation.html"] = "jshtml"
from Data_Generation.generator_sns import (navier_stokes_2d_orig,
                                           navier_stokes_2d_filtered,
                                           navier_stokes_2d_smag)
from Data_Generation.random_forcing import GaussianRF
from timeit import default_timer
import cupy as cp
from cupyx.scipy.signal import convolve2d
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm')

def local_avg(tensor, kernel):
    B, H, W = tensor.shape
    filtered_tensor = cp.zeros((B, H, W))
    for b in range(B):
        filtered_slice = convolve2d(tensor[b, :, :], kernel, mode='same', boundary='wrap')
        filtered_tensor[b, :, :] = filtered_slice
    return filtered_tensor


filename = "./Data_Generation/"
device = torch.device('cuda')

kernel = torch.ones(16, 16) / 256.0
kernel = cp.from_dlpack(kernel.to('cuda'))
downscale = 8

# Viscosity parameter
nu = 1e-4

# Spatial Resolution
s = 256
les_size = 32
sub = 1

# Temporal Resolution
T = 30
delta_t = 1e-3

# Number of solutions to generate
N = 300

bsize = 10

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
f_filtered = local_avg(cp.from_dlpack(f.unsqueeze(0)) , kernel)
f_filtered = torch.tensor(f.squeeze(0)).to(device)
f_filtered = f[::downscale, ::downscale]

# Stochastic forcing function: sigma*dW/dt
stochastic_forcing = {'alpha': 0.005, 'kappa': 10, 'sigma': 0.05}

# Number of snapshots from solution
record_steps = 3000

filtered_sol_col = torch.zeros(N, s // downscale, s // downscale, record_steps).to(device)
nonlinear_diff_col = torch.zeros(N, s // downscale, s // downscale, record_steps).to(device)

for j in range(N // bsize):
    w0 = GRF.sample(bsize)

    _, filtered_sol, nonlinear_diff, _, sol_t = navier_stokes_2d_filtered([1, 1], w0, f, nu, T, kernel, downscale,
                                                                                         delta_t=delta_t, record_steps=record_steps,
                                                                                         stochastic_forcing=None)
    filtered_sol_col[j*bsize:(j+1)*bsize] = filtered_sol
    nonlinear_diff_col[j*bsize:(j+1)*bsize] = nonlinear_diff

# List to store indices of non-NaN batches
valid_indices = []

# Iterate over each batch
for i in range(filtered_sol_col.size(0)):
    batch = filtered_sol_col[i]
    # Check if there are any NaN values in the batch
    if not torch.isnan(batch).any():
        valid_indices.append(i)

# Create a new tensor with only the non-NaN batches
filtered_sol_col = filtered_sol_col[valid_indices]
nonlinear_diff_col = nonlinear_diff_col[valid_indices]

filtered_sol_np = filtered_sol_col.cpu().numpy()
nonlinear_diff_np = nonlinear_diff_col.cpu().numpy()

nu_np = np.array(nu)

filename = 'train_data_185_32_32_3000.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('t', data=sol_t.cpu().numpy())
    file.create_dataset('filtered_sol', data=filtered_sol_np)
    file.create_dataset('nonlinear_diff', data=nonlinear_diff_np)
print(f'Data saved to {filename}')





##############################
#######  Data Loading ########
##############################
# Load data
device = torch.device('cuda')
#
filename_1 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_data_185_32_32_3000.h5'
# Open the HDF5 file
with h5py.File(filename_1, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device=device)
    filtered_sol = torch.tensor(file['filtered_sol'][()], device=device)
    nonlinear_diff = torch.tensor(file['nonlinear_diff'][()], device=device)

T = 10
record_steps = 1000

filtered_sol_smag, nonlinear_smag = navier_stokes_2d_smag([1, 1], filtered_sol[..., 2000], f_filtered, nu, T,
                                                          delta_t=delta_t, record_steps=record_steps)


filtered_sol_smag_np = filtered_sol_smag.cpu().numpy()
nonlinear_smag_np = nonlinear_smag.cpu().numpy()
filename = 'train_data_185_32_32_3000_smag.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('filtered_sol_smag', data=filtered_sol_smag_np)
    file.create_dataset('nonlinear_diff_smag', data=nonlinear_smag_np)



sol_nocorr, _, _, _ = navier_stokes_2d_orig([1, 1], filtered_sol[..., 2000], f_filtered, nu, T,
                                               delta_t=delta_t, record_steps=record_steps)




def relative_mse(tensor1, tensor2):
    """Calculate the Relative Mean Squared Error between two tensors."""
    rel_mse = torch.mean(torch.norm(tensor1 - tensor2, 2, dim=(-2, -1)) / torch.norm(tensor2, 2, dim=(-2, -1)))
    return rel_mse

def cal_mse(tensor1, tensor2):
    """Calculate the Mean Squared Error between two tensors."""
    mse = torch.mean((tensor1 - tensor2)**2)
    return mse

# Assuming 'vorticity_series', 'vorticity_NoG', and 'sol' are preloaded tensors
shifter = 2000
k = 5

# Create a figure and a grid of subplots
fig, axs = plt.subplots(5, 5, figsize=(25, 27), gridspec_kw={'width_ratios': [1]*4 + [1.073]})

# Plot each row using seaborn heatmap
for row in range(5):
    for i in range(5):  # Loop through all ten columns
        ax = axs[row, i]

        j = i * 249
        generated =nonlinear_smag[k, :, :, j].cpu()
        generated_nog = nonlinear_smag[k, :, :, j].cpu()
        truth = nonlinear_diff[k, :, :, shifter + j].cpu()
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








