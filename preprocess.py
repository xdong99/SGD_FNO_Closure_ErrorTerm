import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\DiffusionTerm_Generation')
import matplotlib.pyplot as plt
import torch

import h5py
plt.rcParams["animation.html"] = "jshtml"
from utility import set_seed

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")


##############################
#######  Data Loading ########
##############################
# Load data
device = torch.device('cuda')
#
filename_1 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_data_185_32_32_3000.h5'
filename_2 = 'C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm\\train_data_185_32_32_3000_smag.h5'
# Open the HDF5 file
with h5py.File(filename_1, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device=device)
    filtered_sol = torch.tensor(file['filtered_sol'][()], device=device)
    nonlinear_diff = torch.tensor(file['nonlinear_diff'][()], device=device)

with h5py.File(filename_2, 'r') as file:
    nonlinear_diff_smag = torch.tensor(file['nonlinear_diff_smag'][()], device=device)

##############################
######  Data Preprocess ######
##############################
s = 32
train_batch_size = 180
test_batch_size = 5

# target
nonlinear_train = nonlinear_diff[:train_batch_size, :, :, 2000:]
nonlinear_test = nonlinear_diff[-test_batch_size:, :, :, 2000:]

# conditions
vorticity_train = filtered_sol[:train_batch_size, :, :, 2000:]
vorticity_test = filtered_sol[-test_batch_size:, :, :, 2000:]
nonlinear_prev_train = nonlinear_diff[:train_batch_size, :, :, 1995:2995]
nonlinear_prev_test = nonlinear_diff[-test_batch_size:, :, :, 1995:2995]
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
Ntrain = 180000
Ntest = 5000

train_nonlinear = nonlinear_train[:Ntrain, :, :]
train_nonlinear_smag = nonlinear_smag_train[:Ntrain, :, :]
train_nonlinear_prev = nonlinear_prev_train[:Ntrain, :, :]
train_vorticity = vorticity_train[:Ntrain, :, :]

test_nonlinear = nonlinear_test[:Ntest, :, :]
test_nonlinear_smag = nonlinear_smag_test[:Ntest, :, :]
test_nonlinear_prev = nonlinear_prev_test[:Ntest, :, :]
test_vorticity = vorticity_test[:Ntest, :, :]

filename = 'train_test_180000_32_32.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('t', data=sol_t.cpu().numpy())
    file.create_dataset('train_nonlinear', data=train_nonlinear.cpu())
    file.create_dataset('train_nonlinear_smag', data=train_nonlinear_smag.cpu())
    file.create_dataset('train_vorticity', data=train_vorticity.cpu())
    file.create_dataset('train_nonlinear_prev', data=train_nonlinear_prev.cpu())
    file.create_dataset('test_nonlinear', data=test_nonlinear.cpu())
    file.create_dataset('test_nonlinear_smag', data=test_nonlinear_smag.cpu())
    file.create_dataset('test_vorticity', data=test_vorticity.cpu())
    file.create_dataset('test_nonlinear_prev', data=test_nonlinear_prev.cpu())
print(f'Data saved to {filename}')




##############################
######  Data Preprocess ######
##############################
s = 32
train_batch_size = 22
test_batch_size = 4

# target
nonlinear_train = nonlinear_diff[:train_batch_size, :, :, 2000:2300]
nonlinear_test = nonlinear_diff[-test_batch_size:, :, :, 2000:2300]

# conditions
vorticity_train = filtered_sol[:train_batch_size, :, :, 2000:2300]
vorticity_test = filtered_sol[-test_batch_size:, :, :, 2000:2300]
nonlinear_prev_train = nonlinear_diff[:train_batch_size, :, :, 1995:2295]
nonlinear_prev_test = nonlinear_diff[-test_batch_size:, :, :, 1995:2295]
nonlinear_smag_train = nonlinear_diff_smag[:train_batch_size, :, :, 0:300]
nonlinear_smag_test = nonlinear_diff_smag[-test_batch_size:, :, :, 0:300]


nonlinear_train = nonlinear_train.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_test = nonlinear_test.permute(0,3,1,2).reshape(-1, s, s)

vorticity_train = vorticity_train.permute(0,3,1,2).reshape(-1, s, s)
vorticity_test = vorticity_test.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_smag_train = nonlinear_smag_train.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_smag_test = nonlinear_smag_test.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_prev_train = nonlinear_prev_train.permute(0,3,1,2).reshape(-1, s, s)
nonlinear_prev_test = nonlinear_prev_test.permute(0,3,1,2).reshape(-1, s, s)


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
Ntrain = 6600
Ntest = 1200

train_nonlinear = nonlinear_train[:Ntrain, :, :]
train_nonlinear_smag = nonlinear_smag_train[:Ntrain, :, :]
train_nonlinear_prev = nonlinear_prev_train[:Ntrain, :, :]
train_vorticity = vorticity_train[:Ntrain, :, :]

test_nonlinear = nonlinear_test[:Ntest, :, :]
test_nonlinear_smag = nonlinear_smag_test[:Ntest, :, :]
test_nonlinear_prev = nonlinear_prev_test[:Ntest, :, :]
test_vorticity = vorticity_test[:Ntest, :, :]

filename = 'train_test_short.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('t', data=sol_t.cpu().numpy())
    file.create_dataset('train_nonlinear', data=train_nonlinear.cpu())
    file.create_dataset('train_nonlinear_smag', data=train_nonlinear_smag.cpu())
    file.create_dataset('train_vorticity', data=train_vorticity.cpu())
    file.create_dataset('train_nonlinear_prev', data=train_nonlinear_prev.cpu())
    file.create_dataset('test_nonlinear', data=test_nonlinear.cpu())
    file.create_dataset('test_nonlinear_smag', data=test_nonlinear_smag.cpu())
    file.create_dataset('test_vorticity', data=test_vorticity.cpu())
    file.create_dataset('test_nonlinear_prev', data=test_nonlinear_prev.cpu())
print(f'Data saved to {filename}')

