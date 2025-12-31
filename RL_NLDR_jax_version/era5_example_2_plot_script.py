# script to make plots:
import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
from jax import grad, vmap
import math
from scipy.integrate import solve_ivp
from utils.tools_2 import apply_selected_funcs
import h5py

GAMMA = 0.
library_functions = [ "(_)**2", "(_)**3"]
selection_length = 4
sub_selection_length = 2
d_model = 8
e_layers = 2
learning_rate = 5.
batch_size = 10
results_path = "./results/"
num_samples_total = 1000
num_epochs = 100
svd_trunc_index = 20 # l_val
l_val = svd_trunc_index
r_val = 8

gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]

file_path = "/workspace/venu_files/climate_forecasting/RL_NLDR_jax_version/results/era5_example2_norm_1_wotanh_tdim8_slen4_sslen2_ns1000_bs10_ne100_dm8_el2_lr5.0_dyn_learning_rate_M1/"
result_file = "final_results.pkl"

with open(file_path+result_file, 'rb') as file:
    loaded_data = pickle.load(file)

batch_reward_list = loaded_data['batch_reward_list']
best_sample_list = loaded_data['best_sample_list']
lrs_sample_rewards = loaded_data['lrs_sample_rewards']

# print(batch_reward_list)
# print(best_sample_list[1])
# print(batch_reward_list)

# # plots to make:
# 1. batch rewards
plt.plot(batch_reward_list)
plt.xlabel('batch index', fontsize = 16)
plt.ylabel('reward', fontsize = 16)
plt.title('Batch rewards')
plt.savefig(f'{file_path}batch_rewards.png')
plt.close()


# 2. avg. batch reward per epoch
num_batches_per_epoch = int(num_samples_total/batch_size)

avg_batch_reward_epoch = []
for i in range(num_epochs):
    # print(batch_reward_list[i * num_batches_per_epoch : (i+1)*num_batches_per_epoch ])
    avg_batch_reward_epoch.append( np.mean( batch_reward_list[i * num_batches_per_epoch : (i+1)*num_batches_per_epoch ]  )   )

plt.plot(avg_batch_reward_epoch)
plt.xlabel('epoch index', fontsize = 16)
plt.ylabel('reward', fontsize = 16)
plt.title('Average batch reward per epoch')
plt.tight_layout()
plt.savefig(f'{file_path}avg_batch_reward_per_epoch.png')
plt.close()

# 3. avg. sample reward per epoch
sample_rewards = lrs_sample_rewards['sample_rewards_batch']

avg_sample_reward_epoch = []
for i in range(num_epochs):
    sample_rewards_each_epoch = sample_rewards[i * num_batches_per_epoch : (i+1)*num_batches_per_epoch ]
    # print(sample_rewards_each_epoch)
    all_vals = np.concatenate(sample_rewards_each_epoch)
    avg_sample_reward_epoch.append(all_vals.mean().item())
    
plt.plot(avg_sample_reward_epoch)
plt.xlabel('epoch index', fontsize = 16)
plt.ylabel('reward', fontsize = 16)
plt.title('Average sample reward per epoch')
plt.tight_layout()
plt.savefig(f'{file_path}avg_sample_reward_per_epoch.png')
plt.close()


# probs
#**************************************************************************
# with open('era5_example1_reference_sample.pkl', 'rb') as file:
#     reference_sample = pickle.load(file)

# # prob values:
# ref_sample = tuple(reference_sample['sample_arr'].tolist())

# ref_probs = []

# for i in range(len(track_samples)):
#     for j in range(len(track_samples[i])):
#         if(np.allclose(ref_sample, track_samples[i][j])):
#             ref_probs.append(track_probs[i][j])

# print(ref_probs)
#**************************************************************************

# pick a sample that has low ROM reconstruction error and appears multiple times. See if it
# the assigned prob array increases. Do same for a sample with high reconstruction error
# and see if the prob array decreases.
# print(len(track_samples))
# from collections import Counter
# cnt = Counter(map(tuple, track_samples))
# print(cnt)

# print(len(track_samples), track_samples[0].shape)

# print(len(track_probs), track_probs[0].shape)

# keys = []
# for arr in track_samples:
#     # arr.tolist() is a Python list of Python ints/floats
#     tup = tuple(arr.tolist())
#     print(tup)
#     break
    # keys.append(tup)

# cnt = Counter(keys)
# print(cnt)

best_sample_min_reconstr_err = min(best_sample_list, key=lambda d: d['reconstr_err'])
best_sample_max_reward = max(best_sample_list, key=lambda d: d['sample_reward'])

sel_arr_min_reconstr_err = best_sample_min_reconstr_err['selection_arr']
sel_arr_max_reward = best_sample_max_reward['selection_arr']

with open("era5_example_1_data.pkl", 'rb') as file:
    loaded_data = pickle.load(file)

data_var = loaded_data['u10m_values']
lat_vals = loaded_data['lat_vals']
long_vals = loaded_data['long_vals']

train_split = int(0.8*data_var.shape[0])
S_train_org = data_var[:train_split]
S_test_org = data_var[train_split:]

S_train = S_train_org.reshape(S_train_org.shape[0], S_train_org.shape[1] * S_train_org.shape[2]).T
S_test = S_test_org.reshape(S_test_org.shape[0], S_test_org.shape[1] * S_test_org.shape[2]).T

print(S_train.shape, ":", S_test.shape)
# print(S_train.shape, ":",   S_test.shape) #(18048, 1425) : (4653, 1425)

X_train = S_train[:,:-1]
Y_train = S_train[:,1:]

import torch
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.tensor(X_train, dtype = torch.float32, device = torch_device)
Y_train = torch.tensor(Y_train, dtype = torch.float32, device = torch_device)

U_vals, sing_vals, Vt_vals = torch.linalg.svd(X_train, full_matrices = False)

U_vals = jnp.array(U_vals, device= device)
sing_vals = jnp.array(sing_vals, device = device)
Vt_vals = jnp.array(Vt_vals, device = device)

X_train = jnp.array(X_train, dtype=jnp.float32, device = device)
Y_train = jnp.array(Y_train, dtype=jnp.float32, device = device)

# print(X_train.shape, ":", )

# svd truncation index : 3.

U_l = U_vals[:, :l_val]
Sig_l = sing_vals[:l_val]
Vt_l = Vt_vals[:l_val, :]

A_operator = Y_train @ Vt_l.T @ jnp.linalg.inv(jnp.diag(Sig_l)) @ U_l.T

A_tilde_operator = U_l.T @ A_operator @ U_l

X_tilde = U_l.T @ X_train

phi_mat = jnp.vstack([
    jnp.eye(r_val, dtype=jnp.float32),
    jnp.zeros((l_val - r_val, r_val), dtype=jnp.float32),
])

# \tilde{X}, \hat{X}:
U_r = U_l @ phi_mat
X_hat = U_r.T @ X_train
trunc_dim = X_hat.shape[0]

# sel_arr_min_reconstr_err = best_sample_min_reconstr_err['selection_arr']
# sel_arr_max_reward = best_sample_max_reward['selection_arr']

# test_val = best_sample_min_reconstr_err['phi_bar_mat']
# print(test_val.shape)

# testing reconstruction:
A_hat = phi_mat.T @ A_tilde_operator @ phi_mat
H_hat = phi_mat.T @ A_tilde_operator @ best_sample_min_reconstr_err['phi_bar_mat']


print(A_hat.shape, ":", H_hat.shape)

selected_indices = [i for i, v in enumerate(sel_arr_min_reconstr_err) if v]
idx_array = jnp.array(selected_indices, dtype=jnp.int32)  # (m_sel,)

x0 = Y_train[:, -1]
x_hat0 = U_r.T @ x0

from utils.tools_2 import make_library_functions
funcs_list = make_library_functions(library_functions)
library_functions = tuple(funcs_list)

def step(xh, _):
    mod = apply_selected_funcs(xh, library_functions)
    mod_sel = jnp.take(mod, idx_array)
    # jax.debug.print("{} : {}", mod, mod_sel)
    # xh_next = A_hat @ xh + H_hat_sel @ mod_sel             # (r,)
    xh = A_hat @ xh + H_hat @ mod_sel             # (r,)

    return xh, xh

# Roll forward T steps
_, xh_seq = jax.lax.scan(step, x_hat0, None, length=S_test.shape[1])
# Reconstruct full state and compute ‑norm error
X_pred = U_r @ xh_seq.T
  

print(X_pred.shape, ":", S_test.shape)

# X_reconstr = U_r.cpu().numpy() @ np.array(X_reconstr).T
testing_reconstr_err = []

# testing_data_norm_vals
testing_data_norm_vals = []
pred_data_norm_vals = []

for i in range(S_test.shape[1]):
    testing_data_norm_vals.append(np.linalg.norm(S_test[:,i]))
    pred_data_norm_vals.append(np.linalg.norm(X_pred[:,i]))

plt.plot(testing_data_norm_vals[:100], label = 'test data')
plt.plot(pred_data_norm_vals[:100], label = 'pred data')
plt.xlabel('index', fontsize = 16)
plt.ylabel('Value', fontsize = 16)
plt.legend()
plt.title('Testing data norm values')
plt.savefig(f'{file_path}test_pred_data_norm_vals.png')
plt.close()

# print(X_pred.shape, ":",  S_test.shape)
for i in range(S_test.shape[1]):
    # print(i,":",np.linalg.norm(X_pred[:,i]), ":", np.linalg.norm(S_test[:,i]) )
    testing_reconstr_err.append(np.linalg.norm(S_test[:,i] - X_pred[:,i])/np.linalg.norm(S_test[:,i]))

np.save(f'{file_path}testing_error_reconstruction_era5_example2_norm_1.npy', testing_reconstr_err)

print(min(testing_reconstr_err), ":", max(testing_reconstr_err))
plt.plot(testing_reconstr_err)
plt.xlabel('Time index', fontsize = 16)
plt.ylabel('Error', fontsize = 16)
plt.title('Relative error for testing data predictions')
plt.savefig(f'{file_path}test_pred_error_plot.png')



# # ************************************

# print(avg_reward_epoch)   

# reconstr_err = np.linalg.norm( S_test - X_pred)
# print(reconstr_err)

