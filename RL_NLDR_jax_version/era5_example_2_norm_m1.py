# increasing number of probabilities per sample:
# modifying code to track probabilities of a reference sample that is known to give low
# reconstruction error. 
import numpy as np
import math
import matplotlib.pyplot as plt
from jax import grad, vmap
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import argparse
import os
import functools
from flax.core import FrozenDict
# from experiments.Experiment import Experiment
import pickle
import torch
from utils.tools_2 import apply_selected_funcs
from layers.Enc_Dec import Encoder_Decoder
from utils.tools_2 import random_selection_arr_maker
from layers.output_grad_comp import output_selection
import subprocess
import h5py
import scipy.optimize as opt
from utils.tools_2 import apply_selected_funcs
import jax
from typing import Iterator, Tuple
from tqdm import tqdm
from jax import jit

parser = argparse.ArgumentParser(description='era5_example_2')

GAMMA = 0.
library_functions = [ "(_)**2", "(_)**3"]
selection_length = 4
sub_selection_length = 2
d_model = 8
e_layers = 2
batch_size = 10
results_path = "./results/"
num_samples_total = 1000
num_epochs = 100
svd_trunc_index = 20 # l_val
l_val = svd_trunc_index
r_val = 8
lr_mod = 5.

with open('era5_example1_reference_sample.pkl', 'rb') as file:
    reference_sample = pickle.load(file)

ref_sample_array = reference_sample['sample_arr']

gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]

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

X_train = S_train[:,:-1]
Y_train = S_train[:,1:]

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.tensor(X_train, dtype = torch.float32, device = torch_device)
Y_train = torch.tensor(Y_train, dtype = torch.float32, device = torch_device)

U_vals, sing_vals, Vt_vals = torch.linalg.svd(X_train, full_matrices = False)

U_vals = jnp.array(U_vals, device= device)
sing_vals = jnp.array(sing_vals, device = device)
Vt_vals = jnp.array(Vt_vals, device = device)

X_train = jnp.array(X_train, dtype=jnp.float32, device = device)
Y_train = jnp.array(Y_train, dtype=jnp.float32, device = device)

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

U_r = U_l @ phi_mat
X_hat = U_r.T @ X_train
trunc_dim = X_hat.shape[0]

template_filename = 'era5_example2_norm_1_wotanh_tdim{}_slen{}_sslen{}_ns{}_bs{}_ne{}_dm{}_el{}_lr{}'.format(
    trunc_dim,
    selection_length,
    sub_selection_length,
    num_samples_total,
    batch_size,
    num_epochs,
    d_model,
    e_layers,
    lr_mod
)

path = results_path + template_filename + '/'
if not os.path.exists(path):
    os.makedirs(path)

print(path)

lr_sample_rewards_filename = 'lr_sample_rewards.pkl'
learning_rate_batch_sample_rewards_filename = path + lr_sample_rewards_filename

lrs_sample_rewards = {}
lrs_sample_rewards['modified_learning_rates'] = []
lrs_sample_rewards['sample_rewards_batch'] = []
lrs_sample_rewards['avg_sample_rewards_batch'] = []
lrs_sample_rewards['sample_reconstr_err_batch'] = []

seen = set()
sample_selection_arrays = []
sample_length = trunc_dim * len(library_functions)

while len(sample_selection_arrays) < num_samples_total:
    # build one sample_arr
    pieces = []
    for _ in range(sample_length // selection_length):
        pieces.append(
            random_selection_arr_maker(selection_length, sub_selection_length)
        )
    # concatenate back to a 1D array of length sample_length
    sample_arr = np.concatenate(pieces, axis=0)
    
    # use a tuple of ints as a hashable key
    key = tuple(int(x) for x in sample_arr)
    if key in seen:
        # duplicate: skip, generate a new one
        continue
    
    # new unique sample_arr
    seen.add(key)
    sample_selection_arrays.append(sample_arr)

# stack into an array of shape (num_samples_total, sample_length)
sample_selection_arrays = np.stack(sample_selection_arrays, axis=0)

# print(type(sample_selection_arrays)) # np.array
# sample_arr_dicts = {tuple(row): [] for row in selection_arr_data.tolist()}
# ****************************************************************
from models.models_2.model_reward_norm_1 import Model

# k = len(selection_arr)
# ones = sum(selection_arr)
import itertools
base = [1]*(sub_selection_length) + [0]*(selection_length-sub_selection_length)
perms = list(set(itertools.permutations(base)))  # list of tuples

network = Encoder_Decoder(selection_length + 1, d_model, e_layers)
key = jax.random.PRNGKey(0)
test_inp = [1]*(sub_selection_length+1) + [0]*(selection_length-sub_selection_length)
x_dummy = jnp.array(test_inp)
params = network.init(key, x_dummy)
apply_fn = jax.jit(network.apply)

model_settings = {
    "library_functions": library_functions,
    'gamma': GAMMA,
    'X_train': X_train,
    'Y_train': Y_train,
    'X_tilde': X_tilde,
    'U_r': U_r,
    'A_tilde_operator': A_tilde_operator,
    'X_hat': X_hat,
    'phi_mat': phi_mat,
    'trunc_dim': trunc_dim,
    'selection_length': selection_length,
    'sub_selection_length': sub_selection_length,
    'd_model': d_model,
    'e_layers': e_layers,
    'batch_size': batch_size,
    'reference_sample': None,
    "lam_vec": jnp.logspace(-4, 4, 6),
    "permutations": perms,
    "apply_fn": apply_fn,
    "params": params
}

model = Model(model_settings)


@jax.jit
def model_forward_jit(model_params, batch_sel_arrs: jnp.ndarray):
    # if your Module is called "model", then
    return model.forward(model_params, batch_sel_arrs)


# @jax.jit
def update_params(params, grads, lr_mod, batch_size):
    return jax.tree_map(
        lambda w, g: w + (lr_mod / batch_size) * g,
        params,
        grads
    )


best_sample_list = [] # best sample in each epoch
batch_reward_list = [] # preserving rewards of all batches for plot.

track_samples = []
track_probs = []

num_batches_per_epoch = num_samples_total//batch_size
rng = jax.random.PRNGKey(42)
preserved_params = [model.params]
preserved_lrs = [lr_mod]


# test_sample = jnp.array([1,0,1,0,0,0,1,1,1,1,0,0,0,1,1,0])

# test_sample = jnp.array(sample_selection_arrays[0,:])
# print(test_sample)
# preserved_prob_hists = []  

for epoch in tqdm(range(num_epochs)):        
    best_batch_reward = -jnp.inf

    rng, perm_key = jax.random.split(rng)
    perm = jax.random.permutation(perm_key, sample_selection_arrays.shape[0])
    shuffled = sample_selection_arrays[perm]

    # 2) iterate over mini‑batches of the shuffled data
    num_samples = shuffled.shape[0]

    # call batch training function:

    for batch_idx in range(num_batches_per_epoch):
        # print(batch_idx * batch_size, ":", (batch_idx+1) * batch_size)
        batch_train_data = shuffled[batch_idx * batch_size : (batch_idx+1) * batch_size]   
        batch_sel_arrs = jnp.array(batch_train_data, dtype = jnp.float32, device = device)

        # print(batch_sel_arrs)
        param_vals = model.params
        batch_reconstr_vals, batch_grads, batch_rewards, best_sample_batch, prob_hist = model_forward_jit(model.params, batch_sel_arrs)

        good_mask = ~jnp.isnan(batch_reconstr_vals)                     # (B,)
        good_idxs = jnp.nonzero(good_mask)[0]    # (B,), zeros for bad slots

        if(len(good_idxs) == 0):
            continue

        batch_rewards =  batch_rewards[good_idxs]
        batch_grads = jax.tree_map(lambda g: g[good_idxs], batch_grads)
        prob_hist = prob_hist[good_idxs]
        batch_sel_arrs = batch_sel_arrs[good_idxs]

        # mask = jnp.all(batch_sel_arrs == test_sample, axis=1)  

        # # transfer mask & hist to host (NumPy) to use standard Python logic
        # mask_host     = np.array(mask)
        # prob_hist_host = np.array(prob_hist)

        # if mask_host.any():
        #     # for each matching index, save its history
        #     for idx in np.where(mask_host)[0]:
        #         preserved_prob_hists.append(prob_hist_host[idx])

        # leaves, _ = jax.tree_util.tree_flatten(batch_grads)
        # # stack all values into one big 1D array
        # all_vals = jnp.concatenate([leaf.ravel() for leaf in leaves])
        # # count positives, negatives, and total (you can also count zeros if you like)
        # n_total = all_vals.size
        # n_pos   = jnp.sum(all_vals > 0)
        # n_neg   = jnp.sum(all_vals < 0)
        # n_zero   = jnp.sum(all_vals == 0)
        # # compute fractions
        # frac_pos = n_pos / n_total
        # frac_neg = n_neg / n_total
        # frac_zero = n_zero/ n_total
        # print(frac_pos, ":", frac_neg, ":", frac_zero)

        batch_rewards_scaled = -batch_rewards/jnp.sum(batch_rewards) 

        # global_max_grad = jax.tree_util.tree_reduce(
        #     # reducer: take the greater of accumulator `a` and max of this leaf `x`
        #     lambda a, x: jnp.maximum(a, jnp.max(x)),
        #     batch_grads,
        #     initializer=-jnp.inf
        # )

        # global_max_par = jax.tree_util.tree_reduce(
        #     # reducer: take the greater of accumulator `a` and max of this leaf `x`
        #     lambda a, x: jnp.maximum(a, jnp.max(x)),
        #     model.params,
        #     initializer=-jnp.inf
        # )

        # print(batch_rewards_scaled)
        # print(batch_rewards_scaled, ":", 1e-1/abs(min(batch_rewards_scaled)))

        batch_grads = jax.tree.map(lambda g: g * batch_rewards_scaled[:, None, None], batch_grads)

        track_samples.append(batch_sel_arrs)
        track_probs.append(prob_hist)

        total_batch_grads = jax.tree_map(
            lambda g: g.sum(axis=0),
            batch_grads
        )

        total_batch_reward = jax.tree_map(jnp.sum, batch_rewards)

        batch_reward_list.append(total_batch_reward.item()) # preserving rewards of all batches for plot.

        if(total_batch_reward >  best_batch_reward): 
            best_batch_reward = total_batch_reward
            best_sample_epoch = best_sample_batch

            # update grads of network after each batch of 'm' samples:        
        #     break
        # break

        # lr = update_grads(model, total_batch_grads, path, batch_rewards_scaled)

        lrs_sample_rewards['modified_learning_rates'].append(lr_mod)
        lrs_sample_rewards['sample_rewards_batch'].append(batch_rewards)
        lrs_sample_rewards['sample_reconstr_err_batch'].append(batch_reconstr_vals)

        new_params = update_params(param_vals, total_batch_grads, lr_mod, len(batch_rewards))
        model.params = new_params

    #     break
    # break


    # best sample from batch with highest reward for each epoch.
    preserved_params.append(model.params)
    preserved_lrs.append(lr_mod)
    best_sample_list.append(best_sample_epoch)


# from models.models_2.model_reward_norm_1 import inside_param_vals
# inside_params = np.array(inside_param_vals())


final_results = {
    "batch_reward_list": batch_reward_list,
    "best_sample_list": best_sample_list,
    "lrs_sample_rewards": lrs_sample_rewards,
    'track_samples': track_samples,
    'track_probs': track_probs,
    "saved_params": preserved_params,
    'saved_lrs': preserved_lrs,
    # 'preserved_prob_hists': np.array(preserved_prob_hists),
    # "inside_params": inside_params
}

with open(path + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)




#****************************************************************************************************

