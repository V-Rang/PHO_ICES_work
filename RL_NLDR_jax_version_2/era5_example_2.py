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
from layers.Enc_Dec import Encoder_Decoder
from utils.tools_1 import random_selection_arr_maker, rom_reconstruction_error, sample_err_computation
# from layers.output_grad_comp import output_selection
import subprocess
import h5py
import scipy.optimize as opt
from utils.tools_1 import apply_selected_funcs, make_library_functions, lstsq_l2
import jax
from typing import Iterator, Tuple
from tqdm import tqdm
from jax import jit
from functools import partial

parser = argparse.ArgumentParser(description='era5_example_2')

GAMMA = 0.
library_functions = [ "(_)**2", "(_)**3"]
selection_length = 4
sub_selection_length = 2
d_model = 3
e_layers = 1
learning_rate = 1e-3
batch_size = 100
results_path = "./results/"
num_samples_total = 1000
num_epochs = 100
svd_trunc_index = 20 # l_val
l_val = svd_trunc_index
r_val = 8
lam_vec = jnp.logspace(-4, 4, 6)

funcs_list = make_library_functions(library_functions)
library_functions = tuple(funcs_list)

with open('era5_example1_reference_sample.pkl', 'rb') as file:
    reference_sample = pickle.load(file)

ref_sample_array = reference_sample['sample_arr']

# print(type(ref_sample_array), ":", ref_sample_array)

gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]

# print("Using device:", device) # cuda:0

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

with open("../CAE_transf/tacc_vals.pkl", 'rb') as f:
    tacc_vals = pickle.load(f)

X_train = tacc_vals['X_train']
Y_train = tacc_vals['Y_train']
U_vals = tacc_vals['U_vals']
sing_vals = tacc_vals['sing_vals']
Vt_vals = tacc_vals['Vt_vals']

U_l = U_vals[:, :l_val]
Sig_l = sing_vals[:l_val]
Vt_l = Vt_vals[:l_val, :]

# print(type(U_l), type(X_train))

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
X_hat_mod = apply_selected_funcs(X_hat, library_functions)
lhs_mat = X_tilde - phi_mat @ X_hat

trunc_dim = X_hat.shape[0]
A_hat = phi_mat.T @ A_tilde_operator @ phi_mat

template_filename = 'era5_example2_wotanh_tdim{}_slen{}_sslen{}_ns{}_bs{}_ne{}_dm{}_el{}_unmod_lr'.format(
    trunc_dim,
    selection_length,
    sub_selection_length,
    num_samples_total,
    batch_size,
    num_epochs,
    d_model,
    e_layers
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
    # print(key)
    # print(sample_arr)
  
    if key in seen:
        # duplicate: skip, generate a new one
        continue
    
    # new unique sample_arr
    seen.add(key)
    sample_selection_arrays.append(sample_arr)

# stack into an array of shape (num_samples_total, sample_length)
sample_selection_arrays = np.stack(sample_selection_arrays, axis=0)

sample_err_computation_fn = jax.vmap(sample_err_computation, 
                                in_axes=(
                                None, # X_train
                                None, # Y_train
                                None, # X_hat_mod
                                None, # lhs_mat
                                None, # lam_vec
                                None, # phi_mat
                                None, # A_hat
                                None, # A_tilde
                                None, # U_r
                                None, # library_functions
                                0     # sample_arr
                                )
    )


# sample_errs, _ = sample_err_computation_fn(X_train, Y_train, X_hat_mod,
#                             lhs_mat, lam_vec,   
#                             phi_mat, A_hat, A_tilde_operator, 
#                             U_r, library_functions, 
#                             sample_selection_arrays)


# print(sample_selection_arrays.shape, sample_errs.shape)

chunk_fn = jax.jit(partial(
    sample_err_computation_fn,
    X_train, Y_train, X_hat_mod,
    lhs_mat, lam_vec,
    phi_mat, A_hat, A_tilde_operator, U_r, library_functions
))

# now iterate over sample_selection_arrays in batches:
def batched_sample_errs(sample_selection_arrays, batch_size):
    n = sample_selection_arrays.shape[0]
    errs_list = []
    opts_list = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        chunk = sample_selection_arrays[start:end]
        # if your final chunk is smaller, you can either pad it
        # to batch_size (and then slice off the extra results),
        # or just re‑compile for the smaller shape (it’s usually ok once).
        errs_chunk, _ = chunk_fn(chunk)
        errs_list.append(errs_chunk)
        # opts_list.append(opts_chunk)
    # concatenate back into full arrays
    all_errs = jnp.concatenate(errs_list, axis=0)
    # all_opts = jnp.concatenate(opts_list, axis=0)
    return all_errs

# usage:
sample_errs = batched_sample_errs(sample_selection_arrays, batch_size)

mask = ~np.isnan(sample_errs)

sample_errs = sample_errs[mask]
sample_selection_arrays = sample_selection_arrays[mask]

samples_arrays_errs = {
    'sample_selection_arrays': sample_selection_arrays,
    'sample_errs': sample_errs,
}

# print(sample_errs.shape, sample_selection_arrays.shape) # (963,) (963, 16)
# # (98,) (98, 16)

with open(path + "samples_arrays_errs.pkl", 'wb') as f:
    pickle.dump(samples_arrays_errs, f)

# # ****************************************************************
from models.model_1.model_v1 import Model

import itertools
base = [1]*(sub_selection_length) + [0]*(selection_length-sub_selection_length)
perms = list(set(itertools.permutations(base)))  # list of tuples

# print(perms)
# print(sample_selection_arrays[0])

network = Encoder_Decoder(selection_length, d_model, e_layers)
key = jax.random.PRNGKey(0)
test_inp = [1]*(sub_selection_length) + [0]*(selection_length-sub_selection_length)
x_dummy = jnp.array(test_inp)
params = network.init(key, x_dummy)
apply_fn = jax.jit(network.apply)

model_settings = {
    "apply_fn": apply_fn,
    "permutations": perms,
    'selection_length': selection_length,
    'params': params
}

model = Model(model_settings)

@jax.jit
def model_forward_jit(params, batch_sel_arrs: jnp.ndarray):
    return model.forward(params, batch_sel_arrs)

def update_params(params, grads, lr_mod, batch_size):
    return jax.tree.map(
        lambda w, g: w + (lr_mod / batch_size) * g,
        params,
        grads
    )

best_sample_list = [] # best sample in each epoch
batch_reward_list = [] # preserving rewards of all batches for plot.

track_samples = []
track_probs = []

tracked_errs = []
track_scaled_samples_grads = []

num_samples_total = sample_selection_arrays.shape[0]
num_batches_per_epoch = num_samples_total//batch_size
rng = jax.random.PRNGKey(42)
preserved_params = []
# preserved_lrs = [model.learning_rate]
preserved_sum_batch_grads = []
preserved_lrs = []

for epoch in tqdm(range(num_epochs)):        
    # best_batch_reward = -jnp.inf
    best_batch_reward_each_epoch = -jnp.inf

    rng, perm_key = jax.random.split(rng)
    perm = jax.random.permutation(perm_key, sample_selection_arrays.shape[0])
    shuffled = sample_selection_arrays[perm]
    shuffled_errs = sample_errs[perm]

    num_samples = shuffled.shape[0]
    
    for batch_idx in range(num_batches_per_epoch):
        batch_train_data = shuffled[batch_idx * batch_size : (batch_idx+1) * batch_size]   
        batch_sel_arrs = jnp.array(batch_train_data, dtype = jnp.float32, device = device)

        batch_errs = shuffled_errs[batch_idx * batch_size : (batch_idx+1) * batch_size]   
        batch_sel_errs = jnp.array(batch_errs, dtype = jnp.float32, device = device)

        param_vals = model.params

        batch_grads, batch_prob_hist = model_forward_jit(model.params, batch_sel_arrs)

        # print(batch_prob_hist)
        # print(batch_sel_errs)
    #     break
    # break

        # preserved_params.append(params)

        batch_rewards = -(batch_sel_errs)**2

        def scale_by_error(grad):
            shape = (batch_rewards.shape[0],) + (1,) * (grad.ndim - 1)
            return grad * batch_rewards.reshape(shape)

        error_scaled_sample_grads = jax.tree.map(scale_by_error, batch_grads)
        
        total_batch_grads = jax.tree.map(
                lambda g: jnp.sum(g, axis=0),
                error_scaled_sample_grads
        )

        track_samples.append(batch_sel_arrs)
        # track_scaled_samples_grads.append(error_scaled_sample_grads)
        track_probs.append(batch_prob_hist)
        # tracked_errs.append(batch_sel_errs)

        # total_batch_reward = jnp.sum(batch_rewards)
        total_batch_reward = jax.tree.map(jnp.sum, batch_rewards)

        batch_reward_list.append(total_batch_reward.item()) # preserving rewards of all batches for plot.
        preserved_sum_batch_grads.append(total_batch_grads)

        # print(batch_rewards)

        best_idx = jnp.argmax(batch_rewards)
        best_batch_reward = batch_rewards[best_idx]

        best_sample_batch = {'selection_arr': batch_sel_arrs[best_idx], 
                            'probability_arr': batch_prob_hist[best_idx], 
                            'reconstr_err': batch_sel_errs[best_idx], 
                            'sample_reward': batch_rewards[best_idx] 
                            }

        # print(total_batch_reward, ":", best_batch_reward)

        if(best_batch_reward_each_epoch <  best_batch_reward): 
            best_batch_reward_each_epoch = best_batch_reward
            best_sample_epoch = best_sample_batch

        # best_sample_batch:
        # adv = batch_rewards - jnp.mean(batch_rewards)
        # adv = adv / (jnp.std(adv) + 1e-6)
        # lr_mod = 1e-3/min(abs(adv))           
        lr_mod = learning_rate

        # lrs_sample_rewards['modified_learning_rates'].append(lr_mod)
        # lrs_sample_rewards['sample_rewards_batch'].append(batch_rewards)
        # lrs_sample_rewards['sample_reconstr_err_batch'].append(batch_sel_errs)

        new_params = update_params(param_vals, total_batch_grads, lr_mod, len(batch_rewards))
        model.params = new_params

    preserved_params.append(new_params)



    best_sample_list.append(best_sample_epoch)

final_results = {
    'track_samples': track_samples,
    'track_probs': track_probs,
    "batch_reward_list": batch_reward_list,
    "preserved_params": preserved_params,
    'preserved_sum_batch_grads': preserved_sum_batch_grads
    }

with open(path + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)

#****************************************************************************************************
