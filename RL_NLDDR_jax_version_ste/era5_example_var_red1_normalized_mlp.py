'''
x (Nh) (inp1) -> E1 -> \hat{x}(r,) -> Library -> \hat{x}_{mod}(r * len(lib)) -> 
E2 -> \hat{x}_{nl} (p) (out1)

Dynamics network:
[t, selected r spatial points] -> DFF (p,) (out2) -> Decoder -> xreconstr(Nh) (out3)

Loss1 = out1, out2
Loss2 = inp1, out3
'''
import numpy as np
import math
import jax
import jax.numpy as jnp
import os
import pickle
import jax
import math
from flax.training import checkpoints
import argparse
from utils.tools_1_normalized_m2 import make_library_functions, apply_selected_funcs

# jax.config.update("jax_debug_nans", True)   # crash at first NaN/Inf creation
# jax.config.update("jax_disable_jit", True)  # one step only, to get a Python traceback

parser = argparse.ArgumentParser(description='era5_example_2')

num_epochs = 10
epoch_threshold = 5
lambda_l1 = 0. # initial value
lambda_increment = 1e-3

lr = 1e-3
lam_vec = jnp.logspace(-4, 4, 6)

library_functions = [ "(_)**2", "(_)**3", "(_)**4"]
library_functions = make_library_functions(library_functions)
library_functions = tuple(library_functions)

batch_size = 40
l_val = 20
pred_len = 10
decoder_net = (32, 32)
repulsion_coeff = 1e-1

# for each of the 50 inputs -> make predictions for pred_len. 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import jax 

with open("../data/u_10m_comp_vals_3.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

S_train = loaded_data['train']['data']
train_times = loaded_data['train']['time']
Nh_val = S_train.shape[0]

S_test = loaded_data['test']['data']
test_times = loaded_data['test']['time']

S_val = loaded_data['val']['data']
val_times = loaded_data['val']['time']

lat_vals = loaded_data['lat_vals']
lon_vals = loaded_data['lon_vals']

X_train = S_train[: , :-1]
Y_train = S_train[: , 1:]

U_vals = loaded_data['U_vals']
sing_vals = loaded_data['sing_vals']
Vt_vals = loaded_data['Vt_vals']
r95 = loaded_data['r95']

U_l = U_vals[:, : l_val]
Sig_l = sing_vals[: l_val]
Vt_l = Vt_vals[: l_val, :]

A_operator = Y_train @ Vt_l.T @ jnp.linalg.inv(jnp.diag(Sig_l)) @ U_l.T
A_tilde_operator = U_l.T @ A_operator @ U_l

X_train_tilde = U_l.T @ X_train
Y_train_tilde = U_l.T @ Y_train

min_tilde, max_tilde =  X_train_tilde.min(), X_train_tilde.max()
X_train_tilde_normalized = (X_train_tilde - min_tilde)/ (max_tilde - min_tilde)
Y_train_tilde_normalized = (Y_train_tilde - min_tilde)/ (max_tilde - min_tilde)

c_tilde = 1/(max_tilde - min_tilde) * (  A_tilde_operator @ (min_tilde * jnp.ones(A_tilde_operator.shape[0])  ) - min_tilde*jnp.ones(A_tilde_operator.shape[0])  )

X_val = S_val[: , :-1]
Y_val = S_val[: , 1:]

lhs_mat_train_tilde_norm = Y_train_tilde_normalized - A_tilde_operator @ X_train_tilde_normalized - c_tilde[:,None] 

# print(c_tilde.shape, c_tilde[:, None].shape) # (20,) (20, 1)

results_dir = f'/workspace/venu_files/climate_forecasting/RL_NLDR_jax_version_ste/results/var_red1_ne{num_epochs}_lib{len(library_functions)}_bs{batch_size}_plen{pred_len}_lr{lr}_Nval{Nh_val}_lval{l_val}_l1incr{lambda_increment}_ste_m1_MLP/'
os.makedirs(f"{results_dir}", exist_ok=True)

print(results_dir)

CKPT_DIR = f"{results_dir}checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

from models.models_1.model_ste_mlp import (
    STEFeatureSelectorModel,
    create_train_state,
    train_for_epochs,
)

model = STEFeatureSelectorModel(library_functions = library_functions,
                                lib_dim=int(l_val * len(library_functions)),
                                out_dim=l_val,
                                mlp_hidden=decoder_net)

rng = jax.random.PRNGKey(12)
# sample_input = jnp.zeros((1, l_val), dtype=jnp.float32)
sample_input = jnp.array(np.random.random((1, l_val)), dtype = jnp.float32)
state = create_train_state(rng, model, sample_input, lr)

# print(state.params['decoder'].keys()) # dict_keys(['Dense_0', 'Dense_1', 'Dense_2'])

# eigvals of A_tilde:
# eigvals = jnp.abs(jnp.linalg.eigvals(A_tilde_operator))
# print(jnp.sort(eigvals, descending = True)[:10])

# def step(xh, _): # x-tilde_norm_init # (l_val)
#     linear_term = A_tilde_operator @ xh
#     phi, aux = state.apply_fn({'params': state.params}, xh.reshape(1,-1)) 
#     phi = phi.squeeze(axis = 0)
#     feats_masked, mask = aux['feats_masked'], aux['mask']
#     x_next = linear_term + c_tilde + phi

#     # jax.debug.print("{}: {} | {}:{} | {}:{} | {}:{} | {}:{}", xh.min(), xh.max(), linear_term.min(), linear_term.max(), c_tilde.min(), c_tilde.max(), phi.min(), phi.max(), x_next.min(), x_next.max())

#     jax.debug.print("{}:{} | {}:{} | {}:{} | {}:{}",
#                     linear_term.min(), linear_term.max(), 
#                     c_tilde.min(), c_tilde.max(), 
#                     phi.min(), phi.max(), 
#                     x_next.min(), x_next.max())

#     # jax.debug.print("||xh||: {}, ||linear||: {}, ||phi||: {}, ||c||: {}", 
#     #                 jnp.linalg.norm(xh), 
#     #                 jnp.linalg.norm(linear_term), 
#     #                 jnp.linalg.norm(phi), 
#     #                 jnp.linalg.norm(c_tilde))

#     return x_next, x_next

# # x_init = jnp.array(np.random.random((l_val)))
# x_init = X_train[:,0]
# x_init_tilde = U_l.T @ x_init
# x_init_tilde_norm = (x_init_tilde - min_tilde)/(max_tilde - min_tilde)
# _, x_seq = jax.lax.scan(step, x_init_tilde_norm, None, 5) # (T, l_val)

# print(A_tilde_operator.min(), ":",A_tilde_operator.max()) # -0.10439118 : 0.9998042
# print(c_tilde.min(), ":", c_tilde.max()) # -0.09018245 : 0.13742976

# x_pred_tilde_norm = x_pred_tilde_norm_t.T # (l_val, T)
# x_pred_tilde = (max_tilde - min_tilde) * x_pred_tilde_norm + min_tilde 
# x_pred = U_l @ x_pred_tilde # (Nh, T)
# print(x_pred.shape)
# print(x_seq.shape) # (T, l_val)
# print(state)
# print(X_train.shape, Y_train.shape) # (22701, 1656) (22701, 1656)
# print(X_val.shape, Y_val.shape) # (22701, 987) (22701, 987)

# print(A_tilde_operator.min(), ":", A_tilde_operator.max()) # -0.104394846 : 0.9997977
# print(c_tilde.min(), ":", c_tilde.max()) # -0.2560952 : 0.20082524

# train_for_epochs(
#     state = state,
#     X_train = X_train,
#     Y_train = Y_train,
#     X_val = X_val,
#     Y_val = Y_val,
#     U_l = U_l,
#     min_tilde = min_tilde,
#     max_tilde = max_tilde,
#     A_tilde = A_tilde_operator,
#     c_tilde = c_tilde,
    
#     library_functions = library_functions,

#     lib_dim = l_val,
#     pred_len = pred_len,

#     lambda_l1 = lambda_l1,
#     lambda_increment = lambda_increment,

#     batch_size = batch_size,
#     epochs = num_epochs,
#     epoch_threshold = epoch_threshold,
#     lr = lr
# )

state, preserved_grads, train_loss_hist, val_loss_hist_mask, val_loss_hist_0_mask, aux_results = train_for_epochs(
    state = state,
    X_train = X_train,
    Y_train = Y_train,
    X_val = X_val,
    Y_val = Y_val,
    U_l = U_l,
    min_tilde = min_tilde,
    max_tilde = max_tilde,
    A_tilde = A_tilde_operator,
    c_tilde = c_tilde,
    
    library_functions = library_functions,

    lib_dim = l_val,
    pred_len = pred_len,

    lambda_l1 = lambda_l1,
    lambda_increment = lambda_increment,

    batch_size = batch_size,
    epochs = num_epochs,
    epoch_threshold = epoch_threshold,
    lr = lr
)


# checkpoints.save_checkpoint(
#     ckpt_dir=CKPT_DIR,
#     target=state,       # your TrainState
#     keep=1,             # how many to keep (old ones pruned)
#     step = num_epochs,
#     overwrite=True      # allow overwriting if same step
# )

# final_results = {
#     'preserved_grads': preserved_grads,
#     'train_loss_hist': train_loss_hist,
#     'val_loss_hist_mask': val_loss_hist_mask,
#     'val_loss_hist_0_mask': val_loss_hist_0_mask,
#     'aux_results': aux_results
#     }

# with open(results_dir + "final_results.pkl", "wb") as file:
#     pickle.dump(final_results, file)

#****************************************************************************************************

