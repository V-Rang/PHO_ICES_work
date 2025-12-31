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

# jax.config.update("jax_debug_nans", True)   # crash on first NaN

num_epochs = 100
initial_lr = 1e-3
min_temp = 0.01
start_temp = 10.
lam_vec = jnp.logspace(-4, 4, 6)

# library_functions = [ "(_)**2", "(_)**3", "(_)**4", "jnp.sin(_)", "jnp.cos(_)" ]
library_functions = [ "(_)**2", "(_)**3", "(_)**4"]
library_functions = make_library_functions(library_functions)
library_functions = tuple(library_functions)

l_val = 20
p_val = 20 # number of non-linearities to select from r_val x len(library)

repulsion_coeff = 1e-1
# print(library_functions, type(library_functions))

import os
# passing the arguments to the config
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import jax 
# print(jax.devices()) # [CudaDevice(id=0)]

with open("../data/u_10m_comp_vals_3.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

S_train = loaded_data['train']['data']
train_times = loaded_data['train']['time']

S_test = loaded_data['test']['data']
test_times = loaded_data['test']['time']

S_val = loaded_data['val']['data']
val_times = loaded_data['val']['time']

lat_vals = loaded_data['lat_vals']
lon_vals = loaded_data['lon_vals']

# min_train_val = loaded_data['min_train_val']
# max_train_val = loaded_data['max_train_val']

# print(loaded_data.keys()) # dict_keys(['lat_vals', 'lon_vals', 'train', 'test', 'val', 'U_vals', 'sing_vals', 'Vt_vals', 'r95'])

# print(S_train.shape) # (22701, 1657)
# print(S_test.shape) # (22701, 868)
# print(S_val.shape) # (22701, 988)

# min_train_val, max_train_val = S_train.min(), S_train.max()
# S_train_normalized = (S_train - min_train_val)/ (max_train_val - min_train_val)
# S_val_normalized = (S_val - min_train_val)/ (max_train_val - min_train_val)
# S_test_normalized = (S_test - min_train_val)/ (max_train_val - min_train_val)

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
X_tilde = U_l.T @ X_train
Y_tilde = U_l.T @ Y_train

# rho = jnp.max(jnp.abs(jnp.linalg.eigvals(A_tilde_operator)))
# jax.debug.print("rho(A_tilde)={}", rho)

min_tilde, max_tilde =  X_tilde.min(), X_tilde.max()
X_tilde_normalized = (X_tilde - min_tilde)/ (max_tilde - min_tilde)
Y_tilde_normalized = (Y_tilde - min_tilde)/ (max_tilde - min_tilde)

# print(X_hat.min(), ":", X_hat.max()) # 0.0 : 1.0
# print(X_hat_mod.min(), ":", X_hat_mod.max()) # 0.0 : 1.0

lhs_mat = Y_tilde_normalized - A_tilde_operator @ X_tilde_normalized - 1/(max_tilde - min_tilde) * (  A_tilde_operator @ (min_tilde * jnp.ones(A_tilde_operator.shape[0])  ) - min_tilde*jnp.ones(A_tilde_operator.shape[0])  )[:,None] 

# print(S_train.min(), ":", S_train.max()) # -19.461948 : 22.5715
# print(A_tilde_operator.min(), ":", A_tilde_operator.max()) # -0.10439118 : 0.9998042
# print(A_hat.min(), ":", A_hat.max()) # -0.06317139 : 1.0

# batch_size = max(S_train.shape[1] // 256, 256)
batch_size = 10
Nh_val = S_train.shape[0]

steps_per_epoch = (S_train.shape[1] + batch_size - 1) // batch_size
alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

# # print(alpha_const) 0.9901803093040487

results_dir = f'/workspace/venu_files/climate_forecasting/RL_NLDR_jax_version_3/results/var_red1_ne{num_epochs}_lib{len(library_functions)}_bs{batch_size}_lr{initial_lr}_Nval{Nh_val}_lval{l_val}_rc{repulsion_coeff}_normalized_m2/'
os.makedirs(f"{results_dir}", exist_ok=True)

print(results_dir)

CKPT_DIR = f"{results_dir}checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

from models.models_1.model_cae_m2_normalized import (
    ConcreteAutoencoder,
    create_train_state,
    train_for_epochs,
)

rng = jax.random.PRNGKey(0)
model = ConcreteAutoencoder(
    library_functions= library_functions,
    min_temp     = min_temp,
    start_temp   = start_temp,
    alpha_const  = alpha_const,
    enc_inp_dim  = int(l_val * len(library_functions)),
    enc_out_dim  = p_val,
    lam_vec = lam_vec,
    A_tilde = A_tilde_operator,
    U_l = U_l,
    min_tilde = min_tilde,
    max_tilde = max_tilde,
    frac_dynamics= 0.5,
    repulsion_coeff = repulsion_coeff
)

state = create_train_state(
    rng,
    model,
    input_shape=(batch_size, Nh_val),
    lhs_shape = (batch_size, l_val),
    lr=initial_lr,
)

state, temperature_spatial, train_loss_hist, val_loss_hist, mean_max_prob_spt, logit_vals_hist_spt, preserved_grads = train_for_epochs(
    state = state,
    X_train = X_train,
    Y_train = Y_train,
    lhs_mat = lhs_mat,
    S_val = S_val,
    A_tilde = A_tilde_operator,
    U_l = U_l,
    min_tilde = min_tilde,
    max_tilde = max_tilde,
    library_functions = library_functions,
    num_epochs = num_epochs,
    start_temp = start_temp,
    batch_size = batch_size,
    rng = rng,
    initial_lr = initial_lr,
    threshold = 0.95
)

checkpoints.save_checkpoint(
    ckpt_dir=CKPT_DIR,
    target=state,       # your TrainState
    keep=1,             # how many to keep (old ones pruned)
    step = num_epochs,
    overwrite=True      # allow overwriting if same step
)

final_results = {
    'preserved_grads': preserved_grads,
    'train_loss_hist': train_loss_hist,
    'val_loss_hist': val_loss_hist,
    'mean_max_prob_spt': mean_max_prob_spt,
    'logit_vals_hist_spt': logit_vals_hist_spt,
    }

with open(results_dir + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)

#****************************************************************************************************

