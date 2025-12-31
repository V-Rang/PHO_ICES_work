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
from utils.tools_1 import make_library_functions

parser = argparse.ArgumentParser(description='era5_example_2')

num_epochs = 1000
initial_lr = 5e-3
min_temp = 0.01
start_temp = 10.
library_functions = [ "(_)**2", "(_)**3", "(_)**4", "jnp.sin(_)", "jnp.cos(_)" ]
library_functions = make_library_functions(library_functions)

repulsion_coeff = 1e-1

# print(library_functions, type(library_functions))

with open("../data/u_10m_comp_vals_3.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

# dict_keys(['lat_vals', 'lon_vals', 'train', 'test', 'val', 'U_vals', 'sing_vals', 'Vt_vals', 'r95'])

S_train = loaded_data['train']['data']
train_times = loaded_data['train']['time']

S_test = loaded_data['test']['data']
test_times = loaded_data['test']['time']

S_val = loaded_data['val']['data']
val_times = loaded_data['val']['time']

lat_vals = loaded_data['lat_vals']
lon_vals = loaded_data['lon_vals']

r_val = loaded_data['r95']
batch_size = max(S_train.shape[1] // 256, 256)

# print(S_train.shape[1], batch_size) # 128
 
Nh_val = S_train.shape[0]
r_val = 24
p_val = 20

steps_per_epoch = (S_train.shape[1] + batch_size - 1) // batch_size
alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

results_dir = f'/workspace/venu_files/climate_forecasting/CAE_transf/results/wCAE_wnl_ne{num_epochs}_lib{len(library_functions)}_bs{batch_size}_lr{initial_lr}_Nval{Nh_val}_rval{r_val}_rc{repulsion_coeff}_m2/'
os.makedirs(f"{results_dir}", exist_ok=True)

print(results_dir)

CKPT_DIR = f"{results_dir}checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]

# print(S_train.shape, S_val.shape, S_test.shape) # (22701, 985) (22701, 244) (22701, 124)

lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")
global_coords = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)  # (Nh,2)

from models.models_1.model_1_nl_time_lat_lon_CAE_wnl import (
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
    global_coords= jnp.array(global_coords),
    enc1_inp_dim  = Nh_val,
    enc1_out_dim  = r_val,
    enc2_inp_dim  = int(r_val * len(library_functions)),
    enc2_out_dim  = p_val,
    frac_dynamics= 0.5,
    repulsion_coeff = repulsion_coeff
)

state = create_train_state(
    rng,
    model,
    input_shape=(batch_size, Nh_val),
    lr=initial_lr,
)

state, temperature_spatial, temperature_lib, train_loss_hist, val_loss_hist, mean_max_prob_spt, logit_vals_hist_spt, mean_max_prob_lib, logit_vals_hist_lib, best_train_loss, best_val_loss, preserved_grads = train_for_epochs(
    state = state,
    global_coords=jnp.array(global_coords),
    trunc_dim = r_val,
    S_train = S_train,
    train_times = train_times,
    S_val = S_val,
    val_times = val_times,
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
    'mean_max_prob_lib': mean_max_prob_lib,
    'logit_vals_hist_lib': logit_vals_hist_lib,
    'best_train_loss': best_train_loss,
    'best_val_loss': best_val_loss
    }

with open(results_dir + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)

#****************************************************************************************************

