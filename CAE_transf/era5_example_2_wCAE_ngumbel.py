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

parser = argparse.ArgumentParser(description='era5_example_2')

num_epochs = 1000
batch_size = 16
initial_lr = 1e-1
min_temp = 0.01
start_temp = 10.

with open("../data/u_10m_comp_vals_2.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

S_train = loaded_data['train']['data']
train_times = loaded_data['train']['time']

S_test = loaded_data['test']['data']
test_times = loaded_data['test']['time']

S_val = loaded_data['val']['data']
val_times = loaded_data['val']['time']

lat_vals = loaded_data['lat_vals']
lon_vals = loaded_data['lon_vals']

r_val = loaded_data['r95']
Nh_val = S_train.shape[0]
r_val = 20
    
steps_per_epoch = (S_train.shape[1] + batch_size - 1) // batch_size
alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

results_dir = f'/workspace/venu_files/climate_forecasting/CAE_transf/results/ne{num_epochs}_lr{initial_lr}_Nval{Nh_val}_rval{r_val}_m2_ngumbel/'
os.makedirs(f"{results_dir}", exist_ok=True)

CKPT_DIR = f"{results_dir}checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]

lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")
global_coords = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)  # (Nh,2)


from models.models_1.model_1_nl_time_lat_lon_CAE_ngumbel import (
    ConcreteAutoencoder,
    create_train_state,
    train_for_epochs,
)

rng = jax.random.PRNGKey(0)
model = ConcreteAutoencoder(
    min_temp     = min_temp,
    start_temp   = start_temp,
    alpha_const  = alpha_const,
    global_coords= jnp.array(global_coords),
    enc_inp_dim  = Nh_val,
    enc_out_dim  = r_val,
    frac_dynamics= 0.5
)

state = create_train_state(
    rng,
    model,
    input_shape=(batch_size, Nh_val),
    lr=initial_lr,
)


state, temperature, train_loss_hist, val_loss_hist, mean_max_prob, logit_vals_hist, best_train_loss, best_val_loss = train_for_epochs(
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
    'train_loss_hist': train_loss_hist,
    'val_loss_hist': val_loss_hist,
    'mean_max_prob': mean_max_prob,
    'logit_vals_hist': logit_vals_hist,
    'best_train_loss': best_train_loss,
    'best_val_loss': best_val_loss
    }

with open(results_dir + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)

#****************************************************************************************************
