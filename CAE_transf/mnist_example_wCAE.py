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

parser = argparse.ArgumentParser(description='mnist_example')

num_epochs = 1
initial_lr = 1e-3
min_temp = 0.01
start_temp = 10.

with open("../data/mnist_data.pkl", 'rb') as f:
    mnist_data = pickle.load(f)

x_train = mnist_data['x_train']
x_test = mnist_data['x_test']

# x_train = x_train[:13890]
# x_val = x_test[3000: 3091]
# x_test = x_test[:2780]

# x_train = x_train[:1389]
# x_val = x_test[300: 391]
# x_test = x_test[:278]

# print(x_train.shape, x_val.shape, x_test.shape) # (13890, 784) (91, 784) (2780, 784) 

batch_size = max(len(x_train) // 256, 16)
Nh_val = x_train.shape[1]
r_val = 20

# print(batch_size) # 234
# print(x_train.shape, x_test.shape)

steps_per_epoch = (x_train.shape[0] + batch_size - 1) // batch_size
alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

results_dir = f'/workspace/venu_files/climate_forecasting/CAE_transf/results/mnist_example_ne{num_epochs}_lr{initial_lr}_Nval{Nh_val}_rval{r_val}_mintemp_{min_temp}_wgumbel/'
os.makedirs(f"{results_dir}", exist_ok=True)

print(results_dir)

CKPT_DIR = f"{results_dir}checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]

from models.models_1.gumbel_experiments import (
    ConcreteAutoencoder,
    create_train_state,
    train_for_epochs,
)

rng = jax.random.PRNGKey(0)

model = ConcreteAutoencoder(
    min_temp     = min_temp,
    start_temp   = start_temp,
    alpha_const  = alpha_const,
    enc_inp_dim  = Nh_val,
    enc_out_dim  = r_val
)

state = create_train_state(
    rng,
    model,
    input_shape=(batch_size, Nh_val),
    lr=initial_lr,
)

state, temperature, train_loss_hist, val_loss_hist, mean_max_prob, logit_vals_hist, best_train_loss, best_val_loss, temp_history = train_for_epochs(
    state = state,
    S_train = x_train,
    S_test = x_test,
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
    'best_val_loss': best_val_loss,
    'temp_history': temp_history,

    }

with open(results_dir + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)

#****************************************************************************************************
