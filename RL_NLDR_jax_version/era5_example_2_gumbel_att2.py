import math
import jax.numpy as jnp
import argparse
import os
import pickle
import jax
from utils.tools_2 import make_library_functions
from flax.training import checkpoints


parser = argparse.ArgumentParser(description='era5_example_2')

num_epochs = 100
initial_lr = 1e-1
min_temp = 0.01
start_temp = 10.
library_functions = [ "(_)**2", "(_)**3", "(_)**4", "jnp.sin(_)", "jnp.cos(_)" ]
library_functions = make_library_functions(library_functions)
repulsion_coefficient = 1e-1


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

batch_size = max(S_train.shape[1] // 256, 256)

Nh_val = S_train.shape[0]
l_val = 24
num_features_select = 30


steps_per_epoch = (S_train.shape[1] + batch_size - 1) // batch_size
alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

results_dir = f'/workspace/venu_files/climate_forecasting/RL_NLDR_jax_version/results/wCAE_wnl_ne{num_epochs}_lib{len(library_functions)}_bs{batch_size}_lr{initial_lr}_Nval{Nh_val}_lval{l_val}_rval{r_val}_rc{repulsion_coefficient}/'
os.makedirs(f"{results_dir}", exist_ok=True)

CKPT_DIR = f"{results_dir}checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
else:
    device = jax.devices("cpu")[0]


U_vals = loaded_data['U_vals']
sing_vals = loaded_data['sing_vals']
Vt_vals = loaded_data['Vt_vals']

U_l = U_vals[:, :l_val]
sing_l = sing_vals[:l_val]
Vt_l = Vt_vals[:l_val, :]

X_train = S_train[:, :-1]
Y_train = S_train[:, 1:]

X_val = S_val[:, : -1]
Y_val = S_val[:, 1: ]


# print(type(U_l), type(X_train))

A_operator = Y_train @ Vt_l.T @ jnp.linalg.inv(jnp.diag(sing_l)) @ U_l.T
A_tilde_operator = U_l.T @ A_operator @ U_l
X_tilde = U_l.T @ X_train


from models.models_2.model_gumbel_att2 import (
    ROMModel,
    create_train_state
)


rng = jax.random.PRNGKey(0)
model = ROMModel(
    A_tilde = A_tilde_operator,
    A_hat = A_hat, 
    phi_mat = phi_mat,
    library_functions= library_functions,
    min_temp     = min_temp,
    start_temp   = start_temp,
    alpha_const  = alpha_const,
    r_val  = r_val,
    p_val  = num_features_select,
    U_r = U_r,
    selector_type='gumbel',
    repulsion_coefficient = repulsion_coefficient
)

state = create_train_state(
    rng,
    model,
    batch_size= batch_size,
    r_val = r_val,
    Nh = S_train.shape[0],
    lr=initial_lr,
)


from models.models_2.model_gumbel_att2 import (
    train_for_epochs
)



state, temperature, train_loss_history, val_loss_history, mean_max_prob, logit_vals_hist, best_overall_train_loss, best_overall_val_loss = train_for_epochs(
    state = state,
    phi_mat = phi_mat,
    A_tilde = A_tilde_operator,
    U_r = U_r,
    A_hat = A_hat,
    X_hat = X_hat,

    X_train= X_train,
    Y_train=Y_train,

    X_val= X_val,
    Y_val=Y_val,

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
    'train_loss_hist': train_loss_history,
    'val_loss_hist': val_loss_history,
    'mean_max_prob': mean_max_prob,
    'logit_vals_hist': logit_vals_hist,
    'best_train_loss': best_overall_train_loss,
    'best_val_loss': best_overall_val_loss
    }

with open(results_dir + "final_results.pkl", "wb") as file:
    pickle.dump(final_results, file)