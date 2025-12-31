import os
import math
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Tuple, List
from layers.Enc_Dec import Encoder_Decoder
from utils.tools_1 import apply_selected_funcs, lstsq_l2
from utils.tools_1 import rom_reconstruction_error
from layers.output_grad_comp import output_selection
from utils.tools_1 import make_library_functions
import math
import pandas as pd
from utils.timefeatures import time_features
from flax.training import train_state
from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal
import jax
import optax
import numpy as np
from tqdm import tqdm
import gc
import pynvml
from functools import partial

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, input_shape, lr):
    rng_params, rng_gumbel, rng_Dropout = jax.random.split(rng, num=3)
    dummy_x = jnp.zeros(input_shape, dtype=jnp.float32)
    dummy_temperature_spatial = 1.0
    dummy_temperature_lib = 1.0
    dummy_time = jnp.zeros((input_shape[0], 4), dtype=jnp.float32)
    init_vars = model.init({'params': rng_params, 'gumbel': rng_gumbel, 'dropout': rng_Dropout}, dummy_x, dummy_temperature_spatial, dummy_temperature_lib ,dummy_time)
    params = init_vars['params']
    tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch_inp, temperature_spatial, temperature_lib, batch_time_vals, rngs):
    def loss_fn(params):
        decoder_output, batch_loss, new_temperature_spatial, new_temperature_lib  = state.apply_fn({'params': params}, batch_inp, temperature_spatial, temperature_lib, batch_time_vals, rngs={'gumbel': rngs['gumbel'], 'dropout': rngs['dropout']})
        aux = (decoder_output, new_temperature_spatial, new_temperature_lib)
        return (batch_loss, aux)
    
    (batch_loss, (batch_decoder_output, new_temperature_spatial, new_temperature_lib)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.isfinite(batch_loss)
    grads = jax.tree.map(lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads)
    state = state.apply_gradients(grads=grads)
    return (state, batch_decoder_output, batch_loss, new_temperature_spatial, new_temperature_lib, grads)


class ConcreteSelector_1(nn.Module): # corresp. to enc1
    start_temp: float
    min_temp: float
    alpha_const: float
    input_dim: int # Nhval 
    output_dim: int # rval
    eps: float = 1.0

    @nn.compact
    def __call__(self, x, temp): #(bsize, Nhval)
        logits = self.param('logits', glorot_normal(), (self.output_dim, self.input_dim))
        gumbel_key = self.make_rng('gumbel')
        uniform = jax.random.uniform(gumbel_key, logits.shape, minval=1e-07, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(uniform))
        temp = jnp.maximum(self.min_temp, temp * self.alpha_const)
        noisy_logits = (logits + gumbel) / temp
        samples = jax.nn.softmax(noisy_logits, axis=-1)
        selections = jnp.dot(x, samples.T)
        selected_indices = jnp.argmax(logits, axis=-1)
        return (selections, temp, selected_indices, logits)


class ConcreteSelector_2(nn.Module): # corresp. to enc2
    start_temp: float
    min_temp: float
    alpha_const: float
    input_dim: int # rval x len(lib)
    output_dim: int # pval
    eps: float = 1.0

    @nn.compact
    def __call__(self, x, temp): # x (bsize, r x len(lib))
        logits = self.param('logits', glorot_normal(), (self.output_dim, self.input_dim))
        gumbel_key = self.make_rng('gumbel')
        uniform = jax.random.uniform(gumbel_key, logits.shape, minval=1e-07, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(uniform))
        temp = jnp.maximum(self.min_temp, temp * self.alpha_const)
        noisy_logits = (logits + gumbel) / temp
        samples = jax.nn.softmax(noisy_logits, axis=-1)
        selections = jnp.dot(x, samples.T)

        selected_indices = jnp.argmax(logits, axis=-1)
        return (selections, temp, selected_indices)

def per_example_loss(trunc_input: jnp.ndarray, decoder_output: jnp.ndarray, dff_output: jnp.ndarray, enc2_output: jnp.ndarray, omega_h: float) -> jnp.ndarray:
    """
    One snapshot's contribution to Eq.(11):
      ½ [ ω_h * ||z_true - z_hat||²  +  (1-ω_h) * ||u_tilde - u_pred||² ]
    where
      z_true = Vᵀ u_h  (POD coords),
      z_hat  = decoder(u_pred),
      u_tilde = encoder(z_true),
      u_pred  = dynamics_net(t, μ).
    """
    rec_err = jnp.sum((trunc_input - decoder_output) ** 2)
    dyn_err = jnp.sum((dff_output - enc2_output) ** 2)
    return 0.5 * (omega_h * rec_err + (1.0 - omega_h) * dyn_err)


@partial(jax.jit, static_argnames=('omega_h',))
def batch_loss_compute(trunc_input: jnp.ndarray, decoder_output: jnp.ndarray, dff_output: jnp.ndarray, enc2_output: jnp.ndarray, omega_h: float) -> jnp.ndarray:
    """
    Vectorized, batched version of per_example_loss.
    Returns the mean over the B samples.
    trunc_inp: (b, Nh), 
    decoder_output: (b, Nh),

    dff_output: (b, pval), 
    enc2_output: (b, pval),    
    """

    per_ex = jax.vmap(per_example_loss, in_axes=(0, 0, 0, 0, None))(trunc_input, decoder_output, dff_output, enc2_output, omega_h)
    return jnp.mean(per_ex)

class Decoder(nn.Module):
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x, deterministic: bool=False):
        x = nn.Dense(32)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(32)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class Dff_Network(nn.Module):
    input_dim: int
    output_dim: int
    decoder_input_dim: int

    @nn.compact
    def __call__(self, x, deterministic: bool=False):
        x = nn.Dense(32)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(32)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(self.output_dim)(x)[...,0]
        x = nn.Dense(self.decoder_input_dim)(x)

        return x

N_fixed, r_fixed, p_fixed, lib_length = (22701, 24, 20, 5)
S_train_shape1 = 1657
num_epochs = 1000
min_temp = 0.01
start_temp = 10.
batch_size = max( S_train_shape1 // 256, 256)
steps_per_epoch = ( S_train_shape1 + batch_size - 1) // batch_size
library_functions = [ "(_)**2", "(_)**3", "(_)**4", "jnp.sin(_)", "jnp.cos(_)" ]
library_functions = tuple(make_library_functions(library_functions) )
repulsion_coeff = 1e-1

encoder1_input_dim = N_fixed
encoder1_output_dim = r_fixed

encoder2_input_dim = int(r_fixed * lib_length)
encoder2_output_dim = p_fixed

alpha_const = alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))
frac_dynamics = 0.5

encoder_network_module_tester_1 = ConcreteSelector_1(start_temp = start_temp, 
                                                    min_temp = min_temp, 
                                                    alpha_const= alpha_const, 
                                                    input_dim= encoder1_input_dim, 
                                                    output_dim= encoder1_output_dim)


encoder_network_module_tester_2 = ConcreteSelector_2(start_temp = start_temp, 
                                                    min_temp = min_temp, 
                                                    alpha_const= alpha_const, 
                                                    input_dim= encoder2_input_dim, 
                                                    output_dim= encoder2_output_dim)



dff_network_module_tester = Dff_Network(input_dim=6, output_dim=1, decoder_input_dim=p_fixed)
Decoder_module_tester = Decoder(input_dim=encoder2_output_dim, output_dim=encoder1_input_dim)


@jax.jit
def validation_loss_compute(S_val_t_batch: jnp.array, 
                            val_times_batch, global_coords, logits_spt, selected_indices_spt,
                            selected_indices_lib,  state, temperature_spt, 
                            temperature_lib, batch_size):

    selected_coords = global_coords[selected_indices_spt]

    dff_params = state.params['dff_network']
    decoder_params = state.params['decoder']
    encoder_params_1 = state.params['encoder1']
    encoder_params_2 = state.params['encoder2']

    # (self, x, temperature_spatial, temperature_lib,  time_vals): # x: Nh
    # enc1_output, temperature_spatial, selected_indices_spatial, logits_spt = self.encoder1(x, temperature_spatial) #enc1_out: rval
    # lib_output = apply_selected_funcs(enc1_output, self.library_functions) # rval x len(lib)
    # enc2_output, temperature_lib, selected_indices_lib = self.encoder2(lib_output, temperature_lib) #enc2_out: pval
    # selected_coords = self.global_coords[selected_indices_spatial]
    # coords_expanded = jnp.broadcast_to(selected_coords[None, ...], (x.shape[0], self.enc1_out_dim, 2))
    # time_expanded = jnp.broadcast_to(time_vals[:, None, :], (x.shape[0], self.enc1_out_dim, 4))
    # dff_input = jnp.concatenate([time_expanded, coords_expanded], axis=-1)
    # dff_output = self.dff_network(dff_input) # (pval)
    # decoder_output = self.decoder(dff_output) # pval -> Nh
    # batch_loss = batch_loss_compute(x, decoder_output, dff_output, enc2_output, omega_h=self.frac_dynamics)
    # diff = logits_spt[:, None, :] - logits_spt[None, :, :]   # (K, K, Nh)
    # sqdist = jnp.sum(diff**2, axis=-1)                        # (K, K)
    # mask   = 1.0 - jnp.eye(self.enc1_out_dim)                        # zeros on diag
    # rep_1  = 0.5 * jnp.sum(jnp.exp(-sqdist) * mask)                  # scalar
    # batch_loss = batch_loss + self.repulsion_coeff * rep_1
    # return (decoder_output, batch_loss, temperature_spatial, temperature_lib)

    batch_enc_out_1, _, _ , _ = encoder_network_module_tester_1.apply(
        {"params": encoder_params_1},
        S_val_t_batch,
        temperature_spt,
        rngs={'gumbel': jax.random.PRNGKey(42)}
    )

    batch_lib_output = apply_selected_funcs(batch_enc_out_1, library_functions) # rval x len(lib)

    batch_enc_out_2, _ , _ = encoder_network_module_tester_2.apply(
        {"params": encoder_params_2},
        batch_lib_output,
        temperature_lib,
        rngs={'gumbel': jax.random.PRNGKey(42)}
    )

    selected_coords = global_coords[selected_indices_spt]
    coords_expanded = jnp.broadcast_to(selected_coords[None, ...], (S_val_t_batch.shape[0], encoder1_output_dim, 2))
    time_expanded = jnp.broadcast_to(val_times_batch[:, None, :], (S_val_t_batch.shape[0], encoder1_output_dim, 4))

    batch_dff_input = jnp.concatenate([time_expanded, coords_expanded], axis=-1)

    batch_dff_out = dff_network_module_tester.apply(
        {'params': dff_params},
        batch_dff_input,
        deterministic=True,
        mutable=False
    )

    batch_preds = Decoder_module_tester.apply(
        {'params': decoder_params},
        batch_dff_out,
        deterministic=True,
        mutable=False
    )

    batch_loss = batch_loss_compute(S_val_t_batch, batch_preds, batch_dff_out, batch_enc_out_2, frac_dynamics)

    # diff = logits_spt[:, None, :] - logits_spt[None, :, :]   # (K, K, Nh)
    # sqdist = jnp.sum(diff**2, axis=-1)                        # (K, K)
    # mask   = 1.0 - jnp.eye(encoder1_output_dim)                        # zeros on diag
    # rep_1  = 0.5 * jnp.sum(jnp.exp(-sqdist) * mask)                  # scalar
    # batch_loss = batch_loss + repulsion_coeff * rep_1

    validation_loss = batch_loss * S_val_t_batch.shape[0]

    return validation_loss


@jax.jit
def test_loss_compute(targets: jnp.ndarray, preds: jnp.ndarray, eps: float=1e-08) -> jnp.ndarray:
    """
    Compute ε_rel as in Eq (12):
      ε_rel = (1/N_test) ∑_{i=1}^{N_test} sqrt(
                 ∑_{k=1}^{N_t} ||u_true[i,k] - u_pred[i,k]||²
                 -----------------------------------------
                 ∑_{k=1}^{N_t} ||u_true[i,k]||²
             )
    u_true, u_pred: shape (N_test, N_t, N_h)
    """
    num = jnp.sum((preds - targets) ** 2, axis=1)
    den = jnp.sum(targets ** 2, axis=1) + eps
    rel = jnp.sqrt(num / den)
    return jnp.mean(rel)

# @jax.jit
# def eval_loss(dff_params, decoder_params, dff_inp, targets):
#     dff_output = dff_network_module_tester.apply({'params': dff_params}, dff_inp, deterministic=True, mutable=False)
#     # dff_output = dff_output[..., 0]
#     dff_output = dff_output
#     preds = Decoder_module_tester.apply({'params': decoder_params}, dff_output, deterministic=True, mutable=False)
#     return test_loss_compute(targets, preds)


@jax.jit
def eval_loss_batch(dff_params, decoder_params, batch_dff_inp, batch_targets):
    dff_out = dff_network_module_tester.apply(
        {'params': dff_params},
        batch_dff_inp,
        deterministic=True,
        mutable=False
    )

    preds = Decoder_module_tester.apply(
        {'params': decoder_params},
        dff_out,
        deterministic=True,
        mutable=False
    )

    return test_loss_compute(batch_targets, preds)


def eval_loss_over_dataset(dff_params, decoder_params, 
                           all_dff_inp, all_targets, batch_size=128):

    N = all_dff_inp.shape[0]
    total_loss = 0.0
    for start in range(0, N, batch_size):

        end = start + batch_size

        batch_inp    = all_dff_inp[start:end]
        batch_targs  = all_targets[start:end]

        # note: eval_loss_batch returns a scalar ∈ ℝ
        batch_loss = eval_loss_batch(dff_params, decoder_params,
                                       batch_inp, batch_targs)
    
        total_loss  += float(batch_loss) * (end - start)
    
    return total_loss / N

class ConcreteAutoencoder(nn.Module):
    library_functions: Any
    min_temp: float
    start_temp: float
    alpha_const: float
    global_coords: jnp.array
    enc1_inp_dim: int # Nh
    enc1_out_dim: int # rval 
    enc2_inp_dim: int # rval x len(lib)
    enc2_out_dim: int # pval
    frac_dynamics: float = 0.5
    eps: float = 1.0
    repulsion_coeff: float = 1e-3

    def setup(self):
        self.encoder1 = ConcreteSelector_1(start_temp=self.start_temp, min_temp=self.min_temp, alpha_const=self.alpha_const, input_dim=self.enc1_inp_dim, output_dim=self.enc1_out_dim)
        self.encoder2 = ConcreteSelector_2(start_temp=self.start_temp, min_temp=self.min_temp, alpha_const=self.alpha_const, input_dim=self.enc2_inp_dim, output_dim=self.enc2_out_dim)

        self.dff_network = Dff_Network(input_dim=6, output_dim=1, decoder_input_dim=self.enc2_out_dim)

        self.decoder = Decoder(input_dim=self.enc2_out_dim, output_dim=self.enc1_inp_dim)

    def __call__(self, x, temperature_spatial, temperature_lib,  time_vals): # x: Nh
        enc1_output, temperature_spatial, selected_indices_spatial, logits_spt = self.encoder1(x, temperature_spatial) #enc1_out: rval
    
        lib_output = apply_selected_funcs(enc1_output, self.library_functions) # rval x len(lib)

        enc2_output, temperature_lib, selected_indices_lib = self.encoder2(lib_output, temperature_lib) #enc2_out: pval

        selected_coords = self.global_coords[selected_indices_spatial]
        coords_expanded = jnp.broadcast_to(selected_coords[None, ...], (x.shape[0], self.enc1_out_dim, 2))

        time_expanded = jnp.broadcast_to(time_vals[:, None, :], (x.shape[0], self.enc1_out_dim, 4))

        dff_input = jnp.concatenate([time_expanded, coords_expanded], axis=-1)
        dff_output = self.dff_network(dff_input) # (pval)

        decoder_output = self.decoder(dff_output) # pval -> Nh
         
        # jax.debug.print("{}:{}:{}:{}", x.shape, decoder_output.shape, dff_output.shape, enc2_output.shape)

        batch_loss = batch_loss_compute(x, decoder_output, dff_output, enc2_output, omega_h=self.frac_dynamics)

        # repulsive penalty:
        # diff = logits_spt[:, None, :] - logits_spt[None, :, :]   # (K, K, Nh)
        # sqdist = jnp.sum(diff**2, axis=-1)                        # (K, K)
        # # zero out the diagonal so we only penalize i≠j
        # # sqdist = sqdist * (1 - jnp.eye(sqdist.shape[0]))
        # # compute exp(−sqdist) and sum
        # # rep_pen = jnp.sum(jnp.exp(-sqdist)) * 0.5                # factor ½ to not double‑count
        # # scale by your coefficient

        # mask   = 1.0 - jnp.eye(self.enc1_out_dim)                        # zeros on diag
        # rep_1  = 0.5 * jnp.sum(jnp.exp(-sqdist) * mask)                  # scalar

        # # batch_loss = batch_loss + self.repulsion_coeff * rep_pen

        # batch_loss = batch_loss + self.repulsion_coeff * rep_1


        return (decoder_output, batch_loss, temperature_spatial, temperature_lib)


def make_epoch_starts(key, seq_len, batch_size, non_overlapping=True):
    """
    Returns a 1-D array of start-indices, shuffled for this epoch.
    
    Args:
      key: a jax.random.PRNGKey
      seq_len: total length of your time sequence (e.g. X_hat_mod_t.shape[0])
      batch_size: how many timesteps per batch
      non_overlapping: 
        - If True, uses floor(seq_len / batch_size) non‑overlapping windows.
        - If False, uses all (seq_len - batch_size + 1) sliding windows.
    
    Returns:
      starts: jnp.array of shape (n_batches,), each in [0, seq_len - batch_size]
    """
    if non_overlapping:
        n_batches = seq_len // batch_size
        starts = jnp.arange(n_batches) * batch_size
    else:
        starts = jnp.arange(seq_len - batch_size + 1)
    perm = jax.random.permutation(key, starts.shape[0])
    return starts[perm]


def train_for_epochs(state, global_coords, trunc_dim, S_train, train_times, S_val, val_times, num_epochs, start_temp, batch_size, rng, initial_lr, threshold):
    """Runs the inner training loop and returns updated state + final loss history."""

    S_train_t = S_train.T
    S_val_t = S_val.T
    n_timesteps = S_train_t.shape[0]

    train_times_pd = pd.to_datetime(train_times)
    val_times_pd = pd.to_datetime(val_times)

    train_times_enc = time_features(train_times_pd, freq='h')
    val_times_enc = time_features(val_times_pd, freq='h')

    train_times_enc = train_times_enc.T
    val_times_enc = val_times_enc.T

    val_times_expanded = jnp.broadcast_to(val_times_enc[:, None, :], (S_val_t.shape[0], trunc_dim, 4))

    train_loss_history = []
    val_loss_history = []

    mean_max_prob1 = []
    mean_max_prob2 = []

    logit_vals_hist1 = []
    logit_vals_hist2 = []

    temperature_spatial = start_temp
    temperature_lib = start_temp

    current_lr = initial_lr
    best_val_loss = float('inf')
    best_overall_train_loss, best_overall_val_loss = (float('inf'), float('inf'))
    stall_count = 0
    patience = 100
    best_params = state.params
    lr_decay = 0.5
    stop_training = False
    preserved_grads = []

    for epoch in tqdm(range(1, num_epochs + 1)):
        rng, subkey = jax.random.split(rng)
        starts = make_epoch_starts(subkey, S_train_t.shape[0], batch_size, non_overlapping=True)
        epoch_loss = 0.0
        val_loss = 0.0

        for start in starts:
            S_train_t_batch = S_train_t[start:start + batch_size]
            train_times_batch = train_times_enc[start:start + batch_size]

            rng, rng_g, rng_d = jax.random.split(rng, 3)
            
            state, batch_decoder_output, batch_loss, temperature_spatial, temperature_lib, grad_vals = train_step(
                state,
                S_train_t_batch,
                temperature_spatial,
                temperature_lib,
                train_times_batch,
                rngs={'gumbel': rng_g, 'dropout': rng_d}
            )

            
            logits_enc1 = state.params['encoder1']['logits']
            mean_max1 = jnp.mean(jnp.max(jax.nn.softmax(logits_enc1, axis=-1), axis=-1))
            mean_max_scalar1 = mean_max1.block_until_ready().item()
            mean_max_prob1.append(mean_max_scalar1)

            logits_enc2 = state.params['encoder2']['logits']
            mean_max2 = jnp.mean(jnp.max(jax.nn.softmax(logits_enc2, axis=-1), axis=-1))
            mean_max_scalar2 = mean_max2.block_until_ready().item()
            mean_max_prob2.append(mean_max_scalar2)

            epoch_loss += batch_loss.item() * S_train_t_batch.shape[0]


        logits_enc1_cpu = np.array(logits_enc1)
        del logits_enc1

        logits_enc2_cpu = np.array(logits_enc2)
        del logits_enc2

        selected_indices_spt = jnp.argmax(logits_enc1_cpu, axis=1)
        selected_indices_lib = jnp.argmax(logits_enc2_cpu, axis=1)


        # only preserve grads at end of each epoch.
        grad_vals_cpu = np.array(grad_vals)
        preserved_grads.append(grad_vals_cpu)

        del grad_vals, grad_vals_cpu

        train_loss_history.append(epoch_loss / n_timesteps)

        logit_vals_hist1.append(logits_enc1_cpu)
        selected_indices_spt = jnp.argmax(logits_enc1_cpu, axis=1)

        logit_vals_hist2.append(logits_enc2_cpu)
        selected_indices_lib = jnp.argmax(logits_enc2_cpu, axis=1)

        # validation_loss_compute(S_val_t_batch: jnp.array, 
        #                     val_times_batch, global_coords, logits_spt, selected_indices_spt,
        #                     selected_indices_lib,  state, temperature_spt, 
        #                     temperature_lib, batch_size):

        val_loss = validation_loss_compute(
            S_val_t,
            val_times_enc,
            global_coords,
            logits_enc1_cpu,
            selected_indices_spt,
            selected_indices_lib,
            state,
            temperature_spatial,
            temperature_lib,
            batch_size
        )

        val_loss_history.append(val_loss / S_val_t.shape[0])

        # dff_params = state.params['dff_network']
        # decoder_params = state.params['decoder']
        # selected_coords = global_coords[selected_indices1]

        # coords_expanded = jnp.broadcast_to(selected_coords[None, ...], (S_val_t.shape[0], trunc_dim, 2))
        # val_dff_input = jnp.concatenate([val_times_expanded, coords_expanded], axis=-1)

        # # val_loss = eval_loss(dff_params, decoder_params, val_dff_input, S_val_t)
        # val_loss_cpu = eval_loss_over_dataset(
        #     dff_params,
        #     decoder_params,
        #     val_dff_input,
        #     S_val_t,
        #     batch_size=batch_size
        # )

        # val_loss_history.append(val_loss_cpu)

        # if val_loss_cpu < best_val_loss:
        #     best_val_loss = val_loss_cpu
        #     stall_count = 0
        #     best_params = state.params
        #     best_overall_train_loss = train_loss_history[-1]
        #     best_overall_val_loss = val_loss_history[-1]
        # else:
        #     stall_count += 1

        # if mean_max1 >= threshold:
        #     stop_training = True
        # if stop_training:
        #     print(f'Stopping at epoch {epoch}, mean_max={mean_max1:.3f}, best_val_loss={best_val_loss:.4f}')
        #     break
    
    state = state.replace(params=best_params)

    return (state, temperature_spatial, temperature_lib, train_loss_history, val_loss_history, mean_max_prob1, logit_vals_hist1, mean_max_prob2, logit_vals_hist2, best_overall_train_loss, best_overall_val_loss, preserved_grads)