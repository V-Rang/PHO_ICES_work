
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

def per_example_loss(trunc_input: jnp.ndarray, decoder_output: jnp.ndarray, dff_output: jnp.ndarray, enc_output: jnp.ndarray, omega_h: float) -> jnp.ndarray:
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
    dyn_err = jnp.sum((dff_output - enc_output) ** 2)
    return 0.5 * (omega_h * rec_err + (1.0 - omega_h) * dyn_err)

@partial(jax.jit, static_argnames=('omega_h',))
def batch_loss_compute(trunc_input: jnp.ndarray, decoder_output: jnp.ndarray, dff_output: jnp.ndarray, enc_output: jnp.ndarray, omega_h: float) -> jnp.ndarray:
    """
    Vectorized, batched version of per_example_loss.
    Returns the mean over the B samples.
    """
    per_ex = jax.vmap(per_example_loss, in_axes=(0, 0, 0, 0, None))(trunc_input, decoder_output, dff_output, enc_output, omega_h)
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

class Encoder(nn.Module):
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
    
N_fixed, r_fixed = (24, 20)
encoder_input_dim = N_fixed
encoder_output_dim = r_fixed
dff_network_module_tester = Dff_Network(input_dim=4, output_dim=encoder_output_dim)
Decoder_module_tester = Decoder(input_dim=encoder_output_dim, output_dim=encoder_input_dim)

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

@jax.jit
def eval_loss(dff_params, decoder_params, time_vals, targets):
    dff_output = dff_network_module_tester.apply({'params': dff_params}, time_vals, deterministic=True, mutable=False)
    preds = Decoder_module_tester.apply({'params': decoder_params}, dff_output, deterministic=True, mutable=False)
    return test_loss_compute(targets, preds)

class ConcreteAutoencoder(nn.Module):
    enc_inp_dim: int
    enc_out_dim: int
    frac_dynamics: float = 0.8
    eps: float = 1.0

    def setup(self):
        self.encoder = Encoder(input_dim=self.enc_inp_dim, output_dim=self.enc_out_dim)
        self.dff_network = Dff_Network(input_dim=4, output_dim=self.enc_out_dim)
        self.decoder = Decoder(input_dim=self.enc_out_dim, output_dim=self.enc_inp_dim)

    def __call__(self, x, time_val):
        enc_output = self.encoder(x)
        dff_output = self.dff_network(time_val)
        decoder_output = self.decoder(dff_output)
        batch_loss = batch_loss_compute(x, decoder_output, dff_output, enc_output, omega_h=0.5)
        return (decoder_output, batch_loss)

def create_train_state(rng, model, input_shape, lr):
    rng_params, rng_dropout = jax.random.split(rng, num=2)
    dummy_x = jnp.zeros(input_shape, dtype=jnp.float32)
    dummy_time = jnp.zeros((input_shape[0], 4), dtype=jnp.float32)
    init_vars = model.init({'params': rng_params, 'dropout': rng_dropout}, dummy_x, dummy_time)
    params = init_vars['params']
    tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_trunc_inp, batch_time_vals, rng_dropout):

    def loss_fn(params):
        decoder_output, batch_loss = state.apply_fn({'params': params}, batch_trunc_inp, batch_time_vals, rngs=rng_dropout)
        return (batch_loss, decoder_output)
    (batch_loss, batch_decoder_output), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.isfinite(batch_loss)
    grads = jax.tree.map(lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads)
    state = state.apply_gradients(grads=grads)
    return (state, batch_loss, batch_decoder_output, grads)

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

def train_for_epochs(state, trunc_dim, U_vals, S_train, train_times, S_val, val_times, num_epochs, batch_size, rng, initial_lr):
    """Runs the inner training loop and returns updated state + final loss history."""

    U_svd_trunc = U_vals[:, :trunc_dim]
    S_train_trunc = U_svd_trunc.T @ S_train
    S_train_trunc_t = S_train_trunc.T
    S_val_trunc = U_svd_trunc.T @ S_val
    S_val_trunc_t = S_val_trunc.T
    n_timesteps = S_train_trunc_t.shape[0]
    train_times_pd = pd.to_datetime(train_times)
    val_times_pd = pd.to_datetime(val_times)
    train_times_enc = time_features(train_times_pd, freq='h')
    val_times_enc = time_features(val_times_pd, freq='h')
    train_times_enc = train_times_enc.T
    val_times_enc = val_times_enc.T
    train_loss_history = []
    val_loss_history = []
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
        starts = make_epoch_starts(subkey, S_train_trunc_t.shape[0], batch_size, non_overlapping=True)
        epoch_loss = 0.0
        for start in starts:
            S_train_trunc_t_batch = S_train_trunc_t[start:start + batch_size]
            train_times_batch = train_times_enc[start:start + batch_size]
            rng_d = jax.random.PRNGKey(7)

            state, batch_loss, batch_decoder_output, grad_vals = train_step(state, S_train_trunc_t_batch, train_times_batch, rng_d)

            preserved_grads.append(grad_vals)

            epoch_loss += batch_loss.item() * S_train_trunc_t_batch.shape[0]
    
        train_loss_history.append(epoch_loss / n_timesteps)
        dff_params = state.params['dff_network']
        decoder_params = state.params['decoder']
        val_loss = eval_loss(dff_params, decoder_params, val_times_enc, S_val_trunc_t)
        val_loss_cpu = val_loss.block_until_ready().item()
        val_loss_history.append(val_loss_cpu)
        if val_loss_cpu < best_val_loss:
            best_val_loss = val_loss_cpu
            stall_count = 0
            best_params = state.params
            best_overall_train_loss = train_loss_history[-1]
            best_overall_val_loss = val_loss_history[-1]
        else:
            stall_count += 1
    state = state.replace(params=best_params)
    return (state, train_loss_history, val_loss_history, best_overall_train_loss, best_overall_val_loss, preserved_grads)