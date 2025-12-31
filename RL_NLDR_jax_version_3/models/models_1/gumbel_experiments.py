
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

def per_example_loss(encoder_input: jnp.ndarray, decoder_output: jnp.ndarray) -> jnp.ndarray:
    """
    One snapshot's contribution to Eq.(11):
      ½ [ ω_h * ||z_true - z_hat||²  +  (1-ω_h) * ||u_tilde - u_pred||² ]
    where
      z_true = Vᵀ u_h  (POD coords),
      z_hat  = decoder(u_pred),
      u_tilde = encoder(z_true),
      u_pred  = dynamics_net(t, μ).
    """

    rec_err = jnp.sum((encoder_input - decoder_output) ** 2)
    
    return 0.5 * (rec_err)

@partial(jax.jit)
def batch_loss_compute(encoder_input: jnp.array, decoder_output: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized, batched version of per_example_loss.
    Returns the mean over the B samples.
    """
    per_ex = jax.vmap(per_example_loss, in_axes=(0, 0))(encoder_input, decoder_output)
    return jnp.mean(per_ex)

class Decoder(nn.Module):
    output_dim: int
    @nn.compact
    def __call__(self, x, deterministic: bool=False):
        x = nn.Dense(320)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(320)(x)
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

N_fixed, r_fixed = (784, 20)
encoder_input_dim = N_fixed
encoder_output_dim = r_fixed
Decoder_module_tester = Decoder(output_dim=encoder_input_dim)


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
def eval_loss(decoder_params, decoder_inp, targets):
    preds = Decoder_module_tester.apply({'params': decoder_params}, decoder_inp , deterministic=True, mutable=False)
    return test_loss_compute(targets, preds)

class ConcreteSelector(nn.Module):
    start_temp: float
    min_temp: float
    alpha_const: float
    input_dim: int
    output_dim: int
    eps: float = 1.0

    @nn.compact
    def __call__(self, x, temp):
        logits = self.param('logits', glorot_normal(), (self.output_dim, self.input_dim))

        gumbel_key = self.make_rng('gumbel')
        uniform = jax.random.uniform(gumbel_key, logits.shape, minval=1e-07, maxval=1.0)

        # gumbel = -jnp.log(-jnp.log(uniform))
        # temp = jnp.maximum(self.min_temp, temp * self.alpha_const)
        # noisy_logits = (logits + gumbel) / temp

        noisy_logits = logits / temp
        samples = jax.nn.softmax(noisy_logits, axis=-1)
        selections = jnp.dot(x, samples.T)
        selected_indices = jnp.argmax(logits, axis=-1)
        return (selections, temp, selected_indices)

class ConcreteAutoencoder(nn.Module):
    min_temp: float
    start_temp: float
    alpha_const: float
    enc_inp_dim: int
    enc_out_dim: int
    eps: float = 1.0

    def setup(self):
        self.encoder = ConcreteSelector(start_temp=self.start_temp, min_temp=self.min_temp, alpha_const=self.alpha_const, input_dim=self.enc_inp_dim, output_dim=self.enc_out_dim)
        self.decoder = Decoder(output_dim=self.enc_inp_dim)

    def __call__(self, x, temperature):
        enc_output, temperature, selected_indices = self.encoder(x, temperature)
        decoder_output = self.decoder(enc_output)
        batch_loss = batch_loss_compute(x, decoder_output)

        return (decoder_output, batch_loss, temperature)

def create_train_state(rng, model, input_shape, lr):
    rng_params, rng_gumbel, rng_Dropout = jax.random.split(rng, num=3)
    dummy_x = jnp.zeros(input_shape, dtype=jnp.float32)
    dummy_temperature = 1.0
    init_vars = model.init({'params': rng_params, 'gumbel': rng_gumbel, 'dropout': rng_Dropout}, dummy_x, dummy_temperature)
    params = init_vars['params']
    tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch_inp, temperature, rngs):
    def loss_fn(params):
        decoder_output, batch_loss, new_temperature = state.apply_fn({'params': params}, batch_inp, temperature, rngs={'gumbel': rngs['gumbel'], 'dropout': rngs['dropout']})
        aux = (decoder_output, new_temperature)
        return (batch_loss, aux)
    (batch_loss, (batch_decoder_output, new_temperature)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.isfinite(batch_loss)
    grads = jax.tree.map(lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads)
    state = state.apply_gradients(grads=grads)
    return (state, batch_decoder_output, batch_loss, new_temperature)


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

# S_train: (60k, 784)
# S_test: (10k, 784)
def train_for_epochs(state, S_train, S_test, num_epochs, start_temp, batch_size, rng, initial_lr, threshold):
    """Runs the inner training loop and returns updated state + final loss history."""

    n_timesteps = S_train.shape[0]

    train_loss_history = []
    val_loss_history = []
    mean_max_prob = []
    logit_vals_hist = []
    temperature = start_temp
    current_lr = initial_lr
    best_val_loss = float('inf')
    best_overall_train_loss, best_overall_val_loss = (float('inf'), float('inf'))
    stall_count = 0
    patience = 100
    best_params = state.params
    lr_decay = 0.5
    stop_training = False

    temp_history = []

    for epoch in tqdm(range(1, num_epochs + 1)):
        rng, subkey = jax.random.split(rng)
        starts = make_epoch_starts(subkey, S_train.shape[0], batch_size, non_overlapping=True)
        epoch_loss = 0.0
        for start in starts:
            S_train_batch = S_train[start:start + batch_size]
            rng, rng_g, rng_d = jax.random.split(rng, 3)
            state, batch_decoder_output, batch_loss, temperature = train_step(state, S_train_batch, temperature, rngs={'gumbel': rng_g, 'dropout': rng_d})
            
            temp_history.append(float(temperature))

            logits = state.params['encoder']['logits']
                        
            mean_max = jnp.mean(jnp.max(jax.nn.softmax(logits, axis=-1), axis=-1))
            mean_max_scalar = mean_max.block_until_ready().item()
            mean_max_prob.append(mean_max_scalar)
            epoch_loss += batch_loss.item() * S_train_batch.shape[0]

        train_loss_history.append(epoch_loss / n_timesteps)
        logits_cpu = np.array(logits)
        logit_vals_hist.append(logits_cpu)
        selected_indices = jnp.argmax(logits_cpu, axis=1)

        S_test_sel = jnp.take(S_test, selected_indices, axis = 1)

        decoder_params = state.params['decoder']

        val_loss = eval_loss(decoder_params, S_test_sel, S_test)

        val_loss_cpu = val_loss.block_until_ready().item()

        val_loss_history.append(val_loss_cpu)

        # if val_loss_cpu < best_val_loss:
        #     best_val_loss = val_loss_cpu
        #     stall_count = 0
        #     best_params = state.params
        #     best_overall_train_loss = train_loss_history[-1]
        #     best_overall_val_loss = val_loss_history[-1]
        # else:
        #     stall_count += 1
        # if mean_max >= threshold:
        #     stop_training = True
        # if stop_training:
        #     print(f'Stopping at epoch {epoch}, mean_max={mean_max:.3f}, best_val_loss={best_val_loss:.4f}')
        #     break
    
    state = state.replace(params=best_params)

    return (state, temperature, train_loss_history, val_loss_history, mean_max_prob, logit_vals_hist, best_overall_train_loss, best_overall_val_loss, temp_history)