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
nnz = 20
from flax.training import train_state
from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal
import jax
import optax
import numpy as np
from tqdm import tqdm



class TrainState(train_state.TrainState):
    pass

def reconstr_loss(gt, reconstr):
    mse_err = jnp.mean(jnp.mean((gt - reconstr) ** 2, axis=(1,)))
    return mse_err

@jax.jit
def one_sample_loss(state, x, temperature, rng):
    pred, _, _ = state.apply_fn({'params': state.params}, x[None, :], temperature, rngs={'gumbel': rng, 'dropout': rng})
    pred = pred[0]
    return jnp.mean((x - pred) ** 2)

class Network_Decoder(nn.Module):
    input_dim: int

    @nn.compact
    def __call__(self, x, deterministic: bool=False):
        x = nn.Dense(320)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(320)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(self.input_dim)(x)
        return x

class Network_Decoder(nn.Module):
    x_hat_s1: int

    @nn.compact
    def __call__(self, x, deterministic: bool=False):
        x = nn.Dense(320)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(320)(x)
        x = nn.activation.leaky_relu(x, 0.2)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(self.x_hat_s1)(x)
        return x

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
        gumbel = -jnp.log(-jnp.log(uniform))
        temp = jnp.maximum(self.min_temp, temp * self.alpha_const)
        noisy_logits = (logits + gumbel) / temp
        samples = jax.nn.softmax(noisy_logits, axis=-1)
        selections = jnp.dot(x, samples.T)
        selected_indices = jnp.argmax(logits, axis=-1)
        return (selections, temp, selected_indices)

class ConcreteAutoencoder(nn.Module):
    x_mod_s1: int
    x_nl_s1: int
    x_hat_s1: int
    min_temp: float
    start_temp: float
    alpha_const: float
    sel_rng: int
    eps: float = 1.0

    def setup(self):
        self.selector = ConcreteSelector(self.start_temp, self.min_temp, self.alpha_const, input_dim=self.x_mod_s1, output_dim=self.x_nl_s1)
        self.decoder = Network_Decoder(self.x_hat_s1)

    def __call__(self, x, y_hat_t_batch, temperature):
        x_hat_nl_t_batch, temperature, selected_indices = self.selector(x, temperature)
        y_hat_t_batch_pred = self.decoder(x_hat_nl_t_batch)
        err = reconstr_loss(y_hat_t_batch, y_hat_t_batch_pred)
        return (y_hat_t_batch_pred, err, temperature)

def create_train_state(rng, model, input_shape, trunc_dim, lr):
    rng_params, rng_gumbel, rng_Dropout = jax.random.split(rng, num=3)
    init_vars = model.init({'params': rng_params, 'gumbel': rng_gumbel, 'dropout': rng_Dropout}, jnp.empty(input_shape), jnp.empty((input_shape[0], trunc_dim)), 1.0)
    params = init_vars['params']
    tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07, eps_root=False)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, x_hat_mod_t_batch, y_hat_t_batch, temperature, rngs):

    def loss_fn(params):
        _, loss, new_temperature = state.apply_fn({'params': params}, x_hat_mod_t_batch, y_hat_t_batch, temperature, rngs={'gumbel': rngs['gumbel'], 'dropout': rngs['dropout']})
        return (loss, new_temperature)
    (loss, new_temperature), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.isfinite(loss)
    grads = jax.tree.map(lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads)
    state = state.apply_gradients(grads=grads)
    return (state, loss, new_temperature)

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

def train_for_epochs(state, X_hat_mod_t, Y_hat_t, num_epochs, start_temp, batch_size, rng):
    """Runs the inner training loop and returns updated state + final loss history."""
    n_timesteps = X_hat_mod_t.shape[0]
    train_loss_history = []
    mean_max_prob = []
    logit_vals_hist = []
    temperature = start_temp
    for epoch in tqdm(range(1, num_epochs + 1)):
        rng, subkey = jax.random.split(rng)
        starts = make_epoch_starts(subkey, X_hat_mod_t.shape[0], batch_size, non_overlapping=True)
        epoch_loss = 0.0
        for start in starts:
            X_hat_mod_t_batch = X_hat_mod_t[start:start + batch_size]
            Y_hat_t_batch = Y_hat_t[start:start + batch_size]
            rng, rng_g, rng_d = jax.random.split(rng, 3)
            state, batch_loss, temperature = train_step(state, X_hat_mod_t_batch, Y_hat_t_batch, temperature, rngs={'gumbel': rng_g, 'dropout': rng_d})
            logits = state.params['selector']['logits']
            mean_max = jnp.mean(jnp.max(jax.nn.softmax(logits, axis=-1), axis=-1))
            mean_max_prob.append(mean_max.item())
            logit_vals_hist.append(logits)
            epoch_loss += batch_loss * X_hat_mod_t.shape[0]
        train_loss_history.append((epoch_loss / n_timesteps).item())
    return (state, temperature, train_loss_history, mean_max_prob, logit_vals_hist, rng)