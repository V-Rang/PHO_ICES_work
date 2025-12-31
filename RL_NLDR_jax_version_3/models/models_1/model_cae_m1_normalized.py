import os
import math
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Tuple, List
from layers.Enc_Dec import Encoder_Decoder
from utils.tools_1_normalized import apply_selected_funcs, lstsq_l2
from utils.tools_1_normalized import make_rom_reconstruction_error
from layers.output_grad_comp import output_selection
from utils.tools_1_normalized import make_library_functions, build_library
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

def create_train_state(rng, model, input_shape, lhs_shape,  lr):
    rng_params, rng_gumbel, rng_Dropout = jax.random.split(rng, num=3)
    dummy_x = jnp.zeros(input_shape, dtype=jnp.float32)
    dummy_y = jnp.zeros(input_shape, dtype=jnp.float32)
    dummy_lhs_mat_t = jnp.zeros(lhs_shape, dtype=jnp.float32)

    dummy_temperature_spt = 1.0
    init_vars = model.init({'params': rng_params, 'gumbel': rng_gumbel, 'dropout': rng_Dropout}, dummy_x, dummy_y, dummy_lhs_mat_t, dummy_temperature_spt)
    params = init_vars['params']
    tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


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

# N_fixed, r_fixed, p_fixed, lib_length = (22701, 24, 20, 5)
# S_train_shape1 = 1657
# num_epochs = 1000
# min_temp = 0.01
# start_temp = 10.
# batch_size = max( S_train_shape1 // 256, 256)
# steps_per_epoch = ( S_train_shape1 + batch_size - 1) // batch_size
# library_functions = [ "(_)**2", "(_)**3", "(_)**4", "jnp.sin(_)", "jnp.cos(_)" ]
# library_functions = tuple(make_library_functions(library_functions) )
# repulsion_coeff = 1e-1

# encoder1_input_dim = N_fixed
# encoder1_output_dim = r_fixed

# encoder2_input_dim = int(r_fixed * lib_length)
# encoder2_output_dim = p_fixed

# alpha_const = alpha_const = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))
# frac_dynamics = 0.5

# encoder_network_module_tester_1 = ConcreteSelector_1(start_temp = start_temp, 
#                                                     min_temp = min_temp, 
#                                                     alpha_const= alpha_const, 
#                                                     input_dim= encoder1_input_dim, 
#                                                     output_dim= encoder1_output_dim)


# encoder_network_module_tester_2 = ConcreteSelector_2(start_temp = start_temp, 
#                                                     min_temp = min_temp, 
#                                                     alpha_const= alpha_const, 
#                                                     input_dim= encoder2_input_dim, 
#                                                     output_dim= encoder2_output_dim)



# dff_network_module_tester = Dff_Network(input_dim=6, output_dim=1, decoder_input_dim=p_fixed)
# Decoder_module_tester = Decoder(input_dim=encoder2_output_dim, output_dim=encoder1_input_dim)


@partial(jax.jit, static_argnames=('library_functions',))
def validation_loss_compute(S_val: jnp.array, 
                            phi_mat,
                            A_tilde,
                            phi_bar_opt: jnp.array,
                            U_l,
                            library_functions,
                            A_hat,
                            min_tilde,
                            max_tilde,
                            selected_indices_spt):

    H_hat_opt = phi_mat.T @ A_tilde @ phi_bar_opt

    x0 = S_val[:,0]

    x_tilde0 = U_l.T @ x0
    x_hat0 = phi_mat.T @ x_tilde0                            # (r,)
    # rhs_term3 = (min_tilde)/(max_tilde - min_tilde) *(  phi_mat.T @ (   A_tilde @ jnp.ones(A_tilde.shape[0]) - jnp.ones(A_tilde.shape[0]) )   )

    rhs_term3 = (1/(max_tilde - min_tilde) )  * phi_mat.T @ ( A_tilde @ (min_tilde * jnp.ones(A_tilde.shape[0]) ) - min_tilde * jnp.ones(A_tilde.shape[0]) )

    def step(xh, _):
        mod = apply_selected_funcs(xh, library_functions) # e.g. shape (m,)
        mod_sel = jnp.take(mod, selected_indices_spt) # (K,)

        rhs_term1 = A_hat @ xh
        rhs_term2 = H_hat_opt @ mod_sel

        xh = rhs_term1 + rhs_term2 + rhs_term3

        return xh, xh   

    _, xh_seq = jax.lax.scan(step, x_hat0, None, length=S_val.shape[1]-1)

    X_tilde_rec_batch_normalized = phi_mat @ xh_seq.T
    X_tilde_rec_batch = (max_tilde - min_tilde) * X_tilde_rec_batch_normalized + min_tilde

    X_rec_batch = U_l @ X_tilde_rec_batch

    num = jnp.linalg.norm(S_val[:, 1:] - X_rec_batch)
    den = jnp.linalg.norm(S_val[:, 1:]) + 1e-12       # small eps for safety
    return num / den



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


class ConcreteSelector(nn.Module):
    start_temp: float
    min_temp: float
    alpha_const: float
    input_dim: int # rval x len(lib) 
    output_dim: int # pval
    eps: float = 1.0

    @nn.compact
    def __call__(self, x, temp): #(bsize, rval x len(lib))

        logits = self.param('logits', glorot_normal(), (self.output_dim, self.input_dim))
        gumbel_key = self.make_rng('gumbel')
        uniform = jax.random.uniform(gumbel_key, logits.shape, minval=1e-07, maxval=1.0)
        gumbel = -jnp.log(-jnp.log(uniform))
        temp = jnp.maximum(self.min_temp, temp * self.alpha_const)
        noisy_logits = (logits + gumbel) / temp
        samples = jax.nn.softmax(noisy_logits, axis=-1)
        selections = jnp.dot(x, samples.T)
        selected_indices = jnp.argmax(logits, axis=-1)
        return (selections, temp, selected_indices, samples)


# state, batch_selected_idx, batch_loss, batch_phi_bar_opt, temperature_spt, grad_vals = train_step(
@jax.jit
def train_step(state, X_batch_t, Y_batch_t, lhs_mat_batch_t, temperature_spatial, rngs):
    def loss_fn(params):
        selected_idx, batch_loss, phi_bar_opt, new_temperature_spatial  = state.apply_fn({'params': params}, X_batch_t, Y_batch_t, lhs_mat_batch_t, temperature_spatial, rngs={'gumbel': rngs['gumbel'], 'dropout': rngs['dropout']})
    
        aux = (selected_idx, phi_bar_opt, new_temperature_spatial)
        return (batch_loss, aux)

    (batch_loss, (batch_selected_idx, phi_bar_opt, new_temperature_spatial)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.isfinite(batch_loss)
    
    # jax.debug.print("{}", grads)

    # jax.debug.print("{}:{}", grads['encoder']['logits'].min(), grads['encoder']['logits'].max()) # all nans

    grads = jax.tree.map(lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads)
    state = state.apply_gradients(grads=grads)
    return (state, batch_selected_idx, batch_loss, phi_bar_opt, new_temperature_spatial, grads)

class ConcreteAutoencoder(nn.Module):
    library_functions: Any
    min_temp: float
    start_temp: float
    alpha_const: float
    enc_inp_dim: int # rval x len(lib)
    enc_out_dim: int # pval
    lam_vec: jnp.array
    phi_mat: jnp.array
    A_tilde: jnp.array
    A_hat: jnp.array
    U_l: jnp.array
    min_tilde: float
    max_tilde: float
    frac_dynamics: float = 0.5
    eps: float = 1.0
    repulsion_coeff: float = 1e-3

    def setup(self):
        self.encoder = ConcreteSelector(start_temp=self.start_temp, min_temp=self.min_temp, alpha_const=self.alpha_const, input_dim=self.enc_inp_dim, output_dim=self.enc_out_dim)
        self.rom_err_fn = make_rom_reconstruction_error(self.phi_mat, 
        self.A_hat, self.A_tilde, self.U_l, self.min_tilde, self.max_tilde, self.library_functions)

    def __call__(self, X_batch_t, Y_batch_t, lhs_mat_batch_t, temperature_spt): # x: rval
        X_tilde_batch_t = X_batch_t @ self.U_l #(k, r)
    
        X_tilde_batch_normalized_t = (X_tilde_batch_t - self.min_tilde)/ (self.max_tilde - self.min_tilde)

        X_hat_batch_t = X_tilde_batch_normalized_t @ self.phi_mat

        X_hat_mod_batch_t = build_library(X_hat_batch_t, self.library_functions)  # (k, r x len(lib))

        X_hat_nl_batch_t, temperature_spt, selected_idx, soft_selection = self.encoder(X_hat_mod_batch_t, temperature_spt) # (k, p)

        # print(selected_idx.shape) # (p, )
        # print(selected_idx) # [21 30 24 33  5 12 36  3  9 31 25 13 30  7 36 36 25 34  7 39]
        # print(X_hat_mod_batch_t.shape, X_hat_nl_batch_t.shape) # (256,40), (256,20)
        # jax.debug.print("{}:{}", X_hat_nl_batch_t.min(), X_hat_nl_batch_t.max()) # 0.0:0.0
        # jax.debug.print("{}:{}", lhs_mat_batch_t.min(), lhs_mat_batch_t.max())   # 0.0:0.0

        def _compute_err(lam):
            phi_bar = lstsq_l2(X_hat_nl_batch_t, lhs_mat_batch_t, lam).T # lhs_mat_batch_t : (k, l)
            # return self.rom_err_fn(X_batch_t, Y_batch_t, phi_bar, selected_idx)
            return self.rom_err_fn(X_batch_t, Y_batch_t, phi_bar, soft_selection)

        errs = jax.vmap(_compute_err)(self.lam_vec)
        i_opt = jnp.nanargmin(errs)
        # lam_opt = self.lam_vec[i_opt]
        lam_opt = jax.lax.stop_gradient(self.lam_vec[i_opt])
        err_opt = errs[i_opt]

        jax.debug.print("{}", err_opt) # 2.523542394503168e+17, 1.4174103967629312e+16

        phi_bar_opt = lstsq_l2(X_hat_nl_batch_t, lhs_mat_batch_t, lam_opt).T
        # init_grads = jax.tree.map(lambda p: jnp.zeros_like(p), params)

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

        return (selected_idx, err_opt, phi_bar_opt, temperature_spt)
        # selected_idx, batch_loss, phi_bar_opt, new_temperature_spatial  


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


def train_for_epochs(state, X_train, Y_train, lhs_mat, S_val, phi_mat, A_tilde, U_l, min_tilde, max_tilde, library_functions, A_hat, num_epochs, start_temp, batch_size, rng, initial_lr, threshold):
    """Runs the inner training loop and returns updated state + final loss history."""

    X_train_t = X_train.T
    Y_train_t = Y_train.T
    lhs_mat_t = lhs_mat.T

    train_loss_history = []
    val_loss_history = []

    mean_max_prob1 = []
    logit_vals_hist1 = []

    temperature_spt = start_temp

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
        starts = make_epoch_starts(subkey, X_train_t.shape[0], batch_size, non_overlapping=True)
        epoch_loss = 0.0
        val_loss = 0.0
        n_samples_seen = 0

        min_batch_loss = float('inf')

        for start in starts:
            X_train_batch_t = X_train_t[start:start + batch_size]
            Y_train_batch_t = Y_train_t[start:start + batch_size]
            lhs_mat_batch_t = lhs_mat_t[start:start + batch_size]

            rng, rng_g, rng_d = jax.random.split(rng, 3)

            # print(temperature_spt)
            # print(state.params['encoder']['logits'][:5, :5],'\n____\n')

            state, batch_selected_idx, batch_loss, batch_phi_bar_opt, temperature_spt, grad_vals = train_step(
                state,
                X_train_batch_t,
                Y_train_batch_t,
                lhs_mat_batch_t,
                temperature_spt,
                rngs={'gumbel': rng_g, 'dropout': rng_d}
            )

            # print(grad_vals)

            # batch loss is the rel_loss between the X_rec and Y_batch.

            logits_enc1 = state.params['encoder']['logits']
            mean_max1 = jnp.mean(jnp.max(jax.nn.softmax(logits_enc1, axis=-1), axis=-1))
            mean_max_scalar1 = mean_max1.block_until_ready().item()
            mean_max_prob1.append(mean_max_scalar1)

            epoch_loss += batch_loss.item() * X_train_batch_t.shape[0]
            n_samples_seen += X_train_batch_t.shape[0]

        # working with the 'state' at the end of processing all the batches corresp. to the epoch. 
        logits_enc1_cpu = np.array(logits_enc1)
        del logits_enc1

        epoch_loss /= n_samples_seen

        logit_vals_hist1.append(logits_enc1_cpu)

        selected_indices_spt = jnp.argmax(logits_enc1_cpu, axis=1)

        preserved_grads.append(jax.tree_util.tree_map(lambda x: np.array(x), grad_vals))

        # only preserve grads at end of each epoch.
        # grad_vals_cpu = np.array(grad_vals)
        # preserved_grads.append(grad_vals_cpu)

        del grad_vals

        # del grad_vals, grad_vals_cpu

        train_loss_history.append(epoch_loss)

        # at end of each epoch:
        val_loss = validation_loss_compute(
            S_val, 
            phi_mat,
            A_tilde,
            batch_phi_bar_opt,
            U_l,
            library_functions,
            A_hat,
            min_tilde,
            max_tilde,
            selected_indices_spt)

        if(val_loss < best_val_loss):
            best_val_loss = val_loss
            best_params = state.params
        
        val_loss_history.append(val_loss.item())

    state = state.replace(params=best_params)

    return (state, temperature_spt, train_loss_history, val_loss_history, mean_max_prob1, logit_vals_hist1,  preserved_grads)