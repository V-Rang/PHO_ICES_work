
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
nnz = 20

def _process_one_sample(params, apply_fn, X_hat_mod, lhs_mat, X_train, Y_train, phi_mat, A_hat, A_tilde_operator, U_r, library_functions, lam_vec, perms, selection_length, num_iters, sample_arr):
    jax.debug.print('{}', params)
    mask = sample_arr.astype(bool)
    idxs = jnp.nonzero(mask, size=nnz)[0]
    X_hat_nl = jnp.take(X_hat_mod, idxs, axis=0)

    def _compute_err(lam):
        phi_bar = lstsq_l2(X_hat_nl.T, lhs_mat.T, lam).T
        return rom_reconstruction_error(X_train, Y_train, phi_mat, A_hat, A_tilde_operator, phi_bar, U_r, library_functions, sample_arr)
    errs = jax.vmap(_compute_err)(lam_vec)
    i_opt = jnp.nanargmin(errs)
    lam_opt = lam_vec[i_opt]
    err_opt = errs[i_opt]
    phi_bar_opt = lstsq_l2(X_hat_nl.T, lhs_mat.T, lam_opt).T
    init_grads = jax.tree.map(lambda p: jnp.zeros_like(p), params)

    def _body(carry, sel_slice):
        grads_acc, p_prev = carry
        out = output_selection(params, apply_fn, p_prev, sel_slice, perms)
        grads_acc = jax.tree.map(lambda g_acc, g: g_acc + g, grads_acc, out['gradients'])
        return ((grads_acc, jnp.array([out['output']])), out['output'])
    sel_slices = sample_arr.reshape((num_iters, selection_length))
    (final_grads, _), prob_hist = jax.lax.scan(_body, (init_grads, jnp.array([1.0], jnp.float32)), sel_slices)
    prod_p = jnp.prod(prob_hist)
    sample_reward = -prod_p * err_opt ** 2
    sample_grads = final_grads
    return (err_opt, sample_grads, sample_reward, phi_bar_opt, prob_hist, lam_opt)

class Model:

    def __init__(self, model_settings: Dict[str, Any]):
        self.library_functions = tuple(model_settings['library_functions'])
        self.gamma = model_settings['gamma']
        self.X_train = jnp.array(model_settings['X_train'], dtype=jnp.float32)
        self.Y_train = jnp.array(model_settings['Y_train'], dtype=jnp.float32)
        self.U_r = jnp.array(model_settings['U_r'], dtype=jnp.float32)
        self.A_tilde_operator = jnp.array(model_settings['A_tilde_operator'], dtype=jnp.float32)
        self.X_tilde = jnp.array(model_settings['X_tilde'], dtype=jnp.float32)
        self.X_hat = jnp.array(model_settings['X_hat'], dtype=jnp.float32)
        self.phi_mat = jnp.array(model_settings['phi_mat'], dtype=jnp.float32)
        self.perms = jnp.array(model_settings['permutations'])
        self.trunc_dim = model_settings['trunc_dim']
        self.selection_length = model_settings['selection_length']
        self.sub_selection_length = model_settings['sub_selection_length']
        self.d_model = model_settings['d_model']
        self.e_layers = model_settings['e_layers']
        self.learning_rate = model_settings['learning_rate']
        self.batch_size = model_settings['batch_size']
        self.params = model_settings['params']
        self.apply_fn = model_settings['apply_fn']
        funcs_list = make_library_functions(self.library_functions)
        self.library_functions = tuple(funcs_list)
        self.X_hat_mod = apply_selected_funcs(self.X_hat, self.library_functions)
        self.lhs_mat = self.X_tilde - self.phi_mat @ self.X_hat
        self.lam_vec = jnp.array(tuple(model_settings['lam_vec']))
        self.A_hat = self.phi_mat.T @ self.A_tilde_operator @ self.phi_mat

    def forward(self, inp_sel_arrs: np.ndarray) -> Tuple[Any, ...]:
        batch = jnp.array(inp_sel_arrs, dtype=jnp.int32)
        B, m = batch.shape
        num_iters = m // self.selection_length
        err_computation_fn = jax.vmap(_process_one_sample, in_axes=(None,) * 7)
        batch_reconstr_vals, batch_grads, batch_rewards, phi_bar_opt, prob_hist, lam_opt = err_computation_fn(self.params, self.apply_fn, self.X_hat_mod, self.lhs_mat, self.X_train, self.Y_train, self.phi_mat, self.A_hat, self.A_tilde_operator, self.U_r, self.library_functions, self.lam_vec, self.perms, self.selection_length, num_iters, batch)
        best_idx = jnp.argmax(batch_rewards)
        best_sample_batch = {'selection_arr': batch[best_idx], 'probability_arr': prob_hist[best_idx], 'reconstr_err': batch_reconstr_vals[best_idx], 'sample_reward': batch_rewards[best_idx], 'phi_bar_mat': phi_bar_opt[best_idx], 'lam_opt': lam_opt[best_idx]}
        return (batch_reconstr_vals, batch_grads, batch_rewards, best_sample_batch, prob_hist)

    def update_grads(self, new_grads: Any, path: str, batch_sample_rewards: List[float]) -> float:
        lr_mod = 0.001 / abs(min(batch_sample_rewards))
        self.params = jax.tree.map(lambda w, g: w + lr_mod / len(batch_sample_rewards) * g, self.params, new_grads)
        with open(os.path.join(path, 'params.pkl'), 'wb') as fp:
            pickle.dump(self.params, fp)
            return lr_mod
            return lr_mod