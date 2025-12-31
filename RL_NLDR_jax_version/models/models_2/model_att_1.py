
import os
import math
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Tuple, List
from layers.Enc_Dec import Encoder_Decoder
from utils.tools_2 import apply_selected_funcs, lstsq_l2
from utils.tools_2 import rom_reconstruction_error
from layers.output_grad_comp_att_1 import output_selection
from utils.tools_2 import make_library_functions
import math
nnz = 8

def _process_one_sample(params, apply_fn, perms, selection_length, num_iters, sample_arr):
    init_grads = jax.tree.map(lambda p: jnp.zeros_like(p), params)

    def _body(carry, sel_slice):
        grads_acc, p_prev = carry
        out = output_selection(params, apply_fn, p_prev, sel_slice, perms)
        grads_acc = jax.tree.map(lambda g_acc, g: g_acc + g, grads_acc, out['gradients'])
        return ((grads_acc, jnp.array([out['output']])), out['output'])
    sel_slices = sample_arr.reshape((num_iters, selection_length))
    (final_grads, _), prob_hist = jax.lax.scan(_body, (init_grads, jnp.array([1.0], jnp.float32)), sel_slices)
    return (final_grads, prob_hist)

class Model:

    def __init__(self, model_settings: Dict[str, Any]):
        self.params = model_settings['params']
        self.apply_fn = model_settings['apply_fn']
        self.perms = jnp.array(model_settings['permutations'])
        self.selection_length = model_settings['selection_length']

    def forward(self, inp_params, inp_sel_arrs: np.ndarray) -> Tuple[Any, ...]:
        batch = jnp.array(inp_sel_arrs, dtype=jnp.int32)
        B, m = batch.shape
        num_iters = m // self.selection_length
        grad_computation_fn = jax.vmap(_process_one_sample, in_axes=(None,) * 7)
        self.params = inp_params
        batch_grads, batch_prob_hist = grad_computation_fn(self.params, self.apply_fn, self.perms, self.selection_length, num_iters, batch)
        return (batch_grads, batch_prob_hist)