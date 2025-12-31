import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple, Sequence
import itertools

# out = output_selection(params, apply_fn, p_prev, sel_slice, perms)
def output_selection(params: dict, apply_fn: callable, p_prev: jnp.ndarray, selection_arr: Tuple[int, ...], perms) -> dict:
    """
    For a given tuple `selection_arr`, builds all unique permutations
    with the same number of 1's, runs them through the network,
    forms a softmax over exp(out_i), and returns:
      {
        'output':  softmax_value for the original selection_arr,
        'gradients': pytree of d(softmax)/dparams
      }
    """

    batch_inputs = jnp.concatenate([jnp.broadcast_to(p_prev, (perms.shape[0], 1)), perms], axis=1)
    ys = jax.vmap(lambda inp: apply_fn(params, inp).squeeze())(batch_inputs)
    P_max = jnp.max(ys)
    ys_shifted = ys - P_max
    exp_ys_shifted = jnp.exp(ys_shifted)
    sum_exp_ys_shifted = jnp.sum(exp_ys_shifted)
    log_ys_shited = jnp.log(sum_exp_ys_shifted)
    log_ys_probs = ys_shifted - log_ys_shited
    log_probs = jnp.exp(log_ys_probs)

    def single_grad(inp):
        return jax.grad(lambda P: apply_fn(P, inp).squeeze())(params)
    
    grads_per_perm = jax.vmap(single_grad)(batch_inputs)

    # jax.debug.print("{}", selection_arr.shape) # (4, )

    matches = jnp.all(perms == selection_arr, axis=1)

    # jax.debug.print("{}", perms) # (6, 4)
    # jax.debug.print("{}", selection_arr) # (4,4 )

    idx0 = jnp.argmax(matches)

    prob0 = log_probs[idx0]
    d_out0 = jax.tree_map(lambda leaf: leaf[idx0], grads_per_perm)
    exp_ys_shifted0 = exp_ys_shifted[idx0]
    soft0 = exp_ys_shifted0 / sum_exp_ys_shifted
    sum_j = jax.tree_map(lambda leaf: jnp.tensordot(exp_ys_shifted, leaf, axes=(0, 0)), grads_per_perm)
    # jax.debug.print("{}", sum_j['params'].keys())

    grad_soft = jax.tree_map(lambda g0, gj: soft0 * g0 - exp_ys_shifted0 / (sum_exp_ys_shifted * sum_exp_ys_shifted) * gj, d_out0, sum_j)
    
    return {'output': prob0, 'gradients': grad_soft}