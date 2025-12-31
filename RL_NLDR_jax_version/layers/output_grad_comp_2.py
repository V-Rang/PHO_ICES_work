
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple, Sequence
import itertools

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
    exp_ys = jnp.exp(ys)
    sum_exp_ys = jnp.sum(exp_ys)
    prob_outputs = jnp.log(exp_ys / sum_exp_ys)

    def single_grad(inp):
        return jax.grad(lambda P: apply_fn(P, inp).squeeze())(params)
    
    grads_per_perm = jax.vmap(single_grad)(batch_inputs)
    matches = jnp.all(perms == selection_arr, axis=1)
    idx0 = jnp.argmax(matches)
    prob0 = prob_outputs[idx0]
    d_out0 = jax.tree_map(lambda leaf: leaf[idx0], grads_per_perm)
    sum_j = jax.tree_map(lambda leaf: jnp.tensordot(exp_ys, leaf, axes=(0, 0)), grads_per_perm)
    grad_soft = jax.tree_map(lambda g0, gj: g0 - 1 / sum_exp_ys * gj, d_out0, sum_j)
    return {'output': prob0, 'gradients': grad_soft}