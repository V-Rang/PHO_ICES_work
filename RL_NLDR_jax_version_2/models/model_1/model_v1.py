import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Dict, Any

def get_prob_and_grad(params, apply_fn, perms, sample_arr) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Returns:
      p        : scalar probability = softmax(logits_cur)[idx_next]
      grads    : pytree of same structure as `params`, holding ∂p/∂params
    """
    matches = jnp.all(perms[:, None, :] == sample_arr[None, :, :], axis=-1)
    idxs = jnp.argmax(matches, axis=0)
    cur = idxs[:-1]
    nxt = idxs[1:]


    def one_pair(c, n):
        def f(p):
            logits = apply_fn(p, perms[c])
            probs = jax.nn.softmax(logits)
            return jnp.log(probs[n])

        p0 = jnp.exp(f(params))

        grad0 = jax.grad(f)(params)

        return (grad0, p0)

    sample_grads, sample_probs = jax.vmap(one_pair, in_axes=(0, 0))(cur, nxt)

    sample_grads = jax.tree.map(
            lambda g: jnp.sum(g, axis=0),
            sample_grads
    )

    # jax.debug.print("{}_______________", sample_grads)

    return (sample_grads, sample_probs)


class Model:

    def __init__(self, model_settings):
        self.apply_fn = model_settings['apply_fn']
        self.perms = jnp.array(model_settings['permutations'])
        self.selection_length = model_settings['selection_length']
        self.params = model_settings['params']

    def forward(self, inp_params, batch):
        B, m = batch.shape
        num_iters = m // self.selection_length
        batch = batch.reshape((B, num_iters, self.selection_length))

        # grad_computation_fn = jax.vmap(get_prob_and_grad, in_axes=(None,) * 4)

        grad_computation_fn = jax.vmap(get_prob_and_grad, 
                                       in_axes=(
                                        None, # inp_params
                                        None, # apply_fn
                                        None, # perms
                                        0) # batch
            )

        self.params = inp_params

        batch_grads, batch_prob_hist = grad_computation_fn(self.params, self.apply_fn, self.perms, batch)

        return (batch_grads, batch_prob_hist)