
import numpy as np
import random
from typing import Tuple
import matplotlib.pyplot as plt
import torch
import jax
import jax.numpy as jnp
from functools import partial
from typing import Sequence, Callable
from jax import jit
from typing import Tuple, Sequence
import jax.numpy as jnp
nnz = 8

@jit
def lstsq_l2(A: jnp.ndarray, B: jnp.ndarray, reg_magnitude) -> jnp.ndarray:
    """
    Solve the regularized least-squares problem
        min_X ‖A X - B‖_F^2 + reg_magnitude^2 ‖X‖_F^2
    via the SVD of A.

    Args:
      A: [m, n] design matrix
      B: [m, p] right-hand side
      reg_magnitude: scalar regularization parameter ε

    Returns:
      X: [n, p] solution
    """
    Phi, Sig, Psi_T = jnp.linalg.svd(A, full_matrices=False)
    sinv = Sig / (Sig ** 2 + reg_magnitude ** 2)
    A_pinv = Psi_T.T * sinv @ Phi.T
    return A_pinv @ B

def random_selection_arr_maker(k: int, l: int) -> jnp.ndarray:
    idx = np.random.choice(k, size=l, replace=False)
    arr = np.zeros(k, dtype=int)
    arr[idx] = 1
    return jnp.array(arr)

@partial(jax.jit, static_argnums=(1,))
def apply_selected_funcs(S_hat: jnp.ndarray, lib_funcs: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]) -> jnp.ndarray:
    results = [f(S_hat) for f in lib_funcs]
    return jnp.concatenate(results, axis=0)

def make_library_functions(lib_funcs_strs):
    py_funcs = []
    for func_str in lib_funcs_strs:
        f = eval(f'lambda _: {func_str}', {'jnp': jnp})
        py_funcs.append(f)
    return py_funcs


@partial(jax.jit, static_argnums=(7,))
def rom_reconstruction_error(X_train: jnp.ndarray, Y_train: jnp.ndarray, phi_mat: jnp.ndarray, A_hat: jnp.array, A_tilde: jnp.ndarray, phi_bar_mat: jnp.ndarray, U_r: jnp.ndarray, library_functions: Sequence, selection_arr: jnp.ndarray) -> jnp.ndarray:
    H_hat = phi_mat.T @ A_tilde @ phi_bar_mat
    mask = selection_arr.astype(bool)
    idx_array = jnp.nonzero(mask, size=nnz)[0]
    x0 = X_train[:, 0]
    x_hat0 = U_r.T @ x0

    def step(xh, _):
        mod = apply_selected_funcs(xh, library_functions)
        mod_sel = jnp.take(mod, idx_array)
        xh = A_hat @ xh + H_hat @ mod_sel
        return (xh, xh)
    _, xh_seq = jax.lax.scan(step, x_hat0, None, length=Y_train.shape[1])
    X_rec = U_r @ xh_seq.T
    
    return jnp.linalg.norm(Y_train - X_rec)/jnp.linalg.norm(Y_train)