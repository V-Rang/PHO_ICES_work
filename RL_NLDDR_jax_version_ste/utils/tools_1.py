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

# nnz = 20

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
    # return jnp.concatenate(results, axis=1)
    return jnp.concatenate(results, axis=0)


def make_library_functions(lib_funcs_strs):
    py_funcs = []
    for func_str in lib_funcs_strs:
        f = eval(f'lambda _: {func_str}', {'jnp': jnp})
        py_funcs.append(f)
    return py_funcs


def build_library(X_hat: jnp.ndarray,
                  funcs: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]) -> jnp.ndarray:
    """
    X_hat: (batch, r)          # reduced variables
    funcs: list of callables f: (batch,r)->(batch,r)
    returns: (batch, r * L)    # concatenated along feature dim
    """
    outs = [f(X_hat) for f in funcs]            # each (b,r)
    return jnp.concatenate(outs, axis=1)        # (b, rL)


# @partial(jax.jit, static_argnums=(7,))
# def rom_reconstruction_error(x_mat_t_batch: jnp.ndarray, y_mat_t_batch: jnp.ndarray, phi_mat: jnp.ndarray, A_hat: jnp.array, A_tilde: jnp.ndarray, phi_bar_mat: jnp.ndarray, U_r: jnp.ndarray, library_functions: Sequence, idx_array) -> jnp.ndarray:
#     x_mat_batch = x_mat_t_batch.T
#     y_mat_batch = y_mat_t_batch.T
#     H_hat = phi_mat.T @ A_tilde @ phi_bar_mat
#     x0 = x_mat_batch[:, 0]
#     x_hat0 = U_r.T @ x0

#     def step(xh, _):
#         mod = apply_selected_funcs(xh, library_functions)
#         mod_sel = jnp.take(mod, idx_array)
#         xh = A_hat @ xh + H_hat @ mod_sel
#         return (xh, xh)
#     _, xh_seq = jax.lax.scan(step, x_hat0, None, length=y_mat_batch.shape[1])
#     X_rec = U_r @ xh_seq.T
#     return jnp.linalg.norm(y_mat_batch - X_rec) / jnp.linalg.norm(y_mat_batch)


def make_rom_reconstruction_error(phi_mat, A_hat, A_tilde, U_r, library_functions):
    """
    Build a jitted function that only depends on (X_batch_t, Y_batch_t, phi_bar, selected_idx).
    All other arguments are captured in the closure.
    """

    # Precompute the part of Ĥ that doesn’t change across calls
    H_left = phi_mat.T @ A_tilde                      # shape: (r, m) if phi_bar is (m,)

    # Wrap once so we don't pass callables through JIT each call
    def apply_funcs(xh):
        return apply_selected_funcs(xh, library_functions)  # your existing helper

    @jax.jit
    def rom_reconstruction_error(X_batch_t, Y_batch_t, phi_bar, selected_idx):

        """
        X_batch_t: (nx, T)   full-order states used for IC (transpose of your (T, nx) if needed)
        Y_batch_t: (nx, T)   target rollout to compare against
        phi_bar:   (m,)      coefficients for library functions
        selected_idx: (K,)   integer indices selecting which library functions to use
        """

        H_hat = H_left @ phi_bar                       # shape: (r, K) after later selection

        # jax.debug.print("{}:{}", phi_bar.min(), phi_bar.max()) # 0.0:0.0
        # jax.debug.print("{}:{}", A_hat.min(), A_hat.max()) # -0.06317138671875:1.0
        # jax.debug.print("{}:{}", H_hat.min(), H_hat.max()) # 0.0:0.0

        X_batch = X_batch_t.T                          # (T, nx)
        Y_batch = Y_batch_t.T                          # (T, nx)

        x0 = X_batch[:, 0]                             # (nx,)
        x_hat0 = U_r.T @ x0

        def step(xh, _):
            mod = apply_funcs(xh)                      # e.g. shape (m,)
            mod_sel = jnp.take(mod, selected_idx)      # (K,)

            rhs_term1 = A_hat @ xh
            rhs_term2 = H_hat @ mod_sel

            xh = rhs_term1 + rhs_term2

            return xh, xh   

        _, xh_seq = jax.lax.scan(step, x_hat0, None, length=Y_batch.shape[1] + 1)

        X_rec_batch = U_r @ xh_seq.T

        num = jnp.linalg.norm(Y_batch - X_rec_batch[:, 1: ])
        den = jnp.linalg.norm(Y_batch) + 1e-12       # small eps for safety
        
        return num / den


    return rom_reconstruction_error


# @jax.jit(static_argnames=["library_functions"])
# def rom_reconstruction_error(X_batch_t, Y_batch_t, phi_mat, A_hat, 
#                         A_tilde, phi_bar, U_r, library_functions, selected_idx):

#     H_hat = phi_mat.T @ A_tilde @ phi_bar

#     X_batch = X_batch_t.T
#     Y_batch = Y_batch_t.T

#     x0 = X_batch[:, 0]
#     x_hat0 = U_r.T @ x0

#     def step(xh, _):
#         mod = apply_selected_funcs(xh, library_functions)
#         mod_sel = jnp.take(xh, selected_idx)

#         xh = A_hat @ xh + H_hat @ mod_sel
#         return (xh, xh)

#     _, xh_seq = jax.lax.scan(step, x_hat0, None, length=Y_batch.shape[1])
#     X_rec_batch = U_r @ xh_seq.T

#     return jnp.linalg.norm(Y_batch - X_rec_batch)/jnp.linalg.norm(Y_batch)

