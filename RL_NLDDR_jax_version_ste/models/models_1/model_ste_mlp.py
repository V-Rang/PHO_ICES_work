# ===== Imports =====
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Callable, Any, Tuple, Sequence
from jax.nn.initializers import glorot_normal
from tqdm import tqdm
from functools import partial
from utils.tools_1_normalized_m2 import apply_selected_funcs
from jax import lax

# --------------------------------------------------------------------------------------
# 1) STE utilities
# --------------------------------------------------------------------------------------
def hard_step_with_clip_factory(clip_value: float = 1.0):
    """Returns an STE function f(w) with:
       forward:  hard step 1[w>0]
       backward: identity gradient clipped to [-clip_value, clip_value].
    """

    @jax.custom_vjp
    def f(w):
        return (w > 0).astype(w.dtype)   # hard forward

    def f_fwd(w):
        y = (w > 0).astype(w.dtype)
        return y, None                   # no residuals

    def f_bwd(_, g):
        g_clipped = jnp.clip(g, -clip_value, clip_value)
        return (g_clipped,)

    f.defvjp(f_fwd, f_bwd)

    return f

# --------------------------------------------------------------------------------------
# 2) Library builder (replace with yours)
# --------------------------------------------------------------------------------------
# def build_library_default(x_row: jnp.ndarray) -> jnp.ndarray:
#     """
#     Example per-row library. Replace with your real library builder.
#     Input:  x_row: (r,)
#     Output: feats: (m,) ; here m = 2r (identity + squares).
#     """
#     return jnp.concatenate([x_row, x_row**2], axis=-1)

def build_library(X_hat: jnp.ndarray,
                  funcs: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]) -> jnp.ndarray:
    """
    X_hat: (batch, r)          # reduced variables
    funcs: list of callables f: (batch,r)->(batch,r)
    returns: (batch, r * L)    # concatenated along feature dim
    """
    outs = [f(X_hat) for f in funcs]            # each (b,r)
    return jnp.concatenate(outs, axis=1)        # (b, rL)


# Row-wise wrapper so we can vmap over batch:
# def build_library_batch(X: jnp.ndarray,
#                         row_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
#     """
#     X: (B, r) -> (B, m) via vmap(row_fn).
#     """
#     return jax.vmap(row_fn, in_axes=0, out_axes=0)(X)

# --------------------------------------------------------------------------------------
# 3) Model: X -> library -> STE mask -> MLP head -> lhs
# --------------------------------------------------------------------------------------
class MLP(nn.Module):
    out_dim: int # l
    hidden: Tuple[int, ...] = (32, 32)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.tanh(nn.Dense(self.out_dim)(x)) # constraint output in [-1, 1]

class ConcreteSelector(nn.Module):
    in_out_dim: int # ( l x len(lib) )
    clip_value: float = 1.0  # how aggressively to clip the STE grad

    @nn.compact
    def __call__(self, x): # (B, l x len(lib) )
        logits = self.param('logits', glorot_normal(), (1, self.in_out_dim))
        
        # if sigmoid in backward pass
        y_soft = jax.nn.sigmoid(logits)              
        y_hard = (logits > 0).astype(logits.dtype)           
        mask = y_soft + jax.lax.stop_gradient(y_hard - y_soft)
        # jax.debug.print("soft: {}", y_soft)
        feats_masked = x * mask # (B, l x len(lib) * (1, l x len(lib) ))                   
        
        # jax.debug.print("{}:{}", y_soft, y_hard)

        return feats_masked, mask.squeeze(), y_soft.squeeze() # (B, l x len(lib)), (l x len(lib))

        # is ST grad in backward pass:
        # ste_hard = hard_step_with_clip_factory(self.clip_value)
        # mask = ste_hard(logits)                               # (D,) hard forward, clipped grad
        # feats_masked = x * mask[None, :]                      # (B, D)
        # return feats_masked, mask

# --------------------------------------------------------------------------------------
# 5) Train state and steps
# --------------------------------------------------------------------------------------
class TrainState(train_state.TrainState):
    pass

def create_train_state(rng,
                       model,
                       sample_input: jnp.ndarray, # (1, l_val)
                       lr: float = 1e-3) -> TrainState:
    
    rng_params, rng_gumbel, rng_Dropout = jax.random.split(rng, num=3)

    init_vars = model.init({'params': rng_params, 'gumbel': rng_gumbel, 'dropout': rng_Dropout}, sample_input)
    params = init_vars['params']
    tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


class STEFeatureSelectorModel(nn.Module):
    """
    - Builds library features per row (via row_library_fn).
    - Applies a *global* STE mask over the m library features (broadcast to batch).
      (If you want per-row masks, we can move the mask inside vmap.)
    - MLP head maps masked features (B, m) -> (B, l).
    """

    library_functions: Callable[[jnp.ndarray], jnp.ndarray]
    lib_dim: int     # l x len(lib)
    out_dim: int     # l
    mlp_hidden: Tuple[int, ...] = (128, 128)

    def setup(self):
        self.concrete_selector = ConcreteSelector(self.lib_dim)
        self.decoder = MLP(self.out_dim, self.mlp_hidden)

    def __call__(self, X_tilde_norm_batch_t: jnp.ndarray): # (B, l)
        """
        X_tilde_norm_batch_t: (B, l)
        Returns:
          preds: (B, l)
          aux:   {'mask': (m,), 'feats': (B,m), 'feats_masked': (B,m)}
        """

        feats = build_library(X_tilde_norm_batch_t, self.library_functions) # (B, l x len(lib)) # x_mod

        feats_masked, mask_hard, mask_soft = self.concrete_selector(feats) # (B, l x len(lib))

        preds = self.decoder(feats_masked) # \phi(x_nl)

        feats_0 = jnp.zeros((mask_hard.shape),dtype = jnp.int32)
        preds_0_mask = jax.lax.stop_gradient(self.decoder(feats_0))

        # jax.debug.print("{}\n____\n{}\n____\n{}\n_____\n{}\n______\n{}", X_tilde_norm_batch_t.shape, feats.shape, mask.shape, feats_masked.shape, preds.shape)
        # print(type(feats), type(feats_masked), type(mask), type(preds))
        # print(feats.dtype, feats_masked.dtype, mask.dtype, preds.dtype)

        aux = {'mask_hard': mask_hard, 'mask_soft': mask_soft, 'feats': feats, 'feats_masked': feats_masked, 'preds_0_mask': preds_0_mask}

        return preds, aux

    def decode(self, feats_masked):
        return self.decoder(feats_masked)

# --------------------------------------------------------------------------------------
# 4) Losses
# --------------------------------------------------------------------------------------
def mse_loss(preds: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((preds - target) ** 2)

def total_loss_fn(params,
                  X_batch: jnp.ndarray,
                  lhs_batch: jnp.ndarray,
                  lambda_l1: float = 1e-3):
    """
    Supervised mapping: X_row -> lhs_row, row-wise.
    Loss = MSE + lambda * L1(mask).
    """
    
    preds, aux = state.apply_fn({'params': state.params}, X_batch)

    recon = mse_loss(preds, lhs_batch)          # row-wise regression loss
    # l1 = jnp.mean(jnp.abs(aux['mask']))         # encourages sparsity
    l1 = jnp.sum(jnp.abs(aux['mask']))         # encourages sparsity

    loss = recon + lambda_l1 * l1
    return loss, {'recon': recon, 'l1': l1, 'mask': aux['mask']}


@partial(jax.jit, static_argnames=['lambda_l1', 'min_tilde', 'max_tilde', 'lib_dim', 'b_size', 'pred_len'])
def train_step(state: TrainState,
               X_batch: jnp.ndarray,  # (Nh, b_size) # cols: (0 .. b_size-1)
               Y_batch: jnp.ndarray, # (Nh, b_size + pred_len - 1) 
               
               # 0 -> (1 .. plen), 
               # 1 -> (2 .. plen+1), 
               # b_size-1 -> (b_size -> b_size + plen - 1), 
               # so (1 .. b_size + plen - 1) needed, starting from 1 ahead of X_batch. 
               
               lambda_l1: float,
               A_tilde: jnp.ndarray,
               c_tilde: jnp.ndarray,
               U_l: jnp.ndarray,
               min_tilde: jnp.ndarray,
               max_tilde: jnp.ndarray,
               lib_dim: int,
               b_size: int,
               pred_len: int
               ):

    def loss_fn(params):

        X_tilde = U_l.T @ X_batch                # (l_val, b_size)
        X_tilde_norm = (X_tilde - min_tilde) / (max_tilde - min_tilde)  # (l_val, b_size)

        X_tilde_norm = X_tilde_norm.T            # (b_size, l_val) for vmap
    
        # def single_rollout(x_init_tilde_norm):

        #     def step(xh,_):
        #         phi, aux = state.apply_fn({'params': params}, xh.reshape(1, -1))
        #         phi = phi.squeeze()
        #         mask_t = aux['mask']
        #         mask_soft = aux['mask_soft']
        #         xh = A_tilde @ xh + c_tilde + phi
        #         return xh, (xh, mask_t, mask_soft)

        #     _, (x_seq, mask_seq, mask_soft) = jax.lax.scan(step, x_init_tilde_norm, None, length= pred_len + 1)
            
        #     return x_seq, mask_seq, mask_soft  # (pred_len + 1, l_val)

        def single_rollout(x_init_tilde_norm):
            def step(xh,_):
                phi,_ = state.apply_fn({'params': params}, xh.reshape(1, -1))
                phi = phi.squeeze()
                xh = A_tilde @ xh + c_tilde + phi
                return xh, xh

            _, x_seq = jax.lax.scan(step, x_init_tilde_norm, None, length= pred_len + 1)
            
            return x_seq  # (pred_len + 1, l_val)

        # x_seqs = jax.vmap(single_rollout)(X_tilde_norm)   # (b_size, pred_len+1, l_val)

        # x_seqs, mask_seqs, mask_soft = jax.vmap(single_rollout)(X_tilde_norm)
        x_seqs = jax.vmap(single_rollout)(X_tilde_norm)
        
        # jax.debug.print("{}", mask_seqs.shape)
        # mask_seqs = mask_seqs[:, 1:, :]  # (b_size, pred_len, d)
        # mask_softs = mask_soft[:, 1:, :]  # (b_size, pred_len, d)

        x_seqs = x_seqs[:, 1:, :]                  # Drop first step, (b_size, pred_len, l_val)

        x_seqs = jnp.transpose(x_seqs, (0, 2, 1))   # (b_size, l_val, pred_len)

        # jax.debug.print("{}:{}", x_seqs.min(), x_seqs.max())

        x_seqs = (max_tilde - min_tilde) * x_seqs + min_tilde  # unnorm

        x_preds = jax.vmap(lambda x: U_l @ x)(x_seqs)  # (b_size, Nh, pred_len)

        # jax.debug.print("{}:{}", x_preds.min(), x_preds.max())

        Y_targets = jnp.stack([Y_batch[:, i : i + pred_len] for i in range(X_batch.shape[1])])  # (b_size, Nh, pred_len)

        # jax.debug.print("{}:{}", x_preds.shape, Y_targets.shape) # (bsize, Nh, p_len)

        numerators = jnp.linalg.norm(x_preds - Y_targets, axis=(1,2))        # (b_size,)
        denominators = jnp.linalg.norm(Y_targets, axis=(1,2)) + 1e-8         # avoid division by zero
        relative_errors = numerators / denominators                      # (b_size,)
        recon_loss = jnp.mean(relative_errors)

        # jax.debug.print("{}:{}", numerators.shape, denominators.shape)

        # Estimate L1 penalty (optional)
        mask_hard = state.apply_fn({'params': params}, X_tilde_norm[0].reshape(1, -1))[1]['mask_hard']  # use 1st example
        mask_soft = state.apply_fn({'params': params}, X_tilde_norm[0].reshape(1, -1))[1]['mask_soft']  # use 1st example

        # jax.debug.print("{}:{}", mask_seqs.shape, mask_hard.shape)
        # jax.debug.print("{}\n{}", mask_seqs[0,0,:], mask_hard)
        # jax.debug.print("{}", jnp.allclose(mask_seqs[0,0,:], mask_hard) ) # True
        # jax.debug.print("{}", jnp.allclose(mask_softs[0,0,:], mask_soft) ) # True

        # l1 = jnp.mean(jnp.sum(jnp.abs(mask_seqs)))

        # l1 = jnp.mean(jnp.abs(mask_seqs))
        l1 = jnp.mean(mask_hard)

        return recon_loss + lambda_l1 * l1, {'recon': recon_loss, 'l1': l1, 'mask_hard': mask_hard, 'mask_soft': mask_soft}

    (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss, logs, grads


# validation_log =                    eval_step(state,
#                                         library_functions,
#                                         X_val,
#                                         Y_val, 
#                                         U_l, 
#                                         min_tilde, 
#                                         max_tilde, 
#                                         learned_mask, 
#                                         learned_mask_0, 
#                                         A_tilde, 
#                                         c_tilde,
#                                         batch_size,
#                                         pred_len)


@partial(jax.jit, static_argnames=['library_functions', 'min_tilde', 'max_tilde', 'batch_size', 'pred_len', 'lambda_l1'])
def eval_step(
    state,
    library_functions,
    X_val: jnp.ndarray,
    Y_val: jnp.ndarray,
    U_l: jnp.ndarray,
    min_tilde: jnp.ndarray,
    max_tilde: jnp.ndarray,
    learned_mask,
    learned_mask_0,
    A_tilde: jnp.ndarray,
    c_tilde: jnp.ndarray,
    batch_size: int,
    pred_len: int,
    lambda_l1: float = 0.0,
):

    # print(type(X_val), type(Y_val)) # <DynamicJaxprTracer'> <DynamicJaxprTracer'>

    X_tilde = U_l.T @ X_val
    X_tilde_norm_glb = ((X_tilde - min_tilde) / (max_tilde - min_tilde)).T  # (seq_len, l)

    # ---- Single rollout over pred_len steps starting from x_init_tilde_norm ----
    # def single_rollout_eval(x_init_tilde_norm: jnp.ndarray) -> jnp.ndarray:
    #     def step(carry, _):
    #         xh1, xh0 = carry
    #         X_pair = jnp.stack([xh1, xh0], axis  = 0)
    #         feats = build_library(X_pair, library_functions)

    #         masks = jnp.stack([learned_mask, learned_mask_0], axis = 0)
    #         feats_masked = feats * masks
    
    #         phi_pair = state.apply_fn(
    #                 {'params': state.params},
    #                 feats_masked,
    #                 method=STEFeatureSelectorModel.decode
    #         )   

    #         phi1 = phi_pair[0]    # (l,)
    #         phi0 = phi_pair[1]    # (l,)

    #         xh1_next = A_tilde @ xh1 + c_tilde + phi1                      # (l,)
    #         xh0_next = A_tilde @ xh0 + c_tilde + phi0                      # (l,)

    #         return (xh1_next, xh0_next), (xh1_next, xh0_next)

    #     (x_seq_1, x_seq_0) = jax.lax.scan(
    #         step,
    #         (x_init_tilde_norm, x_init_tilde_norm),   # carry = (xh1, xh0)
    #         None,
    #         length=pred_len
    #     )[1]  # we only need 'ys'

    #     return x_seq_1, x_seq_0


    def single_rollout_eval(x_init_tilde_norm: jnp.ndarray) -> jnp.ndarray:
        def step(carry, _):
            xh1, xh0 = carry

            X_pair = jnp.stack([xh1, xh0], axis=0)        # (2, l)
            feats = build_library(X_pair, library_functions)   # (2, m) where m = l * len(lib)

            # Mask for both in batch
            feats_masked_1 = feats[0] * learned_mask      # (m,)
            feats_masked_0 = feats[1] * learned_mask_0    # (m,)
            feats_masked_pair = jnp.stack([feats_masked_1, feats_masked_0], axis=0)  # (2, m)

            phi_pair = state.apply_fn(
                {'params': state.params},
                feats_masked_pair,
                method=STEFeatureSelectorModel.decode
            )                                            # (2, l)

            phi1 = phi_pair[0]                           # (l,)
            phi0 = phi_pair[1]                           # (l,)

            # Advance both systems
            xh1_next = A_tilde @ xh1 + c_tilde + phi1    # (l,)
            xh0_next = A_tilde @ xh0 + c_tilde + phi0    # (l,)

            return (xh1_next, xh0_next), (xh1_next, xh0_next)

            # feats = build_library(xh.reshape(1,-1), library_functions)
            # feats_masked = feats * learned_mask
            # preds, aux = state.apply_fn(
            #         {'params': state.params},
            #         feats_masked,
            #         method=STEFeatureSelectorModel.decode
            # )   
            # phi1 = preds
            # phi0 = aux['preds_0_mask']
            # xh1_next = A_tilde @ xh1 + c_tilde + phi1                      # (l,)
            # xh0_next = A_tilde @ xh0 + c_tilde + phi0                      # (l,)
            # return (xh1_next, xh0_next), (xh1_next, xh0_next)

        (final_carry, (x_seq_1, x_seq_0)) = jax.lax.scan(
                step,
                (x_init_tilde_norm, x_init_tilde_norm),   # carry = (xh1, xh0)
                None,
                length=pred_len
            )

        return x_seq_1, x_seq_0

        # (x_seq_1, x_seq_0) = jax.lax.scan(
        #     step,
        #     (x_init_tilde_norm),   # carry = (xh1, xh0)
        #     None,
        #     length=pred_len
        # )[1]  # we only need 'ys'

        # return x_seq_1, x_seq_0

        # _, x_seq = jax.lax.scan(step, x_init_tilde_norm, None, length=pred_len)  # (pred_len+1, l)
        # return x_seq

    l1 = jnp.sum(jnp.abs(learned_mask))

    # ---- Accumulate relative error over batches ----
    total_recon_loss = 0.0
    total_count = 0

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    starts = make_epoch_starts(
        subkey,
        n_cols_x = X_val.shape[1],
        batch_size=batch_size,
        pred_len=pred_len,
        n_cols_y = Y_val.shape[1]
        # non_overlapping=True,
    )

    def body(carry, i):
        total_recon_sum_1, total_recon_sum_0, total_count = carry

        s = starts[i]  # starts is a JAX array of int32
        Xb_norm = lax.dynamic_slice_in_dim(X_tilde_norm_glb, s, batch_size, axis=0)
        Yb = lax.dynamic_slice_in_dim(Y_val, s, batch_size+pred_len-1, axis=1)

        # … run your batched rollout, build Y_targets_b via vmap+dynamic_slice_in_dim …
        # suppose `rel_err` is mean relative error for this batch (scalar)

        # x_seqs = jax.vmap(single_rollout_eval)(Xb_norm)             # (B, pred_len+1, l)
        # x_seqs = jnp.transpose(x_seqs, (0, 2, 1))         # (B, l, pred_len)
        # x_seqs = (max_tilde - min_tilde) * x_seqs + min_tilde  # broadcast over (l,)
        # x_preds = jax.vmap(lambda x: U_l @ x)(x_seqs)  # (b_size, Nh, pred_len)

        x_seqs_1, x_seqs_0 = jax.vmap(single_rollout_eval)(Xb_norm)        

        x_seqs_1 = jnp.transpose(x_seqs_1, (0, 2, 1))         # (B, l, pred_len)
        x_seqs_1 = (max_tilde - min_tilde) * x_seqs_1 + min_tilde  # broadcast over (l,)
        x_preds_1 = jax.vmap(lambda x: U_l @ x)(x_seqs_1)  # (b_size, Nh, pred_len)

        x_seqs_0 = jnp.transpose(x_seqs_0, (0, 2, 1))         # (B, l, pred_len)
        x_seqs_0 = (max_tilde - min_tilde) * x_seqs_0 + min_tilde  # broadcast over (l,)
        x_preds_0 = jax.vmap(lambda x: U_l @ x)(x_seqs_0)  # (b_size, Nh, pred_len)

        offsets = jnp.arange(batch_size, dtype=jnp.int32)

        def take_window(off):
            return lax.dynamic_slice_in_dim(Yb, start_index=off, slice_size=pred_len, axis=1)  # (Nh, pred_len)

        Y_targets = jax.vmap(take_window)(offsets)                                            # (B, Nh, pred_len)

        numerators = jnp.linalg.norm(x_preds_1 - Y_targets, axis=(1,2))        # (b_size,)
        denominators = jnp.linalg.norm(Y_targets, axis=(1,2)) + 1e-8         # avoid division by zero
        relative_errors_1 = numerators / denominators                      # (b_size,)
        recon_loss_1 = jnp.mean(relative_errors_1)

        numerators = jnp.linalg.norm(x_preds_0 - Y_targets, axis=(1,2))        # (b_size,)
        denominators = jnp.linalg.norm(Y_targets, axis=(1,2)) + 1e-8         # avoid division by zero
        relative_errors_0 = numerators / denominators                      # (b_size,)
        recon_loss_0 = jnp.mean(relative_errors_0)

        total_recon_sum_1 = total_recon_sum_1 + jnp.sum(recon_loss_1)
        total_recon_sum_0 = total_recon_sum_0 + jnp.sum(recon_loss_0)
        total_count = total_count + batch_size

        return (total_recon_sum_1, total_recon_sum_0, total_count), None

        # total_recon_loss, total_count = carry
        # total_recon_loss = total_recon_loss + recon_loss * batch_size
        # total_count = total_count + batch_size
        # return (total_recon_loss, total_count), None

    init = (0.0, 0.0, 0)

    (tot_sum_1, tot_sum_0, tot_cnt), _ = jax.lax.scan(
        body, init, jnp.arange(starts.shape[0], dtype=jnp.int32)
    )

    # mean recon over all items for both trajectories
    recon_1 = tot_sum_1 / jnp.maximum(tot_cnt, 1)
    recon_0 = tot_sum_0 / jnp.maximum(tot_cnt, 1)

    total_loss_1 = recon_1 + lambda_l1 * l1
    total_loss_0 = recon_0 + lambda_l1 * l1

    logs = {
        'total_loss_1': total_loss_1,
        'total_loss_0': total_loss_0,
        'l1'           : l1,
        'mask'         : learned_mask
    }

    return logs

    # (numer, denom), _ = jax.lax.scan(body, init, jnp.arange(starts.shape[0], dtype=jnp.int32))
    # recon_loss = numer / jnp.maximum(denom, 1)

    # total_loss = recon_loss + lambda_l1 * l1
    # logs: Dict[str, Any] = {'recon': recon_loss, 'l1': l1, 'mask': learned_mask}
    # return total_loss, logs


def make_epoch_starts(key,
                      n_cols_x,
                      batch_size,
                      pred_len,
                      n_cols_y,
                      non_overlapping=False):

    """
    Return randomized start indices s so that:
      X[:, s:s+batch_size]              is valid, and
      Y[:, s:s+batch_size+plen-1]       is valid.

    n_cols_x: number of time columns in X_train
    n_cols_y: number of time columns in Y_train (defaults to n_cols_x)
    """

    # largest allowed start so that Y window fits
    s_max = n_cols_y - (batch_size + pred_len - 1)

    if s_max < 0:
        # no valid batch can be formed
        return jnp.array([], dtype=jnp.int32)

    if non_overlapping:
        # starts: 0, B, 2B, ... <= s_max
        n_batches = s_max // batch_size + 1
        starts = jnp.arange(n_batches) * batch_size

    else:
        # all starts 0..s_max inclusive
        starts = jnp.arange(s_max + 1)

    perm = jax.random.permutation(key, starts.shape[0])

    return starts[perm]


def train_for_epochs(state,
                    X_train: jnp.ndarray,      # (Nh, k) 
                    Y_train: jnp.ndarray,      # (Nh, k) 
                    X_val: jnp.ndarray,      # (Nh, k) 
                    Y_val: jnp.ndarray,      # (Nh, k) 
                    U_l: jnp.array,
                    min_tilde : float,
                    max_tilde: float,
                    A_tilde: jnp.array,
                    c_tilde: jnp.array,
                    library_functions,
                    lib_dim: int, # l_val x len(lib)
                    pred_len: int,
                    lambda_l1: float = 0.,
                    lambda_increment = 1.,
                    batch_size: int = 128,
                    epochs: int = 50,
                    epoch_threshold: int = 5,
                    lr: float = 1e-3,
                    seed: int = 0):

    key = jax.random.PRNGKey(seed)

    N = X_train.shape[0]
    total_time_instances = X_train.shape[1]

    train_hist = []
    
    val_loss_hist_mask = []
    val_loss_hist_0_mask = []
    val_hist_mask = []

    preserved_grads = []
    best_val = float('inf')
    best_params = state.params

    for epoch in tqdm(range(1, epochs + 1)):
        key, subkey = jax.random.split(key)
        # starts = make_epoch_starts(subkey, total_time_instances, batch_size, non_overlapping = True)

        starts = make_epoch_starts(
            subkey,
            n_cols_x=X_train.shape[1],
            batch_size=batch_size,
            pred_len=pred_len,
            n_cols_y=Y_train.shape[1]
            # non_overlapping=False,
        )

        # Training
        running_loss = 0.0; seen = 0
        # counter = 0 

        for s in starts:
        # for s in range(2):

            # print(counter)
            # counter += 1

            Xb = X_train[:, s:s+batch_size] # (Nh, B)
            Yb = Y_train[:, s:s+batch_size+pred_len-1] # (Nh, B + plen - 1)
 
            # batch of bsize samples processed in each batch, each sample has the same mask vals associated with it for all pred_len.      
            state, loss, logs, grads = train_step(state, 
                                                Xb, 
                                                Yb, 
                                                lambda_l1, 
                                                A_tilde, 
                                                c_tilde, 
                                                U_l, min_tilde, max_tilde, lib_dim, batch_size, pred_len)

            running_loss += float(loss) * Xb.shape[1]
            seen += Xb.shape[1]

        # break

        train_epoch_loss = running_loss / max(1, seen)
        train_hist.append(train_epoch_loss)

        # learned_mask = logs['mask_hard'] # len = 90 # mask for the last batch of the epoch.
        # learned_mask_0 = jnp.zeros_like(learned_mask)

        dummy_x = jnp.zeros((1, U_l.shape[1]))  # (1, l)
        _, aux = state.apply_fn({'params': state.params}, dummy_x)
        learned_mask = aux['mask_hard']          # or aux['mask_soft'] if you prefer soft
        learned_mask_0 = jnp.zeros_like(learned_mask)

        # print(type(library_functions)) # <class 'tuple'>
        # return total_loss, logs
        # print(logs['mask'].shape) # 1, 10, 90

        validation_log =                    eval_step(state,
                                                library_functions,
                                                X_val,
                                                Y_val, 
                                                U_l, 
                                                min_tilde, 
                                                max_tilde, 
                                                learned_mask, 
                                                learned_mask_0, 
                                                A_tilde, 
                                                c_tilde,
                                                batch_size,
                                                pred_len)

        validation_loss_mask_1 = validation_log['total_loss_1']
        validation_loss_mask_0 = validation_log['total_loss_0']

        preserved_grads.append(grads)

        val_loss_mask = float(validation_loss_mask_1)
        val_loss_hist_mask.append(val_loss_mask)

        val_loss_0_mask = float(validation_loss_mask_0)
        val_loss_hist_0_mask.append(val_loss_0_mask)

        val_hist_mask.append(learned_mask)

        # Track best
        if val_loss_mask < best_val:
            best_val = val_loss_mask
            best_params = state.params

        # Optional: print or log mask stats

        print(f"[{epoch:03d}] train={train_epoch_loss:.6f}  val={val_loss_mask:.6f}  "
            f"val_loss_0_mask = {val_loss_0_mask:.6f}  "
            # f"mask_L1={jnp.sum(jnp.abs(aux['mask'])):.4f}  "
            f"mask_active~={int(jnp.sum(learned_mask>0.5))}/{learned_mask.shape[0]}")

        if(epoch >= epoch_threshold):
            lambda_l1 += lambda_increment

    # restore best
    state = state.replace(params=best_params)

    return state, preserved_grads, train_hist, val_loss_hist_mask, val_loss_hist_0_mask, logs
