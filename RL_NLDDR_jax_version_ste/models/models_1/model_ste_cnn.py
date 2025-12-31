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
    out_dim: int
    hidden: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.out_dim)(x)

class ConcreteSelector(nn.Module):
    in_out_dim: int
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
        
        return feats_masked, mask.squeeze() # (B, l x len(lib)), (l x len(lib))

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
                       sample_input: jnp.ndarray,
                       lr: float = 1e-3) -> TrainState:
    
    # params = model.init(key, sample_input)['params']
    # tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

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

        feats_masked, mask = self.concrete_selector(feats) # x_nl

        # jax.debug.print("{}\n____\n{}", feats, feats_masked)
        # jax.debug.print("{}\n____\n{}\n_____\n{}", feats.shape, mask.shape, feats_masked.shape)
        # jax.debug.print("{}\n____\n{}\n_____\n{}", feats, mask, feats_masked)

        # mod = apply_selected_funcs(xh, library_functions)      # (m,)
        # mod_sel = mod * mask                                   # (m,)
        # nonlinear_term = state.apply_fn({'params': state.params}, mod_sel, method=STEFeatureSelectorModel.decode)

        preds = self.decoder(feats_masked) # \phi(x_nl)

        feats_0 = jnp.zeros((mask.shape),dtype = jnp.int32)
        preds_0_mask = jax.lax.stop_gradient(self.decoder(feats_0))

        # jax.debug.print("{}\n____\n{}\n____\n{}\n_____\n{}\n______\n{}", X_tilde_norm_batch_t, feats, mask, feats_masked, preds)
        # print(type(feats), type(feats_masked), type(mask), type(preds))
        # print(feats.dtype, feats_masked.dtype, mask.dtype, preds.dtype)

        aux = {'mask': mask, 'feats': feats, 'feats_masked': feats_masked, 'preds_0_mask': preds_0_mask}

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


@jax.jit
def train_step(state: TrainState,
               X_batch: jnp.ndarray, # (B, l)
               lhs_batch: jnp.ndarray, # (B, l)
               lambda_l1: float):

    def loss_fn(params):
        preds, aux = state.apply_fn({'params': params}, X_batch)
        recon = mse_loss(preds, lhs_batch)          # row-wise regression loss
        l1 = jnp.sum(jnp.abs(aux['mask']))         # encourages sparsity
        loss = recon + lambda_l1 * l1
        aux_vals = {'recon': recon, 'l1': l1, 'mask': aux['mask']}
        return (loss, aux_vals)

    (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    jax.debug.print("{}", grads)

    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
    
    state = state.apply_gradients(grads=grads)

    return state, loss, logs, grads


# @jax.jit(static_argnames=('model'))
def eval_step(state: TrainState,
            library_functions,
            X_val: jnp.ndarray,
            Y_val: jnp.ndarray,
            U_l,
            min_tilde,
            max_tilde,
            learned_mask,
            learned_mask_0,
            A_tilde,
            c_tilde):

    # rolling forecast:
    x_init = X_val[:, 0]
    x_init_tilde = U_l.T @ x_init
    X_init_tilde_norm = (x_init_tilde - min_tilde)/(max_tilde - min_tilde)

    def make_step(mask):
        def step(xh, _):
            mod = apply_selected_funcs(xh, library_functions)      # (m,) # x_mod
            mod_sel = mod * mask                                   # (m,) # x_nl
            
            # jax.debug.print("{}\n________\n{}", mod, mod_sel)

            nonlinear_term = state.apply_fn({'params': state.params}, mod_sel, method=STEFeatureSelectorModel.decode)

            xh = A_tilde @ xh + c_tilde + nonlinear_term # (r,)
            return xh, xh
        return step

    # run once per mask
    _, xh_seq_mask1 = jax.lax.scan(make_step(learned_mask),  X_init_tilde_norm, None, length=Y_val.shape[1] + 1)
    _, xh_seq_mask0 = jax.lax.scan(make_step(learned_mask_0),X_init_tilde_norm, None, length=Y_val.shape[1] + 1)

    x_tilde_predict_norm = xh_seq_mask1.T
    x_tilde_predict = (max_tilde - min_tilde) * x_tilde_predict_norm + min_tilde
    x_predict = U_l @ x_tilde_predict

    reconstr_loss = jnp.linalg.norm(x_predict[:,1:] -  Y_val)/jnp.linalg.norm(Y_val)


    x_tilde_predict_norm = xh_seq_mask0.T
    x_tilde_predict = (max_tilde - min_tilde) * x_tilde_predict_norm + min_tilde
    x_predict = U_l @ x_tilde_predict

    reconstr_loss_0_mask = jnp.linalg.norm(x_predict[:,1:] -  Y_val)/jnp.linalg.norm(Y_val)

    return reconstr_loss, reconstr_loss_0_mask

# --------------------------------------------------------------------------------------
# 6) Epoch loop helpers
# --------------------------------------------------------------------------------------
def make_epoch_starts(key, n_rows, batch_size, non_overlapping=True):
    if non_overlapping:
        n_batches = n_rows // batch_size
        starts = jnp.arange(n_batches) * batch_size
    else:
        starts = jnp.arange(max(1, n_rows - batch_size + 1))
    perm = jax.random.permutation(key, starts.shape[0])
    return starts[perm]


def train_for_epochs(state,
                    model,
                    X_train_tilde_norm_t: jnp.ndarray,      # (N, l) 
                    lhs_train_tilde_norm_t: jnp.ndarray,          # (N, l)
                    X_val: jnp.ndarray,        # (N_val, r)
                    Y_val: jnp.ndarray,            # (N_val, l)
                    U_l: jnp.array,
                    min_tilde : float,
                    max_tilde: float,
                    A_tilde: jnp.array,
                    c_tilde: jnp.array,
                    library_functions: Callable[[jnp.ndarray], jnp.ndarray],
                    lib_dim: int, # l x len(lib)
                    out_dim: int, # l 
                    lambda_l1: float = 0.,
                    lambda_increment = 1.,
                    batch_size: int = 128,
                    epochs: int = 50,
                    epoch_threshold: int = 5,
                    lr: float = 1e-3,
                    seed: int = 0):

    key = jax.random.PRNGKey(seed)

    N = X_train_tilde_norm_t.shape[0]
    train_hist, val_hist, val_hist_0_mask = [], [], []
    preserved_grads = []
    best_val = float('inf'); best_params = state.params

    for epoch in tqdm(range(1, epochs + 1)):
        key, subkey = jax.random.split(key)
        starts = make_epoch_starts(subkey, N, batch_size, non_overlapping=True)

        # Training
        running_loss = 0.0; seen = 0
        for s in starts:
            Xb = X_train_tilde_norm_t[s:s+batch_size] # (B, l)
            Lb = lhs_train_tilde_norm_t[s:s+batch_size] # (B, l)
            state, loss, logs, grads = train_step(state, Xb, Lb, lambda_l1)
            
            # {'recon': recon, 'l1': l1, 'mask': aux['mask']}

            running_loss += float(loss) * Xb.shape[0]
            seen += Xb.shape[0]

        train_epoch_loss = running_loss / max(1, seen)
        train_hist.append(train_epoch_loss)

        learned_mask = logs['mask']

        learned_mask_0 = jnp.zeros_like(learned_mask)

        # print(learned_mask,'\n_____\n', learned_mask_0)
        preserved_grads.append(grads)
        val_loss, val_loss_0_mask = eval_step(state,
                                            library_functions,
                                            X_val, 
                                            Y_val, 
                                            U_l, 
                                            min_tilde, 
                                            max_tilde, 
                                            learned_mask, 
                                            learned_mask_0, 
                                            A_tilde, 
                                            c_tilde)

        val_loss = float(val_loss)
        val_hist.append(val_loss)

        val_loss_0_mask = float(val_loss_0_mask)
        val_hist_0_mask.append(val_loss_0_mask)
        
        # Track best
        if val_loss < best_val:
            best_val = val_loss
            best_params = state.params

        # Optional: print or log mask stats

        print(f"[{epoch:03d}] train={train_epoch_loss:.6f}  val={val_loss:.6f}  "
            f"val_loss_0_mask = {val_loss_0_mask:.6f}  "
            # f"mask_L1={jnp.sum(jnp.abs(aux['mask'])):.4f}  "
            f"mask_active~={int(jnp.sum(learned_mask>0.5))}/{learned_mask.shape[0]}")

        if(epoch >= epoch_threshold):
            lambda_l1 += lambda_increment

    # restore best
    state = state.replace(params=best_params)

    return state, preserved_grads, train_hist, val_hist, val_hist_0_mask, logs
