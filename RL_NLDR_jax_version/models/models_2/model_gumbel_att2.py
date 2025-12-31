import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal
from typing import Sequence, Callable, Tuple, Any
from utils.tools_2 import apply_selected_funcs, lstsq_l2
from flax.training import train_state
import optax
from tqdm import tqdm
import numpy as np


def eval_loss_over_dataset(phi_mat, A_tilde, phi_bar_t, U_r, A_hat, X_val_t, Y_val_t, selected_idx, batch_size=128):
    N = X_val_t.shape[0]
    H_hat = phi_mat.T @ A_tilde @ phi_bar_t.T 

    total_loss = 0.0
    for start in range(0, N, batch_size):
        end = start + batch_size

        X_val_t_batch = X_val_t[start: end]
        Y_val_t_batch = Y_val_t[start: end]

        X_val_batch = X_val_t_batch.T
        Y_val_batch = Y_val_t_batch.T

        x0 = X_val_batch[:, 0]
        x_hat0 = U_r.T @ x0

        def step(xh, _):
            mod_sel = jnp.take(xh, selected_idx)
            xh = A_hat @ xh + H_hat @ mod_sel
            return (xh, xh)

        _, xh_seq = jax.lax.scan(step, x_hat0, None, length=Y_val_batch.shape[1])
        X_val_batch_rec = U_r @ xh_seq.T

        batch_loss = jnp.linalg.norm(Y_val_batch - X_val_batch_rec)/jnp.linalg.norm(Y_val_batch)

        total_loss += float(batch_loss) * (end - start)
    
    return total_loss / N


class TrainState(train_state.TrainState):
    pass

def build_library(X_hat: jnp.ndarray,
                  funcs: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]) -> jnp.ndarray:
    """
    X_hat: (batch, r)          # reduced variables
    funcs: list of callables f: (batch,r)->(batch,r)
    returns: (batch, r * L)    # concatenated along feature dim
    """
    outs = [f(X_hat) for f in funcs]            # each (b,r)
    return jnp.concatenate(outs, axis=1)        # (b, rL)

class STGSelector(nn.Module):
    in_dim:  int
    out_dim: int
    sigma_init: float = 0.5
    l0_lambda: float = 1e-4

    @nn.compact
    def __call__(self,
                 X_mod: jnp.ndarray,
                 *,
                 train: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns: (X_sel, gates, hard_idx, l0_pen)
        gates: (out_dim, in_dim) in [0,1]
        """
        mu    = self.param('mu',    glorot_normal(), (self.out_dim, self.in_dim))
        sigma = self.param('sigma', lambda k,s: jnp.full(s, self.sigma_init),
                           (self.out_dim, self.in_dim))

        key = self.make_rng('noise')
        eps = jax.random.normal(key, mu.shape) * sigma
        z   = jnp.clip(mu + eps, 0.0, 1.0) if train else jnp.clip(mu, 0.0, 1.0)

        X_sel = X_mod @ z.T   # (batch, out_dim)
        hard_idx = jnp.argmax(mu, axis=-1)

        # Expected L0 penalty: sum Phi(mu/sigma)
        cdf = 0.5 * (1.0 + jax.scipy.special.erf(mu / (sigma * jnp.sqrt(2.0))))
        exp_l0 = jnp.sum(cdf)
        l0_pen = self.l0_lambda * exp_l0

        return X_sel, z, hard_idx, l0_pen


class GumbelSelector(nn.Module):
    in_dim:  int      # r x L  (size of library)
    out_dim: int      # p   (how many features you want to KEEP)
    start_temp: float = 10.0
    min_temp:   float = 0.1
    alpha:      float = 0.9999   # decay per step
    repulsion_coefficient: float = 1e-1

    @nn.compact
    def __call__(self,
                 X_mod_t_batch: jnp.ndarray,
                 temp:  jnp.ndarray,
                 train: bool = True):

        """
        X_mod: (batch, r x L)
        temp : scalar temperature (will be updated/returned)
        Returns: (X_sel, new_temp, hard_idx, l0_pen)
                 X_sel: (batch, out_dim)
        """

        logits = self.param('logits', glorot_normal(), (self.out_dim, self.in_dim))

        # anneal temperature
        new_temp = jnp.maximum(self.min_temp, temp * self.alpha) if train else self.min_temp

        # Draw Gumbel noise
        gkey = self.make_rng('gumbel')
        u = jax.random.uniform(gkey, logits.shape, minval=1e-7, maxval=1.0)
        g = -jnp.log(-jnp.log(u))

        # relaxed categorical over features, per output slot
        noisy = (logits + g) / new_temp
        soft_samples = jax.nn.softmax(noisy, axis=-1)  # (out_dim, in_dim)

        # selection: (batch, rL) · (in_dim -> out_dim)^T
        X_sel_t_batch = X_mod_t_batch @ soft_samples.T  # (bsize, p_val)

        # indices (hard) just for logging:
        hard_idx = jnp.argmax(logits, axis=-1) # (p_val)

        # optional: expected L0 ≈ mean max prob per row (cheap proxy)
        probs = jax.nn.softmax(logits, axis=-1)
        exp_l0 = jnp.sum(jnp.max(probs, axis=-1))
        l0_pen = self.repulsion_coefficient * exp_l0
        return X_sel_t_batch, new_temp, hard_idx, l0_pen


# X_pred_batch, reconstr_err = reconstruction_error(X_train_t_batch, Y_train_t_batch, 
#                                                 self.phi_bar_t, self.U_l, selected_idx)

def reconstruction_error(X_train_t_batch, Y_train_t_batch, 
                        phi_bar_t, U_l, selected_idx):
    
    X_train_batch = X_train_t_batch.T
    Y_train_batch = Y_train_t_batch.T

    x0 = X_train_batch[:, 0]
    x_tilde0 = U_l.T @ x0

    phi_bar = phi_bar_t.T

    def step(xh, _):
        mod_sel = jnp.take(xh, selected_idx)
        xh = A_tilde @ xh + phi_bar @ mod_sel
        return (xh, xh)

    _, xh_seq = jax.lax.scan(step, x_hat0, None, length=Y_train_batch.shape[1])
    X_rec = U_r @ xh_seq.T

    return X_rec, jnp.linalg.norm(Y_train_batch - X_rec)/jnp.linalg.norm(Y_train_batch)


class ROMModel(nn.Module):
    A_tilde: jnp.array
    library_functions: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]
    l_val: int
    p_val: int
    U_l: jnp.array
    min_temp:   float = 0.01
    start_temp: float = 10.0
    alpha_const: float = 0.999
    selector_type: str = 'gumbel'  # or 'stg'
    repulsion_coefficient: float = 1e-1

    def setup(self):
        if self.selector_type == 'gumbel':
            self.selector = GumbelSelector(
                in_dim=self.l_val * len(self.library_functions),
                out_dim=self.p_val,
                start_temp=self.start_temp,
                min_temp=self.min_temp,
                alpha=self.alpha_const,
                repulsion_coefficient = self.repulsion_coefficient
            )

        # else:
        #     self.selector = STGSelector(
        #         in_dim=self.r * len(self.lib_funcs),
        #         out_dim=self.p,
        #         sigma_init=self.sigma_init,
        #         l0_lambda=self.l0_lambda
        #     )

        self.phi_bar_t = self.param('phi_bar_t',
                                glorot_normal(),
                                (self.p_val, self.l_val))


    def __call__(self, X_tilde_t_batch, X_train_t_batch, Y_train_t_batch, temp):
        
        X_mod_t_batch = build_library(X_tilde_t_batch, self.library_functions)  # (bsize, r_val x len(lib))

        if self.selector_type == 'gumbel':
            X_tilde_nl_t_batch, new_temp, selected_idx, l0_pen = self.selector(X_mod_t_batch, temp)
        # else:
        #     X_hat_nl_t_batch, gates, selected_idx, l0_pen = self.selector(X_mod_t_batch)
        #     new_temp = temp  # unused for STG

        # M-1 
        # H_hat = self.phi_mat.T @ self.A_tilde @ self.phi_bar_t.T 
        X_pred_batch, reconstr_err = reconstruction_error(X_train_t_batch, Y_train_t_batch, 
                                            self.phi_bar_t, self.U_l, selected_idx)

        # M-2
        # Lin_approx = self.phi_mat @ X_hat_t_batch.T
        # nl_approx =  self.phi_bar_t.T @ X_hat_nl_t_batch.T
        # X_pred = Lin_approx + nl_approx
        # reconstr_err = jnp.linalg.norm(X_pred - Y_train_t_batch.T)/ jnp.linalg.norm(Y_train_t_batch.T) 

        return X_pred_batch.T, reconstr_err, new_temp, selected_idx


def create_train_state(rng, model, batch_size, r_val, Nh, lr):
    rng_p, rng_g, rng_d = jax.random.split(rng, 3)

    dummy_X_hat_t_batch = jnp.zeros((batch_size, r_val)) # (bsize, r_val)
    dummy_X_train_t_batch = jnp.zeros((batch_size, Nh)) # (bsize, Nh)
    dummy_Y_train_t_batch = jnp.zeros((batch_size, Nh)) # (bsize, Nh)
    dummy_temp = 1.0

    init_vars = model.init({'params': rng_p, 'gumbel': rng_g, 'dropout': rng_d},
                       dummy_X_hat_t_batch, dummy_X_train_t_batch, dummy_Y_train_t_batch, dummy_temp)

    params = init_vars['params']

    tx = optax.adam(lr)
    # tx = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-07)

    return train_state.TrainState.create(apply_fn=model.apply,
                             params= params,
                             tx=tx)


@jax.jit
def train_step(state, X_hat_t_batch, X_train_t_batch, Y_train_t_batch, temperature, rngs):
    def loss_fn(params):
        X_pred_t_batch, batch_loss, new_temperature, selected_idx = state.apply_fn({'params': params},
                                                                    X_hat_t_batch,
                                                                    X_train_t_batch,
                                                                    Y_train_t_batch,
                                                                    temperature,
                                                                    rngs=rngs)

        aux = (X_pred_t_batch, selected_idx, new_temperature)
        return (batch_loss, aux)

    (batch_loss, (X_pred_t_batch, selected_idx, new_t)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.isfinite(batch_loss)
    grads = jax.tree.map(lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads)

    # grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(loss), g, jnp.zeros_like(g)), grads)
    # new_state = state.apply_gradients(grads=grads)
    # new_state = new_state.replace(temp=new_t)
    # return new_state, loss, X_pred

    state = state.apply_gradients(grads=grads)

    return (state, X_pred_t_batch, batch_loss, new_t )



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



def train_for_epochs(state, phi_mat, A_tilde, U_r, A_hat, X_hat, X_train, Y_train, X_val, Y_val, num_epochs, start_temp, batch_size, rng, initial_lr, threshold):
    """Runs the inner training loop and returns updated state + final loss history."""

    X_hat_t =  X_hat.T

    X_train_t =  X_train.T
    Y_train_t =  Y_train.T

    X_val_t =  X_val.T
    Y_val_t =  Y_val.T

    n_timesteps = X_train_t.shape[0] # (k_train)

    train_loss_history = []
    val_loss_history = []

    mean_max_prob = []
    logit_vals_hist = []

    temperature = start_temp

    current_lr = initial_lr

    best_val_loss = float('inf')

    best_overall_train_loss, best_overall_val_loss = (float('inf'), float('inf'))

    stall_count = 0
    patience = 100
    best_params = state.params
    lr_decay = 0.5
    stop_training = False

    for epoch in tqdm(range(1, num_epochs + 1)):
        rng, subkey = jax.random.split(rng)
        starts = make_epoch_starts(subkey, X_train_t.shape[0], batch_size, non_overlapping=True)
        epoch_loss = 0.0

        for start in starts:
            X_hat_t_batch = X_hat_t[start:start + batch_size]
            X_train_t_batch = X_train_t[start:start + batch_size]
            Y_train_t_batch = Y_train_t[start:start + batch_size]
        

            rng, rng_g, rng_d = jax.random.split(rng, 3)
            # def train_step(state, X_hat_t_batch, X_train_t_batch, Y_train_t_batch, temperature, rngs):
            # return (state, X_pred_t_batch, batch_loss, new_t )

            state, X_pred_t_batch, batch_loss, temperature = train_step(state, X_hat_t_batch, X_train_t_batch, Y_train_t_batch, temperature, rngs={'gumbel': rng_g, 'dropout': rng_d})
            logits_enc = state.params['selector']['logits']
            mean_max = jnp.mean(jnp.max(jax.nn.softmax(logits_enc, axis=-1), axis=-1))
            mean_max_scalar = mean_max.block_until_ready().item()
            mean_max_prob.append(mean_max_scalar)

            # logits_enc2 = state.params['encoder2']['logits']
            # mean_max2 = jnp.mean(jnp.max(jax.nn.softmax(logits_enc2, axis=-1), axis=-1))
            # mean_max_scalar2 = mean_max2.block_until_ready().item()
            # mean_max_prob2.append(mean_max_scalar2)

            # epoch_loss += batch_loss.item() * S_train_t_batch.shape[0]
            epoch_loss += batch_loss.item() * X_train_t_batch.shape[0]

        train_loss_history.append(epoch_loss / n_timesteps)

        logits_cpu = np.array(logits_enc)
        logit_vals_hist.append(logits_cpu)
        selected_indices = jnp.argmax(logits_cpu, axis=1)

        phi_bar_t = state.params['phi_bar_t']

        # print(type(phi_bar_t))
        # print(phi_bar_t.shape) # (30, 24)

        # val_loss = eval_loss(dff_params, decoder_params, val_dff_input, S_val_t)
        # def eval_loss_over_dataset(phi_mat, A_tilde, phi_bar_t, U_r, A_hat, X_val_t, Y_val_t, selected_idx, batch_size=128):

        val_loss_cpu = eval_loss_over_dataset(phi_mat, 
                                              A_tilde,
                                              phi_bar_t,
                                              U_r,
                                              A_hat,
                                              X_val_t, 
                                              Y_val_t, 
                                              selected_indices,
                                              batch_size=batch_size)

        val_loss_history.append(val_loss_cpu)

        if val_loss_cpu < best_val_loss:
            best_val_loss = val_loss_cpu
            stall_count = 0
            best_params = state.params
            best_overall_train_loss = train_loss_history[-1]
            best_overall_val_loss = val_loss_history[-1]
        else:
            stall_count += 1

        if mean_max >= threshold:
            stop_training = True
        if stop_training:
            print(f'Stopping at epoch {epoch}, mean_max={mean_max:.3f}, best_val_loss={best_val_loss:.4f}')
            break
    
    state = state.replace(params=best_params)

    return (state, temperature, train_loss_history, val_loss_history, mean_max_prob, logit_vals_hist, best_overall_train_loss, best_overall_val_loss)