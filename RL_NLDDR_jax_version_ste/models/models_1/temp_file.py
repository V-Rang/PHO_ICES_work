@partial(jax.jit, static_argnames=['library_functions', 'b_size', 'pred_len'])
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
               library_functions: jnp.ndarray,
               lib_dim: int,
               b_size: int,
               pred_len: int
               ):

    def loss_fn(params):

        X_tilde = U_l.T @ X_batch                # (l_val, b_size)
        # X_tilde_norm = (X_tilde - min_tilde[:, None]) / (max_tilde[:, None] - min_tilde[:, None])  # (l_val, b_size)
        X_tilde_norm = (X_tilde - min_tilde) / (max_tilde - min_tilde)  # (l_val, b_size)

        X_tilde_norm = X_tilde_norm.T            # (b_size, l_val) for vmap
    
        def single_rollout(x_init_tilde_norm):

            def step(xh, _):
                phi, _ = state.apply_fn({'params': params}, xh.reshape(1, -1))
                phi = phi.squeeze()
                xh = A_tilde @ xh + c_tilde + phi
                return xh, xh

            _, x_seq = jax.lax.scan(step, x_init_tilde_norm, None, length= pred_len + 1)
            
            return x_seq  # (pred_len + 1, l_val)

        x_seqs = jax.vmap(single_rollout)(X_tilde_norm)   # (b_size, pred_len+1, l_val)
        x_seqs = x_seqs[:, 1:, :]                  # Drop first step, (b_size, pred_len, l_val)

        x_seqs = jnp.transpose(x_seqs, (0, 2, 1))   # (b_size, l_val, pred_len)
        # x_seqs = (max_tilde[:, None] - min_tilde[:, None]) * x_seqs + min_tilde[:, None]  # unnorm
        x_seqs = (max_tilde - min_tilde) * x_seqs + min_tilde  # unnorm

        x_preds = jax.vmap(lambda x: U_l @ x)(x_seqs)  # (b_size, Nh, pred_len)

        # Y_targets = jnp.stack([Y_batch[:, i : i + pred_len] for i in range(X_batch.shape[1])])  # (b_size, Nh, pred_len)
        Y_targets = jnp.stack([Y_batch[:, i : i + pred_len] for i in range(X_batch.shape[1])])  # (b_size, Nh, pred_len)

        numerators = jnp.linalg.norm(x_preds - Y_targets, axis=(1,2))        # (b_size,)
        denominators = jnp.linalg.norm(Y_targets, axis=(1,2)) + 1e-8         # avoid division by zero
        relative_errors = numerators / denominators                      # (b_size,)
        recon_loss = jnp.mean(relative_errors)

        # Estimate L1 penalty (optional)
        mask = state.apply_fn({'params': params}, X_tilde_norm[0].reshape(1, -1))[1]['mask']  # use 1st example
        l1 = jnp.sum(jnp.abs(mask))

        return recon_loss + lambda_l1 * l1, {'recon': recon_loss, 'l1': l1, 'mask': mask}

    (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss, logs, grads

