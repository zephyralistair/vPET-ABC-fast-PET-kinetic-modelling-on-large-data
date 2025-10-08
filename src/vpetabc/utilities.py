from __future__ import annotations

import jax
import jax.numpy as jnp

import pandas as pd

# ---------------------------------------------------------------------
# Preprocessing function
# ---------------------------------------------------------------------
def preprocess_table(df: pd.DataFrame,
                     dt_fine: float = 0.5
                    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    • df: rows exactly as specified above
    • returns
        Cp_fine   : (T_fine,)  – Cp sampled on the uniform fine grid
        A         : (F, T_fine) – frame-averaging sparse matrix (0/1 divided by n)
        tac_voxels: (V, F)     – all voxel TACs stacked
    """
    t_mid   = df.iloc[0].to_numpy(float)          # (F,)
    f_len   = df.iloc[1].to_numpy(float)          # (F,)
    Cp_raw  = df.iloc[2].to_numpy(float)          # (F,)
    TACs    = df.iloc[3:].to_numpy(float)         # (V, F)

    # Fine grid from t=0 to end of last frame
    t_end   = t_mid[-1] + 0.5 * f_len[-1]
    t_fine  = jnp.arange(0.0, t_end + dt_fine, dt_fine)   # (T_fine,)

    # Interpolate Cp onto fine grid (linear)
    Cp_fine = jnp.interp(t_fine, t_mid, Cp_raw)

    # Build sparse frame-averaging matrix A so that
    # Ct_frame = A @ Ct_fine
    starts = t_mid - 0.5 * f_len
    ends   = t_mid + 0.5 * f_len

    def build_row(t0, t1):
        mask = (t_fine >= t0) & (t_fine < t1)
        row  = mask.astype(float)
        # if no fine-grid samples fall inside the frame, fall back to
        # the closest sample to the mid-frame time
        row_sum = row.sum()
        return jax.lax.cond(
            row_sum == 0,
            lambda _: jax.nn.one_hot(jnp.argmin(jnp.abs(t_fine - 0.5*(t0+t1))),
                                    t_fine.size),
            lambda _: row / row_sum,
            operand=None
        )

    A = jax.vmap(build_row)(starts, ends)        # (F, T_fine)

    return Cp_fine, A, jnp.asarray(TACs), t_fine


# ---------------------------------------------------------------------
# Postprocessing function
# ---------------------------------------------------------------------
@jax.jit
def get_conditional_posterior_mean(arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the conditional posterior mean on the preferred model
    Returns conditional mean and the target model (0/1) for each voxel.
    """
    # arr shape (V, k, P)  - voxel x samples x params

    last = arr[..., -1] ## M column, shape (V, k)

    # --- 2. decide, per slice, which value we want ------------------------
    # mean(last) > 0.5  → look for 1s
    # mean(last) ≤ 0.5  → look for 0s
    target = (last.mean(axis=1) > 0.5).astype(arr.dtype)   # 0 or 1, shape (V,)

    # --- 3. build a broadcastable mask ------------------------------------
    mask = (last == target[:, None])[..., None]            # shape (V, k, 1)

    # --- 4. compute the mean of the selected rows -------------------------
    num = (arr * mask).sum(axis=1)                        # (V, P)
    den = mask.sum(axis=1).astype(arr.dtype)              # (V, 1)
    ret = num / den                                       # (V, P)
    return ret, target