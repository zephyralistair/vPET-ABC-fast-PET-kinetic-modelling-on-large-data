# abc_kinetic.py
from __future__ import annotations
from abc import ABC, abstractmethod ## abstract base class
from functools import partial
from typing import Callable, Tuple, Protocol
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import jax.random as jr

import pandas as pd


# ---------------------------------------------------------------------
# 1.  Generic kinetic-model interface
# ---------------------------------------------------------------------
class KineticModel(ABC):
    """Abstract base: any forward model maps params → predicted TAC."""

    @abstractmethod
    def simulate(self,
                 params: jnp.ndarray,
                 Cp:     jnp.ndarray,
                 dt:     float) -> jnp.ndarray:
        """Forward model for a *single* parameter vector."""

    # Batch version is shared and fully vectorised
    @partial(jax.jit, static_argnums=(0,))
    def batch_simulate(self,
                       params: jnp.ndarray,
                       Cp_fine: jnp.ndarray,
                       dt: float,
                       A: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        returns
            (N, T_fine) if A is None
            (N, F)      if A is provided
        """
        Ct_fine = jax.vmap(lambda θ: self.simulate(θ, Cp_fine, dt))(params)
        return Ct_fine if A is None else jnp.matmul(Ct_fine, A.T)   # note transpose


# ---------------------------------------------------------------------
# 2.  Compartment model
# ---------------------------------------------------------------------
class TwoTissueModel(KineticModel):
    """
    5-parameter irreversible/reversible 2-TCM
        dC₁/dt = K₁·Cp  − (k₂+k₃)·C₁ + k₄·C₂
        dC₂/dt = k₃·C₁ − k₄·C₂
        C_T(t) = C₁ + C₂ + v_b·Cp
    Discretised with 4th order explicit Runge-Kutta.  Pure JAX → XLA.
    """

    param_names = ('K1', 'k2', 'k3', 'k4', 'Vb', 'M')

    @partial(jax.jit, static_argnums=(0,))
    def simulate(self,
                 params: jnp.ndarray,
                 Cp:     jnp.ndarray,
                 dt:     float) -> jnp.ndarray:

        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        K1, k2, k3, k4, Vb, M = params

        def deriv(state, Cp_t):
            C1, C2 = state
            dC1 = K1 * Cp_t - (k2 + k3) * C1 + k4 * C2
            dC2 = k3 * C1 - k4 * C2
            return jnp.array([dC1, dC2])
                     
        def step(state, Cp_t):
            C1, C2 = state
            k1 = deriv(state, Cp_t)
            k2 = deriv(state + 0.5 * dt * k1, Cp_t)
            k3 = deriv(state + 0.5 * dt * k2, Cp_t)
            k4 = deriv(state + dt * k3, Cp_t)
            new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            Ct = (1 - Vb) * (new_state[0] + new_state[1]) + Vb * Cp_t
            return new_state, Ct

        init_state = jnp.array([0.0, 0.0])
        _, Ct = jax.lax.scan(step, init_state, Cp)
        return Ct
    
# ---------------------------------------------------------------------
# 7-parameter activated / null neurotransmitter model
# ---------------------------------------------------------------------
class lpntPETModel(KineticModel):
    """
    7-parameter lp-ntPET model¹

        dC_t/dt = R₁ · dC_r/dt  +  k₂ · C_r  −  k₂a · C_t  −  γ · C_t · h(t)

    with the activation kernel

        h(t) = τ(t)^α · exp[ α · (1 − τ(t)) ] · u(t − t_D)
        τ(t) = (t − t_D) / (t_P − t_D)

    Parameters
    ----------
    R₁, k₂, k₂a, γ, t_D, t_P, α, M

    All maths is done on a uniform grid of step dt (explicit Euler).
    The reference-region TAC C_r(t) is passed in via `Cp` for API symmetry.
    """

    param_names = ('R1', 'k2', 'k2a', 'gamma', 'tD', 'tP', 'alpha', 'M')

    @partial(jax.jit, static_argnums=(0,))
    def simulate(self,
                 params: jnp.ndarray,
                 Cr:     jnp.ndarray,      # reference region TAC
                 dt:     float) -> jnp.ndarray:
        
        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        R1, k2, k2a, gamma, tD, tP, alpha, M = params

        # ------------------------ pre-compute helpers -----------------
        T = Cr.size
        t = jnp.arange(T, dtype=Cr.dtype) * dt           # (T,)

        # dCr/dt with forward finite difference (0 for first sample)
        dCr = jnp.concatenate([jnp.zeros((1,), Cr.dtype),
                               jnp.diff(Cr) / dt])       # (T,)

        # safe denominator for τ(t)
        denom = jnp.where(jnp.abs(tP - tD) < 1e-6,
                          1e-6,                          # avoid 0
                          tP - tD)

        tau   = (t - tD) / denom                         # (T,)
        u     = (t >= tD).astype(Cr.dtype)               # Heaviside
        tau   = jnp.where(u, tau, 0)
        h     = u * (tau ** alpha) * jnp.exp(alpha * (1 - tau))
        # bundle per-time inputs for lax.scan
        inputs = jnp.stack([dCr, Cr, h], axis=1)         # (T, 3)

        # ------------------------ Explicit Euler ------------------
        def step(Ct_prev, inp):
            dCr_t, Cr_t, h_t = inp
            dCt = R1 * dCr_t + k2 * Cr_t - k2a * Ct_prev - gamma * Ct_prev * h_t
            Ct  = Ct_prev + dCt * dt
            return Ct, Ct

        Ct0   = 0.0
        _, Ct = jax.lax.scan(step, Ct0, inputs)          # (T,)

        return Ct


# ---------------------------------------------------------------------
# 3.  Vectorised, JIT-compiled rejection-ABC engine
# ---------------------------------------------------------------------
class ABCRejection:
    """
    Rejection ABC with a *fixed* number of simulations (N) and an
    acceptance fraction ε (keep ε·N best simulations).
    """

    def __init__(self,
                 model:          KineticModel,
                 prior_sampler:  Callable[[jr.KeyArray, int], jnp.ndarray],
                 lower_bounds:   jnp.ndarray | None = None,
                 upper_bounds:   jnp.ndarray | None = None,
                 distance_fn:    Callable[[jnp.ndarray, jnp.ndarray], float] | None = None,
                 num_sims:       int = 100_000,
                 accept_frac:    float = 0.01):
        self.model         = model
        self.prior_sampler = prior_sampler
        self.lower_bounds  = lower_bounds
        self.upper_bounds  = upper_bounds
        self.num_sims      = num_sims
        self.accept_frac   = accept_frac
        self.distance_fn   = (distance_fn if distance_fn is not None 
                              else lambda x, y: jnp.sum(
                                  jnp.abs(x[:, None, :] - y[None, :, :]), 
                                  axis=-1))

    def run(self, key, obs, Cp_fine, A, *,
            dt=0.5, batch_size=50_000, progress=True):
        V = obs.shape[0] # number of voxels
        k = max(1, int(self.num_sims * self.accept_frac)) # number of accepted simulations
        P = self.prior_sampler(key, 1).shape[-1] # number of parameters
        # n: number of simulations per batch
        # F: number of frames in the fine grid

        best_d  = jnp.full((k, V), jnp.inf)
        best_th = jnp.empty((k, P, V))

    # -----------------------------------------------------------------
    # JIT-compiled
    # -----------------------------------------------------------------
        @partial(jax.jit, static_argnums=(0, 2))
        def _chunk(self, key, n):
            θ   = self.prior_sampler(key, n, self.lower_bounds, self.upper_bounds) # (n,P)
            Ct  = self.model.batch_simulate(θ, Cp_fine, dt, A)    # (n,F)
            d   = self.distance_fn(Ct, obs)                       # (n,V)
            return θ, d

        def merge(best_d, best_th, d, θ):
            n, P = θ.shape
            V    = best_d.shape[1]
            θ_full = jnp.broadcast_to(θ[..., None], (n, P, V))     # (n,P,V)
            cat_d  = jnp.concatenate([best_d, d], axis=0)          # (k+n,V)
            cat_th = jnp.concatenate([best_th, θ_full], axis=0)
            idx    = jnp.argsort(cat_d, axis=0)[:k]                # (k,V)
            new_d  = jnp.take_along_axis(cat_d, idx, axis=0)
            new_th = jnp.take_along_axis(cat_th, idx[:, None, ...], axis=0)
            return new_d, new_th                                   # (k,V),(k,P,V)

        # --------------- main loop ------------------------------------
        remaining = self.num_sims
        progress_bar = tqdm(total=remaining, desc="Running ABC", disable=not progress)
        while remaining:
            n = min(batch_size, remaining)
            key, sub = jr.split(key)
            θ_b, d_b = _chunk(self, sub, n)          # device-resident (n,P), (n,V)
            best_d, best_th = merge(best_d, best_th, d_b, θ_b)
            remaining -= n
            progress_bar.update(n)

        return jnp.transpose(best_th, (2, 0, 1)) # (V, k, P) - voxel x samples x params


# ---------------------------------------------------------------------
# 4.  Preprocessing function
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
# 5.  Postprocessing function
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
    ret = (arr * mask).mean(axis=1)                        # shape (V, P)
    return ret, target

# ---------------------------------------------------------------------
# 6.  Priors
# ---------------------------------------------------------------------
def TwoTissuePrior(
        key: jr.KeyArray, 
        n: int,
        lows: jnp.ndarray = jnp.array([0., 0., 0., 0, 0.]),
        highs: jnp.ndarray = jnp.array([1., 1., 0.2, 0.2, 0.1])
        ) -> jnp.ndarray:
    """
    Draws n samples of length-6:
        [K1, k2, k3, k4, Vb, M]
    where M ~ Bernoulli(0.5) (0/1).
    Whenever M == 0, k4 is forced to 0. (irreversible model)

    Returns
    -------
    params : (n, 6)  jnp.ndarray
    """

    key, sub_uniform = jr.split(key)
    base = jr.uniform(sub_uniform, shape=(n, 5), minval=lows, maxval=highs)

    # ─── Bernoulli indicator ─────────────────────────────────────────────
    key, sub_bern = jr.split(key)
    ind = jr.bernoulli(sub_bern, p=0.5, shape=(n, 1)).astype(base.dtype)  # (n, 1)

    # If M == 0  →  force k4 (column 3) to 0
    k4 = jnp.where(ind.squeeze() == 0, 0.0, base[:, 3])
    base = base.at[:, 3].set(k4)

    # ─── concatenate and return ─────────────────────────────────────────
    params = jnp.concatenate([base, ind], axis=1)        # (n, 6)
    return params

def lpntPETPrior(
        key: jr.KeyArray, 
        n: int,
        lows: jnp.ndarray = jnp.array([0., 0., 0., 0., 0., 0., 0.]),
        highs: jnp.ndarray = jnp.array([2., 1., 0.4, 0.4, 25., 45., 4.])
        ) -> jnp.ndarray:
    """
    Draws n samples of length-8:
        [R1, k2, k2a, gamma, tD, tP, alpha, M]
    where M ~ Bernoulli(0.5) (0/1).
    Whenever M == 0, gamma is forced to 0. (MRTM model)

    Returns
    -------
    params : (n, 8)  jnp.ndarray
    """

    key, sub_uniform = jr.split(key)
    base = jr.uniform(sub_uniform, shape=(n, 7), minval=lows, maxval=highs)
    base = base.at[:, 5].add(base[:, 4])  # tP = tD + (tP - tD)

    # ─── Bernoulli indicator ─────────────────────────────────────────────
    key, sub_bern = jr.split(key)
    ind = jr.bernoulli(sub_bern, p=0.5, shape=(n, 1)).astype(base.dtype)  # (n, 1)

    # If M == 0  →  force gamma (column 3) to 0
    gamma = jnp.where(ind.squeeze() == 0, 0.0, base[:, 3])
    base = base.at[:, 3].set(gamma)

    # ─── concatenate and return ─────────────────────────────────────────
    params = jnp.concatenate([base, ind], axis=1)        # (n, 8)
    return params
