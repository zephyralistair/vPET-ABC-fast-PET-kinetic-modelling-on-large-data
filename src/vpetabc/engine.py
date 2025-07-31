from __future__ import annotations
from functools import partial
from typing import Callable
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import jax.random as jr

from .models import KineticModel

# ---------------------------------------------------------------------
# Vectorised, JIT-compiled rejection-ABC engine
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