from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

# ---------------------------------------------------------------------
# Priors
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