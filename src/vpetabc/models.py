from __future__ import annotations
from abc import ABC, abstractmethod ## abstract base class
from functools import partial

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------
# Generic kinetic-model interface
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
# Compartment model
# ---------------------------------------------------------------------
class TwoTissueModel(KineticModel):
    """
    5-parameter irreversible/reversible 2-TCM
        dC₁/dt = K₁·Cp  − (k₂+k₃)·C₁ + k₄·C₂
        dC₂/dt = k₃·C₁ − k₄·C₂
        C_T(t) = C₁ + C₂ + v_b·Cp
    Discretised with 4th order explicit Runge-Kutta. Pure JAX → XLA.
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
            k4 = deriv(state +       dt * k3, Cp_t)
            new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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

    All maths is done on a uniform grid of step dt (explicit RK4).
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

        T = Cr.size
        t = jnp.arange(T, dtype=Cr.dtype) * dt           # (T,)

        dCr = jnp.concatenate([jnp.zeros((1,), Cr.dtype),
                               jnp.diff(Cr) / dt])       # (T,)

        denom = jnp.where(jnp.abs(tP - tD) < 1e-6,
                          1e-6,                          # avoid 0
                          tP - tD)

        tau   = (t - tD) / denom                         # (T,)
        u     = (t >= tD).astype(Cr.dtype)               # Heaviside
        tau   = jnp.where(u, tau, 0)
        h     = u * (tau ** alpha) * jnp.exp(alpha * (1 - tau))
        inputs = jnp.stack([dCr, Cr, h], axis=1)         # (T, 3)

        def deriv(Ct, inp):
            dCr_t, Cr_t, h_t = inp
            return (R1 * dCr_t +
                    k2 * Cr_t  -
                    k2a * Ct   -
                    gamma * Ct * h_t)

        def step(C_prev, inp):
            k1 = deriv(C_prev, inp)
            k2 = deriv(C_prev + 0.5 * dt * k1, inp)      # Cp assumed constant on [t,t+dt]
            k3 = deriv(C_prev + 0.5 * dt * k2, inp)
            k4 = deriv(C_prev +       dt * k3, inp)
            C_new = C_prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return C_new, C_new

        Ct0   = 0.0
        _, Ct = jax.lax.scan(step, Ct0, inputs)          # (T,)

        return Ct