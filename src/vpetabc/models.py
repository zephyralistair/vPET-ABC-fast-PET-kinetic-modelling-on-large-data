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
# 2T Compartment model
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
    
# ---------------------------------------------------------------------
# 1T Compartment model with no model selection
# ---------------------------------------------------------------------
class OneTissueModel(KineticModel):
    """
    2-parameter 1TCM model:

        dC_t/dt = K1 * C_p - k2 * C_t

    Parameters
    ----------
    K1, k2
    """

    param_names = ("K1", "k2")

    @partial(jax.jit, static_argnums=(0,))
    def simulate(
        self,
        params: jnp.ndarray,
        Cp: jnp.ndarray,   # plasma/input TAC
        dt: float
    ) -> jnp.ndarray:
        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        K1, k2 = params

        def deriv(Ct, Cp_t):
            return K1 * Cp_t - k2 * Ct

        def step(C_prev, Cp_t):
            k1 = deriv(C_prev, Cp_t)
            k2_ = deriv(C_prev + 0.5 * dt * k1, Cp_t)
            k3 = deriv(C_prev + 0.5 * dt * k2_, Cp_t)
            k4 = deriv(C_prev + dt * k3, Cp_t)
            C_new = C_prev + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
            return C_new, C_new

        Ct0 = jnp.array(0.0, dtype=Cp.dtype)
        _, Ct = jax.lax.scan(step, Ct0, Cp)
        return Ct

# ---------------------------------------------------------------------
# SRTM model
# ---------------------------------------------------------------------
class SRTMModel(KineticModel):
    """
    3-parameter SRTM model (no model selection / indicator).

        dC_t/dt = R1 * dC_r/dt  +  k2 * C_r  -  k2a * C_t
        k2a     = k2 / (1 + BPnd)

    Parameters
    ----------
    R1, k2, BPnd

    All maths is done on a uniform grid of step dt (explicit RK4).
    The reference-region TAC C_r(t) is passed in via `Cr`.
    """

    param_names = ("R1", "k2", "BPnd")

    @partial(jax.jit, static_argnums=(0,))
    def simulate(
        self,
        params: jnp.ndarray,
        Cr: jnp.ndarray,   # reference region TAC
        dt: float
    ) -> jnp.ndarray:

        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        R1, k2, BPnd = params

        # Guard against (1 + BPnd) ~ 0, which would explode k2a
        denom = jnp.where(jnp.abs(1.0 + BPnd) < 1e-6, 1e-6, 1.0 + BPnd)
        k2a = k2 / denom

        T = Cr.size

        # dCr/dt on the same grid (forward diff with 0 at t=0)
        dCr = jnp.concatenate(
            [jnp.zeros((1,), dtype=Cr.dtype), jnp.diff(Cr) / dt],
            axis=0
        )  # (T,)

        inputs = jnp.stack([dCr, Cr], axis=1)  # (T, 2)

        def deriv(Ct, inp):
            dCr_t, Cr_t = inp
            return R1 * dCr_t + k2 * Cr_t - k2a * Ct

        def step(C_prev, inp):
            k1 = deriv(C_prev, inp)
            k2_ = deriv(C_prev + 0.5 * dt * k1, inp)   # assume inputs constant on [t, t+dt]
            k3 = deriv(C_prev + 0.5 * dt * k2_, inp)
            k4 = deriv(C_prev +       dt * k3, inp)
            C_new = C_prev + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
            return C_new, C_new

        Ct0 = jnp.array(0.0, dtype=Cr.dtype)
        _, Ct = jax.lax.scan(step, Ct0, inputs)  # (T,)

        return Ct

# ---------------------------------------------------------------------
# AATH model with time delay
# ---------------------------------------------------------------------
class AATHModel(KineticModel):
    """
    AATH-style model with time delay td:
        irf(t) = F * [H(t) - H(t - Tc)] + K1 * exp(-k2 * (t - Tc)) * H(t - Tc)
        Ct(t)  = (Cp shifted by td) ⊗ irf, discretised via FFT conv, scaled by dt.

    params = (K1, k2, F, Tc, td)
    Notes:
      • Uses linear interpolation to shift Cp by td (left=Cp[0], right=Cp[-1]).
      • Heaviside H(0)=1 to match your NumPy code np.heaviside(·, 1).
      • No explicit blood-volume term here (the intravascular boxcar is in irf).
    """

    param_names = ('K1', 'k2', 'F', 'Tc', 'td')

    @staticmethod
    def _heaviside(x: jnp.ndarray) -> jnp.ndarray:
        # H(x) with H(0)=1 (same convention as np.heaviside(x, 1))
        return jnp.where(x < 0, 0.0, 1.0)

    @staticmethod
    def _interp_shift(Cp: jnp.ndarray, dt: float, td: float) -> jnp.ndarray:
        """
        Linearly interpolate Cp at times (t - td) on a regular grid.
        Boundary behaviour matches np.interp(..., left=Cp[0], right=Cp[-1]).
        """
        T = Cp.shape[0]
        t = jnp.arange(T) * dt
        tq = jnp.clip(t - td, 0.0, t[-1])

        # Convert tq to fractional indices
        fidx  = tq / dt
        idx0f = jnp.floor(fidx)
        idx0  = jnp.clip(idx0f.astype(jnp.int32), 0, T - 1)
        idx1  = jnp.clip(idx0 + 1, 0, T - 1)
        alpha = jnp.clip(fidx - idx0f, 0.0, 1.0)

        Cp0 = jnp.take(Cp, idx0)
        Cp1 = jnp.take(Cp, idx1)
        Cp_td = jnp.where(idx0 < T - 1, (1. - alpha) * Cp0 + alpha * Cp1, Cp[-1])
        return Cp_td

    @staticmethod
    def _fft_convolve_same(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Full convolution via FFT then truncate to len(a).
        Both a and b are length-T here; we keep the first T samples.
        """
        T = a.shape[0]
        L = int(2 * T - 1)
        Af = jnp.fft.rfft(a, n=L)
        Bf = jnp.fft.rfft(b, n=L)
        full = jnp.fft.irfft(Af * Bf, n=L)
        return full[:T]

    @partial(jax.jit, static_argnums=(0,))
    def simulate(self,
                 params: jnp.ndarray,
                 Cp:     jnp.ndarray,
                 dt:     float) -> jnp.ndarray:

        # Guard against NaNs/infs and unpack
        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        K1, k2, F, Tc, td = params

        # Non-negativity for physical parameters (soft constraints)
        K1 = jnp.maximum(K1, 0.0)
        k2 = jnp.maximum(k2, 0.0)
        F  = jnp.maximum(F,  0.0)
        Tc = jnp.maximum(Tc, 0.0)
        # td can be positive or negative by design

        T = Cp.shape[0]
        t = jnp.arange(T) * dt
        H = self._heaviside

        # IRF(t)
        box   = F * (H(t) - H(t - Tc))
        expo  = K1 * jnp.exp(-k2 * jnp.maximum(t - Tc, 0.0)) * H(t - Tc)
        irf   = box + expo

        # Shifted input
        Cp_td = self._interp_shift(Cp, dt, td)

        # Convolution (continuous-time approx: multiply by dt)
        Ct = self._fft_convolve_same(Cp_td, irf) * dt
        return Ct
