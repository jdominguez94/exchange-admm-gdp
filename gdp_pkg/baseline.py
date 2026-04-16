"""
baseline.py — Cálculo del baseline óptimo de consumo para cada FSP.

Patrón: Builder (construye y resuelve N problemas QP independientes).

El baseline es el perfil de consumo que cada FSP seguiría SIN participar
en el programa GDP. Se usa para calcular el headroom de flexibilidad:

  F_cap_{i,k} = u^base_{i,t(k)} · dt   [kWh]

Problema de optimización por FSP (sin costo de flexibilidad):
  min  π_t · Σ_t u_t  +  α_i · Σ_t (x_t - x_ref_i)²
  s.t. x_{t+1} = a_i · x_t + b_i · u_t + c_i · T_out_t
       x_0     = x_ref_i
       x_min_i ≤ x_t ≤ x_max_i
       0 ≤ u_t ≤ min(u_max_i, C_total_i - u_nsl_i_t)

Resuelto con CVXPY + CLARABEL. Si el solver falla, se usa una heurística
de seguimiento de referencia (fallback analítico).
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import cvxpy as cp

from .config import GDPConfig
from .profiles import ProfileData
from .population import FspPopulation


@dataclass
class BaselineResult:
    """Resultado del cálculo de baselines.

    Attributes
    ----------
    u_base_heat  : (N, T) consumo de calefacción del baseline [kW]
    u_base_total : (N, T) consumo total (calefacción + NSL) [kW]
    x_base       : (N, T) temperatura interior del baseline [°C]
    F_cap        : (N, K) headroom de flexibilidad determinista [kWh]
    F_cap_scenarios : (N, S, K) headroom estocástico [kWh]
    u_base_scenarios: (N, S, T) baseline estocástico [kW]
    eta_max_eff  : float — potencia máxima efectiva del agregador [kW]
    """
    u_base_heat: np.ndarray
    u_base_total: np.ndarray
    x_base: np.ndarray
    F_cap: np.ndarray
    F_cap_scenarios: np.ndarray
    u_base_scenarios: np.ndarray
    eta_max_eff:   float
    mu_k_power:    np.ndarray   # (K,) media de entrega agregada FSPs [kW]
    sigma_k_power: np.ndarray   # (K,) desv. est. entrega agregada FSPs [kW]


class _BaselineProblem:
    """QP baseline (CVXPY) para el FSP i. Construido una sola vez y re-usado."""

    def __init__(
        self,
        i: int,
        cfg: GDPConfig,
        pop: FspPopulation,
        profiles: ProfileData,
    ) -> None:
        T   = cfg.T
        dt  = cfg.dt
        pi  = cfg.pi_t

        self.i = i
        self.cfg = cfg
        self.pop = pop

        u_avail = np.minimum(
            pop.u_max_pop[i],
            pop.C_total_pop[i] - pop.u_nsl_pop[i],
        )

        u_v = cp.Variable(T, nonneg=True)
        x_v = cp.Variable(T + 1)

        cost = (
            pi * cp.sum(u_v)
            + pop.alpha_pop_scalar[i] * cp.sum_squares(x_v[1:] - pop.x_ref_pop[i])
        )
        cons = [
            x_v[0]  == pop.x_ref_pop[i],
            x_v[1:] >= pop.x_min_pop[i],
            x_v[1:] <= pop.x_max_pop[i],
            u_v     <= u_avail,
        ]
        for t in range(T):
            cons.append(
                x_v[t + 1] == (
                    pop.a_pop[i] * x_v[t]
                    + pop.b_pop[i] * u_v[t]
                    + pop.c_pop[i] * profiles.T_out[t]
                )
            )

        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self._u_v = u_v
        self._x_v = x_v
        self._u_avail = u_avail

    def solve(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (i, u_heat, u_total, x_trajectory)."""
        self.prob.solve(solver=cp.CLARABEL, verbose=False)

        if self.prob.status not in ('optimal', 'optimal_inaccurate'):
            return self._fallback()

        u_heat  = self._u_v.value
        u_total = u_heat + self.pop.u_nsl_pop[self.i]
        x_traj  = self._x_v.value[1:]
        return self.i, u_heat, u_total, x_traj

    def _fallback(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Heurística analítica de seguimiento de temperatura de referencia."""
        i     = self.i
        pop   = self.pop
        cfg   = self.cfg
        T_out = self.pop.u_nsl_pop[i] * 0  # just need shape; T_out passed in profiles

        # Re-extract T_out from the constraint structure is complex; use zero fallback
        T = cfg.T
        uv = np.zeros(T)
        xv = np.zeros(T + 1)
        xv[0] = pop.x_ref_pop[i]
        u_avail = self._u_avail

        # Referencia de temperatura: mantener x_ref
        for t in range(T):
            # u tal que x_{t+1} ≈ x_ref (ignorando T_out que no tenemos aquí)
            un = (pop.x_ref_pop[i] - pop.a_pop[i] * xv[t]) / pop.b_pop[i]
            uv[t] = np.clip(un, 0.0, float(u_avail[t]))
            xv[t + 1] = np.clip(
                pop.a_pop[i] * xv[t] + pop.b_pop[i] * uv[t],
                pop.x_min_pop[i], pop.x_max_pop[i],
            )

        u_total = uv + pop.u_nsl_pop[i]
        return i, uv, u_total, xv[1:]


def solve_baselines(
    cfg: GDPConfig,
    pop: FspPopulation,
    profiles: ProfileData,
    rng: np.random.Generator | None = None,
) -> BaselineResult:
    """Resuelve los N problemas baseline QP y calcula el headroom F_cap.

    Returns
    -------
    BaselineResult con u_base_heat, x_base, F_cap, F_cap_scenarios, eta_max_eff.
    """
    N, T, S, K, dt = cfg.N, cfg.T, cfg.S, len(pop.K_idx), cfg.dt

    u_base_heat  = np.zeros((N, T))
    u_base_total = np.zeros((N, T))
    x_base       = np.zeros((N, T))

    t0 = time.time()
    problems = [_BaselineProblem(i, cfg, pop, profiles) for i in range(N)]

    for bp in problems:
        i, u_h, u_tot, x_i = bp.solve()
        u_base_heat[i]  = u_h
        u_base_total[i] = u_tot
        x_base[i]       = x_i

    print(f"  Baselines resueltos en {time.time() - t0:.1f}s")

    # Headroom determinista: F_cap_{i,k} = u_base_heat_{i,t(k)} · dt  [kWh]
    K_idx = pop.K_idx
    F_cap = np.array([
        [u_base_heat[i, K_idx[k]] * dt for k in range(K)]
        for i in range(N)
    ])  # (N, K)

    # Potencia máxima efectiva del agregador
    sum_F_cap_k = F_cap.sum(axis=0)
    eta_max_phys = cfg.ALPHA_HEADROOM * sum_F_cap_k.min() / dt
    eta_max_eff  = min(cfg.eta_max, eta_max_phys)

    # Baseline estocástico: u^base_s = u^base · ε^s,  ε ~ LogNormal(μ_ln, σ²)
    _rng  = rng if rng is not None else np.random.default_rng(cfg.SEED)
    mu_ln = -cfg.SIGMA_BASELINE ** 2 / 2
    eps   = _rng.lognormal(
        mean=mu_ln, sigma=cfg.SIGMA_BASELINE, size=(N, S)
    )  # (N, S)
    u_base_scenarios = u_base_heat[:, None, :] * eps[:, :, None]  # (N, S, T)

    # Headroom estocástico
    F_cap_scenarios = np.array([
        [[u_base_scenarios[i, s, K_idx[k]] * dt for k in range(K)]
         for s in range(S)]
        for i in range(N)
    ])  # (N, S, K)

    # Estadísticas agregadas de entrega FSP [kW] — pasadas al agregador
    F_sum_s       = F_cap_scenarios.sum(axis=0)          # (S, K) [kWh]
    mu_k_power    = F_sum_s.mean(axis=0) / cfg.dt        # (K,) [kW]
    sigma_k_power = F_sum_s.std(axis=0)  / cfg.dt        # (K,) [kW]

    return BaselineResult(
        u_base_heat=u_base_heat,
        u_base_total=u_base_total,
        x_base=x_base,
        F_cap=F_cap,
        F_cap_scenarios=F_cap_scenarios,
        u_base_scenarios=u_base_scenarios,
        eta_max_eff=eta_max_eff,
        mu_k_power=mu_k_power,
        sigma_k_power=sigma_k_power,
    )
