"""
aggregator_standalone.py — Subproblema del Agregador en modo standalone.

Sin acoplamiento ADMM (sin λ, ρ, σ_by_scenario).
La flexibilidad total de los FSPs se modela directamente como
F_k ~ N(μ_k, σ_k²) con M escenarios explícitos.

Permite calcular el VSS del agregador en aislamiento para diagnosticar
por qué el modelo estocástico completo produce VSS≈0.

Dos etapas:
  Primera  (here-and-now): η [kW], c_max [kW]
  Segunda  (recourse):     c^{s,m}_k [kWh], r^{s,m}_k [kWh]
             por escenario GDP s y muestra de flexibilidad m.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.stats import norm

from .config import GDPConfig


# ── Utilidades de índices ─────────────────────────────────────────────────────

def build_k_plage(cfg: GDPConfig) -> tuple[dict, np.ndarray]:
    """Construye K_PLAGE y K_DOBLE a partir de GDPConfig (sin FspPopulation).

    Returns
    -------
    K_PLAGE : dict {0: K_AM, 1: K_PM, 2: K_DOBLE}
    K_DOBLE : np.ndarray de índices de periodo de flexibilidad
    """
    pph     = round(1.0 / cfg.dt)                          # periodos por hora
    K_AM    = np.arange(cfg.H_AM_START * pph, cfg.H_AM_END * pph)
    K_PM    = np.arange(cfg.H_PM_START * pph, cfg.H_PM_END * pph)
    K_DOBLE = np.concatenate([K_AM, K_PM])
    return {0: K_AM, 1: K_PM, 2: K_DOBLE}, K_DOBLE


def _lpos(K_PLAGE: dict, K_DOBLE: np.ndarray, sp: int) -> np.ndarray:
    """Índices locales en K_DOBLE para el escenario GDP sp."""
    return np.array([int(np.where(K_DOBLE == k)[0][0]) for k in K_PLAGE[sp]])


# ── Tipos de resultado ────────────────────────────────────────────────────────

@dataclass
class AggStandaloneResult:
    """Resultado de una solución del agregador standalone (SP o EV)."""
    eta:      float   # potencia declarada [kW]
    c_max:    float   # capacidad CLC contratada [kW]
    profit:   float   # profit esperado [CAD]
    cc_rhs:   float   # RHS de la CC [kW]  (nan si use_cc=False)
    cc_slack: float   # (c_max − η) − cc_rhs  (>0 → CC no activa, ≈0 → activa)


# ── Constructor interno del problema CVXPY ────────────────────────────────────

def _build_agg_problem(
    cfg:     GDPConfig,
    K_PLAGE: dict,
    K_DOBLE: np.ndarray,
    F_kWh:   np.ndarray,   # (M, K) [kWh] — escenarios de entrega FSP
    use_cc:  bool,
    mu_k:    np.ndarray,   # (K,) [kW] — usado solo para la CC
    sigma_k: np.ndarray,   # (K,) [kW] — usado solo para la CC
) -> tuple[cp.Problem, cp.Variable, cp.Variable]:
    """Construye el problema CVXPY del agregador sin términos ADMM."""
    M, K    = F_kWh.shape
    S_PLAGE = len(cfg.OMEGA_PLAGE)
    dt      = cfg.dt

    # ── Variables de primera etapa ────────────────────────────────────
    eta   = cp.Variable(nonneg=True)   # [kW]
    c_max = cp.Variable(nonneg=True)   # [kW]

    # ── Variables de segunda etapa ────────────────────────────────────
    # c_vars[sp]: (M, K_sp) activación CLC [kWh]
    # r_vars[sp]: (M, K_sp) shortfall      [kWh]
    c_vars = {sp: cp.Variable((M, len(K_PLAGE[sp])), nonneg=True)
              for sp in range(S_PLAGE)}
    r_vars = {sp: cp.Variable((M, len(K_PLAGE[sp])), nonneg=True)
              for sp in range(S_PLAGE)}

    # ── Objetivo ──────────────────────────────────────────────────────
    rev_avail      = (cfg.p_av - cfg.gamma) * eta - cfg.p_res * c_max
    recourse_terms = []

    for sp in range(S_PLAGE):
        lpos  = _lpos(K_PLAGE, K_DOBLE, sp)
        F_sp  = F_kWh[:, lpos]          # (M, K_sp) constante numpy
        q_sp  = F_sp + c_vars[sp]       # entrega total [kWh]

        term = cfg.OMEGA_PLAGE[sp] / M * (
            cfg.p_act  * cp.sum(q_sp)
            - cfg.p_CLC * cp.sum(c_vars[sp])
            - cfg.p_dev * cp.sum(r_vars[sp])
        )
        recourse_terms.append(term)

    objective = cp.Maximize(rev_avail + cp.sum(recourse_terms))

    # ── Restricciones ─────────────────────────────────────────────────
    cons = [eta >= cfg.eta_min, eta <= cfg.eta_max]

    if use_cc:
        z_alpha = norm.ppf(cfg.alpha_delivery)
        cc_rhs  = float(np.max(-mu_k + z_alpha * sigma_k))
        cons.append(c_max - eta >= cc_rhs)

    for sp in range(S_PLAGE):
        lpos = _lpos(K_PLAGE, K_DOBLE, sp)
        F_sp = F_kWh[:, lpos]

        cons.append(c_vars[sp] <= c_max * dt)                       # cap. CLC
        # Shortfall: r ≥ η·dt − F − c. Si F+c > η·dt, RHS < 0 y r≥0 lo satisface.
        cons.append(r_vars[sp] >= eta * dt - F_sp - c_vars[sp])

    return cp.Problem(objective, cons), eta, c_max


# ── Solvers públicos ──────────────────────────────────────────────────────────

def solve_agg_sp(
    cfg:     GDPConfig,
    K_PLAGE: dict,
    K_DOBLE: np.ndarray,
    mu_k:    np.ndarray,   # (K,) media entrega FSP [kW]
    sigma_k: np.ndarray,   # (K,) desv. est. [kW]
    M:       int  = 200,
    use_cc:  bool = True,
    seed:    int  = 0,
) -> AggStandaloneResult:
    """SP: resuelve con M escenarios explícitos F^m_k ~ N(μ_k, σ_k²)·dt."""
    rng   = np.random.default_rng(seed)
    dt    = cfg.dt
    K     = len(K_DOBLE)
    scale = np.maximum(sigma_k * dt, 1e-9)

    F_kWh = np.maximum(
        0.0,
        rng.normal(mu_k * dt, scale, size=(M, K)),
    )

    prob, eta_var, c_max_var = _build_agg_problem(
        cfg, K_PLAGE, K_DOBLE, F_kWh, use_cc, mu_k, sigma_k,
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)

    eta_val   = float(max(0.0, eta_var.value))
    c_max_val = float(max(0.0, c_max_var.value))
    profit    = float(prob.value) if prob.status in ('optimal', 'optimal_inaccurate') \
                else float('nan')

    if use_cc:
        z_alpha  = norm.ppf(cfg.alpha_delivery)
        cc_rhs   = float(np.max(-mu_k + z_alpha * sigma_k))
        cc_slack = (c_max_val - eta_val) - cc_rhs
    else:
        cc_rhs   = float('nan')
        cc_slack = float('nan')

    return AggStandaloneResult(
        eta=eta_val, c_max=c_max_val, profit=profit,
        cc_rhs=cc_rhs, cc_slack=cc_slack,
    )


def solve_agg_ev(
    cfg:     GDPConfig,
    K_PLAGE: dict,
    K_DOBLE: np.ndarray,
    mu_k:    np.ndarray,   # (K,) media entrega FSP [kW]
    use_cc:  bool = True,
) -> AggStandaloneResult:
    """EV: resuelve con F_k = μ_k·dt (determinístico, M=1, σ=0)."""
    dt    = cfg.dt
    K     = len(K_DOBLE)
    F_kWh = (mu_k * dt).reshape(1, K)   # (1, K) — un único escenario = media

    sigma_zero = np.zeros(K)
    prob, eta_var, c_max_var = _build_agg_problem(
        cfg, K_PLAGE, K_DOBLE, F_kWh, use_cc, mu_k, sigma_zero,
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)

    eta_val   = float(max(0.0, eta_var.value))
    c_max_val = float(max(0.0, c_max_var.value))
    profit    = float(prob.value) if prob.status in ('optimal', 'optimal_inaccurate') \
                else float('nan')

    if use_cc:
        # EV: σ=0 → cc_rhs = max_k(-μ_k); con μ_k > 0 esto es negativo → CC nunca activa por σ
        cc_rhs   = float(np.max(-mu_k))
        cc_slack = (c_max_val - eta_val) - cc_rhs
    else:
        cc_rhs   = float('nan')
        cc_slack = float('nan')

    return AggStandaloneResult(
        eta=eta_val, c_max=c_max_val, profit=profit,
        cc_rhs=cc_rhs, cc_slack=cc_slack,
    )


def eval_eev_closed_form(
    cfg:      GDPConfig,
    K_PLAGE:  dict,
    K_DOBLE:  np.ndarray,
    mu_k:     np.ndarray,   # (K,) [kW]
    sigma_k:  np.ndarray,   # (K,) [kW]
    eta_ev:   float,
    c_max_ev: float,
    M:        int = 200,
    seed:     int = 0,
) -> float:
    """Evalúa las decisiones EV bajo M escenarios del SP (recourse en forma cerrada).

    gap_k   = η_ev·dt − F^m_k
    c_k     = clip(gap_k, 0, c_max_ev·dt)
    r_k     = max(0, gap_k − c_max_ev·dt)
    q_k     = F^m_k + c_k
    """
    rng   = np.random.default_rng(seed)
    dt    = cfg.dt
    K     = len(K_DOBLE)
    scale = np.maximum(sigma_k * dt, 1e-9)

    F_kWh = np.maximum(
        0.0,
        rng.normal(mu_k * dt, scale, size=(M, K)),
    )

    rev_avail = (cfg.p_av - cfg.gamma) * eta_ev - cfg.p_res * c_max_ev
    exp_act   = 0.0

    for sp, w in enumerate(cfg.OMEGA_PLAGE):
        lpos  = _lpos(K_PLAGE, K_DOBLE, sp)
        F_sp  = F_kWh[:, lpos]                         # (M, K_sp) [kWh]
        gap   = eta_ev * dt - F_sp                     # (M, K_sp) [kWh]
        c_sp  = np.clip(gap, 0.0, c_max_ev * dt)
        r_sp  = np.maximum(0.0, gap - c_max_ev * dt)
        q_sp  = F_sp + c_sp

        exp_act += w / M * (
            cfg.p_act  * q_sp.sum()
            - cfg.p_CLC * c_sp.sum()
            - cfg.p_dev * r_sp.sum()
        )

    return rev_avail + exp_act


# ── VSS standalone ────────────────────────────────────────────────────────────

def compute_standalone_vss(
    cfg:     GDPConfig,
    K_PLAGE: dict,
    K_DOBLE: np.ndarray,
    mu_k:    np.ndarray,
    sigma_k: np.ndarray,
    M:       int  = 200,
    use_cc:  bool = True,
    seed:    int  = 0,
) -> dict:
    """Calcula el VSS del agregador standalone.

    VSS = profit_SP − profit_EEV  (≥ 0 por construcción)

    Returns
    -------
    dict con métricas de diagnóstico: decisiones, profits, VSS, estado CC.
    """
    sp_res     = solve_agg_sp(cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k, M, use_cc, seed)
    ev_res     = solve_agg_ev(cfg, K_PLAGE, K_DOBLE, mu_k, use_cc)
    profit_eev = eval_eev_closed_form(
        cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k,
        ev_res.eta, ev_res.c_max, M, seed,
    )
    vss = sp_res.profit - profit_eev

    return dict(
        eta_sp       = sp_res.eta,
        c_max_sp     = sp_res.c_max,
        profit_sp    = sp_res.profit,
        eta_ev       = ev_res.eta,
        c_max_ev     = ev_res.c_max,
        profit_ev    = ev_res.profit,
        profit_eev   = profit_eev,
        vss          = vss,
        cc_binding_sp = (abs(sp_res.cc_slack) < 1e-2) if use_cc else None,
        cc_binding_ev = (abs(ev_res.cc_slack) < 1e-2) if use_cc else None,
        cc_slack_sp  = sp_res.cc_slack,
        cc_slack_ev  = ev_res.cc_slack,
        cc_rhs       = sp_res.cc_rhs,
    )
