"""
aggregator.py — Subproblema del Agregador en el Exchange ADMM (Boyd §7.3.2).

Patrón: Facade — encapsula la construcción y resolución del problema CVXPY
        del agregador en una función llamable simple.

El agregador es el único agente de primera etapa que conoce los tres escenarios
de activación GDP (plage horaria: AM, PM, AM+PM) y optimiza sobre ellos.

Problema (dos etapas estocásticas):
  Primera etapa (here-and-now):
    η  — potencia declarada al operador [kW]   (escalar, no-anticipativa)
    c_max — capacidad CLC contratada [kWh/periodo]  (escalar)

  Segunda etapa (recourse, por escenario s de plage):
    r^s_k — shortfall real [kWh]: déficit tras FSPs + CLC
    c^s_k — activación CLC [kWh]

  max  (p_av - γ)·η  -  p_res·c_max
       + Σ_s ω^s · Σ_{k∈K^s} [
           p_act · q^s_k
         - p_CLC · c^s_k
         - p_dev · r^s_k
         - λ_k · c^s_k
         - (ρ/2) · ‖q^s_k - σ^s_k‖²
       ]

  q^s_k = Σ_i F̃^s_{i,k}  +  c^s_k    (entrega total)
  r^s_k ≥ η·dt - Σ_i F̃^s_{i,k} - c^s_k   (shortfall ≥ 0)
  c^s_k ≤ c_max
  q^s_k ≤ η·dt

Nota: Σ_i F̃^s_{i,k} entra como dato externo (señal ADMM), NO como variable.
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp

from .config import GDPConfig
from .population import FspPopulation


def solve_aggregator(
    cfg: GDPConfig,
    pop: FspPopulation,
    eta_max_eff: float,
    sigma_by_scenario: np.ndarray,
    lam: np.ndarray,
    rho: float,
    sum_F_tilde_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Resuelve el subproblema del agregador.

    Parameters
    ----------
    cfg              : configuración GDP
    pop              : población de FSPs (para K_PLAGE, K_DOBLE, OMEGA)
    eta_max_eff      : potencia máxima efectiva [kW]
    sigma_by_scenario: (S_PLAGE, K_full) target proximal por escenario de plage
    lam              : (K_full,) multiplicadores duales ADMM
    rho              : parámetro de penalización ρ
    sum_F_tilde_k    : (K_full,) Σ_i F̃^s_{i,k} — señal ADMM de la iteración

    Returns
    -------
    (q_full, r_full, c_full, eta_val, c_max_val)
      q_full, r_full, c_full : (K_full,) promedios ω-ponderados sobre plage
      eta_val                : (K_full,) — η escalar expandido a K_full
      c_max_val              : float
    """
    K_DOBLE = np.concatenate([pop.K_AM, pop.K_PM])
    K_full  = len(K_DOBLE)
    S_PLAGE = len(cfg.OMEGA_PLAGE)
    dt      = cfg.dt

    # ── Primera etapa ─────────────────────────────────────────────────
    eta   = cp.Variable(nonneg=True)    # potencia declarada [kW]
    c_max = cp.Variable(nonneg=True)    # cap. CLC contratada [kWh/periodo]

    # ── Segunda etapa: variables por escenario de plage ───────────────
    r_vars = []  # shortfall  [kWh]
    c_vars = []  # activación CLC [kWh]
    for sp in range(S_PLAGE):
        ks = len(pop.K_PLAGE[sp])
        r_vars.append(cp.Variable(ks, nonneg=True))
        c_vars.append(cp.Variable(ks, nonneg=True))

    # ── Objetivo ──────────────────────────────────────────────────────
    rev_avail = (cfg.p_av - cfg.gamma) * eta - cfg.p_res * c_max

    recourse_terms = []
    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])

        # Entrega real: FSPs (señal ADMM fija) + CLC
        sum_F_sp = sum_F_tilde_k[local_pos]   # (ks,) dato externo
        q_sp     = sum_F_sp + c_vars[sp]      # entrega total [kWh]

        rev_act  = cfg.p_act  * cp.sum(q_sp)
        cost_clc = cfg.p_CLC  * cp.sum(c_vars[sp])
        pen_r    = cfg.p_dev  * cp.sum(r_vars[sp])

        # ADMM proximal: acoplamiento sobre q^s_k
        sigma_sp = sigma_by_scenario[sp, local_pos]
        lam_sp   = lam[local_pos]
        lin_dual = lam_sp @ c_vars[sp]
        prox_aug = (rho / 2) * cp.sum_squares(q_sp - sigma_sp)

        term = cfg.OMEGA_PLAGE[sp] * (
            rev_act - cost_clc - pen_r - lin_dual - prox_aug
        )
        recourse_terms.append(term)

    objective = cp.Maximize(rev_avail + cp.sum(recourse_terms))

    # ── Restricciones ─────────────────────────────────────────────────
    cons = [eta >= cfg.eta_min, eta <= eta_max_eff]

    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])
        sum_F_sp = sum_F_tilde_k[local_pos]

        # CLC acotada por capacidad contratada
        cons.append(c_vars[sp] <= c_max)

        # Shortfall = déficit real: r^s_k ≥ η·dt - sum_F_sp[k] - c^s_k
        cons.append(r_vars[sp] >= eta * dt - sum_F_sp - c_vars[sp])

        # Entrega no supera promesa (no sobre-entregar)
        cons.append(sum_F_sp + c_vars[sp] <= eta * dt)

    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return (np.zeros(K_full), np.zeros(K_full),
                np.zeros(K_full), np.zeros(K_full), 0.0)

    eta_scalar = float(np.maximum(0.0, eta.value))
    eta_val    = np.full(K_full, eta_scalar)
    c_max_val  = float(np.maximum(0.0, c_max.value))

    # Reconstruir r, c ponderados sobre K_full
    r_full = np.zeros(K_full)
    c_full = np.zeros(K_full)
    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])
        r_full[local_pos] += cfg.OMEGA_PLAGE[sp] * np.maximum(0.0, r_vars[sp].value)
        c_full[local_pos] += cfg.OMEGA_PLAGE[sp] * np.maximum(0.0, c_vars[sp].value)

    # q = entrega total = FSPs + CLC (ω-ponderado)
    q_full = sum_F_tilde_k + c_full

    return q_full, r_full, c_full, eta_val, c_max_val
