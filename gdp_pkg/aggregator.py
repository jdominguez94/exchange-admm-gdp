"""
aggregator.py — Subproblema del Agregador en el Exchange ADMM (Boyd §7.3.2).

Patrón: Facade — encapsula la construcción y resolución del problema CVXPY
        del agregador en una función llamable simple.

Reformulación con clearing explícito de mercado:
  D_k es la demanda del agregador por flexibilidad FSP — variable de decisión
  de primera etapa, no un parámetro fijo. La restricción de acoplamiento
  Σ_i F_{i,k} = D_k se dualiza explícitamente con el multiplicador λ_k.

Con árbol LMP activo, el problema se extiende a un árbol de dos niveles:
  Nivel 1 — Plage GDP (s): AM / PM / AM+PM     (ω_s = OMEGA_PLAGE)
  Nivel 2 — LMP (l):       Bajo / Medio / Alto  (ν_l = ScenarioTree.nu)

Nodo hoja (s, l): probabilidad conjunta ω_s · ν_l

Problema (dos etapas, árbol GDP × LMP):
  Primera etapa (here-and-now):
    η     — potencia declarada al operador [kW]  (escalar, no-anticipativa)
    c_max — capacidad CLC contratada [kW]         (escalar)
    D_k   — demanda FSP [kWh/dt]  (K_full, no-anticipativa respecto a s y l)

  Segunda etapa (recourse, por nodo hoja (s, l)):
    r^{s,l}_k — shortfall real [kWh]
    c^{s,l}_k — activación CLC [kWh]

  max  (p_av - γ)·η  -  (a·c_max² + b·c_max)
       + Σ_{s,l} ω_s·ν_l · Σ_{k∈K^s} [
           p_act   · q^{s,l}_k          ← q = D_k + c^{s,l}_k
         - p_CLC_l · c^{s,l}_k          ← p_CLC_l = LMP_l ESTOCÁSTICO
         - p_dev   · r^{s,l}_k
       ]
       - Σ_k π_k · λ_k · D_k            ← dual ADMM (clearing explícito)
       - (ρ/2) · ‖D_k - D_k^prev‖²     ← proximal ADMM

  s.t. r^{s,l}_k ≥ η·dt - D_k[K^s] - c^{s,l}_k
       D_k[K^s] + c^{s,l}_k ≤ η·dt
       c^{s,l}_k ≤ c_max·dt
       0 ≤ D_k ≤ η·dt

Sin árbol LMP (scenario_tree=None): L=1, ν=[1], p_CLC_vec=[cfg.p_CLC]
→ comportamiento idéntico al modelo original (backward-compatible).
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
    D_k_prev: np.ndarray,          # (K_full,) demanda FSP del iterado anterior
    lam: np.ndarray,               # (K_full,) multiplicadores duales ADMM
    rho: float,
    pi_k: np.ndarray,              # (K_full,) P(periodo k activo)
    scenario_tree=None,            # ScenarioTree | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Resuelve el subproblema del agregador con clearing explícito (D_k variable).

    Parameters
    ----------
    cfg          : configuración GDP
    pop          : población de FSPs (para K_PLAGE, K_DOBLE, OMEGA)
    eta_max_eff  : potencia máxima efectiva [kW]
    D_k_prev     : (K_full,) demanda FSP del iterado anterior (referencia proximal)
    lam          : (K_full,) multiplicadores duales ADMM
    rho          : parámetro de penalización ρ
    pi_k         : (K_full,) P(periodo k activo) = Σ_{s: k∈K^s} ω^s
    scenario_tree: ScenarioTree con niveles LMP; None → comportamiento original

    Returns
    -------
    (q_full, r_full, c_full, eta_val, c_max_val, D_k_val)
      q_full, r_full, c_full : (K_full,) promedios (ω·ν)-ponderados
      eta_val                : (K_full,) — η escalar expandido a K_full
      c_max_val              : float
      D_k_val                : (K_full,) demanda óptima de flexibilidad FSP
    """
    K_DOBLE = np.concatenate([pop.K_AM, pop.K_PM])
    K_full  = len(K_DOBLE)
    S_PLAGE = len(cfg.OMEGA_PLAGE)
    dt      = cfg.dt

    # ── Setup LMP: escalar (original) o vectorial (árbol) ─────────────
    if scenario_tree is None:
        p_CLC_vec = np.array([cfg.p_CLC])
        nu        = np.array([1.0])
        L         = 1
    else:
        p_CLC_vec = scenario_tree.p_CLC_vec   # (L,)
        nu        = scenario_tree.nu           # (L,)
        L         = scenario_tree.L

    # ── Primera etapa ─────────────────────────────────────────────────
    eta   = cp.Variable(nonneg=True)            # potencia declarada [kW]
    c_max = cp.Variable(nonneg=True)            # cap. CLC contratada [kW]
    D_k   = cp.Variable(K_full, nonneg=True)   # demanda FSP [kWh/dt]

    # ── Segunda etapa: variables indexadas por (sp, l) ────────────────
    r_vars = [[None] * L for _ in range(S_PLAGE)]
    c_vars = [[None] * L for _ in range(S_PLAGE)]
    for sp in range(S_PLAGE):
        ks = len(pop.K_PLAGE[sp])
        for l in range(L):
            r_vars[sp][l] = cp.Variable(ks, nonneg=True)
            c_vars[sp][l] = cp.Variable(ks, nonneg=True)

    # ── Objetivo ──────────────────────────────────────────────────────
    rev_avail = (cfg.p_av - cfg.gamma) * eta - (cfg.c_max_cost_a * cp.square(c_max) + cfg.c_max_cost_b * c_max)

    # Término dual ADMM sobre D_k: −π_k·λ_k·D_k  (global, fuera de escenarios)
    dual_D = (pi_k * lam) @ D_k

    # Término proximal ADMM sobre D_k: −(ρ/2)·‖D_k − D_k^prev‖²  (global)
    prox_D = (rho / 2) * cp.sum_squares(D_k - D_k_prev)

    recourse_terms = []
    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])

        for l in range(L):
            w_leaf  = cfg.OMEGA_PLAGE[sp] * nu[l]
            p_CLC_l = p_CLC_vec[l]

            q_sp = D_k[local_pos] + c_vars[sp][l]   # entrega total [kWh]

            rev_act  = cfg.p_act * cp.sum(q_sp)
            cost_clc = p_CLC_l   * cp.sum(c_vars[sp][l])
            pen_r    = cfg.p_dev * cp.sum(r_vars[sp][l])

            term = w_leaf * (rev_act - cost_clc - pen_r)
            recourse_terms.append(term)

    # Maximizar: f_0  −  dual(D_k)  −  proximal(D_k)
    objective = cp.Maximize(rev_avail + cp.sum(recourse_terms) - dual_D - prox_D)

    # ── Restricciones ─────────────────────────────────────────────────
    cons = [
        eta >= cfg.eta_min,
        eta <= eta_max_eff,
        D_k <= eta * dt,            # cota superior global sobre D_k
    ]

    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])

        for l in range(L):
            cons.append(c_vars[sp][l] <= c_max * dt)
            cons.append(r_vars[sp][l] >= eta * dt - D_k[local_pos] - c_vars[sp][l])
            cons.append(D_k[local_pos] + c_vars[sp][l] <= eta * dt)

    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return (np.zeros(K_full), np.zeros(K_full),
                np.zeros(K_full), np.zeros(K_full), 0.0, np.zeros(K_full))

    eta_scalar = float(np.maximum(0.0, eta.value))
    eta_val    = np.full(K_full, eta_scalar)
    c_max_val  = float(np.maximum(0.0, c_max.value))
    D_k_val    = np.maximum(0.0, D_k.value)

    # ── Reconstruir r, c ponderados sobre K_full ─────────────────────
    r_full = np.zeros(K_full)
    c_full = np.zeros(K_full)
    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])
        for l in range(L):
            w_leaf = cfg.OMEGA_PLAGE[sp] * nu[l]
            r_full[local_pos] += w_leaf * np.maximum(0.0, r_vars[sp][l].value)
            c_full[local_pos] += w_leaf * np.maximum(0.0, c_vars[sp][l].value)

    # q = entrega total = demanda FSP + CLC (ponderado)
    q_full = D_k_val + c_full

    return q_full, r_full, c_full, eta_val, c_max_val, D_k_val
