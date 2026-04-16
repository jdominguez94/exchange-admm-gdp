"""
aggregator.py — Subproblema del Agregador en el Exchange ADMM (Boyd §7.3.2).

Patrón: Facade — encapsula la construcción y resolución del problema CVXPY
        del agregador en una función llamable simple.

El agregador es el único agente de primera etapa que conoce los tres escenarios
de activación GDP (plage horaria: AM, PM, AM+PM) y optimiza sobre ellos.

Con árbol LMP activo, el problema se extiende a un árbol de dos niveles:
  Nivel 1 — Plage GDP (s): AM / PM / AM+PM     (ω_s = OMEGA_PLAGE)
  Nivel 2 — LMP (l):       Bajo / Medio / Alto  (ν_l = ScenarioTree.nu)

Nodo hoja (s, l): probabilidad conjunta ω_s · ν_l

Problema (dos etapas, árbol GDP × LMP):
  Primera etapa (here-and-now):
    η  — potencia declarada al operador [kW]   (escalar, no-anticipativa)
    c_max — capacidad CLC contratada [kW]  (escalar)

  Segunda etapa (recourse, por nodo hoja (s, l)):
    r^{s,l}_k — shortfall real [kWh]
    c^{s,l}_k — activación CLC [kWh]

  max  (p_av - γ)·η  -  p_res·c_max
       + Σ_{s,l} ω_s·ν_l · Σ_{k∈K^s} [
           p_act   · q^{s,l}_k          ← p_act FIJO (pago de HQ)
         - p_CLC_l · c^{s,l}_k          ← p_CLC_l = LMP_l ESTOCÁSTICO
         - p_dev   · r^{s,l}_k          ← p_dev FIJO (penalidad HQ)
         - λ_k · q^{s,l}_k
         - (ρ/2) · ‖q^{s,l}_k - σ^s_k‖²  ← proximal compartido por plage
       ]

  Nota: p_CLC es el costo de oportunidad que el agregador paga al industrial
  (CLC) por reducir su consumo. Este costo = LMP spot NY-HQ (estocástico).

  q^{s,l}_k = Σ_i F̃^s_{i,k}  +  c^{s,l}_k    (entrega total)
  r^{s,l}_k ≥ η·dt - Σ_i F̃^s_{i,k} - c^{s,l}_k
  c^{s,l}_k ≤ c_max·dt
  q^{s,l}_k ≤ η·dt

Sin árbol LMP (scenario_tree=None): L=1, ν=[1], p_CLC_vec=[cfg.p_CLC]
→ comportamiento idéntico al modelo original (backward-compatible).
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp
from scipy.stats import norm

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
    mu_k_power:    np.ndarray,   # (K_full,) media entrega agregada FSPs [kW]
    sigma_k_power: np.ndarray,   # (K_full,) desv. est. [kW]
    scenario_tree=None,          # ScenarioTree | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Resuelve el subproblema del agregador (árbol GDP × LMP opcional).

    Parameters
    ----------
    cfg              : configuración GDP
    pop              : población de FSPs (para K_PLAGE, K_DOBLE, OMEGA)
    eta_max_eff      : potencia máxima efectiva [kW]
    sigma_by_scenario: (S_PLAGE, K_full) target proximal por escenario de plage
    lam              : (K_full,) multiplicadores duales ADMM
    rho              : parámetro de penalización ρ
    sum_F_tilde_k    : (K_full,) Σ_i F̃^s_{i,k} — señal ADMM de la iteración
    mu_k_power       : (K_full,) media potencia FSPs [kW]
    sigma_k_power    : (K_full,) desv. est. potencia FSPs [kW]
    scenario_tree    : ScenarioTree con niveles LMP; None → comportamiento original

    Returns
    -------
    (q_full, r_full, c_full, eta_val, c_max_val)
      q_full, r_full, c_full : (K_full,) promedios (ω·ν)-ponderados
      eta_val                : (K_full,) — η escalar expandido a K_full
      c_max_val              : float
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
    eta   = cp.Variable(nonneg=True)    # potencia declarada [kW]
    c_max = cp.Variable(nonneg=True)    # cap. CLC contratada [kW]

    # ── Segunda etapa: variables indexadas por (sp, l) ────────────────
    # r_vars[sp][l], c_vars[sp][l] ∈ R^{|K^sp|}
    r_vars = [[None] * L for _ in range(S_PLAGE)]
    c_vars = [[None] * L for _ in range(S_PLAGE)]
    for sp in range(S_PLAGE):
        ks = len(pop.K_PLAGE[sp])
        for l in range(L):
            r_vars[sp][l] = cp.Variable(ks, nonneg=True)
            c_vars[sp][l] = cp.Variable(ks, nonneg=True)

    # ── Objetivo ──────────────────────────────────────────────────────
    rev_avail = (cfg.p_av - cfg.gamma) * eta - cfg.p_res * c_max

    recourse_terms = []
    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])

        # Señal ADMM — igual para todo nivel LMP (F es no-anticipativa respecto a LMP)
        sum_F_sp = sum_F_tilde_k[local_pos]   # (ks,) dato externo
        sigma_sp = sigma_by_scenario[sp, local_pos]
        lam_sp   = lam[local_pos]

        for l in range(L):
            w_leaf   = cfg.OMEGA_PLAGE[sp] * nu[l]
            p_CLC_l  = p_CLC_vec[l]

            q_sp = sum_F_sp + c_vars[sp][l]   # entrega total [kWh]

            rev_act  = cfg.p_act  * cp.sum(q_sp)          # p_act FIJO
            cost_clc = p_CLC_l    * cp.sum(c_vars[sp][l]) # p_CLC_l ESTOCÁSTICO
            pen_r    = cfg.p_dev  * cp.sum(r_vars[sp][l]) # p_dev FIJO

            # ADMM proximal: target compartido por plage (precio no afecta cantidades)
            lin_dual = lam_sp @ q_sp
            prox_aug = (rho / 2) * cp.sum_squares(q_sp - sigma_sp)

            term = w_leaf * (rev_act - cost_clc - pen_r - lin_dual - prox_aug)
            recourse_terms.append(term)

    objective = cp.Maximize(rev_avail + cp.sum(recourse_terms))

    # ── Restricciones ─────────────────────────────────────────────────
    z_alpha = norm.ppf(cfg.alpha_delivery)
    cc_rhs  = float(np.max(-mu_k_power + z_alpha * sigma_k_power))  # [kW]
    cons = [eta >= cfg.eta_min, eta <= eta_max_eff, c_max - eta >= cc_rhs]

    for sp in range(S_PLAGE):
        k_idx_local = pop.K_PLAGE[sp]
        local_pos   = np.array([
            np.where(K_DOBLE == k)[0][0] for k in k_idx_local
        ])
        sum_F_sp = sum_F_tilde_k[local_pos]

        for l in range(L):
            # CLC acotada por capacidad contratada
            cons.append(c_vars[sp][l] <= c_max * dt)
            # Shortfall: déficit real ≥ 0
            cons.append(r_vars[sp][l] >= eta * dt - sum_F_sp - c_vars[sp][l])
            # No sobre-entregar
            cons.append(sum_F_sp + c_vars[sp][l] <= eta * dt)

    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return (np.zeros(K_full), np.zeros(K_full),
                np.zeros(K_full), np.zeros(K_full), 0.0)

    eta_scalar = float(np.maximum(0.0, eta.value))
    eta_val    = np.full(K_full, eta_scalar)
    c_max_val  = float(np.maximum(0.0, c_max.value))

    # ── Reconstruir r, c ponderados sobre K_full ─────────────────────
    # Ponderación conjunta: w_leaf = ω_s · ν_l
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

    # q = entrega total = FSPs + CLC (ponderado)
    q_full = sum_F_tilde_k + c_full

    return q_full, r_full, c_full, eta_val, c_max_val
