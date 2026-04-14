"""
vss/use_case.py — Caso de uso: calcular el VSS del Agregador.

Capa: Use Case (Clean Architecture)
  - Orquesta ev_adapter (resolver EV) y domain (evaluar EEV y construir resultado).
  - No conoce CVXPY ni la implementación interna del ADMM.
  - Depende hacia adentro: domain; hacia fuera para inyección: ev_adapter.

VSS = Profit_SP - Profit_EEV ≥ 0
"""

from __future__ import annotations

from ..config import GDPConfig, ADMMConfig
from ..population import FspPopulation
from ..baseline import BaselineResult
from ..admm import ADMMResult
from .domain import (
    DetEquivSolution,
    VSSResult,
    compute_profit,
    compute_eev_profit,
)
from .ev_adapter import solve_ev_problem


def compute_vss(
    cfg: GDPConfig,
    admm_cfg: ADMMConfig,
    pop: FspPopulation,
    baseline: BaselineResult,
    sp_result: ADMMResult,
    verbose: bool = False,
) -> VSSResult:
    """Calcula el VSS (Value of the Stochastic Solution) del Agregador.

    Pasos:
      1. Extrae Profit_SP de sp_result (solución estocástica ya disponible).
      2. Resuelve el Problema EV (S=1) → decisiones de primera etapa.
      3. Evalúa Profit_EEV en forma cerrada bajo escenarios GDP reales.
      4. Retorna VSSResult inmutable.

    Parameters
    ----------
    cfg        : configuración GDP
    admm_cfg   : hiperparámetros ADMM (reutilizados para el EV solve)
    pop        : población de FSPs
    baseline   : resultado de baselines
    sp_result  : resultado del ADMM estocástico (ya ejecutado)
    verbose    : si True, muestra la salida del ADMM interno del EV solve

    Returns
    -------
    VSSResult con profit_sp, profit_eev, vss y ev_solution
    """
    # ── 1. Profit de la solución estocástica (SP) ──────────────────────
    profit_sp = compute_profit(
        p_av=cfg.p_av,
        p_act=cfg.p_act,
        p_CLC=cfg.p_CLC,
        p_dev=cfg.p_dev,
        p_res=cfg.p_res,
        gamma=cfg.gamma,
        dt=cfg.dt,
        omega_plage=cfg.OMEGA_PLAGE,
        k_plage=pop.K_PLAGE,
        k_idx=pop.K_idx,
        F_all=sp_result.F_all,
        eta_val=float(sp_result.eta_k.mean()),
        c_max=sp_result.c_max_opt,
    )

    # ── 2. Resolver Problema EV (determinístico, S=1) ─────────────────
    ev_sol: DetEquivSolution = solve_ev_problem(
        cfg, admm_cfg, pop, baseline, verbose=verbose
    )

    # ── 3. Evaluar EEV bajo escenarios GDP reales ─────────────────────
    profit_eev = compute_eev_profit(
        p_av=cfg.p_av,
        p_act=cfg.p_act,
        p_CLC=cfg.p_CLC,
        p_dev=cfg.p_dev,
        p_res=cfg.p_res,
        gamma=cfg.gamma,
        dt=cfg.dt,
        omega_plage=cfg.OMEGA_PLAGE,
        k_plage=pop.K_PLAGE,
        k_idx=pop.K_idx,
        ev_sol=ev_sol,
    )

    # ── 4. VSS ────────────────────────────────────────────────────────
    return VSSResult(
        profit_sp=profit_sp,
        profit_eev=profit_eev,
        vss=profit_sp - profit_eev,
        ev_solution=ev_sol,
    )
