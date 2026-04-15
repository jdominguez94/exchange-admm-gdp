"""
vss/ev_adapter.py — Adapter que resuelve el Problema de Valor Esperado (EV).

Capa: Adapter (Clean Architecture)
  - Importa infraestructura ADMM existente y tipos de dominio.
  - Depende hacia adentro (domain), nunca al revés.

Responsabilidad:
  Colapsar los S=10 escenarios de incertidumbre a su valor esperado (S=1),
  ejecutar run_exchange_admm() y devolver las decisiones de primera etapa
  como un DetEquivSolution (Value Object del domain).
"""

from __future__ import annotations
import dataclasses
import io
import contextlib

import numpy as np

from ..config import GDPConfig, ADMMConfig
from ..population import FspPopulation
from ..baseline import BaselineResult
from ..admm import run_exchange_admm
from .domain import DetEquivSolution


def solve_ev_problem(
    cfg: GDPConfig,
    admm_cfg: ADMMConfig,
    pop: FspPopulation,
    baseline: BaselineResult,
    verbose: bool = False,
) -> DetEquivSolution:
    """Resuelve el Problema de Valor Esperado (EV) colapsando S→1.

    Construye versiones S=1 de cfg, pop y baseline usando
    dataclasses.replace() (no modifica los objetos originales).

    - T_out_scenarios   : media de S escenarios → (1, T)
    - u_nsl_scenarios   : media de S escenarios → (N, 1, T)
    - u_base_scenarios  : media de S escenarios → (N, 1, T)
    - F_cap_scenarios   : media de S escenarios → (N, 1, K)
    - omega             : [1.0]
    - OMEGA_PLAGE       : sin cambios (escenarios GDP, no de incertidumbre)

    Parameters
    ----------
    cfg       : configuración GDP original (S=10)
    admm_cfg  : hiperparámetros ADMM (reutilizados sin modificación)
    pop       : población de FSPs original
    baseline  : resultado de baselines original
    verbose   : si False, suprime la salida del ADMM interno

    Returns
    -------
    DetEquivSolution con las decisiones de primera etapa (η_ev, c_max_ev, F_ev_all)
    """
    # ── Colapsar escenarios de incertidumbre a S=1 ─────────────────────
    cfg_ev = dataclasses.replace(cfg, S=1)

    pop_ev = dataclasses.replace(
        pop,
        T_out_scenarios=pop.T_out_scenarios.mean(axis=0, keepdims=True),    # (1, T)
        u_nsl_scenarios=pop.u_nsl_scenarios.mean(axis=1, keepdims=True),    # (N, 1, T)
        omega=np.array([1.0]),
    )

    baseline_ev = dataclasses.replace(
        baseline,
        u_base_scenarios=baseline.u_base_scenarios.mean(axis=1, keepdims=True),  # (N, 1, T)
        F_cap_scenarios=baseline.F_cap_scenarios.mean(axis=1, keepdims=True),    # (N, 1, K)
        sigma_k_power=baseline.sigma_k_power,  # EV: misma CC que el SP → VSS ≥ 0 garantizado
    )

    # ── Resolver ADMM determinístico ───────────────────────────────────
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if not verbose else contextlib.nullcontext()

    with ctx:
        result_ev = run_exchange_admm(cfg_ev, admm_cfg, pop_ev, baseline_ev)

    # ── Extraer decisiones de primera etapa ────────────────────────────
    return DetEquivSolution(
        eta_mean=float(result_ev.eta_k.mean()),
        c_max=float(result_ev.c_max_opt),
        F_all=result_ev.F_all.copy(),       # (N, K) — copia para independencia
    )
