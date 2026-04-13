"""
gdp_pkg/vss — Cálculo del VSS (Value of the Stochastic Solution).

Arquitectura: Clean Architecture (sin DDD)
  domain.py    : Value Objects (frozen) + fórmula EEV pura
  ev_adapter.py: Resuelve el problema EV (S=1) usando ADMM existente
  use_case.py  : Orquesta EV solve + evaluación EEV → VSSResult

API pública:
  compute_vss(cfg, admm_cfg, pop, baseline, sp_result) -> VSSResult
  VSSResult.vss       — valor del beneficio estocástico [CAD]
  VSSResult.profit_sp — profit óptimo estocástico [CAD]
  VSSResult.profit_eev— profit usando decisiones EV bajo escenarios reales [CAD]
"""

from .use_case import compute_vss
from .domain import VSSResult, DetEquivSolution

__all__ = ['compute_vss', 'VSSResult', 'DetEquivSolution']
