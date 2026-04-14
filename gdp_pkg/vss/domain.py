"""
vss/domain.py — Value Objects y lógica de dominio pura del VSS.

Capa: Domain (Clean Architecture)
  - Solo importa numpy y dataclasses.
  - Sin dependencias de CVXPY, ADMM, ni ningún otro módulo del paquete.
  - Los dataclasses son frozen (inmutables) para comportarse como Value Objects.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DetEquivSolution:
    """Decisiones de primera etapa del Problema de Valor Esperado (EV).

    Obtenidas resolviendo el ADMM con S=1 (parámetros estocásticos
    colapsados a su media). Son fijas y no-anticipativas.

    Attributes
    ----------
    eta_mean : potencia declarada al operador GDP [kW]
    c_max    : capacidad CLC contratada [kWh/periodo]
    F_all    : (N, K) ofertas de primera etapa de los FSPs [kWh]
    """
    eta_mean: float
    c_max: float
    F_all: np.ndarray   # (N, K)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetEquivSolution):
            return NotImplemented
        return (
            self.eta_mean == other.eta_mean
            and self.c_max == other.c_max
            and np.array_equal(self.F_all, other.F_all)
        )

    def __hash__(self) -> int:  # type: ignore[override]
        return hash((self.eta_mean, self.c_max))


@dataclass(frozen=True)
class VSSResult:
    """Resultado del cálculo VSS (Value of the Stochastic Solution).

    VSS = profit_sp - profit_eev ≥ 0

    Un VSS positivo justifica la complejidad del modelo estocástico:
    el agregador gana `vss` CAD adicionales por modelar la incertidumbre
    frente a usar simplemente el valor esperado de los parámetros.

    Attributes
    ----------
    profit_sp  : profit óptimo de la solución estocástica [CAD]
    profit_eev : profit esperado al aplicar la solución EV bajo los
                 escenarios GDP reales (OMEGA_PLAGE) [CAD]
    vss        : VSS = profit_sp - profit_eev [CAD]
    ev_solution: decisiones de primera etapa del problema EV
    """
    profit_sp: float
    profit_eev: float
    vss: float
    ev_solution: DetEquivSolution


# ── Fórmula de profit (capa domain, sin CVXPY) ────────────────────────────────

def _profit_from_F(
    F_all: np.ndarray,          # (N, K) oferta no-anticipativa de FSPs [kWh]
    eta_val: float,             # potencia declarada [kW]
    c_max: float,               # capacidad CLC contratada [kWh/periodo]
    p_av: float,
    p_act: float,
    p_CLC: float,
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,    # (S_plage,)
    k_plage: dict,              # {sp: array de índices K}
    k_idx: list,                # K_idx — lista de periodos de flexibilidad
) -> float:
    """Profit del agregador usando recourse óptimo cerrado por escenario GDP.

    Válido tanto para SP (F_all = sp_result.F_all) como para EEV (F_all = ev_sol.F_all).

    El recourse c^sp_k = clip(η·dt − F_sum_k, 0, c_max) es óptimo porque
    el gradiente del profit respecto a c^sp_k es positivo en régimen de
    shortfall (p_act − p_CLC + p_dev > 0) y negativo fuera de él
    (p_act − p_CLC < 0) → siempre conviene llenar el gap hasta c_max.

    Nota: q_k y c_k de ADMMResult son promedios ω-ponderados sobre los 3
    escenarios GDP (necesarios para el acoplamiento ADMM en espacio esperado),
    NO valores por escenario. Usarlos directamente en esta fórmula introduce
    una doble ponderación ω que genera un shortfall artificial.
    """
    rev_avail = (p_av - gamma) * eta_val - p_res * c_max
    F_sum     = F_all.sum(axis=0)       # (K,) — suma sobre N FSPs
    k_idx_arr = np.asarray(k_idx)
    exp_act   = 0.0

    for sp, w in enumerate(omega_plage):
        k_pos = np.array([
            int(np.where(k_idx_arr == k)[0][0]) for k in k_plage[sp]
        ])
        F_sp  = F_sum[k_pos]                         # (|K^sp|,) [kWh]
        gap   = eta_val * dt - F_sp                  # déficit por periodo [kWh]
        c_sp  = np.clip(gap, 0.0, c_max)             # CLC óptimo [kWh]
        q_sp  = F_sp + c_sp                          # entrega total [kWh]

        q_total   = float(np.sum(q_sp))
        c_total   = float(np.sum(c_sp))
        n_periods = len(k_pos)
        shortfall = max(0.0, eta_val - q_total / (n_periods * dt))

        exp_act += w * (
            p_act * q_total
            - p_CLC * c_total
            - p_dev * shortfall
        )

    return rev_avail + exp_act


def compute_profit(
    p_av: float,
    p_act: float,
    p_CLC: float,
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,    # (S_plage,)
    k_plage: dict,              # {sp: array de índices K}
    k_idx: list,                # K_idx — lista de periodos de flexibilidad
    F_all: np.ndarray,          # (N, K) oferta no-anticipativa del SP [kWh]
    eta_val: float,             # potencia declarada [kW]
    c_max: float,               # capacidad CLC contratada [kWh/periodo]
) -> float:
    """Calcula el profit de la solución estocástica (SP).

    Reconstruye el recourse óptimo por escenario GDP desde F_all
    (oferta no-anticipativa de los FSPs), en lugar de usar q_k/c_k
    de ADMMResult que son promedios ω-ponderados.
    """
    return _profit_from_F(
        F_all, eta_val, c_max,
        p_av, p_act, p_CLC, p_dev, p_res, gamma, dt,
        omega_plage, k_plage, k_idx,
    )


def compute_eev_profit(
    p_av: float,
    p_act: float,
    p_CLC: float,
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,    # (S_plage,) = [0.55, 0.11, 0.34]
    k_plage: dict,              # {sp: array de índices K}
    k_idx: list,                # K_idx — lista de periodos de flexibilidad
    ev_sol: DetEquivSolution,
) -> float:
    """Evalúa el profit usando las decisiones EV bajo los escenarios GDP reales.

    Recourse de segunda etapa (forma cerrada):
      gap_k  = η_ev·dt − F_ev_sum_k
      c_ev_k = clip(gap_k, 0, c_max_ev)
      q_ev_k = F_ev_sum_k + c_ev_k
    """
    return _profit_from_F(
        ev_sol.F_all, ev_sol.eta_mean, ev_sol.c_max,
        p_av, p_act, p_CLC, p_dev, p_res, gamma, dt,
        omega_plage, k_plage, k_idx,
    )
