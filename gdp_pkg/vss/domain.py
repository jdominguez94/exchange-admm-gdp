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

def compute_profit(
    p_av: float,
    p_act: float,
    p_CLC: float,
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,        # (S_plage,)
    k_plage: dict,                   # {sp: array of K indices}
    k_doble: np.ndarray,             # (K_full,) mapping K_idx → K_full position
    eta_k: np.ndarray,               # (K_full,) o broadcast scalar
    c_max: float,
    q_k: np.ndarray,                 # (K_full,) entrega ponderada por ω [kWh]
    c_k: np.ndarray,                 # (K_full,) CLC ponderado [kWh]
) -> float:
    """Calcula el profit esperado del agregador.

    Fórmula (espejo de admm._compute_profit):
      rev = (p_av - γ)·η_mean - p_res·c_max
      act = Σ_sp ω_sp · [p_act·Σq - p_CLC·Σc - p_dev·max(0, η - mean_q/dt)]
    """
    eta_mean  = float(np.mean(eta_k))
    rev_avail = (p_av - gamma) * eta_mean - p_res * c_max
    exp_act   = 0.0
    for sp, w in enumerate(omega_plage):
        local_pos = np.array([
            int(np.where(k_doble == k)[0][0]) for k in k_plage[sp]
        ])
        q_sp  = q_k[local_pos]
        c_sp  = c_k[local_pos]
        exp_act += w * (
            p_act * float(np.sum(q_sp))
            - p_CLC * float(np.sum(c_sp))
            - p_dev * max(0.0, eta_mean - float(np.sum(q_sp)) / (len(q_sp) * dt))
        )
    return rev_avail + exp_act


def compute_eev_profit(
    p_av: float,
    p_act: float,
    p_CLC: float,
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,        # (S_plage,) = [0.55, 0.11, 0.34]
    k_plage: dict,                   # {sp: array de índices K}
    k_idx: list,                     # lista K_idx (igual que K_DOBLE)
    ev_sol: DetEquivSolution,
) -> float:
    """Evalúa el profit usando las decisiones EV bajo los escenarios GDP reales.

    Recourse de segunda etapa (forma cerrada, sin CVXPY):
      gap_k  = η_ev·dt - F_ev_sum_k
      c_ev_k = clip(gap_k, 0, c_max_ev)
      r_ev_k = max(0, gap_k - c_ev_k)
      q_ev_k = F_ev_sum_k + c_ev_k
    """
    eta_ev      = ev_sol.eta_mean
    c_max_ev    = ev_sol.c_max
    F_ev_sum    = ev_sol.F_all.sum(axis=0)   # (K,)
    k_idx_arr   = np.asarray(k_idx)

    rev_avail = (p_av - gamma) * eta_ev - p_res * c_max_ev
    exp_act   = 0.0

    for sp, w in enumerate(omega_plage):
        active_k = k_plage[sp]
        # Posición de cada k activo dentro de K_idx
        k_pos = np.array([int(np.where(k_idx_arr == k)[0][0]) for k in active_k])

        F_sum = F_ev_sum[k_pos]                          # (|K^sp|,) [kWh]
        gap   = eta_ev * dt - F_sum                      # [kWh]
        c_ev  = np.clip(gap, 0.0, c_max_ev)             # [kWh]
        q_ev  = F_sum + c_ev                             # [kWh]

        q_total   = float(np.sum(q_ev))
        c_total   = float(np.sum(c_ev))
        n_periods = len(active_k)
        shortfall = max(0.0, eta_ev - q_total / (n_periods * dt))

        exp_act += w * (
            p_act * q_total
            - p_CLC * c_total
            - p_dev * shortfall
        )

    return rev_avail + exp_act
