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
    c_max    : capacidad CLC contratada [kW]
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
    c_max: float,               # capacidad CLC contratada [kW]
    p_av: float,
    p_act: float,
    p_CLC: float,               # costo CLC escalar (usado si p_CLC_vec=None)
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,    # (S_plage,)
    k_plage: dict,              # {sp: array de índices K}
    k_idx: list,                # K_idx — lista de periodos de flexibilidad
    *,
    p_CLC_vec: np.ndarray | None = None,  # (L,) costos CLC por nivel LMP
    nu: np.ndarray | None = None,          # (L,) pesos probabilísticos LMP
) -> float:
    """Profit del agregador usando recourse óptimo cerrado por escenario.

    Soporta dos modos:
      Determinístico (p_CLC_vec=None): un solo nivel p_CLC escalar.
      Estocástico LMP (p_CLC_vec provisto): doble loop GDP × LMP.

    Válido tanto para SP (F_all = sp_result.F_all) como para EEV (F_all = ev_sol.F_all).

    El recourse c^{sp}_k = clip(η·dt − F_sum_k, 0, c_max·dt) es óptimo porque:
      p_act + p_dev - p_CLC_l > 0  para todo nivel l en el árbol NY-HQ
      (0.52 + 2.60 - 1.30 = 1.82 > 0  incluso en el escenario High)
    → siempre conviene llenar el gap hasta c_max, independientemente del LMP.

    Nota: q_k y c_k de ADMMResult son promedios ω-ponderados sobre los 3
    escenarios GDP (necesarios para el acoplamiento ADMM en espacio esperado),
    NO valores por escenario. Usarlos directamente en esta fórmula introduce
    una doble ponderación ω que genera un shortfall artificial.
    """
    # ── Setup LMP ───────────────────────────────────────────────────────
    if p_CLC_vec is None:
        p_CLC_vec_eff = np.array([p_CLC])
        nu_eff        = np.array([1.0])
    else:
        assert nu is not None, "nu requerido cuando p_CLC_vec es provisto"
        p_CLC_vec_eff = p_CLC_vec
        nu_eff        = nu

    rev_avail = (p_av - gamma) * eta_val - p_res * c_max
    F_sum     = F_all.sum(axis=0)       # (K,) — suma sobre N FSPs
    k_idx_arr = np.asarray(k_idx)
    exp_act   = 0.0

    for sp, w_s in enumerate(omega_plage):
        k_pos = np.array([
            int(np.where(k_idx_arr == k)[0][0]) for k in k_plage[sp]
        ])
        F_sp  = F_sum[k_pos]                         # (|K^sp|,) [kWh]
        gap   = eta_val * dt - F_sp                  # déficit por periodo [kWh]
        c_sp  = np.clip(gap, 0.0, c_max * dt)        # CLC óptimo [kWh]
        q_sp  = F_sp + c_sp                          # entrega total [kWh]

        q_total   = float(np.sum(q_sp))
        c_total   = float(np.sum(c_sp))
        n_periods = len(k_pos)
        shortfall = max(0.0, eta_val - q_total / (n_periods * dt))

        for p_CLC_l, nu_l in zip(p_CLC_vec_eff, nu_eff):
            exp_act += w_s * nu_l * (
                p_act   * q_total
                - p_CLC_l * c_total    # p_CLC_l = LMP_l (estocástico)
                - p_dev   * shortfall
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
    c_max: float,               # capacidad CLC contratada [kW]
    scenario_tree=None,         # ScenarioTree | None
) -> float:
    """Calcula el profit de la solución estocástica (SP).

    Reconstruye el recourse óptimo por escenario GDP desde F_all
    (oferta no-anticipativa de los FSPs), en lugar de usar q_k/c_k
    de ADMMResult que son promedios ω-ponderados.
    """
    p_CLC_vec, nu = _extract_lmp_params(scenario_tree)
    return _profit_from_F(
        F_all, eta_val, c_max,
        p_av, p_act, p_CLC, p_dev, p_res, gamma, dt,
        omega_plage, k_plage, k_idx,
        p_CLC_vec=p_CLC_vec, nu=nu,
    )


def compute_eev_profit(
    p_av: float,
    p_act: float,
    p_CLC: float,
    p_dev: float,
    p_res: float,
    gamma: float,
    dt: float,
    omega_plage: np.ndarray,      # (S_plage,) = [0.55, 0.11, 0.34]
    k_plage: dict,                # {sp: array de índices K}
    k_idx: list,                  # K_idx — lista de periodos de flexibilidad
    ev_sol: DetEquivSolution,
    F_cap_scenarios: np.ndarray,  # (N, S_climate, K) — escenarios originales del SP
    omega_climate: np.ndarray,    # (S_climate,) — pesos (1/S cada uno)
    scenario_tree=None,           # ScenarioTree | None
) -> float:
    """Evalúa el EEV aplicando las decisiones EV a los escenarios climáticos reales.

    Para cada escenario climático s, la entrega real de los FSPs está limitada
    por el headroom factible bajo ese clima:

      F_actual[i,k,s] = min(F_ev[i,k], F_cap_scen[i,s,k])

    Esto captura el coste de sobre-comprometerse bajo el clima promedio (EV):
    si un escenario frío reduce el headroom por debajo de F_ev, el agregador
    incurre en mayor uso de CLC o shortfall.

    EEV = E_s[ profit(F_actual_s, η_ev, c_max_ev) ]
        = Σ_s ω_s · profit(min(F_ev, F_cap_s), η_ev, c_max_ev)
    """
    p_CLC_vec, nu = _extract_lmp_params(scenario_tree)
    total = 0.0
    for s, w_s in enumerate(omega_climate):
        F_actual_s = np.minimum(ev_sol.F_all, F_cap_scenarios[:, s, :])  # (N, K)
        total += w_s * _profit_from_F(
            F_actual_s, ev_sol.eta_mean, ev_sol.c_max,
            p_av, p_act, p_CLC, p_dev, p_res, gamma, dt,
            omega_plage, k_plage, k_idx,
            p_CLC_vec=p_CLC_vec, nu=nu,
        )
    return total


# ── Helper privado ─────────────────────────────────────────────────────────────

def _extract_lmp_params(
    scenario_tree,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extrae p_CLC_vec y nu del árbol si existe; None si no hay árbol."""
    if scenario_tree is None:
        return None, None
    return scenario_tree.p_CLC_vec, scenario_tree.nu
