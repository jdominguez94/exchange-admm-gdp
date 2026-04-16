"""
scenario_tree.py — Árbol de escenarios LMP (Locational Marginal Price) NY-HQ.

Modela la incertidumbre del precio spot de la interconexión New York – Hydro-Québec
como driver del costo de activación CLC (Charge Limitée Compensée).

Estructura del árbol (dos niveles):
  Nivel 1 — Plage GDP: AM / PM / AM+PM  (probabilidades OMEGA_PLAGE)
  Nivel 2 — LMP:       Bajo / Medio / Alto  (probabilidades ν)

Nodos hoja: (s, l) con probabilidad conjunta  ω_s × ν_l

Economía:
  p_act = 0.52 CAD/kWh  — FIJO (lo que HQ paga al agregador por entrega)
  p_CLC_l = LMP_l       — ESTOCÁSTICO (costo de oportunidad industrial, = LMP)

  Cuando LMP bajo  < p_act: margen positivo en CLC → declarar η alto
  Cuando LMP alto  > p_act: CLC deficitaria per se, pero sigue siendo preferible al
                            shortfall (p_dev = 2.60 >> p_CLC ≤ 1.30 en todos los escenarios)

Calibración (interconexión NY-HQ, eventos Pointe invierno):
  E[p_CLC] = 0.25·0.40 + 0.50·0.70 + 0.25·1.30 = 0.775 ≈ p_CLC actual (0.78 CAD/kWh)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LMPScenario:
    """Un nivel de precio LMP en el árbol de escenarios.

    Attributes
    ----------
    label    : nombre descriptivo ('Low', 'Med', 'High')
    nu       : peso probabilístico (debe sumar 1 sobre todos los niveles)
    p_CLC_eff: costo de activación CLC = LMP del nodo [CAD/kWh]
    """
    label: str
    nu: float
    p_CLC_eff: float


@dataclass(frozen=True)
class ScenarioTree:
    """Árbol de escenarios de dos niveles: GDP plage × LMP.

    Attributes
    ----------
    omega_plage   : (S_plage,) probabilidades de las ramas GDP (AM/PM/AM+PM)
    lmp_scenarios : tupla de LMPScenario (longitud L, ordenada Low→Med→High)
    """
    omega_plage: np.ndarray
    lmp_scenarios: tuple[LMPScenario, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScenarioTree):
            return NotImplemented
        return (
            np.array_equal(self.omega_plage, other.omega_plage)
            and self.lmp_scenarios == other.lmp_scenarios
        )

    def __hash__(self) -> int:
        return hash(tuple(self.lmp_scenarios))

    @property
    def L(self) -> int:
        """Número de niveles LMP."""
        return len(self.lmp_scenarios)

    @property
    def nu(self) -> np.ndarray:
        """(L,) pesos probabilísticos LMP."""
        return np.array([s.nu for s in self.lmp_scenarios])

    @property
    def p_CLC_vec(self) -> np.ndarray:
        """(L,) costos CLC por nivel LMP [CAD/kWh]."""
        return np.array([s.p_CLC_eff for s in self.lmp_scenarios])

    @property
    def p_CLC_mean(self) -> float:
        """E[p_CLC] bajo la distribución LMP [CAD/kWh]."""
        return float(self.nu @ self.p_CLC_vec)

    def leaf_weight(self, s: int, l: int) -> float:
        """Probabilidad conjunta del nodo hoja (s=plage GDP, l=nivel LMP)."""
        return float(self.omega_plage[s] * self.lmp_scenarios[l].nu)


def build_nyhq_scenario_tree(omega_plage: np.ndarray) -> ScenarioTree:
    """Construye el árbol LMP calibrado para la interconexión NY-HQ.

    Niveles calibrados a:
      E[p_CLC] = 0.25·0.40 + 0.50·0.70 + 0.25·1.30 = 0.775 CAD/kWh
               ≈ p_CLC actual (p_act × 1.5 = 0.78 CAD/kWh)

    Fuente: volatilidad histórica LMP hub NY durante eventos Pointe invierno
    (valores típicos rango $30–$1000/MWh, convertidos a CAD/kWh).

    Parameters
    ----------
    omega_plage : (S_plage,) probabilidades GDP — típicamente cfg.OMEGA_PLAGE

    Returns
    -------
    ScenarioTree con L=3 niveles LMP
    """
    lmp_scenarios = (
        LMPScenario(label='Low',  nu=0.25, p_CLC_eff=0.40),
        LMPScenario(label='Med',  nu=0.50, p_CLC_eff=0.70),
        LMPScenario(label='High', nu=0.25, p_CLC_eff=1.30),
    )
    # Validaciones
    nu_sum = sum(s.nu for s in lmp_scenarios)
    assert abs(nu_sum - 1.0) < 1e-9, f"Pesos LMP deben sumar 1, suman {nu_sum}"
    assert abs(omega_plage.sum() - 1.0) < 1e-9, "OMEGA_PLAGE debe sumar 1"

    return ScenarioTree(omega_plage=omega_plage, lmp_scenarios=lmp_scenarios)
