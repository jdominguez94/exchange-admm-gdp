"""
config.py — Configuración central del modelo GDP Bloc 1 (Hydro-Québec).

Usa dataclasses para encapsular todos los parámetros del problema:
  - GDPConfig  : parámetros físicos, financieros y de escenarios
  - ADMMConfig : hiperparámetros del algoritmo Exchange ADMM
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GDPConfig:
    """Parámetros del programa GDP Bloc 1 de Hydro-Québec.

    GDP = Gestion de la Demande de Puissance (Demand Response).
    FSP = Flexible Service Provider (termostatos residenciales).
    CLC = Charge Limitée Compensée (backup industrial).

    Resolución temporal: 15 minutos (dt = 0.25 h), T = 96 periodos/día.
    Ventana de flexibilidad: K_AM ∪ K_PM = 28 periodos.
    """

    # ── Grilla de simulación ──────────────────────────────────────────
    N: int = 60          # número de FSPs residenciales
    T: int = 96          # periodos por día (resolución 15 min)
    dt: float = 0.25     # horas por periodo
    S: int = 10          # escenarios estocásticos de incertidumbre
    SEED: int = 42
    N_WORKERS: int = 96  # workers para paralelismo ProcessPoolExecutor

    # ── Parámetros financieros ────────────────────────────────────────
    # p_av  : crédito de disponibilidad GDP (pago único por temporada) [CAD/kW]
    # p_act : precio de activación (pago por energía entregada) [CAD/kWh]
    # pi_t  : precio de electricidad que paga el FSP [CAD/kWh]
    # p_dev : penalidad por kWh no entregado (shortfall) [CAD/kWh]
    # p_CLC : costo de activación del backup industrial [CAD/kWh]
    # p_res : costo fijo de reservar capacidad CLC [CAD]
    # gamma : overhead del agregador [CAD/kW]
    GDP_RATE_WINTER: float = 5.5
    p_act: float = 0.52
    pi_t: float = 0.0621
    gamma: float = 0.00
    p_CLC_factor: float = 1.5     # p_CLC = p_act * p_CLC_factor
    p_res: float = 2.0
    C_max: float = 15.0           # kWh — capacidad máxima CLC por periodo

    # ── Límites contractuales del agregador ──────────────────────────
    eta_min: float = 10.0         # kW — mínimo contractual
    eta_max: float = 100.0        # kW — techo contractual
    ALPHA_HEADROOM: float = 0.8   # fracción de headroom físico usable

    # ── Error de pronóstico del baseline ─────────────────────────────
    # u^base_s = u^base · exp(N(μ_ln, σ²)),  μ_ln = -σ²/2  ⟹  E[u^base_s]=u^base
    SIGMA_BASELINE: float = 0.03  # 3 %

    # ── Escenarios de activación GDP (plage horaria) ─────────────────
    # Basado en 37 días únicos de activación residencial (HQ, 2024-12 → 2026-03)
    # Diseño conservador: Enero (17 días de Pointe)
    #   s=0  Solo AM  : ω=0.55
    #   s=1  Solo PM  : ω=0.11
    #   s=2  AM + PM  : ω=0.34
    OMEGA_PLAGE: np.ndarray = field(
        default_factory=lambda: np.array([0.55, 0.11, 0.34])
    )

    # ── Ventanas de activación en horas ──────────────────────────────
    H_AM_START: int = 6
    H_AM_END: int = 9
    H_PM_START: int = 16
    H_PM_END: int = 20

    # ── Grupos de incomodidad térmica (alpha) ─────────────────────────
    # 4 grupos de N/4 FSPs. Controlados por GRUPOS_ALPHA.
    # A: Rígido mañana/tarde   (6-8h, 17-20h) — difícil de activar en pico
    # B: Rígido extendido      (7-10h, 17-22h)
    # C: Comparable al precio  (misma forma que A, escala baja)
    # D: Constante todo el día (50 % alto, 50 % bajo)
    GRUPOS_ALPHA: dict = field(
        default_factory=lambda: {'A': 15, 'B': 15, 'C': 15, 'D': 15}
    )
    theta_s: float = 0.2   # factor de escala para umbral de indiferencia θ

    # ── Propiedades derivadas ─────────────────────────────────────────
    @property
    def p_av(self) -> float:
        return self.GDP_RATE_WINTER

    @property
    def p_dev(self) -> float:
        """Penalidad de shortfall: 5 × p_act."""
        return self.p_act * 5

    @property
    def p_CLC(self) -> float:
        return self.p_act * self.p_CLC_factor

    def validate(self) -> None:
        assert np.isclose(self.OMEGA_PLAGE.sum(), 1.0), \
            "OMEGA_PLAGE debe sumar 1"
        assert sum(self.GRUPOS_ALPHA.values()) == self.N, \
            "Grupos alpha deben sumar N"
        assert self.eta_min < self.eta_max, "eta_min < eta_max requerido"


@dataclass
class ADMMConfig:
    """Hiperparámetros del algoritmo Exchange ADMM (Boyd §7.3.2).

    El parámetro de penalización ρ se adapta automáticamente durante
    la iteración según la heurística de Boyd §3.4.1:
      - Si ‖r_p‖ > μ · ‖r_d‖: ρ ← min(τ · ρ, ρ_max)   (primal grande)
      - Si ‖r_d‖ > μ · ‖r_p‖: ρ ← max(ρ / τ, ρ_min)   (dual grande)
    """
    rho: float = 0.05        # penalización ADMM inicial ρ₀
    max_iter: int = 200      # máximo de iteraciones
    eps_primal: float = 1e-3 # tolerancia residual primal ‖r_p‖ < ε_p
    eps_dual: float = 1e-3   # tolerancia residual dual   ‖r_d‖ < ε_d
    mu_res: float = 20.0     # ratio antes de adaptar ρ (Boyd §3.4.1)
    tau_incr: float = 1.2    # factor de incremento/decremento de ρ
    rho_max: float = 10.0    # cota superior de ρ
    rho_min: float = 1e-4    # cota inferior de ρ
