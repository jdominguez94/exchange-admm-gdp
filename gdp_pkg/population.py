"""
population.py — Generación de la población de FSPs (Flexible Service Providers).

Patrón: Factory / Builder.
  FspPopulation  : dataclass que agrupa TODOS los atributos de la población.
  build_population(cfg, profiles, rng) -> FspPopulation

Responsabilidades:
  1. Muestrear parámetros térmicos heterogéneos (a, b, c, x_ref, etc.)
  2. Asignar perfiles NSL por FSP y generar escenarios estocásticos
  3. Construir alpha_pop(N, T) — 4 grupos de comportamiento de incomodidad
  4. Generar escenarios de temperatura exterior T_out_s

Convención de índices GDP:
  K_AM  = periodos t ∈ [6h, 9h)   — 12 periodos en resolución 15 min
  K_PM  = periodos t ∈ [16h, 20h) — 16 periodos
  K_idx = K_AM ∪ K_PM             — 28 periodos de flexibilidad
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import truncnorm

from .config import GDPConfig
from .profiles import ProfileData


# ── Helpers ────────────────────────────────────────────────────────────────

def _truncated_normal(
    mean: float, std: float, low: float, high: float,
    size: int, rng: np.random.Generator
) -> np.ndarray:
    """Muestrea N valores de una Normal truncada a [low, high]."""
    a = (low  - mean) / std
    b = (high - mean) / std
    return truncnorm.rvs(
        a, b, loc=mean, scale=std,
        size=size, random_state=int(rng.integers(1_000_000_000))
    )


def _make_hour_mask(hours_t: np.ndarray, ranges_h: list[tuple[float, float]]) -> np.ndarray:
    """Bool array (T,): True si la hora t cae en alguno de los rangos [h_start, h_end)."""
    m = np.zeros(len(hours_t), dtype=bool)
    for h_start, h_end in ranges_h:
        m |= (hours_t >= h_start) & (hours_t < h_end)
    return m


# ── Dataclass de resultado ─────────────────────────────────────────────────

@dataclass
class FspPopulation:
    """Conjunto completo de atributos de la población de N FSPs.

    Parámetros térmicos (modelo lineal de primer orden):
      x_{t+1} = a·x_t + b·u_t + c·T_out_t

    Atributos
    ---------
    a_pop, b_pop, c_pop : (N,) parámetros del modelo térmico
    alpha_pop           : (N, T) coeficiente de incomodidad por periodo
    alpha_pop_scalar    : (N,)  versión escalar (para baseline)
    u_max_pop           : (N,) potencia máxima del calefactor [kW]
    x_ref_pop           : (N,) temperatura de confort [°C]
    x_min_pop, x_max_pop: (N,) límites de temperatura [°C]
    C_total_pop         : (N,) capacidad eléctrica total [kW]
    u_nsl_pop           : (N, T) carga no shiftable determinista [kWh/periodo]
    u_nsl_scenarios     : (N, S, T) escenarios estocásticos del NSL
    T_out_scenarios     : (S, T) escenarios estocásticos de temperatura [°C]
    omega               : (S,) pesos equiprobables = 1/S
    K_idx               : lista de índices de periodos de flexibilidad
    K                   : número de periodos de flexibilidad (= 28)
    K_AM, K_PM          : arrays de índices de ventana AM y PM
    K_PLAGE             : dict {s: array} — periodos activos por escenario de plage
    group_labels        : (N,) etiqueta de grupo alpha ('A','B','C','D')
    theta               : umbral global de indiferencia [CAD/°C²]
    """
    # Parámetros térmicos
    a_pop: np.ndarray
    b_pop: np.ndarray
    c_pop: np.ndarray
    alpha_pop: np.ndarray         # (N, T)
    alpha_pop_scalar: np.ndarray  # (N,) — escalar para baseline
    u_max_pop: np.ndarray
    x_ref_pop: np.ndarray
    x_min_pop: np.ndarray
    x_max_pop: np.ndarray
    C_total_pop: np.ndarray

    # NSL y escenarios
    u_nsl_pop: np.ndarray         # (N, T) determinista
    u_nsl_scenarios: np.ndarray   # (N, S, T)
    T_out_scenarios: np.ndarray   # (S, T)
    omega: np.ndarray             # (S,) = 1/S

    # Índices de flexibilidad
    K_idx: list
    K_AM: np.ndarray
    K_PM: np.ndarray
    K_PLAGE: dict

    # Metadatos
    group_labels: np.ndarray
    theta: float


def build_population(
    cfg: GDPConfig,
    profiles: ProfileData,
    rng: np.random.Generator,
) -> FspPopulation:
    """Factory: construye la población de FSPs a partir de la configuración.

    Steps
    -----
    1. Muestrear parámetros térmicos (a, b, c, x_ref, u_max, C_total)
    2. Asignar perfiles NSL heterogéneos + jitter
    3. Generar escenarios estocásticos de NSL (LogNormal) y T_out (Normal)
    4. Calcular theta (umbral de indiferencia económica)
    5. Construir alpha_pop (N, T) para 4 grupos de comportamiento
    6. Calcular K_AM, K_PM, K_idx y K_PLAGE
    """
    N, T, S, dt = cfg.N, cfg.T, cfg.S, cfg.dt

    # ── 1. Parámetros térmicos ─────────────────────────────────────────
    a_pop            = _truncated_normal(0.998, 0.001, 0.994, 0.999, N, rng)
    b_pop            = _truncated_normal(0.020, 0.004, 0.010, 0.035, N, rng)
    c_pop            = _truncated_normal(0.002, 0.001, 0.001, 0.005, N, rng)
    alpha_pop_scalar = _truncated_normal(2.0,   0.5,   1.0,   5.0,   N, rng)
    u_max_pop        = _truncated_normal(8.5,   2.0,   6.5,   13.0,  N, rng)
    x_ref_pop        = _truncated_normal(21.0,  0.8,   19.5,  23.0,  N, rng)
    x_min_pop        = np.full(N, 18.0)
    x_max_pop        = np.full(N, 25.0)
    C_total_pop      = _truncated_normal(18.0,  3.0,   15.0,  28.0,  N, rng)

    # ── 2. NSL heterogéneo ─────────────────────────────────────────────
    n_profiles   = profiles.nsl_base_matrix.shape[0]
    nsl_scale    = (0.8 + 0.4 * (u_max_pop - u_max_pop.min())
                    / (u_max_pop.max() - u_max_pop.min() + 1e-9))  # (N,)
    nsl_jitter   = rng.normal(0, 0.03, (N, T))
    u_nsl_pop    = np.clip(
        profiles.nsl_base_matrix[np.arange(N) % n_profiles] * nsl_scale[:, None]
        + nsl_jitter,
        0.0, None
    )  # (N, T) kWh/periodo

    # ── 3. Escenarios estocásticos ─────────────────────────────────────
    # NSL: LogNormal(μ_ln, σ²) con μ_ln = -σ²/2 → E[ε] = 1
    sigma_nsl_15 = profiles.sigma_nsl_15
    ln_mean_nsl  = -0.5 * sigma_nsl_15 ** 2
    ln_noise     = rng.normal(0, 1, (N, S, T))
    u_nsl_scenarios = np.clip(
        u_nsl_pop[:, None, :]
        * np.exp(ln_mean_nsl[None, None, :] + sigma_nsl_15[None, None, :] * ln_noise),
        0.0, None
    )  # (N, S, T)

    # T_out: Normal con σ(t) creciente durante el día
    hours_15_arr    = profiles.hours_15
    sigma_T         = 0.5 + 0.15 * hours_15_arr
    T_out_scenarios = (profiles.T_out[None, :]
                       + rng.normal(0, 1, (S, T)) * sigma_T[None, :])  # (S, T)

    omega = np.full(S, 1.0 / S)

    # ── 4. Umbral global de indiferencia θ ────────────────────────────
    # FSP ofrece flexibilidad si α_i(t) · b_i · Δx · 2·dt < λ*
    # θ = θ_s · p_act / (b_mean · dx_mean · 2·dt)
    b_mean  = b_pop.mean()
    dx_mean = (x_ref_pop - 18.0).mean()  # margen hasta x_min
    theta   = cfg.theta_s * cfg.p_act / (b_mean * dx_mean * 2 * dt)

    # ── 5. alpha_pop (N, T) — 4 grupos ────────────────────────────────
    alpha_pop   = np.zeros((N, T))
    hours_t     = np.arange(T) * dt  # horas decimales

    nA = cfg.GRUPOS_ALPHA['A']
    nB = cfg.GRUPOS_ALPHA['B']
    nC = cfg.GRUPOS_ALPHA['C']
    nD = cfg.GRUPOS_ALPHA['D']
    idx_A = slice(0,         nA)
    idx_B = slice(nA,        nA + nB)
    idx_C = slice(nA + nB,   nA + nB + nC)
    idx_D = slice(nA + nB + nC, N)

    mask_A = _make_hour_mask(hours_t, [(6, 8), (17, 20)])
    mask_B = _make_hour_mask(hours_t, [(7, 10), (17, 22)])
    mask_C = mask_A.copy()

    # Grupo A — Rígido mañana/tarde
    a_hi = rng.uniform(2.0, 4.0, nA) * theta
    a_lo = rng.uniform(0.3, 0.8, nA) * theta
    alpha_pop[idx_A, :]        = a_lo[:, None]
    alpha_pop[idx_A][:, mask_A] = a_hi[:, None]

    # Grupo B — Rígido extendido
    b_hi = rng.uniform(2.0, 4.0, nB) * theta
    b_lo = rng.uniform(0.3, 0.8, nB) * theta
    alpha_pop[idx_B, :]        = b_lo[:, None]
    alpha_pop[idx_B][:, mask_B] = b_hi[:, None]

    # Grupo C — Comparable al precio (escala baja)
    c_peak = rng.uniform(0.8, 1.5, nC) * theta
    c_off  = rng.uniform(0.1, 0.4, nC) * theta
    alpha_pop[idx_C, :]        = c_off[:, None]
    alpha_pop[idx_C][:, mask_C] = c_peak[:, None]

    # Grupo D — Constante (50 % alto, 50 % bajo)
    nD_hi  = nD // 2
    nD_lo  = nD - nD_hi
    d_hi   = rng.uniform(1.5, 3.0, nD_hi) * theta
    d_lo   = rng.uniform(0.1, 0.6, nD_lo) * theta
    alpha_pop[idx_D][:nD_hi, :] = d_hi[:, None]
    alpha_pop[idx_D][nD_hi:, :] = d_lo[:, None]

    group_labels = np.empty(N, dtype=object)
    group_labels[idx_A] = 'A'
    group_labels[idx_B] = 'B'
    group_labels[idx_C] = 'C'
    group_labels[idx_D] = 'D'

    # ── 6. Índices de flexibilidad ─────────────────────────────────────
    K_AM    = np.where((hours_t >= cfg.H_AM_START) & (hours_t < cfg.H_AM_END))[0]
    K_PM    = np.where((hours_t >= cfg.H_PM_START) & (hours_t < cfg.H_PM_END))[0]
    K_DOBLE = np.concatenate([K_AM, K_PM])
    K_idx   = sorted(list(K_AM) + list(K_PM))
    K_PLAGE = {0: K_AM, 1: K_PM, 2: K_DOBLE}

    assert list(K_DOBLE) == K_idx, "K_idx y K_DOBLE deben coincidir"

    return FspPopulation(
        a_pop=a_pop, b_pop=b_pop, c_pop=c_pop,
        alpha_pop=alpha_pop, alpha_pop_scalar=alpha_pop_scalar,
        u_max_pop=u_max_pop, x_ref_pop=x_ref_pop,
        x_min_pop=x_min_pop, x_max_pop=x_max_pop,
        C_total_pop=C_total_pop,
        u_nsl_pop=u_nsl_pop,
        u_nsl_scenarios=u_nsl_scenarios,
        T_out_scenarios=T_out_scenarios,
        omega=omega,
        K_idx=K_idx, K_AM=K_AM, K_PM=K_PM, K_PLAGE=K_PLAGE,
        group_labels=group_labels,
        theta=theta,
    )
