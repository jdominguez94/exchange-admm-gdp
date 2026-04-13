"""
profiles.py — Perfiles base de carga no shiftable (NSL) y temperatura exterior.

Contiene los datos estáticos derivados de mediciones de Hydro-Québec:
  - NSL_PROFILES_KW_H : perfiles horarios de carga no desplazable por tipo de hogar
  - SIGMA_NSL_H       : desviación estándar horaria del NSL
  - T_OUT_HOURLY      : perfil determinista de temperatura exterior (°C) — diseño frio

Función pública:
  build_profiles(cfg) -> ProfileData
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d

from .config import GDPConfig


# ── Datos estáticos ────────────────────────────────────────────────────────
# Perfiles NSL horarios medidos por tipo de hogar [kW]
# Fuente: datos reales HQ (residencias con calefacteur électrique)
NSL_PROFILES_KW_H: dict[str, list[float]] = {
    'H68': [1.5,1.4,1.3,1.2,1.2,1.4,2.0,2.8,3.2,3.0,2.8,2.5,
            2.3,2.2,2.2,2.5,3.0,4.0,5.5,6.5,6.0,4.5,3.0,2.0],
    'H70': [2.0,1.8,1.7,1.6,1.6,1.8,2.2,2.8,3.5,4.0,4.2,4.0,
            3.8,3.5,3.2,3.0,3.2,3.8,4.5,5.0,5.5,6.0,5.0,3.5],
    'H71': [2.5,2.3,2.2,2.0,2.0,2.2,2.8,3.5,3.8,3.5,3.2,3.0,
            2.8,2.8,2.8,3.0,3.2,3.5,3.8,4.0,4.2,4.0,3.5,3.0],
    'H74': [1.0,0.8,0.7,0.6,0.5,0.6,0.8,1.2,1.8,2.2,2.5,2.8,
            3.0,3.2,3.5,3.8,4.0,4.5,5.0,5.5,6.0,5.5,4.0,2.5],
    'H81': [2.0,1.8,1.6,1.5,1.5,1.8,2.5,3.5,4.5,5.0,5.2,5.0,
            4.8,4.5,4.2,4.0,4.2,4.5,5.0,5.5,6.0,6.5,5.5,3.5],
    'H83': [3.0,2.5,2.2,2.0,2.0,2.5,3.5,4.5,5.0,4.5,4.0,3.5,
            3.2,3.0,3.0,3.2,3.5,4.0,4.5,5.0,5.5,6.0,5.0,3.5],
    'H84': [0.3,0.2,0.2,0.1,0.1,0.2,0.5,1.0,1.8,2.2,2.0,1.8,
            1.5,1.2,1.0,1.0,1.2,1.5,1.8,2.0,2.2,2.0,1.5,0.8],
    'H87': [0.5,0.4,0.3,0.3,0.3,0.4,0.8,1.2,1.5,1.5,1.4,1.2,
            1.0,0.9,0.8,0.8,0.9,1.0,1.2,1.5,1.8,2.0,1.5,0.8],
    'H92': [0.8,0.6,0.5,0.4,0.4,0.5,0.8,1.2,1.8,2.2,2.3,2.2,
            2.0,1.8,1.5,1.4,1.5,1.8,2.2,2.5,2.8,2.5,2.0,1.2],
    'H93': [1.2,1.0,0.9,0.8,0.8,1.0,1.5,2.0,2.5,2.8,3.0,3.2,
            3.5,3.8,4.0,4.2,4.3,4.2,3.8,3.5,3.2,3.0,2.5,1.8],
    'H94': [0.5,0.4,0.3,0.3,0.3,0.4,0.6,1.0,1.5,1.8,1.8,1.5,
            1.2,1.0,0.8,0.8,0.9,1.2,1.5,1.8,2.0,1.8,1.2,0.6],
}

# Desviación estándar horaria del NSL (incertidumbre de medición)
SIGMA_NSL_H: np.ndarray = np.array([
    0.20,0.20,0.18,0.18,0.18,0.25,0.40,0.50,
    0.45,0.38,0.32,0.28,0.28,0.28,0.30,0.38,
    0.45,0.50,0.52,0.55,0.50,0.42,0.32,0.22,
])

# Perfil determinista de temperatura exterior [°C] — diseño extremo (enero frio HQ)
T_OUT_HOURLY: np.ndarray = np.array([
    -28, -29, -30, -31, -32, -33, -34, -35,
    -34, -33, -31, -29, -27, -26, -25, -24,
    -25, -26, -27, -28, -29, -30, -31, -32,
], dtype=float)


@dataclass
class ProfileData:
    """Perfiles interpolados a resolución 15 min listos para usar en el modelo.

    Attributes
    ----------
    T_out : (T,) temperatura exterior determinista [°C]
    sigma_nsl_15 : (T,) desviación estándar NSL interpolada
    nsl_base_matrix : (N_profiles, T) perfiles NSL base [kWh/periodo]
    hours_15 : (T,) horas del día en decimal [0, 24)
    profile_names : lista de nombres de perfiles NSL
    """
    T_out: np.ndarray
    sigma_nsl_15: np.ndarray
    nsl_base_matrix: np.ndarray
    hours_15: np.ndarray
    profile_names: list[str]


def build_profiles(cfg: GDPConfig) -> ProfileData:
    """Interpola los perfiles horarios a resolución 15 min (dt = 0.25 h).

    Interpolación lineal sobre la grilla horaria [0..23] hacia [0..23.75]
    con T = 96 puntos equiespaciados.
    """
    hours_h = np.arange(24)
    hours_15 = np.linspace(0, 23, cfg.T)

    # Temperatura exterior interpolada
    T_out = interp1d(hours_h, T_OUT_HOURLY, kind='linear')(hours_15)

    # Desviación estándar NSL interpolada
    sigma_nsl_15 = interp1d(hours_h, SIGMA_NSL_H, kind='linear')(hours_15)

    # Matriz de perfiles NSL (N_profiles, T) — convertida a kWh/periodo
    profile_names = list(NSL_PROFILES_KW_H.keys())
    nsl_base_matrix = np.array([
        interp1d(hours_h, np.array(v, dtype=float), kind='linear')(hours_15)
        for v in NSL_PROFILES_KW_H.values()
    ]) / 4.0  # kW → kWh/periodo (÷ 4 periodos/hora)

    return ProfileData(
        T_out=T_out,
        sigma_nsl_15=sigma_nsl_15,
        nsl_base_matrix=nsl_base_matrix,
        hours_15=hours_15,
        profile_names=profile_names,
    )
