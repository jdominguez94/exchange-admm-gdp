"""
scripts/sweep_vss_params.py — Barrido paramétrico para identificar cuándo VSS > 0.

Umbral teórico (condición de primer orden sobre c_max):
  p_res < (p_act + p_dev - p_CLC) * dt   →   CLC viable   →   VSS puede ser > 0

Con parámetros nominales:
  umbral_p_res = (0.52 + p_dev - 0.78) * 0.25
  umbral_p_dev = p_res / 0.25 + 0.78 - 0.52 = 4*p_res + 0.26

Ejecutar:
    cd /home/jovyan/exchange-admm-gdp
    python scripts/sweep_vss_params.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import numpy as np
from gdp_pkg.config import GDPConfig
from gdp_pkg.aggregator_standalone import (
    build_k_plage, compute_standalone_vss,
)

M      = 400
SEED   = 42
MU_F   = 32.0   # kW — media real de entrega FSP (del ADMM: 8 kWh / 0.25 h)
SIGMA_F_RATIO = 0.15  # 15 % de mu

def sep(title="", width=72):
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * width)


def run_vss(cfg, mu_f, sigma_f_ratio):
    K_PLAGE, K_DOBLE = build_k_plage(cfg)
    K = len(K_DOBLE)
    mu_k    = np.full(K, mu_f)
    sigma_k = sigma_f_ratio * mu_k
    return compute_standalone_vss(
        cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k,
        M=M, use_cc=False, seed=SEED,   # sin CC para medir VSS intrínseco
    )


# ── Umbral teórico ─────────────────────────────────────────────────────────────
def threshold_p_res(p_dev, p_act=0.52, p_CLC_factor=1.5, dt=0.25):
    p_CLC = p_act * p_CLC_factor
    return (p_act + p_dev - p_CLC) * dt

def threshold_p_dev(p_res, p_act=0.52, p_CLC_factor=1.5, dt=0.25):
    p_CLC = p_act * p_CLC_factor
    return p_res / dt - p_act + p_CLC


# ── E-A: Barrido p_res (p_dev nominal) ────────────────────────────────────────
def sweep_p_res():
    sep("Sweep p_res  (p_dev = 1.04 CAD/kWh × p_act,  sin CC)")
    cfg_base = GDPConfig()
    thr = threshold_p_res(cfg_base.p_dev)
    print(f"  Umbral teórico: p_res* = {thr:.4f} CAD/kW"
          f"  (CLC viable si p_res < {thr:.4f})")
    print()
    print(f"  {'p_res':>8}  {'η_sp':>7}  {'c_max_sp':>9}  {'c_max_ev':>9}"
          f"  {'VSS':>12}  {'CLC?':>6}")
    sep()

    for p_res in [2.0, 1.0, 0.5, 0.3, 0.20, 0.15, 0.10, 0.05, 0.01]:
        cfg = dataclasses.replace(cfg_base, p_res=p_res)
        r   = run_vss(cfg, MU_F, SIGMA_F_RATIO)
        clc = "SÍ" if r['c_max_sp'] > 0.01 else "no"
        print(f"  {p_res:8.3f}  {r['eta_sp']:7.2f}  {r['c_max_sp']:9.4f}"
              f"  {r['c_max_ev']:9.4f}  {r['vss']:12.6f}  {clc:>6}")


# ── E-B: Barrido p_dev (p_res nominal) ────────────────────────────────────────
def sweep_p_dev():
    sep("Sweep p_dev  (p_res = 2.0 CAD/kW,  sin CC)")
    cfg_base = GDPConfig()
    thr = threshold_p_dev(cfg_base.p_res)
    print(f"  Umbral teórico: p_dev* = {thr:.4f} CAD/kWh"
          f"  (CLC viable si p_dev > {thr:.4f})")
    print()
    print(f"  {'p_dev':>8}  {'η_sp':>7}  {'c_max_sp':>9}  {'c_max_ev':>9}"
          f"  {'VSS':>12}  {'CLC?':>6}")
    sep()

    for factor in [2, 4, 8, 10, 12, 16, 20, 30]:
        p_dev = cfg_base.p_act * factor
        cfg   = dataclasses.replace(cfg_base, p_act=cfg_base.p_act)
        # Reemplazar p_dev manualmente via propiedad derivada:
        # p_dev = p_act * factor → necesito un wrapper
        # GDPConfig no tiene p_dev directo; p_dev = p_act * 2 es la propiedad
        # Solución: subclasear o usar campo oculto
        cfg2 = _cfg_with_p_dev(cfg_base, p_dev)
        r    = run_vss(cfg2, MU_F, SIGMA_F_RATIO)
        clc  = "SÍ" if r['c_max_sp'] > 0.01 else "no"
        print(f"  {p_dev:8.4f}  {r['eta_sp']:7.2f}  {r['c_max_sp']:9.4f}"
              f"  {r['c_max_ev']:9.4f}  {r['vss']:12.6f}  {clc:>6}")


def _cfg_with_p_dev(base: GDPConfig, p_dev_target: float) -> GDPConfig:
    """Crea una config con p_dev = p_dev_target ajustando el campo que lo genera."""
    # p_dev es propiedad: p_act * 2  → necesitamos cambiar p_act o usar monkey-patch
    # En su lugar, usamos un dataclass temporal con p_act ajustado solo para p_dev:
    # No es limpio — mejor usar un objeto envolvente.
    # Solución simple: modificar el campo interno que genera p_dev
    # En config.py: p_dev = p_act * 2  →  factor fijo = 2
    # Para cambiar p_dev sin tocar p_act, creamos una subclass o override
    class _PatchedCfg(GDPConfig):
        @property
        def p_dev(self):
            return self._p_dev_override
    obj = dataclasses.replace(base)
    obj.__class__ = _PatchedCfg
    object.__setattr__(obj, '_p_dev_override', p_dev_target)
    return obj


# ── E-C: Mapa 2D p_res × p_dev ────────────────────────────────────────────────
def map_2d():
    sep("Mapa 2D VSS  (p_res × p_dev,  sin CC)")
    cfg_base = GDPConfig()

    p_res_vals = [2.0, 0.5, 0.2, 0.1, 0.05]
    p_dev_vals = [1.04, 2.08, 4.16, 8.32, 10.40]  # × p_act × factor

    header = f"  {'p_res \\ p_dev':>14}" + "".join(f"  {p:>9.2f}" for p in p_dev_vals)
    print(header)
    sep()

    for p_res in p_res_vals:
        row = f"  {p_res:14.3f}"
        for p_dev in p_dev_vals:
            cfg = _cfg_with_p_dev(dataclasses.replace(cfg_base, p_res=p_res), p_dev)
            r   = run_vss(cfg, MU_F, SIGMA_F_RATIO)
            vss_str = f"{r['vss']:+.3f}" if r['c_max_sp'] > 0.01 else "  ─    "
            row += f"  {vss_str:>9}"
        print(row)

    print()
    print("  (valores en CAD; '─' = CLC no contratado, VSS ≈ 0)")


# ── E-D: Barrido σ/μ en región viable ─────────────────────────────────────────
def sweep_sigma_viable():
    sep("Sweep σ/μ  en región viable  (p_res=0.10, p_dev=8×p_act, sin CC)")
    cfg_base = GDPConfig()
    cfg = _cfg_with_p_dev(dataclasses.replace(cfg_base, p_res=0.10),
                           cfg_base.p_act * 10)
    thr = threshold_p_res(cfg.p_dev)
    print(f"  p_res={cfg.p_res}  p_dev={cfg.p_dev:.3f}  umbral_p_res={thr:.4f}  → CLC viable")
    print()
    print(f"  {'σ/μ':>6}  {'c_max_sp':>9}  {'c_max_ev':>9}  {'VSS':>12}")
    sep()
    for r_sigma in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        r = run_vss(cfg, MU_F, r_sigma)
        print(f"  {r_sigma:6.0%}  {r['c_max_sp']:9.4f}  {r['c_max_ev']:9.4f}  {r['vss']:12.6f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg_nom = GDPConfig()
    print()
    sep("BARRIDO PARAMÉTRICO  —  Condiciones para VSS > 0")
    print(f"  Parámetros nominales: p_act={cfg_nom.p_act}  p_dev={cfg_nom.p_dev}"
          f"  p_CLC={cfg_nom.p_CLC:.3f}  p_res={cfg_nom.p_res}  dt={cfg_nom.dt}")
    print(f"  μ_F (real ADMM) = {MU_F:.1f} kW  σ_F = {SIGMA_F_RATIO:.0%}·μ")
    print(f"  Umbral global: CLC viable ⟺  p_res < (p_act + p_dev − p_CLC)·dt")
    print()

    sweep_p_res()
    print()
    sweep_p_dev()
    print()
    map_2d()
    print()
    sweep_sigma_viable()
    print()
    sep()
    print()


if __name__ == "__main__":
    main()
