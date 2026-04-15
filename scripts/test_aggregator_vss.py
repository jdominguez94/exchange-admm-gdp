"""
scripts/test_aggregator_vss.py — Diagnóstico de VSS=0 en el Agregador.

Prueba el agregador en aislamiento asumiendo que la flexibilidad total
de los FSPs se conoce con media μ_k y desviación estándar σ_k.
No requiere FSPs, baselines ni ADMM.

Experimentos
------------
E1  Baseline VSS (σ = 15 % de μ)
E2  Barrido σ/μ  (0 % → 30 %)  — ¿a qué nivel emerge VSS > 0?
E3  Toggle CC    (con vs sin chance constraint)
E4  Barrido α    (0.50 → 0.99) — ¿CC más laxa genera VSS?
E5  Decisiones   (η_sp vs η_ev, c_max_sp vs c_max_ev)

Ejecutar:
    cd /home/jovyan/exchange-admm-gdp
    python scripts/test_aggregator_vss.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import numpy as np

from gdp_pkg.config import GDPConfig
from gdp_pkg.aggregator_standalone import (
    build_k_plage,
    compute_standalone_vss,
    solve_agg_sp,
    solve_agg_ev,
)

# ── Parámetros del experimento ────────────────────────────────────────────────

# μ_k  : media de entrega total de los FSPs [kW] por periodo de flexibilidad
# Usamos un valor uniforme por simplicidad; se puede cargar desde un baseline real.
MU_FRACTION = 0.65   # μ_k = MU_FRACTION × η_max  →  media cubre el 65 % de la demanda
M_SCENARIOS = 300    # escenarios Monte Carlo
SEED        = 42


def build_inputs(cfg: GDPConfig):
    """Construye K_PLAGE, K_DOBLE, mu_k y sigma_k nominales."""
    K_PLAGE, K_DOBLE = build_k_plage(cfg)
    K = len(K_DOBLE)
    mu_k    = np.full(K, MU_FRACTION * cfg.eta_max)  # [kW]
    return K_PLAGE, K_DOBLE, mu_k


def sep(title: str = "", width: int = 72) -> None:
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * width)


# ── E1: Baseline ──────────────────────────────────────────────────────────────

def run_e1(cfg: GDPConfig) -> None:
    sep("E1 · Baseline VSS  (σ = 15 % de μ,  α = 0.99, con CC)")
    K_PLAGE, K_DOBLE, mu_k = build_inputs(cfg)
    sigma_k = 0.15 * mu_k

    res = compute_standalone_vss(cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k,
                                  M=M_SCENARIOS, use_cc=True, seed=SEED)
    _print_vss(res)


# ── E2: Barrido σ/μ ──────────────────────────────────────────────────────────

def run_e2(cfg: GDPConfig) -> None:
    sep("E2 · Barrido σ/μ  (α = 0.99, con CC)")
    K_PLAGE, K_DOBLE, mu_k = build_inputs(cfg)

    ratios = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"  {'σ/μ':>6}  {'η_sp':>7}  {'η_ev':>7}  {'c_sp':>7}  {'c_ev':>7}"
          f"  {'VSS':>10}  {'CC_sp':>6}  {'CC_ev':>6}")
    sep()
    for r in ratios:
        sigma_k = r * mu_k
        res     = compute_standalone_vss(cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k,
                                          M=M_SCENARIOS, use_cc=True, seed=SEED)
        cc_sp = "activa" if res['cc_binding_sp'] else "libre "
        cc_ev = "activa" if res['cc_binding_ev'] else "libre "
        print(f"  {r:6.0%}  {res['eta_sp']:7.2f}  {res['eta_ev']:7.2f}"
              f"  {res['c_max_sp']:7.2f}  {res['c_max_ev']:7.2f}"
              f"  {res['vss']:10.4f}  {cc_sp}  {cc_ev}")


# ── E3: Toggle CC ─────────────────────────────────────────────────────────────

def run_e3(cfg: GDPConfig) -> None:
    sep("E3 · Toggle CC  (σ = 15 % de μ)")
    K_PLAGE, K_DOBLE, mu_k = build_inputs(cfg)
    sigma_k = 0.15 * mu_k

    for use_cc, label in [(True, "Con CC "), (False, "Sin CC")]:
        res = compute_standalone_vss(cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k,
                                      M=M_SCENARIOS, use_cc=use_cc, seed=SEED)
        print(f"\n  [{label}]")
        _print_vss(res)


# ── E4: Barrido α ─────────────────────────────────────────────────────────────

def run_e4(cfg: GDPConfig) -> None:
    sep("E4 · Barrido α  (σ = 15 % de μ, con CC)")
    K_PLAGE, K_DOBLE, mu_k = build_inputs(cfg)
    sigma_k = 0.15 * mu_k

    alphas = [0.50, 0.80, 0.90, 0.95, 0.99]
    print(f"  {'α':>6}  {'z_α':>5}  {'η_sp':>7}  {'η_ev':>7}"
          f"  {'c_sp':>7}  {'c_ev':>7}  {'VSS':>10}  {'cc_rhs':>8}")
    sep()
    from scipy.stats import norm
    for alpha in alphas:
        cfg_a   = dataclasses.replace(cfg, alpha_delivery=alpha)
        res     = compute_standalone_vss(cfg_a, K_PLAGE, K_DOBLE, mu_k, sigma_k,
                                          M=M_SCENARIOS, use_cc=True, seed=SEED)
        z       = norm.ppf(alpha)
        cc_rhs  = res['cc_rhs'] if res['cc_rhs'] == res['cc_rhs'] else float('nan')
        print(f"  {alpha:6.2f}  {z:5.2f}  {res['eta_sp']:7.2f}  {res['eta_ev']:7.2f}"
              f"  {res['c_max_sp']:7.2f}  {res['c_max_ev']:7.2f}"
              f"  {res['vss']:10.4f}  {cc_rhs:8.3f}")


# ── E5: Decisiones detalladas ─────────────────────────────────────────────────

def run_e5(cfg: GDPConfig) -> None:
    sep("E5 · Análisis de decisiones (σ = 15 % de μ, con CC)")
    K_PLAGE, K_DOBLE, mu_k = build_inputs(cfg)
    sigma_k = 0.15 * mu_k

    sp_res = solve_agg_sp(cfg, K_PLAGE, K_DOBLE, mu_k, sigma_k,
                           M=M_SCENARIOS, use_cc=True, seed=SEED)
    ev_res = solve_agg_ev(cfg, K_PLAGE, K_DOBLE, mu_k, use_cc=True)

    print(f"\n  {'':20s}  {'SP':>10}  {'EV':>10}  {'Δ (SP−EV)':>12}")
    sep()
    rows = [
        ("η  [kW]",        sp_res.eta,      ev_res.eta),
        ("c_max  [kW]",    sp_res.c_max,    ev_res.c_max),
        ("c_max − η  [kW]",sp_res.c_max - sp_res.eta, ev_res.c_max - ev_res.eta),
        ("CC RHS  [kW]",   sp_res.cc_rhs,   ev_res.cc_rhs),
        ("CC slack  [kW]", sp_res.cc_slack, ev_res.cc_slack),
        ("profit  [CAD]",  sp_res.profit,   ev_res.profit),
    ]
    for label, v_sp, v_ev in rows:
        delta = v_sp - v_ev if (v_sp == v_sp and v_ev == v_ev) else float('nan')
        print(f"  {label:20s}  {v_sp:10.4f}  {v_ev:10.4f}  {delta:12.4f}")

    print(f"\n  μ_k uniforme       = {mu_k[0]:.2f} kW")
    print(f"  σ_k uniforme       = {sigma_k[0]:.2f} kW  ({sigma_k[0]/mu_k[0]:.0%} de μ)")
    print(f"  eta_max            = {cfg.eta_max:.2f} kW")
    print(f"  α                  = {cfg.alpha_delivery:.2f}")

    print("\n  Interpretación:")
    if abs(sp_res.eta - ev_res.eta) < 0.1:
        print("  → η_sp ≈ η_ev: misma declaración de potencia.")
    if abs(sp_res.c_max - ev_res.c_max) < 0.1:
        print("  → c_max_sp ≈ c_max_ev: misma capacidad CLC contratada.")
    if sp_res.cc_slack is not None and abs(sp_res.cc_slack) < 1e-2:
        print("  → CC activa en SP: la restricción de seguridad fija c_max − η.")
    if ev_res.cc_slack is not None and abs(ev_res.cc_slack) < 1e-2:
        print("  → CC activa en EV: la CC también domina sin incertidumbre.")
    elif ev_res.cc_slack is not None and ev_res.cc_slack > 0.1:
        print("  → CC inactiva en EV (EV más liberal que SP en seguridad).")


# ── Helper ────────────────────────────────────────────────────────────────────

def _print_vss(res: dict) -> None:
    cc_sp = ("activa" if res['cc_binding_sp'] else "libre ") \
            if res['cc_binding_sp'] is not None else "N/A"
    cc_ev = ("activa" if res['cc_binding_ev'] else "libre ") \
            if res['cc_binding_ev'] is not None else "N/A"
    print(f"  η_sp={res['eta_sp']:.2f} kW   c_max_sp={res['c_max_sp']:.2f} kW"
          f"   profit_sp={res['profit_sp']:.4f} CAD   CC_sp={cc_sp}")
    print(f"  η_ev={res['eta_ev']:.2f} kW   c_max_ev={res['c_max_ev']:.2f} kW"
          f"   profit_ev={res['profit_ev']:.4f} CAD   CC_ev={cc_ev}")
    print(f"  profit_eev={res['profit_eev']:.4f} CAD")
    vss = res['vss']
    if abs(vss) < 0.01:
        tag = "  ← ≈ 0: incertidumbre no añade valor"
    elif vss > 0:
        tag = "  ← > 0: SP supera al EEV"
    else:
        tag = f"  ← < 0: CC asimétrica penaliza al SP ({vss:.2f} CAD extra de reserva CLC)"
    print(f"  VSS = {vss:.6f} CAD" + tag)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = GDPConfig()

    print()
    sep("DIAGNÓSTICO VSS DEL AGREGADOR STANDALONE")
    print(f"  η_min={cfg.eta_min} kW  η_max={cfg.eta_max} kW  μ_k={MU_FRACTION*cfg.eta_max:.1f} kW")
    print(f"  p_av={cfg.p_av}  p_act={cfg.p_act}  p_CLC={cfg.p_CLC:.3f}  p_dev={cfg.p_dev:.3f}  p_res={cfg.p_res}")
    print(f"  OMEGA_PLAGE={cfg.OMEGA_PLAGE}  α={cfg.alpha_delivery}  M={M_SCENARIOS}")
    print()

    run_e1(cfg)
    print()
    run_e2(cfg)
    print()
    run_e3(cfg)
    print()
    run_e4(cfg)
    print()
    run_e5(cfg)
    print()
    sep("CONCLUSIONES")
    print("""
  1. VSS con CC < 0 (creciente en σ):
       La CC del SP usa z_{1-α}·σ_k > 0 → cc_rhs_sp > cc_rhs_ev.
       El SP queda atado a mayor c_max que el EV, pagando p_res extra.
       El costo p_res·Δc_max supera el beneficio de CLC adicional → VSS < 0.
       Esto NO es un error: refleja que la CC asimétrica hace al SP más
       conservador de lo económicamente óptimo (dado el nivel de σ).

  2. VSS sin CC ≈ 0 (pequeño positivo):
       Sin la CC, SP y EV tienen la misma estructura de restricciones.
       El SP elige un c_max ligeramente menor (usa explícitamente la
       distribución de F) y obtiene ~0.2 CAD adicional.
       El valor de información es prácticamente nulo: la distribución de F
       no ayuda a mejorar las decisiones porque el payoff es casi lineal
       en F (p_act << p_dev, con recourse sencillo).

  3. Diagnóstico del VSS=0 en el modelo completo (ADMM):
       La CC con α=0.99 y la señal σ_k_power > 0 es el mecanismo que
       genera VSS≤0. Al colapsar σ→0 en el EV, se le da una CC más laxa
       y decisiones más baratas, superando al SP.
       Opciones de rediseño:
         a) Misma CC para SP y EV (usar σ_k en ambos) → VSS ≥ 0 siempre
         b) Separar la CC del problema de VSS (medir solo valor de F dist.)
         c) Modelar la CC como restricción estocástica explícita (CVaR)
    """)
    sep()
    print()


if __name__ == "__main__":
    main()
