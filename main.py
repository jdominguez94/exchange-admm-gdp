"""
main.py — Punto de entrada para el modelo GDP Bloc 1 (Hydro-Québec).

Uso:
    python main.py

Patrón: Facade — expone una API simple sobre el pipeline completo:
  1. Construir perfiles base (temperatura, NSL)
  2. Generar población de FSPs con parámetros heterogéneos
  3. Resolver baselines QP y calcular headroom de flexibilidad
  4. Ejecutar Exchange ADMM hasta convergencia
  5. Imprimir y guardar resultados
"""

import os
import time
import pickle
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)

from gdp_pkg.config import GDPConfig, ADMMConfig
from gdp_pkg.profiles import build_profiles
from gdp_pkg.population import build_population
from gdp_pkg.baseline import solve_baselines
from gdp_pkg.admm import run_exchange_admm
from gdp_pkg.vss import compute_vss
from gdp_pkg.scenario_tree import build_nyhq_scenario_tree


def main() -> None:
    # ── Configuración ──────────────────────────────────────────────────
    cfg = GDPConfig(
        N=20, T=96, dt=0.25, S=10,
        SEED=42, N_WORKERS=96,
        GDP_RATE_WINTER=5, 
        p_dev_factor=5,
        p_act=0.52, pi_t=0.0621,
        gamma=0.00, p_CLC_factor=2,
        p_res=2.0, C_max=60.0,
        eta_min=10.0, eta_max=100.0,
        ALPHA_HEADROOM=0.8,
        SIGMA_BASELINE=0.03,
        OMEGA_PLAGE=np.array([0.55, 0.11, 0.34]),
        H_AM_START=6, H_AM_END=9,
        H_PM_START=16, H_PM_END=20,
        GRUPOS_ALPHA={'A': 5, 'B': 5, 'C': 5, 'D': 5},
        theta_s=0.2,
    )
    cfg.validate()

    # ── Árbol de escenarios LMP NY-HQ ─────────────────────────────────
    # E[p_CLC] = 0.775 ≈ p_CLC actual (0.52 × 2 = 1.04... ajustar abajo)
    # Nota: en esta config p_CLC_factor=2 → p_CLC=1.04. El árbol se calibra
    # automáticamente respecto a la media del árbol (0.775 CAD/kWh).
    lmp_tree = build_nyhq_scenario_tree(cfg.OMEGA_PLAGE)

    admm_cfg = ADMMConfig(
        rho=0.05,
        max_iter=200,
        eps_primal=1e-3,
        eps_dual=1e-3,
        mu_res=10.0,
        tau_incr=1.2,
        rho_max=10.0,
        rho_min=1e-4,
    )

    rng = np.random.default_rng(cfg.SEED)

    _print_config(cfg)

    # ── Pipeline ───────────────────────────────────────────────────────
    t0 = time.time()

    print("1. Construyendo perfiles...")
    profiles = build_profiles(cfg)

    print("2. Generando población de FSPs...")
    pop = build_population(cfg, profiles, rng)
    print(f"   θ = {pop.theta:.4f} CAD/°C²  |  K = {len(pop.K_idx)} periodos")

    print("3. Calculando baselines QP...")
    baseline = solve_baselines(cfg, pop, profiles, rng)
    print(f"   η_max_eff = {baseline.eta_max_eff:.2f} kW  |"
          f"  F_cap medio = {baseline.F_cap.mean():.3f} kWh/FSP")

    print("4. Exchange ADMM (árbol LMP NY-HQ: Low/Med/High)...")
    result = run_exchange_admm(cfg, admm_cfg, pop, baseline, scenario_tree=lmp_tree)

    # ── Resultados ─────────────────────────────────────────────────────
    _print_results(cfg, pop, baseline, result)

    print("5. Calculando VSS (Problema EV con S=1, p_CLC=E[LMP])...")
    vss_result = compute_vss(cfg, admm_cfg, pop, baseline, result,
                             scenario_tree=lmp_tree)
    _print_vss(vss_result, lmp_tree)

    # ── Guardar ────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    fname = f'results/gdp_admm_N{cfg.N}_S{cfg.S}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump({
            'cfg': cfg, 'admm_cfg': admm_cfg, 'result': result,
            'vss': vss_result, 'lmp_tree': lmp_tree,
        }, f)
    print(f"\nResultados guardados en: {fname}")
    print(f"Tiempo total: {time.time() - t0:.1f}s")


def _print_config(cfg: GDPConfig) -> None:
    print("=" * 65)
    print("GDP BLOC 1 — FINANCIAL PARAMETERS (15-min resolution)")
    print("=" * 65)
    print(f"  N={cfg.N} FSPs  |  T={cfg.T} periodos  |  S={cfg.S} escenarios")
    print(f"  p_av   = {cfg.p_av:.3f} CAD/kW  (crédito GDP)")
    print(f"  p_act  = {cfg.p_act:.4f} CAD/kWh  (activación, FIJO)")
    print(f"  p_dev  = {cfg.p_dev:.4f} CAD/kWh  (penalidad shortfall, FIJO)")
    print(f"  p_CLC  = ESTOCÁSTICO  ← árbol LMP NY-HQ (Low/Med/High)")
    print(f"  η_max  = {cfg.eta_max:.0f} kW  |  η_min = {cfg.eta_min:.0f} kW")
    print(f"  ω = {cfg.OMEGA_PLAGE}  (AM, PM, AM+PM)")
    print()


def _print_results(cfg, pop, baseline, result) -> None:
    K_DOBLE = np.concatenate([pop.K_AM, pop.K_PM])
    J       = cfg.N + 1

    sum_Ft_sk   = result.F_tilde_all.sum(axis=1)
    sum_Ft_exp  = (pop.omega[:, None] * sum_Ft_sk).sum(axis=0)
    sum_Ft_full = np.zeros(len(K_DOBLE))
    k_mask = np.isin(K_DOBLE, pop.K_idx)
    k_pos  = np.array([pop.K_idx.index(k) for k in K_DOBLE[k_mask]])
    sum_Ft_full[k_mask] = sum_Ft_exp[k_pos]

    balance = result.q_k - result.c_k - sum_Ft_full

    rev_avail  = cfg.p_av * result.eta_k.mean() - cfg.p_res * result.c_max_opt
    exp_act    = 0.0
    for sp in range(len(cfg.OMEGA_PLAGE)):
        local_pos = np.array([np.where(K_DOBLE == k)[0][0]
                               for k in pop.K_PLAGE[sp]])
        q_sp  = result.q_k[local_pos]
        c_sp  = result.c_k[local_pos]
        exp_act += cfg.OMEGA_PLAGE[sp] * (
            cfg.p_act * np.sum(q_sp)
            - cfg.p_CLC * np.sum(c_sp)
            - cfg.p_dev * max(0.0,
                result.eta_k.mean() - np.sum(q_sp) / (len(q_sp) * cfg.dt))
        )
    profit_final = rev_avail + exp_act
    participating = int(np.sum(result.F_all.sum(axis=1) > 1e-3))

    print()
    print("=" * 65)
    print(f"RESULTADOS  (N={cfg.N}, J={J}, dt=15min, S={cfg.S})")
    print("=" * 65)
    print(f"  λ (clearing)  : media={result.lam.mean():.5f}"
          f"  [{result.lam.min():.5f}, {result.lam.max():.5f}] CAD/periodo")
    print(f"  η_k            : media={result.eta_k.mean():.3f} kW"
          f"  (eff={baseline.eta_max_eff:.2f} kW)")
    print(f"  E[q^s_k]       : {result.q_k.mean():.4f} kWh")
    print(f"  E[r^s_k]       : {result.r_k.mean():.4f} kWh  (shortfall)")
    print(f"  E[c^s_k]       : {result.c_k.mean():.4f} kWh  (CLC)")
    print(f"  E[ΣF̃^s_k]      : {sum_Ft_full.mean():.4f} kWh  (FSPs)")
    print(f"  Balance q-c-ΣF̃ : {balance.mean():+.6f}  ← debe → 0")
    print(f"  Profit total   : {profit_final:.4f} CAD"
          f"  (ventana {len(pop.K_idx) * cfg.dt:.0f}h)")
    print(f"  Equiv. mensual : {profit_final / (len(pop.K_idx) * cfg.dt) * 730:.2f} CAD/mes")
    print(f"  C_max óptimo   : {result.c_max_opt:.3f} kWh")
    print(f"  FSPs activos   : {participating}/{cfg.N}")
    print(f"  Iteraciones    : {len(result.hist.res_primal)}")
    print(f"  Wall time      : {result.wall_time:.1f}s")


def _print_vss(vss_result, lmp_tree=None) -> None:
    ev = vss_result.ev_solution
    print()
    print("=" * 65)
    print("VSS — Value of the Stochastic Solution")
    print("=" * 65)
    if lmp_tree is not None:
        for s in lmp_tree.lmp_scenarios:
            print(f"  LMP {s.label:<4}: p_CLC={s.p_CLC_eff:.2f} CAD/kWh"
                  f"  (ν={s.nu:.2f},"
                  f" margen={(0.52 - s.p_CLC_eff):+.2f} CAD/kWh)")
        print(f"  E[p_CLC] = {lmp_tree.p_CLC_mean:.3f} CAD/kWh  (usado por EV)")
        print()
    print(f"  Profit SP  (estocástico) : {vss_result.profit_sp:>10.4f} CAD")
    print(f"  Profit EEV (det. → stoc) : {vss_result.profit_eev:>10.4f} CAD")
    print(f"  VSS = SP − EEV           : {vss_result.vss:>10.4f} CAD"
          f"  ({'≥0 ✓' if vss_result.vss >= -1e-6 else '<!> negativo'})")
    print(f"  η_sp                     : {vss_result.ev_solution.eta_mean:>10.3f} kW  (EV)")
    print(f"  c_max_ev                 : {ev.c_max:>10.3f} kWh")
    print(f"  F_ev medio por FSP       : {ev.F_all.mean():>10.4f} kWh")


if __name__ == '__main__':
    main()
