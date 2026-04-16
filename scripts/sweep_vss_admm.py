"""
scripts/sweep_vss_admm.py — VSS en el modelo ADMM completo para distintos p_dev.

El standalone revela VSS > 0, pero el precio dual λ del ADMM añade un costo
efectivo al CLC: p_CLC_eff = p_CLC + λ*. Para c_max* > 0 en el equilibrio ADMM:

  (p_act + p_dev − p_CLC − λ*) · dt > p_res

Con λ* ≈ 0.74 y p_res = 2.0, necesitamos p_dev > 9.0 CAD/kWh.
Este script verifica esa hipótesis corriendo el ADMM con distintos p_dev.

Ejecutar:
    cd /home/jovyan/exchange-admm-gdp
    python scripts/sweep_vss_admm.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import io, contextlib
import numpy as np

from gdp_pkg.config    import GDPConfig, ADMMConfig
from gdp_pkg.profiles  import build_profiles
from gdp_pkg.population import build_population
from gdp_pkg.baseline  import solve_baselines
from gdp_pkg.admm      import run_exchange_admm
from gdp_pkg.vss       import compute_vss


def sep(title="", width=72):
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * width)


def run_one(p_dev_factor: float, p_res: float | None = None,
            verbose_admm: bool = False) -> dict:
    """Corre el pipeline completo con p_dev = p_act * factor y retorna métricas VSS."""
    kw = dict(p_dev_factor=p_dev_factor)
    if p_res is not None:
        kw['p_res'] = p_res
    cfg = dataclasses.replace(GDPConfig(), **kw)

    admm_cfg = ADMMConfig()

    # Construir pipeline silenciosamente
    rng = np.random.default_rng(cfg.SEED)
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if not verbose_admm else contextlib.nullcontext()
    with ctx:
        profiles = build_profiles(cfg)
        pop      = build_population(cfg, profiles, rng)
        baseline = solve_baselines(cfg, pop, profiles, rng)
        result   = run_exchange_admm(cfg, admm_cfg, pop, baseline)
        vss_res  = compute_vss(cfg, admm_cfg, pop, baseline, result)

    return dict(
        p_dev_factor = p_dev_factor,
        p_dev        = cfg.p_dev,
        p_res        = cfg.p_res,
        profit_sp    = vss_res.profit_sp,
        profit_eev   = vss_res.profit_eev,
        vss          = vss_res.vss,
        eta          = float(result.eta_k.mean()),
        c_max        = float(result.c_max_opt),
        lambda_mean  = float(result.lam_k.mean()) if hasattr(result, 'lam_k') else float('nan'),
        iters        = result.iterations if hasattr(result, 'iterations') else -1,
    )


def main():
    print()
    sep("VSS en ADMM completo — Barrido p_dev  (N=20, S=10)")
    print("  Hipótesis: CLC viable en ADMM ⟺  "
          "(p_act + p_dev − p_CLC − λ*)·dt > p_res")
    print("  Con λ*≈0.74 y p_res=2.0: umbral p_dev > ~9 CAD/kWh")
    print()

    casos = [
        ("Nominal",   2,   None),    # p_dev = 2× p_act = 1.04
        ("2×",        4,   None),    # p_dev = 4× p_act = 2.08
        ("5×",       10,   None),    # p_dev = 10× p_act = 5.20
        ("10×",      20,   None),    # p_dev = 20× p_act = 10.40
        ("20×",      40,   None),    # p_dev = 40× p_act = 20.80
        ("p_res↓",    2,   0.05),    # p_dev nominal, p_res reducido
    ]

    print(f"  {'Caso':>10}  {'p_dev':>7}  {'p_res':>6}  {'η':>7}  "
          f"{'c_max':>7}  {'profit_sp':>10}  {'VSS':>10}  {'VSS>0?':>7}")
    sep()

    for label, factor, p_res in casos:
        print(f"  {'['+label+']':>10}  ejecutando...", end='\r', flush=True)
        try:
            r = run_one(factor, p_res)
            vss_flag = "SÍ ✓" if r['vss'] > 0.5 else ("≈0" if abs(r['vss']) < 0.5 else "NO ✗")
            p_res_s  = f"{r['p_res']:.3f}"
            print(f"  {'['+label+']':>10}  {r['p_dev']:7.3f}  {p_res_s:>6}  "
                  f"{r['eta']:7.2f}  {r['c_max']:7.3f}  {r['profit_sp']:10.4f}"
                  f"  {r['vss']:10.4f}  {vss_flag:>7}")
        except Exception as e:
            print(f"  {'['+label+']':>10}  ERROR: {e}")

    print()
    sep()
    print()


if __name__ == "__main__":
    main()
