"""
admm.py — Loop Exchange ADMM (Boyd §7.3.2) para el problema GDP.

Patrón: Template Method — la estructura del loop es fija; los subproblemas
        (agregador, FSPs) son llamadas a módulos separados.

Algoritmo Exchange ADMM (Gauss-Seidel):
  Convención paper: x̄ = (1/J)·[(q-c) - ΣF̃]  >0 cuando demanda>oferta.

  Step 1 — Agregador (Gauss-Seidel: fija ΣF̃^n, actualiza (q-c)):
    x̄^n     = (1/J)·[(q-c)^n - ΣF̃^n]
    σ^{0,s}_k = ((J-1)/J)·(r^s+c^s)_k - (1/J)·ΣF̃_k   para k ∈ K^s
    (q,r,c,η) ← argmax Π_agg + λ·(q-c) - (ρ/2)·Σπ·‖(q-c)-σ_0‖²

  Step 2a — Recomputa x̄ con (q-c)^{n+1}:
    x̄^{n+½} = (1/J)·[(q-c)^{n+1} - ΣF̃^n]

  Step 2b — FSPs en paralelo (J-1 workers):
    σ_{i,k} = F̃^{s,n}_{i,k} + x̄^{n+½}_k
    F̃^{n+1} ← argmin Πᵢ - λ̃·F̃ + (ρ/2)·Σπ·‖F̃ - σ_i‖²
    s.t. F̃^s_i = F_i ∀s   (no-anticipatividad)

  Step 3 — Actualización dual:
    x̄^{n+1}  = (1/J)·[(q-c)^{n+1} - ΣF̃^{n+1}]
    λ^{n+1}  = λ^n + ρ·[(q-c)^{n+1} - ΣF̃^{n+1}]

  Residuales:
    ‖r_p‖ = ‖E[(q-c)^{n+1} - ΣF̃^{n+1}]‖₂    (desequilibrio esperado)
    ‖r_d‖ = ρ · ‖ΣF̃^{n+1} - ΣF̃^n‖₂          (cambio en oferta)

  Adaptación de ρ (Boyd §3.4.1):
    if ‖r_p‖ > μ·‖r_d‖: ρ ← min(τ·ρ, ρ_max)
    if ‖r_d‖ > μ·‖r_p‖: ρ ← max(ρ/τ, ρ_min)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np

from .config import GDPConfig, ADMMConfig
from .population import FspPopulation
from .baseline import BaselineResult
from .fsp_worker import (
    make_worker_globals, _init_worker, _solve_fsp_worker, solve_fsp_local,
)
from .aggregator import solve_aggregator


@dataclass
class ADMMHistory:
    """Historial de convergencia del loop ADMM."""
    res_primal: list = field(default_factory=list)
    res_dual:   list = field(default_factory=list)
    lam:        list = field(default_factory=list)
    profit:     list = field(default_factory=list)
    q:          list = field(default_factory=list)
    c:          list = field(default_factory=list)
    F_tot:      list = field(default_factory=list)
    rho:        list = field(default_factory=list)
    x_bar:      list = field(default_factory=list)
    c_max:      list = field(default_factory=list)
    eta:        list = field(default_factory=list)


@dataclass
class ADMMResult:
    """Resultado completo del loop ADMM."""
    hist:         ADMMHistory
    u_all:        np.ndarray   # (N, T) consumo óptimo de calefacción [kW]
    x_all:        np.ndarray   # (N, T) temperatura interior [°C]
    F_all:        np.ndarray   # (N, K) oferta de flexibilidad [kWh]
    F_tilde_all:  np.ndarray   # (S, N, K) oferta por escenario [kWh]
    q_k:          np.ndarray   # (K_full,) entrega total ponderada [kWh]
    r_k:          np.ndarray   # (K_full,) shortfall ponderado [kWh]
    c_k:          np.ndarray   # (K_full,) CLC ponderado [kWh]
    eta_k:        np.ndarray   # (K_full,) potencia declarada [kW]
    lam:          np.ndarray   # (K_full,) precio de liquidación [CAD/kWh·dt]
    c_max_opt:    float
    wall_time:    float


def run_exchange_admm(
    cfg: GDPConfig,
    admm_cfg: ADMMConfig,
    pop: FspPopulation,
    baseline: BaselineResult,
) -> ADMMResult:
    """Ejecuta el loop Exchange ADMM Boyd §7.3.2 completo.

    Parameters
    ----------
    cfg      : parámetros GDP
    admm_cfg : hiperparámetros ADMM
    pop      : población de FSPs
    baseline : resultado del cálculo de baselines

    Returns
    -------
    ADMMResult con historia de convergencia y solución óptima.
    """
    N        = cfg.N
    S        = cfg.S
    K_DOBLE  = np.concatenate([pop.K_AM, pop.K_PM])
    K_full   = len(K_DOBLE)
    K        = len(pop.K_idx)
    J        = N + 1              # agentes Boyd (N FSPs + 1 agregador)
    dt       = cfg.dt
    rho      = admm_cfg.rho

    # ── Inicialización ─────────────────────────────────────────────────
    LAM_INIT    = cfg.p_act * 0.1
    F_INIT_FRAC = 0.05
    lam         = np.full(K_full, LAM_INIT)
    F_all       = F_INIT_FRAC * baseline.F_cap               # (N, K)
    F_tilde_all = np.stack([F_all] * S, axis=0)              # (S, N, K)
    q_k         = np.zeros(K_full)
    r_k         = np.zeros(K_full)
    c_k         = np.zeros(K_full)
    eta_k       = np.zeros(K_full)
    c_max_val   = float(cfg.C_max)

    # Mapeo K_idx → K_DOBLE (identidad cuando coinciden)
    k_overlap_mask = np.ones(K_full, dtype=bool)
    k_fsp_pos      = np.arange(K, dtype=int)

    # π_k = P(periodo k activo) = Σ_{s: k∈K^s} ω^s
    pi_k = np.zeros(K_full)
    for sp in range(len(cfg.OMEGA_PLAGE)):
        local_pos = np.array([np.where(K_DOBLE == k)[0][0]
                               for k in pop.K_PLAGE[sp]])
        pi_k[local_pos] += cfg.OMEGA_PLAGE[sp]

    def lam_to_fsp(lam_full: np.ndarray) -> np.ndarray:
        """Proyecta λ (K_full) a λ_fsp (K)."""
        out = np.zeros(K)
        out[k_fsp_pos] = lam_full[k_overlap_mask]
        return out

    # ── Globals para procesos hijos ────────────────────────────────────
    gdict = make_worker_globals(cfg, pop, baseline)

    hist      = ADMMHistory()
    u_all     = np.zeros((N, cfg.T))
    x_all     = np.zeros((N, cfg.T))

    _print_header(J, rho, K_full, admm_cfg, LAM_INIT, F_INIT_FRAC, cfg.p_act,
                  F_all, pi_k, pop)

    # ── Benchmark paralelismo (pre-calentamiento) ──────────────────────
    _benchmark_parallelism(gdict, lam_to_fsp(pi_k * lam), K, rho, cfg.N_WORKERS)

    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=cfg.N_WORKERS,
        initializer=_init_worker,
        initargs=(gdict,),
    ) as pool:
        for n in range(admm_cfg.max_iter):
            sum_Ft_prev_k = F_all.sum(axis=0)                  # (K,) iter anterior

            # ── Step 1: Agregador ──────────────────────────────────────
            sum_Ft_k = F_all.sum(axis=0)                       # (K,) = (K_full,)

            sigma_by_scenario = _compute_sigma_aggregator(
                J, K_full, K_DOBLE, pop, q_k, r_k, c_k, sum_Ft_k, cfg
            )
            q_k, r_k, c_k, eta_k, c_max_val = solve_aggregator(
                cfg, pop, baseline.eta_max_eff,
                sigma_by_scenario, lam, rho, sum_Ft_k,
                mu_k_power=baseline.mu_k_power,
                sigma_k_power=baseline.sigma_k_power,
            )

            # ── Step 2a: x̄^{n+½} ────────────────────────────────────
            x_bar_half = (r_k + c_k - sum_Ft_k) / J            # (K_full,)

            # ── Step 2b: FSPs en paralelo ─────────────────────────────
            lam_expected = pi_k * lam                           # (K_full,)
            lam_fsp      = lam_to_fsp(lam_expected)            # (K,)

            args_list = [
                (i, lam_fsp, F_tilde_all[0, i, :] + x_bar_half, rho)
                for i in range(N)
            ]

            F_new       = np.zeros((N, K))
            F_tilde_new = np.zeros((S, N, K))
            t_fsp       = time.time()

            futures = {pool.submit(_solve_fsp_worker, a): a[0] for a in args_list}
            for fut in as_completed(futures):
                i, u_i, x_i, F_i, Ft_i = fut.result()
                F_new[i]             = F_i
                F_tilde_new[:, i, :] = Ft_i
                u_all[i]             = u_i
                x_all[i]             = x_i
            t_fsp = time.time() - t_fsp

            # ── Step 3: Actualización dual ─────────────────────────────
            sum_Ft_new_k = F_new.sum(axis=0)                   # (K,)

            # Residual primal: E[(q-c) - ΣF̃]
            q_expected = np.zeros(K_full)
            for sp in range(len(cfg.OMEGA_PLAGE)):
                local_pos = np.array([np.where(K_DOBLE == k)[0][0]
                                       for k in pop.K_PLAGE[sp]])
                q_expected[local_pos] += (cfg.OMEGA_PLAGE[sp]
                                          * (r_k[local_pos] + c_k[local_pos]))

            res_p_vec  = q_expected - pi_k * sum_Ft_new_k      # (K_full,)
            lam        = lam + (rho / J) * res_p_vec

            # Residuales escalares
            pres = np.linalg.norm(res_p_vec)
            dres = np.linalg.norm(rho * (sum_Ft_new_k - sum_Ft_prev_k))

            # Adaptación de ρ (Boyd §3.4.1)
            rho = _adapt_rho(rho, pres, dres, admm_cfg)

            # Profit ex-post
            profit = _compute_profit(cfg, pop, K_DOBLE, eta_k, c_max_val, q_k, c_k)

            # Registro de historia
            _record_history(hist, pres, dres, lam, profit, q_k, c_k,
                            sum_Ft_new_k, rho, res_p_vec / J, c_max_val, eta_k)

            elapsed = time.time() - t_start
            _print_iter(n, pres, dres, profit, lam, sum_Ft_new_k, q_k,
                        eta_k, c_max_val, rho, elapsed, t_fsp)

            F_all       = F_new
            F_tilde_all = F_tilde_new

            if pres < admm_cfg.eps_primal and dres < admm_cfg.eps_dual:
                print(f'\n  Convergido en iteración {n + 1}')
                break
        else:
            print(f'\n  Máximo de iteraciones ({admm_cfg.max_iter}) alcanzado')

    wall_time = time.time() - t_start
    print(f'  Tiempo total: {wall_time:.1f}s')

    return ADMMResult(
        hist=hist, u_all=u_all, x_all=x_all,
        F_all=F_all, F_tilde_all=F_tilde_all,
        q_k=q_k, r_k=r_k, c_k=c_k, eta_k=eta_k,
        lam=lam, c_max_opt=c_max_val, wall_time=wall_time,
    )


# ── Helpers internos ───────────────────────────────────────────────────────

def _compute_sigma_aggregator(
    J: int,
    K_full: int,
    K_DOBLE: np.ndarray,
    pop: FspPopulation,
    q_k: np.ndarray,
    r_k: np.ndarray,
    c_k: np.ndarray,
    sum_Ft_k: np.ndarray,
    cfg: GDPConfig,
) -> np.ndarray:
    """Calcula σ^{0,s}_k para el subproblema del agregador (Step 1)."""
    sigma = np.zeros((len(cfg.OMEGA_PLAGE), K_full))
    for sp in range(len(cfg.OMEGA_PLAGE)):
        local_pos = np.array([np.where(K_DOBLE == k)[0][0]
                               for k in pop.K_PLAGE[sp]])
        rc_sp = r_k[local_pos] + c_k[local_pos]
        sigma[sp, local_pos] = (
            ((J - 1) / J) * rc_sp - (1 / J) * sum_Ft_k[local_pos]
        )
    return sigma


def _adapt_rho(
    rho: float,
    pres: float,
    dres: float,
    cfg: ADMMConfig,
) -> float:
    """Heurística de adaptación de ρ (Boyd §3.4.1)."""
    if pres > cfg.mu_res * dres:
        return min(rho * cfg.tau_incr, cfg.rho_max)
    if dres > cfg.mu_res * pres:
        return max(rho / cfg.tau_incr, cfg.rho_min)
    return rho


def _compute_profit(
    cfg: GDPConfig,
    pop: FspPopulation,
    K_DOBLE: np.ndarray,
    eta_k: np.ndarray,
    c_max_val: float,
    q_k: np.ndarray,
    c_k: np.ndarray,
) -> float:
    """Calcula el profit esperado del agregador (ex-post)."""
    rev_avail = cfg.p_av * eta_k.mean() - cfg.p_res * c_max_val
    exp_act   = 0.0
    for sp in range(len(cfg.OMEGA_PLAGE)):
        local_pos = np.array([np.where(K_DOBLE == k)[0][0]
                               for k in pop.K_PLAGE[sp]])
        q_sp  = q_k[local_pos]
        c_sp  = c_k[local_pos]
        exp_act += cfg.OMEGA_PLAGE[sp] * (
            cfg.p_act * np.sum(q_sp)
            - cfg.p_CLC * np.sum(c_sp)
            - cfg.p_dev * max(0.0, eta_k.mean() - np.sum(q_sp) / (len(q_sp) * cfg.dt))
        )
    return rev_avail + exp_act


def _record_history(
    hist: ADMMHistory,
    pres: float, dres: float,
    lam: np.ndarray, profit: float,
    q_k: np.ndarray, c_k: np.ndarray,
    sum_Ft_new_k: np.ndarray, rho: float,
    x_bar: np.ndarray, c_max_val: float,
    eta_k: np.ndarray,
) -> None:
    hist.res_primal.append(pres)
    hist.res_dual.append(dres)
    hist.lam.append(lam.copy())
    hist.profit.append(profit)
    hist.q.append(q_k.copy())
    hist.c.append(c_k.copy())
    hist.F_tot.append(sum_Ft_new_k.copy())
    hist.rho.append(rho)
    hist.x_bar.append(np.linalg.norm(x_bar))
    hist.c_max.append(c_max_val)
    hist.eta.append(eta_k.copy())


def _benchmark_parallelism(
    gdict: dict,
    lam_fsp: np.ndarray,
    K: int,
    rho: float,
    n_workers: int,
) -> None:
    """Mide el speedup real del paralelismo con 3 FSPs de prueba."""
    t_seq = 0.0
    for i in range(3):
        t0 = time.time()
        solve_fsp_local(i, lam_fsp, np.zeros(K), rho, gdict)
        t_seq += time.time() - t0
    print(f'  Benchmark: secuencial={t_seq:.2f}s ({t_seq/3:.2f}s/FSP)')


def _print_header(
    J: int, rho: float, K_full: int, admm_cfg: ADMMConfig,
    lam_init: float, f_frac: float, p_act: float,
    F_all: np.ndarray, pi_k: np.ndarray, pop: FspPopulation,
) -> None:
    K_AM_len = len(pop.K_AM)
    print(f'  Boyd §7.3.2 Exchange  |  J={J}  |  ρ₀={rho:.4f}  |  K_full={K_full}')
    print(f'  λ₀={lam_init:.5f} ({100*lam_init/p_act:.0f}%·p_act)'
          f'  |  F₀={f_frac:.0%}·F_cap'
          f'  |  μ={admm_cfg.mu_res}  τ={admm_cfg.tau_incr}')
    print(f'  π_k AM: {pi_k[:K_AM_len].mean():.3f}'
          f'  π_k PM: {pi_k[K_AM_len:].mean():.3f}')
    print()
    hdr = (f"{'Iter':>6} | {'||r_p||':>10} | {'||r_d||':>10} | "
           f"{'Profit':>10} | {'λ_mean':>9} | {'ΣF_mean':>9} | "
           f"{'q_mean':>8} | {'η_mean':>8} | {'C_max':>7} | {'ρ':>7}")
    print(hdr)
    print('-' * len(hdr))


def _print_iter(
    n: int, pres: float, dres: float, profit: float,
    lam: np.ndarray, sum_Ft: np.ndarray, q_k: np.ndarray,
    eta_k: np.ndarray, c_max_val: float, rho: float,
    elapsed: float, t_fsp: float,
) -> None:
    print(
        f'{n:>6} | {pres:>10.5f} | {dres:>10.5f} | '
        f'{profit:>10.4f} | {lam.mean():>9.5f} | '
        f'{sum_Ft.mean():>9.4f} | {q_k.mean():>8.4f} | '
        f'{eta_k.mean():>8.3f} | {c_max_val:>7.3f} | {rho:>7.4f}'
        f'  [{elapsed:.0f}s|FSP:{t_fsp:.1f}s]'
    )
