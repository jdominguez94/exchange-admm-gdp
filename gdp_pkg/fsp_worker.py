"""
fsp_worker.py — Subproblema CVXPY del FSP en el Exchange ADMM (Boyd §7.3.2).

Patrón: Strategy (cada FSP resuelve su propio subproblema de forma independiente).
        El paralelismo se implementa con ProcessPoolExecutor.

Variable de decisión Boyd:  x_i = F̃^s_{i,k}  (oferta de flexibilidad [kWh])

Problema de minimización del FSP i en la iteración n:
  min  π_t·Σu_t  +  (1/S)·Σ_s Σ_t α_i(t)·(x^s_{i,t} - x_ref_i)²
       - λ_k · F_{i,k}
       + (ρ/2) · Σ_s ω_s · ‖F̃^s_{i,k} - σ_{i,k}‖²
  s.t.
    [Linking 1ª etapa]  F_{i,k} = (ū^base_{i,t(k)} - u_{i,t(k)}) · dt
    [Headroom]          F_{i,k} ≤ F^s_cap_{i,k}  ∀s
    [Capacidad]         0 ≤ u_t  ≤ min(u_max_i, C_total_i - u_nsl^s_{i,t})  ∀s
    [Dinámica]          x^s_{t+1} = a_i·x^s_t + b_i·u_t + c_i·T^s_out_t    ∀s
    [Temperatura]       x_min_i ≤ x^s_t ≤ x_max_i,  x^s_0 = x_ref_i        ∀s
    [No-anticipatividad] F̃^s_{i,k} = F_{i,k}  ∀s

Signal proximal (Boyd §7.3.2):
    σ_{i,k} = F̃^{s,n}_{i,k} + x̄^{n+½}
    x̄^{n+½} = (1/J)·[(q-c)^{n+1} - Σ_i F̃^{s,n}_i]

Señal de precio:  λ̃_k = π_k · λ_k   (precio esperado ponderado por P(k activo))
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp

# Estado global del worker (inicializado por _init_worker en el proceso hijo)
_worker_globals: dict  = {}
_worker_problems: dict = {}


class FspWorker:
    """Problema CVXPY pre-compilado para el FSP i.

    El problema se construye una sola vez (``__init__``) y se re-usa en cada
    iteración ADMM cambiando únicamente los valores de los parámetros CVXPY
    (lam_p, sigma_p, rho_p). Esto evita recompilar el grafo en cada iteración.
    """

    def __init__(self, i: int, g: dict) -> None:
        """
        Parameters
        ----------
        i : índice del FSP en la población
        g : diccionario de globals generado por ``make_worker_globals``
        """
        T_      = g['T'];       K_      = g['K']
        K_idx_  = g['K_idx'];   S_      = g['S']
        dt_     = g['dt'];      pi_t_   = g['pi_t']
        alpha_i  = g['alpha_pop'][i]          # (T,) — varía por hora
        ai       = float(g['a_pop'][i])
        bi       = float(g['b_pop'][i])
        ci_      = float(g['c_pop'][i])
        x_ref_i  = float(g['x_ref_pop'][i])
        x_min_i  = float(g['x_min_pop'][i])
        x_max_i  = float(g['x_max_pop'][i])
        F_cap_scen_i  = g['F_cap_scenarios'][i]    # (S, K)
        u_base_scen_i = g['u_base_scenarios'][i]   # (S, T)
        u_nsl_s       = g['u_nsl_scenarios'][i]    # (S, T)
        T_out_s       = g['T_out_scenarios']        # (S, T)
        C_tot_i  = float(g['C_total_pop'][i])
        omega_   = g['omega']                       # (S,)

        # ── Variables de decisión ──────────────────────────────────────
        self.u         = cp.Variable(T_,        nonneg=True)   # consumo [kW]
        self.F         = cp.Variable(K_,        nonneg=True)   # oferta 1ª etapa [kWh]
        self.F_tilde_s = cp.Variable((S_, K_), nonneg=True)   # oferta 2ª etapa [kWh]
        self.x_s       = [cp.Variable(T_ + 1) for _ in range(S_)]  # temperatura [°C]

        # ── Parámetros ADMM (actualizados en cada iteración) ──────────
        self.lam_p   = cp.Parameter(K_)
        self.sigma_p = cp.Parameter(K_)    # señal proximal determinística
        self.rho_p   = cp.Parameter(nonneg=True)

        # ── Objetivo ──────────────────────────────────────────────────
        energy = (pi_t_ * cp.sum(self.u)
                  + pi_t_ * float(u_nsl_s.mean(axis=0).sum()))

        # Incomodidad térmica con α(t) variable: Σ_s ω_s · Σ_t α_i(t) · (x^s_{t+1} - x_ref)²
        discomfort = (1.0 / S_) * cp.sum([
            cp.sum(cp.multiply(alpha_i, cp.power(self.x_s[s][1:] - x_ref_i, 2)))
            for s in range(S_)
        ])

        # Ingreso de flexibilidad: -λ · F  (primera etapa, no-anticipativa)
        linear_dual = self.lam_p @ self.F

        # Proximal Boyd: (ρ/2) · Σ_s ω_s · ‖F̃^s_i - σ‖²
        proximal = (self.rho_p / 2) * cp.sum([
            omega_[s] * cp.sum_squares(self.F_tilde_s[s, :] - self.sigma_p)
            for s in range(S_)
        ])

        # ── Restricciones ─────────────────────────────────────────────
        cons = []

        # Linking: F_{i,k} = (E_s[u^base_{i,t(k)}] - u_{i,t(k)}) · dt
        u_base_mean_k = u_base_scen_i.mean(axis=0)  # (T,) media sobre S
        for k in range(K_):
            cons.append(
                self.F[k] == (u_base_mean_k[K_idx_[k]] - self.u[K_idx_[k]]) * dt_
            )

        # Headroom factible en todos los escenarios
        for s in range(S_):
            cons.append(self.F <= F_cap_scen_i[s, :])

        # Límite de potencia
        cons.append(self.u <= x_max_i)   # límite físico relajado (x_max proxy)
        cons.append(self.u <= float(g['u_max_pop'][i]))
        for s in range(S_):
            cons.append(self.u <= C_tot_i - u_nsl_s[s])

        # No-anticipatividad: F̃^s_{i,k} = F_{i,k} ∀s
        for s in range(S_):
            cons.append(self.F_tilde_s[s, :] == self.F)

        # Dinámica térmica y límites por escenario
        for s in range(S_):
            cons += [
                self.x_s[s][0]  == x_ref_i,
                self.x_s[s][1:] >= x_min_i,
                self.x_s[s][1:] <= x_max_i,
            ]
            cons.append(
                self.x_s[s][1:] == (
                    ai * self.x_s[s][:-1]
                    + bi * self.u
                    + ci_ * T_out_s[s]
                )
            )

        self.prob = cp.Problem(
            cp.Minimize(energy + discomfort - linear_dual + proximal),
            cons,
        )

        # Guarda para el fallback
        self._i        = i
        self._u_base_i = g['u_base'][i].copy()
        self._x_base_i = g['x_base'][i].copy()
        self._S        = S_
        self._K        = K_

    def solve(
        self,
        lam: np.ndarray,
        sigma_k: np.ndarray,
        rho: float,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resuelve el subproblema ADMM con los parámetros actualizados.

        Returns
        -------
        (i, u_val, x_mean, F_val, F_tilde_val)
        """
        self.lam_p.value   = lam
        self.sigma_p.value = sigma_k
        self.rho_p.value   = float(rho)

        self.prob.solve(solver=cp.ECOS, verbose=False)

        if self.prob.status not in ('optimal', 'optimal_inaccurate'):
            # Fallback: mantener baseline
            return (self._i,
                    self._u_base_i.copy(),
                    self._x_base_i.copy(),
                    np.zeros(self._K),
                    np.zeros((self._S, self._K)))

        x_mean      = np.mean([self.x_s[s].value[1:] for s in range(self._S)], axis=0)
        F_val       = np.maximum(0.0, self.F.value)
        F_tilde_val = np.maximum(0.0, self.F_tilde_s.value)
        return self._i, self.u.value, x_mean, F_val, F_tilde_val


# ── Interfaz de pool de procesos ───────────────────────────────────────────

def make_worker_globals(
    cfg,
    pop,
    baseline,
) -> dict:
    """Construye el diccionario de globals pasado a los procesos hijos.

    Contiene únicamente arrays numpy serializables (no objetos CVXPY).
    """
    return dict(
        N=cfg.N, T=cfg.T, K=len(pop.K_idx), K_idx=pop.K_idx,
        S=cfg.S, dt=cfg.dt, pi_t=cfg.pi_t,
        alpha_pop=pop.alpha_pop,
        group_labels=pop.group_labels,
        a_pop=pop.a_pop, b_pop=pop.b_pop, c_pop=pop.c_pop,
        u_max_pop=pop.u_max_pop, x_ref_pop=pop.x_ref_pop,
        x_min_pop=pop.x_min_pop, x_max_pop=pop.x_max_pop,
        u_base=baseline.u_base_heat,
        x_base=baseline.x_base,
        F_cap=baseline.F_cap,
        u_base_scenarios=baseline.u_base_scenarios,
        F_cap_scenarios=baseline.F_cap_scenarios,
        u_nsl_scenarios=pop.u_nsl_scenarios,
        T_out_scenarios=pop.T_out_scenarios,
        C_total_pop=pop.C_total_pop,
        omega=pop.omega,
    )


def _init_worker(globals_dict: dict) -> None:
    """Inicializador del proceso hijo: guarda globals y limpia caché de problemas."""
    global _worker_globals, _worker_problems
    _worker_globals  = globals_dict
    _worker_problems = {}


def _solve_fsp_worker(
    args: tuple[int, np.ndarray, np.ndarray, float],
) -> tuple:
    """Función ejecutada en el proceso hijo (compatible con pickle)."""
    i, lam, sigma_k, rho = args
    if i not in _worker_problems:
        _worker_problems[i] = FspWorker(i, _worker_globals)
    return _worker_problems[i].solve(lam, sigma_k, rho)


def solve_fsp_local(
    i: int,
    lam: np.ndarray,
    sigma_k: np.ndarray,
    rho: float,
    gdict: dict,
) -> tuple:
    """Resuelve el FSP i en el proceso actual (modo secuencial/benchmark)."""
    if i not in _worker_problems:
        _worker_problems[i] = FspWorker(i, gdict)
    return _worker_problems[i].solve(lam, sigma_k, rho)
