# CLAUDE.md — GDP Bloc 1 · Exchange ADMM Two-Stage Stochastic

## Project Overview

Optimization model for Hydro-Québec's **GDP** (*Gestion de la Demande de Puissance*) demand-response program. Coordinates **N residential FSPs** (electric heating thermostats) through an **Aggregator** using a **Two-Stage Stochastic Exchange ADMM** algorithm.

- **Language**: Python 3.12
- **Solvers**: CVXPY + Clarabel (aggregator/baseline), ECOS (FSP workers)
- **Parallelism**: `ProcessPoolExecutor` with up to 96 workers for FSP subproblems

## Run

```bash
python main.py
```

Results are saved to `results/gdp_admm_N{N}_S{S}.pkl`.

## Key Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| N | 60 | FSPs residenciales |
| T | 96 | Periodos/día (15 min) |
| K | 28 | Periodos de flexibilidad (K_AM ∪ K_PM) |
| S | 10 | Escenarios estocásticos |
| ρ₀ | 0.05 | Penalización ADMM inicial |
| ε_p, ε_d | 1e-3 | Tolerancias de convergencia |

## Project Structure

```
exchange-admm-gdp/
├── gdp_pkg/
│   ├── config.py           # GDPConfig + ADMMConfig (dataclasses)
│   ├── profiles.py         # Perfiles NSL y temperatura exterior
│   ├── population.py       # Factory de población de FSPs heterogéneos
│   ├── baseline.py         # Solver QP de baselines (Builder)
│   ├── fsp_worker.py       # Subproblema CVXPY por FSP (Strategy)
│   ├── aggregator.py       # Subproblema CVXPY del Agregador (Facade)
│   ├── admm.py             # Loop Exchange ADMM (Template Method)
│   └── vss/                # VSS sub-paquete (Clean Architecture)
│       ├── __init__.py     # API pública: compute_vss, VSSResult, DetEquivSolution
│       ├── domain.py       # Value Objects + fórmulas puras (sin CVXPY)
│       ├── ev_adapter.py   # Adapter: colapsa S→1, corre ADMM determinístico
│       └── use_case.py     # Caso de uso: orquesta EV solve + evaluación EEV
└── main.py                 # Entry point (Facade)
```

## Pipeline (main.py)

| Paso | Función | Descripción |
|------|---------|-------------|
| 1 | `build_profiles()` | Perfiles temperatura y NSL (96 periodos, S escenarios) |
| 2 | `build_population()` | Población heterogénea de N FSPs |
| 3 | `solve_baselines()` | Baselines QP + headroom de flexibilidad por FSP |
| 4 | `run_exchange_admm()` | Exchange ADMM estocástico (S=10) → `ADMMResult` |
| 5 | `compute_vss()` | VSS = Profit_SP − Profit_EEV via problema EV (S=1) |

## Algorithm Summary

**Exchange ADMM** (Boyd §7.3.2) with J = N + 1 agents, Gauss-Seidel variant:

1. **Aggregator step** — fixes ΣF̃ᵢ, updates (q−c) via QP
2. **FSP step (parallel)** — each FSP i solves its thermal QP independently
3. **Dual update** — λ ← λ + ρ · [(q−c) − ΣF̃ᵢ]

**Adaptive ρ** (Boyd §3.4.1): scales by τ=1.2 when ‖r_p‖ > μ·‖r_d‖, or divides when ‖r_d‖ > μ·‖r_p‖ (μ=20).

**Stochastic scenarios** (S_plage = 3): AM only (ω=0.55), PM only (ω=0.11), AM+PM (ω=0.34).

## VSS Module (`gdp_pkg/vss/`)

Implements the **Value of the Stochastic Solution** following Clean Architecture (no DDD). Dependencies flow inward only: `ev_adapter` → `use_case` → `domain`.

```
VSS = Profit_SP − Profit_EEV  (≥ 0 always)
```

### Layers

| Capa | Archivo | Responsabilidad |
|------|---------|-----------------|
| Domain | `domain.py` | `DetEquivSolution`, `VSSResult` (frozen dataclasses); `compute_profit()`, `compute_eev_profit()` — solo numpy |
| Use Case | `use_case.py` | `compute_vss()` — orquesta los pasos 1→3 |
| Adapter | `ev_adapter.py` | `solve_ev_problem()` — colapsa S=10→1 con `dataclasses.replace()`, corre ADMM |

### Algoritmo

1. **Profit_SP** — extraído directamente de `ADMMResult` (ya resuelto)
2. **Problema EV** — colapsa escenarios de incertidumbre a su media (S=1); OMEGA_PLAGE (3 GDP scenarios) permanece intacto; resuelve `run_exchange_admm()` determinístico → `(η_ev, c_max_ev, F_ev_all)`
3. **Profit_EEV** — recourse cerrado: `gap = η_ev·dt − F_ev_sum_k`, `c_ev = clip(gap, 0, c_max_ev)`, `q_ev = F_ev_sum_k + c_ev`; evalúa bajo los 3 escenarios GDP originales

### Key invariant

`OMEGA_PLAGE` (escenarios de activación GDP) **no se modifica** en el EV. Solo se colapsan las fuentes de incertidumbre climática/NSL (S=10 → S=1).

## Custom Skills Available

Three domain-specific skills are configured in `.claude/skills/`. Use them with the `/` slash commands or invoke via the Skill tool when relevant.

### `/distributedoptimizationexpert`
**When to use**: Designing or debugging the ADMM decomposition, modifying coupling constraints, analyzing convergence of the Exchange ADMM, tuning ρ, formulating new subproblems (LP/QP/SOCP), or checking DCP convexity rules in CVXPY.

Slash commands:
- `/formulate` — Convert a conceptual problem into rigorous LaTeX
- `/decompose` — Break down a centralized model into a distributed algorithm
- `/solve` — Generate CVXPY implementation
- `/relax` — Identify non-convexities and suggest convex relaxations

### `/uncertaintyexpert`  
**When to use**: Modifying scenario generation (baseline forecast error, outdoor temperature), adding risk measures (CVaR), scenario reduction, multi-stage extensions, computing EVPI/VSS, or handling non-anticipativity constraints.

Slash commands:
- `/tree` — Design a scenario tree for multi-stage problems
- `/recourse` — Formulate the second-stage subproblem and Benders cuts
- `/risk` — Transform risk-neutral → risk-averse (CVaR / Mean-Variance)
- `/metrics` — Compute EVPI and VSS for the current model

### `/mathematician`
**When to use**: Complexity analysis of the ADMM loop, numerical stability of the augmented Lagrangian, verifying matrix conditioning for the QP solvers, or selecting numerical methods.

## Domain Glossary

| Term | Meaning |
|------|---------|
| FSP | Flexible Service Provider — residential electric heating thermostat |
| Aggregator | Entity that bids FSP flexibility into the GDP program |
| GDP | Gestion de la Demande de Puissance — HQ demand-response program |
| Pointe | Winter peak event (~5 days/year, 120 h total, Jan) |
| CLC | Charge Limitée Compensée — industrial backup capacity |
| K_AM / K_PM | Flexibility windows: 06–09 h / 16–20 h at 15-min resolution |
| η | Declared capacity [kW] (first-stage, non-anticipative) |
| F̃ᵢˢ | FSP i's flexibility offer in scenario s [kWh/period] |
| λ | Dual variable (clearing price) [CAD/period] |
| VSS | Value of the Stochastic Solution = Profit_SP − Profit_EEV |
| EV | Expected Value problem — deterministic (S=1) version of the stochastic model |
| EEV | Expected value of the EV solution — profit when EV decisions face real scenarios |
| EVPI | Expected Value of Perfect Information = Profit_WS − Profit_SP |

## Financial Model

```
Revenue = p_av · η − p_res · c_max
        + Σ_s ω_s · Σ_k [p_act · q_k^s − p_CLC · c_k^s − p_dev · r_k^s]
```

Where `p_dev = 5 × p_act` (shortfall penalty) and `p_CLC = 1.5 × p_act` (CLC backup cost).

## Dependencies

```bash
pip install cvxpy clarabel ecos numpy scipy
```
