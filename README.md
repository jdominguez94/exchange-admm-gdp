# GDP Bloc 1 — Exchange ADMM Two-Stage Stochastic

Optimization model for Hydro-Québec's **GDP** (*Gestion de la Demande de Puissance*) demand-response program using a **Two-Stage Stochastic Exchange ADMM** algorithm.

---

## Overview

The **GDP** program is a winter peak-shaving mechanism operated by Hydro-Québec (Québec, Canada). During *Pointe* events (typically January, ~5 days/year, 120 h total), residential customers reduce their electric heating consumption in exchange for a seasonal capacity credit.

This model coordinates **N = 60 residential FSPs** (Flexible Service Providers — electric heating thermostats) through an **Aggregator**, who bids into the GDP program on their behalf. Uncertainty is captured via three **GDP activation scenarios** (AM only, PM only, AM+PM) derived from 37 unique Pointe events between December 2024 and March 2026.

---

## Mathematical Formulation

### Notation

| Symbol | Description | Units |
|--------|-------------|-------|
| $i \in \mathcal{N}$ | FSP index, $N = 60$ | — |
| $t \in \mathcal{T}$ | Time step (15 min), $T = 96$ | — |
| $k \in \mathcal{K}$ | Flexibility period, $K = 28$ | — |
| $s \in \mathcal{S}_\text{plage}$ | GDP activation scenario | — |
| $\omega^s$ | Scenario probability | — |
| $dt$ | Period duration | 0.25 h |
| $\pi_t$ | FSP electricity price | CAD/kWh |
| $p_\text{av}$ | GDP availability credit (once/season) | CAD/kW |
| $p_\text{act}$ | Activation payment | CAD/kWh |
| $p_\text{dev}$ | Shortfall penalty | CAD/kWh |
| $p_\text{CLC}$ | Industrial backup (CLC) cost | CAD/kWh |

### Flexibility Windows

The GDP program activates during two daily windows at 15-min resolution:

$$\mathcal{K}_\text{AM} = \{t : t \in [06{:}00, 09{:}00)\} \quad (12 \text{ periods})$$
$$\mathcal{K}_\text{PM} = \{t : t \in [16{:}00, 20{:}00)\} \quad (16 \text{ periods})$$
$$\mathcal{K} = \mathcal{K}_\text{AM} \cup \mathcal{K}_\text{PM} \quad (K = 28)$$

Three activation scenarios with empirical probabilities from HQ data:

| Scenario | Window | $\omega^s$ |
|----------|--------|-----------|
| $s=0$ | AM only | 0.55 |
| $s=1$ | PM only | 0.11 |
| $s=2$ | AM + PM | 0.34 |

---

### FSP Thermal Model

Each FSP $i$ is an electric heating system modeled as a **first-order linear thermal system**:

$$x^s_{i,t+1} = a_i \, x^s_{i,t} + b_i \, u_{i,t} + c_i \, T^\text{out,s}_t$$

where $x_{i,t}$ is indoor temperature [°C], $u_{i,t}$ is heating power [kW], and $T^\text{out}_t$ is outdoor temperature [°C]. Parameters $(a_i, b_i, c_i)$ are sampled from truncated Gaussian distributions to capture heterogeneity.

### FSP Subproblem (Equations 9–15)

Each FSP $i$ solves independently (in parallel):

$$\min_{u_i, F_i, \tilde{F}^s_i, x^s_i} \quad \pi_t \sum_t u_{i,t} \cdot dt + \frac{1}{S} \sum_s \sum_t \alpha_i(t) \left(x^s_{i,t} - x^\text{ref}_i\right)^2$$

$$- \lambda_k^\top F_i + \frac{\rho}{2} \sum_s \omega_s \, \left\| \tilde{F}^s_i - \sigma_i \right\|^2$$

**Subject to:**

| Constraint | Description |
|-----------|-------------|
| $F_{i,k} = (\bar{u}^\text{base}_{i,t(k)} - u_{i,t(k)}) \cdot dt$ | Linking (1st stage) |
| $F_{i,k} \leq F^s_{\text{cap},i,k} \quad \forall s$ | Physical headroom |
| $0 \leq u_{i,t} \leq \min(u^\text{max}_i, C^\text{total}_i - u^\text{nsl,s}_{i,t})$ | Capacity |
| $x^s_{i,t+1} = a_i x^s_{i,t} + b_i u_{i,t} + c_i T^{\text{out},s}_t$ | Thermal dynamics |
| $x^\text{min}_i \leq x^s_{i,t} \leq x^\text{max}_i, \quad x^s_{i,0} = x^\text{ref}_i$ | Comfort bounds |
| $\tilde{F}^s_{i,k} = F_{i,k} \quad \forall s$ | Non-anticipativity |

**Headroom** (flexibility capacity):
$$F^\text{cap}_{i,k} = u^\text{base}_{i,t(k)} \cdot dt \quad [\text{kWh}]$$

**Thermal discomfort coefficient** $\alpha_i(t)$: time-varying across four behavioral groups:
- **Group A**: Rigid at morning/evening peaks $[06, 08) \cup [17, 20)$
- **Group B**: Extended rigidity $[07, 10) \cup [17, 22)$
- **Group C**: Price-comparable (low scale, marginal)
- **Group D**: Constant throughout the day (50% high / 50% low)

The **indifference threshold**:
$$\theta = \theta_s \cdot \frac{p_\text{act}}{b_\text{mean} \cdot \Delta x_\text{mean} \cdot 2dt}$$

FSP $i$ participates at period $k$ only if $\alpha_i(t_k) < \theta$.

---

### Aggregator Subproblem (Equations 1–8)

The Aggregator solves a **two-stage stochastic program** over the $S_\text{plage} = 3$ GDP scenarios:

**First stage** (here-and-now, non-anticipative):
- $\eta$ — declared capacity to GDP operator [kW]
- $c^\text{max}$ — contracted CLC backup capacity [kWh/period]

**Second stage** (recourse, per scenario $s$):
- $r^s_k$ — actual shortfall [kWh] after FSPs + CLC
- $c^s_k$ — CLC activation [kWh]

$$\max_{\eta, c^\text{max}, r^s, c^s} \quad (p_\text{av} - \gamma)\,\eta - p_\text{res}\,c^\text{max}$$

$$+ \sum_{s} \omega^s \sum_{k \in \mathcal{K}^s} \left[ p_\text{act} \, q^s_k - p_\text{CLC} \, c^s_k - p_\text{dev} \, r^s_k - \lambda_k c^s_k - \frac{\rho}{2} \left\| q^s_k - \sigma^s_k \right\|^2 \right]$$

where $q^s_k = \sum_i \tilde{F}^s_{i,k} + c^s_k$ (total delivery).

**Subject to:**

| Constraint | Description |
|-----------|-------------|
| $\eta_\text{min} \leq \eta \leq \eta^\text{eff}_\text{max}$ | Capacity bounds |
| $c^s_k \leq c^\text{max}$ | CLC bounded by contract |
| $r^s_k \geq \eta \cdot dt - \sum_i \tilde{F}^s_{i,k} - c^s_k$ | Shortfall definition |
| $\sum_i \tilde{F}^s_{i,k} + c^s_k \leq \eta \cdot dt$ | No over-delivery |

---

### Coupling Constraint (Equation 17)

$$q^s_k - c^s_k = \sum_i \tilde{F}^s_{i,k} \quad \forall k \in \mathcal{K}^s, \; \forall s$$

This **market-clearing** condition couples the aggregator's delivery commitment with the sum of individual FSP flexibility offers.

---

## Exchange ADMM Algorithm (Boyd §7.3.2)

The coupling constraint is enforced via the **Exchange ADMM** (Alternating Direction Method of Multipliers) with $J = N + 1$ agents.

### Augmented Lagrangian

The original problem decomposes into $J$ independent subproblems by introducing the augmented dual term:

$$\mathcal{L}_\rho = \sum_j \Pi_j(x_j) + \lambda^\top \left( \sum_j A_j x_j - b \right) + \frac{\rho}{2} \left\| \sum_j A_j x_j - b \right\|^2$$

### Update Rules (Gauss-Seidel variant)

**Convention** (paper): $\bar{x} = \frac{1}{J}\left[(q-c) - \sum_i \tilde{F}_i\right]$, positive when demand exceeds supply.

**Step 1 — Aggregator** (fixes $\sum \tilde{F}^n$, updates $(q-c)$):
$$\bar{x}^n = \frac{1}{J}\left[(q-c)^n - \textstyle\sum_i \tilde{F}^n_i\right]$$
$$\sigma^{0,s}_k = \frac{J-1}{J}(r^s+c^s)_k^n - \frac{1}{J}\textstyle\sum_i \tilde{F}^n_{i,k}, \quad k \in \mathcal{K}^s$$
$$(q, r, c, \eta)^{n+1} \leftarrow \underset{q,r,c,\eta}{\arg\max} \; \Pi_\text{agg} + \lambda^\top(q-c) - \tfrac{\rho}{2}\sum_s\omega^s\|(q-c)^s - \sigma^{0,s}\|^2$$

**Step 2a** — Recompute residual:
$$\bar{x}^{n+\frac{1}{2}} = \frac{1}{J}\left[(q-c)^{n+1} - \textstyle\sum_i \tilde{F}^n_i\right]$$

**Step 2b — FSPs** (in parallel):
$$\sigma_{i,k} = \tilde{F}^{s,n}_{i,k} + \bar{x}^{n+\frac{1}{2}}_k$$
$$\tilde{F}^{n+1}_i \leftarrow \underset{F, \tilde{F}^s, u, x^s}{\arg\min} \; \Pi_i - \lambda^\top F + \frac{\rho}{2}\sum_s\omega_s\|\tilde{F}^s_i - \sigma_i\|^2$$

> When demand > supply: $\bar{x} > 0 \Rightarrow \sigma_i > \tilde{F}^n_i$ → proximal pushes FSPs to offer **more** ✓
>
> When supply > demand: $\bar{x} < 0 \Rightarrow \sigma_i < \tilde{F}^n_i$ → proximal pushes FSPs to offer **less** ✓

**Step 3 — Dual update** (≡ Eq. 22):
$$\bar{x}^{n+1} = \frac{1}{J}\left[(q-c)^{n+1} - \textstyle\sum_i \tilde{F}^{n+1}_i\right]$$
$$\lambda^{n+1} = \lambda^n + \rho \cdot \left[(q-c)^{n+1} - \textstyle\sum_i \tilde{F}^{n+1}_i\right]$$

### Convergence Criteria

$$\|r_p\| = \left\|\mathbb{E}\left[(q-c)^{n+1} - \textstyle\sum_i\tilde{F}^{n+1}_i\right]\right\|_2 < \varepsilon_p \quad \text{(primal residual)}$$
$$\|r_d\| = \rho \cdot \left\|\textstyle\sum_i\tilde{F}^{n+1}_i - \textstyle\sum_i\tilde{F}^n_i\right\|_2 < \varepsilon_d \quad \text{(dual residual)}$$

### Adaptive Penalty (Boyd §3.4.1)

$$\rho^{n+1} = \begin{cases} \min(\tau \rho^n, \rho_\max) & \text{if } \|r_p\| > \mu \|r_d\| \\ \max(\rho^n / \tau, \rho_\min) & \text{if } \|r_d\| > \mu \|r_p\| \\ \rho^n & \text{otherwise} \end{cases}$$

Default: $\mu = 20$, $\tau = 1.2$, $\rho_0 = 0.05$.

---

## Stochastic Uncertainty

Two sources of stochasticity are modeled with $S = 10$ equiprobable scenarios:

**Baseline forecast error** (log-normal multiplicative noise):
$$u^\text{base,s}_{i,t} = u^\text{base}_{i,t} \cdot \varepsilon^s_i, \quad \varepsilon^s_i \sim \text{LogNormal}\!\left(-\tfrac{\sigma^2}{2}, \sigma^2\right), \quad \sigma = 3\%$$

The log-normal parameterization ensures $\mathbb{E}[\varepsilon^s_i] = 1$ (unbiased).

**Outdoor temperature** (additive Gaussian):
$$T^{\text{out},s}_t = T^\text{out}_t + \varepsilon^s_t, \quad \varepsilon^s_t \sim \mathcal{N}\!\left(0, \sigma_T(t)^2\right), \quad \sigma_T(t) = 0.5 + 0.15\,h_t$$

where $h_t$ is the hour of day, yielding larger uncertainty in the afternoon.

**NSL** (Non-Shiftable Load, log-normal):
$$u^\text{nsl,s}_{i,t} = u^\text{nsl}_{i,t} \cdot \exp\!\left(\mu^\text{ln}_t + \sigma^\text{nsl}_t \cdot \xi^s_{i,t}\right), \quad \xi^s_{i,t} \sim \mathcal{N}(0,1)$$

---

## Project Structure

```
gdp/
├── gdp_pkg/
│   ├── __init__.py        # Public API
│   ├── config.py          # GDPConfig + ADMMConfig (Dataclass pattern)
│   ├── profiles.py        # NSL & temperature profiles (Static data)
│   ├── population.py      # FSP population factory (Factory pattern)
│   ├── baseline.py        # Baseline QP solver (Builder pattern)
│   ├── fsp_worker.py      # FSP CVXPY subproblem (Strategy + Pool)
│   ├── aggregator.py      # Aggregator CVXPY subproblem (Facade)
│   └── admm.py            # Exchange ADMM loop (Template Method)
├── main.py                # Entry point (Facade)
└── README.md
```

### Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Dataclass** | `GDPConfig`, `ADMMConfig` | Centralize all parameters; prevents scattered magic numbers |
| **Factory** | `build_population()` | Constructs heterogeneous FSP population in one call |
| **Builder** | `solve_baselines()` | Builds N QP problems, solves them, assembles result |
| **Strategy** | `FspWorker` | Each FSP solves its own independent CVXPY subproblem |
| **Template Method** | `run_exchange_admm()` | Fixed ADMM loop structure; subproblems are injected calls |
| **Facade** | `aggregator.py`, `main.py` | Simple interface hiding CVXPY complexity |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
cvxpy >= 1.4
clarabel    (default solver for aggregator and baseline)
ecos        (solver for FSP workers — faster for small QPs)
```

Install:
```bash
pip install cvxpy clarabel ecos numpy scipy
```

Run:
```bash
python main.py
```

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N$ | 60 | Number of FSPs |
| $T$ | 96 | Time steps/day (15 min) |
| $K$ | 28 | Flexibility periods ($\mathcal{K}_\text{AM} \cup \mathcal{K}_\text{PM}$) |
| $S$ | 10 | Stochastic scenarios |
| $p_\text{av}$ | 5.50 CAD/kW | GDP availability credit (winter) |
| $p_\text{act}$ | 0.52 CAD/kWh | Activation payment |
| $p_\text{dev}$ | 2.60 CAD/kWh | Shortfall penalty ($5 \times p_\text{act}$) |
| $p_\text{CLC}$ | 0.78 CAD/kWh | CLC backup cost ($1.5 \times p_\text{act}$) |
| $\eta_\text{max}$ | 100 kW | Max declared capacity |
| $\rho_0$ | 0.05 | Initial ADMM penalty |
| $\varepsilon_p, \varepsilon_d$ | $10^{-3}$ | Convergence tolerances |

---

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., Eckstein, J. (2011). *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Foundations and Trends in Machine Learning. §3.4.1 (adaptive ρ), §7.3 (exchange problem).
- Hydro-Québec. *Programme GDP Bloc 1* — tarification interruptible résidentielle, données 2024–2026.
- Diamond, S., Boyd, S. (2016). *CVXPY: A Python-embedded modeling language for convex optimization*. JMLR.
