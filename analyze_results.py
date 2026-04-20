"""
analyze_results.py — Análisis visual de los resultados del modelo GDP Bloc 1.

Lee todos los archivos .pkl en results/ y genera figuras en PDF.

Convenciones de tamaño:
  - Figura individual    : (7, 3) inches
  - Figura multi-panel   : ancho 7 inches (doble columna IEEE), alto según filas
"""

import os
import glob
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Estilo global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'lines.linewidth':   1.2,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_DIR = Path('results')
FIGS_DIR    = RESULTS_DIR / 'figures'

# Tamaños IEEE
W_SINGLE = 7.0   # doble columna IEEE [inches]
H_ROW    = 2.2   # altura por fila de subplots [inches]
W_INDIV  = 7.0
H_INDIV  = 3.0


def load_pickles(results_dir: Path) -> list[dict]:
    """Carga todos los .pkl en results_dir."""
    files = sorted(results_dir.glob('*.pkl'))
    if not files:
        print(f'No se encontraron archivos .pkl en {results_dir}')
        return []
    data = []
    for f in files:
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
            d['_fname'] = f.stem
            data.append(d)
        print(f'  Cargado: {f.name}')
    return data


# ── 1. Convergencia ADMM ───────────────────────────────────────────────────────

def plot_admm_convergence(d: dict, out_dir: Path) -> None:
    """4 paneles: residuales primal/dual, profit, ρ."""
    hist = d['result'].hist
    iters = np.arange(1, len(hist.res_primal) + 1)

    fig = plt.figure(figsize=(W_SINGLE, 4 * H_ROW / 2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — Residuales
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(iters, hist.res_primal, label=r'$\|r_p\|$', color='steelblue')
    ax1.semilogy(iters, hist.res_dual,   label=r'$\|r_d\|$', color='tomato')
    cfg = d['cfg']
    admm_cfg = d['admm_cfg']
    ax1.axhline(admm_cfg.eps_primal, ls=':', color='steelblue', lw=0.8, label=r'$\varepsilon_p$')
    ax1.axhline(admm_cfg.eps_dual,   ls=':', color='tomato',    lw=0.8, label=r'$\varepsilon_d$')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Norma residual')
    ax1.set_title('Residuales ADMM')
    ax1.legend(ncol=2)

    # Panel 2 — Profit iterativo
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iters, hist.profit, color='seagreen')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Profit [CAD]')
    ax2.set_title('Profit ex-post (iterativo)')

    # Panel 3 — η y c_max
    ax3 = fig.add_subplot(gs[1, 0])
    eta_trace  = [np.mean(e) for e in hist.eta]
    cmax_trace = hist.c_max
    ax3.plot(iters, eta_trace,  label=r'$\eta$ [kW]', color='navy')
    ax3.plot(iters, cmax_trace, label=r'$c_{\max}$ [kWh]', color='darkorange')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('kW / kWh')
    ax3.set_title(r'Primera etapa: $\eta$ y $c_{\max}$')
    ax3.legend()

    # Panel 4 — ρ adaptativo
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(iters, hist.rho, color='purple')
    ax4.set_xlabel('Iteración')
    ax4.set_ylabel(r'$\rho$')
    ax4.set_title(r'Penalización adaptativa $\rho$')

    fig.suptitle(f'Convergencia ADMM — {d["_fname"]}', fontsize=9, y=1.01)
    _save(fig, out_dir, f'{d["_fname"]}_convergencia')


# ── 2. Perfil de flexibilidad K_full ──────────────────────────────────────────

def plot_flexibility_profile(d: dict, out_dir: Path) -> None:
    """q_k, c_k, r_k y λ_k como función del índice de periodo."""
    result = d['result']
    K_full = len(result.q_k)
    idx    = np.arange(K_full)

    fig, axes = plt.subplots(1, 3, figsize=(W_SINGLE, H_INDIV))

    # q, c, r
    axes[0].bar(idx, result.q_k, color='steelblue', label='$q_k$ (entrega)')
    axes[0].bar(idx, result.c_k, color='tomato',    label='$c_k$ (CLC)', alpha=0.75)
    axes[0].bar(idx, result.r_k, color='gold',      label='$r_k$ (shortfall)', alpha=0.75)
    axes[0].set_xlabel('Período $k$')
    axes[0].set_ylabel('Energía [kWh]')
    axes[0].set_title('Despacho por período')
    axes[0].legend(fontsize=6)

    # η
    axes[1].plot(idx, result.eta_k, color='navy', lw=1.5)
    axes[1].set_xlabel('Período $k$')
    axes[1].set_ylabel('kW')
    axes[1].set_title(r'Potencia declarada $\eta_k$')

    # λ
    axes[2].plot(idx, result.lam, color='purple', lw=1.5)
    axes[2].set_xlabel('Período $k$')
    axes[2].set_ylabel('CAD/período')
    axes[2].set_title(r'Precio de liquidación $\lambda_k$')

    fig.suptitle(f'Perfil de flexibilidad — {d["_fname"]}', fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, f'{d["_fname"]}_perfil_flex')


# ── 3. Heatmap de ofertas FSP ──────────────────────────────────────────────────

def plot_fsp_heatmap(d: dict, out_dir: Path) -> None:
    """Heatmap F_all (N × K)."""
    F = d['result'].F_all         # (N, K)
    N, K = F.shape

    fig, ax = plt.subplots(figsize=(W_INDIV, H_INDIV))
    im = ax.imshow(F, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Flexibilidad ofertada [kWh]')
    ax.set_xlabel('Período de flexibilidad $k$')
    ax.set_ylabel('FSP $i$')
    ax.set_title(f'Ofertas de flexibilidad $F_{{i,k}}$ — {d["_fname"]}')
    _save(fig, out_dir, f'{d["_fname"]}_fsp_heatmap')


# ── 4. Estado térmico FSPs ─────────────────────────────────────────────────────

def plot_thermal(d: dict, out_dir: Path) -> None:
    """Perfiles de consumo u y temperatura x (media ± 1σ sobre FSPs)."""
    u = d['result'].u_all   # (N, T)
    x = d['result'].x_all   # (N, T)
    T  = u.shape[1]
    t  = np.arange(T) * 0.25   # horas

    u_mean, u_std = u.mean(0), u.std(0)
    x_mean, x_std = x.mean(0), x.std(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_SINGLE, H_INDIV))

    ax1.plot(t, u_mean, color='tomato', label='Media')
    ax1.fill_between(t, u_mean - u_std, u_mean + u_std,
                     alpha=0.25, color='tomato', label=r'$\pm1\sigma$')
    ax1.set_xlabel('Hora del día')
    ax1.set_ylabel('Consumo [kW]')
    ax1.set_title('Consumo de calefacción $u_i(t)$')
    ax1.legend()

    ax2.plot(t, x_mean, color='steelblue', label='Media')
    ax2.fill_between(t, x_mean - x_std, x_mean + x_std,
                     alpha=0.25, color='steelblue', label=r'$\pm1\sigma$')
    ax2.set_xlabel('Hora del día')
    ax2.set_ylabel('Temperatura [°C]')
    ax2.set_title(r'Temperatura interior $x_i(t)$')
    ax2.legend()

    fig.suptitle(f'Estado térmico FSPs — {d["_fname"]}', fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, f'{d["_fname"]}_termico')


# ── 5. Árbol de escenarios LMP ────────────────────────────────────────────────

def plot_scenario_tree(d: dict, out_dir: Path) -> None:
    """Visualización del árbol de escenarios LMP NY-HQ."""
    tree = d.get('lmp_tree')
    if tree is None:
        return

    cfg = d['cfg']
    scenarios = tree.lmp_scenarios
    L = len(scenarios)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_SINGLE, H_INDIV))

    # Panel izquierdo — diagrama árbol simplificado
    ax1.axis('off')
    colors_lmp = ['#2ca02c', '#ff7f0e', '#d62728']
    x_root, y_root = 0.1, 0.5

    for l, (scen, col) in enumerate(zip(scenarios, colors_lmp)):
        y_leaf = (l + 0.5) / L
        ax1.annotate(
            '',
            xy=(0.75, y_leaf), xytext=(x_root, y_root),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.5),
        )
        ax1.text(0.78, y_leaf,
                 f"{scen.label}  ν={scen.nu:.2f}\n"
                 f"p_CLC={scen.p_CLC_eff:.2f} CAD/kWh",
                 va='center', fontsize=7.5, color=col)

    ax1.text(x_root - 0.05, y_root,
             f'η, c_max, F_{{i,k}}\n(primera etapa)',
             va='center', ha='right', fontsize=7,
             bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', lw=0.8))
    ax1.text(0.5, 0.97, 'LMP NY-HQ', ha='center', va='top',
             fontsize=8, style='italic')
    ax1.set_xlim(0, 1.3)
    ax1.set_title('Árbol de escenarios LMP')

    # Panel derecho — distribución de p_CLC y margen
    nu_arr    = tree.nu
    pclc_arr  = tree.p_CLC_vec
    labels    = [s.label for s in scenarios]
    bar_cols  = colors_lmp
    bars = ax2.bar(labels, pclc_arr, color=bar_cols, alpha=0.8, width=0.5)
    ax2.axhline(cfg.p_act, ls='--', color='navy',   lw=1.2, label=f'p_act={cfg.p_act:.2f}')
    ax2.axhline(tree.p_CLC_mean, ls='-.',
                color='black', lw=1.2, label=f'E[p_CLC]={tree.p_CLC_mean:.3f}')
    for bar, nu_l in zip(bars, nu_arr):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'ν={nu_l:.2f}', ha='center', va='bottom', fontsize=7)
    ax2.set_ylabel('CAD/kWh')
    ax2.set_title('Distribución LMP')
    ax2.legend()

    fig.suptitle(f'Árbol de escenarios NY-HQ — {d["_fname"]}', fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, f'{d["_fname"]}_scenario_tree')


# ── 6. VSS ─────────────────────────────────────────────────────────────────────

def plot_vss(d: dict, out_dir: Path) -> None:
    """Barras de Profit_SP, Profit_EEV y VSS."""
    vss_r = d.get('vss')
    if vss_r is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_SINGLE, H_INDIV))

    # Panel izquierdo — comparación profits
    labels  = ['Profit SP\n(estocástico)', 'Profit EEV\n(det.→stoc.)']
    values  = [vss_r.profit_sp, vss_r.profit_eev]
    colors  = ['steelblue', 'tomato']
    bars = ax1.bar(labels, values, color=colors, width=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(values) * 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax1.set_ylabel('Profit [CAD]')
    ax1.set_title('Comparación SP vs EEV')

    # Panel derecho — VSS y parámetros EV
    ev = vss_r.ev_solution
    vss_val = vss_r.vss
    sign_col = 'seagreen' if vss_val >= 0 else 'crimson'
    ax2.bar(['VSS = SP − EEV'], [vss_val], color=sign_col, width=0.4, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_ylabel('CAD')
    ax2.set_title(f'VSS = {vss_val:.4f} CAD')
    rel_pct = 100 * vss_val / abs(vss_r.profit_sp) if vss_r.profit_sp != 0 else 0
    ax2.text(0, vss_val / 2,
             f'{rel_pct:+.2f}%\nη_EV={ev.eta_mean:.2f} kW\nc_max_EV={ev.c_max:.2f}',
             ha='center', va='center', fontsize=7.5,
             bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))

    fig.suptitle(f'VSS (Value of the Stochastic Solution) — {d["_fname"]}', fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, f'{d["_fname"]}_vss')


# ── 7. Comparativa multi-run (cuando hay varios pickles) ──────────────────────

def plot_multi_run_comparison(data: list[dict], out_dir: Path) -> None:
    """Compara métricas clave entre distintas ejecuciones."""
    if len(data) < 2:
        return

    fnames  = [d['_fname'] for d in data]
    profits = [d['result'].hist.profit[-1] for d in data]
    iters   = [len(d['result'].hist.res_primal) for d in data]
    etas    = [float(d['result'].eta_k.mean()) for d in data]
    cmaxs   = [float(d['result'].c_max_opt) for d in data]
    vsss    = [d['vss'].vss if d.get('vss') else 0.0 for d in data]

    fig, axes = plt.subplots(2, 3, figsize=(W_SINGLE, 2 * H_ROW))
    x = np.arange(len(fnames))
    w = 0.6

    def _bar(ax, vals, title, ylabel, color):
        ax.bar(x, vals, width=w, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fnames, rotation=25, ha='right', fontsize=6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    _bar(axes[0, 0], profits, 'Profit final [CAD]',    'CAD',       'steelblue')
    _bar(axes[0, 1], iters,   'Iteraciones ADMM',      'iter',      'seagreen')
    _bar(axes[0, 2], etas,    r'$\eta$ media [kW]',    'kW',        'navy')
    _bar(axes[1, 0], cmaxs,   r'$c_{\max}$ [kWh]',    'kWh',       'darkorange')
    _bar(axes[1, 1], vsss,    'VSS [CAD]',             'CAD',       'purple')

    # Convergencia residual (superpuesta)
    ax_c = axes[1, 2]
    for d in data:
        ax_c.semilogy(d['result'].hist.res_primal,
                      label=d['_fname'], lw=0.9)
    ax_c.set_xlabel('Iteración')
    ax_c.set_ylabel(r'$\|r_p\|$')
    ax_c.set_title('Convergencia primal')
    ax_c.legend(fontsize=5)

    fig.suptitle('Comparativa entre ejecuciones', fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, 'multi_run_comparison')


# ── 8. Distribución de flexibilidad por escenario GDP ─────────────────────────

def plot_flexibility_by_gdp(d: dict, out_dir: Path) -> None:
    """F_tilde_all (S, N, K) desglosado por escenario GDP."""
    result = d['result']
    F_tilde = result.F_tilde_all   # (S, N, K)
    S, N, K = F_tilde.shape
    cfg = d['cfg']

    gdp_labels = ['AM', 'PM', 'AM+PM']
    colors = ['steelblue', 'tomato', 'seagreen']

    fig, axes = plt.subplots(1, min(S, 3), figsize=(W_SINGLE, H_INDIV), sharey=True)
    if S == 1:
        axes = [axes]
    for s, (ax, lbl, col) in enumerate(zip(axes, gdp_labels[:S], colors[:S])):
        F_s = F_tilde[s]   # (N, K)
        im = ax.imshow(F_s, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                       vmin=0, vmax=F_tilde.max())
        ax.set_title(f'GDP={lbl}')
        ax.set_xlabel('Período $k$')
        if s == 0:
            ax.set_ylabel('FSP $i$')
    plt.colorbar(im, ax=axes[-1], label='kWh')

    fig.suptitle(f'Ofertas por escenario GDP — {d["_fname"]}', fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, f'{d["_fname"]}_ftilde_gdp')


# ── 9. Heterogeneidad FSP: temperatura + preferencias + flexibilidad ──────────

def plot_fsp_heterogeneity(d: dict, out_dir: Path) -> None:
    """Figura compuesta: perfiles de temperatura, preferencias y primera etapa.

    Layout 3 filas × 2 columnas (IEEE doble columna, 7 × 6.5 in):
      Fila 0 (full-width): x_i(t) para 3 FSPs por grupo (A/B/C/D) sobre las 24h.
                           Fondo: escenarios T_out. Sombras: ventanas K_AM / K_PM.
      Fila 1 (full-width): alpha_pop(i, t) para los mismos 12 FSPs — heatmap de
                           preferencias de incomodidad (heterogeneidad de parámetros).
      Fila 2 — izq.:       F_i,k primera etapa para los 12 FSPs (K periodos).
      Fila 2 — der.:       x_i durante ventanas de flexibilidad (mismo eje K)
                           superpuesto con F_i,k en eje secundario — acoplamiento
                           entre estado térmico y compromiso de primera etapa.
    """
    pop = d.get('pop')
    if pop is None:
        print('    [skip] pop no disponible en el pickle (re-ejecutar main.py)')
        return

    result = d['result']
    cfg    = d['cfg']

    N_FSP  = result.x_all.shape[0]
    T      = result.x_all.shape[1]
    t_h    = np.arange(T) * cfg.dt              # horas decimales [0, 24)
    K_idx  = np.array(pop.K_idx)               # (K,) índices globales
    K      = len(K_idx)
    k_axis = np.arange(K)

    # ── Seleccionar 3 FSPs representativos por grupo ───────────────────
    GROUP_COLORS = {'A': '#1f77b4', 'B': '#d62728', 'C': '#2ca02c', 'D': '#ff7f0e'}
    N_PER_GROUP  = 3
    sel_idx, sel_groups = [], []
    for grp in ['A', 'B', 'C', 'D']:
        candidates = np.where(pop.group_labels == grp)[0]
        chosen = candidates[:N_PER_GROUP]
        sel_idx.extend(chosen.tolist())
        sel_groups.extend([grp] * len(chosen))
    sel_idx    = np.array(sel_idx)
    sel_groups = np.array(sel_groups)

    # Posición de K en el eje temporal (para marcar en fila 0)
    K_AM_arr = pop.K_AM
    K_PM_arr = pop.K_PM

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W_SINGLE, 6.5))
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.2, 1.4, 2.0],
        hspace=0.55, wspace=0.35,
    )

    ax_temp  = fig.add_subplot(gs[0, :])        # fila 0 full-width
    ax_alpha = fig.add_subplot(gs[1, :])        # fila 1 full-width
    ax_F     = fig.add_subplot(gs[2, 0])        # fila 2 izq.
    ax_dual  = fig.add_subplot(gs[2, 1])        # fila 2 der.

    # ──────────────────────────────────────────────────────────────────────
    # Panel 0 — Temperaturas interiores x_i(t) + T_out escenarios
    # ──────────────────────────────────────────────────────────────────────
    # Escenarios T_out como fondo (gris muy claro)
    #for s in range(pop.T_out_scenarios.shape[0]):
    #    ax_temp.plot(t_h, pop.T_out_scenarios[s], color='#aaaaaa',
    #                 lw=0.5, alpha=0.5, zorder=1)
    #ax_temp.plot([], [], color='#aaaaaa', lw=1.0, label=r'$T_{out}$ (escenarios)', alpha=0.6)

    # Sombras ventanas de flexibilidad
    _shade_windows(ax_temp, t_h, K_AM_arr, K_PM_arr)

    # Trayectorias x_i(t) coloreadas por grupo
    plotted_groups = set()
    for i, grp in zip(sel_idx, sel_groups):
        col   = GROUP_COLORS[grp]
        label = f'Grupo {grp}' if grp not in plotted_groups else None
        ax_temp.plot(t_h, result.x_all[i], color=col, lw=1.0,
                     alpha=0.85, label=label, zorder=3)
        # Línea x_ref (comfort) punteada del mismo color
        ax_temp.axhline(pop.x_ref_pop[i], color=col, lw=0.5,
                        ls=':', alpha=0.5, zorder=2)
        plotted_groups.add(grp)

    ax_temp.set_xlim(0, 24)
    ax_temp.set_xlabel('Hora del día [h]')
    ax_temp.set_ylabel('Temperatura [°C]')
    ax_temp.set_title('Perfiles de temperatura interior $x_i(t)$ — heterogeneidad entre FSPs')
    ax_temp.legend(ncol=5, fontsize=6, loc='upper right')

    # ──────────────────────────────────────────────────────────────────────
    # Panel 1 — Heatmap alpha_pop(i, t): preferencias de incomodidad
    # ──────────────────────────────────────────────────────────────────────
    alpha_sel = pop.alpha_pop[sel_idx, :]     # (12, T)
    im = ax_alpha.imshow(
        alpha_sel, aspect='auto', cmap='RdYlGn_r',
        extent=[0, 24, len(sel_idx) - 0.5, -0.5],
        interpolation='nearest',
    )
    # Líneas de separación entre grupos
    grp_boundaries = [N_PER_GROUP, 2 * N_PER_GROUP, 3 * N_PER_GROUP]
    for b in grp_boundaries:
        ax_alpha.axhline(b - 0.5, color='white', lw=1.0)
    # Etiquetas de grupo en eje y
    grp_centers  = [N_PER_GROUP / 2 - 0.5, N_PER_GROUP * 1.5 - 0.5,
                    N_PER_GROUP * 2.5 - 0.5, N_PER_GROUP * 3.5 - 0.5]
    ax_alpha.set_yticks(grp_centers)
    ax_alpha.set_yticklabels(['A', 'B', 'C', 'D'], fontsize=8)
    # Marcas ventanas
    _shade_windows(ax_alpha, t_h, K_AM_arr, K_PM_arr, alpha_val=0.15)
    plt.colorbar(im, ax=ax_alpha, label=r'$\alpha_i(t)$ [CAD/°C²]',
                 orientation='vertical', fraction=0.015, pad=0.01)
    ax_alpha.set_xlabel('Hora del día [h]')
    ax_alpha.set_title('Coeficiente de incomodidad $\\alpha_i(t)$ (preferencias por grupo)')

    # ──────────────────────────────────────────────────────────────────────
    # Panel 2-izq. — Primera etapa F_i,k (K periodos)
    # ──────────────────────────────────────────────────────────────────────
    # Separador AM/PM en el eje k
    k_am_end = len(K_AM_arr)    # primeros k_am_end índices son AM
    ax_F.axvspan(-0.5, k_am_end - 0.5, color='#e8f0fe', alpha=0.5, label='AM')
    ax_F.axvspan(k_am_end - 0.5, K - 0.5, color='#fff3e0', alpha=0.5, label='PM')

    plotted_groups = set()
    for i, grp in zip(sel_idx, sel_groups):
        col   = GROUP_COLORS[grp]
        label = f'Grupo {grp}' if grp not in plotted_groups else None
        ax_F.plot(k_axis, result.F_all[i], color=col, lw=1.0,
                  alpha=0.85, label=label)
        plotted_groups.add(grp)

    ax_F.set_xlim(-0.5, K - 0.5)
    ax_F.set_xlabel('Período de flexibilidad $k$')
    ax_F.set_ylabel('$F_{i,k}$ [kWh]  (1ª etapa)')
    ax_F.set_title('Ofertas de primera etapa $F_{i,k}$')
    ax_F.legend(ncol=2, fontsize=6)

    # ──────────────────────────────────────────────────────────────────────
    # Panel 2-der. — Dual-axis: F_i,k (izq.) y x_i durante K periodos (der.)
    # Muestra el acoplamiento entre compromiso de primera etapa y estado térmico
    # ──────────────────────────────────────────────────────────────────────
    ax_dual2 = ax_dual.twinx()

    ax_dual.axvspan(-0.5, k_am_end - 0.5, color='#e8f0fe', alpha=0.4)
    ax_dual.axvspan(k_am_end - 0.5, K - 0.5, color='#fff3e0', alpha=0.4)

    plotted_groups = set()
    for i, grp in zip(sel_idx, sel_groups):
        col = GROUP_COLORS[grp]
        x_k = result.x_all[i, K_idx]           # temperatura en periodos K
        F_k = result.F_all[i]                  # F primera etapa en K periodos
        lbl_F = f'$F$ Grupo {grp}' if grp not in plotted_groups else None
        lbl_x = f'$x$ Grupo {grp}' if grp not in plotted_groups else None
        ax_dual.plot(k_axis, F_k, color=col, lw=1.2, alpha=0.9,  label=lbl_F)
        ax_dual2.plot(k_axis, x_k, color=col, lw=0.9, alpha=0.55,
                      ls='--', label=lbl_x)
        plotted_groups.add(grp)

    ax_dual.set_xlim(-0.5, K - 0.5)
    ax_dual.set_xlabel('Período de flexibilidad $k$')
    ax_dual.set_ylabel('$F_{i,k}$ [kWh]  (1ª etapa)', fontsize=7)
    ax_dual2.set_ylabel('$x_i(t_k)$ [°C]  (recourse)', fontsize=7, color='gray')
    ax_dual2.tick_params(axis='y', labelcolor='gray', labelsize=6)
    ax_dual.set_title('Primera etapa $F_{i,k}$ y temperatura de recourse $x_i(t_k)$')

    # Leyenda combinada
    h1, l1 = ax_dual.get_legend_handles_labels()
    ax_dual.legend(h1, l1, ncol=2, fontsize=5.5, loc='upper right')

    fig.suptitle(f'Heterogeneidad FSP — {d["_fname"]}', fontsize=9, y=1.01)
    _save(fig, out_dir, f'{d["_fname"]}_fsp_heterogeneity')


def _shade_windows(
    ax,
    t_h: np.ndarray,
    K_AM: np.ndarray,
    K_PM: np.ndarray,
    alpha_val: float = 0.12,
) -> None:
    """Sombrea las ventanas de flexibilidad AM y PM en un eje temporal."""
    if len(K_AM):
        am_start = t_h[K_AM[0]]
        am_end   = t_h[K_AM[-1]] + (t_h[1] - t_h[0])  # hasta el final del último periodo
        ax.axvspan(am_start, am_end, color='steelblue', alpha=alpha_val,
                   label='Ventana AM' if alpha_val > 0.1 else None, zorder=0)
    if len(K_PM):
        pm_start = t_h[K_PM[0]]
        pm_end   = t_h[K_PM[-1]] + (t_h[1] - t_h[0])
        ax.axvspan(pm_start, pm_end, color='tomato', alpha=alpha_val,
                   label='Ventana PM' if alpha_val > 0.1 else None, zorder=0)


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f'{name}.pdf'
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  → {path}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\nCargando resultados desde {RESULTS_DIR}/')
    data = load_pickles(RESULTS_DIR)
    if not data:
        print('Nada que graficar. Ejecute main.py primero.')
        return

    print(f'\nGenerando figuras en {FIGS_DIR}/')

    for d in data:
        print(f'\n  [{d["_fname"]}]')
        plot_admm_convergence(d, FIGS_DIR)
        plot_flexibility_profile(d, FIGS_DIR)
        plot_fsp_heatmap(d, FIGS_DIR)
        plot_thermal(d, FIGS_DIR)
        plot_scenario_tree(d, FIGS_DIR)
        plot_vss(d, FIGS_DIR)
        plot_flexibility_by_gdp(d, FIGS_DIR)
        plot_fsp_heterogeneity(d, FIGS_DIR)

    if len(data) >= 2:
        print('\n  [comparativa multi-run]')
        plot_multi_run_comparison(data, FIGS_DIR)

    print(f'\nListo. {len(list(FIGS_DIR.glob("*.pdf")))} PDFs en {FIGS_DIR}/')


if __name__ == '__main__':
    main()
