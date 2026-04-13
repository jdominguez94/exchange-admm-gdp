"""
gdp_pkg — Exchange ADMM Two-Stage Stochastic para GDP Bloc 1 (Hydro-Québec).

Módulos
-------
config      : GDPConfig, ADMMConfig  (parámetros)
profiles    : build_profiles         (perfiles NSL y temperatura)
population  : build_population       (población de FSPs)
baseline    : solve_baselines        (baselines QP + headroom)
fsp_worker  : FspWorker              (subproblema FSP CVXPY)
aggregator  : solve_aggregator       (subproblema agregador CVXPY)
admm        : run_exchange_admm      (loop Exchange ADMM)
"""

from .config import GDPConfig, ADMMConfig
from .profiles import build_profiles
from .population import build_population
from .baseline import solve_baselines
from .admm import run_exchange_admm, ADMMResult
from .vss import compute_vss, VSSResult, DetEquivSolution

__all__ = [
    'GDPConfig', 'ADMMConfig',
    'build_profiles', 'build_population',
    'solve_baselines', 'run_exchange_admm',
    'ADMMResult',
    'compute_vss', 'VSSResult', 'DetEquivSolution',
]
