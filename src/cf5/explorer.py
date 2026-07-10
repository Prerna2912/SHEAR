"""
CF5 — Parametric Stress Explorer (synchronous backend).

Algorithm
---------
1.  Run base inference via CF2 (cached after first call).
2.  For each numeric input parameter, perturb by ±5%, ±10%, ±20%.
3.  Run V1 inference for each perturbed configuration (reuses cache).
4.  Rank parameters by max |Δ peak τ| across all perturbation levels.
5.  For the top-2 parameters, build a 5×5 joint interaction map.
6.  Return structured ExplorerResult.

Called from FastAPI via asyncio.run_in_executor — this function is blocking
and must not use asyncio primitives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerturbationPoint:
    level: float          # signed fractional level, e.g. -0.10
    param: str
    new_value: float
    delta_peak_tau: float  # peak_tau(perturbed) - peak_tau(base)
    peak_tau: float


@dataclass
class ParameterRanking:
    param: str
    base_value: float
    max_abs_delta: float              # max |Δpeak τ| across all levels
    perturbations: List[PerturbationPoint] = field(default_factory=list)


@dataclass
class InteractionMap:
    param_a: str
    param_b: str
    base_value_a: float
    base_value_b: float
    levels: List[float]           # e.g. [-0.20, -0.10, 0.0, +0.10, +0.20]
    grid: List[List[float]]       # [len(levels) × len(levels)] Δ peak τ from base


@dataclass
class ExplorerResult:
    geometry_type: str
    base_params: Dict[str, Any]
    base_peak_tau: float
    parameter_ranking: List[ParameterRanking]
    interaction_map: Optional[InteractionMap]
    n_inference_calls: int
    total_time_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_tau(mean_tau: list) -> float:
    """Max L2 norm of mean stress across all nodes."""
    arr = np.array(mean_tau, dtype=np.float32)   # [N, 6]
    return float(np.max(np.linalg.norm(arr, axis=-1)))


def _numeric_params(params: Dict[str, Any]) -> Dict[str, float]:
    """Return params that are numeric and non-zero (i.e., perturb-able)."""
    return {
        k: float(v)
        for k, v in params.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and abs(float(v)) > 1e-9
    }


def _safe_perturb(key: str, base: float, factor: float) -> float:
    """Apply factor, ensuring value retains same sign and ≥10% of base magnitude."""
    new = base * factor
    # Clamp to 10% of magnitude in the original direction
    min_mag = abs(base) * 0.10
    if abs(new) < min_mag:
        new = min_mag * (1 if base >= 0 else -1)
    return float(new)


# ---------------------------------------------------------------------------
# Core explorer — runs synchronously in a thread executor
# ---------------------------------------------------------------------------

def run_explorer_sync(
    geometry_type: str,
    base_params: Dict[str, Any],
    n_samples: int = 5,
    grid_size: int = 4,
    ode_steps: int = 10,
    ode_method: str = "euler",
    perturbation_levels: Tuple[float, ...] = (0.10, 0.20),
    interaction_levels: Tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20),
) -> ExplorerResult:
    """
    Run parametric sensitivity analysis and 2D interaction map.

    Uses fast euler ODE with low n_samples/grid_size/ode_steps — sensitivity
    ranking only needs relative Δpeak_tau, not high-fidelity stress values.
    """
    from src.cf2.queue import _blocking_inference, make_job

    t0 = time.time()
    n_calls = 0

    def infer(ptype: str, pparams: dict):
        nonlocal n_calls
        job = make_job(ptype, pparams, n_samples=n_samples, grid_size=grid_size,
                       ode_steps=ode_steps, ode_method=ode_method)
        n_calls += 1
        return _blocking_inference(job, lambda c, t: None)

    # --- 1. Base inference ---------------------------------------------------
    base_result = infer(geometry_type, base_params)
    base_peak = _peak_tau(base_result.mean_tau)

    # --- 2. Identify perturb-able parameters ---------------------------------
    numeric = _numeric_params(base_params)
    if not numeric:
        return ExplorerResult(
            geometry_type=geometry_type,
            base_params=base_params,
            base_peak_tau=base_peak,
            parameter_ranking=[],
            interaction_map=None,
            n_inference_calls=n_calls,
            total_time_ms=int((time.time() - t0) * 1000),
        )

    # --- 3. Sensitivity sweep ------------------------------------------------
    ranking: List[ParameterRanking] = []

    for pname, bval in numeric.items():
        perturbs: List[PerturbationPoint] = []

        for level in perturbation_levels:
            for sign in (-1.0, +1.0):
                new_val = _safe_perturb(pname, bval, 1.0 + sign * level)
                p = {**base_params, pname: new_val}
                try:
                    res = infer(geometry_type, p)
                    peak = _peak_tau(res.mean_tau)
                    delta = peak - base_peak
                except Exception:
                    peak = base_peak
                    delta = 0.0
                perturbs.append(PerturbationPoint(
                    level=sign * level,
                    param=pname,
                    new_value=new_val,
                    delta_peak_tau=delta,
                    peak_tau=peak,
                ))

        max_abs = max(abs(p.delta_peak_tau) for p in perturbs)
        ranking.append(ParameterRanking(
            param=pname,
            base_value=bval,
            max_abs_delta=max_abs,
            perturbations=perturbs,
        ))

    ranking.sort(key=lambda r: -r.max_abs_delta)

    # --- 4. 2D interaction map (top-2 parameters) ----------------------------
    interaction_map: Optional[InteractionMap] = None

    if len(ranking) >= 2:
        pa, pb = ranking[0].param, ranking[1].param
        va, vb = numeric[pa], numeric[pb]
        ilevels = list(interaction_levels)

        grid: List[List[float]] = []
        for fa in ilevels:
            row: List[float] = []
            for fb in ilevels:
                p = dict(base_params)
                p[pa] = _safe_perturb(pa, va, 1.0 + fa) if fa != 0.0 else va
                p[pb] = _safe_perturb(pb, vb, 1.0 + fb) if fb != 0.0 else vb
                try:
                    res = infer(geometry_type, p)
                    delta = _peak_tau(res.mean_tau) - base_peak
                    row.append(float(delta))
                except Exception:
                    row.append(0.0)
            grid.append(row)

        interaction_map = InteractionMap(
            param_a=pa,
            param_b=pb,
            base_value_a=va,
            base_value_b=vb,
            levels=ilevels,
            grid=grid,
        )

    return ExplorerResult(
        geometry_type=geometry_type,
        base_params=base_params,
        base_peak_tau=base_peak,
        parameter_ranking=ranking,
        interaction_map=interaction_map,
        n_inference_calls=n_calls,
        total_time_ms=int((time.time() - t0) * 1000),
    )
