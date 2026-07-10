"""
CF2 — Production V1 Inference Engine.

Provides the ML serving layer that accepts a CF1 graph, runs V1
(SE(3)-equivariant CFM) ODE integration, and returns mean τ,
per-point variance, and all samples.

Quick start (synchronous, no server needed)
-------------------------------------------
    from src.cf2 import infer

    result = infer(
        geometry_type="aerofoil",
        params={"chord": 1.0, "span": 5.0, "alpha_deg": 5.0, "U_inf": 50.0, "Re": 3e6},
        checkpoint="runs/quick/v1/last.pt",
        stats="runs/quick/stats.pt",
        n_samples=5,
    )
    # result.mean_tau   → list [N, 6]
    # result.variance_tau → list [N, 6]

Serving (FastAPI)
-----------------
    uvicorn src.cf2.app:app --host 0.0.0.0 --port 7860
    # POST /api/v1/infer  →  job_id
    # GET  /api/v1/jobs/{job_id}  →  InferenceResult
    # WS   /ws/progress/{job_id}  →  ProgressMessage stream
"""

from .schemas import InferenceResult, GeometryRequest, JobResponse, JobStatus
from .inference import run_inference, geometry_hash, compute_backscatter
from .cache import get_cache

__all__ = [
    "infer",
    "run_inference",
    "geometry_hash",
    "compute_backscatter",
    "get_cache",
    "InferenceResult",
    "GeometryRequest",
    "JobResponse",
    "JobStatus",
]


def infer(
    geometry_type: str,
    params: dict,
    checkpoint: str = "runs/quick/v1/last.pt",
    stats: str = "runs/quick/stats.pt",
    n_samples: int = 20,
    ode_steps: int = 100,
    grid_size: int = 16,
    include_samples: bool = False,
) -> "InferenceResult":
    """
    Synchronous end-to-end inference: CF1 → V1 → InferenceResult.

    Loads the model fresh each call.  For repeated calls use the FastAPI
    server (which keeps the model in memory) or call checkpoint.init_model()
    once and then run_inference() directly.
    """
    import time
    import numpy as np
    t0 = time.time()

    from .checkpoint import load_v1
    from .schemas import InferenceResult, JobStatus
    from src.cf1 import solve as cf1_solve

    model, norm_stats, device = load_v1(checkpoint, stats)

    cf1_out = cf1_solve(geometry_type, params, grid_size=grid_size)

    raw = run_inference(
        graph=cf1_out.graph,
        model=model,
        stats=norm_stats,
        device=device,
        n_samples=n_samples,
        ode_steps=ode_steps,
        include_samples=include_samples,
    )

    g = cf1_out.grad_u.reshape(-1, 3, 3)
    strain_rate = 0.5 * (g + g.transpose(0, 2, 1))

    ghash = geometry_hash(geometry_type, params)
    elapsed_ms = int((time.time() - t0) * 1000)

    return InferenceResult(
        mean_tau=raw["mean_tau"].tolist(),
        variance_tau=raw["variance_tau"].tolist(),
        samples=(raw["samples"].tolist() if raw["samples"] is not None else None),
        positions=cf1_out.positions.tolist(),
        grad_u=cf1_out.grad_u.reshape(-1, 9).tolist(),
        grid_shape=list(cf1_out.grid_shape),
        geometry_hash=ghash,
        geometry_type=geometry_type,
        params=params,
        n_samples=n_samples,
        n_nodes=raw["n_nodes"],
        ode_steps=ode_steps,
        inference_time_ms=elapsed_ms,
        job_id="sync",
        status=JobStatus.COMPLETE,
    )
