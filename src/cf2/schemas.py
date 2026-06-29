"""
CF2 response schema — integration contract for the entire product.

This file is the Day 1 deliverable that CF3 (Deepak) and CF4/CF5 (Sudip) build
their frontends against.  Do not change field names without coordinating.

Serialisation notes
-------------------
* mean_tau, variance_tau, samples are transmitted as nested lists (JSON-safe).
  Consumers should wrap with np.array(…, dtype=np.float32).
* positions is the [N, 3] grid coordinate array from CF1, enabling spatial
  rendering without re-running CF1.
* geometry_hash is the SHA-256 of the canonical geometry parameters dict and
  is used as the cache key in CF2 and the CF5 parametric explorer.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GeometryRequest(BaseModel):
    """Full inference request accepted by POST /api/v1/infer."""
    geometry_type: str = Field(
        ...,
        description="One of: aerofoil, swept_wing, bluff_body, cylinder, "
                    "turbine_blade, flat_plate, ship_hull, bluff_body_wake",
    )
    params: Dict[str, Any] = Field(
        ...,
        description="Geometry-specific parameter dict (see CF1 validation docs).",
    )
    n_samples: int = Field(
        default=20,
        ge=5, le=50,
        description="Number of CFM ODE samples (5–50). Default 20.",
    )
    ode_steps: int = Field(
        default=100,
        ge=10, le=500,
        description="ODE solver step budget (Dopri5 adaptive; this caps max steps).",
    )
    grid_size: int = Field(
        default=16,
        ge=4, le=32,
        description="CF1 grid resolution per side (4–32). Default 16 → 4096 nodes.",
    )
    include_samples: bool = Field(
        default=False,
        description="If True, include all N_samples × N × 6 raw samples in response. "
                    "Large (~8 MB per request at n_samples=20, N=4096). "
                    "Required by CF5 parametric explorer.",
    )


class ExplorerRequest(BaseModel):
    """CF5 parametric explorer request (batch of perturbed geometries)."""
    base_request: GeometryRequest
    perturbations: List[Dict[str, Any]] = Field(
        ...,
        description="List of perturbed param dicts. Each is the full params dict "
                    "for one perturbation level.",
    )


class ExplorerJobRequest(BaseModel):
    """CF5 parametric explorer — backend auto-computes all perturbations."""
    geometry_type: str
    params: Dict[str, Any]
    n_samples: int = Field(default=10, ge=5, le=50,
                           description="Samples per inference call (fewer = faster).")
    grid_size: int = Field(default=16, ge=4, le=32)


class CF4Request(BaseModel):
    """Request CF4 diagnostics for an already-computed inference result."""
    mean_tau: List[List[float]]       # [N, 6]
    variance_tau: List[List[float]]   # [N, 6]
    grad_u: List[List[float]]         # [N, 9]


class CF4Result(BaseModel):
    """CF4 diagnostics result."""
    q_values: List[float]           # [N] Q-criterion
    regime_labels: List[int]        # [N] 0=strain-dom, 1=mixed, 2=vortex-dom
    regime_thresholds: List[float]  # [p33, p66]
    uncertainty: List[float]        # [N] L2 norm of variance_tau
    high_unc_mask: List[bool]       # [N] top-25% uncertainty
    dissipation: List[float]        # [N] Π = -τ:S
    backscatter_mask: List[bool]    # [N] Π < 0
    backscatter_frac: float


class PerturbationPoint(BaseModel):
    level: float
    param: str
    new_value: float
    delta_peak_tau: float
    peak_tau: float


class ParameterRanking(BaseModel):
    param: str
    base_value: float
    max_abs_delta: float
    perturbations: List[PerturbationPoint]


class InteractionMap(BaseModel):
    param_a: str
    param_b: str
    base_value_a: float
    base_value_b: float
    levels: List[float]
    grid: List[List[float]]


class ExplorerResult(BaseModel):
    geometry_type: str
    base_params: Dict[str, Any]
    base_peak_tau: float
    parameter_ranking: List[ParameterRanking]
    interaction_map: Optional[InteractionMap] = None
    n_inference_calls: int
    total_time_ms: int


# ---------------------------------------------------------------------------
# Job / progress models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETE  = "complete"
    ERROR     = "error"


class JobResponse(BaseModel):
    """Returned immediately from POST /api/v1/infer."""
    job_id: str
    status: JobStatus
    geometry_hash: str


class ProgressMessage(BaseModel):
    """Streamed over WebSocket /ws/progress/{job_id}."""
    job_id: str
    sample: int            # number of samples completed so far
    n_samples: int
    status: JobStatus
    message: str = ""


# ---------------------------------------------------------------------------
# Inference result  —  the integration contract
# ---------------------------------------------------------------------------

class InferenceResult(BaseModel):
    """
    Full V1 inference result.  Returned by GET /api/v1/jobs/{job_id} once
    status == complete, and cached by geometry_hash.

    Field sizes (default grid_size=16, n_samples=20):
        mean_tau      : N × 6   float32  (N = grid_size³ = 4096)
        variance_tau  : N × 6   float32
        samples       : n_samples × N × 6  float32  (only if include_samples=True)
        positions     : N × 3   float32  (CF1 grid coordinates)
        strain_rate   : N × 9   float32  (S_ij = (∂u_i/∂x_j + ∂u_j/∂u_i)/2, flat)
    """
    # Core stress predictions
    mean_tau: List[List[float]] = Field(
        description="[N, 6] mean SGS stress in e3nn irreps (1×0e + 1×2e). "
                    "Use src.data.dataset.irreps_to_stress() to get [N, 3, 3].",
    )
    variance_tau: List[List[float]] = Field(
        description="[N, 6] per-node variance across n_samples. "
                    "High values flag where V1 is uncertain.",
    )

    # Optional (set include_samples=True)
    samples: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="[n_samples, N, 6] all raw CFM samples. None unless requested.",
    )

    # Spatial context (from CF1 — needed by CF3 renderer)
    positions: List[List[float]] = Field(
        description="[N, 3] physical grid coordinates (x, y, z) in metres.",
    )
    grad_u: List[List[float]] = Field(
        description="[N, 9] velocity gradient ∂u_i/∂x_j flat (row-major 3×3). "
                    "Used by CF4 for Q-criterion and SGS dissipation.",
    )
    grid_shape: List[int] = Field(
        description="[Nx, Ny, Nz] structured grid dimensions.",
    )

    # Metadata
    geometry_hash: str = Field(
        description="SHA-256 of canonical geometry params. Used as cache key.",
    )
    geometry_type: str
    params: Dict[str, Any]
    n_samples: int
    n_nodes: int
    ode_steps: int
    inference_time_ms: int
    job_id: str
    status: JobStatus = JobStatus.COMPLETE

    # Error (only set when status == error)
    error: Optional[str] = None
