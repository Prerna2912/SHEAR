"""
CF4 — Uncertainty & Regime Map.

Consumes CF1 (grad_u, positions, grid_shape) and CF2 (mean_tau,
variance_tau) outputs to produce a 4-panel diagnostic figure:

  1. Flow Regime Map        — Q-criterion coloured mid-plane slice
  2. Prediction Uncertainty — per-node uncertainty with high-unc zone
  3. SGS Dissipation        — Π = -τ:S (forward scatter vs backscatter)
  4. Backscatter Map        — nodes where V1 predicts energy back-transfer

Usage
-----
    from src.cf4 import diagnose

    fig = diagnose(cf2_result, geometry_type="aerofoil")
    fig.savefig("cf4_diagnostics.png", dpi=150, bbox_inches="tight")
"""

from .compute import analyse, CF4Output
from .plots import make_figure

__all__ = ["diagnose", "analyse", "make_figure", "CF4Output"]


def diagnose(cf2_result, geometry_type: str = "", figsize=(16, 14)):
    """
    One-call end-to-end: CF2 InferenceResult → matplotlib Figure.

    Args:
        cf2_result   : src.cf2.schemas.InferenceResult or any object / dict
                       with .mean_tau, .variance_tau, .grad_u, .grid_shape,
                       .geometry_type, .params attributes (or dict keys).
        geometry_type: override geometry label (uses cf2_result.geometry_type if blank).
        figsize      : passed to make_figure.

    Returns:
        fig : matplotlib Figure
    """
    # Accept both Pydantic model instances and plain dicts (mock fixture)
    def _get(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    mean_tau     = _get(cf2_result, 'mean_tau')
    variance_tau = _get(cf2_result, 'variance_tau')
    grad_u       = _get(cf2_result, 'grad_u')
    grid_shape   = _get(cf2_result, 'grid_shape')
    params       = _get(cf2_result, 'params')
    gtype        = geometry_type or _get(cf2_result, 'geometry_type')

    cf4 = analyse(mean_tau, variance_tau, grad_u)
    return make_figure(cf4, grid_shape, geometry_type=gtype, params=params, figsize=figsize)
