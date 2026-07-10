"""
CF5 — Parametric Stress Explorer.

Backend for the sensitivity analysis and 2D interaction map feature.
All inference is delegated to CF2's queue, which handles CF1 → V1 → cache.
"""

from .explorer import run_explorer_sync, ExplorerResult, ParameterRanking, InteractionMap

__all__ = ["run_explorer_sync", "ExplorerResult", "ParameterRanking", "InteractionMap"]
