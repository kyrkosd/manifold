"""Public API for the Fourier Manifold Anomaly System (FMAS).

Import ``run_analysis`` to run the full pipeline end-to-end.
"""
from __future__ import annotations

from pipeline import (
    FinalReport,
    FourierManifoldPipeline,
    create_pipeline,
    run_analysis,
)

__all__ = [
    "FinalReport",
    "FourierManifoldPipeline",
    "create_pipeline",
    "run_analysis",
]
