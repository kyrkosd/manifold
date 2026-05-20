"""
Phase 1 — Structure Discovery.

Public interface: ``discover_structure(data) → StructureReport`` detects the
topological structure of normalised data and recommends a Fourier engine for
the downstream Graph Fourier Transform phase.
"""
from __future__ import annotations

import numpy as np

from . import dimensionality, graph_builder
from .report import StructureReport
from .report import build as _build_report

__all__ = ["discover_structure", "StructureReport"]


def discover_structure(data: np.ndarray) -> StructureReport:
    """Detect the structure of *data* and recommend a Fourier engine.

    Runs Phase 1 steps [1.1]–[1.5] from the pipeline specification:
    builds the global feature graph, estimates the intrinsic dimension,
    and assembles the StructureReport for downstream phases.

    Parameters
    ----------
    data : (n, d) normalised float64 array (output of Phase 0 ingestion).

    Returns
    -------
    StructureReport : detected topology, intrinsic dimension, feature graph,
        and recommended Fourier engine.
    """
    # Step [1.1]-[1.3]: build feature graph from three complementary metrics.
    graph = graph_builder.build(data)
    # Step [1.4]: estimate the intrinsic dimensionality of the manifold.
    dim = dimensionality.estimate(data)
    # Step [1.5]: assemble and return the structure report.
    return _build_report(graph, dim)
