"""
Phase 3 — Manifold Analysis.

Public interface: ``build_manifold(data, spectral, structure, config) → Manifold``
assembles an atlas of local charts, aligns their eigenbases, and computes
the expected reconstruction residual distribution for each chart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import manifold.atlas as atlas_mod
import manifold.expected_residual as expected_residual_mod
from fourier import SpectralData
from manifold.atlas import Atlas
from manifold.expected_residual import ExpectedResidualDistribution
from structure.report import StructureReport

__all__ = ["Manifold", "build_manifold"]


@dataclass
class Manifold:
    """Container for all Phase-3 outputs.

    Parameters
    ----------
    atlas : Atlas of overlapping charts covering the data manifold.
    expected_residuals : per-chart expected residual distributions,
        keyed by chart index.
    faiss_index : FAISS index over the full dataset (from atlas build).
    spectral : Phase-2 SpectralData forwarded for downstream use.
    """

    atlas: Atlas
    expected_residuals: dict[int, ExpectedResidualDistribution]
    faiss_index: Any = field(repr=False)   # faiss.Index
    spectral: SpectralData


def build_manifold(
    data: np.ndarray,
    spectral: SpectralData,
    structure: StructureReport,
    config: dict,
) -> Manifold:
    """Build the manifold atlas and expected residual distributions.

    Parameters
    ----------
    data : (n, d) normalised float64 data (Phase-0 output).
    spectral : SpectralData from Phase 2.
    structure : StructureReport from Phase 1.
    config : keys: n_charts, overlap_factor, max_retries.

    Returns
    -------
    Manifold
    """
    atlas = atlas_mod.build(data, spectral, structure, config)

    expected_residuals: dict[int, ExpectedResidualDistribution] = {}
    for i, chart in enumerate(atlas.charts):
        region_data = data[chart.region_indices]
        expected_residuals[i] = expected_residual_mod.compute(
            chart, region_data, spectral.bands, spectral.basis
        )

    return Manifold(
        atlas=atlas,
        expected_residuals=expected_residuals,
        faiss_index=atlas.faiss_index,
        spectral=spectral,
    )
