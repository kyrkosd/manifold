"""Shared fixtures for tests/test_anomaly."""
from __future__ import annotations

import numpy as np
import pytest
import structure.graph_builder as gb

from common.types import FourierType, StructureType
from fourier import analyze_fourier
from manifold import build_manifold
from structure.report import StructureReport


@pytest.fixture(scope="module", name="manifold_fixture")
def _manifold_fixture():
    """2-D linear submanifold in 5-D ambient space, built once per test module."""
    rng = np.random.default_rng(0)
    d, intrinsic, n = 5, 2, 100
    orth_mat, _ = np.linalg.qr(rng.standard_normal((d, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coords = rng.standard_normal((n, intrinsic))
    data = coords @ basis.T + rng.standard_normal((n, d)) * 0.02

    fg = gb.build(data)
    structure = StructureReport(
        type=StructureType.GRAPH, intrinsic_dim=intrinsic, graph=fg,
        ordering=None, periodicity=False,
        recommended_fourier=FourierType.GRAPH_FOURIER, confidence=1.0,
    )
    spectral = analyze_fourier(data, structure)
    config = {"n_charts": 3, "overlap_factor": 0.2, "max_retries": 3}
    mf = build_manifold(data, spectral, structure, config)
    return mf, data
