"""Tests for manifold/chart_validator.py.

Two fixture types:
- Good chart: 2-D subspace embedded in 10-D; PCA-like basis gives high injectivity.
- Bad chart: random projection that does not preserve distances.
"""
from __future__ import annotations

import numpy as np
import pytest

import structure.graph_builder as graph_builder
from common import nn_utils
from common.types import AlignmentQuality
from fourier.graph_fourier import GraphFourierEngine
from manifold.chart import Chart, build, _define_chart_map, _compute_chart_inverse
from manifold.chart_validator import (
    _check_continuity,
    _check_injectivity,
    _check_invertibility,
    _compute_distortion,
    _quality_score,
    _split_region,
    validate,
)
from manifold.eigenvector_alignment import AlignedBasis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _subspace_data(
    n: int = 60, ambient: int = 10, intrinsic: int = 2,
    noise: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    orth_mat, _ = np.linalg.qr(rng.standard_normal((ambient, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coeffs = rng.standard_normal((n, intrinsic))
    noise_mat = rng.standard_normal((n, ambient)) * noise
    return coeffs @ basis.T + noise_mat, basis


def _good_chart(seed: int = 1) -> tuple[Chart, np.ndarray]:
    """Build a valid chart for a near-linear 2-D submanifold."""
    data, _ = _subspace_data(n=60, seed=seed)
    region_indices = np.arange(40, dtype=np.intp)
    region_data = data[region_indices]
    fg = graph_builder.build(data)
    ref = GraphFourierEngine(fg).basis
    idx = nn_utils.build_faiss_index(data)
    chart = build(
        region_data=region_data,
        region_indices=region_indices,
        full_data=data,
        graph_builder=graph_builder,
        intrinsic_dim=2,
        reference_basis=ref,
        faiss_index=idx,
        overlap_factor=0.2,
    )
    return chart, region_data


def _bad_chart(seed: int = 0) -> tuple[Chart, np.ndarray]:
    """Construct a pathological chart using a random low-dim projection."""
    rng = np.random.default_rng(seed + 99)
    n, ambient, intrinsic = 40, 10, 2
    data, _ = _subspace_data(n=n, ambient=ambient, seed=seed)
    # Random (not data-driven) projection — poor injectivity.
    rand_vecs = rng.standard_normal((ambient, intrinsic))
    rand_vecs, _ = np.linalg.qr(rand_vecs)
    rand_vecs = rand_vecs[:, :intrinsic]

    # Synthesise a dummy AlignedBasis wrapping the random vectors.
    dummy_basis = AlignedBasis(
        eigenvectors=rand_vecs,
        eigenvalues=np.ones(intrinsic),
        rotation_matrix=np.eye(intrinsic),
        alignment_score=0.1,
        quality=AlignmentQuality.DIVERGENT,
    )

    chart_map = _define_chart_map(rand_vecs)
    chart_inv = _compute_chart_inverse(rand_vecs)
    coords = np.stack([chart_map(p) for p in data])

    return Chart(
        region_indices=np.arange(n, dtype=np.intp),
        expanded_indices=np.arange(n, dtype=np.intp),
        local_basis=dummy_basis,
        selected_indices=[0, 1],
        selected_vectors=rand_vecs,
        coordinates=coords,
        chart_map=chart_map,
        chart_inverse=chart_inv,
        intrinsic_dim=intrinsic,
        ambient_dim=ambient,
    ), data


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    """Tests for Validate."""
    def test_good_chart_is_valid(self):
        """Good chart is valid."""
        chart, region_data = _good_chart()
        valid, _, _reason = validate(chart, region_data)
        assert valid is True

    def test_good_chart_score_positive(self):
        """Good chart score positive."""
        chart, region_data = _good_chart()
        _, score, _ = validate(chart, region_data)
        assert score > 0.0

    def test_validate_returns_three_tuple(self):
        """Validate returns three tuple."""
        chart, region_data = _good_chart()
        result = validate(chart, region_data)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _check_injectivity
# ---------------------------------------------------------------------------

class TestCheckInjectivity:
    """Tests for Check Injectivity."""
    def test_good_chart_passes_injectivity(self):
        """Good chart passes injectivity."""
        chart, region_data = _good_chart()
        assert _check_injectivity(chart, region_data, threshold=0.8) is True

    def test_single_point_passes(self):
        """Single point passes."""
        chart, region_data = _good_chart()
        assert _check_injectivity(chart, region_data[:1], threshold=0.8) is True

    def test_low_threshold_always_passes(self):
        """Low threshold always passes."""
        chart, region_data = _good_chart()
        assert _check_injectivity(chart, region_data, threshold=0.0) is True


# ---------------------------------------------------------------------------
# _check_continuity
# ---------------------------------------------------------------------------

class TestCheckContinuity:
    """Tests for Check Continuity."""
    def test_good_chart_continuous(self):
        """Good chart continuous."""
        chart, region_data = _good_chart()
        assert _check_continuity(chart, region_data) is True

    def test_tiny_data_passes(self):
        """Tiny data passes."""
        chart, region_data = _good_chart()
        assert _check_continuity(chart, region_data[:2]) is True


# ---------------------------------------------------------------------------
# _check_invertibility
# ---------------------------------------------------------------------------

class TestCheckInvertibility:
    """Tests for Check Invertibility."""
    def test_good_chart_invertible(self):
        """Good chart invertible."""
        chart, region_data = _good_chart()
        assert _check_invertibility(chart, region_data, tolerance=0.5) is True

    def test_empty_data_passes(self):
        """Empty data passes."""
        chart, region_data = _good_chart()
        assert _check_invertibility(chart, region_data[:0], tolerance=0.1) is True


# ---------------------------------------------------------------------------
# _compute_distortion
# ---------------------------------------------------------------------------

def test_orthonormal_basis_has_distortion_one():
    """Orthonormal basis has distortion one."""
    # Orthonormal columns → singular values all 1 → condition number 1.
    chart, _ = _good_chart()
    assert _compute_distortion(chart) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _quality_score
# ---------------------------------------------------------------------------

class TestQualityScore:
    """Tests for Quality Score."""
    def test_good_chart_score_above_half(self):
        """Good chart score above half."""
        chart, region_data = _good_chart()
        assert _quality_score(chart, region_data) > 0.5

    def test_score_in_unit_interval(self):
        """Score in unit interval."""
        chart, region_data = _good_chart()
        s = _quality_score(chart, region_data)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# _split_region
# ---------------------------------------------------------------------------

class TestSplitRegion:
    """Tests for Split Region."""
    def test_returns_two_parts(self):
        """Returns two parts."""
        chart, region_data = _good_chart()
        parts = _split_region(chart, region_data)
        assert len(parts) == 2

    def test_parts_cover_all_indices(self):
        """Parts cover all indices."""
        chart, region_data = _good_chart()
        parts = _split_region(chart, region_data)
        all_idx = set(parts[0].tolist()) | set(parts[1].tolist())
        assert all_idx == set(range(len(region_data)))

    def test_parts_are_disjoint(self):
        """Parts are disjoint."""
        chart, region_data = _good_chart()
        parts = _split_region(chart, region_data)
        assert len(set(parts[0].tolist()) & set(parts[1].tolist())) == 0
