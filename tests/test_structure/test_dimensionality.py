"""Tests for structure/dimensionality.py.

Dimension estimators are inherently noisy; assertions use wide tolerances.
The "known manifold" tests embed data in a low-dimensional linear subspace
where the true intrinsic dimension is unambiguous.
"""
from __future__ import annotations

import numpy as np
import pytest

from structure.dimensionality import (
    _confidence_interval,
    _consensus_dimension,
    _eigenvalue_gap,
    _levina_bickel,
    _local_pca_estimate,
    _mle_estimate,
    _pca_dim,
    estimate,
)


# ---------------------------------------------------------------------------
# Shared data factories
# ---------------------------------------------------------------------------

def _linear_subspace(n: int, intrinsic_dim: int, ambient_dim: int,
                     noise: float = 0.02, seed: int = 42) -> np.ndarray:
    """Generate n points on a random linear subspace of dimension intrinsic_dim.

    The embedding is an orthonormal basis of shape (ambient_dim, intrinsic_dim),
    so all structure lies in a intrinsic_dim-dimensional affine subspace.
    """
    rng = np.random.default_rng(seed)
    # Orthonormal embedding matrix: (ambient_dim, intrinsic_dim).
    Q, _ = np.linalg.qr(rng.standard_normal((ambient_dim, intrinsic_dim)))
    coords = rng.standard_normal((n, intrinsic_dim))
    # Add small isotropic noise so the manifold is not degenerate.
    return coords @ Q[:, :intrinsic_dim].T + rng.standard_normal((n, ambient_dim)) * noise


# ---------------------------------------------------------------------------
# estimate() — public entry point
# ---------------------------------------------------------------------------

class TestEstimate:
    def test_returns_positive_int(self):
        # estimate() must always return a positive integer.
        data = _linear_subspace(n=100, intrinsic_dim=2, ambient_dim=8)
        result = estimate(data)
        assert isinstance(result, int) and result >= 1

    def test_bounded_by_n_features_minus_one(self):
        # Result must not exceed d - 1 regardless of input.
        data = _linear_subspace(n=100, intrinsic_dim=5, ambient_dim=8)
        result = estimate(data)
        assert result <= 7  # d - 1 = 8 - 1

    def test_1d_manifold_in_10d(self):
        # Points on a line in 10D; all estimators should agree on dim ≈ 1.
        data = _linear_subspace(n=200, intrinsic_dim=1, ambient_dim=10, noise=0.01)
        result = estimate(data)
        # Allow [1, 3]: noise inflates estimates slightly.
        assert 1 <= result <= 3

    def test_3d_manifold_in_10d(self):
        # Points in a 3D subspace of 10D; consensus should be near 3.
        data = _linear_subspace(n=300, intrinsic_dim=3, ambient_dim=10, noise=0.02)
        result = estimate(data)
        # Wide tolerance [1, 6] because MLE on correlated data is approximate.
        assert 1 <= result <= 6

    def test_small_dataset_does_not_raise(self):
        # Very small data (n < k) must fall back gracefully without error.
        data = _linear_subspace(n=10, intrinsic_dim=2, ambient_dim=5, noise=0.1)
        result = estimate(data)
        assert isinstance(result, int) and result >= 1


# ---------------------------------------------------------------------------
# _eigenvalue_gap
# ---------------------------------------------------------------------------

class TestEigenvalueGap:
    def test_1d_data_returns_1(self):
        # All variance in one direction → largest ratio at index 0 → dim = 1.
        rng = np.random.default_rng(0)
        line_dir = rng.standard_normal(10)
        line_dir /= np.linalg.norm(line_dir)
        data = np.outer(rng.standard_normal(100), line_dir) + rng.standard_normal((100, 10)) * 0.01
        result = _eigenvalue_gap(data)
        assert result <= 2  # practically always 1 for clean 1D data

    def test_two_component_data(self):
        # Data lives in a 2D plane; eigenvalue gap should identify 2.
        data = _linear_subspace(n=200, intrinsic_dim=2, ambient_dim=8, noise=0.01)
        result = _eigenvalue_gap(data)
        assert 1 <= result <= 4

    def test_returns_at_least_1(self):
        # Even for degenerate constant data the result must be ≥ 1.
        data = np.ones((10, 5))
        result = _eigenvalue_gap(data)
        assert result >= 1


# ---------------------------------------------------------------------------
# _local_pca_estimate
# ---------------------------------------------------------------------------

class TestLocalPCAEstimate:
    def test_falls_back_for_small_n(self):
        # n=5 ≤ k=20: fallback to eigenvalue gap must not raise.
        data = _linear_subspace(n=5, intrinsic_dim=2, ambient_dim=6, noise=0.1)
        result = _local_pca_estimate(data, k=20)
        assert isinstance(result, int) and result >= 1

    def test_normal_path_returns_positive_int(self):
        data = _linear_subspace(n=200, intrinsic_dim=3, ambient_dim=8, noise=0.02)
        result = _local_pca_estimate(data)
        assert isinstance(result, int) and result >= 1

    def test_bounded_by_ambient_dim(self):
        data = _linear_subspace(n=150, intrinsic_dim=3, ambient_dim=7, noise=0.02)
        result = _local_pca_estimate(data)
        assert result <= 7


# ---------------------------------------------------------------------------
# _pca_dim
# ---------------------------------------------------------------------------

class TestPCADim:
    def test_single_point_returns_full_dim(self):
        # A 1-point neighbourhood has undefined PCA; returns ambient dimension.
        result = _pca_dim(np.ones((1, 5)))
        assert result == 5

    def test_1d_cluster(self):
        # Points on a line; 1 component explains all variance.
        rng = np.random.default_rng(1)
        t = rng.standard_normal(20)
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        points = np.outer(t, v)
        result = _pca_dim(points)
        assert result == 1

    def test_full_dim_random_data(self):
        # Independent Gaussian data fills the ambient space.
        data = np.random.default_rng(2).standard_normal((30, 4))
        result = _pca_dim(data)
        assert 1 <= result <= 4

    def test_constant_points_returns_1(self):
        # All-identical points have zero variance → dimension 1.
        result = _pca_dim(np.ones((10, 4)))
        assert result == 1


# ---------------------------------------------------------------------------
# _mle_estimate
# ---------------------------------------------------------------------------

class TestMLEEstimate:
    def test_very_small_n_returns_1(self):
        # n=2 → k < _MIN_K_MLE → early return of 1.
        data = np.random.default_rng(3).standard_normal((2, 4))
        result = _mle_estimate(data)
        assert result == 1

    def test_normal_path_positive_int(self):
        data = _linear_subspace(n=200, intrinsic_dim=2, ambient_dim=8, noise=0.02)
        result = _mle_estimate(data)
        assert isinstance(result, int) and result >= 1


# ---------------------------------------------------------------------------
# _levina_bickel
# ---------------------------------------------------------------------------

class TestLevinaBickel:
    def test_single_actual_neighbour_returns_1(self):
        # Only 1 column after skipping self → not enough for log-ratios → 1.
        distances = np.array([[0.0, 1.0]])
        result = _levina_bickel(distances)
        assert result == 1

    def test_known_2d_estimate(self):
        # Simulate distances on a 2D manifold: r_j ∝ j^{1/2} → dim ≈ 2.
        rng = np.random.default_rng(7)
        n, k = 100, 10
        # For a d-dim manifold, distances scale as j^{1/d}.
        j = np.arange(1, k + 1, dtype=float)
        # Euclidean distances for d=2: r_j = C * j^{0.5}
        base_dists = j ** 0.5
        # Add small relative noise; self-distance is column 0.
        noise = 1.0 + rng.standard_normal((n, k)) * 0.05
        distances = np.column_stack([
            np.zeros(n),
            np.tile(base_dists, (n, 1)) * np.abs(noise),
        ])
        result = _levina_bickel(distances)
        # Allow [1, 4] due to noise and discretisation.
        assert 1 <= result <= 4


# ---------------------------------------------------------------------------
# _consensus_dimension and _confidence_interval
# ---------------------------------------------------------------------------

class TestConsensus:
    def test_median_of_estimates(self):
        # Median of [2, 3, 4] is 3; clamped to [1, d-1=9].
        result = _consensus_dimension([2, 3, 4], n_features=10)
        assert result == 3

    def test_clamped_to_at_least_1(self):
        # Even if all estimates are 0, result must be ≥ 1.
        result = _consensus_dimension([0, 0, 0], n_features=5)
        assert result == 1

    def test_clamped_to_n_features_minus_1(self):
        # Estimates larger than d-1 must be clamped down.
        result = _consensus_dimension([10, 12, 15], n_features=5)
        assert result == 4  # d - 1 = 4

    def test_confidence_interval_min_max(self):
        lo, hi = _confidence_interval([1, 4, 2])
        assert lo == 1 and hi == 4

    def test_confidence_interval_single_estimate(self):
        lo, hi = _confidence_interval([3])
        assert lo == 3 and hi == 3
