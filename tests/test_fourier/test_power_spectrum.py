"""Tests for fourier/power_spectrum.py.

Power values are derived from squared GFT coefficients; all assertions
use exact arithmetic wherever possible so tolerances are tight.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import EigenBasis
from fourier.power_spectrum import (
    _compare_spectra,
    _cumulative_power,
    _dominant_frequencies,
    _mean_power,
    _per_point_power,
    _spectral_centroid,
    compute,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _basis(d: int = 4) -> EigenBasis:
    """Identity eigenbasis with eigenvalues [0, 1, …, d-1]."""
    return EigenBasis(eigenvectors=np.eye(d), eigenvalues=np.arange(d, dtype=float))


def _coeffs(n: int = 10, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------------------
# _per_point_power and _mean_power
# ---------------------------------------------------------------------------

class TestPerPointAndMeanPower:
    """Tests for Per Point And Mean Power."""
    def test_per_point_shape_preserved(self):
        # Squaring does not change the (n, d) shape.
        assert _per_point_power(_coeffs(n=8, d=5)).shape == (8, 5)

    def test_per_point_non_negative(self):
        # Squares are always non-negative.
        assert np.all(_per_point_power(_coeffs()) >= 0)

    def test_per_point_known_value(self):
        # [[3, 4]] squared is [[9, 16]].
        np.testing.assert_allclose(
            _per_point_power(np.array([[3.0, 4.0]])), [[9.0, 16.0]]
        )

    def test_mean_power_shape(self):
        # Averaging over n rows collapses the first axis; result is (d,).
        per_point = _per_point_power(_coeffs(n=10, d=6))
        assert _mean_power(per_point).shape == (6,)

    def test_mean_power_known_value(self):
        # Column means of [[1, 4], [9, 16]] are [5, 10].
        np.testing.assert_allclose(
            _mean_power(np.array([[1.0, 4.0], [9.0, 16.0]])), [5.0, 10.0]
        )


# ---------------------------------------------------------------------------
# _cumulative_power and _dominant_frequencies
# ---------------------------------------------------------------------------

class TestCumulativeAndDominant:
    """Tests for Cumulative And Dominant."""
    def test_cumulative_last_is_one(self):
        # Normalised cumsum must end at exactly 1.0 for any non-zero spectrum.
        result = _cumulative_power(np.array([1.0, 2.0, 3.0, 4.0]))
        assert result[-1] == pytest.approx(1.0)

    def test_cumulative_monotone(self):
        # Power can only accumulate; result must be non-decreasing.
        result = _cumulative_power(np.array([0.5, 1.0, 2.0]))
        assert np.all(result[1:] >= result[:-1])

    def test_cumulative_zero_spectrum(self):
        # All-zero spectrum must produce all-zero cumulative (no divide-by-zero).
        np.testing.assert_allclose(_cumulative_power(np.zeros(4)), 0.0)

    def test_dominant_length(self):
        # Default top-5; fewer than 5 components → returns all of them.
        result = _dominant_frequencies(np.arange(10, dtype=float))
        assert len(result) == 5

    def test_dominant_first_is_argmax(self):
        # The index of the maximum power must be first in the sorted list.
        mean = np.array([0.0, 0.0, 5.0, 0.0])
        assert _dominant_frequencies(mean, n=4)[0] == 2


# ---------------------------------------------------------------------------
# _spectral_centroid and _compare_spectra
# ---------------------------------------------------------------------------

class TestSpectralCentroidAndCompare:
    """Tests for Spectral Centroid And Compare."""
    def test_centroid_is_float(self):
        # Return type must always be Python float.
        result = _spectral_centroid(np.array([1.0, 2.0, 3.0]), _basis(3).eigenvalues)
        assert isinstance(result, float)

    def test_centroid_zero_power(self):
        # No power in any mode → centroid defaults to 0.
        assert _spectral_centroid(np.zeros(4), _basis().eigenvalues) == pytest.approx(0.0)

    def test_centroid_single_mode(self):
        # All power in mode k → centroid equals eigenvalue[k].
        # Basis eigenvalues = [0, 1, 2, 3]; all power at index 2 → centroid = 2.
        mean = np.array([0.0, 0.0, 1.0, 0.0])
        assert _spectral_centroid(mean, _basis().eigenvalues) == pytest.approx(2.0)

    def test_compare_identical_spectra(self):
        # L2 distance of a spectrum to itself is 0.
        spec = np.array([0.1, 0.5, 0.3, 0.1])
        assert _compare_spectra(spec, spec) == pytest.approx(0.0)

    def test_compare_orthogonal_spectra(self):
        # [1, 0] and [0, 1] have L2 distance √2.
        dist = _compare_spectra(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert dist == pytest.approx(np.sqrt(2))


# ---------------------------------------------------------------------------
# compute() — integration
# ---------------------------------------------------------------------------

class TestComputeIntegration:
    """Tests for Compute Integration."""
    def test_all_keys_present(self):
        # All five keys must appear regardless of input shape.
        result = compute(_coeffs(n=10, d=4), _basis())
        assert {"per_point", "mean", "cumulative", "dominant", "centroid"}.issubset(result)

    def test_per_point_shape(self):
        # per_point retains (n, d).
        result = compute(_coeffs(n=8, d=5), _basis(5))
        assert result["per_point"].shape == (8, 5)

    def test_mean_shape(self):
        # mean collapses n; shape is (d,).
        result = compute(_coeffs(n=8, d=5), _basis(5))
        assert result["mean"].shape == (5,)

    def test_cumulative_ends_at_one(self):
        # Cumulative power of any non-trivial input must end at 1.
        result = compute(_coeffs(n=10, d=4, seed=1), _basis())
        assert result["cumulative"][-1] == pytest.approx(1.0)
