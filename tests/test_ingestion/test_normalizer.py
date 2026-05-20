"""Tests for ingestion/normalizer.py."""
from __future__ import annotations

import numpy as np
import pytest

from ingestion.normalizer import (
    _select_method,
    _unit_variance,
    _zero_mean,
    inverse_normalize,
    normalize,
)
from ingestion.types import NormParams


# ---------------------------------------------------------------------------
# normalize() — public entry point
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_output_shape_preserved(self):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((20, 4))
        normed, params = normalize(arr)
        assert normed.shape == arr.shape

    def test_zero_mean_after_normalize(self):
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((50, 3)) * 10 + 5
        normed, _ = normalize(arr)
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-10)

    def test_unit_variance_after_normalize(self):
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((50, 3)) * 10 + 5
        normed, _ = normalize(arr)
        np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-10)

    def test_returns_norm_params(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, params = normalize(arr)
        assert isinstance(params, NormParams)
        assert params.method == "standard"
        assert params.means.shape == (2,)
        assert params.stds.shape == (2,)

    def test_nan_preserved(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        normed, _ = normalize(arr)
        assert np.isnan(normed[1, 0])
        assert not np.isnan(normed[1, 1])

    def test_constant_column_unchanged(self):
        arr = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        normed, params = normalize(arr)
        # constant column: mean subtracted, safe_std = 1.0 → values become 0 - 0 / 1 = 0
        np.testing.assert_allclose(normed[:, 0], 0.0, atol=1e-12)
        assert params.stds[0] == pytest.approx(1.0)

    def test_all_nan_column_stays_nan(self):
        arr = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        normed, params = normalize(arr)
        assert np.all(np.isnan(normed[:, 0]))
        assert params.means[0] == pytest.approx(0.0)
        assert params.stds[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _zero_mean
# ---------------------------------------------------------------------------

class TestZeroMean:
    def test_subtracts_column_means(self):
        arr = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        centred, means = _zero_mean(arr)
        np.testing.assert_allclose(means, [3.0, 20.0])
        np.testing.assert_allclose(centred.mean(axis=0), 0.0, atol=1e-12)

    def test_nan_ignored_in_mean_computation(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        centred, means = _zero_mean(arr)
        assert means[1] == pytest.approx(5.0)  # mean of [4, 6]
        assert np.isnan(centred[0, 1])

    def test_all_nan_column_mean_is_zero(self):
        arr = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        _, means = _zero_mean(arr)
        assert means[0] == pytest.approx(0.0)

    def test_returned_means_shape(self):
        arr = np.ones((5, 4))
        _, means = _zero_mean(arr)
        assert means.shape == (4,)


# ---------------------------------------------------------------------------
# _unit_variance
# ---------------------------------------------------------------------------

class TestUnitVariance:
    def test_scales_to_unit_variance(self):
        arr = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])
        # centre first
        arr = arr - arr.mean(axis=0)
        scaled, stds = _unit_variance(arr)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-10)

    def test_zero_variance_col_safe_std_is_one(self):
        arr = np.zeros((5, 2))
        arr[:, 1] = np.arange(5, dtype=float)
        _, stds = _unit_variance(arr)
        assert stds[0] == pytest.approx(1.0)

    def test_nan_col_safe_std_is_one(self):
        arr = np.full((4, 2), np.nan)
        arr[:, 1] = [1.0, 2.0, 3.0, 4.0]
        _, stds = _unit_variance(arr)
        assert stds[0] == pytest.approx(1.0)

    def test_nan_preserved_in_output(self):
        arr = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]])
        scaled, _ = _unit_variance(arr)
        assert np.isnan(scaled[0, 0])


# ---------------------------------------------------------------------------
# inverse_normalize
# ---------------------------------------------------------------------------

class TestInverseNormalize:
    def test_roundtrip(self):
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((20, 4)) * 5 + 3
        normed, params = normalize(arr)
        recovered = inverse_normalize(normed, params)
        np.testing.assert_allclose(recovered, arr, atol=1e-10)

    def test_nan_roundtrip(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        normed, params = normalize(arr)
        recovered = inverse_normalize(normed, params)
        assert np.isnan(recovered[1, 0])
        np.testing.assert_allclose(recovered[0], arr[0], atol=1e-10)
        np.testing.assert_allclose(recovered[2], arr[2], atol=1e-10)

    def test_formula(self):
        means = np.array([1.0, 2.0])
        stds = np.array([3.0, 4.0])
        params = NormParams(means=means, stds=stds, method="standard")
        normed = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = inverse_normalize(normed, params)
        expected = np.array([[1.0, 2.0], [4.0, 6.0]])
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# _select_method
# ---------------------------------------------------------------------------

class TestSelectMethod:
    def test_always_returns_standard(self):
        arr = np.ones((5, 3))
        assert _select_method(arr) == "standard"

    def test_result_is_string(self):
        assert isinstance(_select_method(np.zeros((2, 2))), str)
