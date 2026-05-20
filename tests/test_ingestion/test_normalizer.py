"""Tests for ingestion/normalizer.py.

Covers the public normalize() / inverse_normalize() API and the private
helpers _zero_mean(), _unit_variance(), and _select_method().  Each test
targets one specific contract: NaN propagation, degenerate-column handling,
parameter shapes, or mathematical correctness.
"""
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
        # Normalisation must not alter the (n, d) shape of the input.
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((20, 4))
        normed, params = normalize(arr)
        assert normed.shape == arr.shape

    def test_zero_mean_after_normalize(self):
        # After standardisation each column must have mean ≈ 0.
        # Seed 1 + scale/shift ensure the raw data is far from zero-mean.
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((50, 3)) * 10 + 5
        normed, _ = normalize(arr)
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-10)

    def test_unit_variance_after_normalize(self):
        # After standardisation each column must have std ≈ 1.
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((50, 3)) * 10 + 5
        normed, _ = normalize(arr)
        np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-10)

    def test_returns_norm_params(self):
        # The returned NormParams must have the correct shapes and method tag.
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, params = normalize(arr)
        assert isinstance(params, NormParams)
        assert params.method == "standard"   # only supported method in MVP
        assert params.means.shape == (2,)   # one mean per column
        assert params.stds.shape == (2,)    # one std per column

    def test_nan_preserved(self):
        # NaN values must propagate through unchanged rather than being imputed.
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        normed, _ = normalize(arr)
        assert np.isnan(normed[1, 0])       # NaN in column 0 stays NaN
        assert not np.isnan(normed[1, 1])   # column 1 had no NaN at this row

    def test_constant_column_unchanged(self):
        # Column 0 is constant (value 5); mean=5, safe_std=1 → output = 0.
        arr = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        normed, params = normalize(arr)
        # constant column: mean subtracted then divided by safe_std (1.0) → 0
        np.testing.assert_allclose(normed[:, 0], 0.0, atol=1e-12)
        assert params.stds[0] == pytest.approx(1.0)  # safe_std avoids division by zero

    def test_all_nan_column_stays_nan(self):
        # An all-NaN column has no mean or std; values remain NaN, params are safe.
        arr = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        normed, params = normalize(arr)
        assert np.all(np.isnan(normed[:, 0]))      # all NaN → stays all NaN
        assert params.means[0] == pytest.approx(0.0)  # fallback mean = 0
        assert params.stds[0] == pytest.approx(1.0)   # fallback std = 1 (no scaling)


# ---------------------------------------------------------------------------
# _zero_mean
# ---------------------------------------------------------------------------

class TestZeroMean:
    def test_subtracts_column_means(self):
        # Column 0 mean = 3.0, column 1 mean = 20.0; centred columns must be 0-mean.
        arr = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        centred, means = _zero_mean(arr)
        np.testing.assert_allclose(means, [3.0, 20.0])
        np.testing.assert_allclose(centred.mean(axis=0), 0.0, atol=1e-12)

    def test_nan_ignored_in_mean_computation(self):
        # Column 1 has one NaN; mean computed over remaining values [4, 6] = 5.
        arr = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        centred, means = _zero_mean(arr)
        assert means[1] == pytest.approx(5.0)  # mean of [4, 6]
        assert np.isnan(centred[0, 1])          # NaN propagates to output

    def test_all_nan_column_mean_is_zero(self):
        # nanmean of an all-NaN slice returns NaN; the function replaces it with 0.
        arr = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        _, means = _zero_mean(arr)
        assert means[0] == pytest.approx(0.0)  # sentinel: no centering applied

    def test_returned_means_shape(self):
        # Output means must be (d,) regardless of n.
        arr = np.ones((5, 4))
        _, means = _zero_mean(arr)
        assert means.shape == (4,)


# ---------------------------------------------------------------------------
# _unit_variance
# ---------------------------------------------------------------------------

class TestUnitVariance:
    def test_scales_to_unit_variance(self):
        # Pre-centred data should have std exactly 1 after unit-variance scaling.
        arr = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])
        arr = arr - arr.mean(axis=0)  # centre first so ddof=0 std is exact
        scaled, stds = _unit_variance(arr)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-10)

    def test_zero_variance_col_safe_std_is_one(self):
        # Column 0 is all zeros (std=0); safe_std must be 1.0 to prevent NaN output.
        arr = np.zeros((5, 2))
        arr[:, 1] = np.arange(5, dtype=float)  # column 1 has non-zero variance
        _, stds = _unit_variance(arr)
        assert stds[0] == pytest.approx(1.0)  # degenerate column → no scaling

    def test_nan_col_safe_std_is_one(self):
        # An all-NaN column has undefined std; safe_std fallback must be 1.0.
        arr = np.full((4, 2), np.nan)
        arr[:, 1] = [1.0, 2.0, 3.0, 4.0]  # column 1 is well-defined
        _, stds = _unit_variance(arr)
        assert stds[0] == pytest.approx(1.0)

    def test_nan_preserved_in_output(self):
        # NaN in input must propagate to the scaled output without raising.
        arr = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]])
        scaled, _ = _unit_variance(arr)
        assert np.isnan(scaled[0, 0])  # NaN / safe_std is still NaN


# ---------------------------------------------------------------------------
# inverse_normalize
# ---------------------------------------------------------------------------

class TestInverseNormalize:
    def test_roundtrip(self):
        # Forward then inverse must recover the original array up to float precision.
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((20, 4)) * 5 + 3  # non-trivial mean and scale
        normed, params = normalize(arr)
        recovered = inverse_normalize(normed, params)
        np.testing.assert_allclose(recovered, arr, atol=1e-10)

    def test_nan_roundtrip(self):
        # NaN values must survive the round-trip; non-NaN values must be exact.
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        normed, params = normalize(arr)
        recovered = inverse_normalize(normed, params)
        assert np.isnan(recovered[1, 0])                              # NaN preserved
        np.testing.assert_allclose(recovered[0], arr[0], atol=1e-10)  # row 0 exact
        np.testing.assert_allclose(recovered[2], arr[2], atol=1e-10)  # row 2 exact

    def test_formula(self):
        # Explicit verification: inverse(x) = x * std + mean.
        # normed[0] = [0, 0] → recovered = [0*3+1, 0*4+2] = [1, 2]
        # normed[1] = [1, 1] → recovered = [1*3+1, 1*4+2] = [4, 6]
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
        # In the MVP only "standard" normalisation is supported.
        arr = np.ones((5, 3))
        assert _select_method(arr) == "standard"

    def test_result_is_string(self):
        # The return type contract: must be str regardless of input shape.
        assert isinstance(_select_method(np.zeros((2, 2))), str)
