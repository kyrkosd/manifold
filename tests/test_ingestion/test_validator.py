"""Tests for ingestion/validator.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.exceptions import ValidationError
from ingestion.validator import (
    _check_completeness,
    _check_constant_dims,
    _check_dimensions,
    _check_duplicates,
    _check_min_samples,
    _check_ranges,
    _check_types,
    _estimate_suitability,
    _generate_quality_report,
    validate,
)
from ingestion.types import QualityReport


# ---------------------------------------------------------------------------
# validate() — public entry point
# ---------------------------------------------------------------------------

class TestValidate:
    def _valid_array(self, n=20, d=3):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, d))

    def test_valid_ndarray_returns_validated_data(self):
        arr = self._valid_array()
        result = validate(arr)
        assert result.data.dtype == np.float64
        assert result.data.shape == arr.shape
        assert result.quality.n_samples == arr.shape[0]
        assert result.quality.n_features == arr.shape[1]

    def test_valid_dataframe_converted_correctly(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
        result = validate(df)
        assert isinstance(result.data, np.ndarray)
        assert result.data.dtype == np.float64
        assert result.data.shape == (4, 2)

    def test_non_numeric_ndarray_raises(self):
        arr = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(ValidationError):
            validate(arr)

    def test_non_numeric_dataframe_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        with pytest.raises(ValidationError):
            validate(df)

    def test_single_row_raises(self):
        arr = np.ones((1, 3))
        with pytest.raises(ValidationError):
            validate(arr)

    def test_single_column_raises(self):
        arr = np.ones((5, 1))
        with pytest.raises(ValidationError):
            validate(arr)

    def test_1d_array_raises(self):
        arr = np.ones(5)
        with pytest.raises(ValidationError):
            validate(arr)

    def test_infinite_values_raises(self):
        arr = self._valid_array()
        arr[0, 0] = np.inf
        with pytest.raises(ValidationError):
            validate(arr)

    def test_negative_infinite_raises(self):
        arr = self._valid_array()
        arr[1, 2] = -np.inf
        with pytest.raises(ValidationError):
            validate(arr)

    def test_nan_values_are_allowed(self):
        arr = self._valid_array()
        arr[0, 0] = np.nan
        result = validate(arr)
        assert result.quality.missing_pct > 0.0

    def test_constant_column_logged_as_warning(self, caplog):
        import logging
        arr = self._valid_array()
        arr[:, 1] = 5.0  # constant column
        with caplog.at_level(logging.WARNING, logger="fmas"):
            result = validate(arr)
        assert 1 in result.quality.constant_dims

    def test_duplicate_rows_detected(self):
        row = np.ones((1, 3))
        arr = np.vstack([row] * 20)  # all duplicates
        result = validate(arr)
        assert result.quality.duplicate_count > 0

    def test_suitability_score_is_in_range(self):
        arr = self._valid_array()
        result = validate(arr)
        assert 0.0 <= result.quality.suitability_score <= 1.0

    def test_integer_array_converted_to_float64(self):
        arr = np.arange(20).reshape(4, 5).astype(np.int32)
        result = validate(arr)
        assert result.data.dtype == np.float64

    def test_dataframe_with_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
        result = validate(df)
        assert result.quality.missing_pct > 0.0

    def test_min_2x2_passes(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate(arr)
        assert result.data.shape == (2, 2)


# ---------------------------------------------------------------------------
# _check_dimensions
# ---------------------------------------------------------------------------

class TestCheckDimensions:
    def test_valid_2d(self):
        assert _check_dimensions(np.ones((5, 3))) is True

    def test_exactly_2x2(self):
        assert _check_dimensions(np.ones((2, 2))) is True

    def test_1d_fails(self):
        assert _check_dimensions(np.ones(5)) is False

    def test_single_row_fails(self):
        assert _check_dimensions(np.ones((1, 5))) is False

    def test_single_col_fails(self):
        assert _check_dimensions(np.ones((5, 1))) is False

    def test_3d_fails(self):
        assert _check_dimensions(np.ones((3, 3, 3))) is False


# ---------------------------------------------------------------------------
# _check_completeness
# ---------------------------------------------------------------------------

class TestCheckCompleteness:
    def test_no_nans(self):
        assert _check_completeness(np.ones((5, 3))) == pytest.approx(1.0)

    def test_all_nans(self):
        assert _check_completeness(np.full((5, 3), np.nan)) == pytest.approx(0.0)

    def test_half_nans(self):
        arr = np.ones((2, 2))
        arr[0, 0] = np.nan
        arr[1, 1] = np.nan
        assert _check_completeness(arr) == pytest.approx(0.5)

    def test_empty_array_returns_1(self):
        assert _check_completeness(np.empty((0, 0))) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _check_types
# ---------------------------------------------------------------------------

class TestCheckTypes:
    def test_float_array_ok(self):
        assert _check_types(np.ones((3, 3))) is True

    def test_int_array_ok(self):
        assert _check_types(np.ones((3, 3), dtype=np.int32)) is True

    def test_string_array_fails(self):
        assert _check_types(np.array([["a", "b"]])) is False

    def test_numeric_dataframe_ok(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        assert _check_types(df) is True

    def test_mixed_dataframe_fails(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        assert _check_types(df) is False

    def test_all_string_dataframe_fails(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        assert _check_types(df) is False


# ---------------------------------------------------------------------------
# _check_ranges
# ---------------------------------------------------------------------------

class TestCheckRanges:
    def test_finite_ok(self):
        assert _check_ranges(np.ones((3, 3))) is True

    def test_nan_allowed(self):
        arr = np.array([[1.0, np.nan], [2.0, 3.0]])
        assert _check_ranges(arr) is True

    def test_pos_inf_rejected(self):
        arr = np.array([[1.0, np.inf], [2.0, 3.0]])
        assert _check_ranges(arr) is False

    def test_neg_inf_rejected(self):
        arr = np.array([[1.0, -np.inf], [2.0, 3.0]])
        assert _check_ranges(arr) is False


# ---------------------------------------------------------------------------
# _check_constant_dims
# ---------------------------------------------------------------------------

class TestCheckConstantDims:
    def test_no_constant(self):
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((10, 3))
        assert _check_constant_dims(arr) == []

    def test_one_constant_col(self):
        arr = np.ones((5, 3))
        arr[:, 0] = np.random.standard_normal(5)
        arr[:, 2] = np.random.standard_normal(5)
        result = _check_constant_dims(arr)
        assert 1 in result

    def test_all_nan_col_flagged(self):
        arr = np.ones((5, 2))
        arr[:, 1] = np.nan
        result = _check_constant_dims(arr)
        assert 1 in result

    def test_all_constant(self):
        arr = np.full((5, 3), 7.0)
        assert sorted(_check_constant_dims(arr)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# _check_duplicates
# ---------------------------------------------------------------------------

class TestCheckDuplicates:
    def test_no_duplicates(self):
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((10, 3))
        assert _check_duplicates(arr) == 0

    def test_with_duplicates(self):
        row = np.array([[1.0, 2.0, 3.0]])
        arr = np.vstack([row, row, row, np.array([[4.0, 5.0, 6.0]])])
        assert _check_duplicates(arr) == 2  # two duplicates beyond first occurrence

    def test_all_identical(self):
        arr = np.ones((5, 2))
        assert _check_duplicates(arr) == 4


# ---------------------------------------------------------------------------
# _check_min_samples
# ---------------------------------------------------------------------------

class TestCheckMinSamples:
    def test_sufficient_samples(self):
        arr = np.ones((30, 3))
        assert _check_min_samples(arr) is True

    def test_exactly_at_threshold(self):
        arr = np.ones((30, 3))
        assert _check_min_samples(arr) is True

    def test_insufficient_samples(self):
        arr = np.ones((5, 3))
        assert _check_min_samples(arr) is False


# ---------------------------------------------------------------------------
# _estimate_suitability
# ---------------------------------------------------------------------------

class TestEstimateSuitability:
    def _make_report(self, missing_pct=0.0, constant_dims=None, duplicate_count=0, n_samples=100, n_features=5):
        return QualityReport(
            n_samples=n_samples,
            n_features=n_features,
            missing_pct=missing_pct,
            duplicate_count=duplicate_count,
            constant_dims=constant_dims or [],
            suitability_score=0.0,
        )

    def test_perfect_data(self):
        r = self._make_report()
        score = _estimate_suitability(r)
        assert score == pytest.approx(1.0)

    def test_all_missing_deducts_half(self):
        r = self._make_report(missing_pct=100.0)
        score = _estimate_suitability(r)
        assert score == pytest.approx(0.5)

    def test_all_constant_dims_deducts_30pct(self):
        r = self._make_report(constant_dims=[0, 1, 2, 3, 4], n_features=5)
        score = _estimate_suitability(r)
        assert score == pytest.approx(0.7)

    def test_score_clamped_to_zero(self):
        r = self._make_report(missing_pct=100.0, constant_dims=[0, 1, 2, 3, 4], duplicate_count=100)
        score = _estimate_suitability(r)
        assert score == pytest.approx(0.0)

    def test_score_never_exceeds_one(self):
        r = self._make_report()
        score = _estimate_suitability(r)
        assert score <= 1.0
