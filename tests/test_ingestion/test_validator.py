"""Tests for ingestion/validator.py.

Tests are split across two concerns:
  - Fatal validation errors that raise ValidationError and abort the pipeline.
  - Non-fatal quality checks that produce warnings and populate QualityReport.

Private helpers are tested directly so that coverage catches regressions
before they surface through the public validate() interface.
"""
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
# validate() — shared helper
# ---------------------------------------------------------------------------

# Reused across tests; seed 0 gives reproducible random data with no NaNs or infs.
def _valid_array(n: int = 20, d: int = 3) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n, d))


# ---------------------------------------------------------------------------
# validate() — fatal error paths
# ---------------------------------------------------------------------------

class TestValidateErrors:
    def test_non_numeric_ndarray_raises(self):
        # String dtype cannot be cast to float64; pipeline must abort immediately.
        with pytest.raises(ValidationError):
            validate(np.array([["a", "b"], ["c", "d"]]))

    def test_non_numeric_dataframe_raises(self):
        # Mixed-type DataFrames with object columns must be rejected at the gate.
        with pytest.raises(ValidationError):
            validate(pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]}))

    def test_single_row_raises(self):
        # Fewer than 2 rows → cannot compute inter-sample structure.
        with pytest.raises(ValidationError):
            validate(np.ones((1, 3)))

    def test_single_column_raises(self):
        # Fewer than 2 columns → cannot build a feature graph.
        with pytest.raises(ValidationError):
            validate(np.ones((5, 1)))

    def test_1d_array_raises(self):
        # 1-D input has no column axis; dimension check catches it before conversion.
        with pytest.raises(ValidationError):
            validate(np.ones(5))

    def test_positive_infinite_raises(self):
        # +inf cannot be normalised; its presence terminates validation.
        arr = _valid_array()
        arr[0, 0] = np.inf
        with pytest.raises(ValidationError):
            validate(arr)

    def test_negative_infinite_raises(self):
        # -inf is equally illegal; the range check does not distinguish sign.
        arr = _valid_array()
        arr[1, 2] = -np.inf
        with pytest.raises(ValidationError):
            validate(arr)


# ---------------------------------------------------------------------------
# validate() — successful-path output
# ---------------------------------------------------------------------------

class TestValidateSuccess:
    def test_ndarray_output_shape_and_dtype(self):
        # Validated output must be float64 and preserve the input shape.
        arr = _valid_array()
        result = validate(arr)
        assert result.data.dtype == np.float64
        assert result.data.shape == arr.shape
        # QualityReport dimensions must reflect the validated array, not the input.
        assert result.quality.n_samples == arr.shape[0]
        assert result.quality.n_features == arr.shape[1]

    def test_dataframe_converted_correctly(self):
        # DataFrames must be converted to a float64 ndarray with the same shape.
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
        result = validate(df)
        assert isinstance(result.data, np.ndarray)
        assert result.data.dtype == np.float64
        assert result.data.shape == (4, 2)  # rows = samples, cols = features

    def test_integer_array_converted_to_float64(self):
        # Integer arrays are a valid input type; dtype must be promoted.
        arr = np.arange(20).reshape(4, 5).astype(np.int32)
        assert validate(arr).data.dtype == np.float64

    def test_min_2x2_passes(self):
        # The minimum acceptable shape is (2, 2); verify the boundary is correct.
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert validate(arr).data.shape == (2, 2)

    def test_nan_values_are_allowed(self):
        # NaN is a non-fatal issue; it is allowed and reflected in missing_pct.
        arr = _valid_array()
        arr[0, 0] = np.nan
        assert validate(arr).quality.missing_pct > 0.0

    def test_dataframe_with_nan(self):
        # NaN inside a DataFrame must also be tracked in the quality report.
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
        assert validate(df).quality.missing_pct > 0.0

    def test_constant_column_reported(self, caplog):
        # A zero-variance column triggers a WARNING and is listed in constant_dims.
        import logging
        arr = _valid_array()
        arr[:, 1] = 5.0  # overwrite column 1 with a constant value
        with caplog.at_level(logging.WARNING, logger="fmas"):
            result = validate(arr)
        assert 1 in result.quality.constant_dims  # column index 1 must be flagged

    def test_duplicate_rows_detected(self):
        # All-identical rows are detected and counted in the quality report.
        arr = np.vstack([np.ones((1, 3))] * 20)
        assert validate(arr).quality.duplicate_count > 0

    def test_suitability_score_is_in_unit_range(self):
        # Score is always clamped to [0, 1] regardless of deduction magnitude.
        result = validate(_valid_array())
        assert 0.0 <= result.quality.suitability_score <= 1.0


# ---------------------------------------------------------------------------
# _check_dimensions
# ---------------------------------------------------------------------------

class TestCheckDimensions:
    def test_valid_2d(self):
        assert _check_dimensions(np.ones((5, 3))) is True

    def test_exactly_2x2(self):
        # (2, 2) is the minimum; boundary must be included.
        assert _check_dimensions(np.ones((2, 2))) is True

    def test_1d_fails(self):
        # ndim==1 violates the 2-D requirement.
        assert _check_dimensions(np.ones(5)) is False

    def test_single_row_fails(self):
        # shape[0]==1 < 2; rejected even if ncols is sufficient.
        assert _check_dimensions(np.ones((1, 5))) is False

    def test_single_col_fails(self):
        # shape[1]==1 < 2; rejected even if nrows is sufficient.
        assert _check_dimensions(np.ones((5, 1))) is False

    def test_3d_fails(self):
        # 3-D arrays cannot be treated as a 2-D feature matrix.
        assert _check_dimensions(np.ones((3, 3, 3))) is False


# ---------------------------------------------------------------------------
# _check_completeness
# ---------------------------------------------------------------------------

class TestCheckCompleteness:
    def test_no_nans(self):
        # All values present → completeness = 1.0.
        assert _check_completeness(np.ones((5, 3))) == pytest.approx(1.0)

    def test_all_nans(self):
        # All values missing → completeness = 0.0.
        assert _check_completeness(np.full((5, 3), np.nan)) == pytest.approx(0.0)

    def test_half_nans(self):
        # Two of four values are NaN → completeness = 0.5.
        arr = np.ones((2, 2))
        arr[0, 0] = np.nan
        arr[1, 1] = np.nan
        assert _check_completeness(arr) == pytest.approx(0.5)

    def test_empty_array_returns_1(self):
        # Zero-element arrays have no missing values by convention.
        assert _check_completeness(np.empty((0, 0))) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _check_types
# ---------------------------------------------------------------------------

class TestCheckTypes:
    def test_float_array_ok(self):
        # Default float64 dtype is numeric.
        assert _check_types(np.ones((3, 3))) is True

    def test_int_array_ok(self):
        # Integer dtypes are also numeric and accepted.
        assert _check_types(np.ones((3, 3), dtype=np.int32)) is True

    def test_string_array_fails(self):
        # Object/string dtype is non-numeric; must be rejected.
        assert _check_types(np.array([["a", "b"]])) is False

    def test_numeric_dataframe_ok(self):
        # All-float columns pass the DataFrame dtype check.
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        assert _check_types(df) is True

    def test_mixed_dataframe_fails(self):
        # One non-numeric column contaminates the entire DataFrame.
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        assert _check_types(df) is False

    def test_all_string_dataframe_fails(self):
        # Fully non-numeric DataFrame is also rejected.
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        assert _check_types(df) is False


# ---------------------------------------------------------------------------
# _check_ranges
# ---------------------------------------------------------------------------

class TestCheckRanges:
    def test_finite_ok(self):
        # All finite values; no infinities → passes.
        assert _check_ranges(np.ones((3, 3))) is True

    def test_nan_allowed(self):
        # NaN is not infinite; the range check specifically permits it.
        arr = np.array([[1.0, np.nan], [2.0, 3.0]])
        assert _check_ranges(arr) is True

    def test_pos_inf_rejected(self):
        # +inf is not finite and not NaN → fails.
        arr = np.array([[1.0, np.inf], [2.0, 3.0]])
        assert _check_ranges(arr) is False

    def test_neg_inf_rejected(self):
        # -inf is equally unacceptable; sign does not matter.
        arr = np.array([[1.0, -np.inf], [2.0, 3.0]])
        assert _check_ranges(arr) is False


# ---------------------------------------------------------------------------
# _check_constant_dims
# ---------------------------------------------------------------------------

class TestCheckConstantDims:
    def test_no_constant(self):
        # Random normal data almost certainly has non-zero variance in every column.
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((10, 3))
        assert _check_constant_dims(arr) == []

    def test_one_constant_col(self):
        # Column 1 is the only constant one; it must be the sole flagged index.
        arr = np.ones((5, 3))
        arr[:, 0] = np.random.standard_normal(5)  # variable
        arr[:, 2] = np.random.standard_normal(5)  # variable
        result = _check_constant_dims(arr)
        assert 1 in result

    def test_all_nan_col_flagged(self):
        # nanstd of an all-NaN column is NaN; treated the same as zero variance.
        arr = np.ones((5, 2))
        arr[:, 1] = np.nan
        result = _check_constant_dims(arr)
        assert 1 in result

    def test_all_constant(self):
        # Every column is constant; all three indices must appear in the result.
        arr = np.full((5, 3), 7.0)
        assert sorted(_check_constant_dims(arr)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# _check_duplicates
# ---------------------------------------------------------------------------

class TestCheckDuplicates:
    def test_no_duplicates(self):
        # Random normal rows are almost certainly unique.
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((10, 3))
        assert _check_duplicates(arr) == 0

    def test_with_duplicates(self):
        # Row [1, 2, 3] appears 3 times; 2 are beyond the first occurrence.
        row = np.array([[1.0, 2.0, 3.0]])
        arr = np.vstack([row, row, row, np.array([[4.0, 5.0, 6.0]])])
        assert _check_duplicates(arr) == 2  # two duplicates beyond first occurrence

    def test_all_identical(self):
        # Five identical rows → 4 duplicates (first occurrence is not counted).
        arr = np.ones((5, 2))
        assert _check_duplicates(arr) == 4


# ---------------------------------------------------------------------------
# _check_min_samples
# ---------------------------------------------------------------------------

class TestCheckMinSamples:
    def test_sufficient_samples(self):
        # 30 rows / 3 features = 10×; meets the rule-of-thumb exactly.
        arr = np.ones((30, 3))
        assert _check_min_samples(arr) is True

    def test_exactly_at_threshold(self):
        # The boundary (n == 10d) must be accepted, not rejected.
        arr = np.ones((30, 3))
        assert _check_min_samples(arr) is True

    def test_insufficient_samples(self):
        # 5 rows / 3 features ≈ 1.7×; far below the 10× heuristic.
        arr = np.ones((5, 3))
        assert _check_min_samples(arr) is False


# ---------------------------------------------------------------------------
# _estimate_suitability
# ---------------------------------------------------------------------------

class TestEstimateSuitability:
    # Suitability formula: 1.0 - 0.5*(missing/100) - 0.3*(const/features) - 0.2*(dup/samples)
    # Each deduction term is independent and the result is clamped to [0, 1].

    def _make_report(self, missing_pct=0.0, constant_dims=None, duplicate_count=0,
                     n_samples=100, n_features=5):
        # Helper creates a QualityReport with suitability_score=0.0 (to be computed).
        return QualityReport(
            n_samples=n_samples,
            n_features=n_features,
            missing_pct=missing_pct,
            duplicate_count=duplicate_count,
            constant_dims=constant_dims or [],
            suitability_score=0.0,
        )

    def test_perfect_data(self):
        # No issues → no deductions → score = 1.0.
        r = self._make_report()
        score = _estimate_suitability(r)
        assert score == pytest.approx(1.0)

    def test_all_missing_deducts_half(self):
        # 100 % missing → deduction = 0.5 * 1.0 = 0.5 → score = 0.5.
        r = self._make_report(missing_pct=100.0)
        score = _estimate_suitability(r)
        assert score == pytest.approx(0.5)

    def test_all_constant_dims_deducts_30pct(self):
        # All 5 of 5 features are constant → deduction = 0.3 * 1.0 = 0.3 → score = 0.7.
        r = self._make_report(constant_dims=[0, 1, 2, 3, 4], n_features=5)
        score = _estimate_suitability(r)
        assert score == pytest.approx(0.7)

    def test_score_clamped_to_zero(self):
        # Combined deductions exceed 1.0; clamping must prevent negative scores.
        r = self._make_report(missing_pct=100.0, constant_dims=[0, 1, 2, 3, 4],
                              duplicate_count=100)
        score = _estimate_suitability(r)
        assert score == pytest.approx(0.0)  # clamped; raw would be negative

    def test_score_never_exceeds_one(self):
        # Even with no issues the score must stay ≤ 1.0 (no bonus points).
        r = self._make_report()
        score = _estimate_suitability(r)
        assert score <= 1.0
