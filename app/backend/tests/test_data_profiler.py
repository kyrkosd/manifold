"""Tests for backend/services/data_profiler.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
##import pytest

from backend.models.enums import SuitabilityLevel
from backend.services.data_profiler import profile


def _make_clean(n: int = 200, k: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((n, k)), columns=[f"x{i}" for i in range(k)])


def _make_messy() -> pd.DataFrame:
    n = 30
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": [1.0] * n,  # constant
    })
    # 50% missing in a
    df.loc[:14, "a"] = None
    # duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


class TestProfileCleanData:
    """Verify profiler output for a well-formed numeric DataFrame."""

    def test_high_score(self):
        """Verify a clean dataset receives a suitability score above 0.9."""
        q = profile(_make_clean())
        assert q.suitability_score > 0.9

    def test_ready_level(self):
        """Verify a clean dataset is classified as READY."""
        q = profile(_make_clean())
        assert q.suitability_level == SuitabilityLevel.READY

    def test_no_warnings(self):
        """Verify a clean dataset produces no quality warnings."""
        q = profile(_make_clean())
        assert q.warnings == []

    def test_counts_correct(self):
        """Verify row and column counts in the quality report."""
        q = profile(_make_clean(n=200, k=5))
        assert q.n_rows == 200
        assert q.n_numeric_columns == 5
        assert q.n_non_numeric_columns == 0


class TestProfileMessyData:
    """Verify profiler detects and warns about data quality issues."""

    def test_low_score(self):
        """Verify a messy dataset receives a suitability score below 0.7."""
        q = profile(_make_messy())
        assert q.suitability_score < 0.7

    def test_warns_missing(self):
        """Verify a warning about missing values is present."""
        q = profile(_make_messy())
        assert any("missing" in w.lower() for w in q.warnings)

    def test_warns_constant(self):
        """Verify a warning about constant columns is present."""
        q = profile(_make_messy())
        assert any("constant" in w.lower() for w in q.warnings)

    def test_constant_col_listed(self):
        """Verify the constant column name appears in constant_columns."""
        q = profile(_make_messy())
        assert "c" in q.constant_columns


class TestProfileEdgeCases:
    """Verify profiler handles edge cases gracefully."""

    def test_tiny_dataset_warns(self):
        """Verify a tiny dataset triggers a row-count warning and NOT_SUITABLE level."""
        tiny = _make_clean(n=5, k=2)
        q = profile(tiny)
        assert any("row" in w.lower() for w in q.warnings)
        assert q.suitability_level == SuitabilityLevel.NOT_SUITABLE

    def test_no_numeric_columns(self):
        """Verify a string-only DataFrame scores 0.0 and is NOT_SUITABLE."""
        df = pd.DataFrame({"a": ["x", "y", "z"] * 40, "b": ["p", "q", "r"] * 40})
        q = profile(df)
        assert q.suitability_score == 0.0
        assert q.suitability_level == SuitabilityLevel.NOT_SUITABLE
        assert q.n_numeric_columns == 0

    def test_empty_dataframe(self):
        """Verify an empty DataFrame produces n_rows=0 without raising."""
        q = profile(pd.DataFrame())
        assert q.n_rows == 0

    def test_runtime_estimate_positive(self):
        """Verify the estimated runtime is at least the minimum floor of 5 s."""
        q = profile(_make_clean())
        assert q.estimated_runtime_seconds >= 5.0
