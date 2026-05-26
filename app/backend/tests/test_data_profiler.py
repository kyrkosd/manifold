"""Tests for backend/services/data_profiler.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    def test_high_score(self):
        q = profile(_make_clean())
        assert q.suitability_score > 0.9

    def test_ready_level(self):
        q = profile(_make_clean())
        assert q.suitability_level == SuitabilityLevel.READY

    def test_no_warnings(self):
        q = profile(_make_clean())
        assert q.warnings == []

    def test_counts_correct(self):
        q = profile(_make_clean(n=200, k=5))
        assert q.n_rows == 200
        assert q.n_numeric_columns == 5
        assert q.n_non_numeric_columns == 0


class TestProfileMessyData:
    def test_low_score(self):
        q = profile(_make_messy())
        assert q.suitability_score < 0.7

    def test_warns_missing(self):
        q = profile(_make_messy())
        assert any("missing" in w.lower() for w in q.warnings)

    def test_warns_constant(self):
        q = profile(_make_messy())
        assert any("constant" in w.lower() for w in q.warnings)

    def test_constant_col_listed(self):
        q = profile(_make_messy())
        assert "c" in q.constant_columns


class TestProfileEdgeCases:
    def test_tiny_dataset_warns(self):
        tiny = _make_clean(n=5, k=2)
        q = profile(tiny)
        assert any("row" in w.lower() for w in q.warnings)
        assert q.suitability_level == SuitabilityLevel.NOT_SUITABLE

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"] * 40, "b": ["p", "q", "r"] * 40})
        q = profile(df)
        assert q.suitability_score == 0.0
        assert q.suitability_level == SuitabilityLevel.NOT_SUITABLE
        assert q.n_numeric_columns == 0

    def test_empty_dataframe(self):
        q = profile(pd.DataFrame())
        assert q.n_rows == 0

    def test_runtime_estimate_positive(self):
        q = profile(_make_clean())
        assert q.estimated_runtime_seconds >= 5.0
