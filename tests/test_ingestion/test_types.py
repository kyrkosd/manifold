"""Tests for ingestion/types.py dataclasses.

Verifies the data contracts — field storage, default values, and the
independence of mutable defaults — that downstream pipeline stages rely on.
"""
from __future__ import annotations

import numpy as np

from ingestion.types import CleanData, NormParams, QualityReport
# ValidatedData is an internal ingestion type defined in validator.py, not types.py.


# ---------------------------------------------------------------------------
# NormParams — normalisation parameter bundle
# ---------------------------------------------------------------------------

class TestNormParams:
    """Tests for Norm Params."""
    def test_basic_construction(self):
        # All three fields must be stored exactly as provided.
        means = np.array([1.0, 2.0])
        stds = np.array([0.5, 1.5])
        p = NormParams(means=means, stds=stds, method="standard")
        np.testing.assert_array_equal(p.means, means)
        np.testing.assert_array_equal(p.stds, stds)
        assert p.method == "standard"

    def test_fields_are_stored_by_reference(self):
        # No defensive copy is made; callers own the arrays.
        means = np.zeros(3)
        stds = np.ones(3)
        p = NormParams(means=means, stds=stds, method="standard")
        assert p.means is means   # identity check, not equality
        assert p.stds is stds


# ---------------------------------------------------------------------------
# QualityReport — data-quality summary consumed by downstream phases
# ---------------------------------------------------------------------------

class TestQualityReport:
    """Tests for Quality Report."""
    def test_defaults(self):
        # constant_dims defaults to empty list; suitability_score defaults to 1.0.
        r = QualityReport(n_samples=10, n_features=3, missing_pct=0.0, duplicate_count=0)
        assert r.constant_dims == []
        assert r.suitability_score == 1.0  # perfect score for clean data

    def test_full_construction(self):
        # All fields must survive round-trip through the dataclass constructor.
        r = QualityReport(
            n_samples=100,
            n_features=5,
            missing_pct=10.0,       # 10 % of values are NaN
            duplicate_count=3,
            constant_dims=[1, 4],   # columns 1 and 4 have zero variance
            suitability_score=0.7,
        )
        assert r.n_samples == 100
        assert r.n_features == 5
        assert r.missing_pct == 10.0
        assert r.duplicate_count == 3
        assert r.constant_dims == [1, 4]
        assert r.suitability_score == 0.7

    def test_constant_dims_default_is_independent(self):
        # Each instance gets its own list; mutating one must not affect the other.
        r1 = QualityReport(n_samples=5, n_features=2, missing_pct=0.0, duplicate_count=0)
        r2 = QualityReport(n_samples=5, n_features=2, missing_pct=0.0, duplicate_count=0)
        r1.constant_dims.append(0)   # mutate r1 only
        assert r2.constant_dims == []  # r2 must be unaffected


# ---------------------------------------------------------------------------
# CleanData — post-normalisation data ready for downstream phases
# ---------------------------------------------------------------------------

class TestCleanData:
    """Tests for Clean Data."""
    def test_construction(self):
        # CleanData bundles the normalised array, parameters, and quality report.
        arr = np.zeros((5, 2))
        params = NormParams(means=np.zeros(2), stds=np.ones(2), method="standard")
        report = QualityReport(n_samples=5, n_features=2, missing_pct=0.0, duplicate_count=0)
        cd = CleanData(data=arr, norm_params=params, quality=report)
        assert cd.data is arr           # identity, not equality
        assert cd.norm_params is params
        assert cd.quality is report
