"""Tests for ingestion/types.py dataclasses."""
from __future__ import annotations

import numpy as np
import pytest

from ingestion.types import CleanData, NormParams, QualityReport, ValidatedData


class TestNormParams:
    def test_basic_construction(self):
        means = np.array([1.0, 2.0])
        stds = np.array([0.5, 1.5])
        p = NormParams(means=means, stds=stds, method="standard")
        np.testing.assert_array_equal(p.means, means)
        np.testing.assert_array_equal(p.stds, stds)
        assert p.method == "standard"

    def test_fields_are_stored_by_reference(self):
        means = np.zeros(3)
        stds = np.ones(3)
        p = NormParams(means=means, stds=stds, method="standard")
        assert p.means is means
        assert p.stds is stds


class TestQualityReport:
    def test_defaults(self):
        r = QualityReport(n_samples=10, n_features=3, missing_pct=0.0, duplicate_count=0)
        assert r.constant_dims == []
        assert r.suitability_score == 1.0

    def test_full_construction(self):
        r = QualityReport(
            n_samples=100,
            n_features=5,
            missing_pct=10.0,
            duplicate_count=3,
            constant_dims=[1, 4],
            suitability_score=0.7,
        )
        assert r.n_samples == 100
        assert r.n_features == 5
        assert r.missing_pct == 10.0
        assert r.duplicate_count == 3
        assert r.constant_dims == [1, 4]
        assert r.suitability_score == 0.7

    def test_constant_dims_default_is_independent(self):
        r1 = QualityReport(n_samples=5, n_features=2, missing_pct=0.0, duplicate_count=0)
        r2 = QualityReport(n_samples=5, n_features=2, missing_pct=0.0, duplicate_count=0)
        r1.constant_dims.append(0)
        assert r2.constant_dims == []


class TestValidatedData:
    def test_construction(self):
        arr = np.ones((10, 3))
        report = QualityReport(n_samples=10, n_features=3, missing_pct=0.0, duplicate_count=0)
        vd = ValidatedData(data=arr, quality=report)
        assert vd.data is arr
        assert vd.quality is report


class TestCleanData:
    def test_construction(self):
        arr = np.zeros((5, 2))
        params = NormParams(means=np.zeros(2), stds=np.ones(2), method="standard")
        report = QualityReport(n_samples=5, n_features=2, missing_pct=0.0, duplicate_count=0)
        cd = CleanData(data=arr, norm_params=params, quality=report)
        assert cd.data is arr
        assert cd.norm_params is params
        assert cd.quality is report
