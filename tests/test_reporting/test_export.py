"""Tests for reporting/export.py.

Tests: JSON validity and round-trip, CSV round-trip, DataFrame identity,
summary_dict native types.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd

from anomaly import AnomalyResults
from anomaly.band_scorer import BandScores
from anomaly.residual import ResidualData
from anomaly.threshold import apply as apply_threshold
from reporting.anomaly_report import AnomalyReport, report
from reporting.export import summary_dict, to_csv, to_dataframe, to_json


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_band_scores(rng, n: int, n_bands: int, n_anomaly: int) -> BandScores:
    per_band: dict[int, np.ndarray] = {}
    for b in range(n_bands):
        scores = rng.standard_normal(n)
        scores[-n_anomaly:] += 15.0
        per_band[b] = scores
    abs_mat = np.abs(np.column_stack(list(per_band.values())))
    overall = np.max(abs_mat, axis=1)
    weights = np.ones(n_bands) / n_bands
    return BandScores(per_band=per_band, overall=overall, weights=weights)


def _make_report(n: int = 60, n_bands: int = 3, n_anomaly: int = 6,
                 seed: int = 0) -> AnomalyReport:
    rng = np.random.default_rng(seed)
    d = 4
    band_scores = _make_band_scores(rng, n, n_bands, n_anomaly)
    flags = apply_threshold(band_scores)
    rd = ResidualData(
        total_residuals=rng.standard_normal((n, d)),
        normal_residuals=rng.standard_normal((n, d)),
        per_band_norms={b: np.abs(band_scores.per_band[b]) for b in range(n_bands)},
    )
    results = AnomalyResults(
        scores=band_scores, flags=flags, residual_data=rd,
        n_anomalies=int(np.sum(flags.overall_flags)),
    )
    return report(results, rng.standard_normal((n, d)))


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestToJson:
    """Tests for To Json."""
    def test_returns_string(self):
        """Returns string."""
        r = _make_report()
        assert isinstance(to_json(r), str)

    def test_valid_json(self):
        """Valid json."""
        r = _make_report()
        parsed = json.loads(to_json(r))
        assert isinstance(parsed, dict)

    def test_has_summary_key(self):
        """Has summary key."""
        r = _make_report()
        parsed = json.loads(to_json(r))
        assert "summary" in parsed

    def test_has_type_distribution_key(self):
        """Has type distribution key."""
        r = _make_report()
        parsed = json.loads(to_json(r))
        assert "type_distribution" in parsed

    def test_has_top_anomalies_key(self):
        """Has top anomalies key."""
        r = _make_report()
        parsed = json.loads(to_json(r))
        assert "top_anomalies" in parsed

    def test_summary_values_are_native_types(self):
        """Summary values are native types."""
        r = _make_report()
        parsed = json.loads(to_json(r))
        for val in parsed["summary"].values():
            assert type(val) in (int, float, str, bool, list, dict, type(None))

    def test_no_numpy_types_in_output(self):
        """No numpy types in output."""
        # If numpy types slipped through, json.loads would have raised TypeError.
        # This test is satisfied if test_valid_json passes, but we also verify
        # that every scalar in summary is a Python native type.
        r = _make_report()
        raw = to_json(r)
        parsed = json.loads(raw)
        for val in parsed["summary"].values():
            assert not isinstance(val, np.generic)

    def test_total_points_roundtrips_correctly(self):
        """Total points roundtrips correctly."""
        n = 60
        r = _make_report(n=n)
        parsed = json.loads(to_json(r))
        assert parsed["summary"]["total_points"] == n

    def test_top_anomalies_is_list(self):
        """Top anomalies is list."""
        r = _make_report()
        parsed = json.loads(to_json(r))
        assert isinstance(parsed["top_anomalies"], list)


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

class TestToCsv:
    """Tests for To Csv."""
    def test_creates_file(self):
        """Creates file."""
        r = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(r, path)
            assert os.path.isfile(path)
        finally:
            os.unlink(path)

    def test_roundtrip_shape(self):
        """Roundtrip shape."""
        r = _make_report(n=60, n_bands=3)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(r, path)
            loaded = pd.read_csv(path)
            assert loaded.shape == r.table.shape
        finally:
            os.unlink(path)

    def test_roundtrip_columns(self):
        """Roundtrip columns."""
        r = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(r, path)
            loaded = pd.read_csv(path)
            assert list(loaded.columns) == list(r.table.columns)
        finally:
            os.unlink(path)

    def test_overall_score_preserved(self):
        """Overall score preserved."""
        r = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(r, path)
            loaded = pd.read_csv(path)
            np.testing.assert_allclose(
                loaded["overall_score"].values,
                r.table["overall_score"].values,
                rtol=1e-6,
            )
        finally:
            os.unlink(path)

    def test_no_unnamed_index_column(self):
        """No unnamed index column."""
        # to_csv must use index=False so no "Unnamed: 0" column appears.
        r = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(r, path)
            loaded = pd.read_csv(path)
            assert not any("Unnamed" in c for c in loaded.columns)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    """Tests for To Dataframe."""
    def test_returns_dataframe(self):
        """Returns dataframe."""
        r = _make_report()
        assert isinstance(to_dataframe(r), pd.DataFrame)

    def test_is_same_object_as_table(self):
        """Is same object as table."""
        r = _make_report()
        assert to_dataframe(r) is r.table

    def test_shape_matches_n(self):
        """Shape matches n."""
        n = 60
        r = _make_report(n=n)
        df = to_dataframe(r)
        assert len(df) == n


# ---------------------------------------------------------------------------
# summary_dict
# ---------------------------------------------------------------------------

class TestSummaryDict:
    """Tests for Summary Dict."""
    def test_returns_dict(self):
        """Returns dict."""
        r = _make_report()
        assert isinstance(summary_dict(r), dict)

    def test_includes_summary_keys(self):
        """Includes summary keys."""
        r = _make_report()
        sd = summary_dict(r)
        for key in r.summary:
            assert key in sd

    def test_includes_type_distribution_keys(self):
        """Includes type distribution keys."""
        r = _make_report()
        sd = summary_dict(r)
        for key in r.type_distribution:
            assert key in sd

    def test_all_values_are_python_native(self):
        """All values are python native."""
        r = _make_report()
        sd = summary_dict(r)
        for val in sd.values():
            assert type(val) in (int, float, str, bool, type(None)), (
                f"Non-native type: {type(val)} for value {val}"
            )

    def test_total_anomalies_correct(self):
        """Total anomalies correct."""
        n, n_anomaly = 60, 6
        r = _make_report(n=n, n_anomaly=n_anomaly)
        sd = summary_dict(r)
        # Total anomalies should equal the reported n_anomalies (flags applied).
        assert sd["total_anomalies"] == r.summary["total_anomalies"]
