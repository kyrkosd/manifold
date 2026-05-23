"""Tests for reporting/anomaly_report.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly import AnomalyResults
from anomaly.band_scorer import BandScores
from anomaly.residual import ResidualData
from anomaly.threshold import apply as apply_threshold
from reporting.anomaly_report import (
    AnomalyReport,
    _per_point_table,
    _statistical_summary,
    _top_anomalies,
    _type_distribution,
    report,
)


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _make_results(n: int = 80, n_bands: int = 3, n_anomaly: int = 8,
                  seed: int = 0) -> tuple[AnomalyResults, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = 5

    per_band: dict[int, np.ndarray] = {}
    for b in range(n_bands):
        scores = rng.standard_normal(n)
        scores[-n_anomaly:] += 15.0      # plant anomalies with high z
        per_band[b] = scores

    abs_mat = np.abs(np.column_stack(list(per_band.values())))
    overall = np.max(abs_mat, axis=1)
    weights = np.ones(n_bands) / n_bands
    band_scores = BandScores(per_band=per_band, overall=overall, weights=weights)

    flags = apply_threshold(band_scores)

    total = rng.standard_normal((n, d))
    normal = rng.standard_normal((n, d)) * 0.05
    per_band_norms = {b: np.abs(per_band[b]) for b in range(n_bands)}
    residual_data = ResidualData(
        total_residuals=total,
        normal_residuals=normal,
        per_band_norms=per_band_norms,
    )

    results = AnomalyResults(
        scores=band_scores,
        flags=flags,
        residual_data=residual_data,
        n_anomalies=int(np.sum(flags.overall_flags)),
    )
    data = rng.standard_normal((n, d))
    return results, data


# ---------------------------------------------------------------------------
# report() — top-level
# ---------------------------------------------------------------------------

class TestReport:
    """Tests for Report."""
    def test_returns_anomaly_report(self):
        """Returns anomaly report."""
        results, data = _make_results()
        r = report(results, data)
        assert isinstance(r, AnomalyReport)

    def test_all_fields_populated(self):
        """All fields populated."""
        results, data = _make_results()
        r = report(results, data)
        assert isinstance(r.table, pd.DataFrame)
        assert isinstance(r.type_distribution, dict)
        assert isinstance(r.top_anomalies, list)
        assert isinstance(r.summary, dict)


# ---------------------------------------------------------------------------
# _per_point_table
# ---------------------------------------------------------------------------

class TestPerPointTable:
    """Tests for Per Point Table."""
    def test_is_dataframe(self):
        """Is dataframe."""
        results, _ = _make_results()
        assert isinstance(_per_point_table(results), pd.DataFrame)

    def test_row_count_equals_n(self):
        """Row count equals n."""
        n = 80
        results, _ = _make_results(n=n)
        df = _per_point_table(results)
        assert len(df) == n

    def test_required_columns_present(self):
        """Required columns present."""
        results, _ = _make_results(n_bands=3)
        df = _per_point_table(results)
        for col in ["point_index", "overall_score", "is_anomaly",
                    "band_0_score", "band_1_score", "band_2_score",
                    "top_anomalous_band"]:
            assert col in df.columns

    def test_column_count(self):
        """Column count."""
        # 4 fixed cols + n_bands band-score cols = 4 + n_bands
        n_bands = 3
        results, _ = _make_results(n_bands=n_bands)
        df = _per_point_table(results)
        assert len(df.columns) == 4 + n_bands

    def test_sorted_by_overall_score_descending(self):
        """Sorted by overall score descending."""
        results, _ = _make_results()
        df = _per_point_table(results)
        assert df["overall_score"].is_monotonic_decreasing

    def test_is_anomaly_dtype_bool(self):
        """Is anomaly dtype bool."""
        results, _ = _make_results()
        df = _per_point_table(results)
        assert df["is_anomaly"].dtype == bool

    def test_top_anomalous_band_has_band_prefix(self):
        """Top anomalous band has band prefix."""
        results, _ = _make_results()
        df = _per_point_table(results)
        assert all(v.startswith("band_") for v in df["top_anomalous_band"])

    def test_point_index_covers_all_points(self):
        """Point index covers all points."""
        n = 60
        results, _ = _make_results(n=n)
        df = _per_point_table(results)
        assert set(df["point_index"].tolist()) == set(range(n))


# ---------------------------------------------------------------------------
# _type_distribution
# ---------------------------------------------------------------------------

class TestTypeDistribution:
    """Tests for Type Distribution."""
    def test_returns_dict_with_three_keys(self):
        """Returns dict with three keys."""
        results, _ = _make_results()
        td = _type_distribution(results)
        assert set(td.keys()) == {"spectral", "partial", "global"}

    def test_counts_are_non_negative(self):
        """Counts are non negative."""
        results, _ = _make_results()
        td = _type_distribution(results)
        assert all(v >= 0 for v in td.values())

    def test_total_le_n_anomalies(self):
        """Total le n anomalies."""
        results, _ = _make_results()
        td = _type_distribution(results)
        total = td["spectral"] + td["partial"] + td["global"]
        assert total <= results.n_anomalies

    def test_no_anomalies_gives_all_zeros(self):
        """No anomalies gives all zeros."""
        # Build results with no anomalies by setting all z-scores near zero.
        rng = np.random.default_rng(5)
        n, n_bands = 50, 2
        per_band = {b: rng.standard_normal(n) * 0.001 for b in range(n_bands)}
        abs_mat = np.abs(np.column_stack(list(per_band.values())))
        overall = np.max(abs_mat, axis=1)
        band_scores = BandScores(per_band=per_band, overall=overall,
                                 weights=np.ones(n_bands) / n_bands)
        flags = apply_threshold(band_scores)
        rd = ResidualData(
            total_residuals=rng.standard_normal((n, 3)),
            normal_residuals=rng.standard_normal((n, 3)),
            per_band_norms={b: np.abs(per_band[b]) for b in range(n_bands)},
        )
        results = AnomalyResults(
            scores=band_scores, flags=flags, residual_data=rd,
            n_anomalies=int(np.sum(flags.overall_flags)),
        )
        td = _type_distribution(results)
        if results.n_anomalies == 0:
            assert td == {"spectral": 0, "partial": 0, "global": 0}


# ---------------------------------------------------------------------------
# _top_anomalies
# ---------------------------------------------------------------------------

class TestTopAnomalies:
    """Tests for Top Anomalies."""
    def test_returns_list(self):
        """Returns list."""
        results, _ = _make_results()
        assert isinstance(_top_anomalies(results), list)

    def test_length_at_most_n(self):
        """Length at most n."""
        results, _ = _make_results(n_anomaly=8)
        top = _top_anomalies(results, n=5)
        assert len(top) <= 5

    def test_each_entry_has_required_keys(self):
        """Each entry has required keys."""
        results, _ = _make_results()
        for entry in _top_anomalies(results):
            assert "point_index" in entry
            assert "overall_score" in entry
            assert "flagged_bands" in entry
            assert "band_scores" in entry

    def test_sorted_by_score_descending(self):
        """Sorted by score descending."""
        results, _ = _make_results()
        top = _top_anomalies(results)
        scores = [e["overall_score"] for e in top]
        assert scores == sorted(scores, reverse=True)

    def test_all_entries_are_anomalies(self):
        """All entries are anomalies."""
        results, _ = _make_results()
        top = _top_anomalies(results)
        for entry in top:
            assert results.flags.overall_flags[entry["point_index"]]

    def test_empty_when_no_anomalies(self):
        """Empty when no anomalies."""
        rng = np.random.default_rng(9)
        n, n_bands = 40, 2
        per_band = {b: rng.standard_normal(n) * 0.0001 for b in range(n_bands)}
        abs_mat = np.abs(np.column_stack(list(per_band.values())))
        overall = np.max(abs_mat, axis=1)
        band_scores = BandScores(per_band=per_band, overall=overall,
                                 weights=np.ones(n_bands) / n_bands)
        flags = apply_threshold(band_scores)
        rd = ResidualData(
            total_residuals=rng.standard_normal((n, 3)),
            normal_residuals=rng.standard_normal((n, 3)),
            per_band_norms={b: np.abs(per_band[b]) for b in range(n_bands)},
        )
        results = AnomalyResults(scores=band_scores, flags=flags, residual_data=rd,
                                 n_anomalies=int(np.sum(flags.overall_flags)))
        if results.n_anomalies == 0:
            assert _top_anomalies(results) == []


# ---------------------------------------------------------------------------
# _statistical_summary
# ---------------------------------------------------------------------------

class TestStatisticalSummary:
    """Tests for Statistical Summary."""
    def test_returns_dict(self):
        """Returns dict."""
        results, _ = _make_results()
        assert isinstance(_statistical_summary(results), dict)

    def test_required_keys_present(self):
        """Required keys present."""
        results, _ = _make_results()
        sm = _statistical_summary(results)
        for key in ["total_points", "total_anomalies", "anomaly_rate",
                    "mean_score", "std_score", "max_score"]:
            assert key in sm

    def test_per_band_count_keys_present(self):
        """Per band count keys present."""
        n_bands = 3
        results, _ = _make_results(n_bands=n_bands)
        sm = _statistical_summary(results)
        for b in range(n_bands):
            assert f"band_{b}_anomaly_count" in sm

    def test_total_points_correct(self):
        """Total points correct."""
        n = 70
        results, _ = _make_results(n=n)
        assert _statistical_summary(results)["total_points"] == n

    def test_anomaly_rate_in_unit_interval(self):
        """Anomaly rate in unit interval."""
        results, _ = _make_results()
        rate = _statistical_summary(results)["anomaly_rate"]
        assert 0.0 <= rate <= 1.0

    def test_max_score_is_max_of_overall(self):
        """Max score is max of overall."""
        results, _ = _make_results()
        sm = _statistical_summary(results)
        assert sm["max_score"] == pytest.approx(float(np.max(results.scores.overall)))
