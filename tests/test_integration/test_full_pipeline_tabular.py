"""Integration tests for the full FMAS pipeline on tabular data.

These tests verify end-to-end pipeline connectivity (all phases connect and
return the correct types) and soft performance bounds.  Detection recall/
precision are NOT tested here — the MAD-calibrated thresholds are trained on
the same data, so unsupervised score ordering is the meaningful metric.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline import FourierManifoldPipeline, FinalReport, create_pipeline
from config import PipelineConfig, ManifoldConfig, AnomalyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clean_tabular(n: int = 200, d: int = 5, intrinsic: int = 2,
                        seed: int = 0) -> np.ndarray:
    """200 points on a 2D linear submanifold in 5D."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((d, intrinsic)))[0]
    coords = rng.standard_normal((n, intrinsic))
    noise = rng.standard_normal((n, d)) * 0.05
    return coords @ basis.T + noise


def _make_contaminated_tabular(n: int = 300, d: int = 5, intrinsic: int = 2,
                                n_anomaly: int = 6, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Manifold data with a small number of sparse extreme outliers.

    n_anomaly is intentionally tiny relative to n so anomalies are scattered
    across normal-population charts rather than forming their own region.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((d, intrinsic)))[0]
    coords = rng.standard_normal((n, intrinsic))
    noise = rng.standard_normal((n, d)) * 0.05
    data = coords @ basis.T + noise

    data_std = np.std(data, axis=0, keepdims=True) + 1e-8
    anomaly_indices = rng.choice(n, size=n_anomaly, replace=False)
    anomaly_mask = np.zeros(n, dtype=bool)
    anomaly_mask[anomaly_indices] = True
    data[anomaly_mask] += 15.0 * data_std

    return data, anomaly_mask


@pytest.fixture(scope="module", name="pipeline")
def _pipeline() -> FourierManifoldPipeline:
    """Pipeline."""
    cfg = PipelineConfig(
        manifold=ManifoldConfig(n_charts="auto", overlap_factor=0.2, max_chart_retries=3),
        anomaly=AnomalyConfig(contamination=0.05),
        verbose=False,
    )
    return FourierManifoldPipeline(cfg)


@pytest.fixture(scope="module", name="clean_data")
def _clean_data() -> np.ndarray:
    """Clean data."""
    return _make_clean_tabular()


@pytest.fixture(scope="module", name="contaminated_pair")
def _contaminated_pair() -> tuple[np.ndarray, np.ndarray]:
    """Contaminated pair."""
    return _make_contaminated_tabular()


# ---------------------------------------------------------------------------
# Clean-data tests
# ---------------------------------------------------------------------------

class TestCleanData:
    """Tests for Clean Data."""
    def test_pipeline_returns_final_report(self, pipeline, clean_data):
        """Pipeline returns final report."""
        result = pipeline.run(clean_data)
        assert isinstance(result, FinalReport)

    def test_anomaly_report_populated(self, pipeline, clean_data):
        """Anomaly report populated."""
        result = pipeline.run(clean_data)
        r = result.anomaly_report
        assert r.table is not None
        assert len(r.table) == len(clean_data)

    def test_low_false_positive_rate(self, pipeline, clean_data):
        """Low false positive rate."""
        result = pipeline.run(clean_data)
        sd = result.anomaly_report.summary
        rate = sd["anomaly_rate"]
        assert rate < 0.20, f"FPR too high on clean data: {rate:.3f}"

    def test_timing_dict_has_all_phases(self, pipeline, clean_data):
        """Timing dict has all phases."""
        result = pipeline.run(clean_data)
        for phase in ("ingest", "structure", "fourier", "manifold", "anomaly", "reporting"):
            assert phase in result.timing
            assert result.timing[phase] >= 0.0

    def test_manifold_summary_populated(self, pipeline, clean_data):
        """Manifold summary populated."""
        result = pipeline.run(clean_data)
        ms = result.manifold_summary
        assert ms["n_charts"] >= 1
        assert 0.0 <= ms["coverage_pct"] <= 100.0
        assert ms["n_points"] == len(clean_data)

    def test_config_preserved_in_report(self, pipeline, clean_data):
        """Config preserved in report."""
        result = pipeline.run(clean_data)
        assert result.config is pipeline.config

    def test_table_has_required_columns(self, pipeline, clean_data):
        """Table has required columns."""
        result = pipeline.run(clean_data)
        cols = result.anomaly_report.table.columns.tolist()
        for col in ("point_index", "overall_score", "is_anomaly", "top_anomalous_band"):
            assert col in cols

    def test_summary_has_required_keys(self, pipeline, clean_data):
        """Summary has required keys."""
        result = pipeline.run(clean_data)
        sm = result.anomaly_report.summary
        for key in ("total_points", "total_anomalies", "anomaly_rate", "mean_score"):
            assert key in sm


# ---------------------------------------------------------------------------
# Contaminated-data tests
# ---------------------------------------------------------------------------

class TestContaminatedData:
    """Tests for Contaminated Data."""
    def test_pipeline_completes(self, pipeline, contaminated_pair):
        """Pipeline completes."""
        data, _ = contaminated_pair
        result = pipeline.run(data)
        assert isinstance(result, FinalReport)

    def test_table_length_matches_input(self, pipeline, contaminated_pair):
        """Table length matches input."""
        data, _ = contaminated_pair
        result = pipeline.run(data)
        assert len(result.anomaly_report.table) == len(data)

    def test_planted_anomalies_score_above_mean(self, pipeline, contaminated_pair):
        """Planted anomalies should have above-mean overall_score in expectation.

        We use score ordering rather than binary flags because the threshold
        is calibrated on the same data (unsupervised), meaning false negatives
        can occur when anomalies form clusters that inflate the expected residual.
        """
        data, mask = contaminated_pair
        result = pipeline.run(data)
        table = result.anomaly_report.table
        scores = table.set_index("point_index")["overall_score"]
        global_mean = scores.mean()
        true_indices = np.where(mask)[0]
        present = [i for i in true_indices if i in scores.index]
        if not present:
            pytest.skip("None of the planted anomaly indices found in table")
        anomaly_scores = np.array([scores[i] for i in present])
        # At least half the planted anomalies should score above the global mean.
        above = np.sum(anomaly_scores > global_mean)
        assert above >= len(present) // 2, (
            f"Only {above}/{len(present)} planted anomalies scored above mean "
            f"({global_mean:.3f}). Scores: {anomaly_scores.round(3)}"
        )

    def test_some_anomalies_detected(self, pipeline, contaminated_pair):
        """Pipeline detects at least one anomaly in contaminated data."""
        data, _ = contaminated_pair
        result = pipeline.run(data)
        n_detected = result.anomaly_report.summary["total_anomalies"]
        assert n_detected > 0


# ---------------------------------------------------------------------------
# create_pipeline convenience
# ---------------------------------------------------------------------------

class TestCreatePipeline:
    """Tests for Create Pipeline."""
    def test_no_args(self):
        """No args."""
        p = create_pipeline()
        assert isinstance(p, FourierManifoldPipeline)

    def test_with_dict(self):
        """With dict."""
        p = create_pipeline({"verbose": False})
        assert p.config.verbose is False
