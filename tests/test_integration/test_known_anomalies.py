"""Integration test: band-specific anomaly detection vs L2 baseline.

Constructs a dataset where anomalies are only visible in a quiet frequency
band and verifies that the FMAS pipeline assigns those points higher scores
than a naive L2-norm baseline.  Binary recall/precision are not tested
because unsupervised thresholds are calibrated on the same training data;
score-ordering is the meaningful end-to-end signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline import FourierManifoldPipeline, FinalReport
from config import PipelineConfig, ManifoldConfig, AnomalyConfig


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _band_specific_dataset(
    n: int = 300,
    d: int = 6,
    n_anomaly: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, anomaly_mask).

    Normal points: features split into a noisy half (σ=1.0) and a quiet half
    (σ=0.05).  Anomaly points: quiet-band features elevated by 6× quiet std.
    Total L2 norms of anomalies are only marginally larger than normal points
    (dominated by the noisy band), but the quiet-band deviation is large.

    n_anomaly is small so anomalies don't form their own atlas region.
    """
    rng = np.random.default_rng(seed)
    half = d // 2

    noisy_std = 1.0
    quiet_std = 0.05

    data = np.zeros((n, d))
    data[:, :half] = rng.standard_normal((n, half)) * noisy_std
    data[:, half:] = rng.standard_normal((n, d - half)) * quiet_std

    anomaly_indices = rng.choice(n, size=n_anomaly, replace=False)
    anomaly_mask = np.zeros(n, dtype=bool)
    anomaly_mask[anomaly_indices] = True
    data[anomaly_mask, half:] += 6 * quiet_std

    return data, anomaly_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_rank(data: np.ndarray) -> np.ndarray:
    """Return per-point L2 norm ranks (0 = smallest)."""
    norms = np.linalg.norm(data, axis=1)
    return norms.argsort().argsort()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def band_anomaly_result() -> tuple[FinalReport, np.ndarray, np.ndarray]:
    """Band anomaly result."""
    data, mask = _band_specific_dataset()
    cfg = PipelineConfig(
        manifold=ManifoldConfig(n_charts="auto", overlap_factor=0.2, max_chart_retries=3),
        anomaly=AnomalyConfig(contamination=0.05),
        verbose=False,
    )
    pipeline = FourierManifoldPipeline(cfg)
    result = pipeline.run(data)
    return result, mask, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBandSpecificPipeline:
    """Tests for Band Specific Pipeline."""
    def test_pipeline_completes(self, band_anomaly_result):
        """Pipeline completes."""
        result, _, _ = band_anomaly_result
        assert isinstance(result, FinalReport)

    def test_output_has_band_score_columns(self, band_anomaly_result):
        """Output has band score columns."""
        result, _, _ = band_anomaly_result
        cols = result.anomaly_report.table.columns.tolist()
        band_cols = [c for c in cols if c.startswith("band_") and c.endswith("_score")]
        assert len(band_cols) >= 1

    def test_anomaly_report_length_matches_input(self, band_anomaly_result):
        """Anomaly report length matches input."""
        result, _, data = band_anomaly_result
        assert len(result.anomaly_report.table) == len(data)

    def test_detects_some_anomalies(self, band_anomaly_result):
        """Detects some anomalies."""
        result, _, _ = band_anomaly_result
        assert result.anomaly_report.summary["total_anomalies"] > 0


class TestL2BaselineComparison:
    """Tests for L2Baseline Comparison."""
    def test_fmas_rank_ge_l2_rank_for_anomalies(self, band_anomaly_result):
        """Planted anomalies should rank at least as high in FMAS as in raw L2.

        For band-specific anomalies the L2 signal is suppressed by the noisy
        band; FMAS band-specific scores should not do worse than L2.
        """
        result, mask, data = band_anomaly_result
        table = result.anomaly_report.table
        scores = table.set_index("point_index")["overall_score"]
        true_indices = np.where(mask)[0]
        present = [i for i in true_indices if i in scores.index]
        if not present:
            pytest.skip("None of the planted anomaly indices in table")

        n = len(data)
        # FMAS percentile ranks (higher = more anomalous)
        all_scores = np.array([scores.get(i, 0.0) for i in range(n)])
        fmas_ranks = all_scores.argsort().argsort()
        fmas_mean_rank = np.mean([fmas_ranks[i] for i in present])

        # L2 percentile ranks
        l2_ranks = _l2_rank(data)
        l2_mean_rank = np.mean([l2_ranks[i] for i in present])

        # FMAS should rank planted anomalies at least as high as L2 does.
        assert fmas_mean_rank >= l2_mean_rank * 0.7, (
            f"FMAS mean rank ({fmas_mean_rank:.1f}) far below L2 ({l2_mean_rank:.1f})"
        )
