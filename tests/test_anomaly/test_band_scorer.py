"""Tests for anomaly/band_scorer.py.

Includes the CRITICAL TEST: band-specific anomalies detectable by FMAS
but not by simple L2 reconstruction error.
"""
from __future__ import annotations

import numpy as np
import pytest

from anomaly.band_scorer import (
    BandScores,
    _aggregate_scores,
    _score_band_against_expected,
    _weight_bands,
    score,
)
from anomaly.reconstruction import reconstruct
from anomaly.residual import compute as compute_residuals
from anomaly.threshold import _adaptive_threshold, _flag_points
from common.types import FrequencyBand, FourierType, StructureType
from fourier import analyze_fourier
from manifold import build_manifold
from structure.report import StructureReport
import structure.graph_builder as gb


# ---------------------------------------------------------------------------
# Module-scoped manifold fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", name="manifold_fixture")
def _manifold_fixture():
    """Manifold fixture."""
    rng = np.random.default_rng(2)
    d, intrinsic, n = 5, 2, 100
    orth_mat, _ = np.linalg.qr(rng.standard_normal((d, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coords = rng.standard_normal((n, intrinsic))
    data = coords @ basis.T + rng.standard_normal((n, d)) * 0.02

    fg = gb.build(data)
    structure = StructureReport(
        type=StructureType.GRAPH, intrinsic_dim=intrinsic, graph=fg,
        ordering=None, periodicity=False,
        recommended_fourier=FourierType.GRAPH_FOURIER, confidence=1.0,
    )
    spectral = analyze_fourier(data, structure)
    config = {"n_charts": 3, "overlap_factor": 0.2, "max_retries": 3}
    mf = build_manifold(data, spectral, structure, config)
    return mf, data


# ---------------------------------------------------------------------------
# score() — return type and structure
# ---------------------------------------------------------------------------

class TestScore:
    """Tests for Score."""
    def test_returns_band_scores(self, manifold_fixture):
        """Returns band scores."""
        mf, data = manifold_fixture
        _, raw = reconstruct(data, mf)
        rd = compute_residuals(raw, data, mf)
        result = score(rd, mf)
        assert isinstance(result, BandScores)

    def test_per_band_keys_match_bands(self, manifold_fixture):
        """Per band keys match bands."""
        mf, data = manifold_fixture
        _, raw = reconstruct(data, mf)
        rd = compute_residuals(raw, data, mf)
        result = score(rd, mf)
        assert set(result.per_band.keys()) == set(range(len(mf.spectral.bands)))

    def test_overall_shape(self, manifold_fixture):
        """Overall shape."""
        mf, data = manifold_fixture
        _, raw = reconstruct(data, mf)
        rd = compute_residuals(raw, data, mf)
        result = score(rd, mf)
        assert result.overall.shape == (len(data),)

    def test_overall_is_max_abs_per_band(self, manifold_fixture):
        """Overall is max abs per band."""
        # overall[i] = max |z_b[i]| across bands.
        mf, data = manifold_fixture
        _, raw = reconstruct(data, mf)
        rd = compute_residuals(raw, data, mf)
        result = score(rd, mf)
        abs_matrix = np.abs(np.column_stack(list(result.per_band.values())))
        np.testing.assert_allclose(result.overall, np.max(abs_matrix, axis=1), atol=1e-10)

    def test_weights_length_matches_bands(self, manifold_fixture):
        """Weights length matches bands."""
        mf, data = manifold_fixture
        _, raw = reconstruct(data, mf)
        rd = compute_residuals(raw, data, mf)
        result = score(rd, mf)
        assert len(result.weights) == len(mf.spectral.bands)


# ---------------------------------------------------------------------------
# _score_band_against_expected
# ---------------------------------------------------------------------------

class TestScoreBandAgainstExpected:
    """Tests for Score Band Against Expected."""
    def test_zero_mean_zero_var_at_mean_gives_zero(self):
        """Zero mean zero var at mean gives zero."""
        # band_norms = expected_mean = 0 → z = 0.
        z = _score_band_against_expected(np.zeros(5), 0.0, 0.0)
        np.testing.assert_allclose(z, 0.0, atol=1e-10)

    def test_standard_z_score_formula(self):
        """Standard z score formula."""
        # z = (x - μ) / σ.
        norms = np.array([1.5, 2.0, 0.5])
        z = _score_band_against_expected(norms, 1.0, 0.25)  # σ = 0.5
        expected = (norms - 1.0) / 0.5
        np.testing.assert_allclose(z, expected, atol=1e-10)

    def test_positive_deviation_gives_positive_z(self):
        """Positive deviation gives positive z."""
        z = _score_band_against_expected(np.array([2.0]), 1.0, 1.0)
        assert z[0] > 0.0

    def test_negative_deviation_gives_negative_z(self):
        """Negative deviation gives negative z."""
        z = _score_band_against_expected(np.array([0.0]), 1.0, 1.0)
        assert z[0] < 0.0

    def test_zero_var_nonzero_diff_gives_large_z(self):
        """Zero var nonzero diff gives large z."""
        # expected_var=0, band_norm >> expected_mean → large z.
        z = _score_band_against_expected(np.array([5.0]), 0.0, 0.0)
        assert abs(z[0]) > 1e5

    def test_zero_var_zero_diff_gives_zero_z(self):
        """Zero var zero diff gives zero z."""
        z = _score_band_against_expected(np.array([1.0]), 1.0, 0.0)
        np.testing.assert_allclose(z, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _aggregate_scores
# ---------------------------------------------------------------------------

class TestAggregateScores:
    """Tests for Aggregate Scores."""
    def test_default_is_max_abs(self):
        """Default is max abs."""
        band_scores = {0: np.array([1.0, 3.0]), 1: np.array([2.0, 1.0])}
        result = _aggregate_scores(band_scores)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_weighted_mean(self):
        """Weighted mean."""
        band_scores = {0: np.array([2.0]), 1: np.array([4.0])}
        weights = np.array([0.5, 0.5])
        result = _aggregate_scores(band_scores, weights=weights)
        np.testing.assert_allclose(result, [3.0], atol=1e-10)

    def test_empty_returns_empty(self):
        """Empty returns empty."""
        result = _aggregate_scores({})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _weight_bands
# ---------------------------------------------------------------------------

class TestWeightBands:
    """Tests for Weight Bands."""
    def test_uniform_sums_to_one(self):
        """Uniform sums to one."""
        bands = [
            FrequencyBand(0, 1, "low", 1.0),
            FrequencyBand(2, 3, "mid", 0.5),
            FrequencyBand(4, 5, "high", 0.2),
        ]
        w = _weight_bands(bands)
        assert w.sum() == pytest.approx(1.0)

    def test_uniform_all_equal(self):
        """Uniform all equal."""
        bands = [FrequencyBand(0, 1, "a", 1.0), FrequencyBand(2, 3, "b", 1.0)]
        w = _weight_bands(bands)
        assert w[0] == pytest.approx(w[1])

    def test_empty_bands_returns_empty(self):
        """Empty bands returns empty."""
        assert len(_weight_bands([])) == 0


# ---------------------------------------------------------------------------
# CRITICAL TEST: band-specific anomalies that simple L2 misses
# ---------------------------------------------------------------------------

class TestBandSpecificAnomalyDetection:
    """Core FMAS claim: band-specific scoring detects anomalies invisible to L2.

    Construction:
    - Band 0 (noisy): normal points ~ N(1.0, 0.2) → dominates total L2.
    - Band 1 (quiet): normal points ~ N(0.01, 0.001) → tiny expected variance.
    - Anomaly: band 1 elevated to 0.5 (50× mean), band 0 unchanged.

    Total L2 of normal: sqrt(1.0² + 0.01²) ≈ 1.00
    Total L2 of anomaly: sqrt(1.0² + 0.5²) ≈ 1.12
    MAD threshold on total L2 ≈ 1.58 → anomaly L2 (1.12) < threshold → missed.

    FMAS band-1 z-score of anomaly: (0.5 - 0.01) / 0.001 ≈ 490 → detected.
    """

    def _make_data(self, n_normal=100, n_anomaly=10, seed=0):
        rng = np.random.default_rng(seed)
        n = n_normal + n_anomaly

        b0_normal = np.abs(rng.normal(1.0, 0.1, n_normal))
        b1_normal = np.abs(rng.normal(0.01, 0.001, n_normal))
        b0_anomaly = np.abs(rng.normal(1.0, 0.1, n_anomaly))
        b1_anomaly = np.abs(rng.normal(0.5, 0.01, n_anomaly))

        per_band_norms = {
            0: np.concatenate([b0_normal, b0_anomaly]),
            1: np.concatenate([b1_normal, b1_anomaly]),
        }
        total = np.sqrt(
            np.concatenate([b0_normal, b0_anomaly]) ** 2
            + np.concatenate([b1_normal, b1_anomaly]) ** 2
        )
        anomaly_mask = np.zeros(n, dtype=bool)
        anomaly_mask[n_normal:] = True
        return per_band_norms, total, anomaly_mask, b1_normal

    def test_l2_misses_at_least_some_anomalies(self):
        """L2 misses at least some anomalies."""
        _, total, anomaly_mask, _ = self._make_data()
        threshold = _adaptive_threshold(total)
        l2_flags = _flag_points(total, threshold)

        # L2 threshold is calibrated to band-0 noise; anomalies blend in.
        detected = int(np.sum(l2_flags[anomaly_mask]))
        assert detected < len(anomaly_mask.nonzero()[0]), (
            "L2 detected all anomalies — test fixture did not create hard-enough anomalies."
        )

    def test_fmas_detects_all_band_specific_anomalies(self):
        """Fmas detects all band specific anomalies."""
        per_band_norms, _, anomaly_mask, b1_normal = self._make_data()

        expected_mean = float(np.mean(b1_normal))
        expected_var = float(np.var(b1_normal))

        z_band1 = _score_band_against_expected(
            per_band_norms[1], expected_mean, expected_var
        )

        fmas_flags = np.abs(z_band1) > 3.0
        # All planted anomalies must be flagged.
        assert np.all(fmas_flags[anomaly_mask]), (
            "FMAS failed to detect some band-specific anomalies."
        )

    def test_fmas_low_false_positive_rate(self):
        """Fmas low false positive rate."""
        per_band_norms, _, anomaly_mask, b1_normal = self._make_data()

        expected_mean = float(np.mean(b1_normal))
        expected_var = float(np.var(b1_normal))
        z_band1 = _score_band_against_expected(
            per_band_norms[1], expected_mean, expected_var
        )
        fmas_flags = np.abs(z_band1) > 3.0
        normal_mask = ~anomaly_mask
        fpr = float(np.sum(fmas_flags[normal_mask])) / normal_mask.sum()
        assert fpr < 0.05

    def test_fmas_advantage_over_l2(self):
        """FMAS recall > L2 recall for band-specific anomalies."""
        from anomaly.threshold import _adaptive_threshold, _flag_points

        per_band_norms, total, anomaly_mask, b1_normal = self._make_data()

        # L2 recall
        l2_threshold = _adaptive_threshold(total)
        l2_flags = _flag_points(total, l2_threshold)
        l2_recall = float(np.sum(l2_flags[anomaly_mask])) / anomaly_mask.sum()

        # FMAS band-1 recall
        em = float(np.mean(b1_normal))
        ev = float(np.var(b1_normal))
        z1 = _score_band_against_expected(per_band_norms[1], em, ev)
        fmas_recall = float(np.sum(np.abs(z1[anomaly_mask]) > 3.0)) / anomaly_mask.sum()

        assert fmas_recall > l2_recall, (
            f"Expected FMAS recall ({fmas_recall:.2f}) > L2 recall ({l2_recall:.2f})."
        )
