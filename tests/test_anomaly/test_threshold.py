"""Tests for anomaly/threshold.py."""
from __future__ import annotations

import numpy as np
import pytest

from anomaly.band_scorer import BandScores
from anomaly.threshold import (
    AnomalyFlags,
    _adaptive_threshold,
    _flag_points,
    _multiple_testing_correction,
    _percentile_threshold,
    apply,
)


# ---------------------------------------------------------------------------
# Shared fixture: synthetic BandScores
# ---------------------------------------------------------------------------

def _make_band_scores(n: int = 100, n_bands: int = 3, seed: int = 0) -> BandScores:
    """Return BandScores with n normal points and 5 planted high-score anomalies."""
    rng = np.random.default_rng(seed)
    n_anomaly = 5
    per_band: dict[int, np.ndarray] = {}
    for b in range(n_bands):
        normal = rng.standard_normal(n - n_anomaly)
        anomaly = rng.standard_normal(n_anomaly) + 20.0   # very high z-score
        per_band[b] = np.concatenate([normal, anomaly])

    abs_matrix = np.abs(np.column_stack(list(per_band.values())))
    overall = np.max(abs_matrix, axis=1)
    weights = np.ones(n_bands) / n_bands
    return BandScores(per_band=per_band, overall=overall, weights=weights)


# ---------------------------------------------------------------------------
# apply() — return type and structural correctness
# ---------------------------------------------------------------------------

class TestApply:
    """Tests for Apply."""
    def test_returns_anomaly_flags(self):
        """Returns anomaly flags."""
        bs = _make_band_scores()
        result = apply(bs)
        assert isinstance(result, AnomalyFlags)

    def test_per_band_flags_shape(self):
        """Per band flags shape."""
        n, n_bands = 80, 3
        bs = _make_band_scores(n=n, n_bands=n_bands)
        result = apply(bs)
        for flags in result.per_band_flags.values():
            assert flags.shape == (n,)
            assert flags.dtype == bool

    def test_overall_flags_shape(self):
        """Overall flags shape."""
        n = 80
        bs = _make_band_scores(n=n)
        result = apply(bs)
        assert result.overall_flags.shape == (n,)
        assert result.overall_flags.dtype == bool

    def test_overall_flags_bool(self):
        """Overall flags bool."""
        bs = _make_band_scores()
        result = apply(bs)
        assert result.overall_flags.dtype == bool

    def test_planted_anomalies_are_flagged(self):
        """Planted anomalies are flagged."""
        # Last 5 points have z-score ≈ 20 — must be flagged overall.
        n, n_anomaly = 100, 5
        bs = _make_band_scores(n=n)
        result = apply(bs)
        anomaly_mask = np.zeros(n, dtype=bool)
        anomaly_mask[n - n_anomaly:] = True
        assert np.all(result.overall_flags[anomaly_mask])

    def test_thresholds_dict_has_band_keys(self):
        """Thresholds dict has band keys."""
        n_bands = 3
        bs = _make_band_scores(n_bands=n_bands)
        result = apply(bs)
        assert set(result.thresholds.keys()) == set(range(n_bands))

    def test_overall_threshold_positive(self):
        """Overall threshold positive."""
        bs = _make_band_scores()
        result = apply(bs)
        assert result.overall_threshold >= 0.0


# ---------------------------------------------------------------------------
# _adaptive_threshold
# ---------------------------------------------------------------------------

class TestAdaptiveThreshold:
    """Tests for Adaptive Threshold."""
    def test_gaussian_data_near_3sigma(self):
        """Gaussian data near 3sigma."""
        # For N(0,1) data, threshold ≈ 0 + 3*1.4826*0.675 ≈ 3.
        rng = np.random.default_rng(0)
        scores = np.abs(rng.standard_normal(10_000))
        threshold = _adaptive_threshold(scores)
        assert 2.0 < threshold < 5.0   # loose bounds; MAD is robust

    def test_constant_data_gives_zero(self):
        """Constant data gives zero."""
        scores = np.full(50, 2.5)
        assert _adaptive_threshold(scores) == pytest.approx(2.5, abs=1e-10)

    def test_empty_gives_zero(self):
        """Empty gives zero."""
        assert _adaptive_threshold(np.array([])) == 0.0

    def test_high_outliers_do_not_inflate_threshold(self):
        """High outliers do not inflate threshold."""
        # MAD is robust: a few extreme values should not blow up the threshold.
        scores = np.concatenate([np.ones(100), [1e6]])
        t_with_outlier = _adaptive_threshold(scores)
        t_without = _adaptive_threshold(np.ones(100))
        assert abs(t_with_outlier - t_without) < 1.0   # outlier has small effect


# ---------------------------------------------------------------------------
# _percentile_threshold
# ---------------------------------------------------------------------------

class TestPercentileThreshold:
    """Tests for Percentile Threshold."""
    def test_99th_percentile(self):
        """99th percentile."""
        scores = np.arange(100, dtype=float)
        assert _percentile_threshold(scores, 99.0) == pytest.approx(99.0 * 0.99, abs=0.1)

    def test_empty_gives_zero(self):
        """Empty gives zero."""
        assert _percentile_threshold(np.array([]), 99.0) == 0.0

    def test_100th_percentile_is_max(self):
        """100th percentile is max."""
        scores = np.array([1.0, 3.0, 2.0, 5.0])
        assert _percentile_threshold(scores, 100.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _multiple_testing_correction
# ---------------------------------------------------------------------------

class TestMultipleTestingCorrection:
    """Tests for Multiple Testing Correction."""
    def test_single_band_unchanged(self):
        """Single band unchanged."""
        thresholds = {0: 3.0}
        corrected = _multiple_testing_correction(thresholds)
        assert corrected[0] == pytest.approx(3.0)

    def test_bonferroni_raises_threshold(self):
        """Bonferroni raises threshold."""
        # With multiple bands, each per-band threshold must be at least the original.
        thresholds = {0: 3.0, 1: 3.0, 2: 3.0}
        corrected = _multiple_testing_correction(thresholds)
        for b, t in thresholds.items():
            assert corrected[b] >= t

    def test_all_bands_adjusted_equally(self):
        """All bands adjusted equally."""
        thresholds = {0: 2.0, 1: 2.0, 2: 2.0}
        corrected = _multiple_testing_correction(thresholds)
        values = list(corrected.values())
        assert all(v == pytest.approx(values[0]) for v in values)

    def test_empty_input_returns_empty(self):
        """Empty input returns empty."""
        assert _multiple_testing_correction({}) == {}


# ---------------------------------------------------------------------------
# _flag_points
# ---------------------------------------------------------------------------

class TestFlagPoints:
    """Tests for Flag Points."""
    def test_above_threshold_is_true(self):
        """Above threshold is true."""
        scores = np.array([1.0, 3.0, 5.0])
        flags = _flag_points(scores, 2.0)
        assert flags.tolist() == [False, True, True]

    def test_at_threshold_is_not_flagged(self):
        """At threshold is not flagged."""
        # strictly greater than (>), not >=.
        assert not _flag_points(np.array([3.0]), 3.0)[0]

    def test_all_below_gives_all_false(self):
        """All below gives all false."""
        scores = np.array([0.1, 0.2, 0.3])
        assert not np.any(_flag_points(scores, 1.0))

    def test_output_dtype_is_bool(self):
        """Output dtype is bool."""
        flags = _flag_points(np.array([1.0, 2.0]), 1.5)
        assert flags.dtype == bool
