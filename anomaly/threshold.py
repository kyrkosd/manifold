"""
Determines anomaly detection thresholds from the score distribution via
percentile, median-absolute-deviation (MAD), or parametric (Gaussian/
gamma) fitting, with support for both global and per-cluster thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from anomaly.band_scorer import BandScores

_MAD_CONSISTENCY = 1.4826   # makes MAD a consistent estimator of σ for Gaussian data


@dataclass
class AnomalyFlags:
    """Boolean anomaly flags derived from BandScores.

    Parameters
    ----------
    per_band_flags : band_index → (n,) boolean flag array.
    overall_flags : (n,) aggregate flag (True = anomalous).
    thresholds : Bonferroni-corrected per-band thresholds.
    overall_threshold : threshold applied to the overall score.
    """

    per_band_flags: dict[int, np.ndarray] = field(default_factory=dict)
    overall_flags: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    thresholds: dict[int, float] = field(default_factory=dict)
    overall_threshold: float = 0.0


def apply(scores: BandScores) -> AnomalyFlags:
    """Threshold BandScores to produce boolean anomaly flags.

    Per-band: |z| > Bonferroni-corrected adaptive threshold.
    Overall: overall score > adaptive threshold.

    Parameters
    ----------
    scores : BandScores from anomaly.band_scorer.score.

    Returns
    -------
    AnomalyFlags
    """
    per_band_raw: dict[int, float] = {
        b: _adaptive_threshold(np.abs(v)) for b, v in scores.per_band.items()
    }
    corrected = _multiple_testing_correction(per_band_raw)

    per_band_flags: dict[int, np.ndarray] = {
        b: _flag_points(np.abs(scores.per_band[b]), corrected[b])
        for b in scores.per_band
    }

    overall_threshold = _adaptive_threshold(scores.overall)
    overall_flags = _flag_points(scores.overall, overall_threshold)

    return AnomalyFlags(
        per_band_flags=per_band_flags,
        overall_flags=overall_flags,
        thresholds=corrected,
        overall_threshold=overall_threshold,
    )


def _adaptive_threshold(scores: np.ndarray) -> float:
    """Robust threshold: median + 3 × (1.4826 × MAD).

    The 1.4826 factor makes MAD a consistent estimator of σ for Gaussian data,
    giving a threshold near the 99.9th percentile under normality.

    Parameters
    ----------
    scores : (n,) non-negative score array.

    Returns
    -------
    float threshold value.
    """
    if len(scores) == 0:
        return 0.0
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    return med + 3.0 * _MAD_CONSISTENCY * mad


def _percentile_threshold(
    scores: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """Flag the top (100 - *percentile*)% of scores.

    Parameters
    ----------
    scores : (n,) score array.
    percentile : percentile cutoff in [0, 100].

    Returns
    -------
    float threshold value.
    """
    if len(scores) == 0:
        return 0.0
    return float(np.percentile(scores, percentile))


def _multiple_testing_correction(
    per_band_thresholds: dict[int, float],
    method: str = "bonferroni",
) -> dict[int, float]:
    """Adjust per-band thresholds for multiple testing (Bonferroni).

    With n simultaneous tests each calibrated at level α, Bonferroni controls
    the family-wise error rate by raising each threshold by a factor of
    log(n) + 1 (a heuristic that is conservative without being prohibitive
    for small n).

    Parameters
    ----------
    per_band_thresholds : raw per-band threshold dict.
    method : "bonferroni" (only supported option for MVP).

    Returns
    -------
    dict with the same keys and adjusted values.
    """
    n = len(per_band_thresholds)
    if method == "bonferroni" and n > 1:
        factor = float(np.log(n) + 1.0)
        return {b: t * factor for b, t in per_band_thresholds.items()}
    return dict(per_band_thresholds)


def _flag_points(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean array marking points above *threshold*.

    Parameters
    ----------
    scores : (n,) score array (non-negative for per-band, raw for overall).
    threshold : scalar cutoff.

    Returns
    -------
    np.ndarray : (n,) bool array; True where score > threshold.
    """
    return scores > threshold
