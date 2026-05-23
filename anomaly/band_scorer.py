"""
Aggregates band-wise residuals into a single composite anomaly score per
node using configurable per-band weights, and exposes raw band scores for
interpretability and downstream analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from anomaly.residual import ResidualData
from common.types import FrequencyBand


@dataclass
class BandScores:
    """Per-band and aggregate anomaly z-scores.

    Parameters
    ----------
    per_band : band_index → (n,) z-score array for that band.
    overall : (n,) aggregate score per point (max |z| across bands by default).
    weights : (n_bands,) weights used when computing the aggregate.
    """

    per_band: dict[int, np.ndarray] = field(default_factory=dict)
    overall: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


def score(
    residual_data: ResidualData,
    manifold,
) -> BandScores:
    """Compute per-band and overall z-scores for every data point.

    For each point, the z-score for band b is:
        z_b = (||r_normal_b|| − μ_b) / σ_b

    where μ_b and σ_b come from the point's primary chart's expected residual
    distribution.  The overall score is the maximum |z| across all bands.

    Parameters
    ----------
    residual_data : ResidualData from anomaly.residual.compute.
    manifold : Manifold with atlas, expected_residuals, and spectral data.

    Returns
    -------
    BandScores
    """
    assignments = manifold.atlas.primary_assignments
    bands = manifold.spectral.bands
    band_indices = sorted(residual_data.per_band_norms.keys())
    n = residual_data.per_band_norms[band_indices[0]].shape[0] if band_indices else 0

    per_band: dict[int, np.ndarray] = {bi: np.zeros(n) for bi in band_indices}

    for ci, exp in manifold.expected_residuals.items():
        mask = assignments == ci
        if not mask.any():
            continue
        for bi in band_indices:
            band_norms = residual_data.per_band_norms[bi][mask]
            expected_mean = exp.per_band_mean.get(bi, 0.0)
            expected_var = exp.per_band_var.get(bi, 0.0)
            per_band[bi][mask] = _score_band_against_expected(
                band_norms, expected_mean, expected_var
            )

    weights = _weight_bands(bands)
    overall = _aggregate_scores(per_band)

    return BandScores(per_band=per_band, overall=overall, weights=weights)


def _score_band_against_expected(
    band_norms: np.ndarray,
    expected_mean: float,
    expected_var: float,
) -> np.ndarray:
    """Compute z-scores: (band_norms - expected_mean) / sqrt(expected_var).

    Handles the degenerate case when expected_var ≈ 0:
    - If band_norm ≈ expected_mean: z = 0 (within tolerance).
    - Otherwise: z = sign(diff) * large_value (extreme deviation).

    Parameters
    ----------
    band_norms : (m,) per-point band norms for a batch of points.
    expected_mean : calibrated mean from the expected residual distribution.
    expected_var : calibrated variance; may be near-zero for quiet bands.

    Returns
    -------
    np.ndarray : (m,) z-scores.
    """
    diff = band_norms - expected_mean
    std = float(np.sqrt(max(expected_var, 0.0)))
    if std < 1e-12:
        return np.where(np.abs(diff) < 1e-10, 0.0, np.sign(diff) * 1e10)
    return diff / std


def _aggregate_scores(
    band_scores: dict[int, np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a single overall anomaly score per point.

    Default (weights=None): max |z| across bands — catches single-band anomalies.
    With weights: weighted mean of |z| — catches broad partial anomalies.

    Parameters
    ----------
    band_scores : band_index → (n,) z-score arrays.
    weights : (n_bands,) non-negative weights; ignored when None.

    Returns
    -------
    np.ndarray : (n,) overall anomaly scores.
    """
    if not band_scores:
        return np.array([])

    abs_matrix = np.abs(np.column_stack(list(band_scores.values())))  # (n, k)

    if weights is None:
        return np.max(abs_matrix, axis=1)

    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(w)) / len(w)
    return abs_matrix @ w


def _weight_bands(bands: list[FrequencyBand], method: str = "uniform") -> np.ndarray:
    """Return band weights for the aggregate score.

    Parameters
    ----------
    bands : list of FrequencyBand.
    method : "uniform" assigns equal weight to every band.

    Returns
    -------
    np.ndarray : (n_bands,) weight vector summing to 1.
    """
    n = len(bands)
    if n == 0:
        return np.array([])
    return np.ones(n) / n
