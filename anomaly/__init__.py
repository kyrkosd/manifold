"""
Phase 4 — Anomaly Detection.

Public interface: ``detect_anomalies(data, manifold) → AnomalyResults``
reconstructs data via manifold charts, computes normal-space residuals,
scores them per frequency band against expected distributions, and flags
anomalous points with a robust adaptive threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anomaly import reconstruction
from anomaly import residual as residual_mod
from anomaly.band_scorer import BandScores
from anomaly.band_scorer import score as _score
from anomaly.residual import ResidualData
from anomaly.threshold import AnomalyFlags
from anomaly.threshold import apply as _apply

__all__ = ["AnomalyResults", "detect_anomalies"]


@dataclass
class AnomalyResults:
    """Container for all Phase-4 outputs.

    Parameters
    ----------
    scores : BandScores with per-band z-scores and aggregate overall score.
    flags : AnomalyFlags with boolean masks per band and overall.
    residual_data : ResidualData with total, normal, and band-decomposed residuals.
    n_anomalies : number of points flagged by the overall threshold.
    """

    scores: BandScores
    flags: AnomalyFlags
    residual_data: ResidualData
    n_anomalies: int


def detect_anomalies(
    data: np.ndarray,
    manifold,
) -> AnomalyResults:
    """Run the full Phase-4 anomaly detection pipeline.

    Steps
    -----
    [4.1] Reconstruct each point via its primary chart.
    [4.2] Compute raw reconstruction residuals.
    [4.3] Project residuals to normal space and decompose into frequency bands.
    [4.4] Score per-band norms against expected residual distributions (z-scores).
    [4.5] Apply robust adaptive threshold to produce boolean flags.

    Parameters
    ----------
    data : (n, d) data matrix (same normalisation as Phase 0 output).
    manifold : Manifold from manifold.build_manifold.

    Returns
    -------
    AnomalyResults
    """
    _reconstructed, raw_residuals = reconstruction.reconstruct(data, manifold)
    residual_data = residual_mod.compute(raw_residuals, data, manifold)
    band_scores = _score(residual_data, manifold)
    flags = _apply(band_scores)

    return AnomalyResults(
        scores=band_scores,
        flags=flags,
        residual_data=residual_data,
        n_anomalies=int(np.sum(flags.overall_flags)),
    )
