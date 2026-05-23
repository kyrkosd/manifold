"""
Assembles the final AnomalyReport dataclass from scored results, applied
thresholds, pipeline metadata, and optional per-node explanations (band
contributions, nearest normal neighbours).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from anomaly import AnomalyResults


@dataclass
class AnomalyReport:
    """Complete anomaly report produced by the reporting phase.

    Parameters
    ----------
    table : per-point DataFrame sorted by overall_score descending.
    type_distribution : counts of spectral/partial/global anomalies.
    top_anomalies : list of dicts for the highest-scoring anomalies.
    summary : statistical summary of the detection run.
    """

    table: pd.DataFrame
    type_distribution: dict
    top_anomalies: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def report(results: AnomalyResults, data: np.ndarray) -> AnomalyReport:
    """Assemble a complete AnomalyReport from *results*.

    Parameters
    ----------
    results : AnomalyResults from anomaly.detect_anomalies.
    data : (n, d) original data matrix (shape used for summary only).

    Returns
    -------
    AnomalyReport
    """
    table = _per_point_table(results)
    type_dist = _type_distribution(results)
    top = _top_anomalies(results)
    summary = _statistical_summary(results)
    return AnomalyReport(
        table=table,
        type_distribution=type_dist,
        top_anomalies=top,
        summary=summary,
    )


def _per_point_table(results: AnomalyResults) -> pd.DataFrame:
    """Build a per-point DataFrame sorted by overall_score descending.

    Columns: point_index, overall_score, is_anomaly, band_<k>_score …,
    top_anomalous_band.

    Parameters
    ----------
    results : AnomalyResults.

    Returns
    -------
    pd.DataFrame
    """
    n = len(results.scores.overall)
    band_indices = sorted(results.scores.per_band.keys())

    rows: dict[str, object] = {
        "point_index": np.arange(n, dtype=int),
        "overall_score": results.scores.overall.astype(float),
        "is_anomaly": results.flags.overall_flags.astype(bool),
    }
    for b in band_indices:
        rows[f"band_{b}_score"] = results.scores.per_band[b].astype(float)

    if band_indices:
        abs_matrix = np.abs(
            np.column_stack([results.scores.per_band[b] for b in band_indices])
        )
        top_pos = np.argmax(abs_matrix, axis=1)
        rows["top_anomalous_band"] = [f"band_{band_indices[i]}" for i in top_pos]
    else:
        rows["top_anomalous_band"] = [""] * n

    df = pd.DataFrame(rows)
    return df.sort_values("overall_score", ascending=False).reset_index(drop=True)


def _type_distribution(results: AnomalyResults) -> dict:
    """Count anomalies by the number of flagged bands.

    spectral  — exactly one band flagged.
    partial   — more than one but fewer than all bands.
    global    — every band flagged (or only one band exists).

    Parameters
    ----------
    results : AnomalyResults.

    Returns
    -------
    dict with keys "spectral", "partial", "global".
    """
    band_keys = sorted(results.flags.per_band_flags.keys())
    n_bands = len(band_keys)
    overall_mask = results.flags.overall_flags

    if n_bands == 0 or not overall_mask.any():
        return {"spectral": 0, "partial": 0, "global": 0}

    flag_matrix = np.column_stack(
        [results.flags.per_band_flags[b] for b in band_keys]
    )
    band_counts = flag_matrix[overall_mask].sum(axis=1).astype(int)

    if n_bands == 1:
        return {"spectral": 0, "partial": 0, "global": int(band_counts.sum())}

    return {
        "spectral": int(np.sum(band_counts == 1)),
        "partial":  int(np.sum((band_counts > 1) & (band_counts < n_bands))),
        "global":   int(np.sum(band_counts == n_bands)),
    }


def _top_anomalies(results: AnomalyResults, n: int = 10) -> list[dict]:
    """Return the top *n* anomalies sorted by overall_score descending.

    Each entry includes point_index, overall_score, flagged_bands, and
    per-band z-scores.

    Parameters
    ----------
    results : AnomalyResults.
    n : maximum number of anomalies to return.

    Returns
    -------
    list[dict]
    """
    anomaly_indices = np.where(results.flags.overall_flags)[0]
    if len(anomaly_indices) == 0:
        return []

    scores = results.scores.overall[anomaly_indices]
    order = np.argsort(scores)[::-1][:n]
    top_indices = anomaly_indices[order]

    band_keys = sorted(results.scores.per_band.keys())
    out = []
    for idx in top_indices:
        flagged = [b for b in band_keys if results.flags.per_band_flags[b][idx]]
        out.append({
            "point_index":   int(idx),
            "overall_score": float(results.scores.overall[idx]),
            "is_anomaly":    True,
            "flagged_bands": flagged,
            "band_scores": {
                f"band_{b}": float(results.scores.per_band[b][idx])
                for b in band_keys
            },
        })
    return out


def _statistical_summary(results: AnomalyResults) -> dict:
    """Compute high-level statistics of the anomaly detection run.

    Returns a flat dict with: total_points, total_anomalies, anomaly_rate,
    mean/std/max overall_score, and per-band anomaly counts.

    Parameters
    ----------
    results : AnomalyResults.
    """
    n = len(results.scores.overall)
    n_anom = int(results.n_anomalies)
    scores = results.scores.overall

    per_band_counts = {
        f"band_{b}_anomaly_count": int(np.sum(v))
        for b, v in results.flags.per_band_flags.items()
    }

    return {
        "total_points":    int(n),
        "total_anomalies": n_anom,
        "anomaly_rate":    float(n_anom / n) if n > 0 else 0.0,
        "mean_score":      float(np.mean(scores)) if n > 0 else 0.0,
        "std_score":       float(np.std(scores)) if n > 0 else 0.0,
        "max_score":       float(np.max(scores)) if n > 0 else 0.0,
        **per_band_counts,
    }
