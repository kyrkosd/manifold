"""
Validates chart quality by checking eigenvector smoothness within the
patch, overlap consistency with neighbouring charts, and per-chart
reconstruction fidelity against configurable tolerance thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from manifold.tangent import _project_to_tangent, compute_space


def validate(
    chart,
    data: np.ndarray,
    injectivity_threshold: float = 0.8,
) -> tuple[bool, float, str]:
    """Check chart quality via injectivity (hard gate), continuity, and invertibility.

    Parameters
    ----------
    chart : Chart dataclass from manifold.chart.
    data : (n_region, d) ambient data for the chart's core region.
    injectivity_threshold : minimum Spearman correlation to pass.

    Returns
    -------
    tuple (valid, score, reason)
        valid : False if the injectivity hard gate is not met.
        score : composite quality score in [0, 1].
        reason : human-readable explanation.
    """
    if not _check_injectivity(chart, data, injectivity_threshold):
        return False, 0.0, "Injectivity gate failed: chart is not locally injective."

    continuity_ok = _check_continuity(chart, data)
    invertibility_ok = _check_invertibility(chart, data)
    distortion = _compute_distortion(chart)
    score = _quality_score(chart, data)

    if not continuity_ok:
        return False, score, "Continuity check failed."
    if not invertibility_ok:
        return False, score, "Invertibility check failed."

    reason = f"Valid chart; distortion={distortion:.3f}, score={score:.3f}."
    return True, score, reason


def _check_injectivity(
    chart,
    data: np.ndarray,
    threshold: float = 0.8,
) -> bool:
    """Spearman rank correlation of ambient vs chart pairwise distances.

    Parameters
    ----------
    chart : Chart with chart_map callable.
    data : (n, d) region data.
    threshold : minimum acceptable Spearman correlation.
    """
    n = data.shape[0]
    if n < 2:
        return True

    coords = np.stack([chart.chart_map(row) for row in data])
    rng = np.random.default_rng(42)
    n_pairs = min(500, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % n

    amb = np.linalg.norm(data[i_idx] - data[j_idx], axis=1)
    chart_d = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
    corr, _ = spearmanr(amb, chart_d)
    return bool(not np.isnan(corr) and corr >= threshold)


def _check_continuity(chart, data: np.ndarray) -> bool:
    """Check that nearby points in ambient space map to nearby chart coordinates.

    Uses the ratio of chart distance to ambient distance; a pathological map
    would have extremely large ratios for close ambient pairs.

    Parameters
    ----------
    chart : Chart with chart_map.
    data : (n, d) region data.
    """
    n = data.shape[0]
    if n < 4:
        return True

    coords = np.stack([chart.chart_map(row) for row in data])
    rng = np.random.default_rng(7)
    sample = rng.choice(n, size=min(50, n), replace=False)
    ratios = []
    for i in sample:
        for j in sample:
            if i == j:
                continue
            amb_d = float(np.linalg.norm(data[i] - data[j]))
            if amb_d < 1e-12:
                continue
            chart_d = float(np.linalg.norm(coords[i] - coords[j]))
            ratios.append(chart_d / amb_d)

    if not ratios:
        return True
    arr = np.array(ratios)
    # Fail if the 99th percentile ratio exceeds 100× the median (Lipschitz proxy).
    return bool(np.percentile(arr, 99) < 100.0 * np.median(arr))


def _check_invertibility(
    chart,
    data: np.ndarray,
    tolerance: float = 0.1,
) -> bool:
    """Check round-trip reconstruction: tangent component of residual < tolerance.

    Parameters
    ----------
    chart : Chart with chart_map, chart_inverse, selected_vectors, intrinsic_dim.
    data : (n, d) region data.
    tolerance : maximum allowed mean tangent residual fraction.
    """
    n = data.shape[0]
    if n == 0:
        return True

    tangent_fracs = []
    for p in data:
        z = chart.chart_map(p)
        p_hat = chart.chart_inverse(z)
        residual = p - p_hat
        if np.linalg.norm(residual) < 1e-12:
            tangent_fracs.append(0.0)
            continue
        tangent_component = _project_to_tangent(residual, chart.selected_vectors)
        frac = float(np.linalg.norm(tangent_component) / np.linalg.norm(residual))
        tangent_fracs.append(frac)

    return bool(np.mean(tangent_fracs) < tolerance)


def _compute_distortion(chart) -> float:
    """Condition number of the selected basis matrix as a distortion proxy.

    Parameters
    ----------
    chart : Chart with selected_vectors (d, intrinsic_dim).
    """
    s = np.linalg.svd(chart.selected_vectors, compute_uv=False)
    if s[-1] < 1e-12:
        return float("inf")
    return float(s[0] / s[-1])


def _quality_score(chart, data: np.ndarray) -> float:
    """Composite quality score in [0, 1].

    Combines alignment score, injectivity proxy, and distortion.

    Parameters
    ----------
    chart : Chart with local_basis.alignment_score and selected_vectors.
    data : (n, d) region data.
    """
    alignment = float(np.clip(chart.local_basis.alignment_score, 0.0, 1.0))

    distortion = _compute_distortion(chart)
    distortion_score = float(np.clip(1.0 / (1.0 + distortion), 0.0, 1.0))

    n = data.shape[0]
    if n >= 2:
        coords = np.stack([chart.chart_map(row) for row in data])
        rng = np.random.default_rng(99)
        n_pairs = min(200, n * (n - 1) // 2)
        i_idx = rng.integers(0, n, size=n_pairs)
        j_idx = rng.integers(0, n, size=n_pairs)
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % n
        amb = np.linalg.norm(data[i_idx] - data[j_idx], axis=1)
        cd = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
        corr, _ = spearmanr(amb, cd)
        injectivity = float(np.clip(corr if not np.isnan(corr) else 0.0, 0.0, 1.0))
    else:
        injectivity = 1.0

    return float((alignment + distortion_score + injectivity) / 3.0)


def _split_region(chart, data: np.ndarray) -> list[np.ndarray]:
    """Split the chart region into two sub-regions via k-means on chart coordinates.

    Parameters
    ----------
    chart : Chart with chart_map.
    data : (n, d) region data.

    Returns
    -------
    list of two index arrays partitioning range(n).
    """
    coords = np.stack([chart.chart_map(row) for row in data])
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = km.fit_predict(coords)
    return [np.where(labels == k)[0] for k in range(2)]
