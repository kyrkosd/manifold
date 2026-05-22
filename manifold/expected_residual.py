"""
Estimates the expected per-node reconstruction residual under a normal
(non-anomalous) manifold model, used to calibrate anomaly thresholds
and separate intrinsic approximation error from anomaly signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from common.types import EigenBasis, FrequencyBand
from manifold.tangent import _project_to_normal, compute_space

log = logging.getLogger(__name__)


@dataclass
class ExpectedResidualDistribution:
    """Per-band statistics of the reconstruction residual under normal operation.

    Parameters
    ----------
    per_band_mean : band index → mean ||r_b|| across region points.
    per_band_var : band index → variance of ||r_b|| across region points.
    n_samples : number of points used to fit the distribution.
    """

    per_band_mean: dict[int, float] = field(default_factory=dict)
    per_band_var: dict[int, float] = field(default_factory=dict)
    n_samples: int = 0


def compute(
    chart,
    data: np.ndarray,
    bands: list[FrequencyBand],
    basis: EigenBasis,
) -> ExpectedResidualDistribution:
    """Compute the expected residual distribution for *chart* over *data*.

    Pipeline:
      1. Encode each point via chart_map.
      2. Decode via chart_inverse to get reconstruction.
      3. Compute residual r = p - p̂.
      4. Project r to the normal space (centroid approximation).
      5. Decompose r_normal into frequency bands.
      6. Fit per-band Gaussian statistics.

    Parameters
    ----------
    chart : Chart dataclass.
    data : (n, d) region data (points belonging to this chart).
    bands : list of FrequencyBand objects.
    basis : global EigenBasis (eigenvectors used to define the bands).

    Returns
    -------
    ExpectedResidualDistribution
    """
    if len(data) == 0:
        return ExpectedResidualDistribution(n_samples=0)

    residuals = _encode_decode_residuals(data, chart)
    normal_residuals = _project_residuals_to_normal(residuals, data, chart)
    band_norms = _decompose_to_bands(normal_residuals, bands, basis)

    per_mean: dict[int, float] = {}
    per_var: dict[int, float] = {}
    for band_idx, norms in band_norms.items():
        mean, var = _fit_distribution(norms)
        per_mean[band_idx] = mean
        per_var[band_idx] = var

    return ExpectedResidualDistribution(
        per_band_mean=per_mean,
        per_band_var=per_var,
        n_samples=len(data),
    )


def _encode_decode_residuals(data: np.ndarray, chart) -> np.ndarray:
    """Batch encode-decode each row of *data* and return residuals.

    Parameters
    ----------
    data : (n, d) data matrix.
    chart : Chart with chart_map and chart_inverse.

    Returns
    -------
    np.ndarray : (n, d) residuals p - p̂.
    """
    reconstructed = np.stack(
        [chart.chart_inverse(chart.chart_map(p)) for p in data]
    )
    return data - reconstructed


def _project_residuals_to_normal(
    residuals: np.ndarray,
    data: np.ndarray,
    chart,
) -> np.ndarray:
    """Project residuals onto the normal space using a centroid approximation.

    The tangent/normal spaces are computed at the region centroid and reused
    for all points — this is exact for linear charts and an approximation for
    curved ones.

    Parameters
    ----------
    residuals : (n, d) residual vectors.
    data : (n, d) original data points.
    chart : Chart with chart_map and chart_inverse.

    Returns
    -------
    np.ndarray : (n, d) normal-space components of each residual.
    """
    log.warning(
        "Using centroid-tangent-space approximation for normal projection "
        "(exact for linear charts; approximate for curved manifolds)."
    )
    centroid = np.mean(data, axis=0)
    ts = compute_space(centroid, chart.chart_map, chart.chart_inverse)
    normal_basis = ts.normal_basis   # (d, normal_dim)

    if normal_basis.shape[1] == 0:
        # Chart spans full ambient space; all residuals are zero in normal space.
        return np.zeros_like(residuals)

    return np.stack([_project_to_normal(r, normal_basis) for r in residuals])


def _decompose_to_bands(
    normal_residuals: np.ndarray,
    bands: list[FrequencyBand],
    basis: EigenBasis,
) -> dict[int, np.ndarray]:
    """Project normal residuals onto each frequency band subspace.

    Parameters
    ----------
    normal_residuals : (n, d) normal-space residual vectors.
    bands : list of FrequencyBand with start/end eigenvector indices.
    basis : EigenBasis whose eigenvectors define the frequency decomposition.

    Returns
    -------
    dict mapping band index → (n,) array of projected norms.
    """
    result: dict[int, np.ndarray] = {}
    for band_idx, band in enumerate(bands):
        end = min(band.end + 1, basis.eigenvectors.shape[1])
        band_vecs = basis.eigenvectors[:, band.start:end]   # (d, width)
        if band_vecs.shape[1] == 0:
            result[band_idx] = np.zeros(normal_residuals.shape[0])
            continue
        # Project each normal residual onto this band's subspace.
        proj = normal_residuals @ band_vecs   # (n, width)
        result[band_idx] = np.linalg.norm(proj, axis=1)   # (n,)
    return result


def _fit_distribution(band_norms: np.ndarray) -> tuple[float, float]:
    """Estimate mean and variance of *band_norms* (Gaussian assumption).

    Parameters
    ----------
    band_norms : (n,) non-negative norm values.

    Returns
    -------
    tuple (mean, variance).
    """
    if len(band_norms) == 0:
        return 0.0, 0.0
    return float(np.mean(band_norms)), float(np.var(band_norms))
