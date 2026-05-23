"""
Computes normalised residuals between original and band-reconstructed
signals, decomposed per frequency band so that anomaly character
(low-frequency structural vs. high-frequency noise) can be distinguished.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from common.types import EigenBasis, FrequencyBand
from manifold.tangent import _project_to_normal, compute_normal_space

log = logging.getLogger(__name__)


@dataclass
class ResidualData:
    """Residuals decomposed into total, normal-space, and per-band components.

    Parameters
    ----------
    total_residuals : (n, d) raw reconstruction residuals (p - p̂).
    normal_residuals : (n, d) residuals projected onto the chart normal space.
    per_band_norms : band_index → (n,) per-point ||r_normal_b|| norms.
    """

    total_residuals: np.ndarray               # (n, d)
    normal_residuals: np.ndarray              # (n, d)
    per_band_norms: dict[int, np.ndarray] = field(default_factory=dict)


def compute(
    residuals: np.ndarray,
    data: np.ndarray,
    manifold,
) -> ResidualData:
    """Compute normal-space and band-decomposed residuals.

    Parameters
    ----------
    residuals : (n, d) raw residuals from anomaly.reconstruction.
    data : (n, d) original data points.
    manifold : Manifold with atlas, expected_residuals, and spectral data.

    Returns
    -------
    ResidualData
    """
    normal_residuals = _batch_normal_residuals(residuals, data, manifold)
    bands = manifold.spectral.bands
    basis = manifold.spectral.basis
    per_band_norms = _decompose_normal_to_bands(normal_residuals, bands, basis)

    return ResidualData(
        total_residuals=residuals,
        normal_residuals=normal_residuals,
        per_band_norms=per_band_norms,
    )


def _normal_residual(
    residual: np.ndarray,
    point: np.ndarray,
    chart,
) -> np.ndarray:
    """Project *residual* onto the normal space of *chart* at *point*.

    For MVP linear charts the tangent space is constant, so the result does
    not depend on *point*.  Computing the normal basis here is exact for
    linear charts; for curved manifolds use _batch_normal_residuals which
    applies the centroid approximation.

    Parameters
    ----------
    residual : (d,) reconstruction residual.
    point : (d,) original data point (unused for linear charts; kept for API).
    chart : Chart with selected_vectors.

    Returns
    -------
    np.ndarray : (d,) normal-space component of *residual*.
    """
    normal_basis = compute_normal_space(chart.selected_vectors, chart.ambient_dim)
    return _project_to_normal(residual, normal_basis)


def _batch_normal_residuals(
    residuals: np.ndarray,
    data: np.ndarray,
    manifold,
) -> np.ndarray:
    """Project all residuals onto their chart's normal space (batched by chart).

    Precomputes one normal basis per chart for efficiency.

    Parameters
    ----------
    residuals : (n, d) raw residuals.
    data : (n, d) original data (unused for linear charts).
    manifold : Manifold with atlas.

    Returns
    -------
    np.ndarray : (n, d) normal-space residuals.
    """
    charts = manifold.atlas.charts
    assignments = manifold.atlas.primary_assignments

    # Precompute normal projection matrices: P_N = N @ N.T for each chart.
    normal_projs: dict[int, np.ndarray] = {}
    for ci, chart in enumerate(charts):
        nb = compute_normal_space(chart.selected_vectors, chart.ambient_dim)
        if nb.shape[1] > 0:
            normal_projs[ci] = nb @ nb.T   # (d, d) projection matrix
        else:
            normal_projs[ci] = np.zeros((chart.ambient_dim, chart.ambient_dim))

    normal_residuals = np.zeros_like(residuals)
    for ci, proj_normal in normal_projs.items():
        mask = assignments == ci
        if mask.any():
            normal_residuals[mask] = residuals[mask] @ proj_normal.T  # (m, d) @ (d, d)

    return normal_residuals


def _decompose_normal_to_bands(
    normal_residuals: np.ndarray,
    bands: list[FrequencyBand],
    basis: EigenBasis,
) -> dict[int, np.ndarray]:
    """Decompose normal residuals into per-band ||r_normal_b|| norms.

    For each band, projects normal residuals onto the band's eigenvector subspace
    and computes the per-point norm of the projection.

    Parameters
    ----------
    normal_residuals : (n, d) normal-space residuals.
    bands : list of FrequencyBand with start/end eigenvector indices.
    basis : EigenBasis defining the spectral decomposition.

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
        proj = normal_residuals @ band_vecs                 # (n, width)
        result[band_idx] = np.linalg.norm(proj, axis=1)    # (n,)
    return result
