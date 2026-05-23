"""
Reconstructs each data point from its manifold chart coordinates using
the local eigenbasis, and computes the per-node reconstruction error
as the L2 distance between the original and reconstructed feature vectors.
"""
from __future__ import annotations

import numpy as np


def reconstruct(
    data: np.ndarray,
    manifold,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct every point via its primary chart and return residuals.

    Parameters
    ----------
    data : (n, d) data matrix.
    manifold : Manifold with atlas and chart objects.

    Returns
    -------
    tuple (reconstructed, residuals), each (n, d).
    """
    return _batch_reconstruct(data, manifold)


def _find_chart(point_index: int, manifold):
    """Return the primary Chart for *point_index*.

    Parameters
    ----------
    point_index : row index into the data matrix.
    manifold : Manifold with atlas.

    Returns
    -------
    Chart
    """
    ci = int(manifold.atlas.primary_assignments[point_index])
    return manifold.atlas.charts[ci]


def _encode(point: np.ndarray, chart) -> np.ndarray:
    """Project *point* to chart coordinates.

    Parameters
    ----------
    point : (d,) ambient-space vector.
    chart : Chart with chart_map callable.

    Returns
    -------
    np.ndarray : (intrinsic_dim,) chart coordinates.
    """
    return chart.chart_map(point)


def _decode(coords: np.ndarray, chart) -> np.ndarray:
    """Reconstruct an ambient-space point from chart coordinates.

    Parameters
    ----------
    coords : (intrinsic_dim,) chart coordinates.
    chart : Chart with chart_inverse callable.

    Returns
    -------
    np.ndarray : (d,) ambient-space reconstruction.
    """
    return chart.chart_inverse(coords)


def _batch_reconstruct(
    data: np.ndarray,
    manifold,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct all rows of *data* in memory-bounded batches.

    Groups points by their primary chart so matrix operations are vectorised
    per chart rather than applied point-by-point.

    Parameters
    ----------
    data : (n, d) data matrix.
    manifold : Manifold with atlas.
    batch_size : maximum rows to process per batch.

    Returns
    -------
    tuple (reconstructed, residuals), each (n, d).
    """
    n = data.shape[0]
    reconstructed = np.zeros_like(data)
    assignments = manifold.atlas.primary_assignments
    charts = manifold.atlas.charts

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_data = data[start:end]
        batch_assign = assignments[start:end]

        for ci, chart in enumerate(charts):
            local_mask = batch_assign == ci
            if not local_mask.any():
                continue
            local_data = batch_data[local_mask]           # (m, d)
            basis_vecs = chart.selected_vectors            # (d, k)
            # Encode: coords = basis_vecs.T @ local_data.T → (k, m); decode: basis_vecs @ coords → (d, m)
            coords = local_data @ basis_vecs               # (m, k)
            recon = coords @ basis_vecs.T                  # (m, d)
            reconstructed[start:end][local_mask] = recon

    return reconstructed, data - reconstructed
