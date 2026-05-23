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


def _reconstruct_batch(
    batch_data: np.ndarray,
    batch_assign: np.ndarray,
    charts: list,
) -> np.ndarray:
    """Project one batch of points through their assigned chart bases."""
    result = np.zeros_like(batch_data)
    for ci, chart in enumerate(charts):
        mask = batch_assign == ci
        if not mask.any():
            continue
        local_data = batch_data[mask]
        basis_vecs = chart.selected_vectors
        result[mask] = (local_data @ basis_vecs) @ basis_vecs.T
    return result


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
        reconstructed[start:end] = _reconstruct_batch(
            data[start:end], assignments[start:end], charts
        )
    return reconstructed, data - reconstructed
