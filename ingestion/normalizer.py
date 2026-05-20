"""
Feature normalisation transforms applied during FMAS Phase 0 ingestion.
NaN values are preserved; statistics are computed ignoring them.
"""
from __future__ import annotations

import numpy as np

from common.logging import get_logger
from .types import NormParams

_log = get_logger(__name__)


def normalize(
    data: np.ndarray,
    method: str = "standard",
) -> tuple[np.ndarray, NormParams]:
    """Normalise *data* and return the transformed array with its parameters.

    Parameters
    ----------
    data : (n, d) float64 array.
    method : normalisation strategy; only ``"standard"`` is supported in MVP.

    Returns
    -------
    normalised : (n, d) zero-mean, unit-variance array (NaN values preserved).
    params : NormParams for inverse transformation.
    """
    method = _select_method(data)
    centred, means = _zero_mean(data)
    normalised, stds = _unit_variance(centred)
    return normalised, NormParams(means=means, stds=stds, method=method)


def _zero_mean(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract per-column mean computed ignoring NaN.

    Parameters
    ----------
    data : (n, d) float64 array.

    Returns
    -------
    centred : (n, d) zero-mean array.
    means : (d,) per-column means; 0.0 for all-NaN columns.
    """
    means = np.nanmean(data, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return data - means, means


def _unit_variance(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide by per-column std computed ignoring NaN.

    Zero-variance and all-NaN columns are left unchanged (effective std = 1.0).

    Parameters
    ----------
    data : (n, d) float64 array.

    Returns
    -------
    scaled : (n, d) unit-variance array.
    stds : (d,) effective standard deviations (≥ 1e-10 or 1.0 for degenerate cols).
    """
    raw_stds = np.nanstd(data, axis=0, ddof=0)
    safe_stds = np.where((raw_stds < 1e-10) | np.isnan(raw_stds), 1.0, raw_stds)
    return data / safe_stds, safe_stds


def inverse_normalize(data: np.ndarray, params: NormParams) -> np.ndarray:
    """Reverse a standard normalisation using stored parameters.

    Parameters
    ----------
    data : (n, d) normalised array.
    params : NormParams from the forward normalisation pass.

    Returns
    -------
    np.ndarray : (n, d) array restored to the original data scale.
    """
    return data * params.stds + params.means


def _select_method(data: np.ndarray) -> str:  # noqa: ARG001
    """Return the normalisation method to apply.

    Always returns ``"standard"`` in the MVP; reserved for future heuristics.

    Parameters
    ----------
    data : (n, d) array (not inspected; reserved for future use).

    Returns
    -------
    str : ``"standard"``.
    """
    return "standard"
