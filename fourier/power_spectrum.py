"""
Computes per-node and global power spectra from GFT coefficients,
including spectral energy distribution across eigenvalue bins and
cumulative energy curves used for band boundary selection.
"""
from __future__ import annotations

import numpy as np

from common.types import EigenBasis


def compute(coefficients: np.ndarray, basis: EigenBasis) -> dict:
    """Compute a full power-spectrum summary from GFT coefficients.

    Parameters
    ----------
    coefficients : (n, d) array of GFT coefficients from GraphFourierEngine.
    basis : EigenBasis whose eigenvalues index the spectral components.

    Returns
    -------
    dict with keys:
        per_point  (n, d) squared coefficients per observation.
        mean       (d,) mean spectral power across all n points.
        cumulative (d,) normalised cumulative sum of mean power.
        dominant   list[int] top-5 indices sorted by descending mean power.
        centroid   float power-weighted mean eigenvalue.
    """
    per_point = _per_point_power(coefficients)
    mean = _mean_power(per_point)
    cumulative = _cumulative_power(mean)
    dominant = _dominant_frequencies(mean)
    centroid = _spectral_centroid(mean, basis.eigenvalues)
    return {
        "per_point": per_point,
        "mean": mean,
        "cumulative": cumulative,
        "dominant": dominant,
        "centroid": centroid,
    }


def _per_point_power(coefficients: np.ndarray) -> np.ndarray:
    """Element-wise squared GFT coefficients; shape (n, d)."""
    return coefficients ** 2


def _mean_power(per_point: np.ndarray) -> np.ndarray:
    """Average spectral power across all n observations; shape (d,)."""
    return per_point.mean(axis=0)


def _cumulative_power(mean_power: np.ndarray) -> np.ndarray:
    """Normalised cumulative sum of mean power; final value is 1.0."""
    total = mean_power.sum()
    if total < 1e-12:
        # Guard against all-zero spectrum (e.g., constant data).
        return np.zeros_like(mean_power)
    return np.cumsum(mean_power) / total


def _dominant_frequencies(mean_power: np.ndarray, n: int = 5) -> list[int]:
    """Indices of the top-n spectral components by mean power, descending."""
    top = min(n, len(mean_power))
    return list(np.argsort(mean_power)[::-1][:top])


def _spectral_centroid(mean_power: np.ndarray, eigenvalues: np.ndarray) -> float:
    """Power-weighted mean eigenvalue; indicates the dominant spectral frequency."""
    total = mean_power.sum()
    if total < 1e-12:
        return 0.0
    return float(np.dot(mean_power, eigenvalues) / total)


def _compare_spectra(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """L2 distance between two mean-power spectra of equal length."""
    return float(np.linalg.norm(spec1 - spec2))
