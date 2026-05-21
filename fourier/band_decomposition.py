"""
Partitions the graph spectrum into low-, mid-, and high-frequency bands
using configurable eigenvalue cutoffs, and reconstructs band-limited
signals by masking GFT coefficients and applying the inverse GFT.
"""
from __future__ import annotations

import numpy as np

from common.types import EigenBasis, FrequencyBand

# Ratio between consecutive power values that signals a band boundary.
_GAP_RATIO_THRESHOLD = 3.0
# Number of uniform bands used when no spectral gap is found.
_DEFAULT_N_BANDS = 3


def decompose(
    coefficients: np.ndarray,
    basis: EigenBasis,
    mean_power: np.ndarray,
) -> list[FrequencyBand]:
    """Partition the spectrum into frequency bands using spectral gaps.

    Parameters
    ----------
    coefficients : (n, d) GFT coefficient array from GraphFourierEngine.
    basis : EigenBasis for the feature graph (eigenvalues used for ordering).
    mean_power : (d,) mean spectral power per component from power_spectrum.

    Returns
    -------
    list[FrequencyBand] ordered from lowest (dc) to highest (noise) frequency.
    """
    d = basis.eigenvalues.shape[0]
    gaps = _find_spectral_gaps(mean_power)
    if gaps:
        # Gap-based partition: place band boundaries at detected jumps.
        boundaries = _create_bands_at_gaps(gaps, d)
    else:
        # No significant gaps; fall back to uniform equal-size partition.
        boundaries = _create_uniform_bands(d, _DEFAULT_N_BANDS)
    labels = _label_bands(boundaries)
    bands = []
    for (start, end), label in zip(boundaries, labels):
        # Compute total squared energy in this band slice.
        power = _compute_band_power(coefficients, start, end)
        bands.append(FrequencyBand(start=start, end=end, label=label, power=power))
    return bands


def _find_spectral_gaps(mean_power: np.ndarray) -> list[int]:
    """Return indices where consecutive power ratios exceed the gap threshold.

    A gap at index i places a band boundary between components i and i+1.
    """
    d = len(mean_power)
    if d < 2:
        return []
    # Ratio r[i] = mean_power[i+1] / mean_power[i]; large ratio = upward jump.
    safe_denom = np.maximum(mean_power[:-1], 1e-10)
    ratios = mean_power[1:] / safe_denom
    median_ratio = np.median(ratios)
    if median_ratio < 1e-10:
        return []
    threshold = _GAP_RATIO_THRESHOLD * median_ratio
    return [int(i) for i in np.where(ratios > threshold)[0]]


def _create_bands_at_gaps(gaps: list[int], d: int) -> list[tuple[int, int]]:
    """Build (start, end) band intervals from sorted gap indices."""
    boundaries = []
    start = 0
    for gap in sorted(gaps):
        # Guard against a gap at position d-1 which would create an empty band.
        if gap + 1 < d:
            boundaries.append((start, gap))
            start = gap + 1
    # Final band always extends to the last spectral component.
    boundaries.append((start, d - 1))
    return boundaries


def _create_uniform_bands(d: int, n_bands: int = 3) -> list[tuple[int, int]]:
    """Divide d spectral components into n_bands approximately equal intervals."""
    n = min(n_bands, d)
    # Compute integer split edges; round to nearest integer for even spreading.
    edges = [int(round(d * i / n)) for i in range(n + 1)]
    boundaries = []
    for i in range(n):
        start = edges[i]
        # All but the last band end one before the next edge to avoid overlap.
        end = edges[i + 1] - 1 if i < n - 1 else d - 1
        boundaries.append((start, end))
    return boundaries


def _assign_coefficients(
    coefficients: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Return the slice of coefficient columns in [start, end] inclusive."""
    return coefficients[:, start: end + 1]


def _compute_band_power(
    coefficients: np.ndarray, start: int, end: int
) -> float:
    """Total squared energy of GFT coefficients in band [start, end]."""
    band_slice = _assign_coefficients(coefficients, start, end)
    return float((band_slice ** 2).sum())


def _label_bands(bands: list[tuple[int, int]]) -> list[str]:
    """Assign position-based labels: dc, low, mid, high, noise."""
    n = len(bands)
    if n == 1:
        return ["all"]
    if n == 2:
        return ["dc", "noise"]
    if n == 3:
        # Three-band case: skip separate low/high labels.
        return ["dc", "mid", "noise"]
    # Four or more: assign boundary labels then fill interior with "mid".
    labels = ["mid"] * n
    labels[0] = "dc"
    labels[1] = "low"
    labels[-2] = "high"
    labels[-1] = "noise"
    return labels


def _validate_coverage(bands: list[FrequencyBand], d: int) -> bool:
    """Return True if *bands* cover every index in [0, d-1] without overlap."""
    covered: set[int] = set()
    for band in bands:
        for i in range(band.start, band.end + 1):
            if i in covered:
                return False
            covered.add(i)
    return covered == set(range(d))
