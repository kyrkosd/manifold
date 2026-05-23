"""
Phase 2 — Fourier Analysis.

Public interface: ``analyze_fourier(data, structure) → SpectralData`` applies
the Graph Fourier Transform recommended by Phase 1 and returns spectral
coefficients, power spectrum, and frequency-band decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.types import EigenBasis, FrequencyBand
from structure.report import StructureReport

from . import band_decomposition, power_spectrum
from .graph_fourier import GraphFourierEngine

__all__ = ["SpectralData", "analyze_fourier"]


@dataclass
class SpectralData:
    """Container for all Phase 2 outputs.

    Parameters
    ----------
    coefficients : (n, d) GFT coefficients for every data point.
    power : Power-spectrum summary dict with keys per_point, mean,
        cumulative, dominant, centroid.
    bands : Frequency bands with start/end indices, labels, and power.
    basis : EigenBasis (eigenvectors + eigenvalues) used to compute the GFT.
    band_coefficients : Per-band coefficient slices keyed by band label.
    """

    # (n, d) spectral representation of the input data.
    coefficients: np.ndarray
    # Power-spectrum summary from power_spectrum.compute.
    power: dict
    # Ordered list of FrequencyBand objects (dc → noise).
    bands: list[FrequencyBand]
    # Eigenbasis of the feature-graph Laplacian.
    basis: EigenBasis
    # Sliced coefficients per band; key = band label, value = (n, band_width).
    band_coefficients: dict[str, np.ndarray] = field(default_factory=dict)


def analyze_fourier(data: np.ndarray, structure: StructureReport) -> SpectralData:
    """Apply the Graph Fourier Transform and decompose the spectrum into bands.

    Runs Phase 2 steps [2.1]–[2.5] from the pipeline specification:
    builds the GFT engine, transforms all data points, computes the power
    spectrum, and partitions the spectrum into frequency bands.

    Parameters
    ----------
    data : (n, d) normalised float64 array (output of Phase 0 ingestion).
    structure : StructureReport from Phase 1; must contain a non-None graph.

    Returns
    -------
    SpectralData : full spectral representation for downstream anomaly scoring.

    Raises
    ------
    ValueError : if structure.graph is None.
    """
    if structure.graph is None:
        raise ValueError("structure.graph is None; GFT requires a feature graph.")
    # Steps [2.1]-[2.2]: build eigenbasis and transform every data point.
    engine = GraphFourierEngine(structure.graph)
    coefficients = engine.transform(data)
    # Step [2.3]: compute per-point and mean power spectra.
    power = power_spectrum.compute(coefficients, engine.basis)
    # Steps [2.4]-[2.5]: detect spectral gaps and assign frequency-band labels.
    bands = band_decomposition.decompose(coefficients, engine.basis, power["mean"])
    # Build convenience dict mapping band label → coefficient slice.
    band_coeffs = {b.label: coefficients[:, b.start: b.end + 1] for b in bands}
    return SpectralData(
        coefficients=coefficients,
        power=power,
        bands=bands,
        basis=engine.basis,
        band_coefficients=band_coeffs,
    )
