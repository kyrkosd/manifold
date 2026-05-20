"""
Shared type aliases, enumerations, and base dataclasses used across all
FMAS pipeline stages. All downstream modules import their shared contracts
from here rather than defining them locally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class StructureType(Enum):
    """Topological character of the data manifold."""

    LINEAR = "linear"
    GRID = "grid"
    GRAPH = "graph"
    UNKNOWN = "unknown"


class FourierType(Enum):
    """Fourier engine selected by the structure discovery phase."""

    STANDARD_FFT = "fft"
    MULTIDIM_FFT = "multidim_fft"
    GRAPH_FOURIER = "graph_fourier"


class AnomalyType(Enum):
    """Classification of an anomalous point by the number of affected bands."""

    SPECTRAL = "spectral"    # one band anomalous
    PARTIAL = "partial"      # multiple bands anomalous
    GLOBAL = "global"        # all bands anomalous
    NORMAL = "normal"        # not anomalous


class SpatialType(Enum):
    """Spatial distribution of anomalous points in the manifold."""

    ISOLATED = "isolated"
    REGIONAL = "regional"


class RelationshipType(Enum):
    """Relationship of an anomalous cluster to the parent manifold."""

    SUBMANIFOLD = "on_manifold"
    SEPARATE = "separate_manifold"
    SCATTERED = "scattered_points"


class Severity(Enum):
    """Anomaly severity level."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class BoundaryType(Enum):
    """Character of the boundary between an anomalous region and normal data."""

    SHARP = "sharp"
    GRADUAL = "gradual"
    DIFFUSE = "diffuse"


class AlignmentQuality(Enum):
    """Quality of Procrustes eigenvector alignment across chart boundaries.

    Thresholds are based on the Procrustes residual magnitude:
    EXCELLENT < 0.05, GOOD < 0.15, ACCEPTABLE < 0.30, POOR < 0.50,
    DIVERGENT >= 0.50 (genuine structural difference between regions).
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    DIVERGENT = "divergent"


@dataclass
class FeatureGraph:
    """Graph over data points with an associated node-feature signal.

    Parameters
    ----------
    adjacency:
        Symmetric (n, n) weight matrix; entry [i, j] is the edge weight.
    nodes:
        Ordered list of node identifiers matching adjacency rows/columns.
    stats:
        Summary statistics (e.g. ``n_edges``, ``density``, ``is_connected``).
    """

    adjacency: np.ndarray
    nodes: list[str]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyBand:
    """A contiguous slice of the graph Laplacian spectrum.

    Parameters
    ----------
    start:
        Index of the first eigenvalue in this band (inclusive).
    end:
        Index of the last eigenvalue in this band (inclusive).
    label:
        Human-readable name, e.g. ``"low"``, ``"mid"``, ``"high"``.
    power:
        Total spectral power captured by this band.
    """

    start: int
    end: int
    label: str
    power: float


@dataclass
class EigenBasis:
    """Result of an eigen-decomposition of a symmetric matrix.

    Parameters
    ----------
    eigenvectors:
        (n, k) array; each column is an eigenvector.
    eigenvalues:
        (k,) array of corresponding eigenvalues in ascending order.
    """

    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
