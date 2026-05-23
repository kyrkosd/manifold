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

# ---------------------------------------------------------------------------
# Enumerations — pipeline contracts
# ---------------------------------------------------------------------------

class StructureType(Enum):
    """Topological character of the data manifold."""

    # Points fall along a 1-D curve or affine subspace.
    LINEAR = "linear"
    # Points are arranged on a regular lattice (e.g. image pixels).
    GRID = "grid"
    # Points are connected by an explicit graph; structure is non-Euclidean.
    GRAPH = "graph"
    # Structure detection produced no confident result; treated conservatively.
    UNKNOWN = "unknown"


class FourierType(Enum):
    """Fourier engine selected by the structure discovery phase."""

    # Conventional 1-D DFT; used when StructureType is LINEAR.
    STANDARD_FFT = "fft"
    # Multi-dimensional DFT; used when StructureType is GRID.
    MULTIDIM_FFT = "multidim_fft"
    # Graph Fourier Transform via Laplacian eigenvectors; default for tabular data.
    GRAPH_FOURIER = "graph_fourier"


class AnomalyType(Enum):
    """Classification of an anomalous point by the number of affected bands."""

    # Exactly one frequency band exceeds the expected residual threshold.
    SPECTRAL = "spectral"
    # Two or more bands — but not all — are anomalous.
    PARTIAL = "partial"
    # Every frequency band is anomalous; indicates a large structural deviation.
    GLOBAL = "global"
    # No band is anomalous; point lies within the expected residual distribution.
    NORMAL = "normal"


class SpatialType(Enum):
    """Spatial distribution of anomalous points in the manifold."""

    # Each anomalous point is surrounded only by normal points.
    ISOLATED = "isolated"
    # Anomalous points form a connected cluster covering a local region.
    REGIONAL = "regional"


class RelationshipType(Enum):
    """Relationship of an anomalous cluster to the parent manifold."""

    # Anomalies lie on the manifold surface; their tangent structure matches.
    SUBMANIFOLD = "on_manifold"
    # Anomalies form a distinct manifold with a different intrinsic geometry.
    SEPARATE = "separate_manifold"
    # Anomalies have no coherent structure; they are isolated outliers.
    SCATTERED = "scattered_points"


class Severity(Enum):
    """Anomaly severity level derived from the residual magnitude."""

    # Residual is elevated but within 2σ of the per-chart expected distribution.
    MILD = "mild"
    # Residual is between 2σ and 4σ; worth flagging for review.
    MODERATE = "moderate"
    # Residual exceeds 4σ; high confidence structural anomaly.
    SEVERE = "severe"


class BoundaryType(Enum):
    """Character of the boundary between an anomalous region and normal data."""

    # Anomaly score drops abruptly at the region edge (step-function profile).
    SHARP = "sharp"
    # Score falls off smoothly over a few points (ramp profile).
    GRADUAL = "gradual"
    # Boundary is indistinct; normal and anomalous points intermix.
    DIFFUSE = "diffuse"


class AlignmentQuality(Enum):
    """Quality of Procrustes eigenvector alignment across chart boundaries.

    Thresholds are based on the Procrustes residual magnitude:
    EXCELLENT < 0.05, GOOD < 0.15, ACCEPTABLE < 0.30, POOR < 0.50,
    DIVERGENT >= 0.50 (genuine structural difference between regions).
    """

    # Procrustes residual < 0.05; eigenbases are effectively identical.
    EXCELLENT = "excellent"
    # Residual in [0.05, 0.15); reliable alignment with minor sign ambiguity.
    GOOD = "good"
    # Residual in [0.15, 0.30); alignment is usable but introduces small errors.
    ACCEPTABLE = "acceptable"
    # Residual in [0.30, 0.50); alignment is unreliable; interpret with caution.
    POOR = "poor"
    # Residual >= 0.50; the two regions have fundamentally different geometry.
    DIVERGENT = "divergent"


# ---------------------------------------------------------------------------
# Dataclasses — structured data containers
# ---------------------------------------------------------------------------

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

    adjacency: np.ndarray   # (n, n) symmetric non-negative weight matrix
    nodes: list[str]        # must satisfy len(nodes) == adjacency.shape[0]
    # stats is populated lazily by graph_utils helpers, not at construction.
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

    start: int    # 0-based index into the sorted eigenvalue array
    end: int      # inclusive; end >= start always
    label: str    # used for display and band-level anomaly reporting
    power: float  # sum of squared GFT coefficients in [start, end]


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

    # Columns are L2-normalised; ordering matches eigenvalues (ascending).
    eigenvectors: np.ndarray  # (n, k)
    eigenvalues: np.ndarray   # (k,) non-decreasing; smallest eigenvalue first
