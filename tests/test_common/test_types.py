"""Tests for common/types.py — verifies spec Section 6.1 enum values."""
from __future__ import annotations

import numpy as np
import pytest

from common.types import (
    AlignmentQuality,
    AnomalyType,
    BoundaryType,
    EigenBasis,
    FeatureGraph,
    FourierType,
    FrequencyBand,
    RelationshipType,
    Severity,
    SpatialType,
    StructureType,
)


# --- StructureType ---


def test_structure_type_values() -> None:
    """Structure type values."""
    assert {e.value for e in StructureType} == {"linear", "grid", "graph", "unknown"}


def test_structure_type_members() -> None:
    """Structure type members."""
    assert StructureType.LINEAR.value == "linear"
    assert StructureType.GRAPH.value == "graph"
    assert StructureType.UNKNOWN.value == "unknown"


# --- FourierType ---


def test_fourier_type_values() -> None:
    """Fourier type values."""
    assert {e.value for e in FourierType} == {"fft", "multidim_fft", "graph_fourier"}


def test_fourier_type_graph_fourier() -> None:
    """Fourier type graph fourier."""
    assert FourierType.GRAPH_FOURIER.value == "graph_fourier"


def test_fourier_type_standard_fft() -> None:
    """Fourier type standard fft."""
    assert FourierType.STANDARD_FFT.value == "fft"


# --- AnomalyType ---


def test_anomaly_type_values() -> None:
    """Anomaly type values."""
    assert {e.value for e in AnomalyType} == {"spectral", "partial", "global", "normal"}


def test_anomaly_type_normal_exists() -> None:
    """Anomaly type normal exists."""
    assert AnomalyType.NORMAL.value == "normal"


def test_anomaly_type_spectral_exists() -> None:
    """Anomaly type spectral exists."""
    assert AnomalyType.SPECTRAL.value == "spectral"


# --- SpatialType ---


def test_spatial_type_has_two_members() -> None:
    """Spatial type has two members."""
    assert len(list(SpatialType)) == 2


def test_spatial_type_values() -> None:
    """Spatial type values."""
    assert {e.value for e in SpatialType} == {"isolated", "regional"}


# --- RelationshipType ---


def test_relationship_type_values() -> None:
    """Relationship type values."""
    assert {e.value for e in RelationshipType} == {
        "on_manifold",
        "separate_manifold",
        "scattered_points",
    }


def test_relationship_type_submanifold() -> None:
    """Relationship type submanifold."""
    assert RelationshipType.SUBMANIFOLD.value == "on_manifold"


# --- Severity ---


def test_severity_three_levels() -> None:
    """Severity three levels."""
    assert {e.value for e in Severity} == {"mild", "moderate", "severe"}


def test_severity_mild_exists() -> None:
    """Severity mild exists."""
    assert Severity.MILD.value == "mild"


# --- BoundaryType ---


def test_boundary_type_three_members() -> None:
    """Boundary type three members."""
    assert len(list(BoundaryType)) == 3


def test_boundary_type_values() -> None:
    """Boundary type values."""
    assert {e.value for e in BoundaryType} == {"sharp", "gradual", "diffuse"}


# --- AlignmentQuality ---


def test_alignment_quality_five_members() -> None:
    """Alignment quality five members."""
    assert len(list(AlignmentQuality)) == 5


def test_alignment_quality_values() -> None:
    """Alignment quality values."""
    assert {e.value for e in AlignmentQuality} == {
        "excellent", "good", "acceptable", "poor", "divergent"
    }


def test_alignment_quality_divergent_exists() -> None:
    """Alignment quality divergent exists."""
    assert AlignmentQuality.DIVERGENT.value == "divergent"


# --- FeatureGraph ---


def test_feature_graph_basic_creation() -> None:
    """Feature graph basic creation."""
    adj = np.eye(3)
    fg = FeatureGraph(adjacency=adj, nodes=["a", "b", "c"])
    assert fg.adjacency.shape == (3, 3)
    assert fg.nodes == ["a", "b", "c"]
    assert fg.stats == {}


def test_feature_graph_stats_populated() -> None:
    """Feature graph stats populated."""
    adj = np.ones((2, 2))
    fg = FeatureGraph(adjacency=adj, nodes=["x", "y"], stats={"density": 1.0})
    assert fg.stats["density"] == pytest.approx(1.0)


def test_feature_graph_empty_nodes() -> None:
    """Feature graph empty nodes."""
    fg = FeatureGraph(adjacency=np.zeros((0, 0)), nodes=[])
    assert fg.nodes == []
    assert fg.adjacency.size == 0


# --- FrequencyBand ---


def test_frequency_band_fields() -> None:
    """Frequency band fields."""
    band = FrequencyBand(start=0, end=4, label="low", power=0.8)
    assert band.start == 0
    assert band.end == 4
    assert band.label == "low"
    assert band.power == pytest.approx(0.8)


def test_frequency_band_zero_power() -> None:
    """Frequency band zero power."""
    band = FrequencyBand(start=10, end=20, label="high", power=0.0)
    assert band.power == pytest.approx(0.0)


# --- EigenBasis ---


def test_eigen_basis_shapes() -> None:
    """Eigen basis shapes."""
    vecs = np.random.default_rng(0).standard_normal((10, 3))
    vals = np.array([0.1, 0.5, 1.2])
    eb = EigenBasis(eigenvectors=vecs, eigenvalues=vals)
    assert eb.eigenvectors.shape == (10, 3)
    assert eb.eigenvalues.shape == (3,)


def test_eigen_basis_empty() -> None:
    """Eigen basis empty."""
    eb = EigenBasis(eigenvectors=np.empty((0, 0)), eigenvalues=np.empty(0))
    assert eb.eigenvalues.size == 0


def test_eigen_basis_single_component() -> None:
    """Eigen basis single component."""
    eb = EigenBasis(eigenvectors=np.array([[1.0], [0.0]]), eigenvalues=np.array([0.0]))
    assert eb.eigenvectors.shape == (2, 1)
