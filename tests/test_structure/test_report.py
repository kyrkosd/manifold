"""Tests for structure/report.py.

Verifies the StructureReport dataclass contracts and the build() factory.
For the MVP, build() always returns GRAPH type and GRAPH_FOURIER engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import FeatureGraph, FourierType, StructureType
from structure.report import StructureReport, build
from structure import discover_structure


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_graph(d: int = 4) -> FeatureGraph:
    """Return a simple complete-graph FeatureGraph with d nodes."""
    adj = np.ones((d, d)) - np.eye(d)   # complete graph: all off-diagonal = 1
    nodes = [str(i) for i in range(d)]
    return FeatureGraph(adjacency=adj, nodes=nodes, stats={"n_edges": d * (d - 1) // 2})


# ---------------------------------------------------------------------------
# StructureReport dataclass
# ---------------------------------------------------------------------------

class TestStructureReport:
    """Tests for Structure Report."""
    def test_construction_stores_all_fields(self):
        """Construction stores all fields."""
        # Every field must be stored exactly as provided.
        graph = _make_graph()
        report = StructureReport(
            type=StructureType.GRAPH,
            intrinsic_dim=3,
            graph=graph,
            ordering=None,
            periodicity=False,
            recommended_fourier=FourierType.GRAPH_FOURIER,
            confidence=0.9,
        )
        assert report.type is StructureType.GRAPH
        assert report.intrinsic_dim == 3
        assert report.graph is graph       # identity: no copy made
        assert report.ordering is None
        assert report.periodicity is False
        assert report.recommended_fourier is FourierType.GRAPH_FOURIER
        assert report.confidence == pytest.approx(0.9)

    def test_graph_field_accepts_none(self):
        """Graph field accepts none."""
        # Non-GRAPH structure types have no feature graph.
        report = StructureReport(
            type=StructureType.LINEAR,
            intrinsic_dim=1,
            graph=None,
            ordering=None,
            periodicity=False,
            recommended_fourier=FourierType.STANDARD_FFT,
            confidence=1.0,
        )
        assert report.graph is None


# ---------------------------------------------------------------------------
# build() — MVP factory
# ---------------------------------------------------------------------------

class TestBuild:
    """Tests for Build."""
    def test_type_is_always_graph(self):
        """Type is always graph."""
        # MVP always detects GRAPH structure.
        report = build(_make_graph(), dim_result=3)
        assert report.type is StructureType.GRAPH

    def test_fourier_is_always_graph_fourier(self):
        """Fourier is always graph fourier."""
        # GRAPH_FOURIER is the only supported Fourier engine for tabular data.
        report = build(_make_graph(), dim_result=3)
        assert report.recommended_fourier is FourierType.GRAPH_FOURIER

    def test_ordering_is_none(self):
        """Ordering is none."""
        # Ordering analysis is not implemented in the MVP.
        report = build(_make_graph(), dim_result=3)
        assert report.ordering is None

    def test_periodicity_is_false(self):
        """Periodicity is false."""
        # Periodicity detection is not implemented in the MVP.
        report = build(_make_graph(), dim_result=3)
        assert report.periodicity is False

    def test_confidence_is_one(self):
        """Confidence is one."""
        # The deterministic MVP path always assigns maximum confidence.
        report = build(_make_graph(), dim_result=3)
        assert report.confidence == pytest.approx(1.0)

    def test_intrinsic_dim_is_stored(self):
        """Intrinsic dim is stored."""
        # The dim_result argument must be preserved verbatim.
        report = build(_make_graph(), dim_result=7)
        assert report.intrinsic_dim == 7

    def test_graph_is_stored_by_reference(self):
        """Graph is stored by reference."""
        # The graph must not be copied.
        graph = _make_graph()
        report = build(graph, dim_result=2)
        assert report.graph is graph


# ---------------------------------------------------------------------------
# discover_structure() — end-to-end integration smoke test
# ---------------------------------------------------------------------------

class TestDiscoverStructure:
    """Tests for Discover Structure."""
    def test_returns_structure_report(self):
        """Returns structure report."""
        # End-to-end: discover_structure must return a StructureReport.
        rng = np.random.default_rng(99)
        data = rng.standard_normal((60, 6))
        report = discover_structure(data)
        assert isinstance(report, StructureReport)

    def test_type_is_graph(self):
        """Type is graph."""
        # MVP always reports GRAPH regardless of data shape.
        data = np.random.default_rng(0).standard_normal((60, 5))
        report = discover_structure(data)
        assert report.type is StructureType.GRAPH

    def test_intrinsic_dim_is_positive(self):
        """Intrinsic dim is positive."""
        # Intrinsic dimension must be ≥ 1 for any non-trivial dataset.
        data = np.random.default_rng(1).standard_normal((80, 6))
        report = discover_structure(data)
        assert report.intrinsic_dim >= 1

    def test_graph_is_not_none(self):
        """Graph is not none."""
        # The feature graph must always be built in the MVP path.
        data = np.random.default_rng(2).standard_normal((60, 5))
        report = discover_structure(data)
        assert report.graph is not None

    def test_recommended_fourier_is_graph_fourier(self):
        """Recommended fourier is graph fourier."""
        data = np.random.default_rng(3).standard_normal((60, 5))
        report = discover_structure(data)
        assert report.recommended_fourier is FourierType.GRAPH_FOURIER
