"""Tests for manifold/atlas.py.

Fixture: n=120 points on a 2-D linear subspace embedded in 5-D ambient space.
Feature correlations are strong (all features share two latent factors), so
the local GFT captures the manifold structure well and charts pass validation.
"""
from __future__ import annotations

import numpy as np

from common.types import FourierType, StructureType
from fourier import SpectralData, analyze_fourier
from manifold.atlas import (
    Atlas,
    _assign_primary_charts,
    _find_overlaps,
    _partition_into_regions,
    _verify_coverage,
    build,
)
from structure.report import StructureReport
import structure.graph_builder as graph_builder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_embedded_plane(n: int = 120, d: int = 5, intrinsic: int = 2,
                          seed: int = 0) -> np.ndarray:
    """Return n points near a 2-D subspace in d-D ambient space."""
    rng = np.random.default_rng(seed)
    orth_mat, _ = np.linalg.qr(rng.standard_normal((d, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coords = rng.standard_normal((n, intrinsic))
    return coords @ basis.T + rng.standard_normal((n, d)) * 0.03


def _make_structure(data: np.ndarray, intrinsic_dim: int = 2) -> StructureReport:
    fg = graph_builder.build(data)
    return StructureReport(
        type=StructureType.GRAPH,
        intrinsic_dim=intrinsic_dim,
        graph=fg,
        ordering=None,
        periodicity=False,
        recommended_fourier=FourierType.GRAPH_FOURIER,
        confidence=1.0,
    )


def _make_spectral(data: np.ndarray, structure: StructureReport) -> SpectralData:
    return analyze_fourier(data, structure)


def _make_atlas_fixture(n: int = 120, d: int = 5, n_charts: int = 3, seed: int = 0):
    data = _make_embedded_plane(n=n, d=d, seed=seed)
    structure = _make_structure(data)
    spectral = _make_spectral(data, structure)
    config = {"n_charts": n_charts, "overlap_factor": 0.2, "max_retries": 3}
    atlas = build(data, spectral, structure, config)
    return atlas, data, structure, spectral


# ---------------------------------------------------------------------------
# build() — return type and top-level properties
# ---------------------------------------------------------------------------

class TestAtlasBuild:
    """Tests for Atlas Build."""
    def test_returns_atlas_instance(self):
        """Returns atlas instance."""
        atlas, *_ = _make_atlas_fixture()
        assert isinstance(atlas, Atlas)

    def test_n_points_correct(self):
        """N points correct."""
        atlas, data, *_ = _make_atlas_fixture()
        assert atlas.n_points == len(data)

    def test_has_at_least_one_chart(self):
        """Has at least one chart."""
        atlas, *_ = _make_atlas_fixture()
        assert len(atlas.charts) >= 1

    def test_coverage_pct_positive(self):
        """Coverage pct positive."""
        atlas, *_ = _make_atlas_fixture()
        assert atlas.coverage_pct > 0.0

    def test_coverage_pct_in_unit_interval(self):
        """Coverage pct in unit interval."""
        atlas, *_ = _make_atlas_fixture()
        assert 0.0 <= atlas.coverage_pct <= 1.0

    def test_primary_assignments_shape(self):
        """Primary assignments shape."""
        atlas, data, *_ = _make_atlas_fixture()
        assert atlas.primary_assignments.shape == (len(data),)

    def test_primary_assignments_in_bounds(self):
        """Primary assignments in bounds."""
        atlas, *_ = _make_atlas_fixture()
        n_charts = len(atlas.charts)
        assert int(atlas.primary_assignments.min()) >= 0
        assert int(atlas.primary_assignments.max()) < n_charts

    def test_faiss_index_stored(self):
        """Faiss index stored."""
        atlas, *_ = _make_atlas_fixture()
        assert atlas.faiss_index is not None

    def test_overlaps_is_list(self):
        """Overlaps is list."""
        atlas, *_ = _make_atlas_fixture()
        assert isinstance(atlas.overlaps, list)


# ---------------------------------------------------------------------------
# _partition_into_regions
# ---------------------------------------------------------------------------

class TestPartitionIntoRegions:
    """Tests for Partition Into Regions."""
    def test_returns_list_of_arrays(self):
        """Returns list of arrays."""
        data = _make_embedded_plane()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        regions = _partition_into_regions(data, spectral, structure, n_charts=4)
        assert isinstance(regions, list)
        assert all(isinstance(r, np.ndarray) for r in regions)

    def test_all_points_covered(self):
        """All points covered."""
        data = _make_embedded_plane()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        regions = _partition_into_regions(data, spectral, structure, n_charts=4)
        covered = set()
        for r in regions:
            covered.update(r.tolist())
        assert covered == set(range(len(data)))

    def test_n_regions_matches_request(self):
        """N regions matches request."""
        data = _make_embedded_plane()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        regions = _partition_into_regions(data, spectral, structure, n_charts=3)
        assert len(regions) == 3

    def test_auto_mode_gives_at_least_two(self):
        """Auto mode gives at least two."""
        data = _make_embedded_plane(n=200)
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        regions = _partition_into_regions(data, spectral, structure, n_charts="auto")
        assert len(regions) >= 2


# ---------------------------------------------------------------------------
# _verify_coverage
# ---------------------------------------------------------------------------

class TestVerifyCoverage:
    """Tests for Verify Coverage."""
    def test_full_coverage_returns_true(self):
        """Full coverage returns true."""
        # Build minimal stub charts that together cover points 0..9.
        def _stub_chart(indices):
            return type("C", (), {"region_indices": np.array(indices)})()

        charts = [_stub_chart([0, 1, 2, 3]), _stub_chart([4, 5, 6, 7, 8, 9])]
        assert _verify_coverage(charts, n_points=10) is True

    def test_missing_point_returns_false(self):
        """Missing point returns false."""
        def _stub_chart(indices):
            return type("C", (), {"region_indices": np.array(indices)})()

        charts = [_stub_chart([0, 1, 2])]   # point 3 is missing
        assert _verify_coverage(charts, n_points=4) is False

    def test_empty_charts_returns_false(self):
        """Empty charts returns false."""
        assert _verify_coverage([], n_points=5) is False


# ---------------------------------------------------------------------------
# _assign_primary_charts
# ---------------------------------------------------------------------------

class TestAssignPrimaryCharts:
    """Tests for Assign Primary Charts."""
    def test_shape_is_n_points(self):
        """Shape is n points."""
        def _stub(indices, score):
            return type("C", (), {
                "region_indices": np.array(indices),
                "local_basis": type("B", (), {"alignment_score": score})(),
            })()

        charts = [_stub([0, 1, 2], 0.9), _stub([1, 2, 3], 0.5)]
        result = _assign_primary_charts(charts, n_points=4)
        assert result.shape == (4,)

    def test_best_score_wins(self):
        """Best score wins."""
        # Point 1 is in both charts; chart-0 has higher score → assigned to 0.
        def _stub(indices, score):
            return type("C", (), {
                "region_indices": np.array(indices),
                "local_basis": type("B", (), {"alignment_score": score})(),
            })()

        charts = [_stub([0, 1], 0.95), _stub([1, 2], 0.50)]
        result = _assign_primary_charts(charts, n_points=3)
        assert result[1] == 0   # shared point 1 → chart 0 (higher score)

    def test_exclusive_point_gets_only_chart(self):
        """Exclusive point gets only chart."""
        def _stub(indices, score):
            return type("C", (), {
                "region_indices": np.array(indices),
                "local_basis": type("B", (), {"alignment_score": score})(),
            })()

        charts = [_stub([0], 0.8), _stub([1], 0.7)]
        result = _assign_primary_charts(charts, n_points=2)
        assert result[0] == 0
        assert result[1] == 1


# ---------------------------------------------------------------------------
# _find_overlaps
# ---------------------------------------------------------------------------

class TestFindOverlaps:
    """Tests for Find Overlaps."""
    def test_disjoint_charts_give_empty_overlaps(self):
        """Disjoint charts give empty overlaps."""
        def _stub(indices):
            return type("C", (), {"region_indices": np.array(indices)})()

        charts = [_stub([0, 1, 2]), _stub([3, 4, 5])]
        assert _find_overlaps(charts) == []

    def test_shared_indices_correct(self):
        """Shared indices correct."""
        def _stub(indices):
            return type("C", (), {"region_indices": np.array(indices)})()

        charts = [_stub([0, 1, 2]), _stub([2, 3, 4])]
        overlaps = _find_overlaps(charts)
        assert len(overlaps) == 1
        i, j, shared = overlaps[0]
        assert i == 0 and j == 1
        assert set(shared.tolist()) == {2}

    def test_triple_overlap_gives_three_pairs(self):
        """Triple overlap gives three pairs."""
        def _stub(indices):
            return type("C", (), {"region_indices": np.array(indices)})()

        charts = [_stub([0, 1, 2]), _stub([1, 2, 3]), _stub([2, 3, 4])]
        overlaps = _find_overlaps(charts)
        assert len(overlaps) == 3
