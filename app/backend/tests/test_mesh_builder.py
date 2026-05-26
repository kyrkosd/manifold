"""Tests for backend/services/mesh_builder.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.models.schemas import ManifoldViewerData
from backend.services.manifold_projector import ClusterProjection, ProjectionResult
from backend.services.mesh_builder import MeshBuilder

_MAX_SURFACE_POINTS = 2_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_points(n: int = 500, seed: int = 0) -> np.ndarray:
    """n points uniformly distributed on the unit sphere surface."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _plane_points(n: int = 200) -> np.ndarray:
    """n points in the z=0 plane."""
    rng = np.random.default_rng(1)
    xy = rng.standard_normal((n, 2))
    return np.hstack([xy, np.zeros((n, 1))])


def _minimal_projection(normal_pts: np.ndarray) -> ProjectionResult:
    n = len(normal_pts)
    all_pos = normal_pts
    return ProjectionResult(
        all_positions=all_pos,
        is_anomaly=np.zeros(n, dtype=bool),
        scores=np.zeros(n),
        point_ids=np.arange(n, dtype=np.intp),
        normal_indices=list(range(n)),
        isolated_indices=[],
        clusters=[],
        axes=[0, 1, 2],
        axis_labels=["GFT 0", "GFT 1", "GFT 2"],
    )


@pytest.fixture
def builder():
    return MeshBuilder()


# ---------------------------------------------------------------------------
# Surface mesh
# ---------------------------------------------------------------------------

class TestSurfaceMesh:
    def test_sphere_produces_nonempty_mesh(self, builder):
        pts = _sphere_points(500)
        verts, faces = builder._build_surface_mesh(pts)
        # Sphere is a closed manifold — Delaunay outer-face extraction should give triangles.
        assert len(verts) > 0
        assert len(faces) > 0

    def test_plane_handled_gracefully(self, builder):
        pts = _plane_points(200)
        verts, faces = builder._build_surface_mesh(pts)
        # Coplanar input may give empty mesh; what matters is it doesn't raise.
        assert isinstance(verts, list)
        assert isinstance(faces, list)

    def test_too_few_points_returns_empty(self, builder):
        pts = np.random.default_rng(0).standard_normal((3, 3))
        verts, faces = builder._build_surface_mesh(pts)
        assert faces == []

    def test_large_set_subsampled(self, builder):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((10_000, 3))
        verts, faces = builder._build_surface_mesh(pts)
        assert len(verts) <= _MAX_SURFACE_POINTS + 1

    def test_face_indices_valid(self, builder):
        pts = _sphere_points(300)
        verts, faces = builder._build_surface_mesh(pts)
        if not faces:
            pytest.skip("Mesh is empty — skip index check.")
        n_verts = len(verts)
        for f in faces:
            assert all(0 <= i < n_verts for i in f), f"Invalid face index in {f}"

    def test_no_degenerate_faces(self, builder):
        pts = _sphere_points(300)
        verts, faces = builder._build_surface_mesh(pts)
        for f in faces:
            assert len(set(f)) == 3, f"Degenerate (repeated-vertex) face: {f}"

    def test_empty_input_returns_empty(self, builder):
        verts, faces = builder._build_surface_mesh(np.zeros((0, 3)))
        assert verts == []
        assert faces == []


# ---------------------------------------------------------------------------
# Cluster mesh
# ---------------------------------------------------------------------------

class TestClusterMesh:
    def test_normal_cluster_produces_mesh(self, builder):
        rng = np.random.default_rng(2)
        pts = rng.standard_normal((30, 3))
        verts, faces = builder._build_cluster_mesh(pts)
        assert isinstance(verts, list)
        assert isinstance(faces, list)

    def test_tiny_cluster_no_crash(self, builder):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        verts, faces = builder._build_cluster_mesh(pts)
        assert faces == []

    def test_cluster_face_indices_valid(self, builder):
        rng = np.random.default_rng(3)
        pts = rng.standard_normal((25, 3))
        verts, faces = builder._build_cluster_mesh(pts)
        if not faces:
            return
        n_v = len(verts)
        for f in faces:
            assert all(0 <= i < n_v for i in f)


# ---------------------------------------------------------------------------
# build_viewer_data orchestrator
# ---------------------------------------------------------------------------

class TestBuildViewerData:
    def test_returns_manifold_viewer_data(self, builder):
        proj = _minimal_projection(_sphere_points(200))
        result = builder.build_viewer_data(proj)
        assert isinstance(result, ManifoldViewerData)

    def test_n_points_correct(self, builder):
        pts = _sphere_points(200)
        proj = _minimal_projection(pts)
        result = builder.build_viewer_data(proj)
        assert result.n_points == 200

    def test_point_positions_length(self, builder):
        pts = _sphere_points(200)
        proj = _minimal_projection(pts)
        result = builder.build_viewer_data(proj)
        assert len(result.point_positions) == 200

    def test_zero_anomalies_when_all_normal(self, builder):
        proj = _minimal_projection(_sphere_points(100))
        result = builder.build_viewer_data(proj)
        assert result.n_anomalies == 0
        assert result.n_clusters == 0

    def test_with_cluster_data(self, builder):
        rng = np.random.default_rng(5)
        normal_pts = rng.standard_normal((200, 3))
        cluster_pts = rng.standard_normal((10, 3)) + 5.0

        all_pts = np.vstack([normal_pts, cluster_pts])
        is_anomaly = np.array([False] * 200 + [True] * 10)

        cp = ClusterProjection(
            cluster_id=0,
            indices=list(range(200, 210)),
            positions=cluster_pts,
            reprojected=False,
            divergent_axes=None,
        )
        proj = ProjectionResult(
            all_positions=all_pts,
            is_anomaly=is_anomaly,
            scores=np.where(is_anomaly, 0.9, 0.0),
            point_ids=np.arange(210, dtype=np.intp),
            normal_indices=list(range(200)),
            isolated_indices=[],
            clusters=[cp],
            axes=[0, 1, 2],
            axis_labels=["GFT 0", "GFT 1", "GFT 2"],
        )
        result = builder.build_viewer_data(proj)
        assert result.n_clusters == 1
        assert result.n_anomalies == 10
        assert result.clusters[0].cluster_id == 0

    def test_alpha_parameter_accepted(self, builder):
        proj = _minimal_projection(_sphere_points(200))
        result = builder.build_viewer_data(proj, alpha=5.0)
        assert isinstance(result, ManifoldViewerData)
