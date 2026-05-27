"""Builds alpha-shape surface meshes and cluster wireframes for the 3D viewer."""
from __future__ import annotations

import logging
from collections import Counter
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

import numpy as np

from backend.models.schemas import ClusterMeshData, ManifoldViewerData
from backend.services.manifold_projector import ProjectionResult

log = logging.getLogger(__name__)

_MAX_SURFACE_POINTS = 2_000


# ---------------------------------------------------------------------------
# Module-level geometry helpers (not counted in class WMC)
# ---------------------------------------------------------------------------

def _farthest_point_sample(pts: np.ndarray, n_samples: int) -> np.ndarray:
    """Iterative farthest-point sampling: preserves density better than random."""
    rng = np.random.default_rng(0)
    selected = [int(rng.integers(len(pts)))]
    dists = np.full(len(pts), np.inf)

    for _ in range(n_samples - 1):
        d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))

    return np.array(selected, dtype=int)


def _auto_alpha(pts: np.ndarray) -> float:
    """alpha = 2 / mean_k-NN_distance (inversely proportional to local density)."""
    k = min(7, len(pts) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
    dists, _ = nn.kneighbors(pts)
    mean_nn = dists[:, 1:].mean()
    return 2.0 / (mean_nn + 1e-9)


def _circumradius(tet: np.ndarray) -> float:
    """Circumradius of a tetrahedron (4 × 3 array of 3-D vertices)."""
    p0, p1, p2, p3 = tet
    lhs = 2.0 * np.array([p1 - p0, p2 - p0, p3 - p0])
    rhs = np.array([
        np.dot(p1, p1) - np.dot(p0, p0),
        np.dot(p2, p2) - np.dot(p0, p0),
        np.dot(p3, p3) - np.dot(p0, p0),
    ])
    try:
        center = np.linalg.solve(lhs, rhs)
        return float(np.linalg.norm(center - p0))
    except np.linalg.LinAlgError:
        return np.inf


def _extract_boundary_faces(
    simplices: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Faces shared by exactly one tetrahedron in the masked simplex set."""
    face_count: Counter = Counter()
    for s in simplices[mask]:
        for i in range(4):
            face = tuple(sorted(np.delete(s, i).tolist()))
            face_count[face] += 1
    outer = [list(f) for f, cnt in face_count.items() if cnt == 1]
    return np.array(outer, dtype=int) if outer else np.empty((0, 3), dtype=int)


def _delaunay_mesh(
    pts: np.ndarray,
) -> tuple[list[list[float]], list[list[int]]]:
    """Outer faces of a 3-D Delaunay triangulation (= convex-hull surface)."""
    tri = Delaunay(pts)
    mask = np.ones(len(tri.simplices), dtype=bool)
    outer_faces = _extract_boundary_faces(tri.simplices, mask)
    return pts.tolist(), outer_faces.tolist() if len(outer_faces) else []


def _alpha_shape_mesh(
    pts: np.ndarray,
    alpha: float | None = None,
) -> tuple[list[list[float]], list[list[int]]]:
    """Alpha-shape mesh: keep only Delaunay tetrahedra with small circumradii."""
    tri = Delaunay(pts)

    if alpha is None:
        alpha = _auto_alpha(pts)

    threshold = 1.0 / (alpha + 1e-12)
    radii = np.array([_circumradius(pts[s]) for s in tri.simplices])
    mask = radii < threshold

    if not mask.any():
        log.debug("Alpha %.3f filtered all simplices; using Delaunay fallback.", alpha)
        mask = np.ones(len(tri.simplices), dtype=bool)

    outer_faces = _extract_boundary_faces(tri.simplices, mask)
    return pts.tolist(), outer_faces.tolist() if len(outer_faces) else []


def _try_mesh(pts: np.ndarray, alpha: float | None) -> tuple[list, list]:
    """Try alpha-shape or Delaunay mesh; return empty lists on failure."""
    try:
        return _alpha_shape_mesh(pts, alpha) if alpha is not None else _delaunay_mesh(pts)
    except Exception as exc:
        log.warning("Surface mesh failed: %s — returning empty mesh.", exc)
        return [], []


def _try_cluster_mesh(pts: np.ndarray) -> tuple[list, list]:
    """Try Delaunay cluster mesh; return raw points on failure."""
    try:
        return _delaunay_mesh(pts)
    except Exception as exc:
        log.warning("Cluster mesh failed: %s.", exc)
        return pts.tolist(), []


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class MeshBuilder:
    """Convert a ProjectionResult into ManifoldViewerData (serialisable geometry)."""

    def build_viewer_data(
        self,
        proj: ProjectionResult,
        alpha: float | None = None,
    ) -> ManifoldViewerData:
        """Orchestrate projection → geometry → ManifoldViewerData."""
        normal_pos = (
            proj.all_positions[proj.meta.normal_indices]
            if proj.meta.normal_indices
            else np.zeros((0, 3))
        )

        surface_verts, surface_faces = self.build_surface_mesh(normal_pos, alpha=alpha)

        point_positions = proj.all_positions.tolist()
        point_is_anomaly = proj.is_anomaly.tolist()
        point_scores     = proj.scores.tolist()
        point_ids        = proj.point_ids.tolist()

        clusters_data: list[ClusterMeshData] = []
        for cp in proj.clusters:
            verts, faces = self.build_cluster_mesh(cp.positions)
            clusters_data.append(ClusterMeshData(
                cluster_id=cp.cluster_id,
                vertices=verts,
                faces=faces,
                point_indices=cp.indices,
                centroid=(
                    cp.positions.mean(axis=0).tolist() if len(cp.positions) else [0.0, 0.0, 0.0]
                ),
                reprojected=cp.reprojected,
                divergent_axes=cp.divergent_axes,
            ))

        return ManifoldViewerData(
            surface_vertices=surface_verts,
            surface_faces=surface_faces,
            point_positions=point_positions,
            point_is_anomaly=point_is_anomaly,
            point_scores=point_scores,
            point_ids=point_ids,
            clusters=clusters_data,
            n_points=len(proj.all_positions),
            n_anomalies=int(proj.is_anomaly.sum()),
            n_clusters=len(clusters_data),
            projection_axes=proj.meta.axes,
            axis_labels=proj.meta.axis_labels,
        )

    def build_surface_mesh(
        self,
        pts: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Build an alpha-shape or Delaunay surface mesh from *pts*."""
        if len(pts) < 4:
            return [], []
        if len(pts) > _MAX_SURFACE_POINTS:
            pts = pts[_farthest_point_sample(pts, _MAX_SURFACE_POINTS)]
        return _try_mesh(pts, alpha)

    def build_cluster_mesh(
        self,
        pts: np.ndarray,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Build a Delaunay surface mesh for a single anomaly cluster."""
        if len(pts) < 4:
            return pts.tolist(), []
        return _try_cluster_mesh(pts)
