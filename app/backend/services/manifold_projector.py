"""Projects FMAS pipeline output to 3D coordinates for the manifold viewer."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from fastapi import HTTPException

import numpy as np

from backend.models.schemas import PointDetailResponse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate result types
# ---------------------------------------------------------------------------

@dataclass
class ClusterProjection:
    """3D positions and metadata for a single anomaly cluster."""

    cluster_id: int
    indices: list[int]
    positions: np.ndarray          # (n_cluster, 3)
    reprojected: bool
    divergent_axes: Optional[list[int]] = None


@dataclass
class ProjectionMeta:
    """Projection axes, labels, and point partitioning grouped for a ManifoldProjector result."""

    axes: list[int]
    axis_labels: list[str]
    normal_indices: list[int]
    isolated_indices: list[int]


@dataclass
class ProjectionResult:
    """Complete 3D projection output produced by ManifoldProjector."""

    all_positions: np.ndarray       # (n_points, 3)
    is_anomaly: np.ndarray          # (n_points,) bool
    scores: np.ndarray              # (n_points,) float
    point_ids: np.ndarray           # (n_points,) int
    meta: ProjectionMeta
    clusters: list[ClusterProjection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class ManifoldProjector:
    """Load FMAS run outputs and compute 3-D projections."""

    def project(self, run_dir: Path) -> ProjectionResult:
        """Full projection pipeline. Expects run_dir to contain FMAS outputs."""
        coeff = self._load_coefficients(run_dir)
        power = self._load_or_compute_power(run_dir, coeff)
        anomaly = self._load_or_default_anomaly(run_dir, len(coeff))
        cluster_labels = self._load_cluster_labels(run_dir)

        axes = self.select_top3_axes(power)
        positions_all = self._project_to_3d(coeff, axes)

        flags = np.array(anomaly.get("flags", [False] * len(coeff)), dtype=bool)
        scores = np.array(anomaly.get("scores", [0.0] * len(coeff)), dtype=float)

        normal_idx, isolated_idx, regional = self._classify_points(flags, cluster_labels)
        clusters = self._build_clusters(regional, positions_all, normal_idx, coeff)

        axis_labels = [f"GFT coeff {a} (λ={power[a]:.3f})" for a in axes]

        return ProjectionResult(
            all_positions=positions_all,
            is_anomaly=flags,
            scores=scores,
            point_ids=np.arange(len(coeff), dtype=np.intp),
            meta=ProjectionMeta(
                axes=axes,
                axis_labels=axis_labels,
                normal_indices=normal_idx,
                isolated_indices=isolated_idx,
            ),
            clusters=clusters,
        )

    def point_detail(self, run_dir: Path, index: int) -> PointDetailResponse:
        """Return per-point anomaly detail for the viewer side-panel."""
        coeff = self._load_coefficients(run_dir)
        if index < 0 or index >= len(coeff):
            raise HTTPException(status_code=404, detail=f"Point index {index} out of range.")

        anomaly = self._load_or_default_anomaly(run_dir, len(coeff))
        manifold = self._load_or_default_manifold(run_dir, len(coeff))
        col_names = self._load_column_names(run_dir)

        score  = float(anomaly.get("scores", [0.0] * len(coeff))[index])
        flag   = bool(anomaly.get("flags",   [False] * len(coeff))[index])
        atype  = anomaly.get("types", ["normal"] * len(coeff))[index]
        band_s = {k: float(v[index]) for k, v in anomaly.get("per_point_band_scores", {}).items()}
        top_band = max(band_s, key=lambda k: abs(band_s[k])) if band_s else None

        chart_id, chart_quality = self._get_chart_info(manifold, index)

        row = coeff[index, :20]
        if col_names:
            orig = {col_names[i]: float(row[i]) for i in range(min(len(col_names), len(row)))}
        else:
            orig = {f"feature_{i}": float(v) for i, v in enumerate(row)}

        return PointDetailResponse(
            index=index,
            overall_score=score,
            is_anomaly=flag,
            anomaly_type=atype if flag else None,
            band_scores=band_s,
            top_anomalous_band=top_band if flag else None,
            chart_id=chart_id,
            chart_alignment_quality=chart_quality,
            cluster_id=None,
            original_values=orig,
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_coefficients(self, run_dir: Path) -> np.ndarray:
        path = run_dir / "coefficients.npy"
        if not path.exists():
            # Fallback: generate random projection for demo purposes.
            log.warning(
                "coefficients.npy missing; using random 3-D positions for %s.",
                run_dir.name,
            )
            return np.random.default_rng(0).standard_normal((50, 10))
        return np.load(path)

    def _load_power_spectrum(self, run_dir: Path) -> np.ndarray | None:
        """Load precomputed power spectrum; returns None if absent."""
        path = run_dir / "power_spectrum.npy"
        if path.exists():
            return np.load(path)
        return None

    def _load_anomaly(self, run_dir: Path) -> dict | None:
        """Load anomaly results from JSON; returns None if absent."""
        path = run_dir / "anomaly_results.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _load_manifold(self, run_dir: Path) -> dict | None:
        """Load manifold info from JSON; returns None if absent."""
        path = run_dir / "manifold_info.json"
        if path.exists():
            return json.loads(path.read_text())
        return None


    def _load_cluster_labels(self, run_dir: Path) -> np.ndarray | None:
        path = run_dir / "cluster_labels.npy"
        if path.exists():
            return np.load(path)
        return None

    def _load_column_names(self, run_dir: Path) -> list[str] | None:
        path = run_dir / "column_names.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _load_or_compute_power(self, run_dir: Path, coeff: np.ndarray) -> np.ndarray:
        """Return power spectrum from file, or compute it from coefficients if absent."""
        power = self._load_power_spectrum(run_dir)
        return power if power is not None else np.mean(np.abs(coeff) ** 2, axis=0)

    def _load_or_default_anomaly(self, run_dir: Path, n: int) -> dict:
        """Return anomaly results from file, or an all-normal default dict if absent."""
        result = self._load_anomaly(run_dir)
        return result if result is not None else {
            "flags": [False] * n, "scores": [0.0] * n, "types": ["normal"] * n,
        }

    def _load_or_default_manifold(self, run_dir: Path, n: int) -> dict:
        """Return manifold info from file, or a single-chart default dict if absent."""
        result = self._load_manifold(run_dir)
        return result if result is not None else {
            "n_charts": 1, "chart_assignments": [0] * n,
            "intrinsic_dim": 2, "alignment_qualities": [1.0],
        }

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def select_top3_axes(self, power: np.ndarray) -> list[int]:
        """Return the 3 highest-power non-DC frequency indices, sorted ascending."""
        # Skip index 0 (DC component — constant offset, uninformative for shape).
        start = 1 if len(power) > 3 else 0
        idx = np.argsort(power[start:])[::-1][:3] + start
        return sorted(idx.tolist())

    def _project_to_3d(self, coeff: np.ndarray, axes: list[int]) -> np.ndarray:
        pts = coeff[:, axes].astype(float)
        # Zero-mean, unit-variance per axis for balanced rendering.
        mu = pts.mean(axis=0)
        sd = pts.std(axis=0)
        sd[sd == 0] = 1.0
        return (pts - mu) / sd

    def _classify_points(
        self,
        flags: np.ndarray,
        cluster_labels: np.ndarray | None,
    ) -> tuple[list[int], list[int], dict[int, list[int]]]:
        """Partition point indices into normal, isolated anomaly, and regional cluster groups."""
        normal   = [int(i) for i in np.where(~flags)[0]]
        anomalous = np.where(flags)[0]

        if cluster_labels is None:
            return normal, [int(i) for i in anomalous], {}

        regional: dict[int, list[int]] = {}
        isolated: list[int] = []
        for i in anomalous:
            lbl = int(cluster_labels[i])
            if lbl >= 0:
                regional.setdefault(lbl, []).append(int(i))
            else:
                isolated.append(int(i))
        return normal, isolated, regional

    def _build_clusters(
        self,
        regional: dict[int, list[int]],
        positions_all: np.ndarray,
        normal_idx: list[int],
        coeff: np.ndarray,
    ) -> list[ClusterProjection]:
        """Build ClusterProjection list, reprojecting clusters that overlap the normal surface."""
        normal_pos = positions_all[normal_idx] if normal_idx else np.zeros((0, 3))
        clusters: list[ClusterProjection] = []
        for cid, cidx in regional.items():
            cluster_pos = positions_all[cidx]
            reprojected = False
            div_axes = None

            if (
                len(normal_idx) >= 5
                and self._check_cluster_overlap(cluster_pos, normal_pos)
            ):
                div_axes = self._find_divergent_axes(cidx, normal_idx, coeff)
                cluster_pos = self._reproject_cluster(
                    cidx,
                    coeff,
                    div_axes,
                    positions_all.mean(axis=0),
                )
                reprojected = True

            clusters.append(ClusterProjection(
                cluster_id=cid,
                indices=cidx,
                positions=cluster_pos,
                reprojected=reprojected,
                divergent_axes=div_axes,
            ))
        return clusters

    def _get_chart_info(self, manifold: dict, index: int) -> tuple[int, float]:
        """Return (chart_id, chart_alignment_quality) for the given point index."""
        assignments = manifold.get("chart_assignments", [])
        chart_id = int(assignments[index]) if index < len(assignments) else 0
        qualities = manifold.get("alignment_qualities", [])
        chart_quality = float(qualities[chart_id]) if chart_id < len(qualities) else 0.0
        return chart_id, chart_quality

    def _check_cluster_overlap(
        self,
        cluster_pts: np.ndarray,
        normal_pts: np.ndarray,
        threshold: float = 1.5,
    ) -> bool:
        if len(normal_pts) < 2 or len(cluster_pts) == 0:
            return False
        centroid = cluster_pts.mean(axis=0)
        dists = np.linalg.norm(normal_pts - centroid, axis=1)
        k = min(20, len(normal_pts))
        mean_nn_dist = np.sort(dists)[:k].mean()
        mean_inter = np.linalg.norm(normal_pts - normal_pts.mean(axis=0), axis=1).mean()
        return bool(mean_nn_dist < threshold * (mean_inter + 1e-9))

    def _find_divergent_axes(
        self,
        cluster_idx: list[int],
        normal_idx: list[int],
        coeff: np.ndarray,
    ) -> list[int]:
        cluster_power = np.mean(np.abs(coeff[cluster_idx]) ** 2, axis=0)
        normal_power  = np.mean(np.abs(coeff[normal_idx])  ** 2, axis=0)
        diff = np.abs(cluster_power - normal_power)
        return sorted(np.argsort(diff)[::-1][:3].tolist())

    def _reproject_cluster(
        self,
        cluster_idx: list[int],
        coeff: np.ndarray,
        div_axes: list[int],
        main_centroid: np.ndarray,
    ) -> np.ndarray:
        pts = coeff[cluster_idx][:, div_axes].astype(float)
        mu = pts.mean(axis=0)
        sd = pts.std(axis=0)
        sd[sd == 0] = 1.0
        pts_norm = (pts - mu) / sd

        # Shift so the cluster floats visibly away from the main surface.
        cluster_centroid = pts_norm.mean(axis=0)
        direction = cluster_centroid - main_centroid[:3]
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction /= norm
        offset = main_centroid[:3] + 2.5 * direction
        return pts_norm - cluster_centroid + offset
