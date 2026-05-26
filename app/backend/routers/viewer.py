"""3D manifold viewer API router."""
from __future__ import annotations

import numpy as np
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from backend.config import RUNS_DIR
from backend.models.schemas import ManifoldViewerData, PointDetailResponse
from backend.services.manifold_projector import ManifoldProjector
from backend.services.mesh_builder import MeshBuilder

log = logging.getLogger(__name__)
router = APIRouter(tags=["viewer"])

_projector    = ManifoldProjector()
_mesh_builder = MeshBuilder()
# In-memory cache keyed by run_id; invalidated on reproject.
_viewer_cache: dict[str, ManifoldViewerData] = {}


def _run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return d


def _check_complete(run_dir: Path) -> None:
    """Raise 425/202 if the run is not yet complete."""
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return  # prepared but never started — allow fallback rendering
    payload = json.loads(status_path.read_text())
    status = payload.get("status", "")
    if status == "running":
        raise HTTPException(
    status_code=202,
    detail="Pipeline still running. Try again when complete.",
)

    if status not in ("completed", ""):
        raise HTTPException(status_code=425, detail=f"Run is not complete yet (status={status!r}).")


@router.get("/viewer/{run_id}/data", response_model=ManifoldViewerData)
async def get_viewer_data(run_id: str) -> ManifoldViewerData:
    """Return all 3D geometry for the manifold viewer (cached per run)."""
    run_dir = _run_dir(run_id)
    _check_complete(run_dir)

    if run_id in _viewer_cache:
        return _viewer_cache[run_id]

    try:
        projection = _projector.project(run_dir)
        data = _mesh_builder.build_viewer_data(projection)
    except Exception as exc:
        log.error("Viewer data build failed for run %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build viewer data: {exc}") from exc

    _viewer_cache[run_id] = data
    return data


@router.get("/viewer/{run_id}/point/{point_index}", response_model=PointDetailResponse)
async def get_point_detail(run_id: str, point_index: int) -> PointDetailResponse:
    """Return per-point anomaly detail (fast — no mesh rebuild)."""
    run_dir = _run_dir(run_id)
    return _projector.point_detail(run_dir, point_index)


@router.get("/viewer/{run_id}/cluster/{cluster_id}")
async def get_cluster_detail(run_id: str, cluster_id: int) -> dict:
    """Return summary info about a specific anomaly cluster."""
    
    run_dir = _run_dir(run_id)

    labels_path = run_dir / "cluster_labels.npy"
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="No cluster data for this run.")

    labels = np.load(labels_path)
    cluster_mask = labels == cluster_id
    if not cluster_mask.any():
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")

    anomaly_path = run_dir / "anomaly_results.json"
    scores = []
    if anomaly_path.exists():
        anomaly = json.loads(anomaly_path.read_text())
        all_scores = anomaly.get("scores", [])
        scores = [all_scores[i] for i in np.where(cluster_mask)[0] if i < len(all_scores)]

    return {
        "cluster_id": cluster_id,
        "size": int(cluster_mask.sum()),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "max_score": float(np.max(scores)) if scores else 0.0,
    }


@router.post("/viewer/{run_id}/reproject", response_model=ManifoldViewerData)
async def reproject(
    run_id: str,
    body: dict = Body(default={}),
) -> ManifoldViewerData:
    """Rebuild the surface mesh with a different alpha value; invalidates cache."""
    run_dir = _run_dir(run_id)
    _check_complete(run_dir)

    alpha = body.get("alpha", None)
    if alpha is not None:
        try:
            alpha = float(alpha)
            if alpha <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="alpha must be a positive float.")

    _viewer_cache.pop(run_id, None)

    try:
        projection = _projector.project(run_dir)
        data = _mesh_builder.build_viewer_data(projection, alpha=alpha)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reprojection failed: {exc}") from exc

    _viewer_cache[run_id] = data
    return data
