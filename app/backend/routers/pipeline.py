"""Pipeline launch and status router for the FMAS import interface."""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from backend.config import RUNS_DIR
from backend.models.schemas import LaunchRequest, LaunchResponse
from backend.services.data_store import DataStore
from backend.services.pipeline_bridge import PipelineBridge
from backend.routers.upload import get_store

log = logging.getLogger(__name__)
router = APIRouter(tags=["pipeline"])

_bridge = PipelineBridge()


def get_bridge() -> PipelineBridge:
    """FastAPI dependency: return the shared PipelineBridge instance."""
    return _bridge


@router.post("/launch", response_model=LaunchResponse)
async def launch_pipeline(
    req: LaunchRequest,
    store: DataStore = Depends(get_store),
    bridge: PipelineBridge = Depends(get_bridge),
) -> LaunchResponse:
    """Validate the dataset, prepare a run directory, and launch the pipeline."""
    # Verify data_id exists before doing anything.
    try:
        meta = store.get_metadata(req.data_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    data = store.get_numeric_array(req.data_id)
    if data.size == 0:
        raise HTTPException(status_code=422, detail="Dataset has no numeric columns to analyze.")

    run_id = uuid.uuid4().hex
    config_dict = req.config.model_dump()
    config_dict["column_names"] = meta.get("column_names")

    run_dir = bridge.prepare_run(data, config_dict, run_id, RUNS_DIR)
    result = bridge.launch_run(run_dir)

    status_label = result["status"]
    if status_label == "running":
        message = "Pipeline is running. Poll /api/runs/{run_id}/status for updates."
    else:
        message = result.get("message", "Data prepared. Install FMAS pipeline to run analysis.")

    return LaunchResponse(
        run_id=run_id,
        data_id=req.data_id,
        config=req.config,
        status=status_label,
        message=message,
        viewer_url=f"/viewer?run={run_id}",
    )


@router.get("/runs/{run_id}/status")
async def get_run_status(
    run_id: str,
    bridge: PipelineBridge = Depends(get_bridge),
) -> dict:
    """Return the current status of a pipeline run."""
    return bridge.get_run_status(run_id, RUNS_DIR)


@router.get("/runs")
async def list_runs() -> list[dict]:
    """Return a list of all run directories and their statuses."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if run_dir.is_dir():
            status_path = run_dir / "status.json"
            if status_path.exists():
                payload = json.loads(status_path.read_text())
            else:
                payload = {"status": "prepared"}
            payload["run_id"] = run_dir.name
            runs.append(payload)
    return runs
