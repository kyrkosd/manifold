"""File upload router for the FMAS import interface."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile
##from fastapi.responses import JSONResponse

from backend.config import DATA_DIR, MAX_UPLOAD_BYTES
from backend.models.enums import DataStatus
from backend.models.schemas import PreviewResponse
from backend.services import data_profiler as profiler
from backend.services.data_store import DataStore
from backend.services.file_parser import FileParseError, parse_file, save_upload

log = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])


def get_store() -> DataStore:
    """FastAPI dependency: return the shared DataStore instance."""
    return _store


_store = DataStore(data_dir=DATA_DIR)


@router.post("/upload", response_model=PreviewResponse)
async def upload_file(
    file: UploadFile,
    store: DataStore = Depends(get_store),
) -> PreviewResponse:
    """Upload a file, parse it, profile it, and return a preview response."""
    t0 = time.monotonic()

    # Size guard (read Content-Length header if available; else check after read).
    if file.size and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    saved_path = await save_upload(file, DATA_DIR)
    if saved_path.stat().st_size > MAX_UPLOAD_BYTES:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    try:
        df = parse_file(saved_path)
    except FileParseError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    quality = profiler.profile(df)
    columns = profiler.build_column_info(df)
    preview_rows = profiler.build_preview_rows(df)

    data_id = store.store(df, {
        "source_type": "file",
        "file_name": file.filename,
        "status": DataStatus.READY,
    })

    log.info(
    "Upload %r processed in %.2fs → data_id=%s",
    file.filename,
    time.monotonic() - t0,
    data_id,
)

    return PreviewResponse(
        data_id=data_id,
        file_name=file.filename,
        source_type="file",
        columns=columns,
        preview_rows=preview_rows,
        quality=quality,
        status=DataStatus.READY,
    )


@router.get("/data/{data_id}/preview", response_model=PreviewResponse)
async def get_preview(
    data_id: str,
    store: DataStore = Depends(get_store),
) -> PreviewResponse:
    """Return a cached preview for an already-stored dataset."""
    try:
        df, meta = store.retrieve(data_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    quality = profiler.profile(df)
    columns = profiler.build_column_info(df)
    preview_rows = profiler.build_preview_rows(df)

    return PreviewResponse(
        data_id=data_id,
        file_name=meta.get("file_name"),
        source_type=meta.get("source_type", "file"),
        columns=columns,
        preview_rows=preview_rows,
        quality=quality,
        status=meta.get("status", DataStatus.READY),
    )
