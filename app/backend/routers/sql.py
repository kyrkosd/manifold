"""SQL query router for the FMAS import interface."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from backend.config import DATA_DIR
from backend.models.enums import DataStatus
from backend.models.schemas import PreviewResponse, SQLConnectionRequest
from backend.services import data_profiler as profiler
from backend.services.data_store import DataStore
from backend.services.sql_connector import SQLConnectionError, execute_query, test_connection
from backend.routers.upload import get_store

log = logging.getLogger(__name__)
router = APIRouter(tags=["sql"])


@router.post("/sql/test")
async def sql_test(req: SQLConnectionRequest) -> dict:
    """Test a database connection without executing a real query."""
    return test_connection(req.connection_string)


@router.post("/sql/query", response_model=PreviewResponse)
async def sql_query(
    req: SQLConnectionRequest,
    store: DataStore = Depends(get_store),
) -> PreviewResponse:
    t0 = time.monotonic()
    try:
        df = execute_query(req.connection_string, req.query, req.max_rows)
    except SQLConnectionError as exc:
        msg = str(exc)
        if "timed out" in msg.lower():
            raise HTTPException(status_code=408, detail=msg) from exc
        if "cannot connect" in msg.lower():
            raise HTTPException(status_code=502, detail=msg) from exc
        raise HTTPException(status_code=422, detail=msg) from exc

    quality = profiler.profile(df)
    columns = profiler.build_column_info(df)
    preview_rows = profiler.build_preview_rows(df)

    data_id = store.store(df, {
        "source_type": "sql",
        "file_name": None,
        "status": DataStatus.READY,
    })

    log.info("SQL query processed in %.2fs → data_id=%s", time.monotonic() - t0, data_id)

    return PreviewResponse(
        data_id=data_id,
        file_name=None,
        source_type="sql",
        columns=columns,
        preview_rows=preview_rows,
        quality=quality,
        status=DataStatus.READY,
    )
