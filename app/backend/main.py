"""FastAPI application entry point for the FMAS import interface."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
##from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import DATA_DIR, FRONTEND_DIR, RUNS_DIR
from backend.models.schemas import ErrorResponse
from backend.routers import pipeline as pipeline_router
from backend.routers import sql as sql_router
from backend.routers import upload as upload_router
from backend.routers import viewer as viewer_router
from backend.services.data_store import DataStore
from backend.services.file_parser import FileParseError
from backend.services.sql_connector import SQLConnectionError

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Ensure data directories exist and remove stale datasets on startup."""
    # Ensure runtime directories exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up datasets older than 24 hours on startup.
    store = DataStore(data_dir=DATA_DIR)
    removed = store.cleanup_old(max_age_hours=24)
    if removed:
        log.info("Startup cleanup: removed %d stale datasets.", removed)
    yield


app = FastAPI(
    title="FMAS Import Interface",
    description="Data import and pipeline launch API for the Fourier Manifold Anomaly System.",
    version="0.1.0",
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Register routers.
app.include_router(upload_router.router, prefix="/api")
app.include_router(sql_router.router, prefix="/api")
app.include_router(pipeline_router.router, prefix="/api")
app.include_router(viewer_router.router, prefix="/api")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(FileParseError)
async def file_parse_error_handler(_req: Request, exc: FileParseError):
    """Convert FileParseError to a 422 JSON response."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="File parse error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(SQLConnectionError)
async def sql_error_handler(_req: Request, exc: SQLConnectionError):
    """Convert SQLConnectionError to a 422 JSON response."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="SQL error", detail=str(exc)).model_dump(),
    )


# ---------------------------------------------------------------------------
# Static files and page routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Return 200 OK for health checks."""
    return {"status": "ok"}


@app.get("/viewer", include_in_schema=False)
async def viewer_page():
    """Serve the 3D manifold viewer HTML page."""
    return FileResponse(FRONTEND_DIR / "viewer.html")


@app.get("/", include_in_schema=False)
async def index():
    """Serve the import interface HTML page."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files last so the explicit routes above take priority.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
