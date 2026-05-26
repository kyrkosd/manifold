"""Application configuration for the FMAS import interface."""
from __future__ import annotations

from pathlib import Path

# Absolute path to the app/ directory so services can resolve data paths.
APP_DIR = Path(__file__).parent.parent

DATA_DIR: Path = APP_DIR / "data"
RUNS_DIR: Path = DATA_DIR / "runs"
FRONTEND_DIR: Path = APP_DIR / "frontend"

MAX_UPLOAD_BYTES: int = 500 * 1024 * 1024  # 500 MB
MAX_ROWS: int = 500_000
MAX_STORED_DATASETS: int = 50
DATASET_MAX_AGE_HOURS: int = 24

# When the FMAS pipeline package is installed in the same environment,
# direct imports work because the repo root is on sys.path via pip install -e .
PIPELINE_MODULE: str = "pipeline"
