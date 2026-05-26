"""Temporary parquet-backed storage for imported datasets."""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
MAX_DATASETS = 50


class DataStore:
    """Thread-safe temporary storage mapping data_id → (parquet file, metadata)."""

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, df: pd.DataFrame, metadata: dict) -> str:
        """Persist *df* and return a fresh *data_id*."""
        data_id = uuid.uuid4().hex
        path = self._data_dir / f"{data_id}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")

        record = {
            **metadata,
            "data_id": data_id,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(path),
        }
        with self._lock:
            self._meta[data_id] = record
            self._enforce_limit()

        log.info("Stored dataset %s (%d rows, %d cols).", data_id, len(df), len(df.columns))
        return data_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(self, data_id: str) -> tuple[pd.DataFrame, dict]:
        meta = self._get_meta_safe(data_id)
        df = pd.read_parquet(meta["path"], engine="pyarrow")
        return df, meta

    def get_metadata(self, data_id: str) -> dict:
        return self._get_meta_safe(data_id)

    def get_numeric_array(self, data_id: str) -> np.ndarray:
        """Return a clean float64 ndarray of numeric columns only (no NaN rows)."""
        df, _ = self.retrieve(data_id)
        numeric = df.select_dtypes(include="number")
        arr = numeric.to_numpy(dtype=np.float64)
        # Drop rows that are entirely NaN after column selection.
        valid = ~np.all(np.isnan(arr), axis=1)
        return arr[valid]

    # ------------------------------------------------------------------
    # Delete / cleanup
    # ------------------------------------------------------------------

    def delete(self, data_id: str) -> None:
        with self._lock:
            if data_id not in self._meta:
                return
            path = Path(self._meta.pop(data_id)["path"])
        path.unlink(missing_ok=True)
        log.info("Deleted dataset %s.", data_id)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        now = datetime.now(timezone.utc)
        to_delete = []
        with self._lock:
            for did, meta in list(self._meta.items()):
                ts = datetime.fromisoformat(meta["timestamp"])
                age_h = (now - ts).total_seconds() / 3600
                if age_h > max_age_hours:
                    to_delete.append(did)
        for did in to_delete:
            self.delete(did)
        return len(to_delete)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        with self._lock:
            return [
                {k: v for k, v in m.items() if k != "path"}
                for m in self._meta.values()
            ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_meta_safe(self, data_id: str) -> dict:
        with self._lock:
            if data_id not in self._meta:
                raise KeyError(f"Dataset {data_id!r} not found. It may have expired.")
            return self._meta[data_id]

    def _enforce_limit(self) -> None:
        """Delete oldest datasets when over MAX_DATASETS (call under self._lock)."""
        if len(self._meta) <= MAX_DATASETS:
            return
        oldest = sorted(self._meta.items(), key=lambda kv: kv[1]["timestamp"])
        for did, meta in oldest[: len(self._meta) - MAX_DATASETS]:
            path = Path(meta["path"])
            path.unlink(missing_ok=True)
            del self._meta[did]
            log.info("Evicted oldest dataset %s (limit %d reached).", did, MAX_DATASETS)
