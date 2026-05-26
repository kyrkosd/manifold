"""Tests for backend/services/data_store.py."""
from __future__ import annotations

import threading
##import time

import numpy as np
import pandas as pd
import pytest

from backend.services.data_store import DataStore


@pytest.fixture
def store(tmp_path) -> DataStore:
    return DataStore(data_dir=tmp_path / "data")


class TestStoreAndRetrieve:
    def test_roundtrip(self, store, sample_df):
        did = store.store(sample_df, {"source_type": "file"})
        df2, meta = store.retrieve(did)
        assert len(df2) == len(sample_df)
        assert list(df2.columns) == list(sample_df.columns)

    def test_metadata_stored(self, store, sample_df):
        did = store.store(sample_df, {"source_type": "sql", "file_name": None})
        meta = store.get_metadata(did)
        assert meta["source_type"] == "sql"
        assert meta["n_rows"] == len(sample_df)

    def test_missing_raises_key_error(self, store):
        with pytest.raises(KeyError):
            store.retrieve("nonexistent")


class TestGetNumericArray:
    def test_only_numeric_columns(self, store, sample_df):
        did = store.store(sample_df, {})
        arr = store.get_numeric_array(did)
        # 5 numeric columns, 20 rows (one has NaN but not all-NaN row)
        assert arr.shape[1] == 5
        assert arr.dtype == np.float64

    def test_no_nan_rows(self, store):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, None, 6.0]})
        did = store.store(df, {})
        arr = store.get_numeric_array(did)
        # Row 1 is all-NaN after column selection → dropped.
        assert len(arr) == 2


class TestDeleteAndCleanup:
    def test_delete_removes(self, store, sample_df):
        did = store.store(sample_df, {})
        store.delete(did)
        with pytest.raises(KeyError):
            store.retrieve(did)

    def test_delete_silent_on_missing(self, store):
        store.delete("does_not_exist")  # should not raise

    def test_cleanup_old_removes_stale(self, store, sample_df):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        store.store(sample_df, {})
        future = datetime.now(timezone.utc) + timedelta(hours=25)
        with patch("backend.services.data_store.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.fromisoformat = datetime.fromisoformat
            removed = store.cleanup_old(max_age_hours=24)
        assert removed == 1


class TestMaxLimit:
    def test_evicts_oldest_when_full(self, tmp_path):
        """When MAX_DATASETS is exceeded the oldest entry is evicted."""
        import backend.services.data_store as ds_mod
        original = ds_mod.MAX_DATASETS
        ds_mod.MAX_DATASETS = 3
        try:
            store = DataStore(data_dir=tmp_path / "data")
            df = pd.DataFrame({"x": [1.0, 2.0]})
            ids = [store.store(df, {}) for _ in range(4)]
            assert len(store.list_datasets()) == 3
        finally:
            ds_mod.MAX_DATASETS = original


class TestConcurrentAccess:
    def test_concurrent_store(self, store):
        df = pd.DataFrame({"x": [1.0]})
        results = []

        def _store():
            results.append(store.store(df, {}))

        threads = [threading.Thread(target=_store) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(results) == 10
        assert len(set(results)) == 10  # all unique IDs
