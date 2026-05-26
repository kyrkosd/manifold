"""Tests for backend/services/data_store.py."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import threading
##import time

import backend.services.data_store as ds_mod
import numpy as np
import pandas as pd
import pytest

from backend.services.data_store import DataStore


class TestStoreAndRetrieve:
    """Verify basic store/retrieve/metadata operations."""

    def test_roundtrip(self, store, sample_df):
        """Verify stored DataFrame can be retrieved with identical rows and columns."""
        did = store.store(sample_df, {"source_type": "file"})
        df2, _ = store.retrieve(did)
        assert len(df2) == len(sample_df)
        assert list(df2.columns) == list(sample_df.columns)

    def test_metadata_stored(self, store, sample_df):
        """Verify extra metadata fields are persisted alongside the DataFrame."""
        did = store.store(sample_df, {"source_type": "sql", "file_name": None})
        meta = store.get_metadata(did)
        assert meta["source_type"] == "sql"
        assert meta["n_rows"] == len(sample_df)

    def test_missing_raises_key_error(self, store):
        """Verify retrieving an unknown data_id raises KeyError."""
        with pytest.raises(KeyError):
            store.retrieve("nonexistent")


class TestGetNumericArray:
    """Verify numeric array extraction and NaN row dropping."""

    def test_only_numeric_columns(self, store, sample_df):
        """Verify only numeric columns are returned and dtype is float64."""
        did = store.store(sample_df, {})
        arr = store.get_numeric_array(did)
        # 5 numeric columns, 20 rows (one has NaN but not all-NaN row)
        assert arr.shape[1] == 5
        assert arr.dtype == np.float64

    def test_no_nan_rows(self, store):
        """Verify all-NaN rows are dropped from the numeric array."""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, None, 6.0]})
        did = store.store(df, {})
        arr = store.get_numeric_array(did)
        # Row 1 is all-NaN after column selection → dropped.
        assert len(arr) == 2


class TestDeleteAndCleanup:
    """Verify delete and age-based cleanup behaviour."""

    def test_delete_removes(self, store, sample_df):
        """Verify deleted dataset is no longer retrievable."""
        did = store.store(sample_df, {})
        store.delete(did)
        with pytest.raises(KeyError):
            store.retrieve(did)

    def test_delete_silent_on_missing(self, store):
        """Verify deleting a non-existent data_id does not raise."""
        store.delete("does_not_exist")  # should not raise

    def test_cleanup_old_removes_stale(self, store, sample_df):
        """Verify datasets older than max_age_hours are removed."""

        store.store(sample_df, {})
        future = datetime.now(timezone.utc) + timedelta(hours=25)
        with patch("backend.services.data_store.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.fromisoformat = datetime.fromisoformat
            removed = store.cleanup_old(max_age_hours=24)
        assert removed == 1


class TestEvictionAndConcurrency:
    """Verify the MAX_DATASETS eviction policy and thread-safe concurrent operations."""

    def test_evicts_oldest_when_full(self, tmp_path):
        """When MAX_DATASETS is exceeded the oldest entry is evicted."""
        original = ds_mod.MAX_DATASETS
        ds_mod.MAX_DATASETS = 3
        try:
            store = DataStore(data_dir=tmp_path / "data")
            df = pd.DataFrame({"x": [1.0, 2.0]})
            for _ in range(4):
                store.store(df, {})
            assert len(store.list_datasets()) == 3
        finally:
            ds_mod.MAX_DATASETS = original

    def test_concurrent_store(self, store):
        """Verify 10 concurrent stores each produce a unique data_id."""
        df = pd.DataFrame({"x": [1.0]})
        results = []

        def _store():
            results.append(store.store(df, {}))

        threads = [threading.Thread(target=_store) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(results) == 10
        assert len(set(results)) == 10  # all unique IDs
