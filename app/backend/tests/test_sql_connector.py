"""Tests for backend/services/sql_connector.py."""
from __future__ import annotations

import pytest

from backend.services.sql_connector import (
    SQLConnectionError,
    _add_limit,
    _validate_query,
    execute_query,
    test_connection as _test_connection,
)


class TestValidateQuery:
    def test_select_passes(self):
        _validate_query("SELECT * FROM t")

    def test_with_cte_passes(self):
        _validate_query("WITH cte AS (SELECT 1) SELECT * FROM cte")

    def test_insert_raises(self):
        with pytest.raises(SQLConnectionError, match="INSERT"):
            _validate_query("INSERT INTO t VALUES (1)")

    def test_drop_raises(self):
        with pytest.raises(SQLConnectionError, match="DROP"):
            _validate_query("DROP TABLE t")

    def test_mixed_case_drop_raises(self):
        with pytest.raises(SQLConnectionError):
            _validate_query("Drop TABLE t")

    def test_delete_raises(self):
        with pytest.raises(SQLConnectionError):
            _validate_query("DELETE FROM t WHERE id=1")

    def test_update_raises(self):
        with pytest.raises(SQLConnectionError):
            _validate_query("UPDATE t SET x=1")

    def test_non_select_raises(self):
        with pytest.raises(SQLConnectionError):
            _validate_query("SHOW TABLES")


class TestAddLimit:
    def test_adds_limit_when_absent(self):
        q = _add_limit("SELECT * FROM t", 100)
        assert "LIMIT 100" in q

    def test_does_not_double_limit(self):
        q = _add_limit("SELECT * FROM t LIMIT 50", 100)
        assert q.count("LIMIT") == 1

    def test_strips_trailing_semicolon(self):
        q = _add_limit("SELECT 1;", 10)
        assert q.endswith("LIMIT 10")


class TestExecuteQuery:
    def test_select_from_sqlite(self, sqlite_db):
        df = execute_query(sqlite_db, "SELECT * FROM sensor_data")
        assert len(df) > 0
        assert "temperature" in df.columns

    def test_limit_enforced(self, sqlite_db):
        df = execute_query(sqlite_db, "SELECT * FROM sensor_data", max_rows=5)
        assert len(df) <= 5

    def test_write_rejected(self, sqlite_db):
        with pytest.raises(SQLConnectionError):
            execute_query(sqlite_db, "INSERT INTO sensor_data VALUES (999,1,1,1,1,1,'x','A')")

    def test_bad_connection_raises(self):
        with pytest.raises(SQLConnectionError):
            execute_query("postgresql://bad:bad@localhost:5432/nonexistent", "SELECT 1")


class TestTestConnection:
    def test_sqlite_success(self, sqlite_db):
        result = _test_connection(sqlite_db)
        assert result["success"] is True

    def test_bad_connection_returns_false(self):
        result = _test_connection("postgresql://bad:bad@localhost:5432/nonexistent")
        assert result["success"] is False
        assert result["message"]
