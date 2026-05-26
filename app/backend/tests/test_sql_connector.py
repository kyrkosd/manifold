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
    """Verify write-operation rejection in query validation."""

    def test_select_passes(self):
        """Verify a plain SELECT query passes validation."""
        _validate_query("SELECT * FROM t")

    def test_with_cte_passes(self):
        """Verify a CTE-prefixed SELECT passes validation."""
        _validate_query("WITH cte AS (SELECT 1) SELECT * FROM cte")

    def test_insert_raises(self):
        """Verify INSERT raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError, match="INSERT"):
            _validate_query("INSERT INTO t VALUES (1)")

    def test_drop_raises(self):
        """Verify DROP raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError, match="DROP"):
            _validate_query("DROP TABLE t")

    def test_mixed_case_drop_raises(self):
        """Verify mixed-case DROP raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError):
            _validate_query("Drop TABLE t")

    def test_delete_raises(self):
        """Verify DELETE raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError):
            _validate_query("DELETE FROM t WHERE id=1")

    def test_update_raises(self):
        """Verify UPDATE raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError):
            _validate_query("UPDATE t SET x=1")

    def test_non_select_raises(self):
        """Verify a non-SELECT statement raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError):
            _validate_query("SHOW TABLES")


class TestAddLimit:
    """Verify LIMIT clause injection behaviour."""

    def test_adds_limit_when_absent(self):
        """Verify LIMIT is appended when not present in the query."""
        q = _add_limit("SELECT * FROM t", 100)
        assert "LIMIT 100" in q

    def test_does_not_double_limit(self):
        """Verify an existing LIMIT clause is not duplicated."""
        q = _add_limit("SELECT * FROM t LIMIT 50", 100)
        assert q.count("LIMIT") == 1

    def test_strips_trailing_semicolon(self):
        """Verify trailing semicolons are removed before LIMIT is appended."""
        q = _add_limit("SELECT 1;", 10)
        assert q.endswith("LIMIT 10")


class TestExecuteQuery:
    """Verify query execution against a real SQLite database."""

    def test_select_from_sqlite(self, sqlite_db):
        """Verify a SELECT query against SQLite returns rows with expected columns."""
        df = execute_query(sqlite_db, "SELECT * FROM sensor_data")
        assert len(df) > 0
        assert "temperature" in df.columns

    def test_limit_enforced(self, sqlite_db):
        """Verify max_rows parameter caps the result set."""
        df = execute_query(sqlite_db, "SELECT * FROM sensor_data", max_rows=5)
        assert len(df) <= 5

    def test_write_rejected(self, sqlite_db):
        """Verify INSERT against SQLite raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError):
            execute_query(sqlite_db, "INSERT INTO sensor_data VALUES (999,1,1,1,1,1,'x','A')")

    def test_bad_connection_raises(self):
        """Verify an unreachable database raises SQLConnectionError."""
        with pytest.raises(SQLConnectionError):
            execute_query("postgresql://bad:bad@localhost:5432/nonexistent", "SELECT 1")


class TestTestConnection:
    """Verify connection-test helper returns correct success flags."""

    def test_sqlite_success(self, sqlite_db):
        """Verify test_connection returns success=True for a valid SQLite URL."""
        result = _test_connection(sqlite_db)
        assert result["success"] is True

    def test_bad_connection_returns_false(self):
        """Verify test_connection returns success=False for an invalid URL."""
        result = _test_connection("postgresql://bad:bad@localhost:5432/nonexistent")
        assert result["success"] is False
        assert result["message"]
