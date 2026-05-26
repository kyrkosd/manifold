"""Read-only SQL query execution for the FMAS import interface."""
from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy import create_engine, text

from backend.models.enums import DatabaseType

log = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 60
_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class SQLConnectionError(Exception):
    """Raised when a database connection or query fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(connection_string: str, query: str, max_rows: int = 500_000) -> pd.DataFrame:
    """Connect, validate, and execute *query*; return a DataFrame."""
    _log_safe(connection_string, "Connecting to")
    _validate_query(query)
    bounded = _add_limit(query, max_rows)
    return _connect_and_execute(connection_string, bounded)


def test_connection(connection_string: str) -> dict:
    """Try SELECT 1 and return a result dict (never raises)."""
    try:
        db_type = _detect_database_type(connection_string)
        _connect_and_execute(connection_string, "SELECT 1")
        return {"success": True, "message": "Connected", "database_type": db_type.value}
    except SQLConnectionError as exc:
        return {"success": False, "message": str(exc), "database_type": None}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _detect_database_type(connection_string: str) -> DatabaseType:
    cs = connection_string.lower()
    if cs.startswith("postgresql") or cs.startswith("postgres"):
        return DatabaseType.POSTGRESQL
    if cs.startswith("mysql"):
        return DatabaseType.MYSQL
    if cs.startswith("sqlite"):
        return DatabaseType.SQLITE
    raise SQLConnectionError(
        f"Unsupported database scheme. Use postgresql://, mysql://, or sqlite:///."
    )


def _validate_query(query: str) -> None:
    """Reject any query that isn't a read-only SELECT or CTE."""
    stripped = query.strip()
    match = _WRITE_KEYWORDS.search(stripped)
    if match:
        raise SQLConnectionError(
            f"Write operations are not permitted. Found: {match.group(0).upper()!r}."
        )
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise SQLConnectionError(
            "Query must start with SELECT or WITH (CTE). Only read-only queries are allowed."
        )
    if "WHERE" not in query.upper():
        log.warning("Query has no WHERE clause — may scan the full table.")


def _add_limit(query: str, max_rows: int) -> str:
    """Append LIMIT if the query doesn't already include one."""
    q = query.rstrip().rstrip(";")
    if not re.search(r"\bLIMIT\b", q, re.IGNORECASE):
        q = f"{q} LIMIT {max_rows}"
    return q


def _connect_and_execute(connection_string: str, query: str) -> pd.DataFrame:
    connect_args: dict = {}
    cs = connection_string.lower()
    if cs.startswith("postgresql"):
        connect_args = {"connect_timeout": _TIMEOUT_SECONDS}
    elif cs.startswith("mysql"):
        connect_args = {"connect_timeout": _TIMEOUT_SECONDS}

    engine = None
    try:
        engine = create_engine(connection_string, connect_args=connect_args)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    except Exception as exc:
        raise _classify_error(exc) from exc
    finally:
        if engine:
            engine.dispose()


def _classify_error(exc: Exception) -> SQLConnectionError:
    msg = str(exc).lower()
    if "connection refused" in msg or "could not connect" in msg:
        return SQLConnectionError("Cannot connect. Check host and port.")
    if "authentication failed" in msg or "password" in msg or "access denied" in msg:
        return SQLConnectionError("Authentication failed. Check username and password.")
    if "does not exist" in msg or "no such table" in msg:
        return SQLConnectionError("Table not found. Check your query.")
    if "timeout" in msg or "timed out" in msg:
        return SQLConnectionError(
            f"Query timed out after {_TIMEOUT_SECONDS} seconds. "
            "Try adding filters."
        )

    return SQLConnectionError(str(exc))


def _log_safe(connection_string: str, prefix: str) -> None:
    """Log only the scheme and host, never the password."""
    try:
        parts = connection_string.split("@")
        safe = parts[-1] if "@" in connection_string else connection_string.split("///")[0]
        log.info("%s: %s", prefix, safe[:60])
    except Exception:
        log.info("%s: [redacted]", prefix)
