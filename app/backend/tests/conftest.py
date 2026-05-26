"""Shared pytest fixtures for the FMAS import interface tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """20-row, 5-numeric + 2-string DataFrame matching test_data.csv."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "temperature": rng.normal(75, 10, n).round(2),
        "pressure":    rng.normal(1.2, 0.1, n).round(4),
        "flow_rate":   rng.normal(50, 5, n).round(2),
        "voltage":     rng.normal(220, 5, n).round(2),
        "current":     rng.normal(10, 1, n).round(3),
        "sensor_id":   [f"S{i:03d}" for i in range(n)],
        "location":    rng.choice(["A", "B", "C"], n).tolist(),
    })
    df.loc[5, "temperature"] = None
    return df


@pytest.fixture
def csv_path() -> Path:
    return FIXTURES_DIR / "test_data.csv"


@pytest.fixture
def tsv_path() -> Path:
    return FIXTURES_DIR / "test_data.tsv"


@pytest.fixture
def xlsx_path() -> Path:
    return FIXTURES_DIR / "test_data.xlsx"


@pytest.fixture
def sqlite_db(tmp_path) -> str:
    """SQLite in-memory-style DB written to a temp file."""
    import sqlite3
    db_path = tmp_path / "test.db"
    rng = np.random.default_rng(0)
    n = 100
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE sensor_data ("
        "id INTEGER PRIMARY KEY,"
        "temperature REAL, pressure REAL, flow_rate REAL,"
        "voltage REAL, current REAL,"
        "sensor_id TEXT, location TEXT)"
    )
    rows = [
        (i,
         float(rng.normal(75, 10)),
         float(rng.normal(1.2, 0.1)),
         float(rng.normal(50, 5)),
         float(rng.normal(220, 5)),
         float(rng.normal(10, 1)),
         f"S{i:03d}",
         rng.choice(["A", "B", "C"]))
        for i in range(n)
    ]
    conn.executemany(
        "INSERT INTO sensor_data VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """TestClient pointing at the FastAPI app with a temp data directory."""
    monkeypatch.setattr("backend.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("backend.config.RUNS_DIR", tmp_path / "runs")
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()

    from backend.main import app
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("backend.config.RUNS_DIR", tmp_path / "runs")
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()

    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
