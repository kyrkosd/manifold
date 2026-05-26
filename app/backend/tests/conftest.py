"""Shared pytest fixtures for the FMAS import interface tests."""
from __future__ import annotations

import sqlite3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.services.data_store import DataStore
from backend.services.manifold_projector import ManifoldProjector, ProjectionResult
from backend.services.mesh_builder import MeshBuilder

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# DataStore fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path) -> DataStore:
    """Provide a DataStore backed by a temporary directory."""
    return DataStore(data_dir=tmp_path / "data")


# ---------------------------------------------------------------------------
# MeshBuilder fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> MeshBuilder:
    """Provide a fresh MeshBuilder instance."""
    return MeshBuilder()


# ---------------------------------------------------------------------------
# ManifoldProjector fixtures (module-scoped for test_manifold_projector)
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path) -> Path:
    """Synthetic FMAS run: 1000 pts × 50 features, 3 planted clusters."""
    rng = np.random.default_rng(42)
    n_pts, n_feat = 1000, 50

    coeff = rng.standard_normal((n_pts, n_feat)) * 0.5
    coeff[:, 1:4] += rng.standard_normal((n_pts, 3)) * 2.0

    sep_idx     = list(range(940, 960))
    overlap_idx = list(range(960, 980))
    sep2_idx    = list(range(980, 1000))

    coeff[sep_idx,    1:4] += 50.0
    coeff[overlap_idx, 20] += 50.0
    coeff[sep2_idx,   1:4] -= 50.0

    power = np.mean(np.abs(coeff) ** 2, axis=0)

    all_anomaly = sep_idx + overlap_idx + sep2_idx
    flags  = [i in set(all_anomaly) for i in range(n_pts)]
    scores = [0.9 if f else 0.0 for f in flags]
    types  = ["regional" if f else "normal" for f in flags]

    cluster_labels = np.full(n_pts, -1, dtype=int)
    for i in sep_idx:     cluster_labels[i] = 0
    for i in overlap_idx: cluster_labels[i] = 1
    for i in sep2_idx:    cluster_labels[i] = 2

    anomaly = {
        "flags":  flags,
        "scores": scores,
        "types":  types,
        "per_point_band_scores": {
            "band_0": scores,
            "band_1": [s * 0.5 for s in scores],
        },
    }
    manifold = {
        "n_charts": 2,
        "chart_assignments": [i % 2 for i in range(n_pts)],
        "alignment_qualities": [0.95, 0.90],
        "intrinsic_dim": 2,
    }

    np.save(tmp_path / "coefficients.npy",  coeff)
    np.save(tmp_path / "power_spectrum.npy", power)
    np.save(tmp_path / "cluster_labels.npy", cluster_labels)
    (tmp_path / "anomaly_results.json").write_text(json.dumps(anomaly))
    (tmp_path / "manifold_info.json").write_text(json.dumps(manifold))
    (tmp_path / "column_names.json").write_text(json.dumps([f"feat_{i}" for i in range(n_feat)]))
    return tmp_path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    """Provide a populated FMAS run directory with synthetic data."""
    return _make_run_dir(tmp_path_factory.mktemp("run"))


@pytest.fixture(scope="module")
def projector() -> ManifoldProjector:
    """Provide a ManifoldProjector instance."""
    return ManifoldProjector()


@pytest.fixture(scope="module")
def projection(projector, run_dir) -> ProjectionResult:
    """Provide the full ProjectionResult for the synthetic run."""
    return projector.project(run_dir)


# ---------------------------------------------------------------------------
# Shared DataFrames
# ---------------------------------------------------------------------------

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
    """Return the path to the test CSV fixture file."""
    return FIXTURES_DIR / "test_data.csv"


@pytest.fixture
def tsv_path() -> Path:
    """Return the path to the test TSV fixture file."""
    return FIXTURES_DIR / "test_data.tsv"


@pytest.fixture
def xlsx_path() -> Path:
    """Return the path to the test XLSX fixture file."""
    return FIXTURES_DIR / "test_data.xlsx"


@pytest.fixture
def sqlite_db(tmp_path) -> str:
    """SQLite in-memory-style DB written to a temp file."""
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

    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(tmp_path, monkeypatch):
    """Async HTTPX client wired to the FastAPI app with temp directories."""
    monkeypatch.setattr("backend.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("backend.config.RUNS_DIR", tmp_path / "runs")
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
