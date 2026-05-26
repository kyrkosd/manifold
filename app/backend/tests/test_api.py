"""Integration tests for the FMAS import interface API."""
from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

FIXTURES = Path(__file__).parent / "fixtures"


class TestUpload:
    def test_upload_csv_returns_preview(self, app_client):
        with open(FIXTURES / "test_data.csv", "rb") as f:
            res = app_client.post("/api/upload", files={"file": ("test_data.csv", f, "text/csv")})
        assert res.status_code == 200
        body = res.json()
        assert "data_id" in body
        assert body["source_type"] == "file"
        assert body["quality"]["n_rows"] == 20

    def test_upload_invalid_extension_rejected(self, app_client, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        with open(bad, "rb") as f:
            res = app_client.post("/api/upload", files={"file": ("image.png", f, "image/png")})
        assert res.status_code in (422, 400)

    def test_preview_endpoint(self, app_client):
        with open(FIXTURES / "test_data.csv", "rb") as f:
            upload_res = app_client.post("/api/upload", files={"file": ("test_data.csv", f, "text/csv")})
        data_id = upload_res.json()["data_id"]
        res = app_client.get(f"/api/data/{data_id}/preview")
        assert res.status_code == 200
        assert res.json()["data_id"] == data_id

    def test_preview_not_found(self, app_client):
        res = app_client.get("/api/data/nonexistent/preview")
        assert res.status_code == 404


class TestSQL:
    def test_sql_test_connection(self, app_client, sqlite_db):
        res = app_client.post("/api/sql/test", json={"connection_string": sqlite_db, "query": "SELECT 1"})
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_sql_query_returns_preview(self, app_client, sqlite_db):
        res = app_client.post("/api/sql/query", json={
            "connection_string": sqlite_db,
            "query": "SELECT * FROM sensor_data",
        })
        assert res.status_code == 200
        body = res.json()
        assert body["source_type"] == "sql"
        assert body["quality"]["n_rows"] > 0

    def test_sql_write_rejected(self, app_client, sqlite_db):
        res = app_client.post("/api/sql/query", json={
            "connection_string": sqlite_db,
            "query": "DROP TABLE sensor_data",
        })
        assert res.status_code == 422


class TestLaunch:
    def _upload_and_get_id(self, client) -> str:
        with open(FIXTURES / "test_data.csv", "rb") as f:
            res = client.post("/api/upload", files={"file": ("test_data.csv", f, "text/csv")})
        return res.json()["data_id"]

    def test_launch_returns_run_id(self, app_client):
        data_id = self._upload_and_get_id(app_client)
        res = app_client.post("/api/launch", json={
            "data_id": data_id,
            "config": {"n_charts": "auto", "threshold_method": "adaptive",
                       "normalization": "standard", "max_iterations": 10,
                       "overlap_factor": 0.2, "band_method": "spectral_gaps"},
        })
        assert res.status_code == 200
        body = res.json()
        assert "run_id" in body
        assert body["data_id"] == data_id

    def test_launch_invalid_data_id(self, app_client):
        res = app_client.post("/api/launch", json={
            "data_id": "does_not_exist",
            "config": {"n_charts": "auto", "threshold_method": "adaptive",
                       "normalization": "standard", "max_iterations": 10,
                       "overlap_factor": 0.2, "band_method": "spectral_gaps"},
        })
        assert res.status_code == 404

    def test_launch_invalid_config(self, app_client):
        data_id = self._upload_and_get_id(app_client)
        res = app_client.post("/api/launch", json={
            "data_id": data_id,
            "config": {"n_charts": "auto", "threshold_method": "adaptive",
                       "normalization": "standard", "max_iterations": 10,
                       "overlap_factor": 2.0,   # invalid — outside (0,1)
                       "band_method": "spectral_gaps"},
        })
        assert res.status_code == 422


class TestHealth:
    def test_health_ok(self, app_client):
        res = app_client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
