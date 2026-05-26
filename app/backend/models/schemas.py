"""Pydantic request/response models for the FMAS import interface."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from backend.models.enums import DataStatus, SuitabilityLevel


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SQLConnectionRequest(BaseModel):
    connection_string: str
    query: str = "SELECT 1"
    max_rows: int = 500_000


class PipelineConfigRequest(BaseModel):
    n_charts: int | str = "auto"
    threshold_method: str = "adaptive"
    normalization: str = "standard"
    max_iterations: int = 10
    overlap_factor: float = 0.2
    band_method: str = "spectral_gaps"

    @field_validator("overlap_factor")
    @classmethod
    def _check_overlap(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("overlap_factor must be in the open interval (0, 1)")
        return v

    @field_validator("max_iterations")
    @classmethod
    def _check_iterations(cls, v: int) -> int:
        if v <= 0 or v > 50:
            raise ValueError("max_iterations must be between 1 and 50")
        return v


class LaunchRequest(BaseModel):
    data_id: str
    config: PipelineConfigRequest


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    dtype: str              # "numeric" | "string" | "datetime" | "other"
    is_numeric: bool
    missing_count: int
    missing_pct: float
    sample_values: list[str]


class QualityReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    n_rows: int
    n_columns: int
    n_numeric_columns: int
    n_non_numeric_columns: int
    missing_pct: float
    duplicate_rows: int
    constant_columns: list[str]
    suitability_score: float
    suitability_level: SuitabilityLevel
    warnings: list[str]
    estimated_runtime_seconds: float


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------

class PreviewResponse(BaseModel):
    data_id: str
    file_name: str | None = None
    source_type: str           # "file" or "sql"
    columns: list[ColumnInfo]
    preview_rows: list[dict[str, Any]]
    quality: QualityReport
    status: DataStatus


class LaunchResponse(BaseModel):
    run_id: str
    data_id: str
    config: PipelineConfigRequest
    status: str
    message: str
    viewer_url: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# 3D viewer response models (Prompt 13)
# ---------------------------------------------------------------------------

class ClusterMeshData(BaseModel):
    cluster_id: int
    vertices: list[list[float]]
    faces: list[list[int]]
    point_indices: list[int]
    centroid: list[float]
    reprojected: bool
    divergent_axes: list[int] | None = None


class ManifoldViewerData(BaseModel):
    surface_vertices: list[list[float]]
    surface_faces: list[list[int]]
    point_positions: list[list[float]]
    point_is_anomaly: list[bool]
    point_scores: list[float]
    point_ids: list[int]
    clusters: list[ClusterMeshData]
    n_points: int
    n_anomalies: int
    n_clusters: int
    projection_axes: list[int]
    axis_labels: list[str]


class PointDetailResponse(BaseModel):
    index: int
    overall_score: float
    is_anomaly: bool
    anomaly_type: str | None = None
    band_scores: dict[str, float]
    top_anomalous_band: str | None = None
    chart_id: int
    chart_alignment_quality: float
    cluster_id: int | None = None
    original_values: dict[str, float]
