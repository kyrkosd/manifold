"""Pipeline configuration hierarchy using Pydantic BaseModel."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    normalization_method: str = "standard"


class StructureConfig(BaseModel):
    graph_threshold_method: str = "adaptive"
    dimensionality_methods: list[str] = Field(
        default_factory=lambda: ["eigenvalue_gap", "mle"]
    )


class FourierConfig(BaseModel):
    band_method: str = "spectral_gaps"
    min_bands: int = 3
    max_bands: int = 20


class ManifoldConfig(BaseModel):
    n_charts: int | str = "auto"
    overlap_factor: float = 0.2
    max_chart_retries: int = 3
    alignment_threshold: float = 0.5


class AnomalyConfig(BaseModel):
    threshold_method: str = "adaptive"
    contamination: float = 0.05
    multiple_testing: str = "bonferroni"


class FAISSConfig(BaseModel):
    metric: str = "L2"
    k_neighbors: int = 20
    batch_size: int = 10000


class PipelineConfig(BaseModel):
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    fourier: FourierConfig = Field(default_factory=FourierConfig)
    manifold: ManifoldConfig = Field(default_factory=ManifoldConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    max_iterations: int = 10
    verbose: bool = True


def default_config() -> PipelineConfig:
    return PipelineConfig()


def validate_config(config: PipelineConfig) -> list[str]:
    """Return a list of warning strings for out-of-range config values."""
    warnings: list[str] = []
    if not (0 < config.anomaly.contamination < 0.5):
        warnings.append("contamination should be in (0, 0.5)")
    if not (0 < config.manifold.overlap_factor < 1):
        warnings.append("overlap_factor should be in (0, 1)")
    if config.fourier.min_bands > config.fourier.max_bands:
        warnings.append("min_bands > max_bands")
    if config.faiss.k_neighbors <= 0:
        warnings.append("k_neighbors must be positive")
    if config.manifold.max_chart_retries <= 0:
        warnings.append("max_chart_retries must be positive")
    return warnings
