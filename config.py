"""Pipeline configuration hierarchy using Pydantic BaseModel."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration for the data ingestion phase."""

    normalization_method: str = "standard"

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        return []


class StructureConfig(BaseModel):
    """Configuration for the structure discovery phase."""

    graph_threshold_method: str = "adaptive"
    dimensionality_methods: list[str] = Field(
        default_factory=lambda: ["eigenvalue_gap", "mle"]
    )

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        return []


class FourierConfig(BaseModel):
    """Configuration for the Fourier analysis phase."""

    band_method: str = "spectral_gaps"
    min_bands: int = 3
    max_bands: int = 20

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.min_bands > self.max_bands:
            warnings.append("min_bands > max_bands")
        return warnings


class ManifoldConfig(BaseModel):
    """Configuration for the manifold construction phase."""

    n_charts: int | str = "auto"
    overlap_factor: float = 0.2
    max_chart_retries: int = 3
    alignment_threshold: float = 0.5

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.overlap_factor <= 0 or self.overlap_factor >= 1:
            warnings.append("overlap_factor should be in (0, 1)")
        if self.max_chart_retries <= 0:
            warnings.append("max_chart_retries must be positive")
        return warnings


class AnomalyConfig(BaseModel):
    """Configuration for the anomaly detection phase."""

    threshold_method: str = "adaptive"
    contamination: float = 0.05
    multiple_testing: str = "bonferroni"

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.contamination <= 0 or self.contamination >= 0.5:
            warnings.append("contamination should be in (0, 0.5)")
        return warnings


class FAISSConfig(BaseModel):
    """Configuration for the FAISS nearest-neighbour index."""

    metric: str = "L2"
    k_neighbors: int = 20
    batch_size: int = 10000

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.k_neighbors <= 0:
            warnings.append("k_neighbors must be positive")
        return warnings


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration aggregating all phase configs."""

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    fourier: FourierConfig = Field(default_factory=FourierConfig)
    manifold: ManifoldConfig = Field(default_factory=ManifoldConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    max_iterations: int = 10
    verbose: bool = True

    def validate_values(self) -> list[str]:
        """Aggregate validation warnings across all phase configs."""
        warnings: list[str] = []
        for sub in (self.ingestion, self.structure, self.fourier,
                    self.manifold, self.anomaly, self.faiss):
            warnings.extend(sub.validate_values())
        return warnings


def default_config() -> PipelineConfig:
    """Default config."""
    return PipelineConfig()


def validate_config(config: PipelineConfig) -> list[str]:
    """Return a list of warning strings for out-of-range config values."""
    return config.validate_values()
