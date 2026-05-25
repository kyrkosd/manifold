"""Pipeline configuration hierarchy using Pydantic BaseModel.

Each phase of the FMAS pipeline has its own config class. PipelineConfig
aggregates them all and delegates validation to each sub-config's
validate_values(), which returns a list of human-readable warning strings
rather than raising — this lets callers surface all problems at once.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase-level configs
# ---------------------------------------------------------------------------

class IngestionConfig(BaseModel):
    """Configuration for the data ingestion phase."""

    # "standard" → zero-mean unit-variance; "minmax" and "robust" also accepted.
    normalization_method: str = "standard"

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        return []

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


class StructureConfig(BaseModel):
    """Configuration for the structure discovery phase."""

    # "adaptive" builds the k-NN graph with a data-driven threshold.
    graph_threshold_method: str = "adaptive"
    # Multiple methods are run and the consensus intrinsic dim is used.
    dimensionality_methods: list[str] = Field(
        default_factory=lambda: ["eigenvalue_gap", "mle"]
    )

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        return []

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


class FourierConfig(BaseModel):
    """Configuration for the Fourier analysis phase."""

    # "spectral_gaps" detects band boundaries from eigenvalue distribution gaps.
    band_method: str = "spectral_gaps"
    min_bands: int = 3   # lower bound prevents degenerate single-band output
    max_bands: int = 20  # upper bound avoids over-segmentation on small graphs

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.min_bands > self.max_bands:
            warnings.append("min_bands > max_bands")
        return warnings

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


class ManifoldConfig(BaseModel):
    """Configuration for the manifold construction phase."""

    # "auto" sets k ≈ sqrt(n/100), clamped to [2, 20].
    n_charts: int | str = "auto"
    # Fraction of each chart's core region added as an overlap buffer.
    overlap_factor: float = 0.2
    # Number of times a chart may be split before the region is abandoned.
    max_chart_retries: int = 3
    # Minimum Procrustes alignment score for a chart to pass validation.
    alignment_threshold: float = 0.5

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.overlap_factor <= 0 or self.overlap_factor >= 1:
            warnings.append("overlap_factor should be in (0, 1)")
        if self.max_chart_retries <= 0:
            warnings.append("max_chart_retries must be positive")
        return warnings

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


class AnomalyConfig(BaseModel):
    """Configuration for the anomaly detection phase."""

    # "adaptive" uses MAD-based thresholding; "percentile" uses a fixed quantile.
    threshold_method: str = "adaptive"
    # Expected fraction of anomalies in the dataset; drives threshold calibration.
    contamination: float = 0.05
    # Multiple-testing correction applied across spectral bands.
    multiple_testing: str = "bonferroni"

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        # contamination > 0.5 would flag the majority as anomalous.
        if self.contamination <= 0 or self.contamination >= 0.5:
            warnings.append("contamination should be in (0, 0.5)")
        return warnings

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


class FAISSConfig(BaseModel):
    """Configuration for the FAISS nearest-neighbour index."""

    metric: str = "L2"       # "L2" or "IP" (inner product)
    k_neighbors: int = 20    # k used for region expansion and graph construction
    batch_size: int = 10000  # points processed per FAISS query call

    def validate_values(self) -> list[str]:
        """Return semantic validation warnings for this config."""
        warnings: list[str] = []
        if self.k_neighbors <= 0:
            warnings.append("k_neighbors must be positive")
        return warnings

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


# ---------------------------------------------------------------------------
# Top-level aggregate config
# ---------------------------------------------------------------------------

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
        # Collect warnings from every sub-config so all problems surface at once.
        for sub in (self.ingestion, self.structure, self.fourier,
                    self.manifold, self.anomaly, self.faiss):
            warnings.extend(sub.validate_values())
        return warnings

    def is_valid(self) -> bool:
        """Return True when validate_values produces no warnings."""
        return not self.validate_values()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def default_config() -> PipelineConfig:
    """Return a PipelineConfig populated entirely with default values."""
    return PipelineConfig()


def validate_config(config: PipelineConfig) -> list[str]:
    """Return a list of warning strings for out-of-range config values."""
    return config.validate_values()
