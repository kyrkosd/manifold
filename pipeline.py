"""Top-level orchestration pipeline for the full FMAS workflow."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import structure as structure_mod
from anomaly import AnomalyResults, detect_anomalies
from config import PipelineConfig, default_config
from fourier import SpectralData, analyze_fourier
from ingestion import ingest
from manifold import Manifold, build_manifold
from reporting import AnomalyReport, generate_report
from structure.report import StructureReport


@dataclass
class FinalReport:
    """Complete output of a single pipeline run.

    Parameters
    ----------
    anomaly_report : full AnomalyReport from Phase 5.
    structure : StructureReport from Phase 1.
    manifold_summary : high-level atlas statistics.
    timing : wall-clock seconds per phase, keyed by phase name.
    config : PipelineConfig used for this run.
    """

    anomaly_report: AnomalyReport
    structure: StructureReport
    manifold_summary: dict
    timing: dict
    config: PipelineConfig


class FourierManifoldPipeline:
    """Orchestrates all six FMAS phases end-to-end."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or default_config()
        self._log = logging.getLogger(__name__)

    def validate_config(self) -> list[str]:
        """Return any configuration warnings for the current pipeline config."""
        return self.config.validate_values()

    def run(self, data: np.ndarray | pd.DataFrame) -> FinalReport:
        """Execute the full pipeline and return a FinalReport.

        Parameters
        ----------
        data : (n, d) raw input array or DataFrame.

        Returns
        -------
        FinalReport
        """
        timing: dict[str, float] = {}

        t0 = time.perf_counter()
        clean_data = self._ingest(data)
        timing["ingest"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        structure = self._discover_structure(clean_data)
        timing["structure"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        spectral = self._fourier_analyze(clean_data, structure)
        timing["fourier"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        manifold = self._build_manifold(clean_data, spectral, structure)
        timing["manifold"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = self._detect_anomalies(clean_data, manifold)
        timing["anomaly"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        report = self._generate_report(results, clean_data)
        timing["reporting"] = time.perf_counter() - t0

        if self.config.verbose:
            total = sum(timing.values())
            self._log.info(
                "Pipeline complete in %.2fs: %s",
                total,
                {k: f"{v:.3f}s" for k, v in timing.items()},
            )

        manifold_summary = {
            "n_charts": len(manifold.atlas.charts),
            "coverage_pct": manifold.atlas.coverage_pct,
            "n_points": manifold.atlas.n_points,
        }

        return FinalReport(
            anomaly_report=report,
            structure=structure,
            manifold_summary=manifold_summary,
            timing=timing,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Private phase helpers — each wraps one FMAS phase with the
    # appropriate config slice; keeps run() readable as a sequencer.
    # ------------------------------------------------------------------

    def _ingest(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Run Phase 1 (ingestion) and return the cleaned numpy array."""
        clean = ingest(data)
        return clean.data

    def _discover_structure(self, data: np.ndarray) -> StructureReport:
        """Run Phase 2 (structure discovery) and return a StructureReport."""
        return structure_mod.discover_structure(data)

    def _fourier_analyze(
        self, data: np.ndarray, structure: StructureReport
    ) -> SpectralData:
        """Run Phase 3 (graph Fourier analysis) and return SpectralData."""
        return analyze_fourier(data, structure)

    def _build_manifold(
        self,
        data: np.ndarray,
        spectral: SpectralData,
        structure: StructureReport,
    ) -> Manifold:
        """Run Phase 4 (manifold atlas construction) and return a Manifold.

        ManifoldConfig uses max_chart_retries but the atlas builder expects
        the key "max_retries" — the translation happens here.
        """
        config_dict = {
            "n_charts": self.config.manifold.n_charts,
            "overlap_factor": self.config.manifold.overlap_factor,
            "max_retries": self.config.manifold.max_chart_retries,
        }
        return build_manifold(data, spectral, structure, config_dict)

    def _detect_anomalies(
        self, data: np.ndarray, manifold: Manifold
    ) -> AnomalyResults:
        """Run Phase 5 (anomaly scoring and flagging) and return AnomalyResults."""
        return detect_anomalies(data, manifold)

    def _generate_report(
        self, results: AnomalyResults, data: np.ndarray
    ) -> AnomalyReport:
        """Run Phase 6 (reporting) and return the final AnomalyReport."""
        return generate_report(results, data)


def create_pipeline(config_dict: dict | None = None) -> FourierManifoldPipeline:
    """Instantiate a pipeline from an optional plain-dict config."""
    config = PipelineConfig(**config_dict) if config_dict else default_config()
    return FourierManifoldPipeline(config)


def run_analysis(
    data: np.ndarray | pd.DataFrame, **kwargs
) -> FinalReport:
    """Convenience wrapper: create a pipeline and run it in one call."""
    return create_pipeline(kwargs if kwargs else None).run(data)
