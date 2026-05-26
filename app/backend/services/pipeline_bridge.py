"""Bridge between the import interface and the FMAS pipeline."""
from __future__ import annotations

import importlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class PipelineBridge:
    """Prepare and optionally launch FMAS pipeline runs."""

    def __init__(self) -> None:
        # Check whether the pipeline package is importable in this environment.
        try:
            importlib.import_module("pipeline")
            self.pipeline_available = True
            log.info("FMAS pipeline is available.")
        except ImportError:
            self.pipeline_available = False
            log.warning("FMAS pipeline not importable; runs will be prepared but not executed.")

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare_run(
        self,
        data: np.ndarray,
        config: dict,
        run_id: str,
        output_dir: Path,
    ) -> Path:
        """Save data and config to *output_dir/run_id/*; return that directory."""
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        np.save(run_dir / "data.npy", data)

        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps({
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            }),
            encoding="utf-8",
        )
        _write_status(run_dir, "prepared")
        log.info("Run %s prepared: %s rows × %s cols.", run_id, *data.shape)
        return run_dir

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch_run(self, run_dir: Path) -> dict:
        """Execute the pipeline in a background thread if available."""
        run_id = run_dir.name
        if not self.pipeline_available:
            return {
                "status": "prepared",
                "run_id": run_id,
                "message": "Data saved. Install the FMAS pipeline to run analysis.",
                "run_dir": str(run_dir),
            }

        _write_status(run_dir, "running")
        thread = threading.Thread(target=self._run_pipeline, args=(run_dir,), daemon=True)
        thread.start()
        return {"status": "running", "run_id": run_id}

    def _run_pipeline(self, run_dir: Path) -> None:
        run_id = run_dir.name
        try:
            from pipeline import FourierManifoldPipeline  # type: ignore[import]
            from config import default_config  # type: ignore[import]

            data = np.load(run_dir / "data.npy")
            config_dict = json.loads((run_dir / "config.json").read_text())

            cfg = default_config()
            cfg.manifold.n_charts = config_dict.get("n_charts", "auto")
            cfg.anomaly.threshold_method = config_dict.get("threshold_method", "adaptive")
            cfg.manifold.overlap_factor = config_dict.get("overlap_factor", 0.2)
            cfg.max_iterations = config_dict.get("max_iterations", 10)

            pipeline = FourierManifoldPipeline(config=cfg)
            results = pipeline.run(data)

            _save_pipeline_outputs(run_dir, results, config_dict.get("column_names"))
            _write_status(run_dir, "completed")
            log.info("Run %s completed.", run_id)
        except Exception as exc:
            log.exception("Run %s failed: %s", run_id, exc)
            _write_status(run_dir, "failed", error=str(exc))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_run_status(self, run_id: str, runs_dir: Path) -> dict:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            return {"status": "not_found", "run_id": run_id}
        status_path = run_dir / "status.json"
        if not status_path.exists():
            return {"status": "prepared", "run_id": run_id}
        return json.loads(status_path.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_status(run_dir: Path, status: str, error: str | None = None) -> None:
    payload: dict = {"status": status, "timestamp": datetime.now(timezone.utc).isoformat()}
    if error:
        payload["error"] = error
    (run_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def _save_pipeline_outputs(run_dir: Path, results: object, column_names: list[str] | None) -> None:
    """Persist pipeline outputs so the 3D viewer can load them."""
    try:
        # FinalReport exposes the intermediate results via attributes set during run().
        spectral = getattr(results, "_spectral", None)
        manifold = getattr(results, "_manifold", None)
        anomaly_results = getattr(results, "_anomaly_results", None)

        if spectral is not None:
            np.save(run_dir / "coefficients.npy", spectral.coefficients)
            np.save(run_dir / "power_spectrum.npy", spectral.power_spectrum)

        if anomaly_results is not None and manifold is not None:
            flags_obj = getattr(anomaly_results, "flags", None)
            scores_obj = getattr(anomaly_results, "scores", None)
            _save_anomaly_json(run_dir, flags_obj, scores_obj, anomaly_results)
            _save_manifold_json(run_dir, manifold)

        if column_names:
            (run_dir / "column_names.json").write_text(
                json.dumps(column_names), encoding="utf-8"
            )
    except Exception as exc:
        log.warning("Could not save all pipeline outputs: %s", exc)


def _save_anomaly_json(run_dir: Path, flags_obj, scores_obj, anomaly_results) -> None:
    overall_flags = getattr(flags_obj, "overall_flags", [])
    overall_scores = getattr(scores_obj, "overall", [])
    per_band = getattr(scores_obj, "per_band", {})

    payload = {
        "scores": [float(s) for s in overall_scores],
        "flags": [bool(f) for f in overall_flags],
        "types": getattr(anomaly_results, "anomaly_types", ["normal"] * len(overall_flags)),
        "band_scores": {str(k): [float(v) for v in arr] for k, v in per_band.items()},
        "per_point_band_scores": {str(k): [float(v) for v in arr] for k, v in per_band.items()},
    }
    (run_dir / "anomaly_results.json").write_text(json.dumps(payload), encoding="utf-8")


def _save_manifold_json(run_dir: Path, manifold) -> None:
    atlas = getattr(manifold, "atlas", None)
    charts = getattr(atlas, "charts", []) if atlas else []
    payload = {
        "n_charts": len(charts),
        "chart_assignments": [int(a) for a in getattr(atlas, "primary_assignments", [])],
        "intrinsic_dim": getattr(manifold, "intrinsic_dim", 2),
        "alignment_qualities": [
            float(getattr(getattr(c, "basis", None), "local", None) and
                  getattr(getattr(c.basis, "local", None), "alignment_score", 0.0) or 0.0)
            for c in charts
        ],
    }
    (run_dir / "manifold_info.json").write_text(json.dumps(payload), encoding="utf-8")
