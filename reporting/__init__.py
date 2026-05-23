"""
Phase 5 — Reporting.

Public interface: ``generate_report(results, data) → AnomalyReport``
assembles the per-point table, type distribution, top-anomaly list, and
statistical summary from Phase-4 AnomalyResults.
"""
from __future__ import annotations

import numpy as np

from anomaly import AnomalyResults
from reporting.anomaly_report import AnomalyReport
from reporting.anomaly_report import report as _report

__all__ = ["AnomalyReport", "generate_report"]


def generate_report(results: AnomalyResults, data: np.ndarray) -> AnomalyReport:
    """Generate a complete anomaly report from Phase-4 outputs.

    Parameters
    ----------
    results : AnomalyResults from anomaly.detect_anomalies.
    data : (n, d) original data matrix.

    Returns
    -------
    AnomalyReport
    """
    return _report(results, data)
