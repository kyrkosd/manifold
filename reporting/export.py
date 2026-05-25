"""
Exports AnomalyReport objects to JSON, CSV, and Parquet formats, with
options for including or excluding intermediate artefacts (band scores,
chart assignments) and for streaming large reports to disk.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reporting.anomaly_report import AnomalyReport


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to Python natives.

    The standard json module raises TypeError on numpy types; this subclass
    handles them before falling back to the default encoder.
    """

    def default(self, obj):
        # Each branch converts one numpy family to its Python equivalent.
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            # Recursion-safe: tolist() returns plain Python scalars/lists.
            return obj.tolist()
        return super().default(obj)


def to_json(report: AnomalyReport) -> str:
    """Serialise the report summary and top anomalies to a JSON string.

    All numpy scalar types are converted to Python native equivalents so
    the output is safe to pass to any standard JSON consumer.

    Parameters
    ----------
    report : AnomalyReport.

    Returns
    -------
    str — valid JSON with keys "summary", "type_distribution", "top_anomalies".
    """
    payload = {
        "summary":           report.summary,
        "type_distribution": report.type_distribution,
        "top_anomalies":     report.top_anomalies,
    }
    return json.dumps(payload, cls=_NumpyEncoder, indent=2)


def to_csv(report: AnomalyReport, path: str) -> None:
    """Write the per-point table to a CSV file.

    The output is readable by both pandas (``pd.read_csv``) and Excel.
    No row-index column is written.

    Parameters
    ----------
    report : AnomalyReport.
    path : destination file path (created or overwritten).
    """
    report.table.to_csv(path, index=False)


def to_dataframe(report: AnomalyReport) -> pd.DataFrame:
    """Return the per-point table as a pandas DataFrame.

    Parameters
    ----------
    report : AnomalyReport.

    Returns
    -------
    pd.DataFrame — same object as report.table (no copy).
    """
    return report.table


def summary_dict(report: AnomalyReport) -> dict:
    """Return a flat dict of summary statistics suitable for logging.

    Merges summary and type_distribution into a single flat mapping with
    all values converted to Python native types.

    Parameters
    ----------
    report : AnomalyReport.

    Returns
    -------
    dict with str keys and Python-native values.
    """
    merged: dict = {}
    # summary holds aggregate counts; type_distribution holds per-type counts.
    for key, val in report.summary.items():
        merged[key] = _to_native(val)
    for key, val in report.type_distribution.items():
        merged[key] = _to_native(val)
    return merged


def _to_native(val):
    """Convert a numpy scalar to the equivalent Python built-in type.

    Non-numpy values are returned unchanged so this is safe to call on
    mixed dicts without pre-checking each value's type.
    """
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    # str, int, float, None, etc. — pass through unmodified.
    return val
