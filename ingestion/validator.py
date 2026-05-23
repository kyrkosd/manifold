"""
Schema and constraint validation for raw input datasets before they enter
the FMAS pipeline.  Fatal violations raise ValidationError; non-fatal
issues are logged as warnings and captured in QualityReport.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.exceptions import ValidationError
from common.logging import get_logger

from .types import QualityReport

_log = get_logger(__name__)


@dataclass
class ValidatedData:
    """Internal handoff between validate() and normalize(); not part of the public API.

    Parameters
    ----------
    data : (n, d) float64 ndarray; may contain NaN but no infinities.
    quality : quality report from the validation pass.
    """

    data: np.ndarray
    quality: QualityReport


def validate(data: np.ndarray | pd.DataFrame) -> ValidatedData:
    """Validate raw input and convert to a float64 ndarray.

    Parameters
    ----------
    data : (n, d) ndarray or DataFrame of numeric values.

    Returns
    -------
    ValidatedData : converted array and quality report.

    Raises
    ------
    ValidationError
        Non-numeric data, fewer than 2 rows/columns, or infinite values.
    """
    if not _check_types(data):
        raise ValidationError(
            "Data must be entirely numeric.",
            recovery_suggestion="Drop non-numeric columns before calling ingest().",
        )
    try:
        if isinstance(data, pd.DataFrame):
            arr: np.ndarray = data.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            arr = np.asarray(data, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"Cannot convert data to float64: {exc}.") from exc

    if not _check_dimensions(arr):
        raise ValidationError(
            f"Data must have ≥2 rows and ≥2 columns; got shape {arr.shape}.",
            recovery_suggestion="Provide at least 2 samples and 2 features.",
        )
    if not _check_ranges(arr):
        raise ValidationError(
            "Data contains infinite values.",
            recovery_suggestion="Replace inf / -inf with NaN or finite bounds.",
        )

    completeness = _check_completeness(arr)
    constant_dims = _check_constant_dims(arr)
    duplicate_count = _check_duplicates(arr)
    min_samples_ok = _check_min_samples(arr)

    if completeness < 1.0:
        _log.warning("%.1f%% of values are missing (NaN).", (1.0 - completeness) * 100)
    if constant_dims:
        _log.warning("Zero-variance columns at indices: %s", constant_dims)
    if duplicate_count > 0:
        _log.warning("%d duplicate row(s) detected.", duplicate_count)
    if not min_samples_ok:
        _log.warning(
            "n_samples=%d < 10 × n_features=%d; results may be unreliable.",
            arr.shape[0],
            arr.shape[1],
        )

    checks: dict[str, Any] = {
        "shape": arr.shape,
        "completeness": completeness,
        "constant_dims": constant_dims,
        "duplicate_count": duplicate_count,
        "min_samples": min_samples_ok,
    }
    return ValidatedData(data=arr, quality=_generate_quality_report(checks))


def _check_dimensions(data: np.ndarray) -> bool:
    """Return True if *data* has ≥2 rows and ≥2 columns."""
    return data.ndim == 2 and data.shape[0] >= 2 and data.shape[1] >= 2


def _check_completeness(data: np.ndarray) -> float:
    """Return the fraction of non-NaN values in [0, 1]."""
    if data.size == 0:
        return 1.0
    return float(1.0 - np.isnan(data).mean())


def _check_types(data: Any) -> bool:
    """Return True if all values in *data* are numeric."""
    if isinstance(data, pd.DataFrame):
        return all(pd.api.types.is_numeric_dtype(d) for d in data.dtypes)
    return np.issubdtype(np.asarray(data).dtype, np.number)


def _check_ranges(data: np.ndarray) -> bool:
    """Return True if *data* contains no infinite values (NaN is allowed)."""
    return bool(np.all(np.isfinite(data) | np.isnan(data)))


def _check_constant_dims(data: np.ndarray) -> list[int]:
    """Return column indices whose std is zero or undefined (all-NaN)."""
    stds = np.nanstd(data, axis=0)
    return [i for i, s in enumerate(stds) if s < 1e-10 or np.isnan(s)]


def _check_duplicates(data: np.ndarray) -> int:
    """Return the count of duplicate rows beyond their first occurrence."""
    return int(pd.DataFrame(data).duplicated().sum())


def _check_min_samples(data: np.ndarray) -> bool:
    """Return True if n_samples ≥ 10 × n_features (rule-of-thumb heuristic)."""
    return data.shape[0] >= 10 * data.shape[1]


def _generate_quality_report(checks: dict[str, Any]) -> QualityReport:
    """Assemble a QualityReport from the *checks* dict produced by validate()."""
    n_samples, n_features = checks["shape"]
    missing_pct = (1.0 - checks["completeness"]) * 100.0
    report = QualityReport(
        n_samples=n_samples,
        n_features=n_features,
        missing_pct=missing_pct,
        duplicate_count=checks["duplicate_count"],
        constant_dims=checks["constant_dims"],
        suitability_score=0.0,
    )
    report.suitability_score = _estimate_suitability(report)
    return report


def _estimate_suitability(report: QualityReport) -> float:
    """Compute a [0, 1] suitability score from a QualityReport.

    Deductions: up to 0.50 for missing values, 0.30 for constant dims,
    0.20 for duplicate rows.
    """
    score = 1.0
    score -= (report.missing_pct / 100.0) * 0.5
    if report.n_features > 0:
        score -= (len(report.constant_dims) / report.n_features) * 0.3
    if report.n_samples > 0:
        score -= (report.duplicate_count / report.n_samples) * 0.2
    return float(max(0.0, min(1.0, score)))
