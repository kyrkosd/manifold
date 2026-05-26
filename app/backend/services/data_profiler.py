"""Data quality analysis and suitability scoring for the FMAS import interface."""
from __future__ import annotations

import math
from typing import Any

##import numpy as np
import pandas as pd

from backend.models.enums import SuitabilityLevel
from backend.models.schemas import ColumnInfo, QualityReport


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile(df: pd.DataFrame) -> QualityReport:
    """Analyze *df* and return a complete QualityReport."""
    columns = _analyze_columns(df)
    n_rows, n_cols = df.shape
    n_numeric = sum(1 for c in columns if c.is_numeric)
    n_non_numeric = n_cols - n_numeric
    overall_missing = float(df.isna().mean().mean() * 100)
    dups = _count_duplicates(df)
    constant_cols = _find_constant_columns(df)

    score, level, warnings = _compute_suitability(
        n_rows, n_numeric, overall_missing, dups, len(constant_cols)
    )
    runtime = _estimate_runtime(n_rows, n_numeric)

    return QualityReport(
        n_rows=n_rows,
        n_columns=n_cols,
        n_numeric_columns=n_numeric,
        n_non_numeric_columns=n_non_numeric,
        missing_pct=round(overall_missing, 2),
        duplicate_rows=dups,
        constant_columns=constant_cols,
        suitability_score=round(score, 3),
        suitability_level=level,
        warnings=warnings,
        estimated_runtime_seconds=round(runtime, 1),
    )


def build_column_info(df: pd.DataFrame) -> list[ColumnInfo]:
    return _analyze_columns(df)


def build_preview_rows(df: pd.DataFrame, n: int = 100) -> list[dict[str, Any]]:
    return _build_preview_rows(df, n)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _analyze_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    infos = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
        elif is_numeric:
            dtype = "numeric"
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            dtype = "string"
        else:
            dtype = "other"

        # Count missing: NaN for numeric; NaN + empty string for string types.
        if dtype == "string":
            missing = int(series.isna().sum()) + int((series == "").sum())
        else:
            missing = int(series.isna().sum())
        missing_pct = round(missing / max(len(series), 1) * 100, 2)

        sample = [str(v) for v in series.dropna().head(5).tolist()]

        infos.append(ColumnInfo(
            name=col,
            dtype=dtype,
            is_numeric=is_numeric,
            missing_count=missing,
            missing_pct=missing_pct,
            sample_values=sample,
        ))
    return infos


def _count_duplicates(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    # Sample large DataFrames to keep profiling fast.
    if len(df) > 50_000:
        sample = df.sample(n=10_000, random_state=0)
        rate = sample.duplicated().mean()
        return int(rate * len(df))
    return int(df.duplicated().sum())


def _find_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns whose non-null values are all identical."""
    const = []
    for col in df.select_dtypes(include="number").columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() == 1:
            const.append(col)
    return const


def _compute_suitability(
    n_rows: int,
    n_numeric: int,
    missing_pct: float,
    duplicates: int,
    n_constant: int,
) -> tuple[float, SuitabilityLevel, list[str]]:
    score = 1.0
    warnings: list[str] = []

    if n_rows < 100:
        score -= 0.4
        warnings.append(f"Only {n_rows} rows — FMAS works best with at least 100.")
    elif n_numeric > 0 and n_rows < 10 * n_numeric:
        score -= 0.2
        warnings.append(
            f"Only {n_rows} rows for {n_numeric} numeric features. "
            "A ratio of at least 10:1 (samples:features) is recommended."
        )

    if n_numeric == 0:
        score -= 1.0  # No numeric data at all — FMAS cannot run.
        warnings.append(
            f"Only {n_numeric} numeric column(s) — FMAS needs at least 3 to build a meaningful manifold."
        )
    elif n_numeric < 3:
        score -= 0.5
        warnings.append(
            f"Only {n_numeric} numeric column(s) — FMAS needs at least 3 to build a meaningful manifold."
        )

    if missing_pct > 10:
        score -= 0.3
        warnings.append(f"{missing_pct:.1f}% missing values — large gaps may distort the manifold.")
    elif missing_pct > 1:
        score -= 0.1
        warnings.append(f"{missing_pct:.1f}% missing values — will be imputed during analysis.")

    dup_pct = duplicates / max(n_rows, 1) * 100
    if dup_pct > 10:
        score -= 0.1
        warnings.append(f"{dup_pct:.1f}% duplicate rows detected — consider deduplication.")

    if n_constant > 0:
        penalty = min(0.05 * n_constant, 0.2)
        score -= penalty
        s = "s" if n_constant > 1 else ""
        warnings.append(f"{n_constant} constant column{s} — will be excluded from analysis.")

    score = max(0.0, min(1.0, score))
    if score >= 0.7:
        level = SuitabilityLevel.READY
    elif score >= 0.4:
        level = SuitabilityLevel.NEEDS_ATTENTION
    else:
        level = SuitabilityLevel.NOT_SUITABLE

    return score, level, warnings


def _estimate_runtime(n_rows: int, n_numeric: int) -> float:
    est = 0.001 * n_rows * max(n_numeric, 1)
    return float(max(5.0, min(3600.0, est)))


def _build_preview_rows(df: pd.DataFrame, n: int = 100) -> list[dict[str, Any]]:
    subset = df.head(n)
    rows = []
    for _, row in subset.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if val is None or (isinstance(val, float) and math.isnan(val)):
                rec[col] = "—"  # em dash
            else:
                s = str(val)
                rec[col] = s[:50] if len(s) > 50 else s
        rows.append(rec)
    return rows
