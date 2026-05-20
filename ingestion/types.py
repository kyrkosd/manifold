"""
Ingestion-specific type definitions: raw dataset wrappers and the validated /
normalised dataset contracts that downstream phases consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NormParams:
    """Parameters captured during normalisation, required for the inverse transform.

    Parameters
    ----------
    means : (d,) per-feature means subtracted in the forward pass.
    stds : (d,) per-feature standard deviations used in the forward pass.
    method : normalisation strategy name (e.g. ``"standard"``).
    """

    means: np.ndarray
    stds: np.ndarray
    method: str


@dataclass
class QualityReport:
    """Summary of data-quality checks performed during validation.

    Parameters
    ----------
    n_samples : number of rows in the validated data.
    n_features : number of columns (features).
    missing_pct : percentage of values that are NaN (0–100).
    duplicate_count : duplicate rows beyond first occurrences.
    constant_dims : column indices with zero or undefined variance.
    suitability_score : overall data suitability in [0, 1].
    """

    n_samples: int
    n_features: int
    missing_pct: float
    duplicate_count: int
    constant_dims: list[int] = field(default_factory=list)
    suitability_score: float = 1.0


@dataclass
class ValidatedData:
    """Raw data that has passed all fatal validation checks.

    Parameters
    ----------
    data : (n, d) float64 ndarray; may contain NaN but no infinities.
    quality : quality report from the validation pass.
    """

    data: np.ndarray
    quality: QualityReport


@dataclass
class CleanData:
    """Normalised data ready for downstream FMAS pipeline stages.

    Parameters
    ----------
    data : (n, d) normalised float64 ndarray.
    norm_params : parameters required to invert the normalisation.
    quality : quality report from the validation pass.
    """

    data: np.ndarray
    norm_params: NormParams
    quality: QualityReport
