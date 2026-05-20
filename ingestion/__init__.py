"""
Phase 0 — Data Ingestion.

Public interface: ``ingest(data) → CleanData`` orchestrates validation
followed by standard normalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .normalizer import normalize
from .types import CleanData
from .validator import validate

__all__ = ["ingest"]


def ingest(data: np.ndarray | pd.DataFrame) -> CleanData:
    """Validate, convert, and normalise raw input data.

    Runs Phase 0 steps [0.1] → [0.2] from the pipeline specification:
    validates the data then applies zero-mean, unit-variance normalisation.

    Parameters
    ----------
    data : (n, d) ndarray or DataFrame of numeric values.

    Returns
    -------
    CleanData : normalised array, normalisation params, and quality report.

    Raises
    ------
    ValidationError
        If data is non-numeric, has fewer than 2 rows/columns, or contains
        infinite values.
    """
    validated = validate(data)
    normalised, norm_params = normalize(validated.data)
    return CleanData(
        data=normalised,
        norm_params=norm_params,
        quality=validated.quality,
    )
