"""
Custom exception hierarchy for FMAS.  All pipeline errors subclass
FourierManifoldError so callers can catch the entire hierarchy with a
single except clause, or target specific failure modes precisely.
"""
from __future__ import annotations


class FourierManifoldError(Exception):
    """Base exception for all FMAS pipeline errors.

    Parameters
    ----------
    message:
        Human-readable description of the error.
    recovery_suggestion:
        Optional hint for the caller on how to recover or work around.
    """

    def __init__(self, message: str, recovery_suggestion: str | None = None) -> None:
        super().__init__(message)
        self.recovery_suggestion = recovery_suggestion

    def __str__(self) -> str:
        base = super().__str__()
        if self.recovery_suggestion:
            return f"{base}\nSuggestion: {self.recovery_suggestion}"
        return base


class ValidationError(FourierManifoldError):
    """Raised when input data fails schema or constraint validation."""


class StructureError(FourierManifoldError):
    """Raised when graph construction or topology analysis fails."""


class ChartError(FourierManifoldError):
    """Raised when a manifold chart cannot be constructed or validated."""


class ConvergenceError(FourierManifoldError):
    """Raised when an iterative algorithm fails to converge."""


class DimensionalityError(FourierManifoldError):
    """Raised when intrinsic dimension estimation is inconclusive or invalid."""


class AlignmentError(FourierManifoldError):
    """Raised when eigenvector alignment across chart boundaries fails."""
