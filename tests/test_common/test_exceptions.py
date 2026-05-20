"""Tests for common/exceptions.py."""
from __future__ import annotations

import pytest

from common.exceptions import (
    AlignmentError,
    ChartError,
    ConvergenceError,
    DimensionalityError,
    FourierManifoldError,
    StructureError,
    ValidationError,
)


def test_base_error_message_in_str() -> None:
    err = FourierManifoldError("something went wrong")
    assert "something went wrong" in str(err)


def test_base_error_no_recovery_suggestion_by_default() -> None:
    err = FourierManifoldError("oops")
    assert err.recovery_suggestion is None


def test_base_error_recovery_suggestion_stored() -> None:
    err = FourierManifoldError("oops", recovery_suggestion="try again")
    assert err.recovery_suggestion == "try again"


def test_base_error_recovery_suggestion_appears_in_str() -> None:
    err = FourierManifoldError("bad", recovery_suggestion="increase k")
    assert "increase k" in str(err)


def test_base_error_no_suggestion_str_clean() -> None:
    err = FourierManifoldError("plain error")
    assert "Suggestion" not in str(err)


def test_validation_error_is_subclass() -> None:
    assert issubclass(ValidationError, FourierManifoldError)


def test_validation_error_caught_as_base() -> None:
    with pytest.raises(FourierManifoldError):
        raise ValidationError("bad schema")


def test_structure_error_is_subclass() -> None:
    assert issubclass(StructureError, FourierManifoldError)


def test_chart_error_is_subclass() -> None:
    assert issubclass(ChartError, FourierManifoldError)


def test_convergence_error_is_subclass() -> None:
    assert issubclass(ConvergenceError, FourierManifoldError)


def test_dimensionality_error_is_subclass() -> None:
    assert issubclass(DimensionalityError, FourierManifoldError)


def test_alignment_error_is_subclass() -> None:
    assert issubclass(AlignmentError, FourierManifoldError)


def test_subclass_passes_recovery_suggestion() -> None:
    err = StructureError("disconnected", recovery_suggestion="increase k")
    assert err.recovery_suggestion == "increase k"
    assert "increase k" in str(err)


def test_all_errors_are_exceptions() -> None:
    for cls in (
        FourierManifoldError,
        ValidationError,
        StructureError,
        ChartError,
        ConvergenceError,
        DimensionalityError,
        AlignmentError,
    ):
        assert issubclass(cls, Exception)
