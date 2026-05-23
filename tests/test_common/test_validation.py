"""Tests for common/validation.py.

Each helper raises ValueError (not a domain-specific exception) so that
common/ stays a pure leaf with no dependency on common.exceptions.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pytest

from common.validation import (
    validate_array,
    validate_enum,
    validate_positive,
    validate_positive_definite,
    validate_range,
    validate_smoothness,
    validate_symmetric,
)


# Minimal enum used throughout the enum-related tests.
class _Color(Enum):
    RED = "red"
    BLUE = "blue"


# ---------------------------------------------------------------------------
# validate_array
# ---------------------------------------------------------------------------

def test_validate_array_correct_ndim_does_not_raise() -> None:
    """Validate array correct ndim does not raise."""
    # Normal 2-D array should pass without error.
    validate_array(np.ones((3, 4)), expected_ndim=2)


def test_validate_array_wrong_ndim_raises() -> None:
    """Validate array wrong ndim raises."""
    # A 1-D array is rejected when a 2-D array is required.
    with pytest.raises(ValueError, match="2D"):
        validate_array(np.ones((3,)), expected_ndim=2)


def test_validate_array_1d_expected() -> None:
    """Validate array 1d expected."""
    # 1-D expectation must also be honoured correctly.
    validate_array(np.arange(5), expected_ndim=1)


def test_validate_array_empty_array_correct_ndim() -> None:
    """Validate array empty array correct ndim."""
    # Shape (0, 3) is still 2-D; empty arrays with correct ndim must pass.
    validate_array(np.empty((0, 3)), expected_ndim=2)


# ---------------------------------------------------------------------------
# validate_positive
# ---------------------------------------------------------------------------

def test_validate_positive_ok() -> None:
    """Validate positive ok."""
    # Both a normal positive float and an extremely small positive float pass.
    validate_positive(1.0, "x")
    validate_positive(1e-10, "x")  # near-zero but strictly positive


def test_validate_positive_zero_raises() -> None:
    """Validate positive zero raises."""
    # Zero is not strictly positive; the error message should name the parameter.
    with pytest.raises(ValueError, match="x"):
        validate_positive(0.0, "x")


def test_validate_positive_negative_raises() -> None:
    """Validate positive negative raises."""
    # Any negative value must be rejected.
    with pytest.raises(ValueError):
        validate_positive(-5.0, "count")


# ---------------------------------------------------------------------------
# validate_range
# ---------------------------------------------------------------------------

def test_validate_range_within_bounds() -> None:
    """Validate range within bounds."""
    # A value strictly inside [0, 1] should be accepted.
    validate_range(0.5, 0.0, 1.0, "overlap")


def test_validate_range_at_lower_bound() -> None:
    """Validate range at lower bound."""
    # The lower bound is inclusive.
    validate_range(0.0, 0.0, 1.0, "x")


def test_validate_range_at_upper_bound() -> None:
    """Validate range at upper bound."""
    # The upper bound is inclusive.
    validate_range(1.0, 0.0, 1.0, "x")


def test_validate_range_below_lower_raises() -> None:
    """Validate range below lower raises."""
    # Values just below the lower bound must be rejected; name appears in message.
    with pytest.raises(ValueError, match="overlap"):
        validate_range(-0.1, 0.0, 1.0, "overlap")


def test_validate_range_above_upper_raises() -> None:
    """Validate range above upper raises."""
    # Values just above the upper bound must also be rejected.
    with pytest.raises(ValueError):
        validate_range(1.1, 0.0, 1.0, "x")


# ---------------------------------------------------------------------------
# validate_enum
# ---------------------------------------------------------------------------

def test_validate_enum_valid_member_does_not_raise() -> None:
    """Validate enum valid member does not raise."""
    # A genuine member of the enum class is always valid.
    validate_enum(_Color.RED, _Color)


def test_validate_enum_wrong_type_raises() -> None:
    """Validate enum wrong type raises."""
    # The raw string "red" is not an enum instance; class name in error message.
    with pytest.raises(ValueError, match="_Color"):
        validate_enum("red", _Color)


def test_validate_enum_wrong_enum_type_raises() -> None:
    """Validate enum wrong enum type raises."""
    # A member of a different enum with the same value must still be rejected.
    class _Other(Enum):
        RED = "red"

    with pytest.raises(ValueError):
        validate_enum(_Other.RED, _Color)


def test_validate_enum_none_raises() -> None:
    """Validate enum none raises."""
    # None is never a valid enum member.
    with pytest.raises(ValueError):
        validate_enum(None, _Color)


# ---------------------------------------------------------------------------
# validate_symmetric
# ---------------------------------------------------------------------------

def test_validate_symmetric_identity() -> None:
    """Validate symmetric identity."""
    # The identity matrix is trivially symmetric.
    assert validate_symmetric(np.eye(4)) is True


def test_validate_symmetric_non_square_false() -> None:
    """Validate symmetric non square false."""
    # Non-square matrices cannot be symmetric by definition.
    assert validate_symmetric(np.ones((2, 3))) is False


def test_validate_symmetric_1d_false() -> None:
    """Validate symmetric 1d false."""
    # A 1-D array has no transpose; function must not raise, just return False.
    assert validate_symmetric(np.array([1.0, 2.0])) is False


def test_validate_symmetric_asymmetric_false() -> None:
    """Validate symmetric asymmetric false."""
    # Off-diagonal elements differ; mat_a[0,1]=2, mat_a[1,0]=3.
    mat_a = np.array([[1.0, 2.0], [3.0, 1.0]])
    assert validate_symmetric(mat_a) is False


def test_validate_symmetric_near_symmetric_true() -> None:
    """Validate symmetric near symmetric true."""
    # Floating-point noise within atol=1e-8 must still be considered symmetric.
    mat_a = np.array([[1.0, 2.0], [2.0 + 1e-9, 1.0]])
    assert validate_symmetric(mat_a) is True


# ---------------------------------------------------------------------------
# validate_positive_definite
# ---------------------------------------------------------------------------

def test_validate_positive_definite_identity() -> None:
    """Validate positive definite identity."""
    # Identity is the canonical SPD matrix.
    assert validate_positive_definite(np.eye(3)) is True


def test_validate_positive_definite_negative_eigenvalue_false() -> None:
    """Validate positive definite negative eigenvalue false."""
    # A diagonal matrix with one negative entry is indefinite.
    mat_a = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert validate_positive_definite(mat_a) is False


def test_validate_positive_definite_singular_false() -> None:
    """Validate positive definite singular false."""
    # A rank-1 matrix has a zero eigenvalue; not strictly positive definite.
    mat_a = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert validate_positive_definite(mat_a) is False


def test_validate_positive_definite_non_symmetric_false() -> None:
    """Validate positive definite non symmetric false."""
    # Non-square input must be rejected without raising.
    assert validate_positive_definite(np.ones((2, 3))) is False


def test_validate_positive_definite_2x2_spd() -> None:
    """Validate positive definite 2x2 spd."""
    # [[4,2],[2,3]] has eigenvalues ~1.35 and ~5.65; both positive.
    mat_a = np.array([[4.0, 2.0], [2.0, 3.0]])
    assert validate_positive_definite(mat_a) is True


# ---------------------------------------------------------------------------
# validate_smoothness
# ---------------------------------------------------------------------------

def test_validate_smoothness_linear_is_smooth() -> None:
    """Validate smoothness linear is smooth."""
    # A linear map has zero second derivative; trivially Lipschitz.
    func = lambda x: np.array([2.0 * x[0]])
    points = np.linspace([[0.0]], [[1.0]], 5).reshape(5, 1)
    assert validate_smoothness(func, points) is True


def test_validate_smoothness_single_point() -> None:
    """Validate smoothness single point."""
    # One point means no consecutive pairs to compare; always smooth.
    assert validate_smoothness(lambda x: x, np.array([[1.0]])) is True


def test_validate_smoothness_empty_points() -> None:
    """Validate smoothness empty points."""
    # Empty point set has no pairs; function must return True without error.
    assert validate_smoothness(lambda x: x, np.empty((0, 2))) is True


def test_validate_smoothness_constant_is_smooth() -> None:
    """Validate smoothness constant is smooth."""
    # A constant function has zero variation; Lipschitz condition is satisfied.
    func = lambda x: np.array([42.0])
    points = np.random.default_rng(0).standard_normal((5, 3))
    assert validate_smoothness(func, points) is True
