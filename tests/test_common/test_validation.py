"""Tests for common/validation.py."""
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


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


# --- validate_array ---


def test_validate_array_correct_ndim_does_not_raise() -> None:
    validate_array(np.ones((3, 4)), expected_ndim=2)


def test_validate_array_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="2D"):
        validate_array(np.ones((3,)), expected_ndim=2)


def test_validate_array_1d_expected() -> None:
    validate_array(np.arange(5), expected_ndim=1)


def test_validate_array_empty_array_correct_ndim() -> None:
    validate_array(np.empty((0, 3)), expected_ndim=2)


# --- validate_positive ---


def test_validate_positive_ok() -> None:
    validate_positive(1.0, "x")
    validate_positive(1e-10, "x")


def test_validate_positive_zero_raises() -> None:
    with pytest.raises(ValueError, match="x"):
        validate_positive(0.0, "x")


def test_validate_positive_negative_raises() -> None:
    with pytest.raises(ValueError):
        validate_positive(-5.0, "count")


# --- validate_range ---


def test_validate_range_within_bounds() -> None:
    validate_range(0.5, 0.0, 1.0, "overlap")


def test_validate_range_at_lower_bound() -> None:
    validate_range(0.0, 0.0, 1.0, "x")


def test_validate_range_at_upper_bound() -> None:
    validate_range(1.0, 0.0, 1.0, "x")


def test_validate_range_below_lower_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_range(-0.1, 0.0, 1.0, "overlap")


def test_validate_range_above_upper_raises() -> None:
    with pytest.raises(ValueError):
        validate_range(1.1, 0.0, 1.0, "x")


# --- validate_enum ---


def test_validate_enum_valid_member_does_not_raise() -> None:
    validate_enum(_Color.RED, _Color)


def test_validate_enum_wrong_type_raises() -> None:
    with pytest.raises(ValueError, match="_Color"):
        validate_enum("red", _Color)


def test_validate_enum_wrong_enum_type_raises() -> None:
    class _Other(Enum):
        RED = "red"

    with pytest.raises(ValueError):
        validate_enum(_Other.RED, _Color)


def test_validate_enum_none_raises() -> None:
    with pytest.raises(ValueError):
        validate_enum(None, _Color)


# --- validate_symmetric ---


def test_validate_symmetric_identity() -> None:
    assert validate_symmetric(np.eye(4)) is True


def test_validate_symmetric_non_square_false() -> None:
    assert validate_symmetric(np.ones((2, 3))) is False


def test_validate_symmetric_1d_false() -> None:
    assert validate_symmetric(np.array([1.0, 2.0])) is False


def test_validate_symmetric_asymmetric_false() -> None:
    A = np.array([[1.0, 2.0], [3.0, 1.0]])
    assert validate_symmetric(A) is False


def test_validate_symmetric_near_symmetric_true() -> None:
    A = np.array([[1.0, 2.0], [2.0 + 1e-9, 1.0]])
    assert validate_symmetric(A) is True


# --- validate_positive_definite ---


def test_validate_positive_definite_identity() -> None:
    assert validate_positive_definite(np.eye(3)) is True


def test_validate_positive_definite_negative_eigenvalue_false() -> None:
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert validate_positive_definite(A) is False


def test_validate_positive_definite_singular_false() -> None:
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert validate_positive_definite(A) is False


def test_validate_positive_definite_non_symmetric_false() -> None:
    assert validate_positive_definite(np.ones((2, 3))) is False


def test_validate_positive_definite_2x2_spd() -> None:
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    assert validate_positive_definite(A) is True


# --- validate_smoothness ---


def test_validate_smoothness_linear_is_smooth() -> None:
    func = lambda x: np.array([2.0 * x[0]])
    points = np.linspace([[0.0]], [[1.0]], 5).reshape(5, 1)
    assert validate_smoothness(func, points) is True


def test_validate_smoothness_single_point() -> None:
    assert validate_smoothness(lambda x: x, np.array([[1.0]])) is True


def test_validate_smoothness_empty_points() -> None:
    assert validate_smoothness(lambda x: x, np.empty((0, 2))) is True


def test_validate_smoothness_constant_is_smooth() -> None:
    func = lambda x: np.array([42.0])
    points = np.random.default_rng(0).standard_normal((5, 3))
    assert validate_smoothness(func, points) is True
