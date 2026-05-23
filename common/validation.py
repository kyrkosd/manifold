"""
Input validation utilities for FMAS. Called at phase boundaries to catch
contract violations early and provide clear, actionable error messages.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeVar

import numpy as np

_E = TypeVar("_E", bound=Enum)


def validate_array(data: np.ndarray, expected_ndim: int) -> None:
    """Raise ValueError if *data* does not have *expected_ndim* dimensions.

    Parameters
    ----------
    data:
        Array to validate.
    expected_ndim:
        Required number of dimensions.

    Raises
    ------
    ValueError
        If ``data.ndim != expected_ndim``.
    """
    if data.ndim != expected_ndim:
        raise ValueError(
            f"Expected {expected_ndim}D array, got {data.ndim}D with shape {data.shape}."
        )


def validate_positive(value: float, name: str) -> None:
    """Raise ValueError if *value* is not strictly positive.

    Parameters
    ----------
    value:
        Scalar to check.
    name:
        Variable name used in the error message.

    Raises
    ------
    ValueError
        If ``value <= 0``.
    """
    if value <= 0:
        raise ValueError(f"'{name}' must be strictly positive; got {value}.")


def validate_range(value: float, low: float, high: float, name: str) -> None:
    """Raise ValueError if *value* is not in [*low*, *high*].

    Parameters
    ----------
    value:
        Scalar to check.
    low, high:
        Inclusive bounds.
    name:
        Variable name used in the error message.

    Raises
    ------
    ValueError
        If ``value`` is outside ``[low, high]``.
    """
    if value < low or value > high:
        raise ValueError(f"'{name}' must be in [{low}, {high}]; got {value}.")


def validate_enum(value: object, enum_class: type[_E]) -> None:
    """Raise ValueError if *value* is not a member of *enum_class*.

    Parameters
    ----------
    value:
        Object to check.
    enum_class:
        Expected enum type.

    Raises
    ------
    ValueError
        If *value* is not an instance of *enum_class*.
    """
    if not isinstance(value, enum_class):
        valid = [e.value for e in enum_class]
        raise ValueError(
            f"Expected a {enum_class.__name__} member; got {value!r}. "
            f"Valid values: {valid}."
        )


def validate_symmetric(matrix: np.ndarray) -> bool:
    """Return True if *matrix* is square and symmetric up to 1e-8 tolerance.

    Parameters
    ----------
    matrix:
        Array to check.

    Returns
    -------
    bool
        True if square and ``allclose(matrix, matrix.T, atol=1e-8)``.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return bool(np.allclose(matrix, matrix.T, atol=1e-8))


def validate_positive_definite(matrix: np.ndarray) -> bool:
    """Return True if *matrix* is symmetric positive definite.

    Parameters
    ----------
    matrix:
        Square array to check.

    Returns
    -------
    bool
        True if symmetric and all eigenvalues are strictly positive.
    """
    if not validate_symmetric(matrix):
        return False
    try:
        return bool(np.all(np.linalg.eigvalsh(matrix) > 0))
    except np.linalg.LinAlgError:
        return False


def validate_smoothness(
    func: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
) -> bool:
    """Return True if *func* varies smoothly across *points*.

    Checks that output variation per unit input step stays below a
    Lipschitz bound (1/1e-5) for all consecutive point pairs.

    Parameters
    ----------
    func:
        Function mapping each (d,) point to an output array.
    points:
        (n, d) ordered sequence of evaluation points.

    Returns
    -------
    bool
        True if the Lipschitz ratio is bounded at every consecutive pair.
    """
    if len(points) < 2:
        return True
    _lip_bound = 1e5  # 1.0 / 1e-5
    values = [np.atleast_1d(func(p)) for p in points]
    for i in range(len(points) - 1):
        step = np.linalg.norm(points[i + 1] - points[i])
        if step < 1e-12:
            continue
        change = np.linalg.norm(values[i + 1] - values[i])
        if change / step > _lip_bound:
            return False
    return True
