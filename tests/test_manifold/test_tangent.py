"""Tests for manifold/tangent.py.

Canonical fixture: the 2-sphere S² in ℝ³ with a south-pole stereographic
chart.  At the north pole N=(0,0,1) the chart coordinates are (0,0), the
tangent space spans the xy-plane, and the normal is the z-axis.
"""
from __future__ import annotations

import numpy as np
import pytest

from manifold.tangent import (
    TangentSpace,
    _chart_derivatives,
    _project_to_normal,
    _project_to_tangent,
    _span_basis,
    _tangent_variation,
    compute_normal_space,
    compute_space,
)


# ---------------------------------------------------------------------------
# Sphere fixture (south-pole stereographic projection)
# ---------------------------------------------------------------------------

def _sphere_chart_map(p: np.ndarray) -> np.ndarray:
    """φ: ℝ³ → ℝ²  (stereographic from south pole; singular at (0,0,-1))."""
    x, y, z = p
    # At north pole (0,0,1): maps to (0, 0).
    return np.array([x / (1.0 + z), y / (1.0 + z)])


def _sphere_chart_inverse(uv: np.ndarray) -> np.ndarray:
    """φ⁻¹: ℝ² → ℝ³  (inverse stereographic from south pole)."""
    u, v = uv
    denom = 1.0 + u * u + v * v
    return np.array([2.0 * u / denom, 2.0 * v / denom, (1.0 - u * u - v * v) / denom])


_NORTH_POLE = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ortho_cols(d: int, k: int, seed: int = 0) -> np.ndarray:
    """Return (d, k) matrix with orthonormal columns."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return Q[:, :k]


# ---------------------------------------------------------------------------
# compute_space — shape, type, and sphere geometry
# ---------------------------------------------------------------------------

class TestComputeSpace:
    def test_returns_tangent_space_instance(self):
        # compute_space must return a TangentSpace dataclass.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        assert isinstance(ts, TangentSpace)

    def test_chart_dim_and_ambient_dim(self):
        # S² in ℝ³: chart_dim=2, ambient_dim=3.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        assert ts.chart_dim == 2 and ts.ambient_dim == 3

    def test_tangent_basis_orthonormal_at_north_pole(self):
        # T @ T.T must be the identity (orthonormal columns).
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        T = ts.tangent_basis
        np.testing.assert_allclose(T.T @ T, np.eye(2), atol=1e-6)

    def test_sphere_tangent_spans_xy_plane(self):
        # Tangent plane at north pole must lie in the xy-plane (z-component ≈ 0).
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        # Each tangent basis column must be orthogonal to the z-axis.
        z_axis = np.array([0.0, 0.0, 1.0])
        for j in range(ts.tangent_basis.shape[1]):
            assert abs(np.dot(ts.tangent_basis[:, j], z_axis)) == pytest.approx(0.0, abs=1e-5)

    def test_sphere_normal_is_z_axis(self):
        # Normal at north pole must be the z-axis (outward radial direction).
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        z_axis = np.array([0.0, 0.0, 1.0])
        n = ts.normal_basis[:, 0]
        # Allow for sign; the axis is unique up to ±.
        assert abs(abs(np.dot(n, z_axis)) - 1.0) < 1e-5

    def test_point_is_stored(self):
        # The base point must be stored verbatim in the TangentSpace.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        np.testing.assert_array_equal(ts.point, _NORTH_POLE)


# ---------------------------------------------------------------------------
# _chart_derivatives and _span_basis
# ---------------------------------------------------------------------------

class TestDerivativesAndBasis:
    def test_chart_derivatives_shape(self):
        # Jacobian of φ⁻¹ at (0,0) must be (3, 2) for a sphere chart.
        J = _chart_derivatives(_sphere_chart_inverse, np.array([0.0, 0.0]), eps=1e-5)
        assert J.shape == (3, 2)

    def test_span_basis_orthonormal(self):
        # Orthonormalized columns of a full-rank Jacobian must satisfy T.T @ T = I.
        J = _chart_derivatives(_sphere_chart_inverse, np.array([0.0, 0.0]), eps=1e-5)
        T = _span_basis(J)
        np.testing.assert_allclose(T.T @ T, np.eye(T.shape[1]), atol=1e-6)

    def test_span_basis_shape_preserved(self):
        # _span_basis must return (ambient_dim, chart_dim) matching input.
        cols = _ortho_cols(5, 3)
        T = _span_basis(cols)
        assert T.shape == (5, 3)


# ---------------------------------------------------------------------------
# compute_normal_space
# ---------------------------------------------------------------------------

class TestNormalSpace:
    def test_normal_orthogonal_to_tangent(self):
        # Every normal vector must be orthogonal to every tangent vector.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        cross = ts.tangent_basis.T @ ts.normal_basis  # (2, 1)
        np.testing.assert_allclose(cross, 0.0, atol=1e-6)

    def test_full_dim_chart_gives_empty_normal(self):
        # chart_dim == ambient_dim → normal space has 0 columns.
        T = _ortho_cols(3, 3)   # full-rank 3×3
        N = compute_normal_space(T, ambient_dim=3)
        assert N.shape == (3, 0)

    def test_normal_basis_orthonormal(self):
        # Normal columns must be orthonormal: N.T @ N = I.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        N = ts.normal_basis
        np.testing.assert_allclose(N.T @ N, np.eye(N.shape[1]), atol=1e-6)


# ---------------------------------------------------------------------------
# _project_to_tangent and _project_to_normal
# ---------------------------------------------------------------------------

class TestProjections:
    def test_tangent_plus_normal_recovers_vector(self):
        # For S² at north pole, v = v_tangent + v_normal for any ambient vector.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        v = np.array([1.0, 2.0, 3.0])
        vt = _project_to_tangent(v, ts.tangent_basis)
        vn = _project_to_normal(v, ts.normal_basis)
        np.testing.assert_allclose(vt + vn, v, atol=1e-6)

    def test_tangent_vector_projects_to_itself(self):
        # A tangent vector projected onto the tangent space is unchanged.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        t = ts.tangent_basis[:, 0]   # first tangent basis column
        vt = _project_to_tangent(t, ts.tangent_basis)
        np.testing.assert_allclose(vt, t, atol=1e-6)

    def test_normal_vector_projects_to_itself(self):
        # A normal vector projected onto the normal space is unchanged.
        ts = compute_space(_NORTH_POLE, _sphere_chart_map, _sphere_chart_inverse)
        n = ts.normal_basis[:, 0]
        vn = _project_to_normal(n, ts.normal_basis)
        np.testing.assert_allclose(vn, n, atol=1e-6)


# ---------------------------------------------------------------------------
# _tangent_variation
# ---------------------------------------------------------------------------

class TestTangentVariation:
    def test_single_space_returns_zero(self):
        # One space → no pairs → variation = 0.
        T = _ortho_cols(3, 2)
        assert _tangent_variation([T]) == pytest.approx(0.0)

    def test_identical_spaces_return_zero(self):
        # Two copies of the same basis → principal angles = 0.
        T = _ortho_cols(4, 2, seed=1)
        assert _tangent_variation([T, T]) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_spaces_return_pi_over_two(self):
        # Two 1-D spaces along orthogonal axes → angle = π/2.
        A = np.array([[1.0], [0.0], [0.0]])   # x-axis
        B = np.array([[0.0], [1.0], [0.0]])   # y-axis
        variation = _tangent_variation([A, B])
        assert variation == pytest.approx(np.pi / 2, abs=1e-10)
