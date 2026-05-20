"""Tests for common/math_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from common.math_utils import (
    distance_matrix,
    eigendecompose,
    gram_schmidt,
    matrix_log,
    numerical_hessian,
    numerical_jacobian,
    orthogonal_complement,
    project_onto_subspace,
    safe_normalize,
    smooth_check,
    solve_ode,
    svd_wrapper,
)


# --- eigendecompose ---


def test_eigendecompose_returns_ascending_eigenvalues() -> None:
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    vecs, vals = eigendecompose(A)
    assert vals[0] <= vals[1]


def test_eigendecompose_identity_eigenvalues() -> None:
    vecs, vals = eigendecompose(np.eye(4))
    np.testing.assert_allclose(vals, np.ones(4), atol=1e-10)


def test_eigendecompose_eigenvectors_orthonormal() -> None:
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    vecs, _ = eigendecompose(A)
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(2), atol=1e-10)


def test_eigendecompose_n_components_selects_smallest() -> None:
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    vecs, vals = eigendecompose(A, n_components=2)
    assert vals.shape == (2,)
    np.testing.assert_allclose(vals, [1.0, 2.0], atol=1e-10)


def test_eigendecompose_n_components_clamped_to_n() -> None:
    vecs, vals = eigendecompose(np.eye(3), n_components=100)
    assert vals.shape == (3,)


def test_eigendecompose_single_element_matrix() -> None:
    vecs, vals = eigendecompose(np.array([[7.0]]))
    np.testing.assert_allclose(vals, [7.0], atol=1e-10)


def test_eigendecompose_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        eigendecompose(np.ones((3, 4)))


# --- project_onto_subspace ---


def test_project_onto_subspace_onto_axes() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = project_onto_subspace(np.array([1.0, 2.0, 3.0]), basis)
    np.testing.assert_allclose(result, [1.0, 2.0, 0.0], atol=1e-10)


def test_project_onto_subspace_zero_vector() -> None:
    result = project_onto_subspace(np.zeros(3), np.eye(3))
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-10)


def test_project_onto_subspace_full_space_is_identity() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(project_onto_subspace(v, np.eye(4)), v, atol=1e-10)


# --- orthogonal_complement ---


def test_orthogonal_complement_orthogonal_to_basis() -> None:
    basis = np.array([[1.0, 0.0, 0.0]])
    complement = orthogonal_complement(basis, ambient_dim=3)
    assert complement.shape[1] == 3
    for row in complement:
        np.testing.assert_allclose(basis @ row, np.zeros(1), atol=1e-10)


def test_orthogonal_complement_correct_dimension() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert orthogonal_complement(basis, ambient_dim=3).shape == (1, 3)


def test_orthogonal_complement_full_basis_empty() -> None:
    assert orthogonal_complement(np.eye(3), ambient_dim=3).shape[0] == 0


def test_orthogonal_complement_wrong_dim_raises() -> None:
    with pytest.raises(ValueError):
        orthogonal_complement(np.array([[1.0, 0.0, 0.0]]), ambient_dim=4)


# --- gram_schmidt ---


def test_gram_schmidt_rows_orthonormal() -> None:
    Q = gram_schmidt(np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    np.testing.assert_allclose(np.linalg.norm(Q[0]), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(Q[1]), 1.0, atol=1e-10)
    np.testing.assert_allclose(Q[0] @ Q[1], 0.0, atol=1e-10)


def test_gram_schmidt_single_vector() -> None:
    np.testing.assert_allclose(
        np.linalg.norm(gram_schmidt(np.array([[3.0, 4.0]]))[0]), 1.0, atol=1e-10
    )


def test_gram_schmidt_linearly_dependent_zero_row() -> None:
    Q = gram_schmidt(np.array([[1.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(Q[1]), 0.0, atol=1e-10)


# --- numerical_jacobian ---


def test_numerical_jacobian_linear_map() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    J = numerical_jacobian(lambda x: A @ x, np.zeros(2))
    np.testing.assert_allclose(J, A, atol=1e-5)


def test_numerical_jacobian_quadratic() -> None:
    func = lambda x: np.array([x[0] ** 2 + x[1] ** 2])
    J = numerical_jacobian(func, np.array([1.0, 2.0]))
    np.testing.assert_allclose(J, [[2.0, 4.0]], atol=1e-5)


def test_numerical_jacobian_shape() -> None:
    func = lambda x: np.array([x[0], x[1], x[0] + x[1]])
    assert numerical_jacobian(func, np.zeros(2)).shape == (3, 2)


# --- numerical_hessian ---


def test_numerical_hessian_quadratic_known() -> None:
    # f(x) = x0^2 + 3*x1^2 → H = diag([2, 6])
    func = lambda x: x[0] ** 2 + 3 * x[1] ** 2
    H = numerical_hessian(func, np.array([0.0, 0.0]))
    np.testing.assert_allclose(H, [[2.0, 0.0], [0.0, 6.0]], atol=1e-4)


def test_numerical_hessian_symmetric() -> None:
    func = lambda x: x[0] * x[1] + x[0] ** 2
    H = numerical_hessian(func, np.array([1.0, 1.0]))
    np.testing.assert_allclose(H, H.T, atol=1e-8)


def test_numerical_hessian_cross_term() -> None:
    # f(x) = x0*x1 → H[0,1] = H[1,0] = 1, diagonal = 0
    func = lambda x: x[0] * x[1]
    H = numerical_hessian(func, np.zeros(2))
    np.testing.assert_allclose(H[0, 1], 1.0, atol=1e-4)
    np.testing.assert_allclose(H[1, 0], 1.0, atol=1e-4)


def test_numerical_hessian_shape() -> None:
    func = lambda x: float(x @ x)
    H = numerical_hessian(func, np.zeros(4))
    assert H.shape == (4, 4)


# --- smooth_check ---


def test_smooth_check_linear_is_smooth() -> None:
    A = np.eye(2)
    func = lambda x: A @ x
    points = np.linspace([0.0, 0.0], [1.0, 1.0], 5)
    assert smooth_check(func, points) is True


def test_smooth_check_single_point_is_smooth() -> None:
    assert smooth_check(lambda x: x, np.array([[1.0, 2.0]])) is True


def test_smooth_check_step_function_not_smooth() -> None:
    def step(x: np.ndarray) -> np.ndarray:
        return np.array([1.0 if x[0] > 0.5 else 0.0])

    points = np.array([[0.49], [0.51]])
    # Large discontinuity at 0.5 → not smooth for tight eps
    assert smooth_check(step, points, eps=1e-3) is False


# --- solve_ode ---


def test_solve_ode_exponential_decay() -> None:
    # dy/dt = -y, y(0) = 1 → y(t) = exp(-t)
    func = lambda t, y: -y
    result = solve_ode(func, np.array([1.0]), t_span=(0.0, 1.0))
    # Final value should be close to exp(-1) ≈ 0.368
    np.testing.assert_allclose(result[-1, 0], np.exp(-1.0), atol=1e-5)


def test_solve_ode_returns_2d_array() -> None:
    func = lambda t, y: np.zeros_like(y)
    result = solve_ode(func, np.array([1.0, 2.0]), t_span=(0.0, 0.1))
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_solve_ode_constant_solution() -> None:
    # dy/dt = 0 → y stays at y0
    func = lambda t, y: np.zeros_like(y)
    result = solve_ode(func, np.array([3.0]), t_span=(0.0, 1.0))
    np.testing.assert_allclose(result[:, 0], 3.0, atol=1e-10)


# --- matrix_log ---


def test_matrix_log_of_identity_is_zero() -> None:
    result = matrix_log(np.eye(3))
    np.testing.assert_allclose(result, np.zeros((3, 3)), atol=1e-10)


def test_matrix_log_inverse_of_expm() -> None:
    from scipy.linalg import expm

    A = np.array([[0.1, 0.0], [0.0, 0.2]])
    np.testing.assert_allclose(matrix_log(expm(A)), A, atol=1e-10)


def test_matrix_log_shape_preserved() -> None:
    assert matrix_log(np.eye(4)).shape == (4, 4)


# --- svd_wrapper ---


def test_svd_wrapper_full_shapes() -> None:
    A = np.random.default_rng(0).standard_normal((5, 3))
    U, s, Vt = svd_wrapper(A)
    assert U.shape == (5, 3)
    assert s.shape == (3,)
    assert Vt.shape == (3, 3)


def test_svd_wrapper_truncated_k() -> None:
    A = np.random.default_rng(1).standard_normal((6, 4))
    U, s, Vt = svd_wrapper(A, k=2)
    assert U.shape == (6, 2)
    assert s.shape == (2,)
    assert Vt.shape == (2, 4)


def test_svd_wrapper_singular_values_descending() -> None:
    A = np.diag([3.0, 1.0, 2.0])
    _, s, _ = svd_wrapper(A)
    assert list(s) == sorted(s, reverse=True)


def test_svd_wrapper_k_larger_than_rank_clamped() -> None:
    A = np.random.default_rng(2).standard_normal((4, 3))
    _, s, _ = svd_wrapper(A, k=100)
    assert len(s) == 3  # clamped to min(m, n)


# --- safe_normalize ---


def test_safe_normalize_unit_vector() -> None:
    result = safe_normalize(np.array([3.0, 4.0]))
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)


def test_safe_normalize_zero_vector() -> None:
    np.testing.assert_allclose(safe_normalize(np.zeros(5)), np.zeros(5), atol=1e-10)


def test_safe_normalize_already_unit() -> None:
    v = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(safe_normalize(v), v, atol=1e-10)


# --- distance_matrix ---


def test_distance_matrix_known_distance() -> None:
    D = distance_matrix(np.array([[0.0, 0.0], [3.0, 4.0]]))
    np.testing.assert_allclose(D[0, 1], 5.0, atol=1e-10)
    np.testing.assert_allclose(D[0, 0], 0.0, atol=1e-10)


def test_distance_matrix_symmetric() -> None:
    D = distance_matrix(np.random.default_rng(1).standard_normal((5, 3)))
    np.testing.assert_allclose(D, D.T, atol=1e-10)


def test_distance_matrix_single_point() -> None:
    D = distance_matrix(np.array([[1.0, 2.0]]))
    assert D.shape == (1, 1)
    np.testing.assert_allclose(D[0, 0], 0.0, atol=1e-10)
