"""Tests for common/math_utils.py."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

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
    """Eigendecompose returns ascending eigenvalues."""
    mat_a = np.array([[2.0, 1.0], [1.0, 2.0]])
    _, vals = eigendecompose(mat_a)
    assert vals[0] <= vals[1]


def test_eigendecompose_identity_eigenvalues() -> None:
    """Eigendecompose identity eigenvalues."""
    _, vals = eigendecompose(np.eye(4))
    np.testing.assert_allclose(vals, np.ones(4), atol=1e-10)


def test_eigendecompose_eigenvectors_orthonormal() -> None:
    """Eigendecompose eigenvectors orthonormal."""
    mat_a = np.array([[4.0, 2.0], [2.0, 3.0]])
    vecs, _ = eigendecompose(mat_a)
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(2), atol=1e-10)


def test_eigendecompose_n_components_selects_smallest() -> None:
    """Eigendecompose n components selects smallest."""
    mat_a = np.diag([1.0, 2.0, 3.0, 4.0])
    _, vals = eigendecompose(mat_a, n_components=2)
    assert vals.shape == (2,)
    np.testing.assert_allclose(vals, [1.0, 2.0], atol=1e-10)


def test_eigendecompose_n_components_clamped_to_n() -> None:
    """Eigendecompose n components clamped to n."""
    _, vals = eigendecompose(np.eye(3), n_components=100)
    assert vals.shape == (3,)


def test_eigendecompose_single_element_matrix() -> None:
    """Eigendecompose single element matrix."""
    _, vals = eigendecompose(np.array([[7.0]]))
    np.testing.assert_allclose(vals, [7.0], atol=1e-10)


def test_eigendecompose_non_square_raises() -> None:
    """Eigendecompose non square raises."""
    with pytest.raises(ValueError, match="square"):
        eigendecompose(np.ones((3, 4)))


# --- project_onto_subspace ---


def test_project_onto_subspace_onto_axes() -> None:
    """Project onto subspace onto axes."""
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = project_onto_subspace(np.array([1.0, 2.0, 3.0]), basis)
    np.testing.assert_allclose(result, [1.0, 2.0, 0.0], atol=1e-10)


def test_project_onto_subspace_zero_vector() -> None:
    """Project onto subspace zero vector."""
    result = project_onto_subspace(np.zeros(3), np.eye(3))
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-10)


def test_project_onto_subspace_full_space_is_identity() -> None:
    """Project onto subspace full space is identity."""
    v = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(project_onto_subspace(v, np.eye(4)), v, atol=1e-10)


# --- orthogonal_complement ---


def test_orthogonal_complement_orthogonal_to_basis() -> None:
    """Orthogonal complement orthogonal to basis."""
    basis = np.array([[1.0, 0.0, 0.0]])
    complement = orthogonal_complement(basis, ambient_dim=3)
    assert complement.shape[1] == 3
    for row in complement:
        np.testing.assert_allclose(basis @ row, np.zeros(1), atol=1e-10)


def test_orthogonal_complement_correct_dimension() -> None:
    """Orthogonal complement correct dimension."""
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert orthogonal_complement(basis, ambient_dim=3).shape == (1, 3)


def test_orthogonal_complement_full_basis_empty() -> None:
    """Orthogonal complement full basis empty."""
    assert orthogonal_complement(np.eye(3), ambient_dim=3).shape[0] == 0


def test_orthogonal_complement_wrong_dim_raises() -> None:
    """Orthogonal complement wrong dim raises."""
    with pytest.raises(ValueError):
        orthogonal_complement(np.array([[1.0, 0.0, 0.0]]), ambient_dim=4)


# --- gram_schmidt ---


def test_gram_schmidt_rows_orthonormal() -> None:
    """Gram schmidt rows orthonormal."""
    orth_mat = gram_schmidt(np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    np.testing.assert_allclose(np.linalg.norm(orth_mat[0]), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(orth_mat[1]), 1.0, atol=1e-10)
    np.testing.assert_allclose(orth_mat[0] @ orth_mat[1], 0.0, atol=1e-10)


def test_gram_schmidt_single_vector() -> None:
    """Gram schmidt single vector."""
    np.testing.assert_allclose(
        np.linalg.norm(gram_schmidt(np.array([[3.0, 4.0]]))[0]), 1.0, atol=1e-10
    )


def test_gram_schmidt_linearly_dependent_zero_row() -> None:
    """Gram schmidt linearly dependent zero row."""
    orth_mat = gram_schmidt(np.array([[1.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(orth_mat[1]), 0.0, atol=1e-10)


# --- numerical_jacobian ---


def test_numerical_jacobian_linear_map() -> None:
    """Numerical jacobian linear map."""
    mat_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    jacobian = numerical_jacobian(lambda x: mat_a @ x, np.zeros(2))
    np.testing.assert_allclose(jacobian, mat_a, atol=1e-5)


def test_numerical_jacobian_quadratic() -> None:
    """Numerical jacobian quadratic."""
    func = lambda x: np.array([x[0] ** 2 + x[1] ** 2])
    jacobian = numerical_jacobian(func, np.array([1.0, 2.0]))
    np.testing.assert_allclose(jacobian, [[2.0, 4.0]], atol=1e-5)


def test_numerical_jacobian_shape() -> None:
    """Numerical jacobian shape."""
    func = lambda x: np.array([x[0], x[1], x[0] + x[1]])
    assert numerical_jacobian(func, np.zeros(2)).shape == (3, 2)


# --- numerical_hessian ---


def test_numerical_hessian_quadratic_known() -> None:
    """Numerical hessian quadratic known."""
    # f(x) = x0^2 + 3*x1^2 → hessian = diag([2, 6])
    func = lambda x: x[0] ** 2 + 3 * x[1] ** 2
    hessian = numerical_hessian(func, np.array([0.0, 0.0]))
    np.testing.assert_allclose(hessian, [[2.0, 0.0], [0.0, 6.0]], atol=1e-4)


def test_numerical_hessian_symmetric() -> None:
    """Numerical hessian symmetric."""
    func = lambda x: x[0] * x[1] + x[0] ** 2
    hessian = numerical_hessian(func, np.array([1.0, 1.0]))
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-8)


def test_numerical_hessian_cross_term() -> None:
    """Numerical hessian cross term."""
    # f(x) = x0*x1 → hessian[0,1] = hessian[1,0] = 1, diagonal = 0
    func = lambda x: x[0] * x[1]
    hessian = numerical_hessian(func, np.zeros(2))
    np.testing.assert_allclose(hessian[0, 1], 1.0, atol=1e-4)
    np.testing.assert_allclose(hessian[1, 0], 1.0, atol=1e-4)


def test_numerical_hessian_shape() -> None:
    """Numerical hessian shape."""
    func = lambda x: float(x @ x)
    hessian = numerical_hessian(func, np.zeros(4))
    assert hessian.shape == (4, 4)


# --- smooth_check ---


def test_smooth_check_linear_is_smooth() -> None:
    """Smooth check linear is smooth."""
    mat_a = np.eye(2)
    func = lambda x: mat_a @ x
    points = np.linspace([0.0, 0.0], [1.0, 1.0], 5)
    assert smooth_check(func, points) is True


def test_smooth_check_single_point_is_smooth() -> None:
    """Smooth check single point is smooth."""
    assert smooth_check(lambda x: x, np.array([[1.0, 2.0]])) is True


def test_smooth_check_step_function_not_smooth() -> None:
    """Smooth check step function not smooth."""
    def step(x: np.ndarray) -> np.ndarray:
        return np.array([1.0 if x[0] > 0.5 else 0.0])

    points = np.array([[0.49], [0.51]])
    # Large discontinuity at 0.5 → not smooth for tight eps
    assert smooth_check(step, points, eps=1e-3) is False


# --- solve_ode ---


def test_solve_ode_exponential_decay() -> None:
    """Solve ode exponential decay."""
    # dy/dt = -y, y(0) = 1 → y(t) = exp(-t)
    func = lambda t, y: -y
    result = solve_ode(func, np.array([1.0]), t_span=(0.0, 1.0))
    # Final value should be close to exp(-1) ≈ 0.368
    np.testing.assert_allclose(result[-1, 0], np.exp(-1.0), atol=1e-5)


def test_solve_ode_returns_2d_array() -> None:
    """Solve ode returns 2d array."""
    func = lambda t, y: np.zeros_like(y)
    result = solve_ode(func, np.array([1.0, 2.0]), t_span=(0.0, 0.1))
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_solve_ode_constant_solution() -> None:
    """Solve ode constant solution."""
    # dy/dt = 0 → y stays at y0
    func = lambda t, y: np.zeros_like(y)
    result = solve_ode(func, np.array([3.0]), t_span=(0.0, 1.0))
    np.testing.assert_allclose(result[:, 0], 3.0, atol=1e-10)


# --- matrix_log ---


def test_matrix_log_of_identity_is_zero() -> None:
    """Matrix log of identity is zero."""
    result = matrix_log(np.eye(3))
    np.testing.assert_allclose(result, np.zeros((3, 3)), atol=1e-10)


def test_matrix_log_inverse_of_expm() -> None:
    """Matrix log inverse of expm."""
    mat_a = np.array([[0.1, 0.0], [0.0, 0.2]])
    np.testing.assert_allclose(matrix_log(expm(mat_a)), mat_a, atol=1e-10)


def test_matrix_log_shape_preserved() -> None:
    """Matrix log shape preserved."""
    assert matrix_log(np.eye(4)).shape == (4, 4)


# --- svd_wrapper ---


def test_svd_wrapper_full_shapes() -> None:
    """Svd wrapper full shapes."""
    mat_a = np.random.default_rng(0).standard_normal((5, 3))
    left_vecs, sing_vals, right_vecs = svd_wrapper(mat_a)
    assert left_vecs.shape == (5, 3)
    assert sing_vals.shape == (3,)
    assert right_vecs.shape == (3, 3)


def test_svd_wrapper_truncated_k() -> None:
    """Svd wrapper truncated k."""
    mat_a = np.random.default_rng(1).standard_normal((6, 4))
    left_vecs, sing_vals, right_vecs = svd_wrapper(mat_a, k=2)
    assert left_vecs.shape == (6, 2)
    assert sing_vals.shape == (2,)
    assert right_vecs.shape == (2, 4)


def test_svd_wrapper_singular_values_descending() -> None:
    """Svd wrapper singular values descending."""
    mat_a = np.diag([3.0, 1.0, 2.0])
    _, sing_vals, _ = svd_wrapper(mat_a)
    assert list(sing_vals) == sorted(sing_vals, reverse=True)


def test_svd_wrapper_k_larger_than_rank_clamped() -> None:
    """Svd wrapper k larger than rank clamped."""
    mat_a = np.random.default_rng(2).standard_normal((4, 3))
    _, sing_vals, _ = svd_wrapper(mat_a, k=100)
    assert len(sing_vals) == 3  # clamped to min(m, n)


# --- safe_normalize ---


def test_safe_normalize_unit_vector() -> None:
    """Safe normalize unit vector."""
    result = safe_normalize(np.array([3.0, 4.0]))
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)


def test_safe_normalize_zero_vector() -> None:
    """Safe normalize zero vector."""
    np.testing.assert_allclose(safe_normalize(np.zeros(5)), np.zeros(5), atol=1e-10)


def test_safe_normalize_already_unit() -> None:
    """Safe normalize already unit."""
    v = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(safe_normalize(v), v, atol=1e-10)


# --- distance_matrix ---


def test_distance_matrix_known_distance() -> None:
    """Distance matrix known distance."""
    dist_mat = distance_matrix(np.array([[0.0, 0.0], [3.0, 4.0]]))
    np.testing.assert_allclose(dist_mat[0, 1], 5.0, atol=1e-10)
    np.testing.assert_allclose(dist_mat[0, 0], 0.0, atol=1e-10)


def test_distance_matrix_symmetric() -> None:
    """Distance matrix symmetric."""
    dist_mat = distance_matrix(np.random.default_rng(1).standard_normal((5, 3)))
    np.testing.assert_allclose(dist_mat, dist_mat.T, atol=1e-10)


def test_distance_matrix_single_point() -> None:
    """Distance matrix single point."""
    dist_mat = distance_matrix(np.array([[1.0, 2.0]]))
    assert dist_mat.shape == (1, 1)
    np.testing.assert_allclose(dist_mat[0, 0], 0.0, atol=1e-10)
