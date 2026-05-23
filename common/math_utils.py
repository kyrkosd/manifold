"""Mathematical utilities: eigendecomposition, subspace projections,
orthogonalisation, Jacobians, Hessians, ODE integration, and distances.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist


def eigendecompose(
    matrix: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a real symmetric matrix using ``scipy.linalg.eigh``.
    Parameters
    ----------
    matrix : (n, n) symmetric matrix.  Raises ValueError if not square.
    n_components : k smallest pairs to return; ``None`` returns all n pairs.
    Returns
    -------
    eigenvectors : (n, k) ndarray, columns in ascending eigenvalue order.
    eigenvalues : (k,) ndarray in ascending order.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}.")
    n = matrix.shape[0]
    k = n if n_components is None else min(n_components, n)
    if k < n:
        eigenvalues, eigenvectors = linalg.eigh(matrix, subset_by_index=(0, k - 1))
    else:
        eigenvalues, eigenvectors = linalg.eigh(matrix)
    return eigenvectors, eigenvalues


def project_onto_subspace(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project *vector* onto the subspace spanned by orthonormal *basis* rows.
    Parameters
    ----------
    vector : (d,) input vector.
    basis : (k, d) matrix whose rows are orthonormal.
    Returns
    -------
    np.ndarray : (d,) projection onto column space of ``basis.T``.
    """
    return basis.T @ (basis @ vector)


def orthogonal_complement(basis: np.ndarray, ambient_dim: int) -> np.ndarray:
    """Return rows spanning the orthogonal complement of the row space of *basis*.
    Parameters
    ----------
    basis : (k, ambient_dim) subspace row matrix.
    ambient_dim : must equal ``basis.shape[1]``; raises ValueError otherwise.
    Returns
    -------
    np.ndarray : (ambient_dim - rank, ambient_dim) complement row matrix.
    """
    if basis.shape[1] != ambient_dim:
        raise ValueError(
            f"basis.shape[1]={basis.shape[1]} != ambient_dim={ambient_dim}."
        )
    return linalg.null_space(basis).T


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Orthonormalise rows of *vectors* via modified Gram–Schmidt.
    Parameters
    ----------
    vectors : (k, d) matrix of input row vectors.
    Returns
    -------
    np.ndarray : (k, d) orthonormal matrix; linearly dependent rows → zero.
    """
    k, d = vectors.shape
    Q = np.zeros((k, d), dtype=float)
    for i in range(k):
        v = vectors[i].astype(float)
        for j in range(i):
            v -= np.dot(v, Q[j]) * Q[j]
        norm = np.linalg.norm(v)
        Q[i] = v / norm if norm > 1e-10 else v
    return Q


def numerical_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Estimate the Jacobian of *func* at *point* via central finite differences.
    Parameters
    ----------
    func : (n,) → (m,) mapping.
    point : (n,) evaluation point.
    eps : finite-difference step size.
    Returns
    -------
    np.ndarray : (m, n) Jacobian approximation.
    """
    f0 = np.atleast_1d(func(point))
    m, n = f0.shape[0], point.shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = eps
        J[:, i] = (np.atleast_1d(func(point + delta)) - np.atleast_1d(func(point - delta))) / (2 * eps)
    return J


def numerical_hessian(
    func: Callable[[np.ndarray], float],
    point: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Estimate the Hessian of scalar *func* at *point* via cross central differences.

    H[i,j] = (f(x+eᵢ+eⱼ) - f(x+eᵢ-eⱼ) - f(x-eᵢ+eⱼ) + f(x-eᵢ-eⱼ)) / 4ε²
    Parameters
    ----------
    func : scalar-valued (n,) → float.
    point : (n,) evaluation point.
    eps : finite-difference step size.
    Returns
    -------
    np.ndarray : (n, n) symmetric Hessian approximation.
    """
    n = point.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = eps, eps
            H[i, j] = H[j, i] = (
                func(point + ei + ej) - func(point + ei - ej)
                - func(point - ei + ej) + func(point - ei - ej)
            ) / (4 * eps ** 2)
    return H


def smooth_check(
    func: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    eps: float = 1e-5,
) -> bool:
    """Return True if the function appears smooth at each consecutive pair.

    Two conditions are checked per consecutive pair (pᵢ, pᵢ₊₁):
    1. Jacobian Lipschitz: ‖J(pᵢ₊₁)−J(pᵢ)‖_F / ‖step‖ < 1/eps.
    2. Taylor residual: ‖f(pᵢ₊₁) − f(pᵢ) − J(pᵢ)·step‖ / ‖step‖² < 1/eps.
       For smooth functions the residual is O(‖step‖²); for discontinuous
       functions it is O(1), so the ratio explodes.

    Parameters
    ----------
    func : function under test.
    points : (n, d) ordered evaluation points.
    eps : finite-difference step for Jacobians; also sets Lipschitz bound 1/eps.
    Returns
    -------
    bool : True when both conditions hold at every consecutive pair.
    """
    if len(points) < 2:
        return True
    jacobians = [numerical_jacobian(func, p, eps=eps) for p in points]
    values = [np.atleast_1d(func(p)) for p in points]
    bound = 1.0 / eps
    for i in range(len(points) - 1):
        step_vec = points[i + 1] - points[i]
        step = np.linalg.norm(step_vec)
        if step < 1e-12:
            continue
        if np.linalg.norm(jacobians[i + 1] - jacobians[i], ord="fro") / step > bound:
            return False
        predicted = jacobians[i] @ step_vec
        residual = np.linalg.norm(values[i + 1] - values[i] - predicted)
        if residual / (step ** 2) > bound:
            return False
    return True


def safe_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalise *vector* to unit length; return zeros if norm < 1e-10.
    Parameters
    ----------
    vector : (d,) input vector.
    Returns
    -------
    np.ndarray : (d,) unit vector, or zero vector for near-zero input.
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-10 else np.zeros_like(vector, dtype=float)


def solve_ode(
    func: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    method: str = "RK45",
) -> np.ndarray:
    """Integrate ``func(t, y)`` over *t_span* via ``scipy.integrate.solve_ivp``.
    Parameters
    ----------
    func : right-hand side f(t, y) → dy/dt.
    y0 : (n,) initial state.
    t_span : (t_start, t_end) integration interval.
    method : integrator name; raises RuntimeError if integration fails.
    Returns
    -------
    np.ndarray : (n_steps, n) solution; rows are time steps.
    """
    result = solve_ivp(func, t_span, y0, method=method, rtol=1e-8, atol=1e-10)
    if not result.success:
        raise RuntimeError(f"ODE integration failed: {result.message}")
    return result.y.T


def matrix_log(matrix: np.ndarray) -> np.ndarray:
    """Compute the principal matrix logarithm via ``scipy.linalg.logm``.
    Parameters
    ----------
    matrix : (n, n) square invertible matrix.
    Returns
    -------
    np.ndarray : (n, n) matrix logarithm.
    """
    return linalg.logm(matrix)


def svd_wrapper(
    matrix: np.ndarray,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economy SVD; truncate to the top *k* singular values when *k* is given.
    Parameters
    ----------
    matrix : (m, n) input matrix.
    k : components to retain; ``None`` keeps all min(m, n).
    Returns
    -------
    U, s, Vt : (m, k), (k,), (k, n) left vectors, singular values, right vectors.
    """
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    if k is not None:
        k = min(k, len(s))
        return U[:, :k], s[:k], Vt[:k]
    return U, s, Vt


def distance_matrix(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute a symmetric pairwise distance matrix.
    Parameters
    ----------
    points : (n, d) row-vector array.
    metric : any metric accepted by ``scipy.spatial.distance.cdist``.
    Returns
    -------
    np.ndarray : (n, n) symmetric distance matrix, zeros on the diagonal.
    """
    pts = points.reshape(-1, 1) if points.ndim == 1 else points
    return cdist(pts, pts, metric=metric)
