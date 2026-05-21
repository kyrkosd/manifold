"""
Estimates the tangent plane at each data point from the local chart
eigenvectors, and provides utilities for projecting vectors onto and
off the estimated tangent space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import math_utils


@dataclass
class TangentSpace:
    """Tangent and normal spaces at a single point on a manifold.

    Parameters
    ----------
    tangent_basis : (ambient_dim, chart_dim) orthonormal column matrix.
    normal_basis : (ambient_dim, ambient_dim - chart_dim) orthonormal columns.
    point : (ambient_dim,) the base point in ambient space.
    chart_dim : intrinsic dimension of the manifold.
    ambient_dim : dimension of the embedding space.
    """

    tangent_basis: np.ndarray   # (ambient_dim, chart_dim) orthonormal columns
    normal_basis: np.ndarray    # (ambient_dim, normal_dim) orthonormal columns
    point: np.ndarray           # (ambient_dim,) base point
    chart_dim: int              # intrinsic dimension
    ambient_dim: int            # ambient embedding dimension


def compute_space(
    point: np.ndarray,
    chart_map,
    chart_inverse,
    eps: float = 1e-5,
) -> TangentSpace:
    """Compute the tangent space at *point* via numerical differentiation.

    The tangent space T_pM is the column span of the Jacobian of the chart
    inverse at z = φ(p): T_pM = span(∂φ⁻¹/∂z₁, …, ∂φ⁻¹/∂z_d).

    Parameters
    ----------
    point : (ambient_dim,) base point in ambient space.
    chart_map : callable φ: (ambient_dim,) → (chart_dim,).
    chart_inverse : callable φ⁻¹: (chart_dim,) → (ambient_dim,).
    eps : finite-difference step size for the Jacobian.

    Returns
    -------
    TangentSpace with orthonormal tangent and normal bases.
    """
    # Map the ambient point to chart coordinates.
    z = chart_map(point)
    # Jacobian of φ⁻¹ at z: shape (ambient_dim, chart_dim).
    J = _chart_derivatives(chart_inverse, z, eps)
    tangent_basis = _span_basis(J)
    ambient_dim = tangent_basis.shape[0]
    chart_dim = tangent_basis.shape[1]
    normal_basis = compute_normal_space(tangent_basis, ambient_dim)
    return TangentSpace(
        tangent_basis=tangent_basis,
        normal_basis=normal_basis,
        point=point.copy(),
        chart_dim=chart_dim,
        ambient_dim=ambient_dim,
    )


def _chart_derivatives(
    chart_inverse, chart_coords: np.ndarray, eps: float
) -> np.ndarray:
    """Jacobian of *chart_inverse* at *chart_coords*; shape (ambient_dim, chart_dim).

    Parameters
    ----------
    chart_inverse : callable (chart_dim,) → (ambient_dim,).
    chart_coords : (chart_dim,) evaluation point in chart space.
    eps : finite-difference step size.
    """
    # numerical_jacobian(f, x) → (m, n) where f: (n,) → (m,).
    return math_utils.numerical_jacobian(chart_inverse, chart_coords, eps=eps)


def _span_basis(derivatives: np.ndarray) -> np.ndarray:
    """Orthonormalise the columns of *derivatives* via Gram-Schmidt.

    Parameters
    ----------
    derivatives : (ambient_dim, chart_dim) Jacobian matrix; columns are
        tangent vectors in ambient space.

    Returns
    -------
    np.ndarray : (ambient_dim, chart_dim) orthonormal column matrix.
    """
    # gram_schmidt operates on rows; transpose to treat columns as vectors.
    ortho_rows = math_utils.gram_schmidt(derivatives.T)   # (chart_dim, ambient_dim)
    return ortho_rows.T                                    # (ambient_dim, chart_dim)


def compute_normal_space(tangent_basis: np.ndarray, ambient_dim: int) -> np.ndarray:
    """Orthogonal complement of the tangent space in ambient space.

    Parameters
    ----------
    tangent_basis : (ambient_dim, chart_dim) orthonormal column matrix.
    ambient_dim : dimension of the embedding space.

    Returns
    -------
    np.ndarray : (ambient_dim, normal_dim) orthonormal column matrix.
        Empty (ambient_dim, 0) when chart_dim == ambient_dim.
    """
    chart_dim = tangent_basis.shape[1]
    if chart_dim >= ambient_dim:
        # Normal space is zero-dimensional; return empty column matrix.
        return np.zeros((ambient_dim, 0))
    # orthogonal_complement expects (k, ambient_dim) row matrix.
    normal_rows = math_utils.orthogonal_complement(tangent_basis.T, ambient_dim)
    return normal_rows.T   # (ambient_dim, normal_dim)


def _project_to_tangent(vector: np.ndarray, tangent_basis: np.ndarray) -> np.ndarray:
    """Project *vector* onto the tangent space: Σ ⟨v, eᵢ⟩ eᵢ.

    Parameters
    ----------
    vector : (ambient_dim,) vector to project.
    tangent_basis : (ambient_dim, chart_dim) orthonormal column matrix.
    """
    return tangent_basis @ (tangent_basis.T @ vector)


def _project_to_normal(vector: np.ndarray, normal_basis: np.ndarray) -> np.ndarray:
    """Project *vector* onto the normal space: Σ ⟨v, nⱼ⟩ nⱼ.

    Parameters
    ----------
    vector : (ambient_dim,) vector to project.
    normal_basis : (ambient_dim, normal_dim) orthonormal column matrix.
    """
    if normal_basis.shape[1] == 0:
        return np.zeros_like(vector)
    return normal_basis @ (normal_basis.T @ vector)


def _tangent_variation(
    tangent_spaces: list[np.ndarray], threshold: float = 0.1
) -> float:
    """Mean max principal angle between consecutive tangent spaces.

    Low values indicate a slowly curving manifold; high values indicate
    rapid local geometry change.

    Parameters
    ----------
    tangent_spaces : sequence of (ambient_dim, chart_dim) bases.
    threshold : unused cutoff kept for API compatibility.

    Returns
    -------
    float : mean of the maximum principal angle (radians) across consecutive
        pairs. Returns 0.0 for a list of fewer than two spaces.
    """
    if len(tangent_spaces) < 2:
        return 0.0
    angles = []
    for i in range(len(tangent_spaces) - 1):
        A, B = tangent_spaces[i], tangent_spaces[i + 1]
        # Singular values of A.T @ B equal cosines of principal angles.
        s = np.linalg.svd(A.T @ B, compute_uv=False)
        cos_vals = np.clip(s, -1.0, 1.0)
        # Maximum principal angle between the two subspaces.
        angles.append(float(np.max(np.arccos(cos_vals))))
    return float(np.mean(angles))
