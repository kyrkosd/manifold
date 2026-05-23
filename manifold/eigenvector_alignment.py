"""
Aligns eigenvector signs and orientations across overlapping charts to
produce a globally consistent local basis, resolving sign ambiguity via
greedy propagation through the chart overlap graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.types import AlignmentQuality, EigenBasis


@dataclass
class AlignedBasis:
    """Result of aligning a local eigenbasis to a reference eigenbasis.

    Parameters
    ----------
    eigenvectors : (d, k) aligned eigenvector matrix.
    eigenvalues : (k,) reordered to match reference column order.
    rotation_matrix : (k, k) orthogonal Procrustes rotation applied.
    alignment_score : Mean absolute dot product with reference columns [0, 1].
    quality : Categorical quality level derived from alignment_score.
    """

    eigenvectors: np.ndarray     # (d, k) after sign, ordering, and Procrustes
    eigenvalues: np.ndarray      # (k,) reordered to match reference
    rotation_matrix: np.ndarray  # (k, k) orthogonal; R @ R.T ≈ I
    alignment_score: float       # mean |<aligned_col, ref_col>|; 1.0 = perfect
    quality: AlignmentQuality    # categorical quality derived from score


def align_to_reference(
    local_basis: EigenBasis, reference_basis: EigenBasis
) -> AlignedBasis:
    """Align *local_basis* to *reference_basis* via a three-stage pipeline.

    Stage 1: reorder columns by maximum absolute dot product with reference.
    Stage 2: flip column signs to match reference orientation.
    Stage 3: apply orthogonal Procrustes rotation for fine alignment.

    Parameters
    ----------
    local_basis : EigenBasis from a local chart or region.
    reference_basis : EigenBasis used as the global orientation target.

    Returns
    -------
    AlignedBasis with eigenvectors, rotation matrix, score, and quality.
    """
    vecs, vals = _align_ordering(
        local_basis.eigenvectors,
        local_basis.eigenvalues,
        reference_basis.eigenvectors,
    )
    # Correct sign ambiguity after ordering; must precede Procrustes.
    vecs = _correct_signs(vecs, reference_basis.eigenvectors)
    rotation = _procrustes_rotation(vecs, reference_basis.eigenvectors)
    vecs_aligned = vecs @ rotation
    score = _compute_alignment_quality(vecs_aligned, reference_basis.eigenvectors)
    return AlignedBasis(
        eigenvectors=vecs_aligned,
        eigenvalues=vals,
        rotation_matrix=rotation,
        alignment_score=score,
        quality=_score_to_quality(score),
    )


def _correct_signs(eigenvectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip each column sign so that its dot product with the reference is positive.

    Parameters
    ----------
    eigenvectors : (d, k) local eigenvector matrix (columns are vectors).
    reference : (d, k) reference eigenvector matrix.

    Returns
    -------
    np.ndarray : (d, k) copy of *eigenvectors* with corrected column signs.
    """
    result = eigenvectors.copy()
    for i in range(result.shape[1]):
        # Flip sign when the dot product with the reference column is negative.
        if np.dot(result[:, i], reference[:, i]) < 0:
            result[:, i] = -result[:, i]
    return result


def _align_ordering(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder *eigenvectors* columns to best match *reference* columns.

    Uses the Hungarian algorithm on the absolute dot-product cost matrix
    so that each local column is matched to the most similar reference column.

    Parameters
    ----------
    eigenvectors : (d, k) local eigenvectors.
    eigenvalues : (k,) corresponding eigenvalues.
    reference : (d, k) reference eigenvectors.

    Returns
    -------
    tuple (reordered_eigenvectors, reordered_eigenvalues).
    """
    k = eigenvectors.shape[1]
    # Negate absolute dot products to convert to a minimisation problem.
    cost = -np.abs(eigenvectors.T @ reference)   # (k, k)
    row_ind, col_ind = linear_sum_assignment(cost)
    # Build inverse permutation: new_vecs[:, col_ind[i]] = eigenvectors[:, i]
    inv_perm = np.empty(k, dtype=int)
    inv_perm[col_ind] = row_ind
    return eigenvectors[:, inv_perm], eigenvalues[inv_perm]


def _procrustes_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find the orthogonal matrix R minimising ||source @ R − target||_F.

    SVD of source.T @ target = U S Vᵀ gives R = U diag(1,…,d) Vᵀ where
    the last diagonal entry is adjusted to ensure det(R) = +1.

    Parameters
    ----------
    source : (d, k) matrix to rotate.
    target : (d, k) target matrix.

    Returns
    -------
    np.ndarray : (k, k) orthogonal rotation matrix.
    """
    cross_mat = source.T @ target
    left_vecs, _sing_vals, right_vecs = np.linalg.svd(cross_mat)
    # Correct for potential reflection (det = -1).
    d = float(np.linalg.det(left_vecs @ right_vecs))
    correction = np.diag(np.concatenate([np.ones(len(_sing_vals) - 1), [d]]))
    return left_vecs @ correction @ right_vecs


def _compute_alignment_quality(aligned: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute dot product between corresponding columns of two bases.

    Returns a value in [0, 1]; 1.0 means perfect column-wise alignment.

    Parameters
    ----------
    aligned : (d, k) aligned eigenvector matrix.
    reference : (d, k) reference eigenvector matrix.
    """
    # Column-wise dot products via element-wise product summed over rows.
    dots = np.abs(np.sum(aligned * reference, axis=0))
    return float(np.mean(dots))


def _score_to_quality(score: float) -> AlignmentQuality:
    """Map a [0, 1] alignment score to an AlignmentQuality enum value.

    The mapping uses residual = 1 − score and the enum's documented thresholds.
    """
    residual = 1.0 - score
    if residual < 0.05:
        return AlignmentQuality.EXCELLENT
    if residual < 0.15:
        return AlignmentQuality.GOOD
    if residual < 0.30:
        return AlignmentQuality.ACCEPTABLE
    if residual < 0.50:
        return AlignmentQuality.POOR
    return AlignmentQuality.DIVERGENT


def _detect_genuine_divergence(quality: float, threshold: float = 0.5) -> bool:
    """Return True when *quality* falls below *threshold*.

    A low quality score indicates that the local structure genuinely differs
    from the reference rather than being a trivial sign/permutation variant.
    """
    return quality < threshold
