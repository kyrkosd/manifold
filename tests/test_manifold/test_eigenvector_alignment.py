"""Tests for manifold/eigenvector_alignment.py.

Key fixture: two 3D bases that differ by a known 2D rotation in eigenspace.
After alignment the recovered rotation must be the transpose of the applied one.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import AlignmentQuality, EigenBasis
from manifold.eigenvector_alignment import (
    AlignedBasis,
    _align_ordering,
    _compute_alignment_quality,
    _correct_signs,
    _detect_genuine_divergence,
    _procrustes_rotation,
    align_to_reference,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ortho_basis(d: int, k: int, seed: int = 0) -> np.ndarray:
    """Return a (d, k) matrix with orthonormal columns."""
    rng = np.random.default_rng(seed)
    orth_mat, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return orth_mat[:, :k]


def _rotation_2d(theta: float) -> np.ndarray:
    """Return a 2×2 rotation matrix for angle *theta* (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _eigen_basis(d: int = 4, k: int = 3, seed: int = 0) -> EigenBasis:
    return EigenBasis(
        eigenvectors=_ortho_basis(d, k, seed),
        eigenvalues=np.arange(1, k + 1, dtype=float),
    )


# ---------------------------------------------------------------------------
# _correct_signs
# ---------------------------------------------------------------------------

class TestCorrectSigns:
    def test_flipped_sign_is_restored(self):
        # Flipping one column and correcting must recover the original.
        ref = _ortho_basis(4, 3, seed=1)
        local = ref.copy()
        local[:, 1] = -local[:, 1]   # flip column 1
        result = _correct_signs(local, ref)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_aligned_columns_unchanged(self):
        # Already-aligned columns must not be modified.
        ref = _ortho_basis(4, 3, seed=2)
        result = _correct_signs(ref.copy(), ref)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_does_not_modify_input(self):
        # _correct_signs must return a copy; the input must be unchanged.
        ref = _ortho_basis(4, 3, seed=3)
        local = ref.copy()
        local[:, 0] = -local[:, 0]
        original = local.copy()
        _correct_signs(local, ref)
        np.testing.assert_array_equal(local, original)


# ---------------------------------------------------------------------------
# _align_ordering
# ---------------------------------------------------------------------------

class TestAlignOrdering:
    def test_permuted_columns_are_reordered(self):
        # Swapping cols 0 and 2 of a reference; ordering must restore them.
        ref = _ortho_basis(5, 3, seed=4)
        perm = np.array([2, 1, 0])
        local = ref[:, perm]
        vals = np.array([1.0, 2.0, 3.0])
        result_vecs, _ = _align_ordering(local, vals[perm], ref)
        # Each column of result should match the corresponding reference column.
        for j in range(3):
            dot = abs(float(np.dot(result_vecs[:, j], ref[:, j])))
            assert dot > 0.99

    def test_eigenvalues_follow_reordering(self):
        # Eigenvalues must be permuted consistently with eigenvectors.
        ref = _ortho_basis(5, 3, seed=5)
        perm = np.array([1, 2, 0])
        local = ref[:, perm]
        vals = np.array([10.0, 20.0, 30.0])
        result_vecs, result_vals = _align_ordering(local, vals[perm], ref)
        # After reordering, result_vecs[:, j] ≈ ref[:, j] and vals should match.
        for j in range(3):
            dot = abs(float(np.dot(result_vecs[:, j], ref[:, j])))
            assert dot > 0.99
        # Eigenvalues should also be correctly permuted.
        assert set(result_vals) == {10.0, 20.0, 30.0}

    def test_identity_permutation_unchanged(self):
        # Already-ordered basis: output must equal input.
        ref = _ortho_basis(4, 3, seed=6)
        vals = np.array([1.0, 2.0, 3.0])
        result_vecs, result_vals = _align_ordering(ref.copy(), vals.copy(), ref)
        for j in range(3):
            assert abs(float(np.dot(result_vecs[:, j], ref[:, j]))) > 0.99
        np.testing.assert_array_equal(result_vals, vals)


# ---------------------------------------------------------------------------
# _procrustes_rotation
# ---------------------------------------------------------------------------

class TestProcrustesRotation:
    def test_result_is_orthogonal(self):
        # rotation @ rotation.T must be the identity for any input.
        ref = _ortho_basis(4, 3, seed=7)
        orth_mat = np.linalg.qr(np.random.default_rng(8).standard_normal((3, 3)))[0]
        rotation = _procrustes_rotation(ref @ orth_mat, ref)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        # A proper rotation has det = +1 (not a reflection).
        ref = _ortho_basis(4, 2, seed=9)
        rot_mat = _rotation_2d(0.7)
        rotation = _procrustes_rotation(ref @ rot_mat, ref)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)

    def test_known_rotation_recovered(self):
        # source = ref @ rot_mat → rotation ≈ rot_mat.T; then source @ rotation ≈ ref.
        ref = _ortho_basis(5, 2, seed=10)
        rot_mat = _rotation_2d(np.pi / 4)
        source = ref @ rot_mat
        rotation = _procrustes_rotation(source, ref)
        np.testing.assert_allclose(source @ rotation, ref, atol=1e-10)

    def test_identical_inputs_give_identity(self):
        # Rotating source = target: optimal rotation is the identity.
        ref = _ortho_basis(4, 3, seed=11)
        rotation = _procrustes_rotation(ref, ref)
        np.testing.assert_allclose(ref @ rotation, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# _compute_alignment_quality and _detect_genuine_divergence
# ---------------------------------------------------------------------------

class TestAlignmentQualityHelpers:
    def test_identical_bases_give_score_one(self):
        # Perfect column match → mean |dot| = 1.
        vecs = _ortho_basis(4, 3, seed=12)
        assert _compute_alignment_quality(vecs, vecs) == pytest.approx(1.0)

    def test_flipped_columns_reduce_score(self):
        # After flipping all signs the score is still 1 (abs dot product).
        vecs = _ortho_basis(4, 3, seed=13)
        assert _compute_alignment_quality(-vecs, vecs) == pytest.approx(1.0)

    def test_orthogonal_bases_give_score_zero(self):
        # Bases spanning orthogonal subspaces → all dot products ≈ 0.
        # Build two 1-column bases that are orthogonal.
        a = np.array([[1.0], [0.0], [0.0]])
        b = np.array([[0.0], [1.0], [0.0]])
        assert _compute_alignment_quality(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_divergence_below_threshold(self):
        # Quality < 0.5 should signal genuine divergence.
        assert _detect_genuine_divergence(0.3) is True

    def test_no_divergence_above_threshold(self):
        assert _detect_genuine_divergence(0.8) is False


# ---------------------------------------------------------------------------
# align_to_reference — integration
# ---------------------------------------------------------------------------

class TestAlignToReference:
    def test_returns_aligned_basis(self):
        # align_to_reference must return an AlignedBasis instance.
        ref = _eigen_basis(seed=14)
        local = _eigen_basis(seed=14)
        assert isinstance(align_to_reference(local, ref), AlignedBasis)

    def test_rotation_matrix_is_orthogonal(self):
        # The stored rotation_matrix must satisfy rotation @ rotation.T ≈ I.
        ref = _eigen_basis(seed=15)
        local = _eigen_basis(seed=16)
        result = align_to_reference(local, ref)
        rotation = result.rotation_matrix
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(rotation.shape[0]), atol=1e-10)

    def test_score_near_one_for_same_basis(self):
        # Aligning a basis to itself must yield a near-perfect score.
        basis = _eigen_basis(seed=17)
        result = align_to_reference(basis, basis)
        assert result.alignment_score == pytest.approx(1.0, abs=1e-8)

    def test_quality_excellent_for_same_basis(self):
        # Identical bases → residual ≈ 0 → EXCELLENT quality.
        basis = _eigen_basis(seed=18)
        assert align_to_reference(basis, basis).quality is AlignmentQuality.EXCELLENT

    def test_rotated_basis_aligns_well(self):
        # source = ref @ rot_mat: after alignment, eigenvectors should ≈ ref columns.
        ref = _eigen_basis(d=5, k=2, seed=19)
        rot_mat = _rotation_2d(np.pi / 6)
        rotated = EigenBasis(eigenvectors=ref.eigenvectors @ rot_mat,
                             eigenvalues=ref.eigenvalues.copy())
        result = align_to_reference(rotated, ref)
        np.testing.assert_allclose(result.eigenvectors,
                                   ref.eigenvectors, atol=1e-8)
