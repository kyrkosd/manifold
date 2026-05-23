"""
Defines a single manifold chart: a local coordinate patch characterised
by a centroid, a local eigenbasis, a membership radius, and the set of
data points assigned to that patch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

import manifold.chart_validator as chart_validator
from common import nn_utils
from common.exceptions import ChartError
from common.types import EigenBasis
from fourier.graph_fourier import GraphFourierEngine
from manifold.eigenvector_alignment import AlignedBasis, align_to_reference


@dataclass
class Chart:
    """A local coordinate patch on the data manifold.

    Parameters
    ----------
    region_indices : (n_region,) indices into *full_data* of core region points.
    expanded_indices : (n_expanded,) indices including overlap boundary points.
    local_basis : AlignedBasis for this patch, aligned to a reference.
    selected_indices : (intrinsic_dim,) column indices of selected eigenvectors.
    selected_vectors : (ambient_dim, intrinsic_dim) chart basis vectors.
    coordinates : (n_region, intrinsic_dim) chart coordinates of region points.
    chart_map : callable (ambient_dim,) → (intrinsic_dim,).
    chart_inverse : callable (intrinsic_dim,) → (ambient_dim,).
    intrinsic_dim : manifold intrinsic dimension.
    ambient_dim : ambient embedding dimension.
    """

    region_indices: np.ndarray       # (n_region,)
    expanded_indices: np.ndarray     # (n_expanded,)
    local_basis: AlignedBasis
    selected_indices: list[int]      # length intrinsic_dim
    selected_vectors: np.ndarray     # (ambient_dim, intrinsic_dim)
    coordinates: np.ndarray          # (n_region, intrinsic_dim)
    chart_map: Callable              # (ambient_dim,) → (intrinsic_dim,)
    chart_inverse: Callable          # (intrinsic_dim,) → (ambient_dim,)
    intrinsic_dim: int
    ambient_dim: int


@dataclass
class ChartBuildParams:
    """Fixed build-time parameters for constructing a single chart patch.

    Parameters
    ----------
    graph_builder : module with build(data) -> FeatureGraph.
    intrinsic_dim : target intrinsic dimension.
    faiss_index : pre-built FAISS index over the full dataset.
    overlap_factor : fraction of region size to include as boundary overlap.
    """

    graph_builder: Any
    intrinsic_dim: int
    faiss_index: Any
    overlap_factor: float = 0.2


@dataclass
class _ChartSpec:
    """Intermediate components used to assemble a Chart dataclass."""

    region_indices: np.ndarray
    expanded_indices: np.ndarray
    aligned: AlignedBasis
    sel_indices: list[int]
    intrinsic_dim: int
    ambient_dim: int


def _assemble_chart(region_data: np.ndarray, spec: _ChartSpec) -> Chart:
    """Construct a Chart dataclass from pre-computed eigenbasis components."""
    sel_vectors = spec.aligned.eigenvectors[:, spec.sel_indices]
    chart_map = _define_chart_map(sel_vectors)
    chart_inverse = _compute_chart_inverse(sel_vectors)
    coordinates = _compute_coordinates(region_data, chart_map)
    return Chart(
        region_indices=spec.region_indices,
        expanded_indices=spec.expanded_indices,
        local_basis=spec.aligned,
        selected_indices=spec.sel_indices,
        selected_vectors=sel_vectors,
        coordinates=coordinates,
        chart_map=chart_map,
        chart_inverse=chart_inverse,
        intrinsic_dim=spec.intrinsic_dim,
        ambient_dim=spec.ambient_dim,
    )


def build(
    region_data: np.ndarray,
    region_indices: np.ndarray,
    full_data: np.ndarray,
    reference_basis: EigenBasis,
    params: ChartBuildParams,
) -> Chart:
    """Build a manifold chart for the given region.

    Parameters
    ----------
    region_data : (n_region, d) data points in this chart's core region.
    region_indices : (n_region,) indices into *full_data*.
    full_data : (n, d) full dataset.
    reference_basis : global EigenBasis used for alignment.
    params : ChartBuildParams with graph_builder, intrinsic_dim, faiss_index,
        and overlap_factor.

    Returns
    -------
    Chart

    Raises
    ------
    ChartError
        When chart validation fails (injectivity hard gate not met).
    """
    region_indices_arr, expanded_indices = _expand_region(
        region_indices, full_data, params.faiss_index, params.overlap_factor
    )
    expanded_data = full_data[expanded_indices]
    aligned = _build_local_eigenbasis(expanded_data, params.graph_builder, reference_basis)
    sel_indices = _select_basis_adaptive(aligned, region_data, params.intrinsic_dim)
    spec = _ChartSpec(
        region_indices=region_indices_arr,
        expanded_indices=expanded_indices,
        aligned=aligned,
        sel_indices=sel_indices,
        intrinsic_dim=params.intrinsic_dim,
        ambient_dim=full_data.shape[1],
    )
    chart = _assemble_chart(region_data, spec)
    valid, _score, reason = chart_validator.validate(chart, region_data)
    if not valid:
        raise ChartError(
            f"Chart validation failed: {reason}",
            recovery_suggestion="Split region into smaller sub-regions.",
        )
    return chart


def _collect_outside_neighbours(
    nbr_indices: np.ndarray,
    region_set: set[int],
) -> set[int]:
    """Return neighbour indices that lie outside region_set."""
    outside: set[int] = set()
    for row in nbr_indices:
        for idx in row:
            if int(idx) not in region_set:
                outside.add(int(idx))
    return outside


def _sample_boundary(outside: set[int], n_overlap: int) -> list[int]:
    """Return a deterministic random sample of up to n_overlap outside indices."""
    outside_list = sorted(outside)
    if len(outside_list) <= n_overlap:
        return outside_list
    return np.random.default_rng(0).choice(
        outside_list, size=n_overlap, replace=False
    ).tolist()


def _expand_region(
    region_indices: np.ndarray,
    full_data: np.ndarray,
    faiss_index,
    overlap_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand *region_indices* by adding boundary neighbours from *full_data*.

    Parameters
    ----------
    region_indices : (n_region,) core indices.
    full_data : (n, d) full dataset.
    faiss_index : FAISS index over *full_data*.
    overlap_factor : fraction of n_region to add as overlap.

    Returns
    -------
    tuple (region_indices, expanded_indices)
        region_indices is returned unchanged; expanded_indices is a sorted
        superset that includes overlap boundary points.
    """
    region_set = set(region_indices.tolist())
    region_pts = full_data[region_indices]
    _dists, nbr_indices = nn_utils.query_knn(faiss_index, region_pts, k=10)
    outside = _collect_outside_neighbours(nbr_indices, region_set)
    n_overlap = max(1, int(overlap_factor * len(region_indices)))
    chosen = _sample_boundary(outside, n_overlap)
    expanded = np.array(sorted(region_set | set(chosen)), dtype=np.intp)
    return region_indices, expanded


def _build_local_eigenbasis(
    expanded_data: np.ndarray,
    graph_builder,
    reference_basis: EigenBasis,
) -> AlignedBasis:
    """Build and align a local GFT eigenbasis for *expanded_data*.

    Parameters
    ----------
    expanded_data : (n_expanded, d) data for this patch.
    graph_builder : module exposing build(data) -> FeatureGraph.
    reference_basis : global EigenBasis for alignment.

    Returns
    -------
    AlignedBasis aligned to *reference_basis*.
    """
    local_graph = graph_builder.build(expanded_data)
    engine = GraphFourierEngine(local_graph)
    local_basis = engine.basis
    return align_to_reference(local_basis, reference_basis)


def _injectivity_score(vectors: np.ndarray, data: np.ndarray) -> float:
    """Spearman rank correlation of pairwise distances (ambient vs chart).

    Parameters
    ----------
    vectors : (d, k) basis vectors defining the chart map.
    data : (n, d) data points.

    Returns
    -------
    float in [-1, 1]; higher means more injective.
    """
    n = data.shape[0]
    chart_map = _define_chart_map(vectors)

    coords = _compute_coordinates(data, chart_map)

    rng = np.random.default_rng(1)
    n_pairs = min(500, n * (n - 1) // 2)
    if n >= 2:
        i_idx = rng.integers(0, n, size=n_pairs)
        j_idx = rng.integers(0, n, size=n_pairs)
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % n

        amb_dists = np.linalg.norm(data[i_idx] - data[j_idx], axis=1)
        chart_dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
        corr, _ = spearmanr(amb_dists, chart_dists)
        return float(corr) if not np.isnan(corr) else 0.0
    return 0.0


def _select_basis_adaptive(
    aligned_basis: AlignedBasis,
    region_data: np.ndarray,
    intrinsic_dim: int,
) -> list[int]:
    """Select *intrinsic_dim* eigenvectors that maximise injectivity.

    Starts from the top-*intrinsic_dim* eigenvectors by eigenvalue (ascending
    order → last columns), then tries up to 10 swap candidates.

    Parameters
    ----------
    aligned_basis : AlignedBasis with eigenvectors (d, k).
    region_data : (n_region, d) data for injectivity scoring.
    intrinsic_dim : number of basis vectors to select.

    Returns
    -------
    list[int] of length *intrinsic_dim*; column indices into aligned_basis.eigenvectors.
    """
    vecs = aligned_basis.eigenvectors   # (d, k)
    k = vecs.shape[1]
    # Start with highest-eigenvalue columns (descending: last intrinsic_dim).
    best_indices = list(range(k - intrinsic_dim, k))
    best_score = _injectivity_score(vecs[:, best_indices], region_data)

    remaining = [i for i in range(k) if i not in best_indices]
    attempts = 0
    for swap_out_pos in range(intrinsic_dim):
        for candidate in remaining:
            if attempts >= 10:
                break
            trial = best_indices.copy()
            trial[swap_out_pos] = candidate
            score = _injectivity_score(vecs[:, trial], region_data)
            attempts += 1
            if score > best_score:
                best_score = score
                best_indices = trial
                remaining = [i for i in range(k) if i not in best_indices]
                break
        if attempts >= 10:
            break

    return best_indices


def _define_chart_map(selected_vectors: np.ndarray) -> Callable:
    """Return a callable that projects ambient points to chart coordinates.

    Parameters
    ----------
    selected_vectors : (d, intrinsic_dim) orthonormal chart basis.

    Returns
    -------
    Callable (d,) → (intrinsic_dim,).
    """
    basis_vecs = selected_vectors  # capture by reference

    def chart_map(p: np.ndarray) -> np.ndarray:
        return basis_vecs.T @ p

    return chart_map


def _compute_chart_inverse(selected_vectors: np.ndarray) -> Callable:
    """Return a callable that reconstructs ambient points from chart coordinates.

    Parameters
    ----------
    selected_vectors : (d, intrinsic_dim) orthonormal chart basis.

    Returns
    -------
    Callable (intrinsic_dim,) → (d,).
    """
    basis_vecs = selected_vectors

    def chart_inverse(z: np.ndarray) -> np.ndarray:
        return basis_vecs @ z

    return chart_inverse


def _compute_coordinates(data: np.ndarray, chart_map: Callable) -> np.ndarray:
    """Apply *chart_map* to every row of *data*.

    Parameters
    ----------
    data : (n, d) data matrix.
    chart_map : callable (d,) → (intrinsic_dim,).

    Returns
    -------
    np.ndarray : (n, intrinsic_dim) chart coordinates.
    """
    return np.stack([chart_map(row) for row in data])
