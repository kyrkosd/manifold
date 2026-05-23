"""
Builds and manages the global atlas of overlapping charts that jointly
cover the data manifold, including chart assignment, overlap tracking,
and atlas-level consistency checks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import manifold.chart as chart_mod
import structure.graph_builder as graph_builder_mod
from common import nn_utils
from common.exceptions import ChartError
from common.types import EigenBasis
from fourier import SpectralData
from structure.report import StructureReport

log = logging.getLogger(__name__)


@dataclass
class Atlas:
    """Global atlas of overlapping charts covering the data manifold.

    Parameters
    ----------
    charts : ordered list of Chart objects.
    primary_assignments : (n_points,) index of the primary chart for each point.
    overlaps : list of (chart_i, chart_j, shared_indices) triples.
    n_points : total number of data points.
    coverage_pct : fraction of points covered by at least one chart.
    faiss_index : pre-built FAISS index over the full dataset.
    """

    charts: list                                        # list[Chart]
    primary_assignments: np.ndarray                    # (n_points,) int
    overlaps: list[tuple[int, int, np.ndarray]]
    n_points: int
    coverage_pct: float
    faiss_index: Any = field(repr=False)               # faiss.Index


def build(
    data: np.ndarray,
    spectral: SpectralData,
    structure: StructureReport,
    config: dict,
) -> Atlas:
    """Build the atlas by partitioning data into chart regions.

    Parameters
    ----------
    data : (n, d) full dataset.
    spectral : Phase-2 output (GFT coefficients, bands, basis).
    structure : Phase-1 output (intrinsic_dim, graph).
    config : keys: n_charts (int or "auto"), overlap_factor (float),
        max_retries (int).

    Returns
    -------
    Atlas
    """
    n_points = data.shape[0]
    n_charts = config.get("n_charts", "auto")
    overlap_factor = float(config.get("overlap_factor", 0.2))
    max_retries = int(config.get("max_retries", 3))

    faiss_index = nn_utils.build_faiss_index(data)
    regions = _partition_into_regions(data, spectral, structure, n_charts)
    charts = _build_all_charts(
        data, regions, structure, spectral, faiss_index, overlap_factor, max_retries
    )

    covered = set()
    for c in charts:
        covered.update(c.region_indices.tolist())
    coverage_pct = len(covered) / n_points if n_points > 0 else 0.0

    if not _verify_coverage(charts, n_points):
        log.warning("Atlas does not fully cover all %d points (%.1f%% covered).",
                    n_points, 100 * coverage_pct)

    primary_assignments = _assign_primary_charts(charts, n_points)
    overlaps = _find_overlaps(charts)

    return Atlas(
        charts=charts,
        primary_assignments=primary_assignments,
        overlaps=overlaps,
        n_points=n_points,
        coverage_pct=coverage_pct,
        faiss_index=faiss_index,
    )


def _partition_into_regions(
    data: np.ndarray,
    spectral: SpectralData,
    structure: StructureReport,
    n_charts: int | str,
) -> list[np.ndarray]:
    """Partition data into k regions using mini-batch k-means on low-frequency GFT.

    Parameters
    ----------
    data : (n, d) full dataset.
    spectral : SpectralData with (n, d) GFT coefficients.
    structure : StructureReport with intrinsic_dim.
    n_charts : number of regions, or "auto".

    Returns
    -------
    list of (n_region,) index arrays, one per region.
    """
    n_points = data.shape[0]
    if n_charts == "auto":
        k = min(20, max(2, int(np.sqrt(n_points / 100))))
    else:
        k = max(2, int(n_charts))

    # Low-frequency GFT columns capture global manifold structure.
    n_low = min(2 * structure.intrinsic_dim, spectral.coefficients.shape[1])
    low_freq = spectral.coefficients[:, :n_low]

    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, max_iter=300)
    labels = km.fit_predict(low_freq)

    regions = []
    for ci in range(k):
        mask = labels == ci
        if mask.sum() > 0:
            regions.append(np.where(mask)[0].astype(np.intp))
    return regions


def _build_all_charts(
    data: np.ndarray,
    regions: list[np.ndarray],
    structure: StructureReport,
    spectral: SpectralData,
    faiss_index,
    overlap_factor: float,
    max_retries: int,
) -> list:
    """Build a Chart for each region, with retry-on-failure splitting.

    The first successfully built chart's local basis becomes the Procrustes
    reference for all subsequent charts.

    Parameters
    ----------
    data : (n, d) full dataset.
    regions : list of (n_region,) index arrays.
    structure : StructureReport.
    spectral : SpectralData (provides global EigenBasis for initial reference).
    faiss_index : pre-built FAISS index.
    overlap_factor : passed to chart.build().
    max_retries : maximum split-and-retry attempts per region.

    Returns
    -------
    list[Chart]
    """
    reference_basis = spectral.basis
    charts = []

    for region_indices in regions:
        chart = _attempt_build(
            region_indices, data, structure, faiss_index,
            reference_basis, overlap_factor, max_retries,
        )
        if chart is not None:
            charts.append(chart)
            if len(charts) == 1:
                # Align all subsequent charts to the first chart's local basis.
                reference_basis = EigenBasis(
                    eigenvectors=charts[0].local_basis.eigenvectors,
                    eigenvalues=charts[0].local_basis.eigenvalues,
                )

    return charts


def _attempt_build(
    region_indices: np.ndarray,
    data: np.ndarray,
    structure: StructureReport,
    faiss_index,
    reference_basis: EigenBasis,
    overlap_factor: float,
    max_retries: int,
):
    """Try to build a chart for *region_indices*, splitting on ChartError."""
    pending = [region_indices]
    retries = 0

    while pending and retries <= max_retries:
        curr = pending.pop(0)
        if len(curr) < 4:
            retries += 1
            continue
        try:
            return chart_mod.build(
                region_data=data[curr],
                region_indices=curr,
                full_data=data,
                graph_builder=graph_builder_mod,
                intrinsic_dim=structure.intrinsic_dim,
                reference_basis=reference_basis,
                faiss_index=faiss_index,
                overlap_factor=overlap_factor,
            )
        except ChartError:
            retries += 1
            log.debug("ChartError on region of size %d; splitting (retry %d/%d).",
                      len(curr), retries, max_retries)
            if len(curr) >= 8:
                km = KMeans(n_clusters=2, random_state=0, n_init=5)
                labels = km.fit_predict(data[curr])
                for k in range(2):
                    sub = curr[labels == k]
                    if len(sub) >= 4:
                        pending.append(sub)

    log.warning("Could not build chart for region of %d points after %d retries.",
                len(region_indices), max_retries)
    return None


def _verify_coverage(charts: list, n_points: int) -> bool:
    """Return True when every point index appears in at least one chart."""
    covered = set()
    for c in charts:
        covered.update(c.region_indices.tolist())
    return len(covered) >= n_points


def _assign_primary_charts(charts: list, n_points: int) -> np.ndarray:
    """For each point, assign the chart with the highest alignment score.

    Parameters
    ----------
    charts : list[Chart].
    n_points : total number of data points.

    Returns
    -------
    np.ndarray : (n_points,) integer chart indices.
    """
    assignments = np.zeros(n_points, dtype=int)
    best_scores = np.full(n_points, -np.inf)

    for ci, chart in enumerate(charts):
        score = chart.local_basis.alignment_score
        for idx in chart.region_indices:
            if score > best_scores[idx]:
                best_scores[idx] = score
                assignments[idx] = ci

    return assignments


def _find_overlaps(
    charts: list,
) -> list[tuple[int, int, np.ndarray]]:
    """Find pairs of charts with shared region points.

    Parameters
    ----------
    charts : list[Chart].

    Returns
    -------
    list of (i, j, shared_indices) for every chart pair with non-empty overlap.
    """
    index_sets = [set(c.region_indices.tolist()) for c in charts]
    overlaps = []
    for i in range(len(charts)):
        for j in range(i + 1, len(charts)):
            shared = index_sets[i] & index_sets[j]
            if shared:
                overlaps.append((i, j, np.array(sorted(shared), dtype=np.intp)))
    return overlaps
