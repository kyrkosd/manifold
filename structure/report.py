"""
Assembles the StructureReport that summarises Phase 1 findings and tells
downstream phases which Fourier engine to use.  For the MVP, the detected
type is always GRAPH and the recommended engine is always GRAPH_FOURIER.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.types import FeatureGraph, FourierType, StructureType


@dataclass
class StructureReport:
    """Summary produced by the Structure Discovery phase.

    Parameters
    ----------
    type : detected topological character of the data manifold.
    intrinsic_dim : consensus estimate of the intrinsic dimension.
    graph : feature graph (None when structure is not graph-based).
    ordering : temporal or spatial ordering; always None for MVP.
    periodicity : True if periodic structure was detected; always False for MVP.
    recommended_fourier : Fourier engine most appropriate for this structure.
    confidence : confidence in the detected structure type, in [0, 1].
    """

    # Topological type detected by the structure discovery phase.
    type: StructureType
    # Estimated number of intrinsic degrees of freedom.
    intrinsic_dim: int
    # Feature-level graph; None when structure_type is LINEAR or GRID.
    graph: FeatureGraph | None
    # Ordering information (e.g. time index); reserved for non-GRAPH types.
    ordering: None
    # Periodicity flag; reserved for Phase 1 extension beyond MVP.
    periodicity: bool
    # Fourier engine selected based on the detected structure type.
    recommended_fourier: FourierType
    # Confidence score in [0, 1]; 1.0 for the deterministic MVP path.
    confidence: float


def build(graph: FeatureGraph, dim_result: int) -> StructureReport:
    """Assemble a StructureReport from graph-builder and dimensionality results.

    For the MVP, always assigns GRAPH type and GRAPH_FOURIER engine.
    Ordering and periodicity analysis are stubs returning sentinel values.

    Parameters
    ----------
    graph : feature graph from ``graph_builder.build()``.
    dim_result : intrinsic dimension from ``dimensionality.estimate()``.

    Returns
    -------
    StructureReport
    """
    return StructureReport(
        type=StructureType.GRAPH,
        intrinsic_dim=dim_result,
        graph=graph,
        ordering=None,         # ordering analysis: not implemented in MVP
        periodicity=False,     # periodicity detection: not implemented in MVP
        recommended_fourier=FourierType.GRAPH_FOURIER,
        confidence=1.0,
    )
