"""
Public API of the ``common`` package.  Import from here rather than from
individual sub-modules to keep downstream code decoupled from internal layout.
"""
from __future__ import annotations

from .exceptions import (
    AlignmentError,
    ChartError,
    ConvergenceError,
    DimensionalityError,
    FourierManifoldError,
    StructureError,
    ValidationError,
)
from .graph_utils import (
    clustering_coefficient,
    connected_components,
    degree_matrix,
    dijkstra,
    graph_diameter,
    knn_graph,
    laplacian,
    normalized_laplacian,
    threshold_graph,
)
from .logging import get_logger, setup_logging
from .math_utils import (
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
from .nn_utils import (
    batch_knn,
    build_faiss_index,
    query_knn,
    query_radius,
    rebuild_index,
)
from .types import (
    AlignmentQuality,
    AnomalyType,
    BoundaryType,
    EigenBasis,
    FeatureGraph,
    FourierType,
    FrequencyBand,
    RelationshipType,
    Severity,
    SpatialType,
    StructureType,
)
from .validation import (
    validate_array,
    validate_enum,
    validate_positive,
    validate_positive_definite,
    validate_range,
    validate_smoothness,
    validate_symmetric,
)

__all__ = [
    # types
    "StructureType",
    "FourierType",
    "AnomalyType",
    "SpatialType",
    "RelationshipType",
    "Severity",
    "BoundaryType",
    "AlignmentQuality",
    "FeatureGraph",
    "FrequencyBand",
    "EigenBasis",
    # exceptions
    "FourierManifoldError",
    "ValidationError",
    "StructureError",
    "ChartError",
    "ConvergenceError",
    "DimensionalityError",
    "AlignmentError",
    # logging
    "get_logger",
    "setup_logging",
    # math_utils
    "eigendecompose",
    "project_onto_subspace",
    "orthogonal_complement",
    "gram_schmidt",
    "numerical_jacobian",
    "numerical_hessian",
    "safe_normalize",
    "distance_matrix",
    "smooth_check",
    "solve_ode",
    "matrix_log",
    "svd_wrapper",
    # graph_utils
    "laplacian",
    "normalized_laplacian",
    "degree_matrix",
    "connected_components",
    "knn_graph",
    "threshold_graph",
    "dijkstra",
    "graph_diameter",
    "clustering_coefficient",
    # nn_utils
    "build_faiss_index",
    "query_knn",
    "query_radius",
    "batch_knn",
    "rebuild_index",
    # validation
    "validate_array",
    "validate_positive",
    "validate_range",
    "validate_enum",
    "validate_symmetric",
    "validate_positive_definite",
    "validate_smoothness",
]
