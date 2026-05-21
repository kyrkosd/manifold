"""
Graph Fourier Transform (GFT): computes the normalised graph Laplacian,
its full or truncated eigenbasis, and projects node-feature signals into
the graph spectral domain and back.
"""
from __future__ import annotations

import numpy as np

from common import graph_utils, math_utils
from common.types import EigenBasis, FeatureGraph


class GraphFourierEngine:
    """Encapsulates the GFT eigenbasis for a fixed feature graph.

    Parameters
    ----------
    graph : FeatureGraph whose (d, d) adjacency defines the spectral basis.
    """

    def __init__(self, graph: FeatureGraph) -> None:
        """Build the Laplacian eigenbasis from *graph*.

        Parameters
        ----------
        graph : FeatureGraph with (d, d) adjacency matrix.
        """
        # Ensure float64 and guard against disconnected inputs.
        adj = self._handle_disconnected(graph.adjacency)
        lap = self._compute_laplacian(adj)
        self.basis: EigenBasis = self._eigendecompose(lap)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Forward GFT: project each data row into the graph spectral domain.

        Parameters
        ----------
        data : (n, d) data matrix; each row is one d-dimensional observation.

        Returns
        -------
        np.ndarray : (n, d) array of spectral coefficients.
        """
        return self._apply_gft(data, self.basis)

    def transform_local(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Forward GFT on the subset of rows selected by boolean *mask*.

        Parameters
        ----------
        data : (n, d) full data matrix.
        mask : (n,) boolean selection array; True rows are transformed.

        Returns
        -------
        np.ndarray : (mask.sum(), d) spectral coefficients.
        """
        return self._apply_gft(data[mask], self.basis)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """Inverse GFT: reconstruct spatial signals from spectral coefficients.

        Parameters
        ----------
        coeffs : (n, d) spectral coefficients.

        Returns
        -------
        np.ndarray : (n, d) reconstructed data; equals the original when
            all d spectral components are used.
        """
        # Reconstruction: x = F * U^T where F = coefficients, U = eigenvectors.
        return coeffs @ self.basis.eigenvectors.T

    def _compute_laplacian(self, adjacency: np.ndarray) -> np.ndarray:
        # L = D − W; symmetric positive semi-definite; row sums are zero.
        return graph_utils.laplacian(adjacency)

    def _eigendecompose(self, laplacian: np.ndarray) -> EigenBasis:
        # eigh returns eigenvalues / vectors in ascending eigenvalue order.
        vecs, vals = math_utils.eigendecompose(laplacian)
        return EigenBasis(eigenvectors=vecs, eigenvalues=vals)

    def _apply_gft(self, data: np.ndarray, basis: EigenBasis) -> np.ndarray:
        # Project each row x onto eigenvectors: coeffs = U^T x (rows stay rows).
        return (basis.eigenvectors.T @ data.T).T

    def _handle_disconnected(self, adjacency: np.ndarray) -> np.ndarray:
        # Disconnected graphs yield multiple zero eigenvalues; GFT is still valid
        # because scipy.linalg.eigh always returns an orthonormal eigenbasis.
        return np.asarray(adjacency, dtype=float)
