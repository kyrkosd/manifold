"""
Nearest-neighbour search utilities wrapping FAISS.  Provides index
construction, k-NN and radius queries, batched search for large datasets,
and index rebuilding when the underlying data changes.
"""
from __future__ import annotations

import numpy as np
import faiss

_SMALL_DATA_THRESHOLD = 10_000
_DEFAULT_NPROBE = 10


def build_faiss_index(data: np.ndarray, metric: str = "L2") -> faiss.Index:
    """Build a FAISS index from *data*.

    Uses ``IndexFlatL2`` for n < 10 000 and ``IndexIVFFlat`` (approximate)
    for larger datasets.

    Parameters
    ----------
    data:
        (n, d) array of vectors to index.  Converted to ``float32`` internally.
    metric:
        Distance metric.  Only ``"L2"`` is currently supported.

    Returns
    -------
    faiss.Index
        Populated FAISS index ready for queries.

    Raises
    ------
    ValueError
        If *metric* is not ``"L2"``.
    """
    if metric != "L2":
        raise ValueError(f"Unsupported metric '{metric}'. Only 'L2' is supported.")
    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data_f32.shape
    if n < _SMALL_DATA_THRESHOLD:
        index: faiss.Index = faiss.IndexFlatL2(d)
    else:
        nlist = max(1, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatL2(d)
        ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
        ivf.train(data_f32)
        ivf.nprobe = _DEFAULT_NPROBE
        index = ivf
    index.add(data_f32)
    return index


def query_knn(
    index: faiss.Index,
    points: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search for the *k* nearest neighbours of each row in *points*.

    Parameters
    ----------
    index:
        A trained and populated FAISS index.
    points:
        (q, d) query vectors.
    k:
        Number of neighbours to return per query.

    Returns
    -------
    distances : np.ndarray
        (q, k) squared-L2 distances.
    indices : np.ndarray
        (q, k) integer indices into the indexed dataset.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    distances, indices = index.search(pts, k)
    return distances, indices


def query_radius(
    index: faiss.Index,
    points: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    """Return all neighbours within *radius* Euclidean distance of each point.

    Parameters
    ----------
    index:
        A FAISS index that supports ``range_search`` (e.g. ``IndexFlatL2``).
    points:
        (q, d) query vectors.
    radius:
        Search radius in Euclidean distance units.  Internally squared to
        match FAISS's squared-L2 threshold convention.

    Returns
    -------
    list[np.ndarray]
        Length-q list; each element is an integer array of neighbour indices.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    lims, _D, I = index.range_search(pts, radius ** 2)
    return [I[lims[i] : lims[i + 1]] for i in range(len(pts))]


def batch_knn(
    index: faiss.Index,
    data: np.ndarray,
    k: int,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """k-NN search over all rows of *data* in memory-bounded batches.

    Parameters
    ----------
    index:
        A trained and populated FAISS index.
    data:
        (n, d) query matrix.
    k:
        Number of neighbours per query.
    batch_size:
        Number of rows to process per batch.

    Returns
    -------
    distances : np.ndarray
        (n, k) squared-L2 distances.
    indices : np.ndarray
        (n, k) neighbour indices.
    """
    n = data.shape[0]
    all_distances = np.empty((n, k), dtype=np.float32)
    all_indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        d, i = query_knn(index, data[start:end], k)
        all_distances[start:end] = d
        all_indices[start:end] = i
    return all_distances, all_indices


def rebuild_index(index: faiss.Index, new_data: np.ndarray) -> faiss.Index:
    """Clear *index* and re-populate it with *new_data*.

    Parameters
    ----------
    index:
        An existing FAISS index with the same dimensionality as *new_data*.
    new_data:
        (n, d) replacement dataset.

    Returns
    -------
    faiss.Index
        The same index object, cleared and re-populated.
    """
    index.reset()
    data_f32 = np.ascontiguousarray(new_data, dtype=np.float32)
    index.add(data_f32)
    return index
