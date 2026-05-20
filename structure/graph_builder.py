"""
Builds k-NN and epsilon-ball graphs from normalised feature matrices.
Uses FAISS-backed neighbour search and converts distances to edge weights
via Gaussian or cosine kernels, returning a sparse adjacency matrix.
"""
