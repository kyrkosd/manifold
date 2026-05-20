# fourier-manifold

**Fourier Manifold Anomaly System (FMAS)** — graph-spectral anomaly detection via manifold learning and Fourier decomposition.

## Overview

FMAS detects anomalies in high-dimensional datasets by:
1. Building a k-NN graph over the data points
2. Computing the Graph Fourier Transform on that graph
3. Learning a local manifold atlas of overlapping charts
4. Scoring anomalies by reconstruction residual across frequency bands

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```
pipeline.py          # Top-level orchestration
config.py            # Global configuration
common/              # Shared utilities and types
ingestion/           # Data loading, validation, normalization
structure/           # Graph building and dimensionality estimation
fourier/             # Graph Fourier Transform and spectral analysis
manifold/            # Manifold atlas: charts, alignment, tangent spaces
anomaly/             # Reconstruction, residual scoring, thresholding
reporting/           # Report assembly and export
tests/               # Pytest suite
```
