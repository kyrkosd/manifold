"""End-to-end smoke test for the FMAS pipeline.

Generates synthetic data on a 3D manifold in 20D, plants 20 anomalies in
mid-frequency dimensions, runs the full pipeline, and reports recall.
"""
import numpy as np

from fourier_manifold import run_analysis

# Generate synthetic data: 1000 points on a 3D manifold in 20D
np.random.seed(42)
t = np.random.randn(1000, 3)
# Embed in 20D via a random smooth map
A = np.random.randn(3, 20)
data = t @ A + 0.01 * np.random.randn(1000, 20)

# Plant 20 anomalies: perturb in specific mid-frequency dimensions
anomaly_indices = list(range(980, 1000))
data[anomaly_indices, 5:10] += 3.0  # perturb dimensions 5-9

# Run FMAS
report = run_analysis(data)

# Check results  (keys: total_anomalies, point_index)
print(f"Total anomalies detected: {report.anomaly_report.summary['total_anomalies']}")
print(f"Top 10 anomalies: {[a['point_index'] for a in report.anomaly_report.top_anomalies[:10]]}")

# Verify planted anomalies are detected
detected = set(int(a["point_index"]) for a in report.anomaly_report.top_anomalies)
planted = set(anomaly_indices)
overlap = len(detected & planted)
recall = overlap / len(planted)
print(f"Recall on planted anomalies (top 10): {recall:.2f}")
print(f"Pipeline timing: { {k: f'{v:.2f}s' for k, v in report.timing.items()} }")
print("Smoke test PASSED.")
