"""Tests for backend/services/manifold_projector.py."""
from __future__ import annotations

import json

from fastapi import HTTPException
import numpy as np
import pytest

from backend.services.manifold_projector import ProjectionResult


# ---------------------------------------------------------------------------
# Array shapes & basic correctness
# ---------------------------------------------------------------------------

class TestProjectionShapes:
    """Verify shapes of all arrays in ProjectionResult."""

    def test_positions_shape(self, projection):
        """Verify all_positions has shape (n_points, 3)."""
        assert projection.all_positions.shape == (1000, 3)

    def test_is_anomaly_shape(self, projection):
        """Verify is_anomaly flag count matches planted anomalies."""
        assert projection.is_anomaly.shape == (1000,)
        assert int(projection.is_anomaly.sum()) == 60

    def test_scores_shape(self, projection):
        """Verify scores array has one entry per point."""
        assert projection.scores.shape == (1000,)

    def test_point_ids_arange(self, projection):
        """Verify point_ids is a consecutive 0..n-1 array."""
        assert projection.point_ids.shape == (1000,)
        assert projection.point_ids[0] == 0
        assert projection.point_ids[-1] == 999

    def test_normal_count(self, projection):
        """Verify the count of non-anomalous points."""
        assert len(projection.meta.normal_indices) == 940

    def test_three_clusters(self, projection):
        """Verify three clusters are detected."""
        assert len(projection.clusters) == 3

    def test_axis_labels_length(self, projection):
        """Verify axis_labels and axes each have 3 entries."""
        assert len(projection.meta.axis_labels) == 3
        assert len(projection.meta.axes) == 3


# ---------------------------------------------------------------------------
# Axis selection
# ---------------------------------------------------------------------------

class TestAxisSelection:
    """Verify GFT axis selection logic."""

    def test_dc_component_not_selected(self, projection):
        """Verify the DC component (index 0) is never chosen as an axis."""
        assert (
            0 not in projection.meta.axes
        ), "DC component (index 0) must not be a projection axis."


    def test_top3_axes_are_high_power_features(self, projector, run_dir):
        """Verify amplified features 1-3 are selected as top axes."""
        power = np.load(run_dir / "power_spectrum.npy")
        axes = projector.select_top3_axes(power)
        # Features 1, 2, 3 were amplified — they should be the top-3.
        assert set(axes) == {1, 2, 3}

    def test_axes_sorted(self, projection):
        """Verify projection axes are returned in ascending order."""
        assert projection.meta.axes == sorted(projection.meta.axes)


# ---------------------------------------------------------------------------
# Cluster reprojection
# ---------------------------------------------------------------------------

class TestClusterReprojection:
    """Verify per-cluster reprojection decisions."""

    def _get(self, projection, cid):
        return next(c for c in projection.clusters if c.cluster_id == cid)

    def test_separated_cluster0_not_reprojected(self, projection):
        """Verify separated cluster 0 is not reprojected."""
        assert not self._get(projection, 0).reprojected

    def test_overlapping_cluster1_reprojected(self, projection):
        """Verify overlapping cluster 1 is reprojected."""
        assert self._get(projection, 1).reprojected

    def test_separated_cluster2_not_reprojected(self, projection):
        """Verify separated cluster 2 is not reprojected."""
        assert not self._get(projection, 2).reprojected

    def test_divergent_axes_include_feature20(self, projection):
        """Verify feature 20 is among divergent axes for cluster 1."""
        c1 = self._get(projection, 1)
        assert c1.divergent_axes is not None
        assert 20 in c1.divergent_axes

    def test_divergent_axes_differ_from_projection_axes(self, projection):
        """Verify divergent axes are different from the global projection axes."""
        c1 = self._get(projection, 1)
        assert set(c1.divergent_axes) != set(projection.meta.axes)

    def test_reprojected_positions_shape(self, projection):
        """Verify reprojected cluster positions have shape (n_cluster, 3)."""
        assert self._get(projection, 1).positions.shape == (20, 3)

    def test_non_reprojected_positions_shape(self, projection):
        """Verify non-reprojected cluster positions have shape (n_cluster, 3)."""
        assert self._get(projection, 0).positions.shape == (20, 3)


# ---------------------------------------------------------------------------
# Point detail
# ---------------------------------------------------------------------------

class TestPointDetail:
    """Verify point_detail returns correct per-point data."""

    def test_normal_point_not_flagged(self, projector, run_dir):
        """Verify a normal point has is_anomaly=False and no anomaly_type."""
        d = projector.point_detail(run_dir, 0)
        assert d.is_anomaly is False
        assert d.anomaly_type is None

    def test_anomaly_point_is_flagged(self, projector, run_dir):
        """Verify an anomaly point has is_anomaly=True and score ≈ 0.9."""
        d = projector.point_detail(run_dir, 942)  # inside sep_idx (940-959)
        assert d.is_anomaly is True
        assert abs(d.overall_score - 0.9) < 1e-6

    def test_band_scores_present(self, projector, run_dir):
        """Verify band_0 score is present in point detail."""
        d = projector.point_detail(run_dir, 942)
        assert "band_0" in d.band_scores

    def test_column_names_used(self, projector, run_dir):
        """Verify feature column names appear in original_values."""
        d = projector.point_detail(run_dir, 0)
        assert "feat_0" in d.original_values

    def test_out_of_range_raises(self, projector, run_dir):
        """Verify out-of-range index raises HTTPException."""
        with pytest.raises(HTTPException):
            projector.point_detail(run_dir, 99_999)


# ---------------------------------------------------------------------------
# Missing-file fallbacks
# ---------------------------------------------------------------------------

class TestMissingFiles:
    """Verify graceful fallback when run-dir files are absent."""

    def test_fallback_when_no_coefficients(self, projector, tmp_path):
        """Verify projection returns valid result when coefficients.npy is absent."""
        result = projector.project(tmp_path)
        assert isinstance(result, ProjectionResult)
        assert result.all_positions.shape[1] == 3

    def test_no_cluster_labels_gives_empty_regional(self, projector, tmp_path):
        """Verify missing cluster_labels yields an empty clusters list."""
        rng = np.random.default_rng(0)
        coeff = rng.standard_normal((100, 10))
        np.save(tmp_path / "coefficients.npy", coeff)
        (tmp_path / "anomaly_results.json").write_text(json.dumps({
            "flags":  [True] * 10 + [False] * 90,
            "scores": [0.8] * 10 + [0.0] * 90,
            "types":  ["isolated"] * 10 + ["normal"] * 90,
        }))
        result = projector.project(tmp_path)
        assert result.clusters == []
        assert len(result.meta.isolated_indices) == 10
