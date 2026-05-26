"""Tests for backend/services/manifold_projector.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.manifold_projector import ManifoldProjector, ProjectionResult


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path) -> Path:
    """Synthetic FMAS run: 1000 pts × 50 features, 3 planted clusters.

    Cluster layout:
      0 (indices 940-959): separated — pushed far in positive top-3 direction
      1 (indices 960-979): overlapping in top-3 space; diverges in feature 20
      2 (indices 980-999): separated — pushed far in negative top-3 direction
    """
    rng = np.random.default_rng(42)
    n_pts, n_feat = 1000, 50

    coeff = rng.standard_normal((n_pts, n_feat)) * 0.5
    # Amplify features 1-3 so they dominate the power spectrum → become top-3 axes.
    coeff[:, 1:4] += rng.standard_normal((n_pts, 3)) * 2.0

    sep_idx     = list(range(940, 960))   # cluster 0
    overlap_idx = list(range(960, 980))   # cluster 1
    sep2_idx    = list(range(980, 1000))  # cluster 2

    # ±50 offsets give features 1-3 power ≈ 104, beating feature 20's power ≈ 50.
    coeff[sep_idx,    1:4] += 50.0   # far positive
    coeff[overlap_idx, 20] += 50.0   # diverges only in feature 20, stays normal in 1-3
    coeff[sep2_idx,   1:4] -= 50.0   # far negative

    power = np.mean(np.abs(coeff) ** 2, axis=0)

    all_anomaly = sep_idx + overlap_idx + sep2_idx
    flags  = [i in set(all_anomaly) for i in range(n_pts)]
    scores = [0.9 if f else 0.0 for f in flags]
    types  = ["regional" if f else "normal" for f in flags]

    cluster_labels = np.full(n_pts, -1, dtype=int)
    for i in sep_idx:     cluster_labels[i] = 0
    for i in overlap_idx: cluster_labels[i] = 1
    for i in sep2_idx:    cluster_labels[i] = 2

    anomaly = {
        "flags":  flags,
        "scores": scores,
        "types":  types,
        "per_point_band_scores": {
            "band_0": scores,
            "band_1": [s * 0.5 for s in scores],
        },
    }
    manifold = {
        "n_charts": 2,
        "chart_assignments": [i % 2 for i in range(n_pts)],
        "alignment_qualities": [0.95, 0.90],
        "intrinsic_dim": 2,
    }

    np.save(tmp_path / "coefficients.npy",   coeff)
    np.save(tmp_path / "power_spectrum.npy",  power)
    np.save(tmp_path / "cluster_labels.npy",  cluster_labels)
    (tmp_path / "anomaly_results.json").write_text(json.dumps(anomaly))
    (tmp_path / "manifold_info.json").write_text(json.dumps(manifold))
    (tmp_path / "column_names.json").write_text(
        json.dumps([f"feat_{i}" for i in range(n_feat)])
    )
    return tmp_path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return _make_run_dir(tmp_path_factory.mktemp("run"))


@pytest.fixture(scope="module")
def projector():
    return ManifoldProjector()


@pytest.fixture(scope="module")
def projection(projector, run_dir):
    return projector.project(run_dir)


# ---------------------------------------------------------------------------
# Array shapes & basic correctness
# ---------------------------------------------------------------------------

class TestProjectionShapes:
    def test_positions_shape(self, projection):
        assert projection.all_positions.shape == (1000, 3)

    def test_is_anomaly_shape(self, projection):
        assert projection.is_anomaly.shape == (1000,)
        assert int(projection.is_anomaly.sum()) == 60

    def test_scores_shape(self, projection):
        assert projection.scores.shape == (1000,)

    def test_point_ids_arange(self, projection):
        assert projection.point_ids.shape == (1000,)
        assert projection.point_ids[0] == 0
        assert projection.point_ids[-1] == 999

    def test_normal_count(self, projection):
        assert len(projection.normal_indices) == 940

    def test_three_clusters(self, projection):
        assert len(projection.clusters) == 3

    def test_axis_labels_length(self, projection):
        assert len(projection.axis_labels) == 3
        assert len(projection.axes) == 3


# ---------------------------------------------------------------------------
# Axis selection
# ---------------------------------------------------------------------------

class TestAxisSelection:
    def test_dc_component_not_selected(self, projection):
        assert 0 not in projection.axes, "DC component (index 0) must not be a projection axis."

    def test_top3_axes_are_high_power_features(self, projector, run_dir):
        power = np.load(run_dir / "power_spectrum.npy")
        axes = projector.select_top3_axes(power)
        # Features 1, 2, 3 were amplified — they should be the top-3.
        assert set(axes) == {1, 2, 3}

    def test_axes_sorted(self, projection):
        assert projection.axes == sorted(projection.axes)


# ---------------------------------------------------------------------------
# Cluster reprojection
# ---------------------------------------------------------------------------

class TestClusterReprojection:
    def _get(self, projection, cid):
        return next(c for c in projection.clusters if c.cluster_id == cid)

    def test_separated_cluster0_not_reprojected(self, projection):
        assert not self._get(projection, 0).reprojected

    def test_overlapping_cluster1_reprojected(self, projection):
        assert self._get(projection, 1).reprojected

    def test_separated_cluster2_not_reprojected(self, projection):
        assert not self._get(projection, 2).reprojected

    def test_divergent_axes_include_feature20(self, projection):
        c1 = self._get(projection, 1)
        assert c1.divergent_axes is not None
        assert 20 in c1.divergent_axes

    def test_divergent_axes_differ_from_projection_axes(self, projection):
        c1 = self._get(projection, 1)
        assert set(c1.divergent_axes) != set(projection.axes)

    def test_reprojected_positions_shape(self, projection):
        assert self._get(projection, 1).positions.shape == (20, 3)

    def test_non_reprojected_positions_shape(self, projection):
        assert self._get(projection, 0).positions.shape == (20, 3)


# ---------------------------------------------------------------------------
# Point detail
# ---------------------------------------------------------------------------

class TestPointDetail:
    def test_normal_point_not_flagged(self, projector, run_dir):
        d = projector.point_detail(run_dir, 0)
        assert d.is_anomaly is False
        assert d.anomaly_type is None

    def test_anomaly_point_is_flagged(self, projector, run_dir):
        d = projector.point_detail(run_dir, 942)  # inside sep_idx (940-959)
        assert d.is_anomaly is True
        assert abs(d.overall_score - 0.9) < 1e-6

    def test_band_scores_present(self, projector, run_dir):
        d = projector.point_detail(run_dir, 942)
        assert "band_0" in d.band_scores

    def test_column_names_used(self, projector, run_dir):
        d = projector.point_detail(run_dir, 0)
        assert "feat_0" in d.original_values

    def test_out_of_range_raises(self, projector, run_dir):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            projector.point_detail(run_dir, 99_999)


# ---------------------------------------------------------------------------
# Missing-file fallbacks
# ---------------------------------------------------------------------------

class TestMissingFiles:
    def test_fallback_when_no_coefficients(self, projector, tmp_path):
        result = projector.project(tmp_path)
        assert isinstance(result, ProjectionResult)
        assert result.all_positions.shape[1] == 3

    def test_no_cluster_labels_gives_empty_regional(self, projector, tmp_path):
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
        assert len(result.isolated_indices) == 10
