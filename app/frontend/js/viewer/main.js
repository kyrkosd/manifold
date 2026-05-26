/* Entry point for the FMAS 3D manifold viewer. */
/* global Scene, Controls, DetailPanel, ManifoldMesh, ClusterMesh, ViewerAPI, PointCloud */
(async () => {
  const params = new URLSearchParams(location.search);
  const runId  = params.get('run');

  if (!runId) {
    document.getElementById('viewer-loading').hidden = true;
    document.getElementById('viewer-error-msg').textContent =
      'No run ID specified. Open this page as /viewer?run=<run_id>.';
    document.getElementById('viewer-error').hidden = false;
    return;
  }

  const canvas = document.getElementById('viewer-canvas');
  Scene.init(canvas);
  Controls.init(canvas);
  Controls.initRaycasting(onPointClick);
  DetailPanel.init(runId);

  // Toolbar wiring.
  document.getElementById('btn-reset-camera').addEventListener('click', () => Controls.resetCamera());
  document.getElementById('btn-toggle-surface').addEventListener('click', function () {
    ManifoldMesh.toggle();
    this.classList.toggle('active');
  });
  document.getElementById('btn-toggle-clusters').addEventListener('click', function () {
    ClusterMesh.toggle();
    this.classList.toggle('active');
  });

  // Surface opacity slider.
  const opacitySlider = document.getElementById('surface-opacity');
  if (opacitySlider) {
    opacitySlider.addEventListener('input', () => {
      ManifoldMesh.setOpacity(opacitySlider.value / 100);
    });
  }

  document.getElementById('viewer-loading-msg').textContent = `Loading run ${runId}…`;

  try {
    const data = await ViewerAPI.getViewerData(runId);

    // Metadata badge.
    document.getElementById('toolbar-meta').textContent =
      `${data.n_points.toLocaleString()} pts · ${data.n_anomalies} anomalies · ` +
      `${data.n_clusters} clusters · axes: ${data.axis_labels.join(', ')}`;

    // Axis labels footer.
    const axisEl = document.getElementById('axis-labels');
    if (axisEl) {
      axisEl.textContent =
        `x: ${data.axis_labels[0]} · y: ${data.axis_labels[1]} · z: ${data.axis_labels[2]}`;
    }

    ManifoldMesh.build(data.surface_vertices, data.surface_faces);
    PointCloud.build(data);
    ClusterMesh.build(data.clusters);

    document.getElementById('viewer-loading').hidden = true;
  } catch (err) {
    document.getElementById('viewer-loading').hidden = true;
    document.getElementById('viewer-error-msg').textContent =
      `Could not load manifold data: ${err.message}`;
    document.getElementById('viewer-error').hidden = false;
  }

  function onPointClick(idx) {
    PointCloud.highlight(idx);
    DetailPanel.show(idx);
  }
})();
