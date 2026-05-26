/* Pipeline config form handling and launch for the FMAS import interface. */
/* global App, Preview, API */
/* exported Config */
const Config = {
  init() {
    const overlap = document.getElementById('cfg-overlap');
    const display = document.getElementById('overlap-display');

    overlap.addEventListener('input', () => {
      display.textContent = overlap.value + '%';
      App.config.overlap_factor = overlap.value / 100;
    });

    document.getElementById('cfg-n-charts').addEventListener('change',    e => { App.config.n_charts = e.target.value === 'auto' ? 'auto' : +e.target.value; });
    document.getElementById('cfg-threshold').addEventListener('change',    e => { App.config.threshold_method = e.target.value; });
    document.getElementById('cfg-norm').addEventListener('change',         e => { App.config.normalization = e.target.value; });
    document.getElementById('cfg-iterations').addEventListener('input',    e => { App.config.max_iterations = +e.target.value; });
    document.getElementById('cfg-band').addEventListener('change',         e => { App.config.band_method = e.target.value; });

    document.getElementById('btn-launch').addEventListener('click', () => this.launch());
  },

  updateSummary() {
    const q = App.previewData?.quality;
    if (!q) return;

    const launchBtn = document.getElementById('btn-launch');
    const isUnsuitable = q.suitability_level === 'not_suitable';
    launchBtn.disabled = isUnsuitable;
    if (isUnsuitable) {
      launchBtn.title = 'Dataset is not suitable for FMAS analysis. See warnings.';
    } else {
      launchBtn.title = '';
    }

    const rows     = Preview.formatNumber(q.n_rows);
    const features = q.n_numeric_columns;
    const runtime  = Preview._formatRuntime(q.estimated_runtime_seconds);
    document.getElementById('launch-summary').textContent =
      `${rows} rows × ${features} features → est. runtime ${runtime}`;
  },

  getConfig() {
    return {
      n_charts:         App.config.n_charts,
      threshold_method: App.config.threshold_method,
      normalization:    App.config.normalization,
      max_iterations:   App.config.max_iterations,
      overlap_factor:   App.config.overlap_factor,
      band_method:      App.config.band_method,
    };
  },

  validate() {
    const errors = [];
    const cfg = this.getConfig();
    if (cfg.max_iterations <= 0 || cfg.max_iterations > 50) errors.push('Max iterations must be between 1 and 50.');
    if (cfg.overlap_factor <= 0 || cfg.overlap_factor >= 1)  errors.push('Overlap factor must be between 5% and 95%.');
    return { valid: errors.length === 0, errors };
  },

  async launch() {
    const { valid, errors } = this.validate();
    if (!valid) { App.showError(errors.join(' ')); return; }

    App.showLoading('Launching FMAS pipeline…');
    try {
      const res = await API.launchPipeline(App.dataId, this.getConfig());
      App.hideLoading();
      const viewerUrl = res.viewer_url || `/viewer?run=${res.run_id}`;
      const statusMsg = res.status === 'running'
        ? 'Pipeline running.'
        : res.message || 'Data prepared.';
      const bar = document.querySelector('.launch-bar');
      bar.innerHTML = '';
      const msgDiv = document.createElement('div');
      msgDiv.style.cssText = 'padding:4px 0;color:var(--success);font-size:13px';
      const checkIcon = document.createElement('i');
      checkIcon.className = 'ti ti-check';
      msgDiv.appendChild(checkIcon);
      msgDiv.appendChild(document.createTextNode(` ${statusMsg} Run ID: ${res.run_id}.`));
      bar.appendChild(msgDiv);
      const link = document.createElement('a');
      link.href = viewerUrl;
      link.className = 'btn btn-primary btn-sm';
      link.style.cssText = 'margin-top:8px;display:inline-flex;align-items:center;gap:6px;font-size:13px';
      const cubeIcon = document.createElement('i');
      cubeIcon.className = 'ti ti-3d-cube-sphere';
      link.appendChild(cubeIcon);
      link.appendChild(document.createTextNode(' View 3D manifold'));
      bar.appendChild(link);
    } catch (err) {
      App.hideLoading();
      App.showError(err.error || 'Failed to launch pipeline. Please try again.');
    }
  },
};
