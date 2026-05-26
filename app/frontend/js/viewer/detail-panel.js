/* Side panel for point inspection in the FMAS 3D manifold viewer. */
/* global ViewerAPI, PointCloud */
/* exported DetailPanel */
const DetailPanel = {
  _runId: null,
  _panel: null,
  _isOpen: false,

  init(runId) {
    this._runId = runId;
    this._panel = document.getElementById('detail-panel');
    document.getElementById('detail-close').addEventListener('click', () => this.hide());
  },

  async show(pointIndex) {
    this._panel.hidden = false;
    requestAnimationFrame(() => this._panel.classList.add('open'));
    this._isOpen = true;

    document.getElementById('detail-index').textContent = `#${pointIndex.toLocaleString()}`;
    document.getElementById('detail-body').innerHTML =
      '<div class="detail-loading">Loading…</div>';

    try {
      const d = await ViewerAPI.getPointDetail(this._runId, pointIndex);
      this._render(d);
    } catch (err) {
      document.getElementById('detail-body').innerHTML =
        `<div style="color:var(--danger);font-size:12px">${err.message}</div>`;
    }
  },

  hide() {
    this._panel.classList.remove('open');
    this._panel.addEventListener('transitionend', () => {
      if (!this._isOpen) this._panel.hidden = true;
    }, { once: true });
    this._isOpen = false;
    PointCloud.unhighlight();
  },

  _render(d) {
    const flagColor = d.is_anomaly ? 'var(--danger)' : 'var(--success)';
    const flagLabel = d.is_anomaly ? '⚠ Anomaly'     : '✓ Normal';

    let html = `
      <div class="detail-row">
        <span class="detail-key">Status</span>
        <span class="detail-val" style="color:${flagColor}">${flagLabel}</span>
      </div>
      <div class="detail-row">
        <span class="detail-key">Score</span>
        <span class="detail-val">${d.overall_score.toFixed(3)}</span>
      </div>`;

    if (d.anomaly_type) {
      html += `<div class="detail-row">
        <span class="detail-key">Type</span>
        <span class="detail-val">${d.anomaly_type}</span>
      </div>`;
    }
    if (d.top_anomalous_band) {
      html += `<div class="detail-row">
        <span class="detail-key">Top band</span>
        <span class="detail-val">${d.top_anomalous_band}</span>
      </div>`;
    }
    html += `<div class="detail-row">
      <span class="detail-key">Chart</span>
      <span class="detail-val">#${d.chart_id} (quality ${d.chart_alignment_quality.toFixed(2)})</span>
    </div>`;
    if (d.cluster_id !== null && d.cluster_id !== undefined) {
      html += `<div class="detail-row">
        <span class="detail-key">Cluster</span>
        <span class="detail-val">#${d.cluster_id}</span>
      </div>`;
    }

    const bands = Object.entries(d.band_scores);
    if (bands.length) {
      html += `<div class="detail-section-head">Band scores</div>`;
      html += this._renderBandBars(d.band_scores, d.top_anomalous_band);
    }

    const feats = Object.entries(d.original_values);
    if (feats.length) {
      html += `<div class="detail-section-head">Feature values</div>`;
      feats.slice(0, 20).forEach(([k, v]) => {
        html += `<div class="detail-row">
          <span class="detail-key">${k}</span>
          <span class="detail-val">${Number(v).toFixed(4)}</span>
        </div>`;
      });
    }

    document.getElementById('detail-body').innerHTML = html;
  },

  _renderBandBars(bandScores, topBand) {
    const entries  = Object.entries(bandScores);
    const maxScore = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);
    return entries.map(([k, v]) => {
      const pct   = (Math.abs(v) / maxScore * 100).toFixed(1);
      const color = k === topBand ? 'var(--danger)' : 'var(--accent)';
      return `<div class="detail-band-row">
        <span class="detail-key">Band ${k}</span>
        <div class="band-track"><div class="band-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="detail-val">${Number(v).toFixed(3)}</span>
      </div>`;
    }).join('');
  },
};
