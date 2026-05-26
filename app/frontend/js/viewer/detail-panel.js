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
      const body = document.getElementById('detail-body');
      body.innerHTML = '';
      const errDiv = document.createElement('div');
      errDiv.style.cssText = 'color:var(--danger);font-size:12px';
      errDiv.textContent = err.message;
      body.appendChild(errDiv);
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
    const frag = document.createDocumentFragment();

    const statusVal = document.createElement('span');
    statusVal.className = 'detail-val';
    statusVal.style.color = flagColor;
    statusVal.textContent = flagLabel;
    frag.appendChild(this._row('Status', statusVal));

    const scoreVal = document.createElement('span');
    scoreVal.className = 'detail-val';
    scoreVal.textContent = d.overall_score.toFixed(3);
    frag.appendChild(this._row('Score', scoreVal));

    if (d.anomaly_type) {
      const typeVal = document.createElement('span');
      typeVal.className = 'detail-val';
      typeVal.textContent = d.anomaly_type;
      frag.appendChild(this._row('Type', typeVal));
    }
    if (d.top_anomalous_band) {
      const bandVal = document.createElement('span');
      bandVal.className = 'detail-val';
      bandVal.textContent = d.top_anomalous_band;
      frag.appendChild(this._row('Top band', bandVal));
    }

    const chartVal = document.createElement('span');
    chartVal.className = 'detail-val';
    chartVal.textContent = `#${d.chart_id} (quality ${d.chart_alignment_quality.toFixed(2)})`;
    frag.appendChild(this._row('Chart', chartVal));

    if (d.cluster_id !== null && d.cluster_id !== undefined) {
      const clusterVal = document.createElement('span');
      clusterVal.className = 'detail-val';
      clusterVal.textContent = `#${d.cluster_id}`;
      frag.appendChild(this._row('Cluster', clusterVal));
    }

    const bands = Object.entries(d.band_scores);
    if (bands.length) {
      const head = document.createElement('div');
      head.className = 'detail-section-head';
      head.textContent = 'Band scores';
      frag.appendChild(head);
      frag.appendChild(this._renderBandBars(d.band_scores, d.top_anomalous_band));
    }

    const feats = Object.entries(d.original_values);
    if (feats.length) {
      const head = document.createElement('div');
      head.className = 'detail-section-head';
      head.textContent = 'Feature values';
      frag.appendChild(head);
      feats.slice(0, 20).forEach(([k, v]) => {
        const featVal = document.createElement('span');
        featVal.className = 'detail-val';
        featVal.textContent = Number(v).toFixed(4);
        frag.appendChild(this._row(k, featVal));
      });
    }

    const body = document.getElementById('detail-body');
    body.innerHTML = '';
    body.appendChild(frag);
  },

  _row(key, valEl) {
    const row = document.createElement('div');
    row.className = 'detail-row';
    const keySpan = document.createElement('span');
    keySpan.className = 'detail-key';
    keySpan.textContent = key;
    row.appendChild(keySpan);
    row.appendChild(valEl);
    return row;
  },

  _renderBandBars(bandScores, topBand) {
    const entries  = Object.entries(bandScores);
    const maxScore = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);
    const frag = document.createDocumentFragment();
    entries.forEach(([k, v]) => {
      const pct   = (Math.abs(v) / maxScore * 100).toFixed(1);
      const color = k === topBand ? 'var(--danger)' : 'var(--accent)';
      const row = document.createElement('div');
      row.className = 'detail-band-row';
      const keySpan = document.createElement('span');
      keySpan.className = 'detail-key';
      keySpan.textContent = `Band ${k}`;
      const track = document.createElement('div');
      track.className = 'band-track';
      const fill = document.createElement('div');
      fill.className = 'band-fill';
      fill.style.width = `${pct}%`;
      fill.style.background = color;
      track.appendChild(fill);
      const valSpan = document.createElement('span');
      valSpan.className = 'detail-val';
      valSpan.textContent = Number(v).toFixed(3);
      row.appendChild(keySpan);
      row.appendChild(track);
      row.appendChild(valSpan);
      frag.appendChild(row);
    });
    return frag;
  },
};
