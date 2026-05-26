/* Data preview table and quality cards rendering for the FMAS import interface. */
/* exported Preview */
const Preview = {
  init() {
    this._tableContainer   = document.getElementById('table-container');
    this._qualityContainer = document.getElementById('quality-container');
    this._tableHint        = document.getElementById('table-hint');
  },

  render(previewData) {
    this.renderTable(previewData.columns, previewData.preview_rows, previewData.quality.n_rows);
    this.renderQuality(previewData.quality);
  },

  renderTable(columns, rows, totalRows) {
    const numCols = columns.filter(c => c.is_numeric).map(c => c.name);

    let html = '<table><thead><tr>';
    columns.forEach(c => {
      const cls = c.is_numeric ? 'col-num' : 'col-skip';
      html += `<th class="${cls}">${this._esc(c.name)}</th>`;
    });
    html += '</tr></thead><tbody>';

    rows.forEach(row => {
      html += '<tr>';
      columns.forEach(c => {
        const val = row[c.name];
        const isMissing = val === '—' || val === null || val === undefined || val === '';
        const cls = c.is_numeric ? 'col-num' : 'col-skip';
        const valClass = isMissing ? ' missing' : '';
        html += `<td class="${cls}${valClass}">${this._esc(isMissing ? '—' : val)}</td>`;
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    this._tableContainer.innerHTML = html;

    const numCount = numCols.length;
    const showing  = rows.length;
    this._tableHint.textContent =
      `Showing ${this.formatNumber(showing)} of ${this.formatNumber(totalRows)} rows. ` +
      `Blue = numeric (included). Gray = non-numeric (excluded). ${numCount} numeric column(s).`;
  },

  renderQuality(q) {
    const missingColor = q.missing_pct < 1 ? 'ok' : q.missing_pct < 10 ? 'warn' : 'bad';
    const suitColor    = q.suitability_score >= 0.7 ? 'ok' : q.suitability_score >= 0.4 ? 'warn' : 'bad';
    const suitLabel    = q.suitability_level === 'ready'
      ? 'Ready for analysis'
      : q.suitability_level === 'needs_attention'
      ? 'Needs attention'
      : 'Not suitable';

    const dupColor = q.duplicate_rows === 0 ? 'ok' : 'warn';
    const constColor = q.constant_columns.length === 0 ? 'ok' : 'warn';

    let html = `
      <div class="qcard">
        <div class="qcard-label">Rows</div>
        <div class="qcard-value">${this.formatNumber(q.n_rows)}</div>
      </div>
      <div class="qcard">
        <div class="qcard-label">Numeric columns</div>
        <div class="qcard-value">${q.n_numeric_columns} <span style="font-size:13px;font-weight:400;color:var(--text-secondary)">of ${q.n_columns}</span></div>
      </div>
      <div class="qcard">
        <div class="qcard-label">Missing values</div>
        <div class="qcard-value" style="color:var(--${missingColor === 'ok' ? 'success' : missingColor === 'warn' ? 'warning' : 'danger'})">${this.formatPercent(q.missing_pct)}</div>
        <div class="bar-mini"><div class="bar-mini-fill bar-fill-${missingColor}" style="width:${Math.min(q.missing_pct, 100).toFixed(1)}%"></div></div>
      </div>
      <div class="qcard">
        <div class="qcard-label">Duplicate rows</div>
        <div class="qcard-value" style="color:var(--${dupColor === 'ok' ? 'success' : 'warning'})">${this.formatNumber(q.duplicate_rows)}</div>
      </div>
      <div class="qcard">
        <div class="qcard-label">Constant columns</div>
        <div class="qcard-value" style="color:var(--${constColor === 'ok' ? 'success' : 'warning'})">${q.constant_columns.length}</div>
        ${q.constant_columns.length > 0 ? `<div class="qcard-sub">${q.constant_columns.slice(0,3).map(c => this._esc(c)).join(', ')}${q.constant_columns.length > 3 ? ', …' : ''}</div>` : ''}
      </div>
      <div class="qcard">
        <div class="qcard-label">Suitability</div>
        <div class="qcard-value">${(q.suitability_score * 100).toFixed(0)}%</div>
        <div class="bar-mini"><div class="bar-mini-fill bar-fill-${suitColor}" style="width:${(q.suitability_score * 100).toFixed(1)}%"></div></div>
        <div style="margin-top:6px"><span class="suit-badge badge-${suitColor === 'ok' ? 'ok' : suitColor === 'warn' ? 'warn' : 'bad'}">${suitLabel}</span></div>
        <div class="qcard-sub">Est. runtime: ${this._formatRuntime(q.estimated_runtime_seconds)}</div>
      </div>`;

    if (q.warnings.length > 0) {
      html += `<div class="warnings-list"><p>Warnings</p><ul>`;
      q.warnings.forEach(w => { html += `<li>${this._esc(w)}</li>`; });
      html += `</ul></div>`;
    }

    this._qualityContainer.innerHTML = html;
  },

  formatNumber(n) { return Number(n).toLocaleString(); },
  formatPercent(p) { return Number(p).toFixed(1) + '%'; },

  _formatRuntime(seconds) {
    if (seconds < 60)   return `~${Math.round(seconds)}s`;
    if (seconds < 3600) return `~${Math.round(seconds / 60)} min`;
    return `~${(seconds / 3600).toFixed(1)} hr`;
  },

  _esc(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  },
};
