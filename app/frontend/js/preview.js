/* Data preview table and quality cards rendering for the FMAS import interface. */
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

    const table = document.createElement('table');
    const thead = table.createTHead();
    const headerRow = thead.insertRow();
    columns.forEach(c => {
      const th = document.createElement('th');
      th.className = c.is_numeric ? 'col-num' : 'col-skip';
      th.textContent = c.name;
      headerRow.appendChild(th);
    });

    const tbody = table.createTBody();
    rows.forEach(row => {
      const tr = tbody.insertRow();
      columns.forEach(c => {
        const val = row[c.name];
        const isMissing = val === '—' || val === null || val === undefined || val === '';
        const td = tr.insertCell();
        td.className = (c.is_numeric ? 'col-num' : 'col-skip') + (isMissing ? ' missing' : '');
        td.textContent = isMissing ? '—' : val;
      });
    });

    this._tableContainer.innerHTML = '';
    this._tableContainer.appendChild(table);

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
    const dupColor   = q.duplicate_rows === 0 ? 'ok' : 'warn';
    const constColor = q.constant_columns.length === 0 ? 'ok' : 'warn';

    const frag = document.createDocumentFragment();

    // Rows card
    const rowsCard = this._qcard('Rows');
    rowsCard.appendChild(this._qval(this.formatNumber(q.n_rows)));
    frag.appendChild(rowsCard);

    // Numeric columns card
    const numCard = this._qcard('Numeric columns');
    const numValEl = document.createElement('div');
    numValEl.className = 'qcard-value';
    numValEl.appendChild(document.createTextNode(String(q.n_numeric_columns) + ' '));
    const numSub = document.createElement('span');
    numSub.style.cssText = 'font-size:13px;font-weight:400;color:var(--text-secondary)';
    numSub.textContent = `of ${q.n_columns}`;
    numValEl.appendChild(numSub);
    numCard.appendChild(numValEl);
    frag.appendChild(numCard);

    // Missing values card
    const missingValColor = missingColor === 'ok' ? 'var(--success)' : missingColor === 'warn' ? 'var(--warning)' : 'var(--danger)';
    const missingCard = this._qcard('Missing values');
    const missingValEl = this._qval(this.formatPercent(q.missing_pct));
    missingValEl.style.color = missingValColor;
    missingCard.appendChild(missingValEl);
    missingCard.appendChild(this._minibar(missingColor, Math.min(q.missing_pct, 100)));
    frag.appendChild(missingCard);

    // Duplicate rows card
    const dupCard = this._qcard('Duplicate rows');
    const dupValEl = this._qval(this.formatNumber(q.duplicate_rows));
    dupValEl.style.color = dupColor === 'ok' ? 'var(--success)' : 'var(--warning)';
    dupCard.appendChild(dupValEl);
    frag.appendChild(dupCard);

    // Constant columns card
    const constCard = this._qcard('Constant columns');
    const constValEl = this._qval(String(q.constant_columns.length));
    constValEl.style.color = constColor === 'ok' ? 'var(--success)' : 'var(--warning)';
    constCard.appendChild(constValEl);
    if (q.constant_columns.length > 0) {
      const constSub = document.createElement('div');
      constSub.className = 'qcard-sub';
      constSub.textContent = q.constant_columns.slice(0, 3).join(', ') + (q.constant_columns.length > 3 ? ', …' : '');
      constCard.appendChild(constSub);
    }
    frag.appendChild(constCard);

    // Suitability card
    const suitCard = this._qcard('Suitability');
    suitCard.appendChild(this._qval(`${(q.suitability_score * 100).toFixed(0)}%`));
    suitCard.appendChild(this._minibar(suitColor, q.suitability_score * 100));
    const badgeWrap = document.createElement('div');
    badgeWrap.style.marginTop = '6px';
    const badge = document.createElement('span');
    badge.className = `suit-badge badge-${suitColor === 'ok' ? 'ok' : suitColor === 'warn' ? 'warn' : 'bad'}`;
    badge.textContent = suitLabel;
    badgeWrap.appendChild(badge);
    suitCard.appendChild(badgeWrap);
    const runtimeSub = document.createElement('div');
    runtimeSub.className = 'qcard-sub';
    runtimeSub.textContent = `Est. runtime: ${this._formatRuntime(q.estimated_runtime_seconds)}`;
    suitCard.appendChild(runtimeSub);
    frag.appendChild(suitCard);

    // Warnings
    if (q.warnings.length > 0) {
      const warnDiv = document.createElement('div');
      warnDiv.className = 'warnings-list';
      const warnHead = document.createElement('p');
      warnHead.textContent = 'Warnings';
      warnDiv.appendChild(warnHead);
      const ul = document.createElement('ul');
      q.warnings.forEach(w => {
        const li = document.createElement('li');
        li.textContent = w;
        ul.appendChild(li);
      });
      warnDiv.appendChild(ul);
      frag.appendChild(warnDiv);
    }

    this._qualityContainer.innerHTML = '';
    this._qualityContainer.appendChild(frag);
  },

  formatNumber(n) { return Number(n).toLocaleString(); },
  formatPercent(p) { return Number(p).toFixed(1) + '%'; },

  _formatRuntime(seconds) {
    if (seconds < 60)   return `~${Math.round(seconds)}s`;
    if (seconds < 3600) return `~${Math.round(seconds / 60)} min`;
    return `~${(seconds / 3600).toFixed(1)} hr`;
  },

  _qcard(label) {
    const card = document.createElement('div');
    card.className = 'qcard';
    const labelEl = document.createElement('div');
    labelEl.className = 'qcard-label';
    labelEl.textContent = label;
    card.appendChild(labelEl);
    return card;
  },

  _qval(text) {
    const el = document.createElement('div');
    el.className = 'qcard-value';
    el.textContent = text;
    return el;
  },

  _minibar(colorClass, pct) {
    const bar = document.createElement('div');
    bar.className = 'bar-mini';
    const fill = document.createElement('div');
    fill.className = `bar-mini-fill bar-fill-${colorClass}`;
    fill.style.width = `${Number(pct).toFixed(1)}%`;
    bar.appendChild(fill);
    return bar;
  },
};
window.Preview = Preview;
