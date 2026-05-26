/* SQL connection form handling for the FMAS import interface. */
/* global App, API */
const SQL = {
  _connected: false,

  init() {
    document.getElementById('btn-test-conn').addEventListener('click', () => this.testConnection());
    document.getElementById('btn-run-query').addEventListener('click', () => this.runQuery());
  },

  async testConnection() {
    const cs = document.getElementById('conn-string').value.trim();
    if (!cs) { App.showError('Enter a connection string first.'); return; }

    const btn = document.getElementById('btn-test-conn');
    const status = document.getElementById('conn-status');
    btn.textContent = 'Testing…';
    btn.disabled = true;
    status.textContent = '';
    status.className = 'conn-status';

    try {
      const res = await API.testSqlConnection(cs);
      if (res.success) {
        status.textContent = `✓ Connected (${res.database_type || 'database'})`;
        status.classList.add('ok');
        document.getElementById('btn-run-query').disabled = false;
        this._connected = true;
      } else {
        status.textContent = `✗ ${res.message}`;
        status.classList.add('err');
        document.getElementById('btn-run-query').disabled = true;
        this._connected = false;
      }
    } catch (err) {
      status.textContent = `✗ ${err.error || 'Connection failed'}`;
      status.classList.add('err');
      document.getElementById('btn-run-query').disabled = true;
      this._connected = false;
    } finally {
      btn.textContent = 'Test connection';
      btn.disabled = false;
    }
  },

  async runQuery() {
    const cs    = document.getElementById('conn-string').value.trim();
    const query = document.getElementById('sql-query').value.trim();
    if (!query) { App.showError('Enter a SQL query first.'); return; }

    App.showLoading('Executing query…');
    try {
      const preview = await API.executeSqlQuery(cs, query);
      App.dataId = preview.data_id;
      App.previewData = preview;
      App.hideLoading();
      App.goToStep(1);
    } catch (err) {
      App.hideLoading();
      let msg = err.error || 'Query failed.';
      if (err.status === 400) msg = 'Write operations are not permitted. Use a SELECT query.';
      if (err.status === 408) msg = 'Query timed out. Try adding a WHERE clause or LIMIT.';
      if (err.status === 502) msg = 'Cannot connect to the database. Check your connection string.';
      App.showError(msg);
    }
  },
};
