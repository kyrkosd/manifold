/* API client for the FMAS import interface backend. */
const API = {
  baseUrl: '',

  async _fetch(url, opts = {}, timeoutMs = 30_000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(this.baseUrl + url, { ...opts, signal: ctrl.signal });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw { status: res.status, error: body.error || body.detail || 'Request failed', detail: body.detail };
      }
      return body;
    } catch (err) {
      if (err.name === 'AbortError') throw { status: 0, error: 'Request timed out. The server took too long to respond.', detail: null };
      throw err.status !== undefined ? err : { status: 0, error: String(err), detail: null };
    } finally {
      clearTimeout(timer);
    }
  },

  async uploadFile(file) {
    const fd = new FormData();
    fd.append('file', file);
    return this._fetch('/api/upload', { method: 'POST', body: fd }, 120_000);
  },

  async testSqlConnection(connectionString) {
    return this._fetch('/api/sql/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ connection_string: connectionString, query: 'SELECT 1' }),
    });
  },

  async executeSqlQuery(connectionString, query) {
    return this._fetch('/api/sql/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ connection_string: connectionString, query }),
    }, 120_000);
  },

  async launchPipeline(dataId, config) {
    return this._fetch('/api/launch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data_id: dataId, config }),
    });
  },

  async getRunStatus(runId) {
    return this._fetch(`/api/runs/${runId}/status`);
  },
};
