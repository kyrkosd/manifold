/* API client for the FMAS 3D manifold viewer. */
/* exported ViewerAPI */
const ViewerAPI = {
  async _fetch(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async getViewerData(runId) {
    return this._fetch(`/api/viewer/${runId}/data`);
  },

  async getPointDetail(runId, pointIndex) {
    return this._fetch(`/api/viewer/${runId}/point/${pointIndex}`);
  },

  async reproject(runId, alpha) {
    return this._fetch(`/api/viewer/${runId}/reproject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alpha != null ? { alpha } : {}),
    });
  },
};
