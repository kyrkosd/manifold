/* Point cloud rendering for the FMAS 3D manifold viewer. */
/* global THREE, Scene */
const PointCloud = {
  _points: null,
  _positions: [],
  _isAnomaly: [],
  _scores: [],
  _pointIds: [],
  _colorAttr: null,
  _sizeAttr: null,
  _highlightIdx: null,
  _prevColor: null,
  _prevSize: 0,

  build(data) {
    this.dispose();
    this._positions  = data.point_positions;
    this._isAnomaly  = data.point_is_anomaly;
    this._scores     = data.point_scores;
    this._pointIds   = data.point_ids;

    const n = this._positions.length;
    const flat   = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const sizes  = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      flat[i * 3]     = this._positions[i][0];
      flat[i * 3 + 1] = this._positions[i][1];
      flat[i * 3 + 2] = this._positions[i][2];

      if (this._isAnomaly[i]) {
        // Anomaly: red (#e54040)
        colors[i * 3] = 0.90; colors[i * 3 + 1] = 0.25; colors[i * 3 + 2] = 0.25;
        sizes[i] = Math.min(10.0, 4.0 + (this._scores[i] || 0) * 2.0);
      } else {
        // Normal: blue (#4488cc)
        colors[i * 3] = 0.27; colors[i * 3 + 1] = 0.53; colors[i * 3 + 2] = 0.80;
        sizes[i] = 2.0;
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
    geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors, 3));
    geo.setAttribute('size',     new THREE.Float32BufferAttribute(sizes, 1));

    this._colorAttr = geo.attributes.color;
    this._sizeAttr  = geo.attributes.size;

    const mat = new THREE.PointsMaterial({
      size: 0.06,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.85,
    });

    this._points = new THREE.Points(geo, mat);
    Scene.add(this._points);
  },

  highlight(idx) {
    if (!this._colorAttr || !this._sizeAttr) return;
    if (this._highlightIdx !== null) this.unhighlight();

    this._highlightIdx = idx;
    const c = this._colorAttr;
    this._prevColor = [c.getX(idx), c.getY(idx), c.getZ(idx)];
    this._prevSize  = this._sizeAttr.getX(idx);

    // Highlight: yellow (#ffeb33)
    c.setXYZ(idx, 1.0, 0.92, 0.2);
    this._sizeAttr.setX(idx, 12.0);
    c.needsUpdate = true;
    this._sizeAttr.needsUpdate = true;
  },

  unhighlight() {
    if (this._highlightIdx === null || !this._colorAttr) return;
    const idx = this._highlightIdx;
    this._colorAttr.setXYZ(idx, ...this._prevColor);
    this._sizeAttr.setX(idx, this._prevSize);
    this._colorAttr.needsUpdate = true;
    this._sizeAttr.needsUpdate  = true;
    this._highlightIdx = null;
    this._prevColor    = null;
  },

  dispose() {
    if (this._points) {
      Scene.remove(this._points);
      this._points.geometry.dispose();
      this._points = null;
    }
    this._colorAttr = null;
    this._sizeAttr  = null;
    this._highlightIdx = null;
  },

  getPosition(i) { return this._positions[i] || [0, 0, 0]; },
  isAnomaly(i)    { return this._isAnomaly[i] || false; },
  getScore(i)     { return this._scores[i] || 0; },
  getPointId(i)   { return this._pointIds[i] ?? i; },
  count()         { return this._positions.length; },
};
