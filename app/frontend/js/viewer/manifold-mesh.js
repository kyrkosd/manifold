/* Translucent surface mesh rendering for the FMAS 3D manifold viewer. */
/* global THREE, Scene */
const ManifoldMesh = {
  _mesh: null,
  _visible: true,

  build(vertices, faces) {
    if (!vertices.length || !faces.length) return;
    this.dispose();

    const geo = new THREE.BufferGeometry();
    const flat = [];
    vertices.forEach(v => flat.push(...v));
    geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));

    const idx = [];
    faces.forEach(f => idx.push(...f));
    geo.setIndex(idx);
    geo.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({
      color: 0x60a5fa,
      opacity: 0.18,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    this._mesh = new THREE.Mesh(geo, mat);
    Scene.add(this._mesh);
  },

  dispose() {
    if (this._mesh) {
      Scene.remove(this._mesh);
      this._mesh.geometry.dispose();
      this._mesh = null;
    }
  },

  setVisible(v) {
    this._visible = v;
    if (this._mesh) this._mesh.visible = v;
  },

  setOpacity(alpha) {
    if (this._mesh) this._mesh.material.opacity = Math.max(0, Math.min(0.6, alpha));
  },

  toggle() { this.setVisible(!this._visible); },
};
window.ManifoldMesh = ManifoldMesh;
