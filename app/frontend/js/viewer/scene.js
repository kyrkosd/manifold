/* Three.js scene setup for the FMAS 3D manifold viewer. */
/* global THREE, Controls, ClusterMesh */
const Scene = {
  renderer: null,
  scene: null,
  camera: null,
  animId: null,
  _lastTime: 0,

  init(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    this.camera.position.set(0, 0, 5);

    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(2, 4, 3);
    this.scene.add(ambient, dir);

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this._startLoop();
  },

  resize() {
    const wrap = document.getElementById('canvas-wrap');
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  },

  _startLoop() {
    const loop = (timestamp) => {
      this.animId = requestAnimationFrame(loop);
      const delta = (timestamp - this._lastTime) / 1000;
      this._lastTime = timestamp;

      Controls.update();
      ClusterMesh.animate(delta);
      this.renderer.render(this.scene, this.camera);
    };
    requestAnimationFrame(t => { this._lastTime = t; loop(t); });
  },

  add(obj)    { this.scene.add(obj); },
  remove(obj) { this.scene.remove(obj); },

  clear() {
    // Remove everything except the two lights (indices 0 and 1).
    while (this.scene.children.length > 2) {
      this.scene.remove(this.scene.children[2]);
    }
  },
};
window.Scene = Scene;
