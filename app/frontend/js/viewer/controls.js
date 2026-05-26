/* Orbit controls, panning, zoom, raycasting for the FMAS 3D manifold viewer. */
/* global THREE, Scene, PointCloud */
const Controls = {
  _isDragging: false,
  _isPanning:  false,
  _lastX: 0,
  _lastY: 0,
  _downX: 0,
  _downY: 0,
  _theta: 0,
  _phi: Math.PI / 4,
  _radius: 5,
  _target: new THREE.Vector3(0, 0, 0),

  // Raycasting
  _raycaster: new THREE.Raycaster(),
  _mouse: new THREE.Vector2(),
  _onPointClick: null,

  // Touch
  _touches: [],
  _pinchDist: 0,

  init(canvas) {
    canvas.addEventListener('mousedown',   e => this._onDown(e));
    canvas.addEventListener('mousemove',   e => this._onMove(e));
    canvas.addEventListener('mouseup',     () => this._onUp());
    canvas.addEventListener('mouseleave',  () => this._onUp());
    canvas.addEventListener('wheel',       e => this._onWheel(e), { passive: true });
    canvas.addEventListener('click',       e => this._onClick(e));
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('touchstart',  e => this._onTouchStart(e),  { passive: false });
    canvas.addEventListener('touchmove',   e => this._onTouchMove(e),   { passive: false });
    canvas.addEventListener('touchend',    () => this._onTouchEnd());
    canvas.addEventListener('touchcancel', () => this._onTouchEnd());
  },

  initRaycasting(callback) {
    this._onPointClick = callback;
    this._raycaster.params.Points = { threshold: 0.08 };
  },

  // ------------------------------------------------------------------
  // Mouse
  // ------------------------------------------------------------------

  _onDown(e) {
    this._isDragging = true;
    this._isPanning  = (e.button === 2) || (e.button === 0 && e.shiftKey);
    this._lastX = e.clientX; this._lastY = e.clientY;
    this._downX = e.clientX; this._downY = e.clientY;
  },

  _onUp() { this._isDragging = false; this._isPanning = false; },

  _onMove(e) {
    if (!this._isDragging) return;
    const dx = e.clientX - this._lastX;
    const dy = e.clientY - this._lastY;
    this._lastX = e.clientX; this._lastY = e.clientY;

    if (this._isPanning) {
      const panSpeed = this._radius * 0.001;
      const fwd   = Scene.camera.getWorldDirection(new THREE.Vector3());
      const right = new THREE.Vector3().crossVectors(fwd, Scene.camera.up).normalize();
      this._target.addScaledVector(right, -dx * panSpeed);
      this._target.addScaledVector(Scene.camera.up, dy * panSpeed);
    } else {
      this._theta -= dx * 0.005;
      this._phi    = Math.max(0.05, Math.min(Math.PI - 0.05, this._phi + dy * 0.005));
    }
  },

  _onWheel(e) {
    this._radius = Math.max(0.5, Math.min(50, this._radius + e.deltaY * 0.01));
  },

  _onClick(e) {
    if (Math.abs(e.clientX - this._downX) > 5 || Math.abs(e.clientY - this._downY) > 5) return;
    if (!this._onPointClick || !PointCloud._points) return;

    const canvas = document.getElementById('viewer-canvas');
    const rect   = canvas.getBoundingClientRect();
    this._mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    this._mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._mouse, Scene.camera);

    const hits = this._raycaster.intersectObject(PointCloud._points);
    if (hits.length) this._onPointClick(hits[0].index);
  },

  // ------------------------------------------------------------------
  // Touch
  // ------------------------------------------------------------------

  _onTouchStart(e) {
    e.preventDefault();
    this._touches = Array.from(e.touches);
    if (this._touches.length === 2) {
      this._pinchDist = this._dist2(this._touches);
    } else {
      this._isDragging = true;
      this._lastX = this._touches[0].clientX;
      this._lastY = this._touches[0].clientY;
    }
  },

  _onTouchMove(e) {
    e.preventDefault();
    const touches = Array.from(e.touches);
    if (touches.length === 2) {
      const d = this._dist2(touches);
      this._radius = Math.max(0.5, Math.min(50, this._radius - (d - this._pinchDist) * 0.01));
      this._pinchDist = d;
      // Two-finger pan
      const cx = (touches[0].clientX + touches[1].clientX) / 2;
      const cy = (touches[0].clientY + touches[1].clientY) / 2;
      const px = (this._touches[0].clientX + (this._touches[1] || touches[1]).clientX) / 2;
      const py = (this._touches[0].clientY + (this._touches[1] || touches[1]).clientY) / 2;
      const ps = this._radius * 0.002;
      const right = new THREE.Vector3().crossVectors(
        Scene.camera.getWorldDirection(new THREE.Vector3()), Scene.camera.up).normalize();
      this._target.addScaledVector(right, -(cx - px) * ps);
      this._target.addScaledVector(Scene.camera.up, (cy - py) * ps);
    } else if (touches.length === 1 && this._isDragging) {
      const dx = touches[0].clientX - this._lastX;
      const dy = touches[0].clientY - this._lastY;
      this._theta -= dx * 0.005;
      this._phi    = Math.max(0.05, Math.min(Math.PI - 0.05, this._phi + dy * 0.005));
      this._lastX = touches[0].clientX;
      this._lastY = touches[0].clientY;
    }
    this._touches = touches;
  },

  _onTouchEnd() { this._isDragging = false; this._touches = []; },

  _dist2(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  },

  // ------------------------------------------------------------------
  // Per-frame
  // ------------------------------------------------------------------

  update() {
    const x = this._target.x + this._radius * Math.sin(this._phi) * Math.sin(this._theta);
    const y = this._target.y + this._radius * Math.cos(this._phi);
    const z = this._target.z + this._radius * Math.sin(this._phi) * Math.cos(this._theta);
    Scene.camera.position.set(x, y, z);
    Scene.camera.lookAt(this._target);
  },

  resetCamera() {
    this._theta = 0; this._phi = Math.PI / 4; this._radius = 5;
    this._target.set(0, 0, 0);
  },
};
window.Controls = Controls;
