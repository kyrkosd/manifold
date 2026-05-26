/* Anomaly cluster wireframe meshes for the FMAS 3D manifold viewer. */
/* global THREE, Scene */
/* exported ClusterMesh */
const ClusterMesh = {
  _meshes: [],
  _visible: true,
  _time: 0,        // seconds; incremented in animate()

  build(clusters) {
    this.dispose();
    clusters.forEach(cluster => {
      const group = new THREE.Group();
      group.userData = { clusterId: cluster.cluster_id, reprojected: cluster.reprojected };

      if (cluster.faces.length > 0) {
        // Solid transparent fill.
        const geo = new THREE.BufferGeometry();
        const flat = [];
        cluster.vertices.forEach(v => flat.push(...v));
        geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
        geo.setIndex(cluster.faces.flat());
        geo.computeVertexNormals();

        const solidMat = new THREE.MeshPhongMaterial({
          color: 0xe54040, opacity: 0.15, transparent: true,
          side: THREE.DoubleSide, depthWrite: false,
        });
        group.add(new THREE.Mesh(geo, solidMat));

        // Wireframe overlay.
        const wireMat = new THREE.LineBasicMaterial({
          color: 0xe54040, opacity: 0.6, transparent: true,
        });
        const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), wireMat);
        wire.userData.isPulse = true;   // pulsed in animate()
        group.add(wire);
      } else if (cluster.vertices.length > 0) {
        // Too few points for a mesh — render as large red dots.
        const flat = [];
        cluster.vertices.forEach(v => flat.push(...v));
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
        group.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xe54040, size: 0.12 })));
      }

      // Dashed tether line from reprojected cluster back toward origin.
      if (cluster.reprojected && cluster.centroid) {
        const pts = [
          new THREE.Vector3(...cluster.centroid),
          new THREE.Vector3(0, 0, 0),
        ];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
        const lineMat = new THREE.LineDashedMaterial({
          color: 0xe54040, dashSize: 0.12, gapSize: 0.08,
          opacity: 0.4, transparent: true,
        });
        const tether = new THREE.Line(lineGeo, lineMat);
        tether.computeLineDistances();
        group.add(tether);
      }

      Scene.add(group);
      this._meshes.push(group);
    });
  },

  // Called every frame from scene.js animate() — oscillates wireframe opacity.
  animate(deltaSeconds) {
    this._time += deltaSeconds;
    const opacity = 0.4 + 0.4 * Math.sin(this._time * 2.5);
    this._meshes.forEach(group => {
      group.traverse(child => {
        if (child.userData.isPulse) child.material.opacity = opacity;
      });
    });
  },

  dispose() {
    this._meshes.forEach(g => {
      Scene.remove(g);
      g.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
    });
    this._meshes = [];
  },

  setVisible(v) {
    this._visible = v;
    this._meshes.forEach(g => { g.visible = v; });
  },

  toggle() { this.setVisible(!this._visible); },
};
