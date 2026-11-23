// Shared lab framework: Three.js stage, orbit camera, HTML controls, stats.
const CM = window.ChipModel;
const REDUCED = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
const LABS = {};
function defineLab(id, def) { LABS[id] = def; }
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tcol = n => new THREE.Color(cssv(n).slice(0, 7) || '#888888');
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (x, d = 0) => Number(x).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
function si(x, unit) {
  const a = Math.abs(x);
  if (a >= 1e15) return (x / 1e15).toFixed(2) + ' P' + unit;
  if (a >= 1e12) return (x / 1e12).toFixed(2) + ' T' + unit;
  if (a >= 1e9) return (x / 1e9).toFixed(2) + ' G' + unit;
  if (a >= 1e6) return (x / 1e6).toFixed(2) + ' M' + unit;
  if (a >= 1e3) return (x / 1e3).toFixed(1) + ' k' + unit;
  return x.toFixed(a < 10 ? 2 : 0) + ' ' + unit;
}

class Stage {
  constructor(host, o = {}) {
    this.host = host; this.o = o; this.alive = true; this.visible = true; this.t0 = performance.now();
    this.theta = o.theta ?? 0.7; this.phi = o.phi ?? 0.95; this.radius = o.radius ?? 14;
    this.target = new THREE.Vector3(...(o.target || [0, 0, 0]));
    this.onFrame = null; this.dirty = true;
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    } catch (e) {
      host.innerHTML = '<div class="nowebgl">This lab needs WebGL, which this browser could not start. The stats and notes below still update.</div>';
      this.renderer = null; this.scene = new THREE.Scene(); this.root = new THREE.Group(); return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    host.appendChild(this.renderer.domElement);
    if (o.hint !== false) { const h = document.createElement('div'); h.className = 'hint'; h.textContent = 'drag to rotate'; host.appendChild(h); }
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(o.fov ?? 38, 1, 0.1, 2000);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8890b0, 0.95));
    const d = new THREE.DirectionalLight(0xffffff, 0.75); d.position.set(5, 12, 7); this.scene.add(d);
    this.root = new THREE.Group(); this.scene.add(this.root);
    const el = this.renderer.domElement; let drag = null;
    el.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; this.touched = true; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', e => {
      if (!drag) return;
      this.theta -= (e.clientX - drag[0]) * 0.008; this.phi = clamp(this.phi - (e.clientY - drag[1]) * 0.006, 0.08, 1.52);
      drag = [e.clientX, e.clientY]; this.dirty = true;
    });
    ['pointerup', 'pointercancel'].forEach(n => el.addEventListener(n, () => { drag = null; }));
    el.addEventListener('wheel', e => { if (!e.ctrlKey && !e.metaKey) return; e.preventDefault(); this.radius = clamp(this.radius * (1 + e.deltaY * 0.0012), 2, 400); this.dirty = true; }, { passive: false });
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(host);
    this.io = new IntersectionObserver(es => { this.visible = es[0].isIntersecting; }); this.io.observe(host);
    this.resize(); this.loop = this.loop.bind(this); requestAnimationFrame(this.loop);
  }
  resize() {
    if (!this.renderer) return;
    const w = this.host.clientWidth || 300, h = this.host.clientHeight || 300;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.dirty = true;
  }
  place() {
    const r = this.radius * Math.max(1, 1.25 / this.camera.aspect), s = Math.sin(this.phi);
    this.camera.position.set(this.target.x + r * s * Math.sin(this.theta), this.target.y + r * Math.cos(this.phi), this.target.z + r * s * Math.cos(this.theta));
    this.camera.lookAt(this.target);
  }
  loop(now) {
    if (!this.alive) return;
    requestAnimationFrame(this.loop);
    if (!this.renderer || !this.visible) return;
    const dt = Math.min(0.05, (now - (this.last || now)) / 1000); this.last = now;
    if (this.o.spin && !this.touched && !REDUCED) { this.theta += dt * this.o.spin; this.dirty = true; }
    let wants = false;
    if (this.onFrame) wants = this.onFrame(dt) === true;
    if (wants || this.dirty) { this.place(); this.renderer.render(this.scene, this.camera); this.dirty = false; }
  }
  destroy() {
    this.alive = false; if (this.ro) this.ro.disconnect(); if (this.io) this.io.disconnect();
    if (this.renderer) { clearGroup(this.root); this.renderer.dispose(); this.renderer.forceContextLoss && this.renderer.forceContextLoss(); }
    this.host.innerHTML = '';
  }
}
function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    clearGroup(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) { if (c.material.map) c.material.map.dispose(); c.material.dispose(); }
  }
}
const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: o.rough ?? 0.55, metalness: o.metal ?? 0.1, transparent: o.alpha !== undefined, opacity: o.alpha ?? 1, depthWrite: o.alpha === undefined || o.alpha > 0.5 });
function box(w, h, d, c, o) { return new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(c, o)); }
function at(m, x, y, z) { m.position.set(x, y, z); return m; }
function inst(geo, material, n) {
  const m = new THREE.InstancedMesh(geo, material, Math.max(1, n));
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.frustumCulled = false; m.count = 0; return m;
}
const _d = new THREE.Object3D();
function setI(m, i, x, y, z, sx = 1, sy = 1, sz = 1) { _d.position.set(x, y, z); _d.scale.set(sx, sy, sz); _d.updateMatrix(); m.setMatrixAt(i, _d.matrix); }
function doneI(m, n) { m.count = n; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }
function colorI(m, i, c) { m.setColorAt(i, c); }
function segs(points, c, o = 1) {
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: c, transparent: o < 1, opacity: o }));
}
function label(text, o = {}) {
  const size = o.size ?? 44, pad = 10, cv = document.createElement('canvas'), x = cv.getContext('2d');
  const font = `${o.weight ?? 600} ${size}px ${o.mono ? 'JetBrains Mono, monospace' : 'Archivo, Arial, sans-serif'}`;
  x.font = font; const w = Math.ceil(x.measureText(text).width) + pad * 2, h = size + pad * 2;
  cv.width = w; cv.height = h; x.font = font; x.textBaseline = 'middle';
  x.fillStyle = o.color ?? cssv('--ink'); x.fillText(text, pad, h / 2 + 1);
  const t = new THREE.CanvasTexture(cv); t.minFilter = THREE.LinearFilter;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: o.depth ?? false }));
  const k = (o.scale ?? 1) * 0.012; s.scale.set(w * k, h * k, 1); s.renderOrder = 5; return s;
}

class Lab {
  constructor(host, def, state) {
    this.def = def; this.state = state; this.els = {};
    host.innerHTML = '';
    const head = document.createElement('div'); head.className = 'lab-head';
    head.innerHTML = `<h4>${def.title}</h4><span>${def.sub || ''}</span>`;
    this.stageEl = document.createElement('div'); this.stageEl.className = 'stage';
    this.side = document.createElement('div'); this.side.className = 'lab-side';
    this.statsEl = document.createElement('div'); this.statsEl.className = 'lab-stats';
    this.noteEl = document.createElement('div'); this.noteEl.className = 'lab-note';
    host.append(head, this.stageEl, this.side, this.statsEl, this.noteEl);
    if (def.height) this.stageEl.style.minHeight = def.height + 'px';
    this.stage = new Stage(this.stageEl, def.stage || {});
    this.render = () => {}; def.build(this);
    this.render(); this.stage.dirty = true;
  }
  id(k) { return 'lab-' + this.def.key + '-' + k; }
  slider(k, text, values, init, f = v => v) {
    // values is a list, so nonlinear scales stay exact.
    if (this.state[k] === undefined) this.state[k] = init;
    let idx = values.indexOf(this.state[k]); if (idx < 0) idx = values.indexOf(init); if (idx < 0) idx = 0;
    const d = document.createElement('div'); d.className = 'ctl';
    d.innerHTML = `<label for="${this.id(k)}"><span>${text}</span><output></output></label><input id="${this.id(k)}" type="range" min="0" max="${values.length - 1}" step="1" value="${idx}">`;
    const inp = d.querySelector('input'), out = d.querySelector('output');
    const sync = () => { const v = values[+inp.value]; this.state[k] = v; out.textContent = f(v); };
    sync(); inp.addEventListener('input', () => { sync(); this.render(k); this.stage.dirty = true; });
    this.side.appendChild(d); this.els[k] = { inp, values, sync };
  }
  setSlider(k, v) { const e = this.els[k]; if (!e) return; let b = 0, bd = 1e30; e.values.forEach((x, i) => { const dd = Math.abs(Math.log((x + 1e-9) / (v + 1e-9))); if (dd < bd) { bd = dd; b = i; } }); e.inp.value = b; e.sync(); }
  select(k, text, opts, init) {
    if (this.state[k] === undefined) this.state[k] = init;
    const d = document.createElement('div'); d.className = 'ctl';
    d.innerHTML = `<label for="${this.id(k)}"><span>${text}</span></label><select id="${this.id(k)}">${opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select>`;
    const s = d.querySelector('select'); s.value = this.state[k];
    s.addEventListener('change', () => { this.state[k] = s.value; this.render(k); this.stage.dirty = true; });
    this.side.appendChild(d); this.els[k] = { sel: s };
  }
  buttons(list) {
    const d = document.createElement('div'); d.className = 'btns';
    const out = list.map(([t, fn]) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.addEventListener('click', () => { fn(b); this.stage.dirty = true; }); d.appendChild(b); return b; });
    this.side.appendChild(d); return out;
  }
  stats(rows) { this.statsEl.innerHTML = rows.map(([l, v, c]) => `<div class="stat ${c || ''}"><b>${v}</b><span>${l}</span></div>`).join(''); }
  note(html) { this.noteEl.innerHTML = `<p>${html}</p>`; }
  destroy() { this.stage.destroy(); }
}
const labState = {};
function mountLab(host) {
  const key = host.dataset.lab, def = LABS[key];
  if (!def) return null;
  def.key = key; labState[key] = labState[key] || {};
  host.classList.add('lab');
  return new Lab(host, def, labState[key]);
}
