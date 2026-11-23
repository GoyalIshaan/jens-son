// Labs: memory, software pipeline, roofline, pod topology, generations, die budget, workbench.
const TPUS = CM.tpu;

defineLab('memory', {
  title: 'Tile reuse and the memory stack', sub: 'tiles, ridge point and bytes in flight', height: 420,
  stage: { theta: 0.72, phi: 1.1, radius: 25, target: [0, 3.2, 0], spin: 0.08 },
  build(L) {
    const s = L.state, g = L.stage.root;
    L.select('chip', 'Chip', TPUS.map((t, i) => [i, t.name]), '4');
    L.slider('b', 'Bytes per element', [1, 2, 4], 2); L.slider('T', 'Output tile side T', [32, 64, 128, 256, 512, 632, 1024, 2048, 4096], 256, fmt);
    L.slider('Tk', 'K slice held on chip', [64, 128, 256, 512, 1024, 2048], 256, fmt); L.slider('lat', 'Memory latency, ns', [200, 300, 500, 800, 1200], 500);
    const dyn = new THREE.Group(); g.add(dyn);
    const parts = new THREE.InstancedMesh(new THREE.SphereGeometry(.09, 8, 6), mat(tcol('--via')), 90); parts.frustumCulled = false; g.add(parts);
    const P = Array.from({ length: 90 }, (_, i) => ({ u: Math.random(), a: Math.random() * 6.28, r: Math.random(), lvl: i % 3 }));
    let ys = [0, 3, 5.4, 7];
    L.stage.onFrame = dt => {
      if (REDUCED) return false;
      P.forEach((p, i) => { const sp = [.3, .75, 1.5][p.lvl]; p.u += dt * sp; if (p.u > 1) p.u -= 1; const y = ys[p.lvl] + (ys[p.lvl + 1] - ys[p.lvl]) * p.u, r = (3.6 - p.lvl * 1.1) * p.r; setI(parts, i, Math.cos(p.a) * r, y, Math.sin(p.a) * r); });
      doneI(parts, P.length); return true;
    };
    L.render = () => {
      const t = TPUS[+s.chip], d = CM.derive(t), { b, T, Tk, lat } = s;
      const need = 2 * (2 * T * Tk) * b + T * T * 4, cap = t.vmem * 2 ** 20, I = CM.tileIntensity(T, T, b), ridge = t.bf16 * 1e12 * (2 / b) / (t.hbmBW * 1e9);
      const frac = Math.min(1, I / ridge), infl = t.hbmBW * 1e9 * lat * 1e-9, req = infl / 4096, need1 = ridge * b;
      clearGroup(dyn);
      const levels = [['Host, SSD', 11, '--line', 'TB'], [`HBM ${t.hbmGiB} GiB, ${fmt(t.hbmBW)} GB/s`, 8.6, '--diff'], [`VMEM ${t.vmem} MiB`, 6, '--m1'], ['Registers, accumulators', 3.1, '--poly']];
      levels.forEach((l, i) => { const m = at(box(l[1], .32, l[1], tcol(l[2]), { alpha: .82 }), 0, ys[i], 0); dyn.add(m); const lb = label(l[0], { scale: .8 }); at(lb, l[1] / 2 + .4, ys[i] + .5, l[1] / 2); dyn.add(lb); });
      const scale = 6 / Math.sqrt(cap), side = Math.min(5.8, Math.sqrt(need) * scale), acc = Math.sqrt(T * T * 4) * scale;
      dyn.add(at(box(side, .5, side, tcol('--via'), { alpha: .55 }), 0, ys[2] + .42, 0)); dyn.add(at(box(Math.min(acc, side), .22, Math.min(acc, side), tcol('--poly')), 0, ys[2] + .8, 0));
      const over = need > cap;
      L.stats([['Ridge at this precision', fmt(ridge) + ' F/B'], ['Tile intensity T/b', fmt(I) + ' F/B', I >= ridge ? 'good' : 'bad'], ['Share of peak', pct(frac, 0), frac > .9 ? 'good' : 'bad'], ['Minimum T for ridge', fmt(need1)], ['Double-buffered bytes', si(need, 'B'), over ? 'bad' : ''], ['Bytes in flight needed', si(infl, 'B')], ['4 KB requests in flight', fmt(req)]]);
      L.note(`With T = ${fmt(T)} and ${b}-byte elements, each byte fetched does ${fmt(I)} FLOPs. ${I >= ridge ? 'That clears the ridge, so this matmul can run at full speed from HBM.' : `The ridge asks for ${fmt(ridge)}, so memory limits you to ${pct(frac, 0)} of peak. Raise T to about ${fmt(need1)}.`} ${over ? 'This tile set does not fit in VMEM, so a smaller K slice or tile is required.' : `The buffers use ${pct(need / cap, 1)} of VMEM, shown as the amber slab, so raw matmul reuse is not what fills the scratchpad.`} Keeping ${fmt(t.hbmBW)} GB/s flowing at ${lat} ns needs ${si(infl, 'B')} in flight, about ${fmt(req)} requests, and every queue on the path has to hold them.`);
    };
  },
});

defineLab('pipeline', {
  title: 'Software pipelining in space and time', sub: 'load, compute and store overlapping', height: 380,
  stage: { theta: 0.22, phi: 0.92, radius: 21, target: [-0.6, 0, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    L.select('mode', 'Schedule', [['pipe', 'Software pipelined'], ['seq', 'Sequential']], 'pipe');
    L.slider('N', 'Iterations', [2, 3, 4, 6, 8, 12, 16, 24], 8); L.slider('S', 'Stages', [2, 3, 4, 5, 6], 3);
    L.slider('slow', 'Slowest stage, relative time', [1, 1.25, 1.5, 2, 3], 1, v => v + '×');
    const blocks = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(0xffffff), 200); blocks.frustumCulled = false; g.add(blocks);
    const dyn = new THREE.Group(); g.add(dyn);
    const head = at(box(.06, 1.2, 1, tcol('--poly'), { alpha: .7 }), 0, .4, 0); g.add(head);
    let total = 1, t = 0, zspan = 1;
    L.stage.onFrame = dt => { if (REDUCED) return false; t += dt / 7; if (t > 1.1) t = 0; head.position.x = -9 + Math.min(1, t) * 18; head.scale.z = zspan; return true; };
    L.render = () => {
      const { mode, N, S, slow } = s, dur = Array.from({ length: S }, (_, i) => i === Math.floor(S / 2) ? slow : 1);
      const fin = Array.from({ length: N }, () => Array(S).fill(0)), st = Array.from({ length: N }, () => Array(S).fill(0)); let clock = 0;
      for (let i = 0; i < N; i++) for (let j = 0; j < S; j++) {
        const start = mode === 'seq' ? clock : Math.max(j ? fin[i][j - 1] : 0, i ? fin[i - 1][j] : 0);
        st[i][j] = start; fin[i][j] = start + dur[j]; clock = fin[i][j];
      }
      total = Math.max(...fin.map(r => r[S - 1])); const busy = dur.reduce((a, b) => a + b, 0) * N, util = busy / (total * S), kx = 18 / total;
      let c = 0; const col = new THREE.Color(), base = [tcol('--m1'), tcol('--diff'), tcol('--via'), tcol('--m2'), tcol('--poly')];
      for (let i = 0; i < N; i++) for (let j = 0; j < S; j++) { setI(blocks, c, -9 + (st[i][j] + dur[j] / 2) * kx, .25 + (i % 2) * .12, (j - (S - 1) / 2) * 1.8, dur[j] * kx * .94, .5, 1.3); col.copy(base[i % 5]); blocks.setColorAt(c, col); c++; }
      doneI(blocks, c); clearGroup(dyn); zspan = S * 1.8; const names = ['load', 'compute', 'store', 'stage 4', 'stage 5', 'stage 6'];
      for (let j = 0; j < S; j++) { const lb = label(j < 3 && S === 3 ? names[j] : 'stage ' + (j + 1), { scale: 1.15 }); at(lb, -10.9, .4, (j - (S - 1) / 2) * 1.8); dyn.add(lb); }
      const ideal = (N + S - 1) * Math.max(...dur), bufs = mode === 'pipe' ? S - 1 : 0;
      L.stats([['Total time', fmt(total, 1)], ['Average unit utilization', pct(util), util > .7 ? 'good' : 'bad'], ['Prologue steps', mode === 'pipe' ? S - 1 : 0], ['Steady-state share', mode === 'pipe' ? pct(Math.max(0, N - S + 1) / (N + S - 1), 0) : '0%'], ['Live buffers needed', bufs]]);
      L.note(mode === 'pipe' ? `Each lane is a hardware unit and each color is one loop iteration. Total steps follow N + S − 1, so ${N} iterations of ${S} stages take ${fmt(total, 1)} units${slow > 1 ? `, and the ${slow}× stage sets the pace for everyone` : ''}. Short loops spend most of their time in prologue and epilogue, which is why you want long loops and few stages. Every overlap needs a live buffer.`
        : `Run in order, only one lane is busy at a time, so each unit is idle for most of the ${fmt(total, 1)} units. Switch to the pipelined schedule to see the same work in far less time.`);
    };
  },
});

defineLab('roofline', {
  title: 'Rooflines across generations', sub: 'attainable TFLOPS against arithmetic intensity, log scales', height: 430,
  stage: { theta: 0.62, phi: 1.16, radius: 25, target: [2.6, 2.6, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    const chips = [...TPUS.map(t => ({ n: t.name + ' BF16', p: t.bf16, b: t.hbmBW })), { n: 'Ironwood FP8', p: 4614, b: 7300 }, { n: 'GB300 NVFP4', p: 15000, b: 8000 }];
    L.select('c', 'Highlight chip', chips.map((c, i) => [i, c.n]), '4');
    L.slider('B', 'Decode batch', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], 64); L.slider('w', 'Bytes per weight', [0.5, 1, 2], 2);
    L.slider('q', 'Queries per KV group', [1, 2, 4, 8, 16, 32, 64, 128], 64); L.slider('kv', 'Bytes per KV element', [0.5, 1, 2], 1); L.slider('pf', 'Prefill tokens per step', [128, 256, 512, 1024, 4096], 1024, fmt);
    const X = i => (Math.log10(i) - 1.2) * 4.2, Y = p => Math.log10(p) * 1.55;
    const dyn = new THREE.Group(); g.add(dyn);
    L.render = () => {
      clearGroup(dyn); const ci = +s.c;
      chips.forEach((c, i) => {
        const z = (i - (chips.length - 1) / 2) * 1.5, r = c.p * 1e12 / (c.b * 1e9), pts = [], top = [], ramp = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000];
        const seq = [...ramp.filter(v => v < r), r, 30000].filter(v => v >= 0.3 * 1 && v > 16 || v === r);
        const sel = i === ci, color = sel ? tcol('--m1') : tcol('--mute');
        for (let k = 0; k < seq.length - 1; k++) {
          const a = seq[k], b = seq[k + 1], ya = Y(CM.attainable(a, c.p * 1e12, c.b * 1e9) / 1e12), yb = Y(CM.attainable(b, c.p * 1e12, c.b * 1e9) / 1e12);
          const x0 = X(a), x1 = X(b), geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.Float32BufferAttribute([x0, 0, z, x1, 0, z, x1, yb, z, x0, 0, z, x1, yb, z, x0, ya, z], 3)); geo.computeVertexNormals();
          dyn.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: sel ? .55 : .16, side: THREE.DoubleSide, depthWrite: false })));
          pts.push(x0, ya, z, x1, yb, z);
        }
        dyn.add(segs(pts, sel ? tcol('--m1') : tcol('--mute'), sel ? 1 : .7));
        const lb = label(c.n, { scale: sel ? .82 : .62, color: cssv(sel ? '--ink' : '--mute') }); at(lb, X(30000) + 1.4, Y(c.p) * .98, z); dyn.add(lb);
      });
      const c = chips[ci], z = (ci - (chips.length - 1) / 2) * 1.5, peak = c.p * 1e12, bw = c.b * 1e9, r = peak / bw;
      const ff = CM.ffIntensity(s.B, s.w), at2 = CM.attnDecodeIntensity(s.q, s.kv), pf = CM.ffIntensity(s.pf, s.w);
      const mk = [[ff, '--poly', 'decode FF'], [at2, '--via', 'decode attention'], [pf, '--diff', 'prefill FF']];
      mk.forEach(([I, cc, n], k) => { const a = CM.attainable(I, peak, bw) / 1e12, y = Y(a), xv = X(Math.max(I, 18)); const m = new THREE.Mesh(new THREE.SphereGeometry(.2, 16, 12), mat(tcol(cc))); at(m, xv, y, z); dyn.add(m); const lb = label(n, { scale: .68, color: cssv(cc) }); at(lb, xv, y + .55 + k * .32, z); dyn.add(lb); });
      const f = (I) => pct(Math.min(1, I / r), I / r > .1 ? 0 : 1), Bneed = CM.decodeBatchForRidge(r, s.w);
      L.stats([['Ridge', fmt(r) + ' F/B'], ['Decode FF', fmt(ff, ff < 10 ? 1 : 0) + ' F/B'], ['FF share of peak', f(ff), ff >= r ? 'good' : 'bad'], ['Attention share of peak', f(at2), at2 >= r ? 'good' : 'bad'], ['Prefill share of peak', f(pf), pf >= r ? 'good' : 'bad'], ['Batch to reach ridge', fmt(Bneed)]]);
      L.note(`Each roof is one chip. The slope is memory-bound, and the flat top is compute-bound. Notice the ridge moving right across generations, which is the chip asking for more reuse. On ${c.n}, batch ${s.B} at ${s.w} bytes per weight puts decode feed-forward at ${f(ff)} of peak. Attention does not move with batch at all: try the batch slider, then try queries per KV group. ${ff < r ? `This chip needs batch ${fmt(Bneed)} to leave the slope.` : 'Feed-forward has reached the roof, so more batch only adds latency.'}`);
    };
  },
});

defineLab('torus', {
  title: 'Pods, hops and failures', sub: 'how topology turns link bandwidth into useful bandwidth', height: 430,
  stage: { theta: 0.72, phi: 1.02, radius: 21, target: [0, 0, 0], spin: 0.1 },
  build(L) {
    const s = L.state, g = L.stage.root;
    L.select('shape', 'Topology', [['4x4x4', '4×4×4 torus (Google cube)'], ['4x4', '4×4 torus (Roune donut)'], ['8x8', '8×8 torus'], ['6x6x6', '6×6×6 torus'], ['mesh444', '4×4×4 mesh, no wraps'], ['router', 'Single router, 64 chips']], '4x4x4');
    L.select('traffic', 'Traffic', [['nbr', 'All-reduce: neighbors only'], ['rand', 'MoE: uniform random']], 'nbr');
    L.slider('bw', 'Chip link bandwidth, GB/s total', [200, 400, 600, 900, 1800, 3600], 600, fmt); L.slider('X', 'All-reduce size, MB', [1, 4, 16, 64, 256], 16);
    const nodes = new THREE.InstancedMesh(new THREE.BoxGeometry(.5, .5, .5), mat(0xffffff), 220), pulses = new THREE.InstancedMesh(new THREE.SphereGeometry(.09, 8, 6), mat(tcol('--via')), 140);
    [nodes, pulses].forEach(m => { m.frustumCulled = false; g.add(m); });
    let wires = null, N = [], adj = [], live = [], failed = -1;
    L.buttons([['Fail a chip', () => { failed = Math.floor(Math.random() * N.length); L.render(); }], ['Repair', () => { failed = -1; L.render(); }]]);
    function bfs(a, b) { const pr = new Map([[a, -1]]), q = [a]; for (let i = 0; i < q.length; i++) { const u = q[i]; if (u === b) break; for (const v of adj[u]) if (!pr.has(v) && v !== failed) { pr.set(v, u); q.push(v); } } if (!pr.has(b)) return null; const p = []; for (let u = b; u !== -1; u = pr.get(u)) p.unshift(u); return p; }
    L.stage.onFrame = dt => {
      if (REDUCED || !N.length) return false;
      let guard = 0; while (live.length < 110 && guard++ < 400) {
        const a = Math.floor(Math.random() * N.length); if (a === failed) continue;
        let b; if (s.traffic === 'nbr' && s.shape !== 'router') b = adj[a][Math.floor(Math.random() * adj[a].length)]; else b = Math.floor(Math.random() * N.length);
        if (a === b || b === failed) continue; const r = bfs(a, b); if (r && r.length > 1) live.push({ r, t: 0 });
      }
      let i = 0; live = live.filter(p => { p.t += dt * 3; return p.t < p.r.length - 1; });
      for (const p of live) { const k = Math.floor(p.t), f = p.t - k, A = N[p.r[k]], B = N[p.r[k + 1]]; setI(pulses, i++, A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f); }
      doneI(pulses, i); return true;
    };
    L.render = () => {
      const sh = s.shape; N = []; adj = []; live = []; const pts = []; let dims = [], wrap = true;
      if (sh === 'router') {
        for (let i = 0; i < 64; i++) { const a = i / 64 * 6.283, y = (i % 4 - 1.5) * .9; N.push([Math.cos(a) * 6, y, Math.sin(a) * 6]); adj.push([64]); }
        N.push([0, 0, 0]); adj.push(Array.from({ length: 64 }, (_, i) => i)); for (let i = 0; i < 64; i++) pts.push(...N[i], 0, 0, 0);
      } else {
        wrap = sh !== 'mesh444'; dims = (sh === 'mesh444' ? '4x4x4' : sh).split('x').map(Number); while (dims.length < 3) dims.push(1);
        const [a, b, c] = dims, p = Math.min(2.6, 8 / Math.max(a, b, c)), id = (x, y, z) => (z * b + y) * a + x;
        for (let z = 0; z < c; z++) for (let y = 0; y < b; y++) for (let x = 0; x < a; x++) { N.push([(x - (a - 1) / 2) * p, (z - (c - 1) / 2) * p, (y - (b - 1) / 2) * p]); adj.push([]); }
        const con = (u, v, far, ax) => { adj[u].push(v); adj[v].push(u); const A = N[u], B = N[v]; if (far) { const o = [0, 0, 0]; o[ax === 0 ? 1 : 0] = p * .28; const e = [A[0] + (ax === 0 ? -p * .6 : 0) * Math.sign(A[0] - B[0]) * -1, A[1], A[2]]; pts.push(...A, A[0] + o[0], A[1] + o[1], A[2] + o[2], A[0] + o[0], A[1] + o[1], A[2] + o[2], B[0] + o[0], B[1] + o[1], B[2] + o[2], B[0] + o[0], B[1] + o[1], B[2] + o[2], ...B); } else pts.push(...A, ...B); };
        for (let z = 0; z < c; z++) for (let y = 0; y < b; y++) for (let x = 0; x < a; x++) {
          if (x + 1 < a) con(id(x, y, z), id(x + 1, y, z)); else if (wrap && a > 2) con(id(x, y, z), id(0, y, z), true, 0);
          if (y + 1 < b) con(id(x, y, z), id(x, y + 1, z)); else if (wrap && b > 2) con(id(x, y, z), id(x, 0, z), true, 1);
          if (z + 1 < c) con(id(x, y, z), id(x, y, z + 1)); else if (wrap && c > 2) con(id(x, y, z), id(x, y, 0), true, 2);
        }
      }
      const cN = tcol('--m1'), cF = tcol('--poly'), cR = tcol('--m2');
      N.forEach((q, i) => { setI(nodes, i, q[0], q[1], q[2], i === 64 && sh === 'router' ? 2.4 : 1, i === 64 && sh === 'router' ? 2.4 : 1, i === 64 && sh === 'router' ? 2.4 : 1); nodes.setColorAt(i, i === failed ? cF : (sh === 'router' && i === 64) ? cR : cN); });
      doneI(nodes, N.length); if (wires) { g.remove(wires); wires.geometry.dispose(); wires.material.dispose(); } wires = segs(pts, tcol('--mute'), .4); g.add(wires);
      const real = dims.filter(d => d > 1), P = sh === 'router' ? 64 : real.reduce((a, b) => a * b, 1), nbr = s.traffic === 'nbr';
      let hops; if (sh === 'router') hops = nbr ? 2 : CM.routerAvgHops(P); else if (nbr) hops = wrap ? 1 : 2; else hops = wrap ? CM.torusAvgHops(real) : real.reduce((a, w) => a + (w * w - 1) / (3 * w), 0);
      const useful = CM.usefulBw(1, s.bw, hops), Xb = s.X * 1e6, ar = CM.allReduceBytes(Xb, P) / (useful * 1e9) * 1e6;
      const domain = failed < 0 ? 'none' : sh === 'router' ? '1 chip' : wrap ? `${P} chip torus broken` : `${P} chip mesh degraded`;
      L.stats([['Chips', P], ['Average hops', fmt(hops, 2)], ['Useful bandwidth', fmt(useful) + ' GB/s', hops > 2.5 ? 'bad' : hops <= 1 ? 'good' : 'warn'], ['Efficiency', pct(Math.min(1, 1 / hops), 0)], ['All-reduce time', fmt(ar, ar < 10 ? 1 : 0) + ' µs'], ['Failure impact', domain, failed >= 0 ? 'bad' : '']]);
      L.note(`Useful bandwidth = link bandwidth ÷ average hops. ${sh === 'router' ? 'A router is a node, so every bit crosses two links, which spends half your ports on the way through.' : nbr && wrap ? 'Collectives touch only neighbors here, so a torus uses every link fully.' : !wrap && nbr ? 'Without wrap links, collectives run at half torus speed.' : `Random traffic averages ${fmt(hops, 1)} hops, so MoE on this topology pays about ${fmt(hops, 1)}× in link bandwidth.`} ${failed >= 0 ? (sh === 'router' ? 'One chip is lost, and the others carry on.' : 'Traffic now detours, and a torus job loses its regular structure. This is the case optical switches exist for: Google can swap in a spare cube.') : 'Press Fail a chip to see the failure domain.'}`);
    };
  },
});

defineLab('generations', {
  title: 'Five generations of TPU', sub: 'Table 1 of the Ironwood paper, log₂ scale relative to TPU v2', height: 450,
  stage: { theta: 0.62, phi: 1.08, radius: 25, target: [0, 1.6, 0] },
  build(L) {
    const s = L.state, g = L.stage.root; s.sel = s.sel ?? 4;
    L.select('mode', 'View', [['abs', 'Growth since TPU v2'], ['bal', 'Balance: each resource per TFLOP']], 'abs');
    const gens = TPUS.map(t => t.name.replace('TPU ', ''));
    const btns = L.buttons(gens.map((n, i) => [n, () => { s.sel = i; L.render(); }]));
    const metrics = [['Peak BF16', t => t.bf16, '--poly'], ['HBM bandwidth', t => t.hbmBW, '--diff'], ['HBM capacity', t => t.hbmGiB, '--via'], ['VMEM', t => t.vmem, '--m1'], ['ICI per chip', t => t.links * t.linkBW, '--m2']];
    const notes = [
      'TPU v2 sets the template: two TensorCores, BF16 arrays with FP32 accumulation, HBM, VLIW control, ICI links. Later chips scale this block diagram.',
      'v3 doubles arrays and raises clock by a third with liquid cooling, while HBM bandwidth rises only 29%. The ridge doubles, so software must find more reuse. Heat and link bugs arrive with it.',
      'v4 is the system generation: 3D torus, optical switches, more SparseCores. It takes six slower links over four faster ones. Ridge reaches 229 as compute keeps outrunning memory.',
      'v5p corrects the balance. Arrays stay the same and clock rises 67%, while VMEM goes up 4×, HBM capacity 3×, bandwidth 2.3×. It is the only generation where the ridge falls.',
      'Ironwood doubles array side, adds native FP8, moves to two dies and eight HBM3E stacks, and adds spare rows, self-test and replay. Compute jumps 5×, and ICI per TFLOP falls 5×.',
    ];
    const dyn = new THREE.Group(); g.add(dyn);
    L.render = () => {
      clearGroup(dyn); btns.forEach((b, i) => b.classList.toggle('on', i === s.sel));
      const bal = s.mode === 'bal', px = 3.3, pz = 1.5, base = TPUS[0];
      dyn.add(at(box(px * 5.2, .05, pz * 5.4, tcol('--line'), { alpha: .5 }), 0, 0, 0));
      TPUS.forEach((t, i) => {
        const x = (i - 2) * px; const lb = label(gens[i] + ' ' + t.year, { scale: i === s.sel ? 1 : .76, color: cssv(i === s.sel ? '--ink' : '--mute') }); at(lb, x, -0.5, pz * 3); dyn.add(lb);
        metrics.forEach(([n, f, c], j) => {
          if (bal && j === 0) return; let r = f(t) / f(base); if (bal) r /= t.bf16 / base.bf16;
          const h = Math.log2(r) * (bal ? .75 : .62), z = (j - 2) * pz;
          dyn.add(at(box(px * .62, Math.abs(h) + .03, pz * .72, tcol(c), { alpha: i === s.sel ? .98 : .62 }), x, h / 2, z));
          if (i === s.sel) { const v = label((r >= 1 ? fmt(r, r < 10 ? 1 : 0) : r.toFixed(2)) + '×', { scale: .62, mono: true }); at(v, x, h + (h >= 0 ? .4 : -.4), z); dyn.add(v); }
        });
      });
      metrics.forEach(([n, f, c], j) => { if (bal && j === 0) return; const lb = label(n, { scale: .68, color: cssv(c) }); at(lb, -px * 2.75, .15, (j - 2) * pz); dyn.add(lb); });
      const t = TPUS[s.sel], d = CM.derive(t), clk = s.sel === 4 ? 2.2 : d.clockGHz;
      L.stats([['Peak BF16', fmt(t.bf16) + ' TF'], ['Implied clock', clk.toFixed(2) + ' GHz' + (s.sel === 4 ? '*' : '')], ['Ridge', fmt(d.ridgeBf16) + ' F/B'], ['VMEM per TFLOP', d.vmemPerTf.toFixed(2) + ' MiB'], ['ICI per TFLOP', d.iciPerTf.toFixed(2) + ' GB/s'], ['Batch at ridge', fmt(d.batchForRidge)]]);
      L.note((bal ? 'Bars below the plate mean a resource has shrunk relative to compute since v2. ' : '') + notes[s.sel] + (s.sel === 4 ? ' *The printed array count implies 4.4 GHz; see Source audit.' : ''));
    };
  },
});

defineLab('die', {
  title: 'Die budget', sub: 'schematic layout with teaching constants, not a floorplan of any product', height: 470,
  stage: { theta: 0.68, phi: 0.92, radius: 40, target: [0, 1, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    const NODE = { 7: { mac: 220, sram: .45, leak: .03 }, 5: { mac: 130, sram: .34, leak: .035 }, 3: { mac: 85, sram: .31, leak: .04 } }, PREC = { int8: { a: 1, e: .2, b: 1, n: 'INT8' }, fp8: { a: 2, e: .28, b: 1, n: 'FP8' }, bf16: { a: 5, e: .5, b: 2, n: 'BF16' } };
    L.select('pre', 'Start from', [['iw', 'Ironwood-like, 2 dies'], ['v4', 'TPU v4-like'], ['edge', 'Small inference die']], 'iw');
    L.select('node', 'Process', [[7, '7 nm class'], [5, '5 nm class'], [3, '3 nm class']], '3');
    L.select('prec', 'Array precision', [['int8', 'INT8'], ['fp8', 'FP8'], ['bf16', 'BF16 with FP32 accumulate']], 'bf16');
    L.slider('side', 'Array side', [32, 64, 128, 256, 512], 256); L.slider('cnt', 'Arrays', [1, 2, 4, 8, 16, 32], 8); L.slider('f', 'Clock, GHz', [.6, .8, 1, 1.2, 1.4, 1.75, 2, 2.2, 2.5, 3], 2.2);
    L.slider('sram', 'SRAM, MiB', [8, 16, 32, 64, 128, 256, 384, 512], 128); L.slider('hbm', 'HBM stacks', [0, 1, 2, 4, 6, 8, 12], 8); L.slider('dies', 'Compute dies', [1, 2, 4], 2); L.slider('ex', 'Explode layers', [0, 1, 2, 3, 4, 5, 6], 2);
    const pre = { iw: ['3', 'bf16', 256, 8, 2.2, 128, 8, 2], v4: ['7', 'bf16', 128, 8, 1.05, 32, 4, 1], edge: ['5', 'int8', 64, 4, 1, 32, 0, 1] };
    L.els.pre.sel.addEventListener('change', () => { const p = pre[s.pre]; s.node = p[0]; L.els.node.sel.value = p[0]; s.prec = p[1]; L.els.prec.sel.value = p[1]; ['side', 'cnt', 'f', 'sram', 'hbm', 'dies'].forEach((k, i) => L.setSlider(k, p[i + 2])); L.render(); });
    const [heatB] = L.buttons([['Show heat', b => { s.heat = !s.heat; b.classList.toggle('on', s.heat); b.textContent = s.heat ? 'Show blocks' : 'Show heat'; L.render(); }]]);
    const dyn = new THREE.Group(); g.add(dyn);
    L.render = () => {
      clearGroup(dyn); const n = NODE[s.node], p = PREC[s.prec], { side, cnt, f, sram, hbm, dies, ex } = s;
      const macs = side * side * cnt, aArr = macs * n.mac * p.a / 1e6, aSram = sram * n.sram, aVec = aArr * .5 + 30, aPhy = hbm * 15, aSer = 50, total = aArr + aSram + aVec + aPhy + aSer + (dies > 1 ? dies * 15 : 0), perDie = total / dies;
      const V = clamp(CM.voltageForFreq(f, .3, 1.3, .75, 2), .5, 1.05), es = (V / .75) ** 2, pArr = macs * f * 1e9 * p.e * 1e-12 * es * .75, pOth = pArr * .35, pHbm = hbm * 16, pLeak = total * n.leak * (V / .75), P = pArr + pOth + pHbm + pLeak;
      const peak = 2 * macs * f * 1e9, bw = hbm * 900e9, sideMm = Math.sqrt(perDie), yld = CM.yieldPoisson(perDie, .1), needEdge = (hbm / dies) * 11 + 8, haveEdge = 4 * sideMm * .8;
      const K = 1 / 3, W = sideMm * K, gap = .5, heatC = (w) => { const c = tcol('--m1'), h = clamp(w / 3.5, 0, 1); return c.lerp(tcol('--poly'), h); };
      dyn.add(at(box(Math.max(W * dies + 20, 14), .3, W + 12, tcol('--diff'), { alpha: .7 }), 0, 0, 0));
      if (hbm) dyn.add(at(box(W * dies + 15, .1, W + 5, tcol('--line'), { alpha: .6 }), 0, .5 + ex * .35, 0));
      const yDie = .75 + ex * .75, dtot = W * dies + gap * (dies - 1);
      for (let d = 0; d < dies; d++) {
        const cx = -dtot / 2 + W / 2 + d * (W + gap), phyW = W * Math.min(.22, (aPhy + aSer) / dies / perDie), band = W * Math.min(.5, aSram / dies / perDie / (1 - .0));
        const wAv = W - phyW * 2, hA = W - band;
        dyn.add(at(box(W, .22, W, tcol('--line')), cx, yDie, 0));
        const per = Math.ceil(cnt / dies), gc = Math.ceil(Math.sqrt(per)), gr = Math.ceil(per / gc), cw = wAv / gc, ch = (hA * .8) / gr;
        const dArr = pArr / Math.max(aArr, 1e-6), dS = (pOth * .5) / Math.max(aSram, 1), dV = (pOth * .5) / Math.max(aVec, 1);
        for (let i = 0; i < per; i++) { const x = cx - wAv / 2 + (i % gc + .5) * cw, z = -W / 2 + band + hA * .2 + (Math.floor(i / gc) + .5) * ch; dyn.add(at(box(cw * .9, .3, ch * .9, s.heat ? heatC(dArr) : tcol('--m1')), x, yDie + .2, z)); }
        dyn.add(at(box(wAv * .98, .26, band * .92, s.heat ? heatC(dS) : tcol('--diff')), cx, yDie + .17, -W / 2 + band / 2));
        dyn.add(at(box(wAv * .98, .26, hA * .17, s.heat ? heatC(dV) : tcol('--via')), cx, yDie + .17, -W / 2 + band + hA * .1));
        [-1, 1].forEach(sd => dyn.add(at(box(phyW * .9, .26, W * .96, s.heat ? heatC(.3) : tcol('--m2')), cx + sd * (W / 2 - phyW / 2), yDie + .17, 0)));
      }
      for (let i = 0; i < hbm; i++) { const sd = i % 2 ? 1 : -1, row = Math.floor(i / 2), rows = Math.ceil(hbm / 2), z = (row - (rows - 1) / 2) * (W / Math.max(rows, 1) * .98 + .4), x = sd * (dtot / 2 + 2.5), hh = Math.min(3.2, W / Math.max(rows, 1)); for (let k = 0; k < 6; k++) dyn.add(at(box(3, .16, hh * .9, k === 0 ? tcol('--via') : tcol('--poly'), { alpha: .95 }), x, yDie + k * (.22 + ex * .04), z)); }
      if (ex > 0) dyn.add(at(box(Math.max(W * dies + 15, 12), .1, W + 5, tcol('--m1'), { alpha: .14 }), 0, yDie + 2 + ex * .9, 0));
      const lb = label(`${fmt(perDie)} mm² per die`, { scale: .9 }); at(lb, 0, yDie + 1.5, W / 2 + 2); dyn.add(lb);
      const over = perDie > 858, hot = P / total, edgeBad = needEdge > haveEdge, ridge = bw ? peak / bw : Infinity;
      L.stats([['Peak', si(peak, 'FLOP/s')], ['Die area', fmt(perDie) + ' mm²', over ? 'bad' : perDie > 700 ? 'warn' : ''], ['Yield per die', pct(yld, 0), yld < .4 ? 'bad' : ''], ['Power', fmt(P) + ' W', P > 1400 ? 'bad' : P > 900 ? 'warn' : ''], ['Average density', hot.toFixed(2) + ' W/mm²', hot > 2 ? 'bad' : ''], ['Array density', (pArr / aArr).toFixed(2) + ' W/mm²'], ['Voltage', V.toFixed(2) + ' V'], ['Ridge', bw ? fmt(ridge) + ' F/B' : 'no HBM'], ['Edge used', pct(needEdge / haveEdge, 0), edgeBad ? 'bad' : '']]);
      const shares = [['arrays', aArr], ['SRAM', aSram], ['vector and control', aVec], ['PHYs', aPhy + aSer]].map(([k, v]) => `${k} ${pct(v / total, 0)}`).join(', ');
      const bind = over ? 'Area binds: this die exceeds the 858 mm² reticle. Add dies or trim SRAM.' : edgeBad ? 'Beachfront binds: the die edge cannot fit these PHYs. More dies add perimeter.' : P > 1400 ? 'Power binds. Try a lower clock with a wider array: voltage drops and energy per operation falls with its square.' : hot > 2 ? 'Heat density is high. Spread the arrays, or slow them down.' : 'Nothing binds yet, so you have room to spend.';
      L.note(`${bind} Area split: ${shares}. Clock ${f} GHz needs about ${V.toFixed(2)} V on the teaching curve, which scales energy per operation by ${es.toFixed(2)}×. Constants are rough defaults, so use this to see relationships and not to predict a real chip.`);
    };
  },
});

defineLab('workbench', {
  title: 'Goal to system', sub: 'decode sizing, worked example defaults', height: 370,
  stage: { theta: 0.45, phi: 1.12, radius: 19, target: [0, 2.4, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    L.slider('P', 'Model, billions of weights', [1, 3, 8, 13, 34, 70, 180, 405], 70); L.slider('w', 'Bytes per weight', [.5, 1, 2], 1);
    L.slider('Ly', 'Layers', [16, 24, 32, 48, 64, 80, 96, 126], 80); L.slider('G', 'KV groups', [1, 2, 4, 8, 16, 32], 8); L.slider('seq', 'Context', [1024, 2048, 4096, 8192, 16384, 32768, 131072], 8192, fmt);
    L.slider('kv', 'Bytes per KV element', [.5, 1, 2], 1); L.slider('B', 'Users per system', [1, 8, 32, 64, 128, 256, 512, 1024], 256); L.slider('r', 'Tokens/s per user', [10, 20, 30, 50, 100, 200], 50);
    L.select('chip', 'Chip', [['iw', 'Ironwood FP8: 4.6 PF, 7.3 TB/s, 192 GiB'], ['gb', 'GB300 FP4: 15 PF, 8 TB/s, 288 GB'], ['v5', 'TPU v5p: 459 TF, 2.8 TB/s, 96 GiB'], ['ed', 'Modest: 500 TF, 1 TB/s, 64 GB']], 'iw');
    const C = { iw: [4614e12, 7.3e12, 206e9, 600e9], gb: [15000e12, 8e12, 288e9, 1800e9], v5: [459e12, 2.765e12, 103e9, 600e9], ed: [500e12, 1e12, 64e9, 200e9] };
    const dyn = new THREE.Group(); g.add(dyn);
    L.render = () => {
      clearGroup(dyn); const { P, w, Ly, G, seq, kv, B, r } = s, [pk, bw, cap, net] = C[s.chip], wB = P * 1e9 * w, kvU = 2 * Ly * seq * G * 128 * kv, kvT = kvU * B, step = 1 / r;
      const cBW = (wB + kvT) / step / bw, cCap = (wB + kvT) / cap, flops = 2 * P * 1e9 * B * r, cCmp = flops / (pk * .6), chips = Math.max(1, Math.ceil(Math.max(cBW, cCap, cCmp) * 1.15));
      const I = 2 * P * 1e9 * B / (wB + kvT), ridge = pk / bw, d = Math.round(Math.sqrt(P * 1e9 / (12 * Ly)) / 128) * 128 || 1024, X = d * B * 2, netBW = chips > 1 ? 2 * Ly * CM.allReduceBytes(X, chips) / step : 0;
      const vals = [['Compute', cCmp, '--poly'], ['Bandwidth', cBW, '--diff'], ['Capacity', cCap, '--via']], mx = Math.max(...vals.map(v => v[1]), .1);
      vals.forEach(([n, v, c], i) => { const h = Math.max(.05, v / mx * 5.2), x = (i - 1) * 3.2; dyn.add(at(box(2, h, 2, tcol(c), { alpha: .92 }), x, h / 2, 0)); const a = label(n, { scale: .85 }); at(a, x, -.45, 1.4); dyn.add(a); const b = label(fmt(v, v < 10 ? 1 : 0) + ' chips', { scale: .78, mono: true }); at(b, x, h + .45, 0); dyn.add(b); });
      dyn.add(at(box(11, .05, 4, tcol('--line'), { alpha: .6 }), 0, 0, 0));
      const bind = vals.slice().sort((a, b) => b[1] - a[1])[0][0];
      L.stats([['Weights', si(wB, 'B')], ['KV per user', si(kvU, 'B')], ['KV total', si(kvT, 'B')], ['Traffic needed', si((wB + kvT) * r, 'B/s')], ['Intensity', fmt(I) + ' F/B', I < ridge ? 'bad' : 'good'], ['Compute', si(flops, 'FLOP/s')], ['Chips, 15% headroom', fmt(chips)], ['Network per chip', si(netBW, 'B/s'), netBW > net ? 'bad' : '']]);
      L.note(`${bind} sets the chip count here. ${kvT > wB ? `KV traffic is ${fmt(kvT / wB, 1)}× weight traffic, so batching has stopped helping and KV is what you are buying.` : 'Weights dominate traffic, so more batch is still almost free.'} Intensity is ${fmt(I)} FLOP per byte against a ridge of ${fmt(ridge)}, so this system is ${I < ridge ? 'memory-bound and its arrays will sit idle much of the time' : 'compute-bound'}. Try doubling context, halving KV groups, or dropping KV to 4 bits. Compute assumes 60% utilization, and d_model is estimated from model size.`);
    };
  },
});

defineLab('token', {
  title: 'One token through your chip', sub: 'array beats, waiting and ROM on the 1 × 16 design', height: 380,
  stage: { theta: 0.5, phi: 1.12, radius: 19, target: [0, 2.6, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    const FFN = { dense: { E: 1, k: 1, h: 256 }, big: { E: 4, k: 1, h: 256 }, small: { E: 4, k: 1, h: 64 }, two: { E: 4, k: 2, h: 64 }, fine: { E: 8, k: 1, h: 32 } };
    L.select('ffn', 'FFN', [['dense', 'Dense, width 256'], ['big', 'MoE: 4 experts, top-1, width 256'], ['small', 'MoE: 4 experts, top-1, width 64'], ['two', 'MoE: 4 experts, top-2, width 64'], ['fine', 'MoE: 8 experts, top-1, width 32']], 'dense');
    L.slider('W', 'Vector lanes', [1, 2, 4, 8, 16], 4); L.slider('T', 'Context', [8, 16, 32], 32);
    L.select('kv', 'K and V heads', [['2', 'Two, as planned'], ['1', 'One, shared']], '2');
    const dyn = new THREE.Group(); g.add(dyn);
    L.render = () => {
      clearGroup(dyn);
      const o = { W: s.W, T: s.T, moe: FFN[s.ffn], kvHeads: +s.kv }, a = CM.tokenPlan(o), b = CM.tokenPlan({ ...o, deferLN: true }), base = CM.tokenPlan({ W: s.W, T: s.T });
      const unit = 6.4 / 6200, rom = a.weights / 320000 * 6.4;
      const tower = (x, name, total, array) => {
        const ah = array * unit, wh = Math.max(.02, (total - array) * unit);
        dyn.add(at(box(2, ah, 2, tcol('--m1'), { alpha: .92 }), x, ah / 2, 0));
        dyn.add(at(box(2, wh, 2, tcol('--via'), { alpha: .92 }), x, ah + wh / 2, 0));
        const lb = label(name, { scale: .72 }); at(lb, x, -.55, 1.3); dyn.add(lb);
        const lv = label(fmt(total), { scale: .78 }); at(lv, x, ah + wh + 1.15, 0); dyn.add(lv);
      };
      tower(-5, 'One step at a time', a.serial, a.arrayBeats); tower(-1.8, 'Overlapped', a.overlapped, a.arrayBeats); tower(1.4, 'LN deferred', b.overlapped, b.arrayBeats);
      dyn.add(at(box(2, rom, 2, tcol('--diff'), { alpha: .92 }), 5, rom / 2, 0));
      const lr = label('Weight ROM', { scale: .72 }); at(lr, 5, -.55, 1.3); dyn.add(lr);
      const lk = label(fmt(a.weights / 1024, 0) + ' KiB', { scale: .78 }); at(lk, 5, rom + 1.15, 0); dyn.add(lk);
      dyn.add(at(box(14, .05, 4, tcol('--line'), { alpha: .6 }), 0, 0, 0));
      const over = a.overlapped / a.arrayBeats - 1, vs = a.overlapped / base.overlapped - 1;
      L.stats([['Array beats', fmt(a.arrayBeats)], ['Beats per token', fmt(a.overlapped)], ['Waiting on top', pct(over), over > .12 ? 'bad' : over > .08 ? 'warn' : 'good'], ['Against dense', (vs > 0 ? '+' : '') + pct(vs), vs < 0 ? 'good' : ''], ['Weight bytes', fmt(a.weights)]]);
      const moeNote = s.ffn === 'dense' ? 'Pick an MoE option to see that speed follows the experts you run, while ROM follows the experts you store.' : `Speed follows the ${FFN[s.ffn].k} expert${FFN[s.ffn].k > 1 ? 's' : ''} run per token. ROM follows all ${FFN[s.ffn].E} stored. Whether the model stays good has to be tested in your Python golden.`;
      L.note(`Blue is the array working, amber is the array waiting. Done one step at a time, this token takes ${fmt(a.serial)} beats. Overlapped, it takes ${fmt(a.overlapped)}. ${moeNote}`);
    };
  },
});
