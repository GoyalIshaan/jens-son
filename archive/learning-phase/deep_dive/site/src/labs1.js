// Labs: systolic wavefront, matmul box, reduction structures, fabrics.
const P2 = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16];

function systolicScene(L, s, stage, auto) {
  const g = new THREE.Group(); stage.root.add(g);
  const cap = 16 * 16, tok = 900;
  const peM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(0xffffff, { rough: 0.45 }), cap);
  const accM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(0xffffff, { rough: 0.35, metal: 0.3 }), cap);
  const aM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--m1')), tok);
  const bM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--poly')), tok);
  [peM, accM, aM, bM].forEach(m => { m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.frustumCulled = false; g.add(m); });
  const cOff = tcol('--line'), cHot = tcol('--via'), cAcc = tcol('--diff'), cOn = tcol('--m1');
  const pitch = 1, tmp = new THREE.Color();
  const sim = { t: -1.5, playing: !REDUCED, hold: 0 };
  function total() { const K = s.K, R = s.R, C = s.C; return K + R + C - 2; }
  function draw() {
    const R = s.R, C = s.C, K = s.K, ws = s.flow === 'ws', t = sim.t, ti = Math.floor(t), fr = t - ti;
    let i = 0, ai = 0, bi = 0, active = 0, macs = 0;
    const x0 = -(C - 1) / 2 * pitch, z0 = -(R - 1) / 2 * pitch;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const k = ti - r - c, on = k >= 0 && k < K;
      const done = clamp(ti - r - c + 1, 0, K); macs += done; if (on) active++;
      const x = x0 + c * pitch, z = z0 + r * pitch;
      setI(peM, i, x, 0, z, 0.82, 0.14, 0.82);
      tmp.copy(cOff); if (on) tmp.lerp(cHot, 0.85); peM.setColorAt(i, tmp);
      if (ws) { setI(accM, i, x, 0.2, z, 0.34, 0.26, 0.34); tmp.copy(cAcc); accM.setColorAt(i, tmp); }
      else { const h = 0.02 + 1.7 * done / K; setI(accM, i, x, 0.08 + h / 2, z, 0.42, h, 0.42); tmp.copy(cAcc).lerp(cOn, done / K * 0.35); accM.setColorAt(i, tmp); }
      if (on) {
        setI(aM, ai++, x - pitch * (1 - fr) * 0.9 + 0.08, 0.3, z - 0.16, 0.2, 0.13, 0.13);
        if (ws) { const sz = 0.1 + 0.16 * (r + 1) / R; setI(bM, bi++, x + 0.16, 0.3, z - pitch * (1 - fr) * 0.9 + 0.08, sz, sz, 0.2); }
        else setI(bM, bi++, x + 0.16, 0.3, z - pitch * (1 - fr) * 0.9 + 0.08, 0.13, 0.13, 0.2);
      }
      i++;
    }
    // Waiting operands form the skew triangle at the array edges.
    const vis = Math.min(K, 14);
    for (let r = 0; r < R && ai < tok; r++) for (let k = 0; k < vis; k++) {
      const wait = k + r - t; if (wait <= 0 || ai >= tok) continue;
      setI(aM, ai++, x0 - 0.8 - wait * 0.42, 0.3, z0 + r * pitch - 0.16, 0.26, 0.13, 0.13);
    }
    if (!ws) for (let c = 0; c < C && bi < tok; c++) for (let k = 0; k < vis; k++) {
      const wait = k + c - t; if (wait <= 0 || bi >= tok) continue;
      setI(bM, bi++, x0 + c * pitch + 0.16, 0.3, z0 - 0.8 - wait * 0.42, 0.13, 0.13, 0.26);
    }
    peM.count = R * C; accM.count = R * C; aM.count = ai; bM.count = bi;
    [peM, accM, aM, bM].forEach(m => { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; });
    return { active, macs, cyc: Math.max(0, ti + 1) };
  }
  stage.onFrame = dt => {
    if (sim.playing) {
      sim.t += dt * (s.speed || 3);
      if (sim.t > total() + 0.2) { sim.hold += dt; sim.t = total() + 0.2; if (sim.hold > 1.6) { sim.t = -1.5; sim.hold = 0; } }
    }
    const st = draw(); if (L && L.tick) L.tick(st); return sim.playing;
  };
  sim.draw = draw; sim.total = total; return sim;
}

defineLab('hero', {
  build() {}, title: '', hero: true,
});

defineLab('systolic', {
  title: 'Systolic wavefront', sub: 'schedule identical to tile_engine.sv: k = t − r − c', height: 430,
  stage: { theta: 0.62, phi: 0.82, radius: 22, target: [0, 0.2, 0] },
  build(L) {
    const s = L.state;
    L.select('flow', 'Dataflow', [['os', 'Output-stationary (your RTL)'], ['ws', 'Weight-stationary (TPU)']], 'os');
    L.slider('R', 'Rows R', P2, 8); L.slider('C', 'Columns C', P2, 8);
    L.slider('K', 'Depth: K, or rows M if weight-stationary', [1, 2, 4, 8, 12, 16, 24, 32, 48, 64], 16);
    L.slider('speed', 'Cycles per second', [1, 2, 3, 5, 8, 12], 3);
    const sim = systolicScene(L, s, L.stage, false);
    const [pb] = L.buttons([[sim.playing ? 'Pause' : 'Play', b => { sim.playing = !sim.playing; b.textContent = sim.playing ? 'Pause' : 'Play'; }],
      ['Step', () => { sim.playing = false; pb.textContent = 'Play'; sim.t = Math.floor(sim.t) + 1; sim.draw(); }],
      ['Restart', () => { sim.t = -1.5; sim.hold = 0; }]]);
    L.tick = st => {
      const T = sim.total(), R = s.R, C = s.C, K = s.K, peak = st.cyc * R * C, sus = K / T;
      L.stats([['Cycle', `${Math.min(st.cyc, T)} / ${T}`], ['PEs active now', `${st.active} of ${R * C}`], ['MACs done', fmt(st.macs)],
        ['Utilization so far', peak ? pct(st.macs / peak) : '0%'], ['Whole tile utilization', pct(sus), sus > .7 ? 'good' : sus < .3 ? 'bad' : 'warn']]);
    };
    L.render = () => {
      const R = s.R, C = s.C, K = s.K, T = sim.total(), u = K / T;
      L.note(s.flow === 'ws'
        ? `Green posts are weights, parked in place. Blue activations flow right, and amber-red partial sums grow as they flow down. Each column is a pipelined adder chain. ${K} rows keep ${R}×${C} PEs busy for ${pct(u)} of ${T} cycles, so the array wants far more rows than this before fill and drain fade away.`
        : `Blue A values flow right and red B values flow down, skewed so element k reaches PE(r,c) at cycle k+r+c. Green towers are accumulators. Total time is K+R+C−2 = ${T} cycles, so this isolated tile runs at ${pct(u)}. The diagonal band of amber PEs is the wavefront. Raise K and the fill and drain cost shrinks. Raise R and C at fixed K and it grows.`);
    };
  },
});

defineLab('beat', {
  title: 'Beat lab', sub: 'fill, overlap and packing: same MAC budget, different cells', height: 430,
  stage: { theta: 0.62, phi: 0.8, radius: 24, target: [0, 0.2, 0] },
  build(L) {
    const s = L.state, g = new THREE.Group(); L.stage.root.add(g);
    const cap = 1024;
    const macM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(0xffffff, { rough: 0.45 }), cap);
    const plateM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(0xffffff, { rough: 0.7 }), cap);
    const aM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--m1')), 64);
    const bM = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--poly')), 64);
    [macM, plateM, aM, bM].forEach(m => { m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.frustumCulled = false; g.add(m); });
    const cOff = tcol('--line'), cA = tcol('--via'), cB = tcol('--diff'), cFeed = tcol('--m1'), tmp = new THREE.Color();
    const sim = { t: -1.5, playing: !REDUCED, hold: 0 };
    L.select('feed', 'How operands reach a cell', [['hop', 'Hop cell to cell (systolic)'], ['bcast', 'Broadcast along each row'], ['diag', 'Feed from the diagonal (Axon)'], ['bcast2', 'Broadcast both']], 'hop');
    L.select('mode', 'Tiles', [['iso', 'One at a time, like tile_engine.sv'], ['ovl', 'Back to back']], 'iso');
    L.slider('P', 'MAC budget', [64, 256, 1024], 256, v => fmt(v));
    L.slider('tm', 'Tm = Tn: outputs per cell side', [1, 2, 4], 1);
    L.slider('tk', 'Tk: K elements per beat', [1, 2, 4, 8], 1);
    L.slider('K', 'Reduction depth K', [8, 16, 32, 64, 128, 256], 32);
    L.slider('T', 'Tiles in the stream', [1, 2, 3, 4, 8], 3);
    L.slider('speed', 'Beats per second', [2, 4, 8, 16, 32], 8);
    const shape = () => {
      const n = Math.max(0, Math.log2(s.P / (s.tm * s.tm * s.tk)));
      return { Rc: 2 ** Math.floor(n / 2), Cc: 2 ** Math.ceil(n / 2) };
    };
    const calc = () => { const { Rc, Cc } = shape(); return CM.blockedOs({ Rc, Cc, Tm: s.tm, Tn: s.tm, Tk: s.tk, K: s.K, T: s.T, feed: s.feed, overlap: s.mode === 'ovl' }); };
    function draw() {
      const { Rc, Cc } = shape(), f = calc(), tm = s.tm, tk = s.tk, t = sim.t, ti = Math.floor(t);
      const gap = 0.34, cw = tm + gap, x0 = -(Cc * cw - gap) / 2 + 0.5, z0 = -(Rc * cw - gap) / 2 + 0.5, lh = 0.2;
      let i = 0, p = 0, busy = 0, a = 0, b = 0;
      for (let r = 0; r < Rc; r++) for (let c = 0; c < Cc; c++) {
        const arr = CM.arrival(r, c, s.feed); let tile = -1;
        for (let n = 0; n < s.T; n++) {
          const k = ti - n * f.perTile - arr;
          if (k >= 0 && k < f.stream) tile = n;
        }
        if (tile >= 0) busy++;
        const cx = x0 + c * cw + (tm - 1) / 2, cz = z0 + r * cw + (tm - 1) / 2;
        setI(plateM, p, cx, -0.1, cz, tm + 0.14, 0.08, tm + 0.14);
        tmp.copy(cOff); if (s.feed === 'diag' && r === c) tmp.lerp(cFeed, 0.8); plateM.setColorAt(p++, tmp);
        for (let u = 0; u < tm; u++) for (let v = 0; v < tm; v++) for (let w = 0; w < tk && i < cap; w++) {
          setI(macM, i, x0 + c * cw + v, 0.06 + w * lh, z0 + r * cw + u, 0.82, lh * 0.72, 0.82);
          tmp.copy(cOff); if (tile >= 0) tmp.copy(tile % 2 ? cB : cA); macM.setColorAt(i++, tmp);
        }
      }
      // Edge bars grow with the operands each cell row or column takes per beat.
      const len = 0.35 * tm * tk;
      if (s.feed !== 'diag') {
        for (let r = 0; r < Rc && a < 64; r++) setI(aM, a++, x0 - 0.9 - len / 2, 0.1, z0 + r * cw + (tm - 1) / 2, len, 0.16, 0.3 * tm);
        for (let c = 0; c < Cc && b < 64; c++) setI(bM, b++, x0 + c * cw + (tm - 1) / 2, 0.1, z0 - 0.9 - len / 2, 0.3 * tm, 0.16, len);
      }
      macM.count = i; plateM.count = p; aM.count = a; bM.count = b;
      [macM, plateM, aM, bM].forEach(m => { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; });
      return { busy, cells: Rc * Cc, beat: Math.max(0, ti + 1), f };
    }
    const show = st => {
      const f = st.f, slow = s.mode === 'ovl' && f.perTile > f.stream;
      L.stats([['Beat', `${Math.min(st.beat, f.beats)} / ${f.beats}`], ['Cells busy now', `${st.busy} of ${st.cells}`], ['Fill per ramp', `${f.fill} beats`],
        ['Whole-run utilization', pct(f.util), f.util > .8 ? 'good' : f.util < .4 ? 'bad' : 'warn'], ['Operands in per beat', fmt(f.edge)], ['Flops per MAC', fmt(f.flopsPerMac), slow ? 'warn' : '']]);
    };
    L.stage.onFrame = dt => {
      const T = calc().beats;
      if (sim.playing) {
        sim.t += dt * s.speed;
        if (sim.t > T + 0.2) { sim.hold += dt; sim.t = T + 0.2; if (sim.hold > 1.6) { sim.t = -1.5; sim.hold = 0; } }
      }
      show(draw()); return sim.playing;
    };
    const [pb] = L.buttons([[sim.playing ? 'Pause' : 'Play', b => { sim.playing = !sim.playing; b.textContent = sim.playing ? 'Pause' : 'Play'; }],
      ['Step', () => { sim.playing = false; pb.textContent = 'Play'; sim.t = Math.floor(sim.t) + 1; show(draw()); }],
      ['Restart', () => { sim.t = -1.5; sim.hold = 0; }]]);
    L.render = () => {
      sim.t = -1.5; sim.hold = 0;
      const { Rc, Cc } = shape(), f = calc();
      L.stage.radius = 10 + 1.5 * Math.max(Rc, Cc) * (s.tm + 0.34);
      const over = f.macs > s.P ? ` This cell is bigger than the budget, so the array grew to ${fmt(f.macs)} MACs.` : '';
      const rd = s.mode === 'ovl' && f.perTile > f.stream ? ` Readout now limits each tile to ${f.perTile} beats, because a column shifts ${f.R} results but K÷Tk is only ${f.stream}.` : '';
      const flow = s.mode === 'ovl' ? `The next tile follows one wave behind, in the other color, so the ${f.fill}-beat ramp is paid once for all ${s.T} tiles.` : `Each tile waits for the last one to finish, so the ${f.fill}-beat ramp is paid ${s.T} times.`;
      L.note(`${Rc}×${Cc} cells, each ${s.tm}×${s.tm}×${s.tk} MACs: a ${f.R}×${f.C} output tile. Stacked layers are Tk, and the bars at the edge show how wide each feed is. K=${s.K} takes ${f.stream} beats per tile. ${flow}${rd}${over}`);
    };
  },
});

defineLab('cube', {
  title: 'The matmul box', sub: 'weight-stationary tiles over C[M×N] = A[M×K] · B[K×N]', height: 440,
  stage: { theta: 0.75, phi: 1.0, radius: 26, target: [0, 1.2, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    const presets = {
      ffp: ['FF up, prefill', 2048, 32768, 8192], ffd: ['FF up, decode, batch 256', 256, 32768, 8192], ffd1: ['FF up, decode, batch 1', 1, 4096, 4096],
      qk: ['Attention QKᵀ, decode (64 queries)', 64, 8192, 128], qkp: ['Attention QKᵀ, prefill', 2048, 2048, 128], m257: ['Awkward: 257 output columns', 256, 257, 4096],
      img: ['First image layer, K=3', 4096, 64, 3], dw: ['Depthwise 3×3, K=9', 4096, 64, 9], mine: ['Your GPT-2: QKV in decode', 1, 192, 64], mine2: ['Your GPT-2: FFN in prefill', 32, 256, 64],
    };
    L.select('preset', 'Shape preset', Object.entries(presets).map(([k, v]) => [k, v[0]]), 'qk');
    const Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], Ns = [16, 64, 128, 192, 256, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768], Ks = [3, 9, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    L.slider('M', 'M, rows (tokens × batch)', Ms, 64, fmt); L.slider('N', 'N, output features', Ns, 8192, fmt); L.slider('K', 'K, reduction depth', Ks, 128, fmt);
    L.slider('A', 'Array side', [8, 16, 32, 64, 128, 256, 512], 128); L.slider('rho', 'Weight bandwidth ρ', [0.5, 1, 2, 4, 8, 16, 32], 1, v => v + '×');
    L.els.preset.sel.addEventListener('change', () => { const p = presets[s.preset]; L.setSlider('M', p[1]); L.setSlider('N', p[2]); L.setSlider('K', p[3]); L.render(); });
    const CAP = 12, unit = 0.92;
    const useful = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(0xffffff, {}), CAP * CAP);
    const pad = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--poly'), { alpha: 0.2 }), CAP * CAP);
    const cur = at(box(1, 1, 1, tcol('--via'), { alpha: 0.5 }), 0, 0, 0);
    [useful, pad].forEach(m => { m.frustumCulled = false; g.add(m); });
    g.add(cur);
    const dyn = new THREE.Group(); g.add(dyn);
    let sweep = 0, geo = null;
    L.stage.onFrame = dt => {
      if (!geo || REDUCED) return false;
      sweep += dt * 2.2; const n = geo.tn * geo.tk, i = Math.floor(sweep) % n;
      const a = i % geo.tn, b = Math.floor(i / geo.tn) % geo.tk;
      const fx = a === geo.tn - 1 ? geo.fn : 1, fz = b === geo.tk - 1 ? geo.fk : 1;
      cur.position.set(geo.x0 + a * geo.px - (1 - fx) * unit * geo.px / 2, geo.h * geo.fm / 2 + 0.02, geo.z0 + b * geo.pz - (1 - fz) * unit * geo.pz / 2);
      cur.scale.set(unit * geo.px * fx * 1.03, geo.h * geo.fm * 1.02 + 0.03, unit * geo.pz * fz * 1.03); return true;
    };
    L.render = () => {
      const { M, N, K, A, rho } = s;
      const g0 = CM.wsGemm(M, N, K, A, A, rho), tk = Math.ceil(K / A), tn = Math.ceil(N / A), per = Math.max(M, A / rho);
      const dk = Math.min(tk, CAP), dn = Math.min(tn, CAP), span = 11, px = span / Math.max(dn, dk), pz = px, h = 3.8;
      const fk = K / (tk * A), kLast = (K - (tk - 1) * A) / A, nLast = (N - (tn - 1) * A) / A, fm = M / per;
      let i = 0, j = 0; const x0 = -(dn - 1) / 2 * px, z0 = -(dk - 1) / 2 * pz, c1 = tcol('--m1'), c2 = tcol('--diff');
      for (let b = 0; b < dk; b++) for (let a = 0; a < dn; a++) {
        const fx = a === dn - 1 ? nLast : 1, fz = b === dk - 1 ? kLast : 1, x = x0 + a * px, z = z0 + b * pz;
        setI(useful, i, x - (1 - fx) * unit * px / 2, h * fm / 2, z - (1 - fz) * unit * pz / 2, unit * px * fx, h * fm, unit * pz * fz);
        useful.setColorAt(i, (a + b) % 2 ? c1 : c2); i++;
        if (fx < 1 || fz < 1 || fm < 1) { setI(pad, j++, x, h / 2, z, unit * px, h, unit * pz); }
      }
      useful.count = i; pad.count = j; useful.instanceMatrix.needsUpdate = true; pad.instanceMatrix.needsUpdate = true; if (useful.instanceColor) useful.instanceColor.needsUpdate = true;
      clearGroup(dyn);
      const lx = label(`N → ${fmt(N)}${tn > dn ? ` (${tn} tiles, showing ${dn})` : ''}`, { scale: .78 }); at(lx, 0, -0.5, span / 2 + 1.3); dyn.add(lx);
      const lz = label(`K → ${fmt(K)}${tk > dk ? ` (${tk} tiles)` : ''}`, { scale: .78 }); at(lz, -span / 2 - 2.1, -0.5, 0); dyn.add(lz);
      const ly = label(`rows per weight tile: ${fmt(M)} of ${fmt(per)}`, { scale: .72 }); at(ly, 0, h + 0.9, -span / 2); dyn.add(ly);
      geo = { tn: dn, tk: dk, fn: nLast, fk: kLast, fm, x0, z0, px, pz, h }; sweep = 0;
      const kill = fk < 1 ? `K fills ${pct(fk)} of the array's rows` : null, nFill = N / (tn * A), mFill = M / per;
      const worst = Math.min(fk, nFill, mFill);
      const rows = CM.minRowsForFull(A, rho), cycles = g0.cycles;
      L.stats([['Steady utilization', pct(g0.steady), g0.steady > .75 ? 'good' : g0.steady < .25 ? 'bad' : 'warn'], ['K fill', pct(fk), fk < .75 ? 'bad' : ''], ['N fill', pct(nFill), nFill < .75 ? 'bad' : ''], ['M fill', pct(mFill), mFill < .75 ? 'bad' : ''],
        ['Weight tiles', fmt(tk * tn)], ['Rows needed', fmt(rows)], ['Array cycles', fmt(cycles)]]);
      const why = worst === mFill && mFill < 1 ? `M is the limit. Each weight tile needs ${fmt(rows)} rows to hide its load, and you have ${fmt(M)}. The red volume above the blocks is time the array spends waiting for weights. More batch or more weight bandwidth fixes this.`
        : worst === fk && fk < 1 ? `${kill}, and K is the one dimension batching cannot fix. Red volume is PEs with no terms left to add. A smaller array or heterogeneous cores are the real answers.`
        : worst === nFill && nFill < 1 ? `N leaves padding in the last column of tiles. That is the 257-versus-256 tax. Pad the model dimension, or choose a width that divides it.`
        : `Every dimension fills the array. This is the feed-forward case that makes large arrays attractive.`;
      L.note(why + ` The amber block walks the tile order. Compare array cycles as you change array side: utilization alone can mislead, since a smaller array can have better utilization and take longer.`);
    };
  },
});

defineLab('reduction', {
  title: 'Chain, tree and grouped reduction', sub: 'same adders, different depth and wiring', height: 400,
  stage: { theta: 0.3, phi: 1.22, radius: 19, target: [0, 3.6, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    L.select('mode', 'Structure', [['chain', 'Chain (array column)'], ['tree', 'Tree (dot-product unit)'], ['group', 'Grouped: small trees, then chain']], 'tree');
    L.slider('n', 'Inputs to sum', [8, 16, 32, 64, 128], 32); L.slider('G', 'Group size G', [2, 4, 8, 16], 4);
    L.slider('ab', 'Operand bits', [4, 8, 16], 8);
    const nodes = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 10), mat(0xffffff), 300); nodes.frustumCulled = false; g.add(nodes);
    let wires = null, levels = [], maxL = 1, t = 0; const tmp = new THREE.Color(), cA = tcol('--line'), cB = tcol('--via'), cD = tcol('--m1');
    function paint() {
      levels.forEach((lv, i) => { tmp.copy(lv <= t ? cB : lv === 0 ? cD : cA); nodes.setColorAt(i, tmp); });
      if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;
    }
    L.stage.onFrame = dt => { if (REDUCED) return false; t += dt * Math.max(1.2, maxL / 5); if (t > maxL + 1.5) t = -0.5; paint(); return true; };
    L.render = () => {
      const { mode, n, G, ab } = s; const pts = []; const pos = []; levels = [];
      const W = 15, H = 8, add = (x, y, z, l) => { pos.push([x, y, z]); levels.push(l); return pos.length - 1; };
      let leaves = []; for (let i = 0; i < n; i++) leaves.push(add((i - (n - 1) / 2) * W / n, 0, Math.sin(i * 1.7) * 0.8, 0));
      let depth = 0, longest = 0, flops = 0;
      const link = (a, b) => { pts.push(...pos[a], ...pos[b]); const d = Math.hypot(pos[a][0] - pos[b][0], pos[a][2] - pos[b][2]) / (W / n); longest = Math.max(longest, d); };
      function tree(ids, l0) { let cur = ids, l = l0; while (cur.length > 1) { l++; const nx = []; for (let i = 0; i < cur.length; i += 2) { if (i + 1 >= cur.length) { nx.push(cur[i]); continue; } const a = pos[cur[i]], b = pos[cur[i + 1]]; const id = add((a[0] + b[0]) / 2, l, (a[2] + b[2]) / 2 * 0.8, l); link(cur[i], id); link(cur[i + 1], id); nx.push(id); } flops += nx.length; cur = nx; } return [cur[0], l]; }
      if (mode === 'tree') { const [, l] = tree(leaves, 0); depth = l; }
      else if (mode === 'chain') { let acc = leaves[0]; for (let i = 1; i < n; i++) { const b = pos[leaves[i]]; const id = add(b[0], i, b[2] * 0.3, i); link(acc, id); link(leaves[i], id); acc = id; flops++; } depth = n - 1; }
      else { const heads = []; let td = 0; for (let i = 0; i < n; i += G) { const [h, l] = tree(leaves.slice(i, i + G), 0); heads.push(h); td = l; } let acc = heads[0], l = td; for (let i = 1; i < heads.length; i++) { l++; const b = pos[heads[i]]; const id = add(b[0], l, b[2] * 0.5, l); link(acc, id); link(heads[i], id); acc = id; flops++; } depth = l; }
      maxL = Math.max(1, depth); const ys = H / maxL;
      pos.forEach((p, i) => { p[1] *= ys; setI(nodes, i, p[0], p[1], p[2], .19, .19, .19); });
      nodes.count = pos.length; nodes.instanceMatrix.needsUpdate = true;
      for (let i = 1; i < pts.length; i += 3) pts[i] *= ys;
      if (wires) { g.remove(wires); wires.geometry.dispose(); wires.material.dispose(); }
      wires = segs(pts, tcol('--ink'), .85); g.add(wires); paint();
      const adders = n - 1, acc = CM.accBits(ab, ab, n), flopsPerMac = 2 * ab + (mode === 'chain' ? 4 * ab : mode === 'group' ? 4 * ab / G : 4 * ab / n);
      L.stats([['Adders', fmt(adders)], ['Depth in adders', fmt(depth), depth > 32 ? 'warn' : 'good'], ['Longest wire, in PE pitches', fmt(longest, 1), longest > 8 ? 'warn' : 'good'], ['Exact sum width', acc + ' bits'], ['Flops per MAC, one accumulator per group', fmt(flopsPerMac, 1)]]);
      const msg = { chain: `A chain has depth ${depth}, and every wire stays one pitch long. This is one column of a weight-stationary array. Pipelined, it delivers a result every cycle after a ${depth}-cycle fill.`, tree: `A tree has depth ${depth} with the same ${adders} adders, and its top-level wires span ${fmt(longest, 0)} pitches. Good for latency, hard to route at large fan-in.`, group: `Groups of ${G} use short trees, then a chain joins them. You share one accumulator between ${G} multipliers, trading critical path for fewer flops. Most real matrix engines sit near here.` };
      L.note(msg[mode]);
    };
  },
});

defineLab('fabric', {
  title: 'Fabrics in 3D', sub: 'ports, wires, hops and bank conflicts', height: 420,
  stage: { theta: 0.55, phi: 0.8, radius: 24, target: [0, 0, 0] },
  build(L) {
    const s = L.state, g = L.stage.root;
    L.select('topo', 'Topology', [['ring', 'Ring'], ['mesh', 'Mesh'], ['torus', 'Torus'], ['xbar', 'Crossbar']], 'xbar');
    L.slider('P', 'Ports', [4, 9, 16, 25, 36, 49, 64], 16); L.slider('W', 'Port width, bits', [64, 128, 256, 512, 1024], 512);
    L.slider('B', 'SRAM banks behind the fabric', [8, 16, 32, 64, 128, 256], 32);
    const nodes = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--m1')), 140), xp = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(tcol('--via')), 4200), pulses = new THREE.InstancedMesh(new THREE.SphereGeometry(.16, 10, 8), mat(tcol('--poly')), 60);
    [nodes, xp, pulses].forEach(m => { m.frustumCulled = false; g.add(m); });
    let wires = null, P = [], adj = null, routes = [], live = [];
    function route(a, b) { const seen = new Map([[a, -1]]), q = [a]; while (q.length) { const u = q.shift(); if (u === b) break; for (const v of adj[u]) if (!seen.has(v)) { seen.set(v, u); q.push(v); } } const p = []; for (let u = b; u !== -1 && u !== undefined; u = seen.get(u)) p.unshift(u); return p; }
    L.stage.onFrame = dt => {
      if (REDUCED || !P.length) return false;
      while (live.length < 36) { const a = Math.floor(Math.random() * P.length), b = Math.floor(Math.random() * P.length); if (a === b) continue; live.push({ r: s.topo === 'xbar' ? null : route(a, b), a, b, t: 0 }); }
      let i = 0; live = live.filter(p => { p.t += dt * 2.6; const n = p.r ? p.r.length - 1 : 1; return p.t < n; });
      for (const p of live) { let x, z, y = .35; if (p.r) { const k = Math.floor(p.t), f = p.t - k, A = P[p.r[k]], B = P[p.r[k + 1]]; x = A[0] + (B[0] - A[0]) * f; z = A[2] + (B[2] - A[2]) * f; } else { const A = P[p.a], B = routes[p.b], f = p.t; x = A[0] + (B[0] - A[0]) * f; z = A[2] + (B[2] - A[2]) * f; y = .1 + A[1] * (1 - f) + B[1] * f; } setI(pulses, i++, x, y, z); }
      doneI(pulses, i); return true;
    };
    L.render = () => {
      const { topo, W, B } = s; let Pn = s.P; const k = Math.round(Math.sqrt(Pn)); P = []; adj = []; live = []; routes = [];
      const pts = []; let xc = 0; const span = 13;
      if (topo === 'ring') { for (let i = 0; i < Pn; i++) { const a = i / Pn * Math.PI * 2; P.push([Math.cos(a) * 6.5, 0, Math.sin(a) * 6.5]); adj.push([(i + 1) % Pn, (i + Pn - 1) % Pn]); } for (let i = 0; i < Pn; i++) pts.push(...P[i], ...P[(i + 1) % Pn]); }
      else if (topo === 'mesh' || topo === 'torus') {
        const wrap = topo === 'torus', p = span / Math.max(k - 1, 1); Pn = k * k;
        for (let r = 0; r < k; r++) for (let c = 0; c < k; c++) { P.push([(c - (k - 1) / 2) * p, 0, (r - (k - 1) / 2) * p]); const n = []; if (c + 1 < k) n.push(r * k + c + 1); else if (wrap) n.push(r * k); if (c > 0) n.push(r * k + c - 1); else if (wrap) n.push(r * k + k - 1); if (r + 1 < k) n.push((r + 1) * k + c); else if (wrap) n.push(c); if (r > 0) n.push((r - 1) * k + c); else if (wrap) n.push((k - 1) * k + c); adj.push(n); }
        for (let u = 0; u < Pn; u++) for (const v of adj[u]) if (u < v) { const A = P[u], Bp = P[v], far = Math.hypot(A[0] - Bp[0], A[2] - Bp[2]) > p * 1.5; if (far) { const lift = 2.2; pts.push(...A, A[0], lift, A[2], A[0], lift, A[2], Bp[0], lift, Bp[2], Bp[0], lift, Bp[2], ...Bp); } else pts.push(...A, ...Bp); }
      } else {
        const p = span / Math.max(Pn - 1, 1);
        for (let i = 0; i < Pn; i++) { P.push([-6.5, 0, (i - (Pn - 1) / 2) * p]); routes.push([6.5, 1.1, (i - (Pn - 1) / 2) * p]); }
        for (let i = 0; i < Pn; i++) for (let j = 0; j < Pn; j++) { const xx = -5.6 + 11.2 * (j + .5) / Pn; const q = Math.min(p, 11.2 / Pn) * .34; setI(xp, xc++, xx, .55, P[i][2], q, q * .7, q); }
        for (let i = 0; i < Pn; i++) { pts.push(...P[i], 5.9, .55, P[i][2]); const xx = -5.6 + 11.2 * (i + .5) / Pn; pts.push(xx, .55, -span / 2 - .3, xx, .55, span / 2 + .3); }
      }
      doneI(xp, xc); const sz = topo === 'xbar' ? Math.min(.5, 8 / Pn) : .5; P.forEach((q, i) => setI(nodes, i, q[0], q[1], q[2], .7, .36, topo === 'xbar' ? sz : .7));
      let cnt = P.length; if (topo === 'xbar') routes.forEach((q, i) => setI(nodes, cnt++, q[0], q[1], q[2], .7, .36, sz)); doneI(nodes, cnt);
      if (wires) { g.remove(wires); wires.geometry.dispose(); wires.material.dispose(); } wires = segs(pts, tcol('--ink'), .6); g.add(wires);
      const hops = topo === 'ring' ? CM.ringAvgHops(Pn) : topo === 'mesh' ? CM.meshAvgHops(k) : topo === 'torus' ? CM.torusAvgHops([k, k]) : 1;
      const links = topo === 'ring' ? Pn : topo === 'mesh' ? 2 * k * (k - 1) : topo === 'torus' ? 2 * k * k : Pn * Pn;
      const bis = topo === 'ring' ? 2 : topo === 'mesh' ? k : topo === 'torus' ? 2 * k : Pn / 2;
      const thr = Math.min(1, CM.bankThroughput(Pn, B) / Pn), mmw = Pn * W * 80e-6 / 4;
      L.stats([['Ports', fmt(Pn)], [topo === 'xbar' ? 'Crosspoints' : 'Links', fmt(links), topo === 'xbar' && links > 1000 ? 'bad' : ''], ['Average hops', fmt(hops, 2)], ['Bisection, in ports', fmt(bis)], ['Wire width if flat crossbar', mmw.toFixed(2) + ' mm', mmw > 1 ? 'warn' : ''], ['Random bank throughput', pct(thr, 0), thr < .7 ? 'bad' : thr > .85 ? 'good' : 'warn']]);
      const m = { ring: `A ring has ${links} links and ${fmt(hops, 1)} average hops, with only 2 links across any cut. Cheap, and it fills up fast as ports grow.`, mesh: `A ${k}×${k} mesh keeps wires short. Average distance is ${fmt(hops, 1)} hops, and its edges have no wraparound, so collectives run at half the speed of a torus.`, torus: `Wrap links lift the bisection to ${bis} and cut average distance to ${fmt(hops, 1)} hops, at the cost of long links. Folded layouts shorten them.`, xbar: `One hop for everyone, at ${fmt(links)} crosspoints. Doubling ports quadruples this. With ${fmt(Pn)} ports of ${W} bits, the wires alone need about ${mmw.toFixed(2)} mm at 80 nm pitch on 4 layers.` };
      L.note(m[topo] + ` Behind the fabric, ${B} banks give ${pct(thr, 0)} throughput under random access. Well-scheduled matmul access needs no such margin, since software can avoid conflicts.`);
    };
  },
});
