// Pure analytical chip model shared by the site labs and the test suite.
// Teaching model only: no synthesis, STA or measured silicon behind it.
// GEMM naming everywhere: C[M x N] = A[M x K] * B[K x N].
//   M = tokens/rows of activations, K = reduction depth, N = output features.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ChipModel = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  const ceil = Math.ceil, log2 = Math.log2;

  // ---- Systolic array utilization -------------------------------------------------
  // Output-stationary R x C array. Matches tile_engine.sv: one tile = K + R + C - 2 cycles.
  function osTile(R, C, K) { return K + R + C - 2; }
  function osGemm(M, N, K, R, C, overlap) {
    const tiles = ceil(M / R) * ceil(N / C);
    const perTile = overlap ? Math.max(K, 1) : osTile(R, C, K);
    const cycles = tiles * perTile + (overlap ? R + C - 2 : 0);
    const useful = M * N * K;
    return { tiles, cycles, useful, util: useful / (cycles * R * C), padUtil: (M * N) / (tiles * R * C) };
  }
  // Weight-stationary array: R rows hold a K slice, C columns hold an N slice.
  // rho = weight-load bandwidth / activation bandwidth (Roune's bw_ratio inverted).
  // A tile holds its weights for max(M, R/rho) cycles when loads are double buffered.
  function wsGemm(M, N, K, R, C, rho) {
    rho = rho || 1;
    const tiles = ceil(K / R) * ceil(N / C);
    const perTile = Math.max(M, R / rho);
    const cycles = tiles * perTile + R + C - 2;
    const useful = M * N * K;
    return { tiles, cycles, useful, util: useful / (cycles * R * C), steady: (M * N * K) / (tiles * perTile * R * C) };
  }
  // Beats until the farthest PE sees its first operands, by how operands reach it.
  // hop: both operands hop from the edges. bcast: one operand reaches a whole row in one beat.
  // bcast2: both do. diag: Axon-style, injected on the diagonal and spreading both ways.
  function arrival(r, c, feed) {
    return feed === 'bcast' ? r : feed === 'bcast2' ? 0 : feed === 'diag' ? Math.abs(r - c) : r + c;
  }
  function fillBeats(R, C, feed) {
    if (feed === 'bcast') return R - 1;
    if (feed === 'bcast2') return 0;
    if (feed === 'diag') return Math.max(R, C) - 1;
    return R + C - 2;
  }
  // Output-stationary array of Rc x Cc cells. Each cell does Tm x Tn x Tk MACs per beat:
  // Tm rows and Tn columns of outputs, Tk reduction elements. Tk is the grouped PE's G.
  // T tiles run back to back if overlap is set. Readout shifts one value per column per beat.
  function blockedOs(o) {
    const { Rc, Cc, K } = o, Tm = o.Tm || 1, Tn = o.Tn || 1, Tk = o.Tk || 1, T = o.T || 1, feed = o.feed || 'hop';
    const stream = ceil(K / Tk), fill = fillBeats(Rc, Cc, feed), R = Rc * Tm, C = Cc * Tn;
    const macs = Rc * Cc * Tm * Tn * Tk;
    const perTile = o.overlap ? Math.max(stream, R) : stream + fill;
    const beats = o.overlap ? T * perTile + fill : T * perTile;
    const useful = T * R * C * K;
    const opFlops = 8 * (Tm + Tn) * Tk, accFlops = 32 * Tm * Tn * (o.overlap ? 2 : 1);
    return { stream, fill, R, C, macs, perTile, beats, util: useful / (beats * macs),
      edge: Rc * Tm * Tk + Cc * Tn * Tk, acc: R * C, flopsPerMac: (opFlops + accFlops) / (Tm * Tn * Tk),
      treeDepth: Math.ceil(Math.log2(Tk)), fanout: Math.max(Tm, Tn) };
  }
  // Delay registers in the edge triangle: row r needs r delays.
  function skewRegs(n) { return n * (n - 1) / 2; }
  function minRowsForFull(Rdim, rho) { return Rdim / (rho || 1); }
  // Block-diagonal trick from Roune: a width-w problem on a width-W array.
  function narrowK(W, w, blockDiagonal) {
    const f = w / W;
    return blockDiagonal ? Math.min(1, f) : f * f;
  }

  // ---- Accumulation ---------------------------------------------------------------
  function accBits(aBits, bBits, K) { return aBits + bBits + ceil(log2(K)); }
  function chainDepth(n) { return n - 1; }
  function treeDepth(n) { return ceil(log2(n)); }

  // ---- Fabrics --------------------------------------------------------------------
  function crossbarCrosspoints(P) { return P * P; }
  function ringAvgHops(P) { return P / 4; }
  function torus1dAvgHops(w) { return w / 4; }              // Roune: uniform random, even w
  function meshAvgHops(k) { return 2 * (k * k - 1) / (3 * k); }
  function torusAvgHops(dims) { return dims.reduce((s, w) => s + w / 4, 0); }
  function routerAvgHops(P) { return 2 * (P - 1) / P; }     // self-traffic costs 0
  function usefulBw(linkUtil, allLinkBw, avgHops) { return linkUtil * allLinkBw / avgHops; }
  // Uniform random requests: expected banks serving per cycle.
  function bankThroughput(P, B) { return B * (1 - Math.pow(1 - 1 / B, P)); }
  function torusBisection(dims, linkBw) {
    const sorted = dims.slice().sort((a, b) => a - b);
    const area = sorted[0] * sorted[1];          // cut across the longest dimension
    return 2 * area * linkBw;
  }

  // ---- Roofline and workloads -----------------------------------------------------
  function ridge(peakFlops, bw) { return peakFlops / bw; }
  function attainable(intensity, peakFlops, bw) { return Math.min(peakFlops, intensity * bw); }
  function ffIntensity(batch, bytesPerWeight) { return 2 * batch / bytesPerWeight; }
  function decodeBatchForRidge(ridgeFlopPerByte, bytesPerWeight) { return ridgeFlopPerByte * bytesPerWeight / 2; }
  function attnDecodeIntensity(queriesPerKv, bytesPerElem) { return 2 * queriesPerKv / bytesPerElem; }
  function tileIntensity(Tm, Tn, bytes) { return 2 * Tm * Tn / ((Tm + Tn) * bytes); }
  function kvBytes(o) {
    return 2 * o.batch * o.layers * o.seq * o.groups * o.width * o.bytes * (o.idle || 1) * (o.pipe || 1);
  }
  function allReduceBytes(X, P) { return 2 * X * (1 - 1 / P); }

  // ---- Software pipelining --------------------------------------------------------
  function pipeSteps(N, S) { return N + S - 1; }
  function pipeUtil(N, S) { return N / (N + S - 1); }

  // ---- Power, voltage, frequency (alpha-power law teaching curve) -----------------
  function freqAtVoltage(V, Vt, a, V0, f0) {
    const g = v => Math.pow(v - Vt, a) / v;
    return f0 * g(V) / g(V0);
  }
  function voltageForFreq(f, Vt, a, V0, f0) {
    let lo = Vt + 0.02, hi = 1.3;
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2;
      if (freqAtVoltage(m, Vt, a, V0, f0) < f) lo = m; else hi = m;
    }
    return (lo + hi) / 2;
  }
  function dynEnergyScale(V, V0) { return (V / V0) * (V / V0); }
  function yieldPoisson(areaMm2, d0PerCm2) { return Math.exp(-(areaMm2 / 100) * d0PerCm2); }

  // ---- Published TPU table (Jouppi et al. 2026, Table 1) --------------------------
  const tpu = [
    { name: 'TPU v2', year: 2017, bf16: 46, fp8: null, mxu: 2, dim: 128, vmem: 32, stacks: 2, hbmGiB: 16, hbmBW: 700, tcs: 2, sc: 2, pod: 256, topo: '2D', links: 4, linkBW: 62, bisect: 1984, relPerf: 1, relPerfW: 1, relTDP: 1 },
    { name: 'TPU v3', year: 2018, bf16: 123, fp8: null, mxu: 4, dim: 128, vmem: 32, stacks: 4, hbmGiB: 32, hbmBW: 900, tcs: 2, sc: 2, pod: 1024, topo: '2D', links: 4, linkBW: 70, bisect: 4480, relPerf: 10, relPerfW: 1.8, relTDP: 5.6 },
    { name: 'TPU v4', year: 2021, bf16: 275, fp8: null, mxu: 8, dim: 128, vmem: 32, stacks: 4, hbmGiB: 32, hbmBW: 1200, tcs: 2, sc: 4, pod: 4096, topo: '3D', links: 6, linkBW: 50, bisect: 25600, relPerf: 100, relPerfW: 4.9, relTDP: 20 },
    { name: 'TPU v5p', year: 2023, bf16: 459, fp8: 459, mxu: 8, dim: 128, vmem: 128, stacks: 6, hbmGiB: 96, hbmBW: 2765, tcs: 2, sc: 4, pod: 8960, topo: '3D', links: 6, linkBW: 100, bisect: 64000, relPerf: 350, relPerfW: 5.2, relTDP: 67 },
    { name: 'Ironwood', year: 2025, bf16: 2307, fp8: 4614, mxu: 4, dim: 256, vmem: 128, stacks: 8, hbmGiB: 192, hbmBW: 7300, tcs: 2, sc: 4, pod: 9216, topo: '3D', links: 6, linkBW: 100, bisect: 76800, relPerf: 3600, relPerfW: 29.3, relTDP: 123 },
  ];
  function impliedClockGHz(t) { return t.bf16 * 1e12 / (t.mxu * t.dim * t.dim * 2) / 1e9; }
  function derive(t) {
    return {
      clockGHz: impliedClockGHz(t),
      ridgeBf16: t.bf16 * 1e12 / (t.hbmBW * 1e9),
      vmemPerTf: t.vmem / t.bf16,
      hbmGiBPerTf: t.hbmGiB / t.bf16,
      iciPerTf: t.links * t.linkBW / t.bf16,
      batchForRidge: decodeBatchForRidge(t.bf16 * 1e12 / (t.hbmBW * 1e9), 2),
    };
  }

  // ---- One decoded token, beat by beat, including the work around the array ---------
  // The chip: 1 x C array, Tk multipliers per PE, x broadcast, so no fill. Defaults are the
  // recommended 1 x 16, Tk = 4 from Systolic arrays, at context T = 32.
  // Vector side: W elements per beat through one edge pipeline (requant, table, residual, stats).
  // Assumptions: pipe = beats through that pipeline, rsq = beats for one 1/sqrt (seed plus Newton
  // on one multiplier), one output buffer behind the accumulators, so the array stalls if the
  // vector side is more than one pass behind.
  // moe: { E, k, h } replaces the FFN. E = 1 with no router is the dense model.
  function tokenPlan(o) {
    o = o || {};
    const C = o.C || 16, Tk = o.Tk || 4, W = o.W || 1, T = o.T || 32, pipe = o.pipe === undefined ? 3 : o.pipe, rsq = o.rsq === undefined ? 8 : o.rsq;
    const L = o.layers || 2, d = o.d || 64, heads = o.heads || 2, dh = o.dh || 32, vocab = o.vocab || 256, kvHeads = o.kvHeads || heads;
    const moe = o.moe || { E: 1, k: 1, h: o.h || 256 }, router = moe.E > 1;
    const qkv = d + 2 * kvHeads * dh, deferLN = !!o.deferLN;
    let tA = 0, tV = 0, arrayBeats = 0, ops = 0, lastDone = 0;
    const mm = (N, K, dep) => {
      let done = dep;
      for (let p = 0; p < ceil(N / C); p++) {
        const n = Math.min(C, N - p * C), beats = ceil(K / Tk);
        let end = Math.max(tA, p ? 0 : dep) + beats;
        end = Math.max(end, lastDone);            // one buffer: wait for the pass before last to leave
        lastDone = tV;
        tA = end; arrayBeats += beats;
        tV = Math.max(tV, end) + n / W; done = tV + pipe;
      }
      return done;
    };
    const vec = (n, dep) => { tV = Math.max(tV, dep) + n / W; return tV + pipe; };
    // The array eats Tk normalized values a beat. Fewer lanes than that throttles the first pass.
    const norm = x => (deferLN ? x : x + rsq + Math.max(0, d / W - d / Tk));
    let x = 0;
    for (let l = 0; l < L; l++) {
      const proj = mm(qkv, d, norm(x));
      const sc = [], ex = [];
      for (let h = 0; h < heads; h++) sc.push(mm(T, dh, proj));
      for (let h = 0; h < heads; h++) ex.push(vec(T, sc[h]));
      let av = 0;
      for (let h = 0; h < heads; h++) av = mm(dh, T, ex[h]);
      const a = mm(d, d, av);
      let dep = norm(a);   // the router and the first expert both read this normalized vector
      if (router) dep = mm(moe.E, d, norm(a)) + 1;
      let y = dep;
      for (let e = 0; e < moe.k; e++) { const u = mm(moe.h, d, y); y = mm(d, moe.h, u); }
      x = y;
    }
    const end = mm(vocab, d, norm(x));
    // Same work with nothing overlapped: every step walks its vector once, then the next step starts.
    const outputs = L * (qkv + heads * T + heads * dh + d + moe.k * (moe.h + d) + (router ? moe.E : 0)) + vocab;
    const lnSteps = (2 * L + 1) * d * 3, smSteps = L * heads * T * 4, actSteps = L * moe.k * moe.h, resSteps = 2 * L * d, argmax = vocab;
    const vectorOps = outputs + lnSteps + smSteps + actSteps + resSteps + argmax;
    const serial = arrayBeats + vectorOps / W + (2 * L + 1) * rsq + L * heads * rsq;
    const weights = L * (d * qkv + d * d + moe.E * 2 * d * moe.h + (router ? d * moe.E : 0)) + vocab * d;
    const total = Math.ceil(end - 1e-6);
    return { arrayBeats, overlapped: total, exposed: total - arrayBeats, serial: Math.ceil(serial), vectorOps, outputs, weights, macsPerBeat: C * Tk };
  }

  return { tokenPlan, osTile, osGemm, wsGemm, arrival, fillBeats, blockedOs, skewRegs, minRowsForFull, narrowK, accBits, chainDepth, treeDepth,
    crossbarCrosspoints, ringAvgHops, torus1dAvgHops, meshAvgHops, torusAvgHops, routerAvgHops, usefulBw,
    bankThroughput, torusBisection, ridge, attainable, ffIntensity, decodeBatchForRidge, attnDecodeIntensity,
    tileIntensity, kvBytes, allReduceBytes, pipeSteps, pipeUtil, freqAtVoltage, voltageForFreq, dynEnergyScale,
    yieldPoisson, tpu, impliedClockGHz, derive };
});
