// Integer experiments behind the "LUTs, nonlinear units and MoE" chapter.
// Every number quoted in docs/18_nonlinear.md that is not a pure count comes from here.
// Run: node deep_dive/model/nonlinear.js   Teaching experiments on synthetic data, not a trained model.
const rng = (s => () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32)(12345);
const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))); // GPT-2's form
const clip = (v, a, b) => Math.max(a, Math.min(b, v));
const out = {};

// ---- 1. GELU table on int8 ---------------------------------------------------------
{
  const s = 6 / 127; // same scale in and out, inputs cover about -6..6
  const tbl = Array.from({ length: 256 }, (_, i) => clip(Math.round(gelu((i - 128) * s) / s), -128, 127));
  let worst = 0, trivial = 0;
  for (let i = 0; i < 256; i++) {
    const q = i - 128;
    worst = Math.max(worst, Math.abs(tbl[i] - gelu(q * s) / s));
    if (tbl[i] === q || tbl[i] === 0) trivial++;
  }
  out.gelu = { entries: 256, bytes: 256, worstLSB: worst, trivial, interesting: 256 - trivial, sample: [-20, 0, 20, 60].map(q => [q, +(q * s).toFixed(3), tbl[q + 128]]) };
}

// ---- 2. Wider inputs: direct table, then straight-line segments --------------------
{
  const lo = -6, hi = 6, rows = [];
  const lsb8 = 6 / 127;
  for (const seg of [8, 16, 32, 64, 128]) {
    const w = (hi - lo) / seg, knots = Array.from({ length: seg + 1 }, (_, i) => gelu(lo + i * w));
    let worst = 0;
    for (let i = 0; i <= 200000; i++) {
      const x = lo + (hi - lo) * i / 200000, j = Math.min(seg - 1, Math.floor((x - lo) / w)), f = (x - lo) / w - j;
      worst = Math.max(worst, Math.abs(knots[j] + (knots[j + 1] - knots[j]) * f - gelu(x)));
    }
    rows.push({ seg, tableBytes: (seg + 1) * 2, worstAbs: +worst.toFixed(5), worstLSB8: +(worst / lsb8).toFixed(3) });
  }
  out.interp = { rows, direct16bitIn16bitOutBytes: 65536 * 2 };
}

// ---- 3. Reciprocal and inverse square root: seed table plus Newton -----------------
{
  const run = (bits, steps, f, newton, lo, hi) => {
    const n = 2 ** bits, w = (hi - lo) / n, seed = Array.from({ length: n }, (_, i) => f(lo + (i + .5) * w));
    let worst = 0;
    for (let i = 0; i < 100000; i++) {
      const x = lo + (hi - lo) * (i + .5) / 100000; let y = seed[Math.min(n - 1, Math.floor((x - lo) / w))];
      for (let k = 0; k < steps; k++) y = newton(x, y);
      worst = Math.max(worst, Math.abs(y - f(x)) / f(x));
    }
    return worst;
  };
  const rec = (x, y) => y * (2 - x * y), rsq = (x, y) => y * (3 - x * y * y) / 2;
  out.newton = {};
  for (const bits of [3, 4, 5]) for (const st of [0, 1, 2]) {
    out.newton[`recip ${2 ** bits} entries, ${st} steps`] = +run(bits, st, x => 1 / x, rec, 1, 2).toExponential(2);
    out.newton[`rsqrt ${2 ** bits} entries, ${st} steps`] = +run(bits, st, x => 1 / Math.sqrt(x), rsq, 1, 4).toExponential(2);
  }
}

// ---- 4. Softmax into the int8 array: normalize first or divide last? ---------------
// The array is signed int8, so the "probabilities" fed to it must fit in 0..127.
{
  const T = 32, trials = 20000, s = 8 / 127; // scores cover about -8..8
  const expT = Array.from({ length: 256 }, (_, d) => Math.round(127 * Math.exp(-d * s)));
  const nz = expT.filter(v => v > 0).length;
  const res = {};
  for (const [name, spread] of [['flat', 0.5], ['medium', 2], ['peaky', 5]]) {
    let eF = 0, eD = 0, sumF = 0, sumD = 0;
    for (let t = 0; t < trials; t++) {
      const sc = Array.from({ length: T }, () => clip(Math.round(gauss() * spread / s), -128, 127));
      const v = Array.from({ length: T }, () => clip(Math.round(gauss() * 40), -128, 127));
      const m = Math.max(...sc), p = sc.map(x => Math.exp((x - m) * s)), Z = p.reduce((a, b) => a + b);
      const truth = p.reduce((a, x, i) => a + x * v[i], 0) / Z;
      const e = sc.map(x => expT[m - x]), S = e.reduce((a, b) => a + b);
      // normalize first: p_i = e_i/S scaled to 0..127, then array, then undo the 127.
      const q = e.map(x => Math.round(127 * x / S));
      const first = q.reduce((a, x, i) => a + x * v[i], 0) / 127;
      // divide last: raw e_i into the array, one divide at the end.
      const last = e.reduce((a, x, i) => a + x * v[i], 0) / S;
      const a = Math.abs(first - truth), b = Math.abs(last - truth);
      sumF += a; sumD += b; eF = Math.max(eF, a); eD = Math.max(eD, b);
    }
    res[name] = { meanFirst: +(sumF / trials).toFixed(3), worstFirst: +eF.toFixed(2), meanLast: +(sumD / trials).toFixed(3), worstLast: +eD.toFixed(2) };
  }
  out.softmax = { nonzeroEntries: nz, firstTen: expT.slice(0, 10), res };
}

// ---- 5. Base-2 exp: a shift and a 16-entry table -----------------------------------
{
  const frac = Array.from({ length: 16 }, (_, i) => 2 ** (-(i + .5) / 16));
  let worst = 0;
  for (let i = 0; i < 200000; i++) {
    const x = 12 * i / 200000, y = x * Math.LOG2E, n = Math.floor(y), f = y - n;
    worst = Math.max(worst, Math.abs(frac[Math.floor(f * 16)] * 2 ** -n - Math.exp(-x)));
  }
  out.exp2 = { entries: 16, worstAbs: +worst.toFixed(4), inUnitsOf127: +(worst * 127).toFixed(2) };
}

// ---- 6. LayerNorm: exact one-pass statistics, and matmul before the statistics -----
{
  let maxDiff = 0, exactVar = true;
  for (let t = 0; t < 2000; t++) {
    const d = 64, x = Array.from({ length: d }, () => clip(Math.round(gauss() * 30), -128, 127));
    const sum = x.reduce((a, b) => a + b, 0), sq = x.reduce((a, b) => a + b * b, 0);
    const onePass = d * sq - sum * sum; // d^2 * variance, integers, no cancellation error
    const twoPass = x.reduce((a, b) => a + (d * b - sum) ** 2, 0) / d;
    if (onePass !== twoPass) exactVar = false;
    // W(x - mu) * r  ==  r * (W x - mu * colsum(W))
    const w = Array.from({ length: d }, () => clip(Math.round(gauss() * 30), -128, 127));
    const mu = sum / d, r = 1 / Math.sqrt(onePass / d / d + 1e-5);
    const before = x.reduce((a, v, i) => a + (v - mu) * r * w[i], 0);
    const after = r * (x.reduce((a, v, i) => a + v * w[i], 0) - mu * w.reduce((a, b) => a + b));
    maxDiff = Math.max(maxDiff, Math.abs(before - after));
  }
  out.layernorm = { onePassExact: exactVar, deferredIdentityMaxDiff: maxDiff, sumSqBitsWorst: Math.ceil(Math.log2(64 * 128 * 128)) + 1 };
}

// ---- 7. MoE prefill on an array with rows: padding from uneven routing -------------
{
  const trials = 200000, res = {};
  for (const [R, E] of [[8, 4], [4, 4], [8, 8], [1, 4]]) {
    let tot = 0, worst = 0;
    for (let t = 0; t < trials; t++) {
      const n = new Array(E).fill(0);
      for (let i = 0; i < 32; i++) n[Math.floor(rng() * E)]++;
      const tiles = n.reduce((a, c) => a + Math.ceil(c / R), 0);
      tot += tiles; worst = Math.max(worst, tiles);
    }
    res[`R${R} E${E}`] = { denseTiles: 32 / R, meanTiles: +(tot / trials).toFixed(2), worst, overhead: +((tot / trials) / (32 / R)).toFixed(2) };
  }
  out.moePrefill = res;
}

// ---- 8. Router near-ties: how often does 1 LSB of error change the expert? ---------
{
  const res = {};
  for (const sigma of [10, 20, 40]) {
    let flips = 0; const trials = 400000;
    for (let t = 0; t < trials; t++) {
      const l = [0, 0, 0, 0].map(() => Math.round(gauss() * sigma)).sort((a, b) => b - a);
      if (l[0] - l[1] <= 1) flips++;
    }
    res[`sigma ${sigma}`] = +(100 * flips / trials).toFixed(2);
  }
  out.routerGapAtMost1LSBPercent = res;
}

// ---- 9. Gumbel-max: sampling with no softmax ---------------------------------------
{
  const logits = [2, 1, 0, -1], Z = logits.reduce((a, l) => a + Math.exp(l), 0), n = 1000000, c = [0, 0, 0, 0];
  for (let t = 0; t < n; t++) {
    let b = 0, bv = -Infinity;
    logits.forEach((l, i) => { const v = l - Math.log(-Math.log(rng() + 1e-12)); if (v > bv) { bv = v; b = i; } });
    c[b]++;
  }
  out.gumbel = { want: logits.map(l => +(Math.exp(l) / Z).toFixed(4)), got: c.map(v => +(v / n).toFixed(4)) };
}

console.log(JSON.stringify(out, null, 1));
