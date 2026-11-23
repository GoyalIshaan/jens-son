// Beat-by-beat simulators used to check the formulas in model.js. Values really move, so a wrong
// schedule gives a wrong matrix. These are teaching models, not RTL.
'use strict';

// Output-stationary, Rc x Cc cells of Tm x Tn x Tk MACs, tiles back to back.
// `first` and `last` tags ride with the operands. At `last` each accumulator hands its sum to a
// shadow slot. Once a column's bottom cell has finished, the column shifts out one value per beat.
function simOs({ Rc, Cc, Tm = 1, Tn = 1, Tk = 1, K, tiles }) {
  const R = Rc * Tm, C = Cc * Tn, kb = Math.ceil(K / Tk), T = tiles.length;
  const cell = () => ({ a: null, b: null });
  const g = Array.from({ length: Rc }, () => Array.from({ length: Cc }, cell));
  const acc = Array.from({ length: R }, () => new Array(C).fill(0));
  const shadow = Array.from({ length: R }, () => new Array(C).fill(null));
  const out = tiles.map(() => Array.from({ length: R }, () => new Array(C).fill(null)));
  const shifting = new Array(C).fill(0);
  let overwrites = 0, lastMac = -1, lastOut = -1, macBeats = 0;
  const aIn = (rc, s) => {
    if (s < 0 || s >= T * kb) return null;
    const t = Math.floor(s / kb), kk = s % kb, v = [];
    for (let i = 0; i < Tm; i++) for (let j = 0; j < Tk; j++) { const k = kk * Tk + j; v.push(k < K ? tiles[t].A[rc * Tm + i][k] : 0); }
    return { v, t, first: kk === 0, last: kk === kb - 1 };
  };
  const bIn = (cc, s) => {
    if (s < 0 || s >= T * kb) return null;
    const t = Math.floor(s / kb), kk = s % kb, v = [];
    for (let i = 0; i < Tn; i++) for (let j = 0; j < Tk; j++) { const k = kk * Tk + j; v.push(k < K ? tiles[t].B[k][cc * Tn + i] : 0); }
    return { v };
  };
  for (let beat = 0; beat < T * Math.max(kb, R) * 2 + 4 * (R + C) + 8; beat++) {
    // Readout first, so a slot emptied this beat can be refilled this beat.
    for (let c = 0; c < C; c++) if (shifting[c] > 0) {
      // A one-bit tag per slot: only the draining tile moves, so the next tile's early values stay put.
      const bottom = shadow[R - 1][c], dt = bottom && bottom.t;
      if (bottom) { out[bottom.t][bottom.r][c] = bottom.v; lastOut = beat; shadow[R - 1][c] = null; }
      for (let r = R - 1; r > 0; r--) { const v = shadow[r - 1][c]; if (v && v.t === dt) { shadow[r][c] = v; shadow[r - 1][c] = null; } }
      shifting[c]--;
    }
    // One hop per beat, far cells first. Skew at the edge is the s = beat - rc index.
    for (let rc = Rc - 1; rc >= 0; rc--) for (let cc = Cc - 1; cc >= 0; cc--) {
      g[rc][cc].a = cc ? g[rc][cc - 1].a : aIn(rc, beat - rc);
      g[rc][cc].b = rc ? g[rc - 1][cc].b : bIn(cc, beat - cc);
    }
    let any = false;
    for (let rc = 0; rc < Rc; rc++) for (let cc = 0; cc < Cc; cc++) {
      const { a, b } = g[rc][cc]; if (!a || !b) continue;
      any = true;
      for (let i = 0; i < Tm; i++) for (let n = 0; n < Tn; n++) {
        const r = rc * Tm + i, c = cc * Tn + n;
        let p = 0; for (let j = 0; j < Tk; j++) p += a.v[i * Tk + j] * b.v[n * Tk + j];
        acc[r][c] = a.first ? p : acc[r][c] + p;
        if (a.last) {
          if (shadow[r][c]) overwrites++;
          shadow[r][c] = { v: acc[r][c], t: a.t, r };
          if (r === R - 1) shifting[c] = R;
        }
      }
    }
    if (any) { lastMac = beat; macBeats++; }
  }
  return { out, beats: lastMac + 1, lastOut, overwrites };
}

// Weight-stationary R x C. Each tile holds weights W[R][C] and streams M activation rows.
// The first row of a tile carries a flip tag, so each PE swaps to its shadow weight as the wave
// reaches it. New weights enter a column right behind the flip, deepest row first.
function simWs({ R, C, M, tiles }) {
  const T = tiles.length, active = [], shadow = [], fresh = [];
  for (let r = 0; r < R; r++) { active.push(new Array(C).fill(null)); shadow.push(new Array(C).fill(null)); fresh.push(new Array(C).fill(false)); }
  const acts = Array.from({ length: R }, () => new Array(C).fill(null));
  const sums = Array.from({ length: R }, () => new Array(C).fill(null));
  const out = tiles.map(() => Array.from({ length: M }, () => new Array(C).fill(0)));
  let tokens = [], early = 0, lastOut = -1;
  const send = (t, c, beat) => { for (let j = 0; j < R; j++) tokens.push({ c, row: R - 1 - j, at: 0, start: beat + j, w: tiles[t].W[R - 1 - j][c] }); };
  for (let c = 0; c < C; c++) send(0, c, 0); // First load is exposed: R beats.
  const start = R;
  for (let beat = 0; beat < start + T * M + 2 * (R + C) + 4; beat++) {
    // Weight tokens fall one row per beat and park at their row.
    tokens = tokens.filter(k => {
      if (beat < k.start) return true;
      k.at = beat - k.start;
      if (k.at < k.row) return true;
      if (fresh[k.row][k.c]) early++;
      shadow[k.row][k.c] = k.w; fresh[k.row][k.c] = true;
      return false;
    });
    for (let r = R - 1; r >= 0; r--) for (let c = C - 1; c >= 0; c--) {
      let a;
      if (c) a = acts[r][c - 1];
      else { const s = beat - start - r; a = s >= 0 && s < T * M ? { v: tiles[Math.floor(s / M)].X[s % M][r], t: Math.floor(s / M), m: s % M, flip: s % M === 0 } : null; }
      acts[r][c] = a;
      if (!a) { sums[r][c] = null; continue; }
      if (a.flip) {
        if (!fresh[r][c]) early++;
        active[r][c] = shadow[r][c]; fresh[r][c] = false;
        if (r === 0 && a.t + 1 < T) send(a.t + 1, c, beat + 1);
      }
      const up = r ? sums[r - 1][c] : null;
      sums[r][c] = { v: (up ? up.v : 0) + a.v * active[r][c], t: a.t, m: a.m };
    }
    for (let c = C - 1; c >= 0; c--) { const s = sums[R - 1][c]; if (s) { out[s.t][s.m][c] = s.v; lastOut = beat; } }
  }
  return { out, beats: lastOut - start + 1, early };
}

module.exports = { simOs, simWs };
