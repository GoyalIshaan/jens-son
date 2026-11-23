// Recomputes every numeric answer without using the stored value, by enumeration where possible,
// and runs the beat simulators for the timing claims. Run: node deep_dive/quiz/verify.js
'use strict';
const { TOPICS, QUESTIONS, shuffleFor } = require('./questions.js');
const { simOs, simWs } = require('../model/sim_array.js');
const mk = (r, c) => Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => (i * 3 + j * 5) % 7 - 3));
const pes = (R, C, f) => { let n = 0; for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) if (f(r, c)) n++; return n; };
const busy = (r, c, K, b) => b >= r + c && b <= r + c + K - 1;
const peak = (R, C, K) => { let m = 0; for (let b = 0; b < K + R + C; b++) m = Math.max(m, pes(R, C, (r, c) => busy(r, c, K, b))); return m; };
const firstK = (test) => { for (let k = 1; k < 1e6; k++) if (test(k)) return k; };
const firstT = firstK;
const osBeats = (R, C, K, T = 1) => simOs({ Rc: R, Cc: C, K, tiles: Array.from({ length: T }, () => ({ A: mk(R, K), B: mk(K, C) })) }).beats;
const tri = n => { let s = 0; for (let r = 0; r < n; r++) s += r; return s; };
const calc = {
  t1: () => 3 + 2 + 6 - 1,
  t2: () => pes(5, 3, (r, c) => busy(r, c, 6, 3)),
  t3: () => peak(4, 4, 4),
  t4: () => firstK(k => peak(6, 4, k) === 24),
  t6: () => { const b = osBeats(8, 8, 10); return 64 * b - 64 * 10; },
  t7: () => firstK(k => k / (k + 30) >= 0.9),
  d1: () => { const r = simWs({ R: 8, C: 8, M: 1, tiles: [{ W: mk(8, 8), X: mk(1, 8) }] }); console.log('  simWs 8x8 M=1 beats', r.beats, 'formula', 1 + 14); return 1 + 128 + 128 - 2; },
  d3: () => 1024 / 256,
  d5: () => 96 / 256 * 100,
  d7: () => 4 * (8 + 8),
  f2: () => { for (let k = 1; k < 2000; k++) if (Math.abs((k + 126) * 0.5 - (k + 63) * 0.625) < 1e-9) return k; },
  f3: () => Math.floor(((800 - 350) / 100) / 0.012 + 1e-9),
  f5: () => (512 + 510) / (512 + 255),
  f6: () => 8 * 8 * (2 + 2) * 8,
  f7: () => { for (let S = 1; S <= 128; S *= 2) if (128 / S + Math.log2(S) <= 12) return S; },
  f9: () => (128 + 510) / (128 + 254),
  o2: () => firstT(T => T * 256 / (T * 256 + 254) >= 0.9),
  o3: () => pes(8, 8, (r, c) => r + c <= 9 && r + c + 9 >= 10),
  o6: () => firstK(L => 128 / L <= 32),
  o7: () => 100 / Math.max(100, 256) * 100,
  p1: () => { const r = simOs({ Rc: 4, Cc: 4, Tk: 4, K: 30, tiles: [{ A: mk(4, 30), B: mk(30, 4) }] }); return r.beats; },
  p3: () => (4 * 2 + 8 * 1) * 2,
  p4: () => 2 * 4 * 2 * 2,
  p5: () => Math.ceil(48 / 16) * Math.ceil(96 / 4),
  p6: () => { const c = 1 * 9 + 7 * 2; if (256 * 2 + 16 * c + 63 !== 23 * 41) throw new Error('p6'); return c; },
  p7: () => 64 / (2 * 2),
  p8: () => 8 * (2 + 2) / (2 * 2) + 32 / 2 + 32 / 2,
  s1: () => tri(32) * 2 * 8,
  s2: () => [0, 0, 1, 1, 2, 2, 3, 3].reduce((a, b) => a + b),
  s4: () => tri(256) * 8,
  g1: () => 32 / 64 * 100,
  g2: () => 1920 + 13 * (32 - 1),
};
let bad = 0;
for (const q of QUESTIONS.filter(q => q.type === 'n')) {
  if (!calc[q.id]) { console.log('NO CHECK', q.id); bad++; continue; }
  const v = calc[q.id](), ok = Math.abs(v - q.ans) <= (q.tol || 0) + 1e-9;
  if (!ok) { console.log('MISMATCH', q.id, 'stored', q.ans, 'computed', v); bad++; }
}
// Claims made inside choice questions and explanations.
const claims = [
  ['t1 sim: 5x3 K=6 tile beats', osBeats(5, 3, 6), 12],
  ['t8 ratio', +(16384 * 128 / 382 / (4096 * 128 / 254)).toFixed(1), 2.7],
  ['o1 stream', 512 + 254, 766],
  ['p2 back to back 8x8 sim', osBeats(8, 8, 64, 64), 4110],
  ['p2 back to back 4x4 Tk4 sim', simOs({ Rc: 4, Cc: 4, Tk: 4, K: 64, tiles: Array.from({ length: 256 }, () => ({ A: mk(4, 64), B: mk(64, 4) })) }).beats, 4102],
  ['p2 isolated', 64 * 78, 4992], ['p2 isolated tk', 256 * 22, 5632],
  ['o5 overwrites at K<R', simOs({ Rc: 8, Cc: 2, K: 2, tiles: [0, 1, 2, 3].map(() => ({ A: mk(8, 2), B: mk(2, 2) })) }).overwrites > 0, true],
  ['p9 identity', (3 + 4) * (5 + 2), 3 * 2 + 5 * 4 + 3 * 5 + 2 * 4],
];
for (const [n, a, b] of claims) if (a !== b) { console.log('CLAIM FAIL', n, a, b); bad++; }
// Structure.
const ids = new Set(QUESTIONS.map(q => q.id));
if (QUESTIONS.length !== 50 || ids.size !== 50) { console.log('COUNT', QUESTIONS.length, ids.size); bad++; }
const slots = [0, 0, 0, 0]; let longest = 0, ch = 0;
for (const q of QUESTIONS) {
  if (!TOPICS.includes(q.topic) || !q.hint || !q.why?.length || !q.see) { console.log('FIELDS', q.id); bad++; }
  if (q.type === 'c') { ch++; if (q.opts.length !== 4 || new Set(q.opts).size !== 4) { console.log('OPTS', q.id); bad++; }
    slots[shuffleFor(q.id, 4).indexOf(0)]++;
    if (q.opts[0].length === Math.max(...q.opts.map(o => o.length))) longest++; }
}
console.log('topics', TOPICS.map(t => t + ' ' + QUESTIONS.filter(q => q.topic === t).length).join(', '));
console.log('choice', ch, 'numeric', 50 - ch, '| right-answer slots', slots.join(' '), '| right answer is longest in', longest);
console.log(bad ? `FAILED: ${bad}` : 'ALL CHECKS PASSED');
process.exit(bad ? 1 : 0);
