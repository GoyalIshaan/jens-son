// Checks the model against every worked number quoted from the source PDFs.
const m = require('./model.js');
let pass = 0, fail = 0;
function eq(name, got, want, tol = 1e-6) {
  const ok = Math.abs(got - want) <= tol * Math.max(1, Math.abs(want));
  ok ? pass++ : fail++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${got}${ok ? '' : ' want ' + want}`);
}
// Roune: 64-wide vectors on a 128 array give 25%, block diagonal gives 50%.
eq('narrow K 64 on 128', m.narrowK(128, 64, false), 0.25);
eq('block diagonal 64 on 128', m.narrowK(128, 64, true), 0.5);
// Roune: N=1 on 128 with 1:1 bandwidth is 1/128.
eq('WS steady M=1 R=128', m.wsGemm(1, 128, 128, 128, 128, 1).steady, 1 / 128);
eq('WS steady M=128 R=128', m.wsGemm(128, 128, 128, 128, 128, 1).steady, 1);
eq('min rows, 2x weight BW', m.minRowsForFull(128, 2), 64);
// tile_engine.sv schedule.
eq('tile cycles 2x2x2', m.osTile(2, 2, 2), 4);
eq('tile util 2x2x2', m.osGemm(2, 2, 2, 2, 2, false).util, 0.5);
// Roune KV example: 4,194 GB and 4 GB after retrieval.
const kv = { batch: 64, layers: 32, seq: 1e6, groups: 8, width: 128, bytes: 1 };
eq('KV 1M tokens GB', m.kvBytes(kv) / 1e9, 4194.304);
eq('KV 1k tokens GB', m.kvBytes({ ...kv, seq: 1e3 }) / 1e9, 4.194304);
eq('KV over 32 chips GB', m.kvBytes(kv) / 1e9 / 32, 131.072);
// Roune MoE distances.
eq('torus 4-wide', m.torus1dAvgHops(4), 1);
eq('router 4 nodes', m.routerAvgHops(4), 1.5);
eq('4x4 torus', m.torusAvgHops([4, 4]), 2);
eq('4x4x4 torus', m.torusAvgHops([4, 4, 4]), 3);
eq('16x16x24 torus', m.torusAvgHops([16, 16, 24]), 14);
// All-reduce bytes.
eq('all-reduce P=2', m.allReduceBytes(1, 2), 1);
eq('all-reduce P=128', m.allReduceBytes(1, 128), 2 * 127 / 128);
// Software pipeline: 4 iterations of 3 stages = 6 steps, not 12.
eq('pipe steps', m.pipeSteps(4, 3), 6);
// Accumulator: 8x8 products over 65536 terms need exactly 32 bits.
eq('acc bits K=65536', m.accBits(8, 8, 65536), 32);
eq('acc bits K=256', m.accBits(8, 8, 256), 24);
eq('tree depth 256', m.treeDepth(256), 8);
// Decode: GQA and speculation. Queries per KV = 4 * 128 / 8 = 64.
eq('decode attention intensity', m.attnDecodeIntensity(64, 1), 128);
// Bisection reproduces Table 1 with implied pod shapes.
const shapes = { 'TPU v2': [16, 16], 'TPU v3': [32, 32], 'TPU v4': [16, 16, 16], 'TPU v5p': [16, 20, 28], Ironwood: [16, 24, 24] };
for (const t of m.tpu) {
  const s = shapes[t.name];
  eq(`${t.name} shape chips`, s.reduce((a, b) => a * b, 1), t.pod);
  eq(`${t.name} bisection`, s.length === 2 ? 2 * Math.min(...s) * t.linkBW : m.torusBisection(s, t.linkBW), t.bisect);
}
// Beat-level simulators: real values move, so these check results as well as timing.
const { simOs, simWs } = require('./sim_array');
let seed = 7; const rnd = () => (seed = seed * 1103515245 + 12345 & 0x7fffffff, (seed >> 8) % 17 - 8);
const mk = (a, b) => Array.from({ length: a }, () => Array.from({ length: b }, rnd));
const mul = (A, B) => A.map(r => B[0].map((_, c) => r.reduce((s, v, k) => s + v * B[k][c], 0)));
const same = (x, y) => JSON.stringify(x) === JSON.stringify(y);
const ok = (name, v) => eq(name, v ? 1 : 0, 1);
for (const [Rc, Cc, Tm, Tn, Tk, K, T] of [[3, 3, 1, 1, 1, 8, 3], [4, 2, 1, 1, 1, 8, 2], [2, 2, 2, 2, 1, 8, 3], [2, 2, 1, 1, 4, 8, 3], [2, 2, 2, 1, 2, 12, 2], [1, 4, 1, 1, 4, 16, 2]]) {
  const tiles = Array.from({ length: T }, () => ({ A: mk(Rc * Tm, K), B: mk(K, Cc * Tn) }));
  const r = simOs({ Rc, Cc, Tm, Tn, Tk, K, tiles }), f = m.blockedOs({ Rc, Cc, Tm, Tn, Tk, K, T, overlap: true });
  const name = `OS ${Rc}x${Cc} cell ${Tm}x${Tn}x${Tk} K=${K}`;
  ok(name + ' correct', tiles.every((t, i) => same(r.out[i], mul(t.A, t.B))));
  eq(name + ' beats', r.beats, f.beats);
  eq(name + ' no overwrite', r.overwrites, 0);
}
// Readout rule: K/Tk beats per tile must cover R shifts. One short of the limit must collide.
{
  const go = (R, K) => { const tiles = Array.from({ length: 4 }, () => ({ A: mk(R, K), B: mk(K, 3) })); return simOs({ Rc: R, Cc: 3, K, tiles }).overwrites; };
  eq('readout keeps up at K=R', go(6, 6), 0);
  ok('readout collides at K=R-1', go(6, 5) > 0);
}
for (const [R, C, M] of [[3, 3, 3], [4, 2, 6], [2, 5, 2]]) {
  const T = 3, tiles = Array.from({ length: T }, () => ({ W: mk(R, C), X: mk(M, R) }));
  const r = simWs({ R, C, M, tiles });
  ok(`WS ${R}x${C} M=${M} correct`, tiles.every((t, i) => same(r.out[i], mul(t.X, t.W))));
  eq(`WS ${R}x${C} M=${M} beats`, r.beats, T * M + R + C - 2);
  eq(`WS ${R}x${C} M=${M} swaps in time`, r.early, 0);
}
{
  const tiles = Array.from({ length: 3 }, () => ({ W: mk(4, 2), X: mk(3, 4) }));
  ok('WS swap misses at M=R-1', simWs({ R: 4, C: 2, M: 3, tiles }).early > 0);
}
// Each feed's closed-form fill equals the slowest PE's arrival.
for (const feed of ['hop', 'bcast', 'bcast2', 'diag']) for (const [R, C] of [[8, 8], [4, 16], [1, 64]]) {
  let w = 0; for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) w = Math.max(w, m.arrival(r, c, feed));
  eq(`fill ${feed} ${R}x${C}`, m.fillBeats(R, C, feed), w);
}
eq('grouped PE flops agree with reduction chapter', m.blockedOs({ Rc: 1, Cc: 1, Tk: 4, K: 4 }).flopsPerMac, 24);
eq('skew triangle 256', m.skewRegs(256), 32640);
// One token around the array: the plan must collapse to the known beat counts.
eq('token 1x16 Tk4 array beats', m.tokenPlan().arrayBeats, 1920);
eq('token 1x64 Tk1 array beats', m.tokenPlan({ C: 64, Tk: 1 }).arrayBeats, 2048);
eq('token 1x8 Tk8 array beats', m.tokenPlan({ C: 8, Tk: 8 }).arrayBeats, 1920);
eq('token weights match 114,688', m.tokenPlan().weights, 114688);
eq('token vector steps', m.tokenPlan().vectorOps, 4160);
eq('free vector side hides everything', m.tokenPlan({ W: 1e9, pipe: 0, rsq: 0 }).exposed, 0);
eq('deferred LN is never slower', Math.min(0, m.tokenPlan({ W: 4 }).overlapped - m.tokenPlan({ W: 4, deferLN: true }).overlapped), 0);
eq('router costs a full pass, 2 layers', m.tokenPlan({ moe: { E: 4, k: 1, h: 256 } }).arrayBeats - 1920, 32);
eq('E4 top-1 h256 storage', m.tokenPlan({ moe: { E: 4, k: 1, h: 256 } }).weights, 311808);
eq('E4 top-1 h64 array beats', m.tokenPlan({ moe: { E: 4, k: 1, h: 64 } }).arrayBeats, 1184);
eq('E4 top-1 h64 storage', m.tokenPlan({ moe: { E: 4, k: 1, h: 64 } }).weights, 115200);
eq('one KV head array beats', m.tokenPlan({ kvHeads: 1 }).arrayBeats, 1792);
eq('SwiGLU 176 FFN beats per layer', 2 * Math.ceil(176 / 16) * 16 + 4 * 176 / 4, 528);
console.log('\nImplied clock and balance, BF16 peak, 2 ops per MAC');
for (const t of m.tpu) {
  const d = m.derive(t);
  console.log(`${t.name.padEnd(9)} clock ${d.clockGHz.toFixed(3)} GHz | ridge ${d.ridgeBf16.toFixed(0)} F/B | batch@ridge ${d.batchForRidge.toFixed(0)} | VMEM MiB/TF ${d.vmemPerTf.toFixed(3)} | HBM GiB/TF ${d.hbmGiBPerTf.toFixed(3)} | ICI GB/s/TF ${d.iciPerTf.toFixed(2)} | per-chip TDP x${(t.relTDP / (t.pod / 256)).toFixed(2)}`);
}
const iw = m.tpu[4];
console.log('Ironwood if 8 x 256^2 BF16:', (iw.bf16 * 1e12 / (8 * 65536 * 2) / 1e9).toFixed(3), 'GHz; FP8 4 x 512^2:', (iw.fp8 * 1e12 / (4 * 262144 * 2) / 1e9).toFixed(3), 'GHz');
console.log('Carbon check, tonnes:', 3.14e23 / 1e18 * 265 / 1e6);
console.log('Pod FLOPs ratio:', iw.pod * iw.fp8 / (256 * 46), ' perf/W from columns:', 3600 / 123);
console.log('Pod HBM GiB:', 9216 * 192, 'PiB', 9216 * 192 / 1024 / 1024);
console.log('Bank throughput P=B=16:', m.bankThroughput(16, 16) / 16, ' 2x banks:', m.bankThroughput(16, 32) / 16);
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
