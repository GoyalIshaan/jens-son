"""GPTQ: rounding that uses real activations so errors cancel at each layer's output.

Plain rounding makes each weight as close as possible to its original. GPTQ
instead keeps each matrix's *output* close: after rounding one column of
weights, it nudges the not-yet-rounded columns to make up for the error,
using H = sum of x xᵀ over example inputs x (which inputs move together).

The output is the same storage format as plain rounding (codes + group scales
+ table), so it runs on the same hardware.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", 2022. This is the standard algorithm with
lazy block updates, per-group scales set when the group's first column is
reached, and 1% damping.
"""
import torch

from formats import QTensor, Spec, quantize_weight


@torch.no_grad()
def gptq_matrix(W, H, fmt, block=128, damp=0.01):
    """W [N, K] (any float dtype, on GPU), H [K, K] fp32. Returns a QTensor."""
    W = W.float() if W.dtype != torch.float32 else W.clone()   # never modify the caller's tensor
    N, K = W.shape
    spec = Spec(W, fmt)
    g = spec.g
    assert K % g == 0 and block % g == 0
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=H.device)
    L = torch.linalg.cholesky(H)
    U = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    del L, H

    codes = torch.empty(N, K, dtype=torch.int8 if len(spec.table) <= 128 else torch.int16, device=W.device)
    scales = torch.empty(N, K // g, dtype=torch.float32, device=W.device)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        s = None
        for j in range(i2 - i1):
            col = i1 + j
            if col % g == 0:                                   # new group: set its scale now
                s = spec.scale(W1[:, j:j + g])
                scales[:, col // g] = s
            w = W1[:, j]
            idx, qv = spec.round(w / s)
            codes[:, col] = idx
            err = (w - qv * s) / U1[j, j]
            W1[:, j:] -= err[:, None] * U1[j, j:][None, :]
            E1[:, j] = err
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    qt = QTensor(codes.view(N, K // g, g), scales, spec.table, g, (N, K))
    qt.gmax = spec.gmax
    return qt


@torch.no_grad()
def quantize_model_gptq(gold, numerics, calib, log=print):
    """Sequential GPTQ over the golden model.

    calib: list of 1-D token tensors (each one calibration sequence).
    For each layer: run the calibration set through it with its original weights,
    collecting H for every matrix; GPTQ every matrix; then rerun the layer with
    the quantized weights to produce the next layer's inputs.
    The embedding (a lookup, no inputs) uses plain rounding; the LM head uses GPTQ.
    Leaves gold.W holding QTensors (and bf16 for anything numerics keeps in bf16).
    """
    import time
    cfg = gold.cfg
    gold.num = numerics
    gold.weight_error = {}
    t_all = time.time()

    def record_error(name, q):
        from qwen35_golden import weight_kind
        ref = gold.raw[name].to(q.codes.device)
        e = gold.weight_error.setdefault(weight_kind(name), [0.0, 0.0, 0, numerics.weight_format(name)])
        for r in range(0, ref.shape[0], 32768):
            rr = ref[r:r + 32768].float()
            e[0] += (q.dequant_slice(r, r + 32768) - rr).pow(2).sum().item()
            e[1] += rr.pow(2).sum().item()
        e[2] += ref.numel()

    # free the GPU copies of everything that will be quantized; each layer's bf16 weights are
    # brought back to the GPU only while that layer is processed (fits 24 GB GPUs)
    for n in list(gold.W):
        if numerics.weight_format(n):
            gold.W[n] = None
    torch.cuda.empty_cache()

    # embedding: plain rounding (it is a lookup; GPTQ needs inputs)
    name = "embed_tokens.weight"
    fmt = numerics.weight_format(name)
    if fmt:
        q = quantize_weight(gold.raw[name].to(gold.dev), fmt)
        gold.W[name] = q
        record_error(name, q)
    hs = [gold.embed(ids).cpu() for ids in calib]              # per-sequence hidden states (host RAM)

    for i, t in enumerate(cfg.layer_types):
        t0 = time.time()
        names = [n for n in gold.raw if n.startswith(f"layers.{i}.") and numerics.weight_format(n)]
        for n in names:
            gold.W[n] = gold.raw[n].to(gold.dev)
        gold.collect = {n: (torch.zeros(gold.raw[n].shape[1], gold.raw[n].shape[1], device=gold.dev), 0)
                        for n in names}
        for h in hs:
            _layer(gold, i, t, h.to(gold.dev))                 # collect H with original weights
        coll, gold.collect = gold.collect, None
        for n in names:
            H, cnt = coll[n]
            q = gptq_matrix(gold.W[n], H * (2.0 / max(cnt, 1)), numerics.weight_format(n))
            gold.W[n] = q
            record_error(n, q)
        del coll
        hs = [_layer(gold, i, t, h.to(gold.dev)).cpu() for h in hs]   # next layer's inputs, quantized
        torch.cuda.empty_cache()
        log(f"  layer {i:2d} ({t}) gptq done in {time.time()-t0:.1f}s")

    name = "lm_head.weight"
    fmt = numerics.weight_format(name)
    if fmt:
        gold.W[name] = gold.raw[name].to(gold.dev)
        H = None
        cnt = 0
        for h in hs:
            x = gold.rms(h.to(gold.dev), "norm.weight")
            H = x.t() @ x if H is None else H + x.t() @ x
            cnt += x.shape[0]
        q = gptq_matrix(gold.W[name], H * (2.0 / cnt), fmt)
        gold.W[name] = q
        record_error(name, q)
    log(f"  gptq total {time.time()-t_all:.0f}s")


def _layer(gold, i, t, h):
    """Run one decoder layer on one calibration sequence (fresh cache)."""
    cache = gold.new_cache()
    positions = torch.arange(h.shape[0], device=h.device)
    x = gold.rms(h, f"layers.{i}.input_layernorm.weight")
    if t == "full_attention":
        h = h + gold.full_attention(i, x, cache, positions)
    else:
        h = h + gold.deltanet(i, x, cache)
    return h + gold.ffn(i, gold.rms(h, f"layers.{i}.post_attention_layernorm.weight"))
