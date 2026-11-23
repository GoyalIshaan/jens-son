"""Builds model_map/index.html from template.html plus per-op data.

All shapes come from design/calc/qwen35_budget.py, so the page and the design
notes can't disagree. Run:  python3 model_map/build.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "design", "calc"))
import qwen35_budget as B  # noqa: E402

H, F, V = B.HIDDEN, B.FFN, B.VOCAB
QH, KVH, HD = B.Q_HEADS, B.KV_HEADS, B.HEAD_DIM
LQK, LV, LD, CK = B.LIN_QK_HEADS, B.LIN_V_HEADS, B.LIN_DIM, B.CONV_K
CONV_CH = (2 * LQK + LV) * LD                      # q, k, v channels = 8192


def mm(id_, name, k, n, what, fused=None):
    return dict(id=id_, name=name, kind="matmul", what=what,
                shape=f"[1×{k}] · [{k}×{n}] → [1×{n}]",
                weights=k * n, macs=k * n, act_in=k, act_out=n, fused=fused)


def ew(id_, name, n, flops_per, what, kind="vector", **extra):
    d = dict(id=id_, name=name, kind=kind, what=what, shape=f"[1×{n}]",
             weights=0, macs=0, ew_flops=n * flops_per, act_in=n, act_out=n)
    d.update(extra)
    return d


norm = lambda id_, what="Scale the 4,096 values so their RMS is 1, then multiply by a learned gain.": \
    ew(id_, "RMSNorm", H, 4, what)

ffn_ops = [
    norm("ffn_norm"),
    mm("ffn_gate", "FFN gate", H, F, "Project up to 12,288. This branch decides how much of each feature passes.", fused="gate+up"),
    mm("ffn_up", "FFN up", H, F, "Second projection to 12,288 from the same input. Fused with gate in hardware.", fused="gate+up"),
    ew("ffn_act", "SiLU(gate) × up", F, 5, "SwiGLU: smooth gate times the up branch, element by element."),
    mm("ffn_down", "FFN down", F, H, "Project back down to 4,096. The only matrix whose dot products are 12,288 long."),
    ew("ffn_res", "Residual add", H, 1, "Add the FFN output to the running hidden state."),
]

delta_ops = [
    norm("dn_norm"),
    mm("dn_qkvz", "in_proj q,k,v,z", H, 2 * LQK * LD + 2 * LV * LD,
       "One fused projection: q (16×128), k (16×128), v (32×128) and output gate z (32×128)."),
    mm("dn_ba", "in_proj β, a", H, 2 * LV,
       "Two numbers per value head: β (write strength) and a (forget rate). Tiny and awkward: 64 outputs."),
    dict(id="dn_conv", name="Causal conv1d (k=4) + SiLU", kind="conv",
         what="Each of the 8,192 q/k/v channels mixes its last 4 values. Needs the previous 3 inputs (conv state).",
         shape=f"[4×{CONV_CH}] depthwise → [1×{CONV_CH}]", weights=CONV_CH * CK,
         macs=CONV_CH * CK, act_in=CONV_CH, act_out=CONV_CH,
         state_values=CONV_CH * (CK - 1)),
    ew("dn_l2", "L2-normalise q, k", 2 * LQK * LD, 3, "Each 128-wide q and k head vector is scaled to length 1."),
    ew("dn_gates", "β = σ(b), decay = exp(−e^A · softplus(a+δ))", 2 * LV, 8,
       "Turn the 64 raw numbers into a write strength and a decay per head."),
    dict(id="dn_rule", name="Gated delta rule (32 heads)", kind="recurrence",
         what="Per head, a 128×128 memory S: decay it, read S·k, write β·(v − S·k)·kᵀ, read out S·q. "
              "Fixed cost whatever the context length.",
         shape=f"{LV} × S[{LD}×{LD}]", weights=0,
         macs=B.STATE_VALUES // B.N_LIN * 4, act_in=2 * LQK * LD + LV * LD, act_out=LV * LD,
         state_values=B.STATE_VALUES // B.N_LIN),
    ew("dn_gnorm", "Gated RMSNorm: norm(o) × SiLU(z)", LV * LD, 7, "Normalise each head's output, then gate it with z."),
    mm("dn_out", "out_proj", LV * LD, H, "Mix the 32 heads back into the 4,096-wide hidden state."),
    ew("dn_res", "Residual add", H, 1, "Add the mixer output to the running hidden state."),
]

attn_ops = [
    norm("at_norm"),
    mm("at_q", "q_proj (+ output gate)", H, 2 * QH * HD,
       "16 query heads of 256, plus a 4,096-wide gate (attn_output_gate). Hence 8,192 outputs.", fused="q,k,v"),
    mm("at_k", "k_proj", H, KVH * HD, "4 key heads of 256. Each is shared by 4 query heads (GQA).", fused="q,k,v"),
    mm("at_v", "v_proj", H, KVH * HD, "4 value heads of 256.", fused="q,k,v"),
    ew("at_qknorm", "q_norm, k_norm", (QH + KVH) * HD, 4, "RMSNorm on each 256-wide head."),
    ew("at_rope", "RoPE on 64 of 256 dims", (QH + KVH) * HD // 4, 6,
       "Rotate a quarter of each head (partial_rotary_factor 0.25) by an angle set by position."),
    dict(id="at_kvw", name="Append to KV cache", kind="kv", what="Write this token's 4 keys and 4 values.",
         shape=f"[2×{KVH}×{HD}]", weights=0, macs=0, act_in=2 * KVH * HD, act_out=0,
         kv_write=2 * KVH * HD),
    dict(id="at_score", name="Scores q·Kᵀ + softmax", kind="attention",
         what="Each query head dots with every past key of its group, then softmax. Grows with context L.",
         shape="per KV head: [4×256] · [256×L] → [4×L]", weights=0, macs=0,
         macs_per_ctx=QH * HD, ew_per_ctx=QH * 5, act_in=QH * HD, act_out_per_ctx=QH,
         kv_read_per_ctx=KVH * HD),
    dict(id="at_pv", name="Weighted sum of V", kind="attention",
         what="Mix all past values by the softmax weights. Grows with context L.",
         shape="per KV head: [4×L] · [L×256] → [4×256]", weights=0, macs=0,
         macs_per_ctx=QH * HD, act_out=QH * HD, kv_read_per_ctx=KVH * HD),
    ew("at_gate", "× σ(gate)", QH * HD, 5, "Output gate: scale each output by a sigmoid of the gate from q_proj."),
    mm("at_o", "o_proj", QH * HD, H, "Mix the 16 heads back into 4,096."),
    ew("at_res", "Residual add", H, 1, "Add the mixer output to the running hidden state."),
]

edge_ops = {
    "embed": [dict(id="emb", name="Embedding lookup", kind="lookup",
                   what="Fetch one 4,096-wide row of the 248,320-row table. One row, not a matmul.",
                   shape=f"table [{V}×{H}] → row [1×{H}]", weights=0, stored=V * H,
                   row_read=H, macs=0, act_in=1, act_out=H)],
    "head": [norm("final_norm"),
             mm("lm_head", "LM head", H, V, "Score every vocabulary entry. The largest single matrix (12.8% of bytes)."),
             ew("sample", "Top-k / argmax over 248,320", V, 2, "Pick the next token. Stream the scores, keep the best few.")],
}

# --- consistency with the budget script
dn_w = sum(o["weights"] for o in delta_ops)
at_w = sum(o["weights"] for o in attn_ops)
ff_w = sum(o["weights"] for o in ffn_ops)
assert dn_w == B.P["lin"], (dn_w, B.P["lin"])
assert at_w == B.P["full"], (at_w, B.P["full"])
assert ff_w == B.P["ffn"]
total = B.N_LIN * (dn_w + ff_w) + B.N_FULL * (at_w + ff_w) + H * V
assert total == B.P["read_per_token"]
for c in (4096, 32768):
    mac = sum(o.get("macs", 0) for o in delta_ops + ffn_ops) * B.N_LIN \
        + sum(o.get("macs", 0) + o.get("macs_per_ctx", 0) * c for o in attn_ops + ffn_ops) * B.N_FULL \
        + H * V
    assert abs(2 * mac - B.flops_per_token(c)) / B.flops_per_token(c) < 0.001, (2 * mac, B.flops_per_token(c))

layers = [("attention" if (i + 1) % B.FULL_EVERY == 0 else "deltanet") for i in range(B.LAYERS)]

data = dict(
    model=dict(hidden=H, ffn=F, vocab=V, layers=B.LAYERS, n_full=B.N_FULL, n_lin=B.N_LIN,
               q_heads=QH, kv_heads=KVH, head_dim=HD, lin_qk_heads=LQK, lin_v_heads=LV,
               lin_dim=LD, conv_k=CK, max_ctx=B.MAX_CTX, native_ctx=262144,
               read_per_token=B.P["read_per_token"], total=B.P["total"],
               kv_per_token=B.KV_VALUES_PER_TOKEN, state_values=B.STATE_VALUES,
               conv_state_values=B.CONV_STATE_VALUES),
    layers=layers,
    ops=dict(deltanet=delta_ops, attention=attn_ops, ffn=ffn_ops, **edge_ops),
    card=dict(bw=B.CARD_BW, util=B.CARD_UTIL),
    formats=[dict(name=n, bits=(b if b is not None else None)) for n, b in B.FORMATS],
    kv_formats=[dict(name=n, bytes=b) for n, b in B.KV_BYTES.items()],
)
# fill the mixed-precision row the same way the budget script does
ffn_all = B.ffn_weights()
mixed = (ffn_all * (3 + 16 / 32) + (B.P["read_per_token"] - ffn_all) * 4.5) / B.P["read_per_token"]
for f in data["formats"]:
    if f["bits"] is None:
        f["bits"] = mixed

tpl = open(os.path.join(HERE, "template.html")).read()
out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
chip = open(os.path.join(HERE, "chip_content.js")).read()
assert "</script" not in chip
out = out.replace("/*__CHIP__*/", chip)
open(os.path.join(HERE, "index.html"), "w").write(out)
print("wrote model_map/index.html", len(out), "bytes; checks passed")
