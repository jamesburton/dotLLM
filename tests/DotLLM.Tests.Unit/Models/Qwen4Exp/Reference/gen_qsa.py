"""HF reference for the Qwen4-Exp QSA attention layer (indexer + sparse attention), issue #816 sub-PR 3.

Runs the real `Qwen4ExpTextAttention` (layer 3 of the tiny model) in eager mode on T=40 tokens (10 blocks, budget 4 blocks:
sparse engaged for later queries) and T=14 (3 blocks <= budget: exactly dense). Dumps weights, the block input, the pooled
indexer keys, the block scores, the selection mask and the layer output, plus a dense-attention (budget huge) output on the
T=40 input so a test can prove sparse != dense.
"""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from qwen4exp_ref_common import FixtureWriter  # noqa: E402
from qwen4exp_tiny_model import build_model  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb  # noqa: E402


def causal_mask(T, dtype=torch.float32):
    m = torch.zeros(1, 1, T, T, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), torch.finfo(dtype).min)
    return m


def run(model, attn, x, T):
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    pe = model.rotary_emb(x, pos)
    mask = causal_mask(T)
    with torch.no_grad():
        sel = attn.indexer(x, pe, mask, None)                 # [1,1,T,T] additive float mask (0 selected / min dropped)
        out, _ = attn(x, pe, mask, past_key_values=None)
    return pe, sel, out


def pooled_and_scores(attn, x, pe, T):
    """Replicates the indexer pooled keys / block scores (for fixture dumps; the mask is asserted equal to HF's)."""
    ix = attn.indexer
    D, R = ix.index_head_dim, ix.compress_ratio
    cos, sin = pe
    with torch.no_grad():
        qk = ix.index_qk_proj(x)
        q, k = torch.split(qk, [ix.index_n_heads * D, D], dim=-1)
        q = q.reshape(1, T, ix.index_n_heads, D)
        raw = k.reshape(1, T, D)
        qn = ix.q_layernorm(q)
        qn = apply_rotary_pos_emb(qn, cos=cos, sin=sin, unsqueeze_dim=2)
        nb = T // R
        pooled = raw[0, : nb * R].view(nb, R, D).float().mean(1)
        pooled = ix.k_layernorm(pooled)
        starts = [b * R for b in range(nb)]
        pooled = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0, starts], sin=sin[0, starts]).squeeze(1)
        scores = torch.full((T, nb), -1.0)
        for t in range(T):
            vis = (t + 1) // R
            if vis:
                s = torch.matmul(qn[0, t].float(), pooled[:vis].float().T)
                scores[t, :vis] = torch.relu(s).sum(0) / math.sqrt(D)
    return raw[0], qn[0], pooled, scores


def main(out):
    model = build_model(seed=21)
    attn = model.layers[3].self_attn
    w = FixtureWriter(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=32, rope_dim=8, idx_heads=3, idx_dim=16,
                      block=4, budget_tokens=16, eps=1e-6, rope_theta=1.0e7)
    for name, p in (("q_proj", attn.q_proj.weight), ("k_proj", attn.k_proj.weight), ("v_proj", attn.v_proj.weight),
                    ("o_proj", attn.o_proj.weight), ("index_qk_proj", attn.indexer.index_qk_proj.weight)):
        w.add(name, p)
    w.add("q_norm", 1.0 + attn.q_norm.weight)
    w.add("k_norm", 1.0 + attn.k_norm.weight)
    w.add("idx_q_norm", 1.0 + attn.indexer.q_layernorm.weight)
    w.add("idx_k_norm", 1.0 + attn.indexer.k_layernorm.weight)

    for T in (40, 14):
        torch.manual_seed(100 + T)
        x = torch.randn(1, T, 64)
        pe, sel, out_t = run(model, attn, x, T)
        raw, qn, pooled, scores = pooled_and_scores(attn, x, pe, T)
        mask = (sel[0, 0] == 0).float()                        # 1 where the indexer allows the key
        causal = torch.tril(torch.ones(T, T))
        w.add(f"t{T}.x", x[0])
        w.add(f"t{T}.out", out_t[0])
        w.add(f"t{T}.sel_mask", mask * causal)
        w.add(f"t{T}.raw_keys", raw)
        w.add(f"t{T}.pooled", pooled)
        w.add(f"t{T}.idx_q", qn)                                 # normed + roped indexer queries [T, heads, D]
        w.add(f"t{T}.scores", scores)
        # consistency: the selection derived from our replicated scores must equal HF's mask
        R, topk = 4, 4
        for t in range(T):
            vis = (t + 1) // R
            chosen = set(torch.topk(scores[t, :vis], min(topk, vis)).indices.tolist()) if vis else set()
            exp = torch.zeros(T)
            for b in chosen:
                exp[b * R:(b + 1) * R] = 1
            exp[vis * R: t + 1] = 1
            assert torch.equal(exp, (mask * causal)[t]), f"replicated selection != HF mask at t={t}"
        if T == 40:
            # same weights + input, indexer budget huge -> every block selected == dense causal attention
            dense_model = build_model(seed=21, indexer_budget=2000)
            _, _, dense_out = run(dense_model, dense_model.layers[3].self_attn, x, T)
            w.add("t40.dense_out", dense_out[0])
            assert not torch.allclose(dense_out, out_t, atol=1e-4), "sparse and dense coincide: fixture not discriminating"
    w.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
