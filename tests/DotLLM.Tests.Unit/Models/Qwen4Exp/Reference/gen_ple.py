"""HF reference for the Qwen4-Exp PLE (n-gram hash embedding) branch, issue #816 sub-PR 2.

Two fixtures:
  ple_branch.json : a tiny `Qwen4ExpTextPLELayer` (random weights, T=24 tokens with EOS resets at several offsets,
                    dilated conv with 9-row history) with every intermediate dumped.
  ple_hash_real.json : the hash index builder with the REAL model constants (45-bit multipliers, 16 head ranges of ~20M
                    rows, vocab 248320, eos 248044) computed by HF's own int64 ops, to prove exact-int64 arithmetic.
Gammas are exported GGUF-convention (1 + weight), conv weight as the checkpoint's [C, 1, K] flattened to [C, K].
"""
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from qwen4exp_ref_common import FixtureWriter  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
    _build_layer_multipliers,
)

S, H, T = 4, 48, 24
EOS = 5
EPS = 1e-6


def branch(out):
    torch.manual_seed(7)
    cfg = SimpleNamespace(
        hidden_size=H, hc_count=S, ple_embed_dim=128, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        rms_norm_eps=EPS, vocab_size=997, ngram_vocab_size_base=40, seed=1234, eos_token_id=EOS,
        make_ngram_vocab_size_divisible_by=16,
    )
    m = Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0).eval()
    with torch.no_grad():
        m.ple_embedding.ngram_embedding.weight.normal_()
        for n in (m.norm_key, m.norm_query, m.norm_conv):
            n.weight.copy_(torch.randn_like(n.weight) * 0.3)
        m.key_proj.weight.copy_(torch.randn_like(m.key_proj.weight) * 0.15)
        m.value_proj.weight.copy_(torch.randn_like(m.value_proj.weight) * 0.15)
        m.conv1d.weight.copy_(torch.randn_like(m.conv1d.weight) * 0.7)

    ids = torch.randint(6, 997, (1, T))
    for p in (0, 7, 8, 13, 21, 22):  # EOS at the very start, adjacent EOS pair, and 1/2 back from later tokens
        ids[0, p] = EOS
    scales = torch.tensor([0.3, 1.0, 3.0, 9.0]).view(1, 1, S, 1)
    r = (torch.randn(1, T, S, H) * scales).flatten(-2)

    captured = {}
    m.ple_embedding.ngram_embedding.register_forward_pre_hook(lambda mod, args: captured.update(ids=args[0].clone()))
    with torch.no_grad():
        ref_out = m(r, ids, None)
        emb = m.ple_embedding(ids, None)
        key_normed = m.norm_key(m.key_proj(emb)).unflatten(-1, (S, H))
        value = m.value_proj(emb)
        query_normed = m.norm_query(r).unflatten(-1, (S, H))
        g = (key_normed * query_normed).sum(dim=-1, keepdim=True) / (H ** 0.5)
        gate = torch.sigmoid(g.abs().clamp_min(1e-6).sqrt() * g.sign())
        gated = gate * value.unsqueeze(-2)
        normed = m.norm_conv(gated.flatten(-2))
        conv = m._short_conv(normed, None)
        mine = gated.flatten(-2) + conv
    assert torch.allclose(mine, ref_out, atol=1e-5), "generator replication diverged from the module"

    w = FixtureWriter(hidden_size=H, hc_count=S, seq_len=T, eos=EOS, eps=EPS, ngram=3, heads_per_ngram=2,
                      row_dim=32, conv_kernel=4, ple_embed_dim=128, vocab=997)
    w.add("ids", ids[0].to(torch.int64))
    w.add("multipliers", m.ple_embedding.layer_multipliers)
    w.add("offsets", m.ple_embedding.ngram_heads_offsets)
    w.add("vocab_sizes", m.ple_embedding.ngram_heads_vocab_sizes)
    w.add("rows", captured["ids"][0])                         # [T, numHeads] int64 table rows
    w.add("table", m.ple_embedding.ngram_embedding.weight)    # [rows, 32]
    w.add("emb", emb[0])
    w.add("key_proj", m.key_proj.weight)
    w.add("value_proj", m.value_proj.weight)
    w.add("norm_key", 1.0 + m.norm_key.weight)
    w.add("norm_query", 1.0 + m.norm_query.weight)
    w.add("norm_conv", 1.0 + m.norm_conv.weight)
    w.add("conv1d", m.conv1d.weight[:, 0, :])                 # [C, K]
    w.add("residual", r[0])
    w.add("gate", gate[0, :, :, 0])
    w.add("gated", gated.flatten(-2)[0])
    w.add("normed", normed[0])
    w.add("conv_out", conv[0])
    w.add("output", ref_out[0])                               # gated + silu(conv(...))
    w.save(out)


def hash_real(out):
    mult = [23703573157769, 20109073645365, 8052911324071]
    offsets = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455,
               200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
    sizes = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093,
             20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
    # sanity: our constants equal HF's own derivation for ple layer 0 (seed 1234)
    assert _build_layer_multipliers(248320, 3, 0, 1234).tolist() == mult
    eos, vocab, n = 248044, 248320, 512
    torch.manual_seed(3)
    ids = torch.randint(0, vocab, (1, n))
    ids[0, torch.randint(0, n, (40,))] = eos
    ids[0, 100:103] = vocab - 1                 # max-magnitude products
    ids[0, 200] = 0
    stub = SimpleNamespace(eos_token_id=eos)
    shifted = [Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos(stub, ids, s) for s in range(3)]
    mult_t = torch.tensor(mult, dtype=torch.long)
    blocks = []
    for ngram in (2, 3):
        a = (ngram - 2) * 8
        mixed = shifted[0] * mult_t[0]
        for p in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[p] * mult_t[p])
        vs = torch.tensor(sizes[a:a + 8], dtype=torch.long)
        os_ = torch.tensor(offsets[a:a + 8], dtype=torch.long)
        blocks.append(torch.remainder(mixed.unsqueeze(-1), vs.view(1, 1, -1)) + os_.view(1, 1, -1))
    rows = torch.cat(blocks, dim=-1)
    w = FixtureWriter(eos=eos, vocab=vocab, n=n, ngram=3, heads_per_ngram=8)
    w.add("ids", ids[0].to(torch.int64))
    w.add("multipliers", torch.tensor(mult, dtype=torch.long))
    w.add("offsets", torch.tensor(offsets, dtype=torch.long))
    w.add("vocab_sizes", torch.tensor(sizes, dtype=torch.long))
    w.add("rows", rows[0])
    w.save(out)


if __name__ == "__main__":
    branch(sys.argv[1])
    hash_real(sys.argv[2])
