"""HF reference for the whole Qwen4-Exp text model (tiny random weights), issue #816 sub-PR 4.

Runs the real `Qwen4ExpTextModel` (4 layers: GDN, GDN+PLE, GDN, QSA; NK=2 != NV=4 GDN heads; GQA 4:2; 3 indexer heads; 8 experts
top-3 + shared expert; budget 16 tokens so the indexer prunes beyond ~19 tokens) on 48 tokens containing EOS resets, and dumps:
  * every weight in the GGUF convention llama.cpp's converter produces (folded 1+w gammas, V heads reordered grouped->tiled,
    A_log -> -exp(A_log), indexer qk split, PLE table as one tensor), under the GGUF tensor names;
  * the per-layer residual stream after every decoder layer, the final head-mixer output and logits (lm_head is a random matrix).
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from qwen4exp_ref_common import FixtureWriter  # noqa: E402
from qwen4exp_tiny_model import build_model  # noqa: E402


def reorder_v(t, dim, nk, r, hd):
    """grouped (by K head) -> tiled V head order along `dim` (llama.cpp _reorder_v_heads)."""
    shape = list(t.shape)
    new_shape = shape[:dim] + [nk, r, hd] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def main(out):
    seed = 33
    # moe/shared intermediate 32 (not the default 24) so the expert banks can be block-quantised (K multiple of 32) in the quantised-checkpoint test
    model = build_model(seed=seed, moe_intermediate_size=32, shared_expert_intermediate_size=32)
    cfg = model.config
    H, S = cfg.hidden_size, cfg.hc_count
    nk, nv, dk, dv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    r = nv // nk
    T = 48

    w = FixtureWriter(hidden_size=H, hc_count=S, num_layers=cfg.num_hidden_layers, vocab=cfg.vocab_size,
                      seq_len=T, eos=5, eps=float(cfg.rms_norm_eps), nk=nk, nv=nv, dk=dk, dv=dv,
                      heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                      rope_dim=int(cfg.head_dim * 0.25), rope_theta=1.0e7, idx_heads=cfg.indexer_n_heads,
                      idx_dim=cfg.indexer_head_dim, block=cfg.indexer_compress_ratio, budget=cfg.indexer_budget,
                      experts=cfg.num_experts, top_k=cfg.num_experts_per_tok, moe_inter=cfg.moe_intermediate_size,
                      shared_inter=cfg.shared_expert_intermediate_size, hc_lowrank=cfg.hc_lowrank,
                      ngram=cfg.ngram_size, heads_per_ngram=cfg.heads_per_ngram, ple_layer=0, conv_k=cfg.linear_conv_kernel_dim,
                      ple_conv_k=cfg.ple_conv_kernel_size, seed=seed)

    def gr(prefix, mod, inject=True):
        w.add(f"{prefix}norm.weight", 1.0 + mod.hc_norm.weight)
        w.add(f"{prefix}down.weight", mod.input_mix_weight_down.weight)
        w.add(f"{prefix}up.weight", mod.input_mix_weight_up.weight)
        if inject:
            w.add(f"{prefix}inject.weight", mod.block_inject_weight.weight)

    w.add("token_embd.weight", model.embed_tokens.weight)
    torch.manual_seed(seed + 1)
    lm_head = torch.randn(cfg.vocab_size, H) * 0.3
    w.add("output.weight", lm_head)
    gr("output_hc_", model.hyper_connection_mixer, inject=False)

    for il, layer in enumerate(model.layers):
        b = f"blk.{il}."
        gr(b + "hc_attn_", layer.attn_hyper_connection)
        gr(b + "hc_ffn_", layer.mlp_hyper_connection)
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            qkv = la.in_proj_qkv.weight
            kd = nk * dk
            q, k, v = qkv[:kd], qkv[kd:2 * kd], qkv[2 * kd:]
            w.add(b + "attn_qkv.weight", torch.cat([q, k, reorder_v(v, 0, nk, r, dv)], 0))
            w.add(b + "attn_gate.weight", reorder_v(la.in_proj_z.weight, 0, nk, r, dv))
            w.add(b + "ssm_beta.weight", reorder_v(la.in_proj_b.weight, 0, nk, r, 1))
            w.add(b + "ssm_alpha.weight", reorder_v(la.in_proj_a.weight, 0, nk, r, 1))
            w.add(b + "ssm_a", reorder_v((-la.A_log.float().exp()).unsqueeze(-1), 0, nk, r, 1)[:, 0])
            w.add(b + "ssm_dt.bias", reorder_v(la.dt_bias.unsqueeze(-1), 0, nk, r, 1)[:, 0])
            cw = la.conv1d.weight[:, 0, :]
            w.add(b + "ssm_conv1d.weight", torch.cat([cw[:2 * kd], reorder_v(cw[2 * kd:], 0, nk, r, dv)], 0))
            w.add(b + "ssm_norm.weight", la.norm.weight)           # plain gain, not folded
            w.add(b + "ssm_out.weight", reorder_v(la.out_proj.weight, 1, nk, r, dv))
        else:
            sa = layer.self_attn
            w.add(b + "attn_q.weight", sa.q_proj.weight)
            w.add(b + "attn_k.weight", sa.k_proj.weight)
            w.add(b + "attn_v.weight", sa.v_proj.weight)
            w.add(b + "attn_output.weight", sa.o_proj.weight)
            w.add(b + "attn_q_norm.weight", 1.0 + sa.q_norm.weight)
            w.add(b + "attn_k_norm.weight", 1.0 + sa.k_norm.weight)
            nq = cfg.indexer_n_heads * cfg.indexer_head_dim
            qk = sa.indexer.index_qk_proj.weight
            w.add(b + "indexer.q_proj.weight", qk[:nq])
            w.add(b + "indexer.k_proj.weight", qk[nq:])
            w.add(b + "indexer.q_norm.weight", 1.0 + sa.indexer.q_layernorm.weight)
            w.add(b + "indexer.k_norm.weight", 1.0 + sa.indexer.k_layernorm.weight)
        if layer.ple is not None:
            p = layer.ple
            w.add(b + "ple_key.weight", p.key_proj.weight)
            w.add(b + "ple_value.weight", p.value_proj.weight)
            w.add(b + "ple_norm_key.weight", 1.0 + p.norm_key.weight)
            w.add(b + "ple_norm_query.weight", 1.0 + p.norm_query.weight)
            w.add(b + "ple_norm_conv.weight", 1.0 + p.norm_conv.weight)
            w.add(b + "ple_conv1d.weight", p.conv1d.weight[:, 0, :])
            w.add("per_layer_token_embd.weight", p.ple_embedding.ngram_embedding.weight)
            w.add("ple.layer_multipliers", p.ple_embedding.layer_multipliers)
            w.add("ple.head_offsets", p.ple_embedding.ngram_heads_offsets)
            w.add("ple.head_vocab_sizes", p.ple_embedding.ngram_heads_vocab_sizes)
        mlp = layer.mlp
        inter = cfg.moe_intermediate_size
        gate_up = mlp.experts.gate_up_proj
        w.add(b + "ffn_gate_inp.weight", mlp.gate.weight)
        w.add(b + "ffn_gate_exps.weight", gate_up[:, :inter, :])
        w.add(b + "ffn_up_exps.weight", gate_up[:, inter:, :])
        w.add(b + "ffn_down_exps.weight", mlp.experts.down_proj)
        w.add(b + "ffn_gate_shexp.weight", mlp.shared_expert.gate_proj.weight)
        w.add(b + "ffn_up_shexp.weight", mlp.shared_expert.up_proj.weight)
        w.add(b + "ffn_down_shexp.weight", mlp.shared_expert.down_proj.weight)
        w.add(b + "ffn_gate_inp_shexp.weight", mlp.shared_expert_gate.weight[0])

    torch.manual_seed(seed + 2)
    ids = torch.randint(6, cfg.vocab_size, (1, T))
    for p in (0, 9, 10, 11, 25, 30, 47):
        ids[0, p] = 5
    captured = []
    hooks = [layer.register_forward_hook(lambda m, a, o: captured.append(o.detach().clone())) for layer in model.layers]
    with torch.no_grad():
        res = model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    w.add("ids", ids[0].to(torch.int64))
    for il, o in enumerate(captured):
        w.add(f"l_out.{il}", o[0])
    final = res.last_hidden_state[0]
    w.add("hidden_final", final)
    w.add("logits", final @ lm_head.T)
    w.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
