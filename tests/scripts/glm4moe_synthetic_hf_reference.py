"""HF transformers oracle for the synthetic glm4moe GGUF (issue #1599).

Usage:  python glm4moe_synthetic_hf_reference.py <synthetic.gguf> [ids.txt]

Loads the F32 tensors of the GGUF written by SyntheticGlm4MoeGguf into a Glm4MoeForCausalLM (F32, eager attention), runs the
12-token prompt in one forward pass and prints the sha256 of the GGUF plus the base64 of the [seq, vocab] F32 logits, ready to
paste into Glm4MoeForwardTests (FixtureSha256 / Golden). Needs torch and transformers >= 4.56 (Glm4Moe).

Use a STOCK torch wheel (pip install torch --index-url https://download.pytorch.org/whl/cpu). A locally built torch 2.14 gave
run-to-run different MoE logits with the default multi-threaded CPU kernels (deterministic only with torch.set_num_threads(1)).
"""
import base64
import hashlib
import sys

import numpy as np
import torch
from gguf import GGUFReader
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM


def main() -> None:
    path = sys.argv[1]
    ids_path = sys.argv[2] if len(sys.argv) > 2 else None
    ids = [3, 14, 8, 1, 5, 9, 0, 11, 2, 15, 7, 6]
    if ids_path:
        ids = [int(x) for x in open(ids_path).read().split()]

    rd = GGUFReader(path)
    kv = {}
    for k, f in rd.fields.items():
        if len(f.data) == 1 and f.types and f.types[0].name in ("UINT32", "INT32", "UINT64", "FLOAT32", "BOOL"):
            kv[k] = f.parts[f.data[0]][0].item()
    t = {x.name: torch.from_numpy(np.array(x.data, dtype=np.float32)) for x in rd.tensors}

    p = "glm4moe."
    trunk = kv[p + "block_count"] - kv.get(p + "nextn_predict_layers", 0)
    n_experts = kv[p + "expert_count"]
    cfg = Glm4MoeConfig(
        vocab_size=t["token_embd.weight"].shape[0],
        hidden_size=kv[p + "embedding_length"],
        intermediate_size=kv[p + "feed_forward_length"],
        moe_intermediate_size=kv[p + "expert_feed_forward_length"],
        num_hidden_layers=trunk,
        num_attention_heads=kv[p + "attention.head_count"],
        num_key_value_heads=kv[p + "attention.head_count_kv"],
        head_dim=kv[p + "attention.key_length"],
        partial_rotary_factor=kv[p + "rope.dimension_count"] / kv[p + "attention.key_length"],
        n_routed_experts=n_experts,
        n_shared_experts=kv[p + "expert_shared_count"],
        num_experts_per_tok=kv[p + "expert_used_count"],
        first_k_dense_replace=kv[p + "leading_dense_block_count"],
        n_group=1,
        topk_group=1,
        norm_topk_prob=bool(kv[p + "expert_weights_norm"]),
        routed_scaling_factor=kv[p + "expert_weights_scale"],
        rope_theta=kv[p + "rope.freq_base"],
        rms_norm_eps=kv[p + "attention.layer_norm_rms_epsilon"],
        attention_bias=True,
        use_qk_norm=False,
        max_position_embeddings=kv[p + "context_length"],
        tie_word_embeddings=False,
        num_nextn_predict_layers=0,
        attn_implementation="eager",
    )
    model = Glm4MoeForCausalLM(cfg).float().eval()
    sd = model.state_dict()
    new = {}

    def put(hf, g):
        assert tuple(sd[hf].shape) == tuple(g.shape), (hf, tuple(sd[hf].shape), tuple(g.shape))
        new[hf] = g.contiguous()

    put("model.embed_tokens.weight", t["token_embd.weight"].reshape(-1, cfg.hidden_size))
    put("model.norm.weight", t["output_norm.weight"])
    put("lm_head.weight", t["output.weight"].reshape(cfg.vocab_size, cfg.hidden_size))
    for l in range(trunk):
        b, h = f"blk.{l}.", f"model.layers.{l}."
        put(h + "input_layernorm.weight", t[b + "attn_norm.weight"])
        put(h + "post_attention_layernorm.weight", t[b + "post_attention_norm.weight"])
        for g, hf in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            w = t[b + f"attn_{g}.weight"]
            put(h + f"self_attn.{hf}.weight", w.reshape(-1, cfg.hidden_size))
            put(h + f"self_attn.{hf}.bias", t[b + f"attn_{g}.bias"])
        put(h + "self_attn.o_proj.weight", t[b + "attn_output.weight"].reshape(cfg.hidden_size, -1))
        if l < cfg.first_k_dense_replace:
            put(h + "mlp.gate_proj.weight", t[b + "ffn_gate.weight"].reshape(-1, cfg.hidden_size))
            put(h + "mlp.up_proj.weight", t[b + "ffn_up.weight"].reshape(-1, cfg.hidden_size))
            put(h + "mlp.down_proj.weight", t[b + "ffn_down.weight"].reshape(cfg.hidden_size, -1))
        else:
            mid = cfg.moe_intermediate_size
            put(h + "mlp.gate.weight", t[b + "ffn_gate_inp.weight"].reshape(n_experts, cfg.hidden_size))
            put(h + "mlp.gate.e_score_correction_bias", t[b + "exp_probs_b.bias"])
            ge = t[b + "ffn_gate_exps.weight"].reshape(n_experts, mid, cfg.hidden_size)
            ue = t[b + "ffn_up_exps.weight"].reshape(n_experts, mid, cfg.hidden_size)
            de = t[b + "ffn_down_exps.weight"].reshape(n_experts, cfg.hidden_size, mid)
            if (h + "mlp.experts.gate_up_proj") in sd:        # transformers >= 5: fused expert parameters
                put(h + "mlp.experts.gate_up_proj", torch.cat([ge, ue], dim=1))
                put(h + "mlp.experts.down_proj", de)
            else:
                for e in range(n_experts):
                    put(h + f"mlp.experts.{e}.gate_proj.weight", ge[e])
                    put(h + f"mlp.experts.{e}.up_proj.weight", ue[e])
                    put(h + f"mlp.experts.{e}.down_proj.weight", de[e])
            put(h + "mlp.shared_experts.gate_proj.weight", t[b + "ffn_gate_shexp.weight"].reshape(-1, cfg.hidden_size))
            put(h + "mlp.shared_experts.up_proj.weight", t[b + "ffn_up_shexp.weight"].reshape(-1, cfg.hidden_size))
            put(h + "mlp.shared_experts.down_proj.weight", t[b + "ffn_down_shexp.weight"].reshape(cfg.hidden_size, -1))
    missing = [k for k in sd if k not in new]
    assert not missing, f"unmapped HF params: {missing}"
    model.load_state_dict(new, strict=True)

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float().numpy()
    print("sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
    print("shape", logits.shape, "absmax", float(np.abs(logits).max()), "std", float(logits.std()))
    print("b64", base64.b64encode(logits.astype("<f4").tobytes()).decode())


if __name__ == "__main__":
    main()
