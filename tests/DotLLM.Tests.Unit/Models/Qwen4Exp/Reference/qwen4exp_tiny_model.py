"""Tiny random-weight Qwen4-Exp config/model shared by the HF reference generators (issue #816).

4 layers: GDN, GDN (+PLE, 0-based layer 1), GDN, QSA. NK != NV (2 vs 4), GQA 4:2, 3 indexer heads, 8 experts / top-3,
indexer budget 16 tokens / ratio 4 -> top-4 blocks so the sparse path engages after ~19 tokens.
"""
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel


def tiny_config(**over):
    kw = dict(
        vocab_size=100, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        moe_intermediate_size=24, shared_expert_intermediate_size=24, num_experts=8, num_experts_per_tok=3,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, hc_count=4, hc_lowrank=20, ple_layer_ids=[2], ple_embed_dim=64, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=50, split_ngram_parts=4, indexer_n_heads=3, indexer_kv_heads=1,
        indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4, output_gate_type="sigmoid", eos_token_id=5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], pad_token_id=0,
        max_position_embeddings=512,
    )
    kw.update(over)
    cfg = Qwen4ExpTextConfig(**kw)
    cfg._attn_implementation = "eager"
    return cfg


def randomize(model, seed=0, scale=1.0):
    """Give every parameter a non-degenerate random value (HF zero-inits several of them)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log") or name.endswith("dt_bias"):
                p.copy_(torch.rand(p.shape, generator=g) * 2.0 - 1.0)
            elif "linear_attn.norm" in name:
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.2)  # plain gain (not 1+w)
            elif "norm" in name and name.endswith("weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)        # (1 + w) gains
            elif "embed_tokens" in name or "ngram_embedding" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            elif "conv1d" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.6)
            elif "indexer" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5 * scale)
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) * (0.7 / max(fan_in, 1) ** 0.5) * scale)
    return model


def build_model(seed=0, **over):
    torch.manual_seed(seed)
    model = Qwen4ExpTextModel(tiny_config(**over)).eval()
    return randomize(model, seed)
