"""Inspect quant types for a list of tensor names in a GGUF."""
import sys
from gguf import GGUFReader

reader = GGUFReader(sys.argv[1])
names = sys.argv[2:] if len(sys.argv) > 2 else [
    "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
    "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
    "blk.0.ssm_conv1d.weight", "blk.0.ssm_out.weight",
    "blk.0.ffn_gate_inp.weight",
    "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
    "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
    "blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
    "token_embd.weight", "output.weight",
    "blk.39.attn_q.weight", "blk.39.attn_k.weight",
]
for n in names:
    t = next((t for t in reader.tensors if t.name == n), None)
    if t is None:
        print(f"{n}: (missing)")
        continue
    print(f"{n}: {t.tensor_type.name} shape={list(t.shape)}")
