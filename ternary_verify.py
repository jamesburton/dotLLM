import os, sys, torch
sys.path.insert(0, os.path.join("scripts", "lora"))
import bitdistill as bd
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
bd.convert_to_bitnet_student(m)
st = torch.load(".docs/bitdistill/qwen3_0p6b_poc/final/student_state.pt", map_location="cpu")
m.load_state_dict(st, strict=False)

def stats(name, bl):
    w = bl.weight.float()
    wq = bd.weight_quant_ternary(w)          # exactly what the ternary forward uses (alpha=1)
    uniq = torch.unique(wq)
    scale = float(uniq.abs().max())
    levels = sorted(set((wq / (scale if scale > 0 else 1.0)).round().int().unique().tolist()))
    zeros = float((wq == 0).float().mean()) * 100
    print(f"{name:28s} distinct={uniq.numel()} levels={levels} scale={scale:.5f} sparsity(zeros)={zeros:.1f}%")

L0 = m.model.layers[0]
stats("L0.self_attn.q_proj", L0.self_attn.q_proj)
stats("L0.self_attn.o_proj", L0.self_attn.o_proj)
stats("L0.mlp.gate_proj", L0.mlp.gate_proj)
stats("L0.mlp.up_proj", L0.mlp.up_proj)
stats("L0.mlp.down_proj", L0.mlp.down_proj)

nbit = sum(1 for mod in m.modules() if isinstance(mod, bd.BitLinear))
print(f"\ntotal ternary BitLinear layers: {nbit}")
print(f"embed_tokens dtype={m.model.embed_tokens.weight.dtype} (kept FP, not ternarized)")
print(f"a BitLinear's stored master dtype={L0.mlp.gate_proj.weight.dtype} (FP master; ternary applied in forward / by dotLLM I2_S at load)")
