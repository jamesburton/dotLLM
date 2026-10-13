import os, sys, json, torch
sys.path.insert(0, os.path.join("scripts", "lora"))
import bitdistill as bd
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = ".docs/bitdistill/qwen3_0p6b_poc/final/student_state.pt"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# build + convert + load on CPU, THEN move to cuda (new BitLinear modules are created
# during convert -> must move the whole model after convert, like bitdistill.train()).
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
bd.convert_to_bitnet_student(m)
state = torch.load(CKPT, map_location="cpu")
miss, unexp = m.load_state_dict(state, strict=False)
print(f"[ref] loaded student_state (missing={len(miss)} unexpected={len(unexp)})")
m = m.to("cuda")
m.eval(); bd.set_quant_alpha(m, 1.0); m.config.use_cache = True

out = {"prompts": []}
for prompt in ["The capital of France is", "2 + 2 =", "def add(a, b):"]:
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        logits = m(ids, use_cache=False).logits[0, -1].float()
        top5 = torch.topk(logits, 5)
        gen = m.generate(ids, max_new_tokens=24, do_sample=False,
                         pad_token_id=(tok.eos_token_id or 0))
    entry = {
        "prompt": prompt,
        "prompt_ids": ids[0].tolist(),
        "first_tok_top5": [[tok.decode([i]), round(v, 4)]
                           for v, i in zip(top5.values.tolist(), top5.indices.tolist())],
        "gen_ids": gen[0].tolist(),
        "gen_text": tok.decode(gen[0], skip_special_tokens=True),
    }
    out["prompts"].append(entry)
    print(f"[ref] {prompt!r} -> {entry['gen_text']!r}")
    print(f"      first-tok top5: {entry['first_tok_top5']}")

with open(".docs/bitdistill/qwen3_0p6b_dotllm/pytorch_reference.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("[ref] wrote pytorch_reference.json")
