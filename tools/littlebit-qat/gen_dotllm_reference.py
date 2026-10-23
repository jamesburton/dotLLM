"""Reference data for the dotLLM LittleBit CPU model tests (issue #864).

Usage: python gen_dotllm_reference.py <littlebit export dir> <out dir> [--windows 40]

Writes into <out dir>:
  generation_reference.json  3 prompts: prompt ids, 40 greedy continuation ids (bf16 torch model, trainer's forward), and for every
                             teacher-forced position the torch argmax plus its top-8 (id, logit) pairs.
  wikitext2_first40.tokens   whitespace-separated token ids of the first N non-overlapping 2048-token windows, built exactly as
                             eval_ppl.ppl builds them (join rows with \n\n, tokenise once). Feed to `dotllm perplexity --tokens-file`.
  Prints the trainer's own PPL over those windows (the figure dotLLM must reproduce).
"""
import json, math, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from littlebit import load_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset

ckpt, out = sys.argv[1], sys.argv[2]
nwin = int(sys.argv[sys.argv.index("--windows") + 1]) if "--windows" in sys.argv else 40
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(ckpt)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = load_checkpoint(ckpt, device=dev)

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    if n < 2:\n        return n", "In 1969, the first humans"]
NEW = 40
cases = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=NEW, do_sample=False, use_cache=True, min_new_tokens=NEW)
        lg = model(input_ids=gen, use_cache=False).logits.float()[0]
    n0 = ids.shape[1]
    steps = []
    for pos in range(n0 - 1, gen.shape[1] - 1):
        top = torch.topk(lg[pos], 8)
        steps.append(dict(argmax=int(top.indices[0]), top_ids=top.indices.tolist(), top_logits=[round(v, 4) for v in top.values.tolist()]))
    cases.append(dict(prompt=p, prompt_ids=ids[0].tolist(), generated_ids=gen[0, n0:].tolist(), steps=steps,
                      text=tok.decode(gen[0, n0:])))
# Same teacher-forced positions through an FP32 copy of the model (activations and accumulation in fp32): the bf16 torch forward
# rounds every stage to 8 mantissa bits, so near-tied logits flip against ANY fp32 engine; this is the like-for-like oracle.
m32 = load_checkpoint(ckpt, dtype=torch.float32, device=dev)
for c in cases:
    full = torch.tensor([c["prompt_ids"] + c["generated_ids"]], device=dev)
    with torch.no_grad():
        lg32 = m32(input_ids=full, use_cache=False).logits.float()[0]
    n0 = len(c["prompt_ids"])
    c["steps_fp32"] = []
    for pos in range(n0 - 1, full.shape[1] - 1):
        top = torch.topk(lg32[pos], 8)
        c["steps_fp32"].append(dict(argmax=int(top.indices[0]), top_ids=top.indices.tolist(), top_logits=[round(v, 4) for v in top.values.tolist()]))
    with torch.no_grad():
        g32 = m32.generate(full[:, :n0], max_new_tokens=NEW, do_sample=False, min_new_tokens=NEW)
    c["generated_ids_fp32"] = g32[0, n0:].tolist()
del m32
json.dump(dict(new_tokens=NEW, cases=cases), open(f"{out}/generation_reference.json", "w"))
if "--skip-ppl" in sys.argv: sys.exit(0)

text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
all_ids = tok(text, return_tensors="pt").input_ids
seq = 2048
n = min(all_ids.shape[1] // seq, nwin)
open(f"{out}/wikitext2_first{n}.tokens", "w").write(" ".join(map(str, all_ids[0, : n * seq].tolist())))
nll = 0.0
with torch.no_grad():
    for i in range(n):
        x = all_ids[:, i * seq:(i + 1) * seq].to(dev)
        l = model(input_ids=x, use_cache=False).logits.float()
        nll += torch.nn.functional.cross_entropy(l[0, :-1], x[0, 1:], reduction="sum").item()
print(f"windows={n} total_ids={all_ids.shape[1]} trainer-style PPL={math.exp(nll / (n * (seq - 1))):.4f}")
