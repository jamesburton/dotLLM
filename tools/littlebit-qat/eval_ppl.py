"""WikiText-2 test perplexity (spec 5): join rows with \n\n, tokenise once, non-overlapping seq windows."""
import argparse, math, torch


def ppl(model, tok, seq=2048, device="cuda", max_windows=None):
    from datasets import load_dataset
    ids = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]),
              return_tensors="pt").input_ids
    n = ids.shape[1] // seq
    if max_windows: n = min(n, max_windows)
    nll = 0.0
    with torch.no_grad():
        for i in range(n):
            x = ids[:, i * seq:(i + 1) * seq].to(device)
            lg = model(input_ids=x, use_cache=False).logits.float()
            nll += torch.nn.functional.cross_entropy(lg[0, :-1], x[0, 1:], reduction="sum").item()
    return math.exp(nll / (n * (seq - 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id (teacher) or exported littlebit dir")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--max-windows", type=int)
    a = ap.parse_args()
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from littlebit import load_checkpoint
    tok = AutoTokenizer.from_pretrained(a.model)
    if os.path.exists(os.path.join(a.model, "littlebit_config.json")):
        m = load_checkpoint(a.model, device="cuda")
    else:
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    print("wikitext2 ppl", ppl(m, tok, a.seq, "cuda", a.max_windows))
