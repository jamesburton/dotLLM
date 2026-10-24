"""Build the QAT token stream (spec 'c4_wiki'): WikiText-2 train + a C4 en shard slice, tokenised per
document with default special tokens, concatenated, chunked into seq-length blocks (remainder dropped).
Output: flat uint32 .bin (vocab > 65535). Fetch goes through huggingface_hub (default HF cache)."""
import argparse, gzip, json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--c4-tokens", type=int, default=20_000_000)
    ap.add_argument("--c4-shard", default="en/c4-train.00000-of-01024.json.gz")
    a = ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    chunks = []
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(wt["text"]))["input_ids"]
    chunks.append(np.asarray(ids, dtype=np.uint32)); print("wikitext2 train tokens", len(ids), flush=True)
    path = hf_hub_download("allenai/c4", a.c4_shard, repo_type="dataset")
    n, batch = 0, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) == 512:
                for d in tok(batch)["input_ids"]:
                    chunks.append(np.asarray(d, dtype=np.uint32)); n += len(d)
                batch = []
                if n >= a.c4_tokens: break
    print("c4 tokens", n, flush=True)
    allt = np.concatenate(chunks)
    allt = allt[: (len(allt) // a.seq) * a.seq]
    allt.tofile(a.out)
    print("wrote", a.out, len(allt), "tokens,", len(allt) // a.seq, "blocks")


if __name__ == "__main__":
    main()
