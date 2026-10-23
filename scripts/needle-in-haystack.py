"""Needle-in-a-haystack retrieval against a running dotllm server (issue #1101).

Hides a passkey sentence early in a long wikitext context, asks for it at the end, and records whether the answer
contains it, plus prompt size, time to first token and the server's prefill/decode timings. With --target-tokens past a
model's native window (262,144 for Bonsai-2 / Qwen3.8) this checks that YaRN actually carries attention that far.

Example (Strix, YaRN on):
  python needle-in-haystack.py --url http://127.0.0.1:18101 --target-tokens 300000 --depth 0.02 --out needle.json
"""
import argparse, json, pathlib, random, time, urllib.error, urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--url", default="http://127.0.0.1:18101", help="server base URL")
ap.add_argument("--model", default=None, help="model id (default: first of GET /v1/models)")
ap.add_argument("--corpus", default=str(REPO_ROOT / ".docs" / "corpora" / "wikitext2-test.raw"))
ap.add_argument("--target-tokens", type=int, required=True, help="approximate prompt size in tokens")
ap.add_argument("--depth", type=float, default=0.02, help="needle position as a fraction of the haystack (0 = start)")
ap.add_argument("--passkey", type=int, default=None, help="the number to hide (default: random 5-digit, seeded)")
ap.add_argument("--seed", type=int, default=1101)
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--timeout", type=int, default=3 * 3600, help="request timeout in seconds")
ap.add_argument("--out", default="needle.json")
args = ap.parse_args()
BASE = args.url.rstrip("/")


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(text):
    return post("/v1/tokenize", {"text": text, "model": model})["count"]


model = args.model or json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=60).read())["data"][0]["id"]
passkey = args.passkey if args.passkey is not None else random.Random(args.seed).randint(10000, 99999)
needle = f" The secret passkey is {passkey}. Remember it: the secret passkey is {passkey}. "
question = ("\n\nThe document above contains one sentence stating a secret passkey. What is the secret passkey? "
            "Reply with the number only.")

corpus = open(args.corpus, encoding="utf-8").read()
# Chars per token from a sample, then size the haystack; repeat the corpus if it is too short.
sample = corpus[:200_000]
cpt = len(sample) / max(1, count_tokens(sample))
want_chars = int(args.target_tokens * cpt)
hay = (corpus * (want_chars // len(corpus) + 1))[:want_chars]
cut = hay.rfind(". ", 0, max(2, int(len(hay) * args.depth))) + 1   # a sentence boundary at or before the depth
hay = hay[:cut] + needle + hay[cut:]
content = "Read the following document.\n\n" + hay + question

prompt_tokens = count_tokens(content)
print(f"model {model}; {cpt:.2f} chars/token; prompt ~{prompt_tokens} tokens; needle at char {cut} "
      f"({cut / len(hay):.3%}); passkey {passkey}", flush=True)

body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": args.max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "enable_thinking": False}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time(); ttft = None; reply = []; timings = {}; usage = {}; err = ""
try:
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("error"):
                err = json.dumps(d["error"])[:300]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece and ttft is None:
                    ttft = time.time() - t0
                if delta.get("content"):
                    reply.append(delta["content"])
            timings = d.get("timings") or timings
            usage = d.get("usage") or usage
except urllib.error.HTTPError as e:
    err = f"HTTP {e.code}: {e.read().decode()[:300]}"
except Exception as e:  # noqa: BLE001 - record the failure
    err = repr(e)[:300]

answer = "".join(reply)
result = {
    "model": model, "target_tokens": args.target_tokens, "prompt_tokens": prompt_tokens,
    "needle_depth": cut / len(hay), "passkey": passkey, "answer": answer, "found": str(passkey) in answer,
    "ttft_s": ttft, "wall_s": time.time() - t0, "timings": timings, "usage": usage, "error": err,
}
pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
