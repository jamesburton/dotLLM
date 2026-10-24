#!/usr/bin/env python3
"""Capture llama.cpp reference embeddings + token ids for a BERT-class embedding GGUF (issue #739).

Runs llama-server (CPU, -ngl 0) on the SAME gguf the integration test loads, once with the
checkpoint's own pooling (no --pooling flag) and queries /tokenize + /v1/embeddings.
Vectors are L2-normalised by llama.cpp (--embd-normalize default 2).

usage: capture_llamacpp_bert_embeddings.py <llama-server.exe> <model.gguf> <out.json> <repo> <file> [--pooling P] [--prefix STR]
"""
import json, subprocess, sys, time, urllib.request, datetime

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A man is eating food.",
    "What is the capital of France?",
    "dotLLM is a native .NET inference engine.",
    "Café naïve résumé Zürich",
    "北京是中国的首都 and mixed English",
    "Hello",
    "The 2024 results: revenue rose 12.5% (YoY), beating estimates -- but costs also climbed.",
    ("Transformers process sequences with self-attention, letting every token attend to every other token. "
     "Encoder-only models such as BERT use bidirectional attention and are pooled into a single vector, "
     "which makes them well suited to semantic search, clustering and retrieval-augmented generation. "
     "Sentence embeddings are usually compared with cosine similarity after L2 normalisation, so tiny "
     "numerical differences in the forward pass show up directly as a lower cosine against the reference."),
]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    server, model, out, repo, fname = sys.argv[1:6]
    pooling = None
    prefix = ""
    if "--pooling" in sys.argv:
        pooling = sys.argv[sys.argv.index("--pooling") + 1]
    if "--prefix" in sys.argv:
        prefix = sys.argv[sys.argv.index("--prefix") + 1]
    port = 18739
    args = [server, "-m", model, "--embeddings", "-ngl", "0", "--port", str(port), "--host", "127.0.0.1",
            "-c", "4096", "-ub", "4096", "-b", "4096", "--no-warmup", "-np", "1"]
    if pooling:
        args += ["--pooling", pooling]
    ver = subprocess.run([server, "--version"], capture_output=True, text=True)
    vlines = (ver.stdout + ver.stderr).splitlines()
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("llama-server never became healthy")
        texts = [prefix + t for t in TEXTS]
        toks = [post(f"http://127.0.0.1:{port}/tokenize", {"content": t, "add_special": True})["tokens"] for t in texts]
        r = post(f"http://127.0.0.1:{port}/v1/embeddings", {"model": "ref", "input": texts})
        embs = [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
        props = post(f"http://127.0.0.1:{port}/props", {}) if False else None
    finally:
        proc.kill()
    doc = {
        "reference": {
            "tool": "llama-server (llama.cpp)",
            "version": next((l for l in vlines if l.startswith("version")), ""),
            "built_with": next((l for l in vlines if l.startswith("built")), ""),
            "flags": " ".join(a for a in args[1:] if a not in (model,)).replace(str(port), "<port>"),
            "captured": datetime.date.today().isoformat(),
            "pooling_flag": pooling or "(checkpoint default)",
            "prefix": prefix,
            "notes": ["-ngl 0 pins the llama.cpp CPU backend", "vectors are L2-normalised (default --embd-normalize 2)",
                      "tokens come from /tokenize add_special=true, i.e. llama.cpp's own WPM tokenizer"],
        },
        "model": {"repo": repo, "file": fname, "hidden_size": len(embs[0])},
        "usage_prompt_tokens": r["usage"]["prompt_tokens"],
        "inputs": [{"text": t, "tokens": k} for t, k in zip(texts, toks)],
        "embeddings": embs,
    }
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", out, len(embs), "x", len(embs[0]))


main()
