import json, subprocess, sys, time, urllib.request, os, glob
sys.path.insert(0, "C:/Development/dotLLM/.claude/worktrees/agent-a83b71ed9d48197fe/tests/scripts")
HF = os.path.expanduser("~/.cache/huggingface/hub")
def g(p): return glob.glob(f"{HF}/{p}")[0]
MODELS = {
 "minilm-q8":  g("models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots/*/all-MiniLM-L6-v2-Q8_0.gguf"),
 "minilm-f16": g("models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots/*/all-MiniLM-L6-v2-ggml-model-f16.gguf"),
 "mxbai-q8":   g("models--ChristianAzinn--mxbai-embed-large-v1-gguf/snapshots/*/mxbai-embed-large-v1.Q8_0.gguf"),
 "mxbai-f32":  g("models--ChristianAzinn--mxbai-embed-large-v1-gguf/snapshots/*/mxbai-embed-large-v1_fp32.gguf"),
 "nomic-q8":   g("models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/*/nomic-embed-text-v1.5.Q8_0.gguf"),
 "nomic-q4km": g("models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/*/nomic-embed-text-v1.5.Q4_K_M.gguf"),
 "nomic-f32":  g("models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/*/nomic-embed-text-v1.5.f32.gguf"),
}
src = open("C:/Development/dotLLM/.claude/worktrees/agent-a83b71ed9d48197fe/tests/scripts/capture_llamacpp_bert_embeddings.py", encoding="utf-8").read()
ns = {}; exec(src.split("def post")[0], ns); TEXTS = ns["TEXTS"]
LS = {
 "ll8683cpu": ("C:/Development/llama.cpp/llama-server.exe", 0),
 "ll9016cpu": ("C:/Development/llama.cpp-b9016/llama-server.exe", 0),
 "ll9672cpu": ("C:/Development/llamacpp-vulkan/llama-server.exe", 0),
 "ll9747cpu": ("C:/Users/james/.cache/lemonade/bin/llamacpp/vulkan/llama-server.exe", 0),
 "ll9016vk":  ("C:/Development/llama.cpp-b9016/llama-server.exe", 99),
 "ll9747vk":  ("C:/Users/james/.cache/lemonade/bin/llamacpp/vulkan/llama-server.exe", 99),
}
def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
def run_ll(prov, model):
    exe, ngl = LS[prov]; port = 18801
    args = [exe, "-m", MODELS[model], "--embeddings", "-ngl", str(ngl), "--port", str(port), "--host", "127.0.0.1",
            "-c", "4096", "-ub", "4096", "-b", "4096", "--no-warmup", "-np", "1"]
    if ngl == 0: args += ["--device", "none"]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(400):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok": break
            except Exception: time.sleep(0.5)
        else: raise RuntimeError("no health")
        toks = [post(f"http://127.0.0.1:{port}/tokenize", {"content": t, "add_special": True})["tokens"] for t in TEXTS]
        r = post(f"http://127.0.0.1:{port}/v1/embeddings", {"input": TEXTS})
        return {"tokens": toks, "emb": [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]}
    finally:
        p.kill(); p.wait()
def run_ollama(model):
    name = "dotllm-prov-" + model
    open("Modelfile", "w").write(f"FROM {MODELS[model]}\n")
    subprocess.run(["ollama", "create", name, "-f", "Modelfile"], check=True, capture_output=True)
    r = post("http://127.0.0.1:11434/api/embed", {"model": name, "input": TEXTS, "options": {"num_gpu": 0}, "keep_alive": 0})
    return {"count": r.get("prompt_eval_count"), "emb": r["embeddings"]}
if __name__ == "__main__":
    out = json.load(open("results.json")) if os.path.exists("results.json") else {}
    for model in MODELS:
        for prov in list(LS) + ["ollama"]:
            k = f"{model}|{prov}"
            if k in out: continue
            try:
                out[k] = run_ollama(model) if prov == "ollama" else run_ll(prov, model)
                print("ok", k, flush=True)
            except Exception as e:
                out[k] = {"error": str(e)}; print("ERR", k, e, flush=True)
            json.dump(out, open("results.json", "w"))
