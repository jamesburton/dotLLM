"""Teacher-forced top-1: llama.cpp greedy text (128 tok) fed through dotLLM; does our argmax follow llama's tokens?"""
import os, subprocess, sys
S = os.path.dirname(os.path.abspath(__file__))
WT = r"C:\Development\dotLLM\.claude\worktrees\agent-a08e22aa7248079ed"
D = os.path.expanduser(r"~\.dotllm\test-cache\nvidia\nemotron-3.5-lightning")
GGUF = D + r"\nemotron-3.5-lightning-Q4_K_M-trunk52.gguf"
PROBE = WT + r"\benchmarks\NemotronProbe\bin\Release\net10.0\NemotronProbe.exe"
LLAMA = r"C:\Development\llama.cpp"
device = os.environ.get("NEM_DEV", "cpu")
def run(c): return subprocess.run(c, capture_output=True)
for name in sys.argv[1:]:
    pf = os.path.join(S, "cases", name + ".prompt.txt")
    prompt = open(pf, encoding="utf-8", newline="").read()
    gf = os.path.join(S, "cases", name + ".llama.txt")
    if not os.path.exists(gf):
        rl = run([LLAMA + r"\llama-completion.exe", "-m", GGUF, "-dev", "none", "-ngl", "0", "-p", prompt, "-n", os.environ.get("NEM_N", "128"),
                  "--temp", "0", "-no-cnv", "--no-display-prompt", "-c", "6144", "--no-warmup"])
        t = rl.stdout.decode("utf-8", "replace").replace("\r", "").replace(" [end of text]", "")
        open(gf, "w", encoding="utf-8", newline="").write(t)
    gen = open(gf, encoding="utf-8", newline="").read().rstrip()  # llama.cpp prints newlines after EOG; not model tokens
    full = os.path.join(S, "cases", name + ".full.txt"); open(full, "w", encoding="utf-8", newline="").write(prompt + gen)
    ids_p = run([PROBE, "tok", GGUF, pf]).stdout.decode().split()
    ids_f = run([PROBE, "tok", GGUF, full]).stdout.decode().split()
    canon = ids_f[:len(ids_p)] == ids_p
    idf = os.path.join(S, "cases", name + ".full.ids"); open(idf, "w").write(" ".join(ids_f))
    r = run([PROBE, "forced", GGUF, idf, str(len(ids_p)), device])
    print(f"### {name} [{device}] prompt={len(ids_p)} gen_tokens~{len(ids_f)-len(ids_p)} prompt-boundary-canonical={canon}")
    print(r.stdout.decode().strip() or r.stderr.decode()[-500:], flush=True)
