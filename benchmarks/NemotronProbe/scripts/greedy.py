"""Greedy text identity: dotLLM CPU `run` vs llama.cpp CPU `llama-completion`, on identical rendered prompts."""
import json, os, subprocess, sys, time

S = os.path.dirname(os.path.abspath(__file__))
WT = r"C:\Development\dotLLM\.claude\worktrees\agent-a08e22aa7248079ed"
D = os.path.expanduser(r"~\.dotllm\test-cache\nvidia\nemotron-3.5-lightning")
GGUF = os.environ.get("NEM_GGUF", D + r"\nemotron-3.5-lightning-Q4_K_M-trunk52.gguf")
GGUF_DOT = os.environ.get("NEM_GGUF_DOT", GGUF)
CLI = WT + r"\src\DotLLM.Cli\bin\Release\net10.0\DotLLM.Cli.exe"
PROBE = WT + r"\benchmarks\NemotronProbe\bin\Release\net10.0\NemotronProbe.exe"
LLAMA = r"C:\Development\llama.cpp"
N = int(os.environ.get("NEM_N", "128"))
OUT = os.path.join(S, "cases"); os.makedirs(OUT, exist_ok=True)
only = sys.argv[1:]

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)

def render(msgs, tools=None, name="x"):
    mp = os.path.join(OUT, name + ".msgs.json"); json.dump(msgs, open(mp, "w", encoding="utf-8"))
    cmd = [PROBE, "render", GGUF, mp]
    if tools:
        tp = os.path.join(OUT, name + ".tools.json"); json.dump(tools, open(tp, "w", encoding="utf-8")); cmd.append(tp)
    r = run(cmd)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    return r.stdout.decode("utf-8")

NOTHINK = "<think></think>"
def user(c): return {"role": "user", "content": c}
def asst(c): return {"role": "assistant", "content": c}
def sysm(c): return {"role": "system", "content": c}

def nothink(text):  # swap generation prompt to the non-thinking form
    assert text.endswith("<|im_start|>assistant\n<think>\n"), text[-60:]
    return text[: -len("<think>\n")] + NOTHINK

doc = open(WT + r"\docs\KV_CACHE.md", encoding="utf-8").read().replace("\r\n", "\n")
filler_src = ""
for f in ["docs/ARCHITECTURE.md", "docs/SCHEDULING.md", "docs/SAMPLING.md", "docs/LORA.md"]:
    filler_src += open(os.path.join(WT, f), encoding="utf-8", errors="replace").read().replace("\r\n", "\n") + "\n"
filler = filler_src[:15500]
needle = "\n\nNOTE FROM THE ARCHIVIST: the vault passphrase is TANGERINE-7342-OWL. Remember it.\n\n"
half = len(filler) // 2
hay = filler[:half] + needle + filler[half:]

cases = {}
cases["arith_think"] = render([user("What is 487 * 36 + 129? Work it out step by step.")], name="arith_think")
cases["arith"] = nothink(render([user("What is 487 * 36 + 129? Give the final number only after a one-line calculation.")], name="arith"))
cases["json"] = nothink(render([user('Extract name, city and age from this text as a JSON object and output only the JSON: "Priya Raman, a 34 year old architect from Leeds, said she loves the new library."')], name="json"))
cases["code"] = nothink(render([user("Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Include a short docstring. Code only.")], name="code"))
cases["summary"] = nothink(render([user("Summarise the following passage in two sentences.\n\n" + doc[:2600])], name="summary"))
cases["multiturn"] = nothink(render([
    sysm("You are a concise assistant."),
    user("Hi! My name is Marta and I'm planning a trip to Kyoto in April."),
    asst("Hello Marta! April is cherry-blossom season in Kyoto, a lovely time to visit. What would you like help with?"),
    user("What was my name again, and which month am I going?")], name="multiturn"))
tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}]
cases["tool"] = nothink(render([user("What's the weather like in Paris right now?")], tools=tools, name="tool"))
cases["needle4k"] = nothink(render([user(hay + "\n\nQuestion: what is the vault passphrase mentioned in the text above? Answer with the passphrase only.")], name="needle4k"))

results = {}
for name, prompt in cases.items():
    if only and name not in only: continue
    pf = os.path.join(OUT, name + ".prompt.txt")
    open(pf, "w", encoding="utf-8", newline="").write(prompt)
    # token ids on both sides
    t_dot = run([PROBE, "tok", GGUF, pf]).stdout.decode().split()
    t_ll = run([LLAMA + r"\llama-tokenize.exe", "-m", GGUF, "-f", pf, "--ids", "--no-bos", "--log-disable"]).stdout.decode().strip()
    t_ll = [x for x in t_ll.replace("[", " ").replace("]", " ").replace(",", " ").split() if x.lstrip("-").isdigit()]
    same_tok = t_dot == t_ll
    # llama.cpp
    t0 = time.time()
    rl = run([LLAMA + r"\llama-completion.exe", "-m", GGUF, "-dev", "none", "-ngl", "0", "-p", prompt, "-n", str(N), "--temp", "0",
              "-no-cnv", "--no-display-prompt", "-c", "6144", "--no-warmup"])
    ll_text = rl.stdout.decode("utf-8", "replace")
    ll_err = rl.stderr.decode("utf-8", "replace")
    tl = time.time() - t0
    # dotLLM
    t0 = time.time()
    rd = run([CLI, "run", GGUF_DOT, "-d", os.environ.get("NEM_DOT_DEVICE", "cpu"), "-p", prompt, "-n", str(N), "--json"])
    td = time.time() - t0
    line = [l for l in rd.stdout.decode("utf-8", "replace").splitlines() if l.startswith("{")][-1]
    j = json.loads(line)
    dot_text = j["text"]
    # strip llama's trailing " [end of text]" marker
    ll_clean = ll_text.replace("\r", "").replace(" [end of text]", "").rstrip("\n")
    dot_text = dot_text.replace("\r", "")
    n_common = 0
    for a, b in zip(dot_text, ll_clean):
        if a != b: break
        n_common += 1
    ident = dot_text.strip() == ll_clean.strip()
    results[name] = dict(prompt_tokens_dot=len(t_dot), prompt_tokens_llama=len(t_ll), tokenization_identical=same_tok,
                         text_identical=ident, common_prefix_chars=n_common, dot_len=len(dot_text), ll_len=len(ll_clean),
                         dot_tok_s=j["timings"]["decode_tok_s"], dot_prefill_tok_s=j["timings"]["prefill_tok_s"],
                         dot_gen=j["usage"]["generated_tokens"], secs_llama=round(tl, 1), secs_dot=round(td, 1))
    print(f"### {name}: {json.dumps(results[name])}")
    print("--- dotLLM:\n" + dot_text[:700])
    print("--- llama.cpp:\n" + ll_clean[:700], flush=True)
    json.dump(results, open(os.path.join(OUT, "results_%s.json" % os.environ.get("NEM_TAG", "cpu")), "w"), indent=1)
