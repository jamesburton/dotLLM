"""Chat-template fidelity: dotLLM Jinja renderer vs llama.cpp (minja) /apply-template on identical messages/tools."""
import json, os, subprocess, time, urllib.request
S = os.path.dirname(os.path.abspath(__file__))
WT = r"C:\Development\dotLLM\.claude\worktrees\agent-a08e22aa7248079ed"
D = os.path.expanduser(r"~\.dotllm\test-cache\nvidia\nemotron-3.5-lightning")
GGUF = D + r"\nemotron-3.5-lightning-Q4_K_M-trunk52.gguf"
PROBE = WT + r"\benchmarks\NemotronProbe\bin\Release\net10.0\NemotronProbe.exe"
srv = subprocess.Popen([r"C:\Development\llama.cpp\llama-server.exe", "-m", GGUF, "-dev", "none", "-ngl", "0", "--jinja", "--port", "8099", "-c", "512", "--no-warmup"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2); break
        except Exception:
            time.sleep(2)
    cases = {
        "multiturn": ([{"role": "system", "content": "You are a concise assistant."}, {"role": "user", "content": "Hi! I'm Marta."},
                       {"role": "assistant", "content": "Hello Marta!"}, {"role": "user", "content": "What was my name?"}], None),
        "tool": ([{"role": "user", "content": "What's the weather like in Paris right now?"}],
                 [{"name": "get_weather", "description": "Get the current weather for a city.",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}]),
        "tool_roundtrip": ([{"role": "user", "content": "Weather in Paris?"},
                            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                            {"role": "tool", "tool_call_id": "c1", "content": "18C and sunny"}], None),
    }
    for name, (msgs, tools) in cases.items():
        if name == "tool_roundtrip":
            continue  # probe render takes plain role/content messages only
        mp = os.path.join(S, "cases", "tc_" + name + ".msgs.json"); json.dump(msgs, open(mp, "w"))
        cmd = [PROBE, "render", GGUF, mp]
        body = {"messages": msgs}
        if tools:
            tp = os.path.join(S, "cases", "tc_" + name + ".tools.json"); json.dump(tools, open(tp, "w")); cmd.append(tp)
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        mine = subprocess.run(cmd, capture_output=True).stdout.decode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:8099/apply-template", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        theirs = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt"]
        same = mine == theirs
        print(f"### {name}: identical={same} (dotLLM {len(mine)} chars, llama.cpp {len(theirs)} chars)")
        if not same:
            k = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
            print("  first diff at", k, "\n  dotLLM :", repr(mine[max(0, k-60):k+80]), "\n  llama  :", repr(theirs[max(0, k-60):k+80]))
finally:
    srv.kill()
