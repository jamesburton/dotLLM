"""Small-task suite for a fast test model. usage: small-model-tasks.py <base_url> [model]"""
import json, sys, time, urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "m"


def chat(messages, tools=None, max_tokens=200):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d, time.time() - t


def j(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    return json.loads(s[s.find("{"): s.rfind("}") + 1])


WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

TASKS = [
    ("arith-mul", [{"role": "user", "content": "What is 17*23? Answer with just the number."}], None,
     lambda m, tc: "391" in m),
    ("arith-add", [{"role": "user", "content": "What is 1234+5678? Answer with just the number."}], None,
     lambda m, tc: "6912" in m.replace(",", "")),
    ("json-extract", [{"role": "user", "content": "Extract the fields from this text and answer with ONLY a JSON object with keys name (string), age (integer), city (string): 'Alice, who is 31 years old, lives in Paris.'"}], None,
     lambda m, tc: (lambda o: o["name"] == "Alice" and int(o["age"]) == 31 and o["city"] == "Paris")(j(m))),
    ("classify", [{"role": "user", "content": "Classify the sentiment of this review as exactly one word, positive or negative: 'I love this product, it works great!'"}], None,
     lambda m, tc: m.strip().strip(".").lower() == "positive"),
    ("short-qa", [{"role": "user", "content": "What is the capital of Australia? Answer in one word."}], None,
     lambda m, tc: "canberra" in m.lower()),
    ("summary", [{"role": "user", "content": "Summarize in one sentence of at most 20 words: 'The city council voted on Tuesday to build a new bridge over the river. Construction will start in March and is expected to cost 12 million dollars. Residents hope it will reduce traffic.'"}], None,
     lambda m, tc: "bridge" in m.lower() and len(m.split()) <= 30),
    ("tool-call", [{"role": "user", "content": "What is the weather in Paris right now?"}], WEATHER,
     lambda m, tc: bool(tc) and tc[0]["function"]["name"] == "get_weather" and "paris" in tc[0]["function"]["arguments"].lower()),
]

res = []
for name, msgs, tools, check in TASKS:
    try:
        d, dt = chat(msgs, tools)
        ch = d["choices"][0]
        m = ch["message"].get("content") or ""
        tc = ch["message"].get("tool_calls")
        try:
            ok = bool(check(m, tc))
        except Exception as e:
            ok = False
        u = d.get("usage", {})
        res.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name:12s} {dt:6.2f}s prompt={u.get('prompt_tokens')} gen={u.get('completion_tokens')} finish={ch.get('finish_reason')} out={m.strip()[:110]!r} tc={json.dumps(tc)[:120] if tc else None}")
    except Exception as e:
        res.append(False)
        print(f"ERR  {name}: {e}")
print(f"RESULT {sum(res)}/{len(res)}")
