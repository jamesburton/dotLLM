"""Golden renders for the Liquid AI LFM3 chat template (#1560).

Usage: python generate_lfm3_render_cases.py lfm3-chat-template.jinja lfm3-render-cases.json

lfm3-chat-template.jinja is LiquidAI/LFM3-1.7B chat_template.jinja verbatim. It is rendered with a jinja2 environment that
mirrors transformers' apply_chat_template compiler (sandboxed, trim_blocks, lstrip_blocks, HF's tojson with
ensure_ascii=False and default separators, raise_exception, {% generation %} as a pass-through block), so the expected
strings are HF semantics, not a re-implementation of dotLLM's evaluator.
"""
import json
import sys
from jinja2 import Environment
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment
from jinja2 import nodes

class GenerationExt(Extension):
    tags = {"generation"}
    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_pass", []), [], [], body).set_lineno(lineno)
    def _pass(self, caller):
        return caller()

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

def raise_exception(m):
    raise Exception(m)

env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationExt])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(sys.argv[1], encoding="utf-8", newline="").read())

weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}
cases = [
    ("user_only", dict(messages=[{"role": "user", "content": "Hello!"}])),
    ("system_user", dict(messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}])),
    ("multi_turn", dict(messages=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}])),
    ("multi_turn_thinking_stripped", dict(messages=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "<think>hmm</think>A1"}, {"role": "user", "content": "Q2"}])),
    ("assistant_reasoning_last", dict(messages=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1", "reasoning_content": "because"}])),
    ("no_gen_prompt", dict(messages=[{"role": "user", "content": "Hello!"}], add_generation_prompt=False)),
    ("thinking_off", dict(messages=[{"role": "user", "content": "Hello!"}], enable_thinking=False)),
    ("preserve_thinking", dict(messages=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1", "reasoning_content": "r1"}, {"role": "user", "content": "Q2"}], preserve_thinking=True)),
    ("tools", dict(messages=[{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Weather in Paris?"}], tools=[weather])),
    ("tools_no_system", dict(messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[weather])),
    ("tool_roundtrip", dict(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "metric": True}}}]},
        {"role": "tool", "content": "{\"temp\": 18}", "tool_call_id": "c1"},
    ], tools=[weather])),
    ("tool_then_answer", dict(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "sunny", "tool_call_id": "c1"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": "Thanks"},
    ], tools=[weather])),
    ("image_content", dict(messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}])),
    ("unicode", dict(messages=[{"role": "user", "content": "Zażółć gęślą jaźń \u2603 \U0001F600"}])),
]
out = []
for name, kw in cases:
    ctx = dict(bos_token="<|startoftext|>", eos_token="<|im_end|>", add_generation_prompt=True)
    ctx.update(kw)
    out.append(dict(name=name, context=ctx, expected=tpl.render(**ctx)))
json.dump(out, open(sys.argv[2], "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(out))
for o in out:
    print(o["name"], ascii(o["expected"])[:160])
