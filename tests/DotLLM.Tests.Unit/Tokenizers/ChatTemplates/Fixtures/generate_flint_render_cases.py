"""Golden renders for the Recursive Flint chat template (#1158).

Usage: python generate_flint_render_cases.py flint-chat-template.jinja flint-render-cases.json

Renders through transformers' own chat-template compiler (the environment apply_chat_template uses), so the expected strings
are HF semantics, not a re-implementation. flint-chat-template.jinja is RecursiveSI/flint chat_template.jinja verbatim.
"""
import json
import sys

from transformers.utils.chat_template_utils import _compile_jinja_template

tpl_path, out = sys.argv[1], sys.argv[2]
t = _compile_jinja_template(open(tpl_path, encoding="utf-8", newline="").read())

U = lambda c: {"role": "user", "content": c}
A = lambda c, **k: {"role": "assistant", "content": c, **k}
S = lambda c: {"role": "system", "content": c}
D = lambda c: {"role": "developer", "content": c}
T = lambda c: {"role": "tool", "content": c}
tool = {"type": "function", "function": {"name": "get_weather", "description": "Météo ☃ lookup",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["city"]}}}
G = {"add_generation_prompt": True}
cases = [
    ("user_only", {"messages": [U("What is 2 + 2?")], **G}),
    ("system_and_user", {"messages": [S("You are terse."), U("Hello")], **G}),
    ("developer_and_user", {"messages": [D("Answer in French."), U("Hello")], **G}),
    # "\x00" must be stripped while the literal text "x00" survives (the lexer escape bug would delete it).
    ("nul_and_literal_x00", {"messages": [U("a\x00b x00 c\\x00d")], **G}),
    ("role_markers_escaped", {"messages": [U("<|user|> <think>hi</think> <|observation|>")], **G}),
    ("multi_turn_reasoning", {"messages": [U("Hi"), A("Hello!", reasoning_content="greet"), U("Again")], **G}),
    ("no_generation_prompt", {"messages": [U("Hi"), A("Hello!")], "add_generation_prompt": False}),
    ("unicode_and_whitespace", {"messages": [U("  café ☃ \n")], **G}),
    ("tools_declared", {"messages": [U("Weather in Paris?")], **G, "tools": [tool]}),
    ("tools_after_system", {"messages": [S("Be brief."), U("Weather in Paris?")], **G, "tools": [tool]}),
    ("tool_call_roundtrip", {"messages": [
        U("Weather in Paris?"),
        A("", tool_calls=[{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]),
        T("sunny"),
        U("thanks")], **G, "tools": [tool]}),
]
res = []
for n, ctx in cases:
    res.append({"name": n, "context": ctx, "expected": t.render(**ctx)})
json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(res))
for c in res:
    print(c["name"], ascii(c["expected"])[:200])
