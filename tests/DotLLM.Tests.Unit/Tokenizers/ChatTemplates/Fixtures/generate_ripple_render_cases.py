"""Golden renders for the ricursiveAI Ripple-65B-Preview chat template (#1643).

Usage: python generate_ripple_render_cases.py ripple-65b-preview-chat-template.jinja ripple-render-cases.json

Renders through transformers' own chat-template compiler (the environment apply_chat_template uses), so the expected strings
are HF semantics, not a re-implementation. ripple-65b-preview-chat-template.jinja is ricursiveAI/Ripple-65B-Preview
chat_template.jinja verbatim (also embedded byte-identically as tokenizer.chat_template in the AtomicChat and prometheusAIR GGUFs).
Note: tokenizer_config.json carries a bare ChatML stub; the .jinja file is the one transformers and both GGUFs use.
"""
import json
import sys

from transformers.utils.chat_template_utils import _compile_jinja_template

tpl_path, out = sys.argv[1], sys.argv[2]
t = _compile_jinja_template(open(tpl_path, encoding="utf-8", newline="").read())

U = lambda c: {"role": "user", "content": c}
A = lambda c, **k: {"role": "assistant", "content": c, **k}
S = lambda c: {"role": "system", "content": c}
T = lambda c, i: {"role": "tool", "content": c, "tool_call_id": i}
TC = lambda i, n, a: {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
bash = {"type": "function", "function": {"name": "execute_bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command."}},
                       "required": ["command"]}}}
read = {"type": "function", "function": {"name": "read_file", "description": "Read a file (UTF-8, ünïcode ☃ ok).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                       "required": ["path"]}}}
G = {"add_generation_prompt": True}
cases = [
    ("plain", {"messages": [U("Hello!")], **G}),
    ("system", {"messages": [S("You are terse."), U("Hi")], **G}),
    ("no_generation_prompt", {"messages": [U("Hi"), A("Hello.")], "add_generation_prompt": False}),
    ("multi_turn_reasoning", {"messages": [U("2+2?"), A("4", reasoning_content="Simple sum."), U("And 3+3?")], **G}),
    ("multi_turn_reasoning_clear_thinking", {"messages": [U("2+2?"), A("4", reasoning_content="Simple sum."), U("And 3+3?")],
                                             **G, "clear_thinking": True}),
    # content that already opens <think> is not re-wrapped
    ("reasoning_already_inline", {"messages": [U("q"), A("<think>\nr\n</think>\nanswer", reasoning_content="r"), U("again")], **G}),
    ("tools_with_system", {"messages": [S("You are an agent."), U("Where am I?")], **G, "tools": [bash, read]}),
    ("tools_without_system", {"messages": [U("List files")], **G, "tools": [bash, read]}),
    ("parallel_tool_calls_and_results", {"messages": [
        U("pwd and read a.txt"),
        A("", reasoning_content="Two calls.", tool_calls=[TC("call_0", "execute_bash", {"command": "pwd"}),
                                                           TC("call_1", "read_file", {"path": "a.txt", "limit": 10})]),
        T("/home/u", "call_0"),
        T("hello", "call_1")], **G, "tools": [bash, read]}),
    ("content_plus_call_then_answer", {"messages": [
        U("go"),
        A("Running it.", tool_calls=[TC("call_0", "execute_bash", {"command": "ls -la"})]),
        T("total 0", "call_0"),
        A("Empty dir.")], "add_generation_prompt": False, "tools": [bash]}),
    ("system_mid_conversation", {"messages": [S("first"), U("a"), A("b"), S("second"), U("c")], **G}),
]
res = []
for n, ctx in cases:
    res.append({"name": n, "context": ctx, "expected": t.render(**ctx)})
json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(res))
for c in res:
    print(c["name"], ascii(c["expected"])[:160])
