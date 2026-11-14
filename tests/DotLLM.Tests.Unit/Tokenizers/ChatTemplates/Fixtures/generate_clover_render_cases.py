"""Golden renders for the DiffuSpace Clover-35B-A3B-Preview chat template (#1794).

Usage: python generate_clover_render_cases.py clover-35b-a3b-chat-template.jinja clover-render-cases.json

Renders through transformers' own chat-template compiler (the environment apply_chat_template uses), so the expected strings
are HF semantics, not a re-implementation. clover-35b-a3b-chat-template.jinja is Diffuspace/Clover-35B-A3B-Preview
chat_template.jinja verbatim (Apache-2.0; byte-identical to tokenizer.chat_template in the official Q4_K_M GGUF). It is a
text-only rewrite of Qwen3.6's template with three switches: enable_thinking, preserve_thinking (default true) and
add_non_thinking_prefix (default true).
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
    ("enable_thinking_false", {"messages": [U("Hi")], **G, "enable_thinking": False}),
    ("multi_turn_no_preserve_thinking", {"messages": [U("2+2?"), A("4", reasoning_content="Simple sum."), U("And 3+3?")],
                                         **G, "preserve_thinking": False}),
    ("assistant_without_reasoning", {"messages": [U("a"), A("b"), U("c")], **G}),
    ("assistant_without_reasoning_no_prefix", {"messages": [U("a"), A("b"), U("c")], **G, "add_non_thinking_prefix": False}),
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
    # Clover's template refuses a mid-conversation system turn (Qwen3.6's accepts it).
    ("system_mid_conversation", {"messages": [S("first"), U("a"), A("b"), S("second"), U("c")], **G}),
]
res = []
for n, ctx in cases:
    try:
        res.append({"name": n, "context": ctx, "expected": t.render(**ctx)})
    except Exception as e:  # jinja2 TemplateError from raise_exception
        res.append({"name": n, "context": ctx, "expectedError": str(e)})
json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(res))
for c in res:
    print(c["name"], ascii(c.get("expected", "ERROR: " + c.get("expectedError", "")))[:160])
