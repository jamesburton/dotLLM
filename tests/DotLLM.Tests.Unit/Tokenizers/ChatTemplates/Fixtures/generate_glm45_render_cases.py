"""Golden renders for the GLM-4.5-Air chat template (#1604).

Usage: python -I generate_glm45_render_cases.py glm-4.5-air-chat-template.jinja glm-4.5-air-render-cases.json

Renders through transformers' own chat-template compiler (the environment apply_chat_template uses), so the expected strings are
HF semantics, not a re-implementation. glm-4.5-air-chat-template.jinja is zai-org/GLM-4.5-Air chat_template.jinja verbatim
(snapshot a24ceef6ce4f3536971efe9b778bdaa1bab18daa).
"""
import json
import sys

from transformers.utils.chat_template_utils import _compile_jinja_template

tpl_path, out = sys.argv[1], sys.argv[2]
t = _compile_jinja_template(open(tpl_path, encoding="utf-8", newline="").read())

U = lambda c: {"role": "user", "content": c}
A = lambda c, **k: {"role": "assistant", "content": c, **k}
S = lambda c: {"role": "system", "content": c}
T = lambda c: {"role": "tool", "content": c}
TC = lambda n, a: {"type": "function", "function": {"name": n, "arguments": a}}
weather = {"type": "function", "function": {"name": "get_weather", "description": "Météo ☃ lookup",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city"]}}}
calc = {"type": "function", "function": {"name": "add", "description": "Add numbers",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}
G = {"add_generation_prompt": True}
cases = [
    ("user_only", {"messages": [U("What is 2 + 2?")], **G}),
    ("system_and_user", {"messages": [S("You are terse."), U("Hello")], **G}),
    ("multi_turn", {"messages": [U("Hi"), A("Hello!"), U("Again"), A("Welcome back."), U("Bye")], **G}),
    ("no_generation_prompt", {"messages": [U("Hi"), A("Hello!")], "add_generation_prompt": False}),
    ("enable_thinking_false", {"messages": [U("What is 2 + 2?")], **G, "enable_thinking": False}),
    ("enable_thinking_true", {"messages": [U("What is 2 + 2?")], **G, "enable_thinking": True}),
    ("nothink_suffix_already_present", {"messages": [U("Quick one /nothink")], **G, "enable_thinking": False}),
    ("enable_thinking_false_multi_turn", {"messages": [S("Be brief."), U("Hi"), A("Hello!"), U("Again")], **G, "enable_thinking": False}),
    ("reasoning_content_in_history", {"messages": [U("Hi"), A("Hello!", reasoning_content="greet the user"), U("Again")], **G}),
    ("reasoning_content_after_last_user", {"messages": [U("Hi"), A("Hello!", reasoning_content="greet the user")], "add_generation_prompt": False}),
    ("inline_think_in_content", {"messages": [U("Hi"), A("<think>\nthinking it over\n</think>\n\nHello!"), U("Again")], **G}),
    ("tools_declared", {"messages": [U("Weather in Paris?")], **G, "tools": [weather]}),
    ("tools_two_after_system", {"messages": [S("Be brief."), U("Weather in Paris, then 1+2?")], **G, "tools": [weather, calc]}),
    ("tool_call_roundtrip", {"messages": [
        U("Weather in Paris?"),
        A("", tool_calls=[TC("get_weather", {"city": "Paris", "days": 3})]),
        T("sunny"),
        U("thanks")], **G, "tools": [weather]}),
    ("parallel_tool_calls_and_results", {"messages": [
        U("Weather in Paris and Rome?"),
        A("Checking both.", reasoning_content="two lookups", tool_calls=[TC("get_weather", {"city": "Paris"}), TC("get_weather", {"city": "Rome"})]),
        T("sunny"), T("rainy")], **G, "tools": [weather]}),
    ("tool_result_then_answer_thinking_off", {"messages": [
        U("What is 1+2?"),
        A("", tool_calls=[TC("add", {"a": 1, "b": 2})]),
        T("3")], **G, "tools": [calc], "enable_thinking": False}),
    ("string_and_json_argument_values", {"messages": [
        U("go"),
        A("", tool_calls=[TC("note", {"text": "plain string", "tags": ["a", "é"], "meta": {"k": 1.5, "ok": True}, "none": None})]),
        T("done")], **G}),
    ("content_parts_list", {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}], **G}),
]
res = []
for n, ctx in cases:
    res.append({"name": n, "context": ctx, "expected": t.render(**ctx)})
json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(res))
for c in res:
    print(c["name"], ascii(c["expected"])[:300])
