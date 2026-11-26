"""Golden renders for the Thinking Machines Inkling 2 chat template (#2026).

Usage: python generate_inkling2_render_cases.py inkling2-chat-template.jinja inkling2-render-cases.json

Renders through transformers' own chat-template compiler (the environment apply_chat_template uses, including the `{% generation %}`
extension), so the expected strings are HF semantics, not a re-implementation. inkling2-chat-template.jinja is
thinkingmachines/Inkling-2-Mini chat_template.jinja verbatim (byte-identical to the template embedded in
ggml-org/Inkling-2-Mini-GGUF `tokenizer.chat_template`). Template errors are recorded as `expectedError`.
"""
import json
import sys

from transformers.utils.chat_template_utils import _compile_jinja_template

tpl_path, out = sys.argv[1], sys.argv[2]
t = _compile_jinja_template(open(tpl_path, encoding="utf-8", newline="").read())

U = lambda c: {"role": "user", "content": c}
A = lambda c, **k: {"role": "assistant", "content": c, **k}
S = lambda c: {"role": "system", "content": c}
T = lambda c, **k: {"role": "tool", "content": c, **k}
tool = {"type": "function", "function": {"name": "get_weather", "description": "Météo ☃ lookup",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["city"]}}}
cases = [
    ("user_only_default_effort", {"messages": [U("What is 2 + 2?")]}),
    ("system_then_user", {"messages": [S("You are terse."), U("Hello")]}),
    ("effort_low", {"messages": [U("Hi")], "reasoning_effort": "low"}),
    ("effort_none", {"messages": [U("Hi")], "reasoning_effort": "none"}),
    ("effort_max", {"messages": [U("Hi")], "reasoning_effort": "max"}),
    ("effort_numeric", {"messages": [U("Hi")], "reasoning_effort": 0.5}),
    ("effort_numeric_zero", {"messages": [U("Hi")], "reasoning_effort": 0.0}),
    ("effort_string_padded", {"messages": [U("Hi")], "reasoning_effort": " medium "}),
    ("effort_only_system_messages", {"messages": [S("Be brief.")], "reasoning_effort": "high"}),
    ("effort_unknown", {"messages": [U("Hi")], "reasoning_effort": "wild"}),
    ("effort_out_of_range", {"messages": [U("Hi")], "reasoning_effort": 1.5}),
    ("multi_turn_reasoning", {"messages": [U("Hi"), A("Hello!", reasoning_content="greet"), U("Again")]}),
    ("assistant_named", {"messages": [U("Hi"), A("Hello!", name="Ada"), U("Again")]}),
    ("tools_declared", {"messages": [U("Weather in Paris?")], "tools": [tool]}),
    ("tool_call_roundtrip", {"messages": [
        U("Weather in Paris?"),
        A("", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"days": 3, "city": "Paris"}}}]),
        T("sunny", tool_call_id="c1"),
        U("thanks")], "tools": [tool]}),
    ("tool_result_named", {"messages": [U("x"), T("42", name="calc")]}),
    ("content_parts_and_image", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Look:"}, {"type": "image_url", "image_url": {"url": "x"}}, "tail"]}]}),
    ("audio_refused", {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]}),
    ("unknown_role", {"messages": [{"role": "developer", "content": "x"}]}),
    ("unicode_and_whitespace", {"messages": [U("  café ☃ \n")]}),
]
res = []
for n, ctx in cases:
    try:
        res.append({"name": n, "context": ctx, "expected": t.render(**ctx)})
    except Exception as e:  # template-raised errors are part of the contract
        res.append({"name": n, "context": ctx, "expectedError": str(e)})
json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(res))
for c in res:
    print(c["name"], ascii(c.get("expected", c.get("expectedError")))[:220])
