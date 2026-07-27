# scripts/lora/dump_bitnet_format.py
"""Task 1 (BitNet tool-use): extract the tool-aware BitNet chat-template rendering
for a fixed messages+tools fixture, so a C# test can assert dotLLM renders the same string.
Run: PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface C:/Python311/python.exe scripts/lora/dump_bitnet_format.py
"""
import json, os
from transformers import AutoTokenizer

BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"
TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "bitnet_tooluse.jinja")
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIX, exist_ok=True)

# Fixed inputs shared with the C# test (kept tiny + deterministic).
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
ASSISTANT_TOOL_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    template = open(TEMPLATE_FILE, encoding="utf-8").read()
    tok.chat_template = template

    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    with open(os.path.join(FIX, "bitnet_tooluse_reference.txt"), "w", encoding="utf-8") as f:
        f.write(rendered)

    # Flatten tools to dotLLM's ToolDefinition shape (name/description/parameters-as-json-string).
    dotllm_tools = [{
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "parameters": json.dumps(t["function"]["parameters"]),
    } for t in TOOLS]
    with open(os.path.join(FIX, "bitnet_format_inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"messages": MESSAGES, "tools": dotllm_tools,
                   "assistant_tool_call": ASSISTANT_TOOL_CALL,
                   "bos_token": tok.bos_token or "", "eos_token": tok.eos_token or ""},
                  f, indent=2, ensure_ascii=False)

    print("Wrote fixtures to", FIX)
    print("--- rendered (first 800 chars) ---")
    print(rendered[:800])
    assert "<tools>" in rendered, "ERROR: <tools> block missing from rendered output"
    assert "get_weather" in rendered, "ERROR: tool name missing from rendered output"
    assert rendered.rstrip().endswith("Assistant:"), f"ERROR: does not end at assistant turn, ends: {rendered[-50:]!r}"
    print("--- assertions passed ---")


if __name__ == "__main__":
    main()
