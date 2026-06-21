# scripts/lora/dump_qwen_format.py
"""U0: extract Qwen3's authoritative chat-template rendering for a fixed
messages+tools fixture, so a C# test can assert dotLLM renders the same string.
Run: PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface C:/Python311/python.exe scripts/lora/dump_qwen_format.py
"""
import json, os
from transformers import AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIX, exist_ok=True)

# Fixed inputs shared with the C# test (kept tiny + deterministic).
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
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
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)

def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    template = tok.chat_template
    with open(os.path.join(FIX, "qwen3_chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(template)

    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    with open(os.path.join(FIX, "qwen3_tooluse_reference.txt"), "w", encoding="utf-8") as f:
        f.write(rendered)

    # Flatten tools to dotLLM's ToolDefinition shape (name/description/parameters-as-json-string).
    dotllm_tools = [{
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "parameters": json.dumps(t["function"]["parameters"]),
    } for t in TOOLS]
    with open(os.path.join(FIX, "format_inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"messages": MESSAGES, "tools": dotllm_tools,
                   "assistant_tool_call": ASSISTANT_TOOL_CALL,
                   "bos_token": tok.bos_token or "", "eos_token": tok.eos_token or ""},
                  f, indent=2, ensure_ascii=False)

    print("Wrote fixtures to", FIX)
    print("--- rendered (first 600 chars) ---")
    print(rendered[:600])

if __name__ == "__main__":
    main()
