"""Task 1 (BitNet tool-use): verify the tool-aware BitNet chat template renders correctly.

Run: PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface C:/Python311/python.exe -m pytest scripts/lora/tests/test_bitnet_tooluse_render.py -v
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")
TEMPLATE_FILE = os.path.join(TEMPLATES_DIR, "bitnet_tooluse.jinja")

BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"

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


def _load_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open(TEMPLATE_FILE, encoding="utf-8").read()
    return tok


def test_template_file_exists():
    assert os.path.isfile(TEMPLATE_FILE), f"Template file not found: {TEMPLATE_FILE}"


def test_rendered_contains_tools_block():
    tok = _load_tok()
    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    assert "<tools>" in rendered, f"Expected <tools> block. Got:\n{rendered[:600]}"
    assert "</tools>" in rendered, "Missing </tools>"


def test_rendered_contains_tool_name():
    tok = _load_tok()
    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    assert "get_weather" in rendered, f"Tool name 'get_weather' missing. Got:\n{rendered[:600]}"


def test_rendered_ends_at_assistant_turn():
    tok = _load_tok()
    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    assert rendered.rstrip().endswith("Assistant:"), (
        f"Expected prompt to end at 'Assistant:'. Ends with: {rendered[-80:]!r}"
    )


def test_rendered_contains_tool_call_instructions():
    tok = _load_tok()
    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    assert "<tool_call>" in rendered, "Expected <tool_call> instruction in tools preamble"
    assert "</tool_call>" in rendered, "Expected </tool_call> instruction in tools preamble"


def test_rendered_contains_eot_id_separator():
    tok = _load_tok()
    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    assert "<|eot_id|>" in rendered, "Expected BitNet turn separator <|eot_id|>"


def test_rendered_matches_reference_fixture():
    """Parity: rendered output must match the pre-generated reference fixture."""
    ref_path = os.path.join(FIXTURES_DIR, "bitnet_tooluse_reference.txt")
    assert os.path.isfile(ref_path), (
        f"Reference fixture not found at {ref_path}. "
        "Run 'python scripts/lora/dump_bitnet_format.py' first."
    )
    reference = open(ref_path, encoding="utf-8").read()
    tok = _load_tok()
    rendered = tok.apply_chat_template(
        MESSAGES, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )

    def norm(s): return s.replace("\r\n", "\n").rstrip()

    assert norm(rendered) == norm(reference), (
        f"Rendered output differs from reference fixture.\n"
        f"Expected:\n{reference[:600]}\n\nActual:\n{rendered[:600]}"
    )


def test_hermes_parser_roundtrips_tool_call():
    """Python-side Hermes parse: extract name+arguments from <tool_call> block."""
    # Simple inline parser matching HermesToolCallParser logic
    open_tag = "<tool_call>"
    close_tag = "</tool_call>"
    text = ASSISTANT_TOOL_CALL
    start = text.index(open_tag) + len(open_tag)
    end = text.index(close_tag, start)
    json_blob = text[start:end].strip()
    parsed = json.loads(json_blob)
    assert parsed["name"] == "get_weather"
    assert parsed["arguments"]["city"] == "Tokyo"
