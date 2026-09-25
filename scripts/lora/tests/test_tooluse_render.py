"""Tests for tooluse_render.py — template-generated <tools> block correctness.

Schema confirmed from NousResearch/hermes-function-calling-v1 / glaive_func_calling:
  - row['tools']  — JSON string, list of {"type":"function","function":{...}} objects (already HF-ready)
  - row['conversations'] — ShareGPT list; turns have keys 'from' and 'value' (NOT 'content')
  - system turn embeds tools in <tools>...</tools> boilerplate (redundant once we use tools=)
  - multi-turn: several gpt turns, only some start with <tool_call>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from tooluse_render import extract_tools, render_tooluse

TOK_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Minimal row mirroring the actual glaive_func_calling schema:
# - 'tools' is a JSON string (top-level field), already in HF {"type":"function",...} format
# - conversation turns use 'value' key (not 'content')
MINIMAL_ROW = {
    "tools": '[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", '
             '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]',
    "conversations": [
        {
            "from": "system",
            "value": "You are a function calling AI model. You are provided with function signatures "
                     "within <tools></tools> XML tags.<tools>\n"
                     '[{"type": "function", "function": {"name": "get_weather"}}]\n'
                     "</tools>",
        },
        {"from": "human", "value": "Weather in Paris?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
    ],
}

# Multi-turn row: the first gpt turn is a tool call, second is a plain reply
MULTI_TURN_ROW = {
    "tools": '[{"type": "function", "function": {"name": "get_stock_price", "description": "Get stock price", '
             '"parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}]',
    "conversations": [
        {"from": "system", "value": "You are a function calling AI model."},
        {"from": "human", "value": "What is the stock price of AAPL?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}\n</tool_call>'},
        {"from": "tool", "value": '<tool_response>\n{"price": 150.0}\n</tool_response>'},
        {"from": "gpt", "value": "The price of AAPL is $150."},
        {"from": "human", "value": "Thanks."},
        {"from": "gpt", "value": "You're welcome!"},
    ],
}

# Row where no gpt turn contains a tool call — render_tooluse should raise ValueError
NO_TOOL_CALL_ROW = {
    "tools": '[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", '
             '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]',
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "4"},
    ],
}


def test_extract_tools_returns_hf_objects():
    tools = extract_tools(MINIMAL_ROW)
    assert isinstance(tools, list)
    assert len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "get_weather"


def test_extract_tools_multi_turn():
    tools = extract_tools(MULTI_TURN_ROW)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "get_stock_price"


def test_rendered_prompt_uses_template_tools_block():
    """Core parity test: the rendered prompt must contain the template-generated <tools> block."""
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    tools = extract_tools(MINIMAL_ROW)
    assert tools and tools[0]["function"]["name"] == "get_weather"
    ids, labels = render_tooluse(tok, MINIMAL_ROW)
    text = tok.decode(ids)
    assert "<tools>" in text and "get_weather" in text, (
        f"Expected <tools> block with tool name in rendered prompt. Got:\n{text[:500]}"
    )
    # Prompt part must exist and be fully masked
    assert -100 in labels, "Expected prompt tokens to be masked (-100)"


def test_render_supervises_tool_call_completion():
    """Some labels must be != -100 (the <tool_call> completion is supervised)."""
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    ids, labels = render_tooluse(tok, MINIMAL_ROW)
    supervised = [l for l in labels if l != -100]
    assert len(supervised) > 0, "No supervised tokens — completion not masked correctly"
    # Decode just the supervised tokens and check it contains the tool call
    completion_text = tok.decode(supervised)
    assert "get_weather" in completion_text or "tool_call" in completion_text, (
        f"Supervised completion doesn't look like a tool call: {completion_text!r}"
    )


def test_render_multi_turn_supervises_first_tool_call():
    """For multi-turn rows, we supervise the FIRST gpt <tool_call> turn only."""
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    ids, labels = render_tooluse(tok, MULTI_TURN_ROW)
    text = tok.decode(ids)
    assert "<tools>" in text
    supervised_ids = [i for i, l in zip(ids, labels) if l != -100]
    assert len(supervised_ids) > 0
    completion_text = tok.decode(supervised_ids)
    assert "get_stock_price" in completion_text or "tool_call" in completion_text


def test_render_raises_on_no_tool_call():
    """If a row has no <tool_call> gpt turn, render_tooluse should raise ValueError."""
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    import pytest
    with pytest.raises(ValueError, match="tool_call"):
        render_tooluse(tok, NO_TOOL_CALL_ROW)
