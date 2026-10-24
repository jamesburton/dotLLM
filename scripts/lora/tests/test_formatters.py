import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_formatters import format_row

def test_instruction_passthrough_messages():
    row = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}]}
    msgs = format_row("instruction", row)
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "Hello!"

def test_coding_builds_messages_from_alpaca():
    row = {"instruction": "Write a function that adds two numbers.",
           "input": "", "output": "def add(a, b):\n    return a + b"}
    msgs = format_row("coding", row)
    assert msgs[0]["role"] == "user"
    assert "adds two numbers" in msgs[0]["content"]
    assert msgs[-1]["role"] == "assistant"
    assert "def add" in msgs[-1]["content"]

def test_tooluse_maps_sharegpt_conversations():
    row = {"conversations": [
        {"from": "system", "content": "You are a function calling AI."},
        {"from": "human", "content": "Weather in Paris?"},
        {"from": "gpt", "content": '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>'}]}
    msgs = format_row("tooluse", row)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"].startswith("<tool_call>")
