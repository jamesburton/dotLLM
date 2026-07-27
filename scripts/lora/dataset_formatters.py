"""Map raw dataset rows to chat messages (role/content), ending on the assistant
turn. The trainer then renders these via tokenizer.apply_chat_template — identical
to dotLLM's serving render (see FORMAT.md)."""

_SHAREGPT_ROLE = {"system": "system", "human": "user", "user": "user",
                  "gpt": "assistant", "assistant": "assistant", "tool": "tool"}

def _sharegpt_text(turn: dict) -> str:
    return turn.get("content", turn.get("value", ""))

def format_row(task: str, row: dict) -> list[dict]:
    if task == "instruction":
        msgs = row["messages"]
        return [{"role": m["role"], "content": m["content"]} for m in msgs]
    if task == "coding":
        instr = row["instruction"]
        if row.get("input"):
            instr = f"{instr}\n\n{row['input']}"
        return [{"role": "user", "content": instr},
                {"role": "assistant", "content": row["output"]}]
    if task == "tooluse":
        out = []
        for turn in row["conversations"]:
            role = _SHAREGPT_ROLE[turn["from"]]
            out.append({"role": role, "content": _sharegpt_text(turn)})
        return out
    raise ValueError(f"unknown task: {task}")
