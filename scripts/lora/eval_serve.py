"""dotLLM serving shim for the task-LoRA eval harness (U2 Phase C, Task 7).

Shells out to the dotLLM `run` CLI in `--json` mode and returns the generated text.
Prompts are pre-rendered with `tokenizer.apply_chat_template` (the U0-verified
train==serve form) and passed RAW: `run` only applies its own chat template when
`--tools` is given, so a pre-rendered prompt with no `--tools` is used verbatim.

Build the Release CLI once before looping:
    dotnet build src/DotLLM.Cli -c Release
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Repo root = three levels up from this file (scripts/lora/eval_serve.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PROJECT = REPO_ROOT / "src" / "DotLLM.Cli"


def generate(
    model_gguf: str,
    prompt: str,
    lora: list[str] | str | None = None,
    device: str = "gpu",
    max_tokens: int = 128,
    temp: float = 0.0,
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
    seed: int | None = None,
    tools_file: str | None = None,
    tool_choice: str | None = None,
    timeout: int = 600,
) -> dict:
    """Run one generation and return the parsed `RunJsonResult` dict.

    `lora` may be a single adapter dir, a list of dirs (stacked), or None (base).
    When `tools_file` is given, `--tools @<file>` is passed and `run` renders the
    prompt via the model's chat template with those tools (so `prompt` should be the
    raw user query, not a pre-rendered string); `tool_choice` (e.g. "required")
    additionally constrains decoding to a valid tool call. Parsed calls are then in
    the returned dict under `toolCalls`.
    Returns the full JSON object; callers usually want `result["text"]` / `result["toolCalls"]`.
    """
    loras = [] if lora is None else ([lora] if isinstance(lora, str) else list(lora))

    cmd = [
        "dotnet", "run", "--project", str(CLI_PROJECT), "-c", "Release",
        "--",
        "run", model_gguf,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--device", device,
        "--json",
    ]
    for d in loras:
        cmd += ["--lora", d]
    if tools_file is not None:
        cmd += ["--tools", f"@{tools_file}"]
    if tool_choice is not None:
        cmd += ["--tool-choice", tool_choice]
    if repeat_penalty is not None:
        cmd += ["--repeat-penalty", str(repeat_penalty)]
    if repeat_last_n is not None:
        cmd += ["--repeat-last-n", str(repeat_last_n)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dotLLM run failed (exit {proc.returncode}).\n"
            f"CMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr[-2000:]}\nSTDOUT:\n{proc.stdout[-2000:]}"
        )

    # `--json` prints a single JSON object. `dotnet run` may emit build chatter on the
    # first invocation; isolate the JSON object (first '{' to last '}').
    out = proc.stdout
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"No JSON object in CLI output:\n{out[-2000:]}")
    return json.loads(out[start:end + 1])


def generate_text(model_gguf: str, prompt: str, **kw) -> str:
    return generate(model_gguf, prompt, **kw)["text"]


if __name__ == "__main__":
    # Smoke test: base generation returns non-empty text. Usage: eval_serve.py <gguf> [lora_dir]
    gguf = sys.argv[1]
    lora = sys.argv[2] if len(sys.argv) > 2 else None
    res = generate(gguf, "Write one short sentence about Tokyo.", lora=lora, max_tokens=40)
    print("FINISH:", res.get("finishReason"))
    print("TIMINGS_ms:", res.get("timings"))
    print("TEXT:", repr(res.get("text"))[:500])
    assert res.get("text"), "expected non-empty generated text"
    print("OK")
