"""
compare_logits_py_reference.py — PyTorch reference fixture generator for dotLLM.

PURPOSE
-------
This script is the "PyTorch oracle" half of dotLLM's bf16/F16 numerical
correctness suite (roadmap step P2.6). It loads a HuggingFace safetensors
checkpoint via `transformers.AutoModelForCausalLM`, runs a single forward
pass over a user-supplied prompt in bf16, and writes the resulting
logits to a JSON fixture. The companion C# test
`RealHfSafetensorsEndToEndTests.Qwen25_0_5B_LogitsMatchPyTorchReference`
loads the same prompt through dotLLM's own loader + forward path and
compares logits element-wise, asserting loose but meaningful
drift bounds (see the test's XML docs for calibration).

EXECUTION MODEL
---------------
This script is NOT invoked automatically by CI or by `dotnet test`. It's
run manually, once per model-version bump, to regenerate the JSON
fixture that is checked into the repo. The C# test reads the JSON; if
the fixture is missing the test skips cleanly with a message pointing
at this file.

Hardcoded package versions (in the install snippet below) exist because
the SDPA kernel in PyTorch / transformers has drifted across releases
and differences there would pollute the drift signal we're measuring.
Pinning makes the reference reproducible.

USAGE
-----
    # From the repository root (C:/Development/dotLLM-mamba3):
    python -m venv .venv-pyref
    .venv-pyref\\Scripts\\activate        # Windows
    # source .venv-pyref/bin/activate    # macOS / Linux

    pip install --index-url https://download.pytorch.org/whl/cpu \\
        torch==2.4.1
    pip install transformers==4.45.0 safetensors==0.4.5

    python tests/scripts/compare_logits_py_reference.py \\
        --model-path "C:/Users/james/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987" \\
        --prompt "The capital of France is" \\
        --output-path tests/DotLLM.Tests.Integration/Models/Loaders/references/qwen2.5-0.5b-reference.json

JSON SCHEMA
-----------
{
  "model_path": "...",
  "prompt": "...",
  "input_ids": [int, ...],
  "logits_shape": [seq_len, vocab_size],
  "logits": [[float, ...], ...],      # seq_len rows of vocab_size floats
  "dtype": "bfloat16",
  "python_version": "3.x.y",
  "torch_version": "...",
  "transformers_version": "..."
}
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a PyTorch reference logits JSON for dotLLM drift testing."
    )
    parser.add_argument("--model-path", required=True, help="Path to HF snapshot directory.")
    parser.add_argument("--prompt", required=True, help="Prompt string to forward.")
    parser.add_argument("--output-path", required=True, help="Destination JSON path.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16 — matches HF canonical storage).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of modeling code shipped in the snapshot "
             "(required for DeepSeek-V2/V3 and other checkpoints that "
             "declare an auto_map for custom classes).",
    )
    args = parser.parse_args()

    # Deferred imports so `--help` works without torch installed.
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import transformers  # type: ignore

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model path does not exist: {model_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=args.trust_remote_code)

    print(f"Loading model from {model_path} (dtype={args.dtype}, "
          f"trust_remote_code={args.trust_remote_code}) ...")
    # low_cpu_mem_usage requires `accelerate`; we keep the dep surface small
    # and take the higher peak-RSS hit — 0.5 B params in bf16 is ~1 GiB and
    # fits comfortably in RAM on any dev box that can run this script.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    print(f"Encoding prompt: {args.prompt!r}")
    # add_special_tokens=False to match dotLLM's raw-prompt tokenisation
    # path — the existing Qwen25_0_5B_GeneratesText_FromTokenizedPrompt
    # test feeds the raw `tok.Encode(prompt)` ids into Forward without BOS.
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    print(f"Input ids ({input_ids.shape[-1]}): {input_ids[0].tolist()}")

    print("Running forward pass ...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
    # logits: [batch, seq_len, vocab]
    logits = outputs.logits[0]  # strip batch dim -> [seq_len, vocab]
    # Upcast to float32 for JSON serialisation (bf16 isn't JSON-friendly and
    # would silently truncate anyway; we want the model's bf16 output values,
    # just transported as f32).
    logits_f32 = logits.to(torch.float32).cpu()
    seq_len, vocab_size = logits_f32.shape
    print(f"Logits shape: [{seq_len}, {vocab_size}]")
    print(
        f"Logits stats: min={logits_f32.min().item():.4f} "
        f"max={logits_f32.max().item():.4f} "
        f"mean={logits_f32.mean().item():.4f} "
        f"std={logits_f32.std().item():.4f}"
    )

    payload = {
        "model_path": str(model_path).replace("\\", "/"),
        "prompt": args.prompt,
        "input_ids": input_ids[0].tolist(),
        "logits_shape": [int(seq_len), int(vocab_size)],
        "logits": logits_f32.tolist(),
        "dtype": args.dtype,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    print(f"Writing {output_path} ...")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {size_mb:.1f} MiB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
