# Cloud CUDA Parity Runbook — Qwen3MoeHybrid

End-to-end CUDA forward parity for `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` (29.6 GB).
Local 12 GB cards cannot fit the full model — this runbook is the final gate
before any release that touches `CudaQwen3MoeHybridTransformerModel`.

## Prerequisites

- **GPU**: A100 80 GB or H100 80 GB (or any single card with ≥ 40 GB free).
  An RTX 4090 24 GB will *not* fit at Q6_K_XL — the model needs ~30 GB of
  Q-format weight residence plus headroom for KV cache, activations, and a
  modest cuBLAS dequant scratch on prefill.
- **CUDA toolkit 12.x** with `nvcc` for PTX rebuild.
- **.NET 10 SDK preview** (whatever the repo currently builds against — check
  `global.json` if present, otherwise the latest `10.0.x` preview).
- **HuggingFace login** if the GGUF needs to be downloaded again
  (`huggingface-cli login`).

## Provisioning

Vast.ai / RunPod / Lambda — any of them works. A 1-hour A100 80 GB instance
is more than enough; budget ~$2 with current pricing.

Recommended image: `nvidia/cuda:12.6.0-devel-ubuntu24.04` plus
`.NET 10 SDK preview` installed via the canonical `dotnet-install.sh`.

## Setup

```bash
git clone https://github.com/jamesburton/dotLLM.git
cd dotLLM
git checkout feature/qwen3.6   # or main once merged

# Build native PTX (preserves -fmad=false — required for bit-perfect parity).
cd native && ./build_ptx.sh && cd ..

# Build managed code.
dotnet build -c Release

# Download the GGUF (29.6 GB). Skip if already cached.
huggingface-cli download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
    --local-dir ~/.dotllm/models

export DOTLLM_QWEN3MOEHYBRID_GGUF_PATH=$HOME/.dotllm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
```

## Step 1 — Capture CPU activation dumps (one-time)

```bash
mkdir -p /tmp/qwen35_dumps
export DOTLLM_TENSOR_DUMP_DIR=/tmp/qwen35_dumps

# Greedy decode of a short canonical prompt; captures per-layer activations
# via the env-controlled TensorDump infrastructure.
dotnet run --project samples/DotLLM.Sample.Console -c Release -- \
    "$DOTLLM_QWEN3MOEHYBRID_GGUF_PATH" \
    "The capital of France is" \
    --greedy --max 1

unset DOTLLM_TENSOR_DUMP_DIR
```

This produces ~700 `.bin` files in `/tmp/qwen35_dumps/`, each holding a
`[shape] f32` activation at every per-layer hook point in
`Qwen3MoeHybridTransformerModel.ForwardFullAttnBody` and `ForwardGdnBody`.

Time: 3–5 min on a 64-core box (CPU-only forward — GPU not involved).

## Step 2 — Run the layer-by-layer CUDA parity test

```bash
export DOTLLM_QWEN3MOEHYBRID_CPU_DUMP_DIR=/tmp/qwen35_dumps

dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release \
    --no-build \
    --filter "FullyQualifiedName~CudaQwen3MoeHybridRealGgufLayerParityTests" \
    --logger "console;verbosity=normal"
```

Expected: 4 tests pass. Reports per-layer `rms` and worst-case `|diff|` for
each of the 30 GDN layers + 10 full-attn layers; the gate is RMS-based
(see `RmsTol` in the test file for the current bound and rationale).

If a layer reports `[OUTLIER]` (per-element `|diff|` > 1e-1) but RMS stays
within bounds, that's an activation outlier in the model itself, not a
kernel bug. If RMS breaches, drill into the kernel chain at that layer —
the test failure message points at the worst-case index for the
investigation.

## Step 3 — End-to-end greedy decode parity vs llama.cpp

```bash
# Install llama.cpp with CUDA support if not already present.
# (Use llama.cpp release binaries or build from source with GGML_CUDA=1.)

# dotLLM greedy decode.
dotnet run --project samples/DotLLM.Sample.Console -c Release -- \
    "$DOTLLM_QWEN3MOEHYBRID_GGUF_PATH" \
    "The capital of France is" \
    --greedy --max 50 \
    | tee /tmp/dotllm_decode.txt

# llama.cpp greedy decode on the same model + prompt.
llama-cli \
    --model "$DOTLLM_QWEN3MOEHYBRID_GGUF_PATH" \
    --prompt "The capital of France is" \
    --n-predict 50 \
    --temp 0 \
    --top-k 1 \
    --no-conversation \
    --gpu-layers 99 \
    | tee /tmp/llama_decode.txt

diff /tmp/dotllm_decode.txt /tmp/llama_decode.txt
```

Expected: token sequences match for at least the first 20 tokens (the
canonical Phase 10 success criterion). Divergence after 20 tokens is
acceptable — accumulated FP rounding across long sequences eventually
flips a top-1 even between two correct implementations, particularly
near low-entropy positions.

## Step 4 — Decode throughput vs llama.cpp

```bash
# dotLLM
dotnet run --project samples/DotLLM.Sample.Console -c Release -- \
    "$DOTLLM_QWEN3MOEHYBRID_GGUF_PATH" \
    "The capital of France is" \
    --greedy --max 50

# Look at the printed [Prefill: ... ms (... tok/s), Decode: ... ms (... tok/s)] line.

# llama.cpp
llama-bench \
    --model "$DOTLLM_QWEN3MOEHYBRID_GGUF_PATH" \
    --n-gen 50 \
    --n-prompt 32 \
    --threads 1 \
    --gpu-layers 99
```

Phase 10 target: dotLLM decode within 2× of llama.cpp on the same model
and GPU. If the gap is wider than 2×, the next perf pass tackles the
hottest kernel surfaced by `nvprof` / `nsys`.

## Step 5 — Report back

Open a short PR comment / issue with:
- GPU model + driver version.
- Step 2 result (per-layer RMS pass/fail).
- Step 3 result (token-for-token match for first 20 tokens?).
- Step 4 numbers: dotLLM prefill tok/s, decode tok/s; llama.cpp same.
- Any `[OUTLIER]` layers from Step 2 (informational, not a fail).

That's the gate. Without these numbers we cannot claim Phase 10 done at
the success criteria level.
