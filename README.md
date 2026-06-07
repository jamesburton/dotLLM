<div align="center">

# dotLLM

**High-performance LLM inference engine written natively in C#/.NET**

[![CI](https://github.com/kkokosa/dotLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/kkokosa/dotLLM/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![.NET](https://img.shields.io/badge/.NET-10-purple.svg)](https://dotnet.microsoft.com/)
[![Website](https://img.shields.io/badge/website-dotllm.dev-0a7aca.svg)](https://dotllm.dev/)

[Website](https://dotllm.dev/) · [Documentation](docs/) · [Roadmap](docs/ROADMAP.md) · [Discussions](https://github.com/kkokosa/dotLLM/discussions)

</div>

---

## About

dotLLM is a ground-up LLM inference engine for .NET — not a wrapper around llama.cpp or Python libraries. All orchestration, model loading, tokenization, sampling, and CPU compute are implemented in pure C#, with CUDA GPU acceleration via PTX kernels loaded through the CUDA Driver API (no native shared library). It targets transformer-based models (Llama, Mistral, Phi, Qwen, DeepSeek) with SIMD-optimized CPU and CUDA GPU backends.

> **Status**: Phase 6 complete — speculative decoding, paged KV-cache, Native AOT (experimental), and startup warm-up on top of the OpenAI-compatible API server, built-in chat UI, constrained decoding (JSON/schema/regex/grammar), tool calling, and prompt caching. CUDA GPU backend with CPU/GPU hybrid offloading and KV-cache quantization. SIMD-optimized CPU inference with Q4_K_M, chat templates, streaming, multi-threading, NUMA pinning. Supports Llama, Mistral, Phi, Qwen. Phase 7 (diagnostics & interpretability) in progress — logprobs landed. See [Roadmap](#roadmap).

## Key Features

### Performance
- **Zero-GC inference** — unmanaged memory (`NativeMemory.AlignedAlloc`, 64-byte aligned) for all tensor data; no managed heap allocations on the hot path
- **SIMD vectorization** — `TensorPrimitives` + hand-tuned `System.Runtime.Intrinsics` for quantized matmul, RMSNorm, RoPE, softmax
- **Memory-mapped model loading** — GGUF files loaded via `MemoryMappedFile`; OS demand-paging means multi-GB models load in milliseconds
- **Quantized inference** — FP16, Q8_0, Q4_K_M and other GGUF quantization formats; fused scale×int dot-product kernels operating directly on quantized blocks

### Architecture Support
- **Transformer models** — Llama, Mistral, Phi, Qwen, DeepSeek via parameterized `TransformerBlock` and `ModelConfig`
- **Attention mechanisms** — MHA, MQA, GQA via parameterized `ModelConfig`, with `IAttentionStrategy` for kernel selection
- **Position encoding** — RoPE, ALiBi, absolute, none — pluggable via `IPositionEncoding`
- **Composable sampling** — `ISamplerStep` chain: repetition penalty → temperature → top-k → top-p → min-p → categorical sample

### Serving
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, tool calling, streaming via ASP.NET
- **Paged KV-cache** — PagedAttention with block-level allocation, prefix caching, and copy-on-write
- **Speculative decoding** — draft-verify-accept with KV-cache rollback (greedy mode today; non-greedy planned — see issue #121)
- **Structured output** — FSM/PDA-based constrained decoding guaranteeing valid JSON, JSON Schema, regex, and grammar
- **(Planned) Continuous batching** — iteration-level scheduling with preemption and priority queuing — Phase 9, see [Roadmap](#roadmap)

### Extensibility
- **Pluggable backends** — `IBackend` interface with separate packages per backend (CPU, CUDA, ROCm)
- **Diagnostic hooks** — zero-cost `IInferenceHook` points for activation capture, logit lens, SAE integration
- **(Planned) LoRA adapters** — runtime loading, no weight merging, concurrent multi-adapter serving — Phase 7, see [Roadmap](#roadmap)
- **(Planned) OpenTelemetry observability** — `System.Diagnostics.Metrics` + `Activity` for throughput, latency, and per-request tracing — Phase 7 / Phase 9, see [Roadmap](#roadmap)

## Architecture Overview

dotLLM is organized as a layered architecture where each layer depends only on the layers below it:

```
┌─────────────────────────────────────────┐
│            DotLLM.Server                │  ASP.NET OpenAI-compatible API
├─────────────────────────────────────────┤
│            DotLLM.Engine                │  KV-cache, scheduler, samplers,
│                                         │  constraints, speculative decoding
├──────────┬──────────┬───────────────────┤
│ DotLLM.  │ DotLLM.  │ DotLLM.Cpu/Cuda   │  GGUF/SafeTensors, BPE/SPM,
│ Models   │Tokenizers│ (backends)        │  SIMD kernels / CUDA kernels
├──────────┴──────────┴───────────────────┤
│            DotLLM.Core                  │  Interfaces, tensor types, config
└─────────────────────────────────────────┘
```

Each project ships as a separate NuGet package, so users pull in only what they need. `DotLLM.Core` defines all abstractions (`ITensor`, `IBackend`, `IModel`, `ISamplerStep`, etc.) while concrete implementations live in their respective projects.

## Getting Started

There are two paths: grab a pre-built release and run it, or clone the repo and build from source.

### Use a pre-built release

Pick one of three install options.

**Option A — install as a global .NET tool** (requires .NET 10 runtime):

```bash
dotnet tool install -g DotLLM.Cli --prerelease

# Download a model once, then use it anywhere
dotllm model pull QuantFactory/SmolLM-135M-GGUF

dotllm run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 64
dotllm serve QuantFactory/SmolLM-135M-GGUF               # OpenAI-compatible API + chat UI
```

**Option B — download a self-contained binary** (no .NET install needed — the runtime is bundled):

Grab the archive for your platform from the [latest release](https://github.com/kkokosa/dotLLM/releases/latest):

- Windows x64: `dotllm-<version>-win-x64.zip`
- Linux x64: `dotllm-<version>-linux-x64.tar.gz`
- macOS (Apple Silicon): `dotllm-<version>-osx-arm64.tar.gz`

Unpack and run:

```bash
# Linux / macOS
tar -xzf dotllm-<version>-linux-x64.tar.gz
cd dotllm-<version>-linux-x64
./dotllm model pull QuantFactory/SmolLM-135M-GGUF
./dotllm run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 64
./dotllm serve QuantFactory/SmolLM-135M-GGUF             # OpenAI-compatible API + chat UI
```

```powershell
# Windows
Expand-Archive dotllm-<version>-win-x64.zip -DestinationPath .
cd dotllm-<version>-win-x64
.\dotllm.exe model pull QuantFactory/SmolLM-135M-GGUF
.\dotllm.exe run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 64
```

*Experimental:* Native AOT builds for Linux and Windows are also attached to each release (`dotllm-<version>-aot-<rid>.{zip,tar.gz}`) — smaller and faster to start, but please [file an issue](https://github.com/kkokosa/dotLLM/issues/new) if you hit a crash.

**Option C — reference the libraries from your .NET app** — see [NuGet Packages](#nuget-packages) below.

### Build from source

Clone the repository and build with the .NET 10 SDK.

**Prerequisites:**

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- [Python 3.10+](https://www.python.org/) with `pip install rich InquirerPy` — only needed for the benchmark scripts under `scripts/`
- *Optional:* [llama.cpp](https://github.com/ggerganov/llama.cpp) for comparison benchmarks (see [llama.cpp setup](#llamacpp-setup))

```bash
git clone https://github.com/kkokosa/dotLLM.git
cd dotLLM
dotnet build -c Release
```

When built from source, replace `dotllm <subcommand>` in the [Usage](#usage) examples below with `dotnet run --project src/DotLLM.Cli -c Release -- <subcommand>`.

## Usage

dotLLM ships a single CLI tool with four command groups:

- **`dotllm model`** — download, list, search, and inspect GGUF models
- **`dotllm run`** — single-shot text generation with a performance summary
- **`dotllm chat`** — interactive multi-turn REPL with chat template formatting
- **`dotllm serve`** — OpenAI-compatible HTTP API with a built-in web chat UI

Models are identified by a local `.gguf` path or a HuggingFace repo ID (e.g., `QuantFactory/SmolLM-135M-GGUF`). **Models must be downloaded explicitly with `dotllm model pull` before they can be used** — `run`, `chat`, and `serve` read from `~/.dotllm/models/` and do not auto-fetch.

### Manage models

```bash
# Search HuggingFace for GGUF repos
dotllm model search llama --limit 5

# Download a repo (streams the .gguf files + tokenizer metadata into ~/.dotllm/models/)
dotllm model pull QuantFactory/SmolLM-135M-GGUF

# List everything cached locally
dotllm model list

# Show architecture, quantizations, and tokenizer info for a cached repo
dotllm model info QuantFactory/SmolLM-135M-GGUF

# Remove a cached repo
dotllm model delete QuantFactory/SmolLM-135M-GGUF
```

### Run — single-shot generation

Encodes a prompt, streams tokens to stdout, and prints a performance + memory summary.

```bash
# Greedy generation (default: temperature=0, max-tokens=128)
dotllm run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 64

# Sampled generation
dotllm run QuantFactory/SmolLM-135M-GGUF -p "Once upon a time" -n 128 -t 0.7 --top-k 40 --top-p 0.95

# JSON output (for scripting / piping)
dotllm run QuantFactory/SmolLM-135M-GGUF -p "Hello" --json

# Select a specific quantization when a repo has multiple .gguf files
dotllm run QuantFactory/SmolLM-135M-GGUF -p "Test" -q Q8_0

# GPU inference (requires NVIDIA GPU + CUDA Toolkit)
dotllm run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" --device gpu

# NUMA / P-core aware CPU threading
dotllm run QuantFactory/SmolLM-135M-GGUF -p "Test" --threads 8 --decode-threads 4 --numa-pin

# KV-cache quantization (Q8_0 / Q4_0) to fit longer contexts in memory
dotllm run QuantFactory/SmolLM-135M-GGUF -p "..." --cache-type-k q8_0 --cache-type-v q8_0

# Constrained JSON output
dotllm run QuantFactory/SmolLM-135M-GGUF -p "List 3 colors as JSON." --response-format json_object

# Speculative decoding — target + draft must share the same vocabulary.
# Example pair (validated by scripts/test_models_speculative.py):
#   target: Llama-3.2-3B-Instruct Q8_0     draft: Llama-3.2-1B-Instruct Q4_K_M
dotllm model pull bartowski/Llama-3.2-3B-Instruct-GGUF
dotllm model pull bartowski/Llama-3.2-1B-Instruct-GGUF
dotllm run bartowski/Llama-3.2-3B-Instruct-GGUF -q Q8_0 \
    -p "Explain what a CPU cache is in one sentence." -n 48 \
    --speculative-model bartowski/Llama-3.2-1B-Instruct-GGUF --speculative-k 5
```

Sample output:

```
── dotllm | Llama 30L/576H | Q8_0 | 16 threads | greedy ──────────────────
The capital of France is Paris. Paris is a city of romance and culture,

╭──────────────────────────────────────────────────────────────────────────╮
│                                                                          │
│  Generation Complete                                      163.27 tok/s   │
│                                                                          │
│  Performance                                                             │
│  Prefill            12.3 ms       6 tokens       487.80 tok/s            │
│  Decode             91.8 ms      15 tokens       163.40 tok/s            │
│  Sampling            0.1 ms      15 tokens                               │
│  ──────────────────────────────────────────────────────────              │
│  Total             104.2 ms      21 tokens       201.54 tok/s            │
│  Load              456.7 ms                                              │
│                                                                          │
│  Memory                                                                  │
│  Weights         136.73 MiB      (memory-mapped)                         │
│  Compute           2.25 MiB                                              │
│  KV Cache        158.20 MiB      (192 slots)                             │
│  ──────────────────────────────────────────────────────────              │
│  Total           297.18 MiB                                              │
│                                                                          │
│  length | 6 prompt, 15 generated                                         │
╰──────────────────────────────────────────────────────────────────────────╯
```

### Chat — interactive REPL

Multi-turn chat with persistent history, using the model's built-in chat template (falling back to ChatML). Prompt caching reuses KV-cache state across turns so subsequent turns skip redundant prefill.

```bash
# Basic chat
dotllm chat QuantFactory/SmolLM-135M-GGUF

# With a system prompt and sampling
dotllm chat QuantFactory/SmolLM-135M-GGUF --system "You are a helpful assistant." -t 0.8 --top-p 0.95

# GPU + KV-cache quantization for long contexts
dotllm chat bartowski/Llama-3.2-3B-Instruct-GGUF --device gpu --cache-type-k q8_0 --cache-type-v q8_0
```

In-session commands: `/exit` or `/quit` to leave, `/clear` to reset history (keeps the system prompt), `/system <text>` to change the system prompt.

Sample session:

```
── dotllm chat | Llama 30L/576H | Q8_0 | 16 threads | greedy ─────────────
Type /exit to quit, /clear to reset history, /system <text> to set system prompt.

>>> Hello, how are you?
I'm doing well, thank you for asking! How can I help you today?
[42 prompt tokens, 18 generated tokens, 28 ms TTFT, 487.8 prefill tok/s, 163.4 decode tok/s]

>>> What is 2+2?
2 + 2 = 4.
[78 prompt tokens, 12 generated tokens, 45 ms TTFT, 312.5 prefill tok/s, 155.2 decode tok/s]

>>> /clear
History cleared.
>>> /exit
```

### Serve — OpenAI-compatible API + chat UI

Starts a local HTTP server exposing an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/tokenize`, streaming SSE, tool calling) plus a built-in single-page web chat UI. Paged KV-cache, prompt caching, and startup warm-up are on by default. The browser opens automatically unless `--no-browser` is set.

```bash
# Start the server with a loaded model and open the chat UI
dotllm serve QuantFactory/SmolLM-135M-GGUF

# Bind a public interface, custom port, API-only, no auto-browser
dotllm serve QuantFactory/SmolLM-135M-GGUF --host 0.0.0.0 --port 9000 --no-ui --no-browser

# Start without a model — pick one from the chat UI
dotllm serve

# GPU with partial hybrid offload and more warm-up iterations
dotllm serve bartowski/Llama-3.2-3B-Instruct-GGUF --device gpu --gpu-layers 24 --warmup-iterations 5

# Speculative decoding — draft must share the target's vocabulary
dotllm serve bartowski/Llama-3.2-3B-Instruct-GGUF -q Q8_0 \
    --speculative-model bartowski/Llama-3.2-1B-Instruct-GGUF --speculative-k 5
```

Any OpenAI-compatible client works against the running server:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM-135M",
    "messages": [{"role":"user","content":"Say hi in one word."}],
    "stream": true
  }'
```

To embed the same endpoints inside your own ASP.NET Core app, see [Host the OpenAI API in your ASP.NET app](#host-the-openai-api-in-your-aspnet-app) below.

### CLI option reference

**Common options** (shared by `run`, `chat`, and `serve`):

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--device` | `-d` | `cpu` | Compute device: `cpu`, `gpu`, `gpu:0`, `gpu:1` |
| `--gpu-layers` | | *(all if `gpu`, 0 if `cpu`)* | Transformer layers on GPU (hybrid offload) |
| `--threads` | | 0 (auto) | CPU threads for inference |
| `--decode-threads` | | 0 (auto) | Decode threads (capped at memory channels) |
| `--numa-pin` | | false | Pin workers to NUMA-local cores (multi-socket) |
| `--pcore-only` | | false | Pin workers to P-cores only (Intel hybrid) |
| `--quant` | `-q` | *(auto)* | Quant filter when a repo has multiple `.gguf` files (e.g., `Q4_K_M`) |
| `--cache-type-k` | | `f32` | KV-cache key quant: `f32`, `q8_0`, `q4_0` |
| `--cache-type-v` | | `f32` | KV-cache value quant: `f32`, `q8_0`, `q4_0` |
| `--speculative-model` | | *(none)* | Draft model for speculative decoding (must share vocab) |
| `--speculative-k` | | 5 | Draft tokens per speculative step |

**Sampling & constraints** (shared by `run` and `chat`):

| Option | Short | Default (`run`) | Default (`chat`) | Description |
|--------|-------|-----------------|------------------|-------------|
| `--max-tokens` | `-n` | 128 | 512 | Max tokens per generation |
| `--temp` | `-t` | 0 (greedy) | 0 (greedy) | Sampling temperature |
| `--top-k` | | 0 (off) | 0 (off) | Top-K sampling |
| `--top-p` | | 1.0 | 1.0 | Nucleus threshold |
| `--min-p` | | 0 (off) | 0 (off) | Min-P threshold |
| `--repeat-penalty` | | 1.0 | 1.0 | Repetition penalty |
| `--repeat-last-n` | | 0 (full) | 0 (full) | Penalty lookback window |
| `--seed` | `-s` (run only) | *(random)* | *(random)* | Random seed for reproducibility |
| `--cache-window` | | 0 | 0 | Full-precision tail window for KV quant |
| `--paged` | | off | off | Use paged (block-based) KV-cache |
| `--response-format` | | `text` | `text` | `text`, `json_object`, `json_schema`, `regex`, `grammar` |
| `--schema` | | — | — | JSON Schema (or `@file.json`) for `json_schema` |
| `--pattern` | | — | — | Regex pattern for `regex` |
| `--grammar` | | — | — | GBNF grammar (or `@file.gbnf`) for `grammar` |
| `--tools` | | — | — | Tool definitions JSON (or `@file.json`) |

**`run`-only:**

- `--prompt` / `-p` — input prompt (**required**)
- `--json` — emit a single JSON result object (suppresses formatted output)

**`chat`-only:**

- `--system` / `-s` — system prompt
- `--tool-choice` — `auto` (default), `none`, `required`, or a function name
- `--no-prompt-cache` — disable KV-cache reuse across turns
- `--prompt-cache-size` — max cached sessions (default: `1`)
- `--verbose` / `-v` — debug output (finish reason, raw text, tool-call details)

**`serve`-only:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | | `localhost` | Bind address |
| `--port` | `-p` | `8080` | Port to listen on |
| `--no-ui` | | false | Disable the built-in chat UI (API only) |
| `--no-browser` | | false | Don't auto-open the browser |
| `--no-paged` | | false | Disable paged KV-cache (paged is **on** by default for `serve`) |
| `--no-prompt-cache` | | false | Disable KV-cache reuse across requests |
| `--prompt-cache-size` | | `4` | Max cached sessions |
| `--no-warmup` | | false | Disable startup warm-up passes |
| `--warmup-iterations` | | `3` | Warm-up iteration count |

> **Short-flag gotcha:** `-p` is **prompt** under `run` but **port** under `serve`. `-s` is **seed** under `run` but **system** under `chat`. When in doubt, use the long form.

## Development

### Debug build

Building in `Debug` configuration (`-c Debug`) enables a `debug` command group with diagnostic tools for inspecting GGUF files and model internals. These commands are excluded from Release builds via `#if DEBUG`.

```bash
# Build in Debug mode
dotnet build src/DotLLM.Cli -c Debug

# Inspect GGUF file structure
dotnet run --project src/DotLLM.Cli -c Debug -- debug gguf-header model.gguf
dotnet run --project src/DotLLM.Cli -c Debug -- debug gguf-metadata model.gguf
dotnet run --project src/DotLLM.Cli -c Debug -- debug gguf-tensors model.gguf
dotnet run --project src/DotLLM.Cli -c Debug -- debug gguf-config model.gguf

# Tokenizer round-trip verification
dotnet run --project src/DotLLM.Cli -c Debug -- debug tokenize model.gguf --text "Hello world"

# Single forward pass with top-10 logit diagnostics
dotnet run --project src/DotLLM.Cli -c Debug -- debug forward-pass model.gguf --prompt "Hello"

# Inspect embedding vector for a token ID
dotnet run --project src/DotLLM.Cli -c Debug -- debug embed-lookup model.gguf --token-id 1
```

| Command | Description |
|---------|-------------|
| `debug gguf-header` | GGUF header structure (magic, version, tensor/metadata counts) |
| `debug gguf-metadata` | All metadata key-value pairs |
| `debug gguf-tensors` | Tensor descriptors (name, shape, quantization type, offset) |
| `debug gguf-config` | Extracted `ModelConfig` (architecture, layers, dims, RoPE params) |
| `debug tokenize` | Encode text → token IDs → decode, verify round-trip fidelity |
| `debug forward-pass` | Single forward pass, top-10 predicted tokens with softmax probabilities |
| `debug embed-lookup` | Raw embedding vector for a given token ID |

> Debug builds are significantly slower than Release (~2-10x) because JIT optimizations, inlining, and SIMD vectorization are reduced. Always use `-c Release` for performance measurements.

### Tests

**Unit and integration tests:**

```bash
dotnet test
```

> Integration tests automatically download several GGUF models (~4.5 GB total) from HuggingFace to `~/.dotllm/test-cache/` on first run. The first `dotnet test` will take a while; subsequent runs use the cache. To run only unit tests (no downloads): `dotnet test tests/DotLLM.Tests.Unit`.

> **GPU tests** (tagged `Category=GPU`) require an NVIDIA GPU and run full model inference — they can take 20-30 minutes. They are skipped automatically on machines without CUDA. To exclude them explicitly: `dotnet test tests/DotLLM.Tests.Unit/ --filter "Category!=GPU"`

**Model correctness smoke tests** (`scripts/test_models.py`) run dotLLM CLI with greedy decoding across architectures (Llama, Mistral, Phi, Qwen) and verify expected output:

```bash
# Build CLI first
dotnet build src/DotLLM.Cli -c Release

# List available test cases and which models are cached
python scripts/test_models.py --list

# Run tests for all cached models
python scripts/test_models.py

# Download missing models and run all tests
python scripts/test_models.py --download

# Run only specific architectures
python scripts/test_models.py --filter phi,qwen
```

Models are downloaded from HuggingFace to `~/.dotllm/models/` on first use and cached for subsequent runs.

Sample output:

```
Test                                Arch       Result      Time  Details
- **2026-04** — **First public release (v0.1.0-preview.1)** — dotLLM goes public. [NuGet packages](#nuget-packages) for all 10 libraries + `DotLLM.Cli` as a global `dotnet tool`. Self-contained single-file downloads for Windows / Linux / macOS (Apple Silicon) and experimental Native AOT builds for Linux / Windows attached to every [GitHub Release](https://github.com/kkokosa/dotLLM/releases). Companion website at [dotllm.dev](https://dotllm.dev/) ([#119](https://github.com/kkokosa/dotLLM/issues/119))
- **2026-04** — **Wave 7**: CPU performance cleanup pass — `TopKSampler` replaces full `Array.Sort` with a hand-rolled size-K min-heap (`O(N log K)`, stack-resident scratch); `JsonSchemaConstraint` adds first-char bucketing to skip the ~160 MB of struct clones per mask build when the tracker rejects most leading characters, plus LRU eviction instead of the previous full-flush cache overflow; `Dequantize.Q5_0` gains an AVX2 path matching Q8_0's throughput (reuses `MatMulQ5_0.ExtractQ5HighBits` / `vpshufb` bit-extraction); `BpeTokenizer` pre-splits special tokens via the existing `Trie.TryMatchLongest` instead of the O(n × m) linear scan; `ComputeThreadPool` now pins the caller (inference) thread to the first candidate P-core on first `Dispatch`, eliminating the hybrid-CPU stall where pinned P-core workers idled at the barrier waiting for an E-core caller. New BenchmarkDotNet suites for TopK sampling, schema mask build, and special-token encode ([#109](https://github.com/kkokosa/dotLLM/issues/109))
- **2026-04** — **Phase 7 begins**: Logprobs — OpenAI-compatible `logprobs: true` + `top_logprobs: N` (0-20) on `/v1/chat/completions` and `/v1/completions`. Per-token log-softmax captured before sampling, returned in both streaming SSE chunks and non-streaming responses. Chat UI gains opt-in logprobs visualization: color-coded token confidence (green/lime/yellow/orange/red), hover tooltips with top-K alternatives and probabilities, diagnostic cues for low confidence, ambiguity, and sampling effect. `DotLLM.Sample.Logprobs` console sample with ANSI-colored output ([#101](https://github.com/kkokosa/dotLLM/issues/101))
- **2026-04** — **Phase 6 complete**: Speculative decoding — draft-verify-accept loop with modified rejection sampling. A small draft model proposes K candidate tokens; the target model verifies in one batched forward pass. Greedy fast-path for temperature=0. `IKvCache.Rollback()` for KV-cache truncation on rejection. `IDecodingConstraint.Clone()` for constraint rollback. Serve UI gains draft model selector and K slider. `--speculative-model` and `--speculative-k` CLI options. Speculative acceptance rate in response timings ([#98](https://github.com/kkokosa/dotLLM/issues/98))
- **2026-04** — Paged KV-cache — block-based KV-cache memory management (the allocation half of PagedAttention): shared block pool, block tables, ref counting, copy-on-write. Foundation for advanced prefix sharing (step 37, hard requirement) and speculative decoding (step 43, cheap rollback/fork). Attention kernels still operate on contiguous buffers via staging-buffer gather — true paged attention kernels are a future step. `--paged` (opt-in for CLI), on by default for `serve`. `--no-ui` flag for API-only hosting. See [docs/KV_CACHE.md](docs/KV_CACHE.md) ([#96](https://github.com/kkokosa/dotLLM/issues/96))
- **2026-04** — Native AOT (experimental) — opt-in `dotnet publish -p:PublishAot=true` produces a single-file `dotllm` binary with ~50ms startup (vs ~500ms JIT). Source-generated JSON serialization across all projects, `CreateSlimBuilder` for ASP.NET, `TrimmerRoots.xml` for Spectre.Console.Cli type preservation. JIT remains default for best throughput (Dynamic PGO). See [docs/AOT.md](docs/AOT.md) ([#94](https://github.com/kkokosa/dotLLM/issues/94))
- **2026-04** — **Phase 6 begins**: Warm-up — configurable dummy inference passes at server startup trigger .NET Tier-1 JIT promotion (Dynamic PGO) and exercise CUDA/cuBLAS pipelines. `/ready` probe gates on warm-up completion. `--no-warmup` to disable, `--warmup-iterations N` to configure ([#92](https://github.com/kkokosa/dotLLM/issues/92))
- **2026-04** — **Phase 5 complete**: simple prompt caching completes the constrained decoding & API phase — all 7 steps done (JSON mode, JSON Schema, regex/CFG, tool calling, API server, chat UI, prompt caching)
- **2026-04** — Simple prompt caching — `PrefixCache` keeps live KV-cache instances across generation calls. On each turn, element-wise prefix match finds cached tokens and skips redundant prefill, processing only new suffix tokens. LRU eviction with configurable max sessions. Enabled by default in `chat` and `serve` commands (`--no-prompt-cache` to disable). Cached token count reported in CLI stats, API `timings.cached_tokens`, and Chat UI stats bar. Near-100% cache hit rate in multi-turn chat, dramatically reducing TTFT on subsequent turns ([#90](https://github.com/kkokosa/dotLLM/issues/90))
- **2026-04** — Built-in web chat UI — `dotllm serve model.gguf` starts the API server and opens a browser to a bundled single-page chat UI (vanilla JS + TailwindCSS, embedded as resources in the DLL). Per-message inference stats (prefill/decode tok/s, TTFT), live sampling parameter control, model hot-swap from the UI, system prompt, verbose mode. New endpoints: `/props`, `/v1/config`, `/v1/models/available`, `/v1/models/load`. Streaming SSE now includes `usage` + `timings` in the final chunk ([#86](https://github.com/kkokosa/dotLLM/issues/86))
- **2026-04** — ASP.NET OpenAI-compatible API server — `DotLLM.Server` with `/v1/chat/completions` (streaming SSE + non-streaming), `/v1/completions`, `/v1/models`, `/v1/tokenize`, `/v1/detokenize`, health/ready probes. Chat template formatting, tool calling with `IToolCallParser` detection, `response_format` constrained decoding. Model loading at startup via `--model`/`--device` CLI args. Sequential request processing via semaphore ([#84](https://github.com/kkokosa/dotLLM/issues/84))
- **2026-04** — Tool calling — `IToolCallParser` implementations for Llama 3.1+, Hermes/Qwen, Mistral, and generic fallback. Auto-detection factory selects parser from model architecture and chat template content. `ToolCallSchemaBuilder` generates JSON Schema from tool definitions for constrained decoding (`tool_choice=required`). `ToolCallDetector` for post-generation detection, `StreamingToolCallAccumulator` for streaming. `--tools` and `--tool-choice` CLI options with multi-turn tool use in chat REPL. Parallel tool calls supported ([#82](https://github.com/kkokosa/dotLLM/issues/82))
- **2026-04** — Regex + CFG constrained decoding — `RegexConstraint` compiles patterns to minimized DFA (Thompson NFA → subset construction → Hopcroft minimization) with equivalence-class compression. `GrammarConstraint` parses GBNF grammars into PDA with InlineArray-based call stack. Both use zero-alloc struct simulators and dictionary-cached token masks. `--response-format regex --pattern <pattern>` and `--response-format grammar --grammar <gbnf|@file>` CLI support ([#80](https://github.com/kkokosa/dotLLM/issues/80))
- **2026-03** — JSON Schema constrained decoding — `JsonSchemaConstraint` layers schema tracking on `JsonCharParser` to enforce type constraints, required properties, enum values, nested structures. Schema compiled into flat node array with property-name tries. Zero-alloc `Clone()` via struct-copy. `--response-format json_schema --schema <json|@file>` CLI support ([#78](https://github.com/kkokosa/dotLLM/issues/78))
- **2026-03** — **Phase 5 begins**: JSON mode constrained decoding — `JsonConstraint` FSM guarantees syntactically valid JSON output via per-token vocabulary masking. Stack-based PDA (RFC 8259), AVX2-vectorized logit masking, state-keyed mask cache. `--response-format json_object` CLI flag ([#76](https://github.com/kkokosa/dotLLM/issues/76))
- **2026-03** — KV-cache quantization: Q8_0 and Q4_0 KV-cache compression on CPU and GPU (3.7–7.1× memory reduction). Separate `--cache-type-k`/`--cache-type-v` options, mixed-precision window `--cache-window N` keeps recent tokens in full precision. Dual-region storage with quantize-on-evict, per-tile dequantization in tiled attention ([#74](https://github.com/kkokosa/dotLLM/issues/74))
- **2026-03** — CPU/GPU hybrid layer offloading: `--gpu-layers N` to run first N layers on GPU, remainder on CPU. Automatic FP16→FP32 hidden state transfer at boundary. Split KV-cache (GPU FP16 + CPU FP32). Partial VRAM usage proportional to offloaded layers ([#72](https://github.com/kkokosa/dotLLM/issues/72))
- **2026-03** — CUDA GPU backend: PTX kernels via CUDA Driver API P/Invoke (no native shared library), cuBLAS HGEMM for prefill, custom quantized GEMV for decode (Q8_0, Q4_K, Q6_K), FP16 activation pipeline, on-the-fly weight dequantization, GPU KV-cache, `--device gpu` CLI flag, `--device both` benchmarking ([#70](https://github.com/kkokosa/dotLLM/issues/70))
- **2026-03** — NUMA-aware threading: adaptive spin-wait dispatch (generation counter with event fallback), NUMA topology detection (Windows/Linux), P-core/E-core awareness, CPU affinity pinning, auto-reduced decode thread count ([#57](https://github.com/kkokosa/dotLLM/issues/57))
- **2026-03** — Operator fusion: fused RMSNorm+quantize (decode-only, eliminates normOut intermediate buffer) and tiled SwiGLU (1KB L1-resident sigmoid buffer) reduce DRAM roundtrips on the decode hot path ([#56](https://github.com/kkokosa/dotLLM/issues/56))
- **2026-03** — Fast approximate exp/softmax: Schraudolph IEEE-754 bit-manipulation trick replaces polynomial exp (~3 SIMD ops vs ~12) in attention softmax. AVX2/AVX-512 fused shift+exp+sum pass eliminates 3 separate TensorPrimitives calls. Sampling softmax keeps full precision ([#55](https://github.com/kkokosa/dotLLM/issues/55))
- **2026-03** — Tiled attention with online softmax: O(N) memory flash-attention-style algorithm replaces O(N²) score matrix materialization, eliminates 64 MB/head allocations at ctx 4096, uses ~1 KB stack per head ([#54](https://github.com/kkokosa/dotLLM/issues/54))
- **2026-03** — Row-interleaved weight repacking: R4 layout stores 4 consecutive rows' blocks contiguously at model load time, improving cache/TLB locality for all quantized GEMV kernels ([#52](https://github.com/kkokosa/dotLLM/issues/52))
- **2026-03** — Q8_1 input quantization: precomputed block sums for Q5_0 kernels, 2-block loop unrolling, eliminates ~4 SIMD ops/block from Q5_0 vec_dot hot path ([#51](https://github.com/kkokosa/dotLLM/issues/51))
- **2026-03** — Fused decode dispatch: Q/K/V (3→1) and Gate/Up (2→1) projection fusion saves ~72 dispatches/layer, ~4% decode throughput improvement ([#50](https://github.com/kkokosa/dotLLM/issues/50))
- **2026-03** — **Phase 2 complete**: additional model architectures (Mistral, Phi, Qwen), sliding window attention, fused QKV support, `IModel` interface, `ModelLoader` helper ([#34](https://github.com/kkokosa/dotLLM/issues/34))
- **2026-03** — Streaming token generation: `IAsyncEnumerable<GenerationToken>` API with UTF-8-safe incremental text, `CancellationToken` support, and per-token finish reason/timings ([#31](https://github.com/kkokosa/dotLLM/issues/31))
- **2026-03** — Chat template engine: Jinja2-subset interpreter (lexer→parser→evaluator), `IChatTemplate` implementation, `GgufChatTemplateFactory`, `dotllm chat` REPL command ([#30](https://github.com/kkokosa/dotLLM/issues/30))
- **2026-03** — Mixed quantization + Q8_K: Q8_K input quantization (float32 scale, 256-element blocks, precomputed bsums), true 4-row fused K-quant kernels, re-enabled Q4_K×Q8_K/Q5_K×Q8_K/Q6_K×Q8_K fused GEMV/GEMM ([#29](https://github.com/kkokosa/dotLLM/issues/29))
- **2026-03** — Q4_K_M dequantization and vec_dot kernels: Q4_K, Q5_K, Q6_K scalar + AVX2 dequant and fused matmul kernels with full model-level dispatch ([#28](https://github.com/kkokosa/dotLLM/issues/28))
- **2026-03** — BDN inference benchmarks: end-to-end benchmarks with custom tok/s columns, auto model download, llama.cpp comparison script ([#42](https://github.com/kkokosa/dotLLM/issues/42))
- **2026-03** — Engine inference timings: `InferenceTimings` on `InferenceResponse`, `onTokenGenerated` callback, CLI refactored to use `TextGenerator` ([#41](https://github.com/kkokosa/dotLLM/issues/41))
- **2026-03** — Multi-threaded CPU inference: zero-alloc `ComputeThreadPool` with `delegate*` dispatch, parallel GEMV/GEMM and head-parallel attention ([#36](https://github.com/kkokosa/dotLLM/issues/36))
- **2026-03** — SIMD kernel tuning: FMA float accumulation, 4-row batched GEMV, AVX-512 paths, SIMD quantization ([#26](https://github.com/kkokosa/dotLLM/issues/26))
- **2026-03** — Phase 1 complete: sampling pipeline + stop conditions — first coherent multi-token generation ([#24](https://github.com/kkokosa/dotLLM/pull/24))
- **2026-03** — KV-cache: eval drops from 1091 ms/token to 227 ms/token (~4.8× speedup)
- **2026-03** — Llama forward pass: first token generation from embedding to logits
- **2026-02** — BPE Tokenizer with SentencePiece and tiktoken support ([#16](https://github.com/kkokosa/dotLLM/pull/16))

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **1 — End-to-End Generation** | GGUF loading, dequantization, CPU ops, tokenizer, attention, forward pass, KV-cache, sampling | Done (9/9) |
| **2 — Practical Local Inference** | Engine metrics, benchmarks, Q4_K_M, chat templates, streaming, multi-threading, more architectures | Done (10/10) |
| **3 — CPU Performance** | Decode dispatch, Q8_1 input, weight repacking, outer-product GEMM, tiled attention, fast exp, fusion, NUMA | In Progress (7/8) |
| **4 — GPU Acceleration** | CUDA backend, CPU/GPU hybrid, KV-cache quantization | Done (3/3) |
| **5 — Constrained Decoding & API** | JSON mode, JSON Schema, regex/CFG, tool calling, OpenAI API server, chat UI, prompt caching | Done (7/7) |
| **6 — Improved Serving** | Warm-up, NativeAOT, paged KV-cache, speculative decoding | Done (4/4) |
| **7 — Diagnostics & Interpretability** | Logprobs, hook system, logit lens, SAE integration, LoRA adapters | In Progress (1/5) |
| **8 — Model Expansion** | MLA attention, ALiBi, SmolLM3, Gemma 4, Mixture of Experts | Planned (0/5) |
| **9 — Production Serving** | Continuous batching, prefix sharing, advanced scheduling, rate limiting, metrics & tracing | Planned (0/5) |

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed steps, dependencies, and milestones.

## Documentation

- [Architecture & data flow](docs/ARCHITECTURE.md)
- [GGUF binary format](docs/GGUF_FORMAT.md)
- [Quantization formats](docs/QUANTIZATION.md)
- [Attention mechanisms](docs/ATTENTION.md)
- [Position encoding](docs/POSITION_ENCODING.md)
- [Tokenizers & chat templates](docs/TOKENIZERS.md)
- [Sampling pipeline](docs/SAMPLING.md)
- [Constrained decoding](docs/CONSTRAINED_DECODING.md)
- [Tool calling](docs/TOOL_CALLING.md)
- [KV-cache management](docs/KV_CACHE.md)
- [GPU inference](docs/GPU.md)
- [CUDA backend architecture](docs/CUDA.md)
- [Batch scheduling](docs/SCHEDULING.md)
- [Native AOT deployment](docs/AOT.md)
- [Full roadmap](docs/ROADMAP.md)

## Contributing

Contributions are welcome! dotLLM uses an issue-driven workflow — every change starts with a [GitHub issue](https://github.com/kkokosa/dotLLM/issues) describing the work. Pick an existing issue or open a new one, then submit a PR targeting `main`.

## Contact

Questions, ideas, or feedback? Open a thread in [GitHub Discussions](https://github.com/kkokosa/dotLLM/discussions).

## Author

Built by **[Konrad Kokosa](https://kokosa.dev/)** — .NET MVP, author of *Pro .NET Memory Management* (2nd ed.), and AI/agents engineer at Nethermind. Over 20 years of .NET performance work.

- Website: [dotllm.dev](https://dotllm.dev/)
- Personal: [kokosa.dev](https://kokosa.dev/)
- GitHub: [@kkokosa](https://github.com/kkokosa)

## License

dotLLM is licensed under the [GNU General Public License v3.0](LICENSE).

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for GGUF format, quantization kernels, and CUDA implementations
- [Hugging Face](https://huggingface.co/) — model ecosystem, transformers reference implementations, tokenizer specs
- [.NET team](https://github.com/dotnet/runtime) — `TensorPrimitives`, `System.Runtime.Intrinsics`, `MemoryMappedFile`, and the runtime that makes this possible
