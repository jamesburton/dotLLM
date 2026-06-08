# Architecture — dotLLM

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│  DotLLM.Server (ASP.NET Minimal API)                            │
│  ├── /v1/chat/completions, /v1/completions, /v1/embeddings      │
│  ├── /v1/models, /v1/tokenize, /v1/detokenize                  │
│  ├── Tool calling protocol handler                              │
│  └── Rate limiting, API key auth, request priority              │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Engine (Orchestration)                                  │
│  ├── InferenceEngine          — Main entry point                │
│  ├── IScheduler               — Continuous batch scheduling     │
│  ├── PagedKvCacheManager      — Block allocation, prefix cache  │
│  ├── SamplerPipeline          — Composable ISamplerStep chain   │
│  ├── ConstraintEngine         — FSM/PDA for structured output   │
│  ├── ISpeculativeDecoder      — Draft-verify-accept loop        │
│  └── IAdapterManager          — LoRA runtime management         │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Models                     DotLLM.Tokenizers            │
│  ├── GGUF loader (mmap)            ├── BPE (tiktoken-style)     │
│  ├── SafeTensors loader            ├── SentencePiece            │
│  ├── LlamaModel                    ├── HuggingFace tokenizer    │
│  ├── MistralModel                  └── Chat template engine     │
│  ├── PhiModel, QwenModel                                        │
│  └── DeepSeekModel (MLA)                                        │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Diagnostics              DotLLM.Telemetry               │
│  ├── Hook registry                ├── IInferenceMetrics          │
│  ├── Activation capture           └── IRequestTracer             │
│  ├── Logit lens                                                  │
│  └── SAE integration                                             │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Core (Interfaces & Abstractions)                        │
│  ├── ITensor, TensorShape, DType                                │
│  ├── IBackend, DevicePlacement                                  │
│  ├── IAttentionStrategy                                         │
│  ├── IPositionEncoding                                           │
│  ├── ISamplerStep, ILogitProcessor, IStopCondition              │
│  ├── IDecodingConstraint, TokenMask                             │
│  ├── IInferenceHook, HookPoint                                  │
│  └── ModelConfig, InferenceOptions                              │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Cpu              │  DotLLM.Cuda                         │
│  ├── CpuBackend          │  ├── CudaBackend                     │
│  ├── SIMD kernels        │  ├── P/Invoke interop                │
│  └── TensorPrimitives    │  └── Handle management               │
├──────────────────────────┼──────────────────────────────────────┤
│  P/Invoke boundary       │                                      │
├──────────────────────────┼──────────────────────────────────────┤
│  Native C/CUDA Library   │                                      │
│  ├── cuBLAS GEMM         │  ├── Flash attention .cu             │
│  ├── Quantized matmul    │  ├── Fused RoPE/RMSNorm/SiLU .cu    │
│  ├── NCCL wrappers       │  └── GPU memory pool                 │
└──────────────────────────┴──────────────────────────────────────┘
```

## Data Flow: Model Loading

```
GGUF file on disk
  │
  ├─ Header parsing ──→ magic, version, tensor count, metadata count
  ├─ Metadata parsing ──→ ModelConfig (architecture, dims, vocab, RoPE params)
  │                       ChatTemplate (Jinja2 string)
  │                       Tokenizer vocabulary + merges + scores
  │
  └─ Tensor data section
       │
       MemoryMappedFile.CreateFromFile()
       │
       ├─ Tensor descriptors ──→ (name, shape, quantization type, offset)
       │
       └─ Memory-mapped region ──→ OS demand-pages from disk
            │                       No managed heap allocation
            │
            ├─ CPU tensors: raw pointer via SafeMemoryMappedViewHandle
            └─ GPU tensors: cudaMemcpy from mmap'd host → device memory
```

## Data Flow: Inference Request

```
HTTP POST /v1/chat/completions
  │
  ├─ Parse request (messages, tools, sampling params, constraints)
  ├─ Apply chat template ──→ IChatTemplate.Apply(messages) ──→ prompt string
  ├─ Tokenize ──→ ITokenizer.Encode(prompt) ──→ int[] token_ids
  ├─ Prefix cache lookup ──→ match existing KV-cache blocks
  ├─ Enqueue in scheduler with priority
  │
  └─ Scheduler admits request when KV-cache capacity available
       │
       ├─ PREFILL (compute-bound)
       │    For each layer:
       │      Norm → Q/K/V projection → RoPE → Attention → Residual
       │      → Norm → FFN (+LoRA delta) → Residual
       │      [Hooks fire at each stage if registered]
       │    Store K, V in KV-cache blocks
       │
       ├─ DECODE LOOP (memory-bandwidth-bound)
       │    Each iteration:
       │      Forward pass for single token (using cached K, V)
       │      → Sampler pipeline: logit_bias → constraint → penalties
       │        → temperature → top_k → top_p → min_p → sample
       │      → Check stop conditions
       │      → Advance constraint FSM
       │      → Yield token via SSE stream
       │
       └─ Response: tokens + usage + finish_reason
```

## Data Flow: Speculative Decoding

```
Draft model generates K candidates → Target model verifies in single forward pass
→ Accept left-to-right via rejection sampling → Rollback rejected tokens
  (KV-cache entries + constraint state rolled back)
```

See [SPECULATIVE.md](SPECULATIVE.md) for full design.

## Data Flow: Hybrid CPU-prefill / iGPU-decode

```
                ┌─────────────────────────────────────────────┐
                │  Memory-mapped GGUF (single OS page cache)  │
                └──────┬───────────────────────────┬──────────┘
                       │ mmap view A                │ mmap view B
                       ▼                            ▼
              ┌─────────────────┐         ┌─────────────────────┐
              │ TransformerModel │         │ VulkanTransformerModel │
              │  (CPU SIMD)      │         │ (iGPU compute)         │
              └────────┬─────────┘         └──────────┬─────────────┘
                       │ Forward(promptIds, ...)      │
                       ▼                              │
              ┌────────────────────┐                  │
              │ SimpleKvCache (FP32)│                 │
              │  [L, maxSeq, kvStride]                │
              └────────┬───────────┘                  │
                       │  per-layer KeysSpan /        │
                       │  ValuesSpan ───────────────► │
                       │                              ▼
                       │                  ┌──────────────────────┐
                       │   IngestFromHost │ VulkanKvCache         │
                       └─────────────────►│ (device-local FP32)   │
                                          └──────┬───────────────┘
                                                 │ decode loop
                                                 ▼
                                          iGPU produces token T₀
                                          (T₁, T₂, ... continue here)
```

**Why it's a dotLLM-native trick.** Lemonade-server (and the AMD GAIA stack
sitting on top of it) routes each backend — llama.cpp Vulkan, llama.cpp CPU,
ryzenai-server, vLLM ROCm — to a **separate subprocess**. KV state never
crosses the process boundary, so there is no in-process handoff point. dotLLM
is a single .NET process: every `IModel` lives in the same address space and
can mmap the same GGUF file, and copying KV bytes between heaps is a memcpy
(on a unified-memory APU like Strix Halo, the bytes never even leave system
DRAM — only the driver's tiling layout changes). See
`.planning/notes/gaia-lemonade-research.md` §6 H4.

**When it's a win.** CPU prefill (AVX-512 + Q4 dequant) competes with iGPU
prefill on short prompts because the iGPU pays a one-shot pipeline-cache /
descriptor-binding warm-up tax (~30-80 ms on Strix Halo) that dominates a
64-token prefill. Above ~256 tokens the iGPU's batched-GEMM throughput
amortises that tax and wins. Decode is memory-bandwidth-bound — the iGPU's
~212 GB/s measured DDR5 bandwidth keeps it ahead regardless of prompt
length. Switching at the prefill/decode boundary captures both biases.

**Where it lives in the code.**

| File | Role |
|---|---|
| `src/DotLLM.Core/Backends/BackendCapabilities.cs` | Stock profiles (Cpu / VulkanIgpu / CudaDiscrete) plus the per-strategy crossover threshold (`DOTLLM_HYBRID_PREFILL_CROSSOVER` env override). |
| `src/DotLLM.Engine/Strategies/HybridPrefillDecodeStrategy.cs` | `RunPrefill` + `Handoff` orchestration. Holds the two `IModel` instances and an `HybridKvHandoff` delegate. Engine itself does NOT reference any backend; the delegate is wired up at the call site. |
| `src/DotLLM.Engine/KvCache/SimpleKvCache.cs` | `KeysSpan(layer)` / `ValuesSpan(layer)` accessors over the contiguous FP32 layer buffers (host side). |
| `src/DotLLM.Vulkan/VulkanKvCache.cs` | `IngestFromHost(layer, length, keys, values)` — host-staging → device-local KV destination. |
| `src/DotLLM.Engine/TextGenerator.cs` | Optional `hybridStrategy` constructor parameter; when the prompt is below the crossover the prefill block routes through the strategy, otherwise the existing single-backend path runs unchanged. |

**KV layout compatibility.** Both `SimpleKvCache` and `VulkanKvCache` use
the identical logical layout `[maxSeqLen, numKvHeads × headDim]` FP32 per
layer. The handoff is therefore a per-layer memcpy with no reshape or
re-quantisation. The Vulkan path's device-local memory may be tiled by the
driver — `vkCmdCopyBuffer` from a host-visible staging buffer handles the
swizzle transparently.

**Crossover semantics.** `HybridPrefillDecodeStrategy.ShouldRunHybrid(N)`
returns `true` iff `0 < N < CrossoverTokens`. Hybrid mode is disabled
automatically when:

- the prefix cache returns a hit (the CPU side starts from a fresh
  `SimpleKvCache`; reconciling with a populated decode-side cache is a
  follow-up),
- a speculative draft model is wired up (the draft cache lives on the
  decode device; co-prefilling draft + target on different devices needs
  separate work).

In both cases the existing single-backend prefill path runs.

**Acceptance** (per `.planning/notes/gaia-lemonade-research.md` §6 H4):

- 64-token prompt, first-32-tokens latency ≥ 10 % lower than pure-Vulkan
  baseline.
- 1024-token prompt, no regression (gating routes long prompts back to
  pure-Vulkan prefill).
- Bit-identical output within FP32 reorder noise vs the pure-Vulkan path.

Tracked by `HybridPrefillDecodeBenchmarks` (PromptTokens 16/64/256/1024
× Mode PureVulkan/Hybrid) and `HybridPrefillDecodeTests`.

## NuGet Package Graph

```
DotLLM (pure .NET) ─── DotLLM.Server (ASP.NET)
├── Core, Models, Tokenizers, Cpu, Engine, Diagnostics, Telemetry

DotLLM.Backend.Cuda12 (native binaries) ── depends on DotLLM.Core
DotLLM.Backend.ROCm (future) ── depends on DotLLM.Core
```

## Threading Model

- **Server**: ASP.NET thread pool, fully async.
- **Scheduler**: Single dedicated thread, communicates via `Channel<T>`.
- **Inference**: Synchronous compute on scheduler thread. GPU ops async (kernel launch + stream sync).
- **Hooks**: Synchronous on inference thread — must be fast.
- **Streaming**: Tokens pushed via `Channel<T>` to `IAsyncEnumerable<string>`.