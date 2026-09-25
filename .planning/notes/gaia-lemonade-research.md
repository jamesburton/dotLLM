# AMD GAIA + lemonade-server: Hybrid Execution Research

**Audience**: dotLLM agent prompting / planning
**Research date**: 2026-05-14
**Scope**: How GAIA and lemonade-server orchestrate hybrid execution across AMD silicon (CPU + Radeon iGPU + XDNA2 NPU), with focus on Strix Halo (Ryzen AI Max+ 395, gfx1151), and prioritized recommendations for dotLLM.

---

## 1. Executive Summary

**GAIA is a thin agent framework, not an inference runtime.** It is a Python (and a port to C++17) library distributed via `pip install amd-gaia` that implements the agent loop, tool registry, MCP client, RAG, voice (Whisper+Kokoro) and Qwen3-VL vision pipelines. **All actual LLM inference is delegated over HTTP to lemonade-server.** GAIA's hardware story ("NPU + iGPU on Ryzen AI") is entirely Lemonade's hardware story — GAIA itself contains no kernel-level code, no quantization, no device placement logic. The MIT-licensed `github.com/amd/gaia` repo confirms this: it is an agent SDK with a Lemonade Server prerequisite.

**Lemonade is the real inference orchestrator.** It is an Apache-2.0 C++ daemon (`lemond`) with a Tauri/React UI and CLI, exposing OpenAI-, Ollama-, and Anthropic-compatible REST plus a WebSocket Realtime audio API on port 13305. Architecturally it is a **router-of-subprocesses**: each backend (llama.cpp Vulkan/ROCm/CPU/Metal, FastFlowLM `flm` on NPU, RyzenAI-LLM `ryzenai-llm` on NPU, vLLM ROCm, whisper.cpp, stable-diffusion.cpp, Kokoro TTS) is launched as a separate process. The `Router` class forwards HTTP requests, manages LRU eviction per model type, enforces NPU exclusivity, and resolves a recipe → backend → device family pipeline based on `SystemInfo` hardware probing. Critically, **Lemonade does no in-process kernel work** — it is a process supervisor with hardware probing, model registry, and routing.

**The "hybrid" execution AMD markets has two distinct meanings and only one is actually integrated into Lemonade.** (a) The *OGA hybrid mode* (Ryzen AI 1.3+) partitions a single model across NPU (prefill, compute-bound) and iGPU (decode, memory-bound) inside a single ONNXRuntime GenAI process — this is implemented in OnnxRuntime-GenAI and AMD's VitisAI EP, surfaced via the `ryzenai-llm` recipe, and Lemonade just spawns the `ryzenai-server` subprocess against it. (b) The *coarse hardware routing* in Lemonade itself: pick llama.cpp+Vulkan for iGPU, llama.cpp+ROCm for dGPU, FLM/RyzenAI-LLM for NPU, vLLM ROCm for Strix Halo Linux experimental — based on a static `RECIPE_DEFS` table mapping recipes to OS+device family. There is **no per-layer or per-kernel placement decision made by Lemonade**, and **no zero-copy or unified-memory exploitation beyond what each subprocess does internally**. On Strix Halo specifically, OGA hybrid is *not currently validated* (`Ryzen AI OGA flow supports Strix and Krackan Point processors. Phoenix (PHX) and Hawk (HPT) processors are not supported`; Strix Halo not listed), so hybrid NPU+iGPU is **not** GAIA's primary Strix Halo story today — Vulkan+Radeon 8060S via llama.cpp is.

---

## 2. GAIA Architecture (what + where + how)

**Repo**: `https://github.com/amd/gaia`  **License**: MIT (2024-2026 AMD)  **Package**: `pip install amd-gaia`

**What it is**: An agent framework. From the README: *"GAIA is AMD's open-source framework for building intelligent AI agents that run 100% locally on AMD Ryzen AI hardware."* The Python entry point is `gaia.agents.base.agent.Agent` with a tool registration decorator (`@tool`). A C++17 port exists under `cpp/` (`#include <gaia/agent.h>`) for embedding without Python.

**What it contains**:
- Agent loop, tool registry, MCP client
- RAG (document indexing, semantic search over 50+ file formats)
- Voice integration: Whisper ASR + Kokoro TTS (these are themselves Lemonade backends)
- Vision: Qwen3-VL-4B (via Lemonade)
- Tauri/Electron desktop UI
- Plugin system distributed via PyPI

**What it does *not* contain**: any tensor code, any kernels, any quantization, any device placement, any inference runtime. The repo's `src/` is Python orchestration plus a C++ agent loop port. Confirmed by listing repo contents — there is no `cuda/`, `rocm/`, `vulkan/`, no GGUF parser, no GEMM.

**How it talks to hardware**: It doesn't. The Quickstart explicitly lists Lemonade Server as a prerequisite. The marketplace logo grid in Lemonade's README lists GAIA alongside Claude Code, Open WebUI, AnythingLLM, etc. as a *client* of Lemonade.

**System requirements (from README)**: Min Ryzen AI 300-series; **recommended Ryzen AI Max+ 395** (Strix Halo); Windows 11 or Linux; 16 GB min / 64 GB recommended RAM. Note 64 GB recommended — not 96/128 — confirming GAIA is not the layer reaching for unified-memory-resident MoE models.

**Takeaway for dotLLM**: GAIA itself is uninteresting — it is an agent SDK competing with LangChain/LangGraph. The hardware story sits one layer down in Lemonade. dotLLM's planned OpenAI-compatible server (`DotLLM.Server`) is the analog of Lemonade, *not* of GAIA. If dotLLM wants the AMD marketplace foothold, dotLLM should aim to be *runnable behind* a GAIA-equivalent (any OpenAI-compatible client) and *interchangeable with* Lemonade — which it already is via the OpenAI API contract.

---

## 3. lemonade-server (what + how it integrates)

**Repo**: `https://github.com/lemonade-sdk/lemonade`  **License**: Apache 2.0  **Language split**: C++17 server + React/Tauri UI + Python integration tests.

### 3.1 Process topology

From `AGENTS.md`:
- **`lemond`** — pure HTTP server, port 13305. Routes requests, manages model load/unload.
- **`lemonade`** — CLI client (`lemonade list`, `pull`, `run`, `launch`, `backends`). Talks to `lemond` over HTTP, discovers it via UDP beacon.
- **`LemonadeServer.exe`** — Windows tray app that embeds `lemond`, auto-started via Startup folder.
- **`lemonade-tray`** — macOS/Linux equivalent tray.
- **Tauri desktop app** — React 19 + TypeScript, Rust host. Discovers running `lemond`, never embeds it.

**Critical invariant** (from `AGENTS.md`): *"Backends run as subprocesses (llama-server, whisper-server, sd-server, koko, flm, ryzenai-server). They must NOT run in-process."* Lemonade is a process supervisor.

### 3.2 The backend abstraction

`WrappedServer` (`src/cpp/include/lemon/wrapped_server.h`) is the C++ ABC. Each backend:
- inherits `WrappedServer`
- implements `load()`, `unload()`, `chat_completion()`, `completion()`, `responses()`, optionally `install()` / `download_model()`
- runs as a subprocess; `WrappedServer` forwards HTTP requests via `forward_request("/v1/chat/completions", request)`
- reports a `DeviceType` bitmask (`DEVICE_CPU | DEVICE_GPU | DEVICE_NPU`) from `model_types.h`

The eight concrete backends and their device mapping (`src/cpp/server/backends/`):

| Backend file | Recipe name | Launches | Device |
|---|---|---|---|
| `llamacpp_server.cpp` | `llamacpp` | `llama-server` binary from ggml-org/llama.cpp + Vulkan/HIP/CPU/Metal variant | GPU (or CPU) |
| `fastflowlm_server.cpp` | `flm` | `flm` binary from FastFlowLM/FastFlowLM | NPU (XDNA2) |
| `ryzenaiserver.cpp` | `ryzenai-llm` | `ryzenai-server` binary from `lemonade-sdk/ryzenai-server` | NPU (XDNA2) |
| `vllm_server.cpp` | `vllm` | vLLM with ROCm (Linux only, Strix Halo experimental) | GPU |
| `whisper_server.cpp` | `whispercpp` | whisper-server | CPU / NPU / Vulkan |
| `sd_server.cpp` | `sd-cpp` | sd-server | CPU / ROCm |
| `kokoro_server.cpp` | `kokoro` | koko | CPU |

### 3.3 Hardware probing and recipe selection

`src/cpp/server/system_info.cpp` is the single source of truth for what runs where. Key data structures:

```cpp
// ROCm architecture mapping
const std::map<std::string, std::string> ROCM_ARCH_MAPPING = {
    {"gfx1030", "gfx103X"}, // RDNA2
    {"gfx1100", "gfx110X"}, // RDNA3
    {"gfx1150", "gfx1150"}, // Strix Point iGPU
    {"gfx1151", "gfx1151"}, // Strix Halo iGPU
    {"gfx1200", "gfx120X"}, // RDNA4
};

const std::map<std::string, std::string> DEVICE_FAMILY_NAMES = {
    {"gfx1151", "Radeon 8050S/8060S (Strix Halo)"},
    {"XDNA2",   "AMD XDNA 2"},
    // ...
};

// Recipe -> backend -> OS -> device family table
static const std::vector<RecipeBackendDef> RECIPE_DEFS = {
    {"llamacpp", "metal",  {"macos"},          {{"metal",{}}}},
    {"llamacpp", "vulkan", {"windows","linux"},{{"cpu",{"x86_64"}},{"amd_gpu",{}}}},
    {"llamacpp", "rocm",   {"windows","linux"},{{"amd_gpu",{"gfx1150","gfx1151","gfx103X","gfx110X","gfx120X"}}}},
    {"llamacpp", "cpu",    {"windows","linux"},{{"cpu",{"x86_64"}}}},
    {"flm",         "npu", {"windows","linux"},{{"amd_npu",{"XDNA2"}}}},
    {"ryzenai-llm", "npu", {"windows"},        {{"amd_npu",{"XDNA2"}}}},
    {"vllm",        "rocm",{"linux"},          {{"amd_gpu",{"gfx1150","gfx1151","gfx110X","gfx120X"}}}},
    // ...
};
```

The selection is **static** — there is no profiler, no runtime cost model, no adaptive switching. The `server_models.json` registry pins each model to a recipe (or list of recipes); the `Router` calls `create_backend_server()` (`router.cpp`) which is a literal `if (recipe == "flm") ... else if (recipe == "ryzenai-llm") ...` switch. Comment from `system_info.cpp`: *"IMPORTANT: Backend order matters! For recipes with multiple backends (e.g., llamacpp), the order in this table defines the preference order. First listed = most preferred."* So on Linux+Strix Halo, Vulkan is preferred over ROCm; on macOS, Metal over Vulkan over CPU.

### 3.4 Router behavior

`src/cpp/server/router.cpp`:
- Holds `std::vector<std::unique_ptr<WrappedServer>> loaded_servers_`.
- LRU eviction *per model type* (LLM, embedding, reranking, audio, image, TTS) — keyed by `last_access_time`.
- **NPU exclusivity** (Invariant #2 in AGENTS.md): *"Exclusive-NPU recipes (`ryzenai-llm`, `whispercpp` on NPU) evict ALL other NPU models before loading. FastFlowLM (`flm`) can coexist with other FLM types (max 1 per FLM type) but not with exclusive-NPU recipes."* Helpers: `find_npu_server()`, `evict_all_npu_servers()`, `find_flm_server_by_type()`.
- "Nuclear option" — on non-file-not-found load errors, evicts all models and retries.
- Load serialization via `load_mutex_` + `is_loading_` + cv: only one model loads at a time across the daemon.

### 3.5 What hybrid NPU+iGPU actually means inside Lemonade

There is **no Lemonade-level hybrid execution**. The `ryzenai-llm` recipe spawns AMD's prebuilt `ryzenai-server` binary which itself uses OnnxRuntime-GenAI with the VitisAI EP and a custom op library (`onnx_custom_ops.dll` for hybrid models, `onnxruntime_vitis_ai_custom_ops.dll` for NPU-only). The actual NPU↔iGPU partitioning happens *inside that subprocess*, opaque to Lemonade. Lemonade's contribution is: route the request to the right subprocess, manage its lifecycle, prevent another NPU model from stealing the device.

From AMD's hybrid OGA technical article: *"The prefill phase is compute-intensive and benefits from running on the NPU with its high AI Engine capability at up to 50 TOPS, while the decode phase is memory intensive and is executed on the iGPU, which is well-suited for handling high-bandwidth operations."* This NPU-prefill / iGPU-decode split is the heart of the marketing message but it's an OnnxRuntime-GenAI/VitisAI EP implementation detail, not Lemonade's.

### 3.6 vLLM as the Strix Halo escape hatch

vLLM is listed as **experimental** and the README explicitly notes *"validated only on gfx1151 (Strix Halo)"*. This is significant: AMD/Lemonade are using vLLM ROCm (Linux only) as the high-end Strix Halo path for serving — particularly relevant for MoE/large models that benefit from continuous batching. This is also a static routing decision; the user opts in via the recipe.

---

## 4. Hybrid execution mechanics (concrete techniques observed)

### 4.1 OnnxRuntime-GenAI hybrid mode (the AMD marketing story)

**Where it lives**: OnnxRuntime-GenAI + AMD VitisAI Execution Provider. Lemonade just spawns `ryzenai-server`.
**Processors supported**: STX (Strix Point), KRK (Krackan). **Strix Halo NOT listed** in `ryzenai.docs.amd.com 1.7.1`.
**Quantization**: AWQ INT4 weights with BF16 activations is the published format. Pre-optimized model zoo includes Llama-2/3, Mistral, DeepSeek-distill, Qwen2/2.5/3, Gemma-2, Phi-3/3.5/4.
**Partition mechanism**: ONNX graph rewriting at model-prep time (`oga_model_prepare` tool) marks ops as NPU-eligible (matmul-heavy prefill GEMMs go to XDNA2 via VitisAI EP; everything else, plus decode-heavy KV-cache attention, runs on iGPU via the standard DML/CUDA-equivalent ORT GPU EP).
**Long context**: `genai_config.json` gets `"hybrid_opt_chunk_context": "1"`, `"chunk_size": 2048` — implying chunked prefill to fit NPU SRAM/working set.
**Memory model**: Documentation does not detail zero-copy NPU↔iGPU. The pragmatic reality on a Windows AI PC is that NPU buffers are XRT-managed and iGPU buffers are DXGI/D3D12 (or HIP on ROCm) — crossing requires staging copies through system RAM. **This is a published optimization opportunity** that AMD has not described publicly.

### 4.2 Strix Halo unified memory exploitation (the *real* hybrid story for our target hardware)

This is **not** GAIA/Lemonade technology — it's llama.cpp + amdgpu kernel driver behavior, which Lemonade just inherits by spawning llama.cpp. But these are the techniques that matter for dotLLM on the same silicon.

**Hardware facts (from llm-tracker.info and AMD developer articles)**:
- Radeon 8060S: 40 RDNA3.5 CUs, peak ~59.4 FP16 TFLOPS (with WMMA/wave32).
- DDR5-8000, 256-bit bus → 256 GB/s theoretical, ~212 GB/s measured via `rocm_bandwidth_test`.
- Decode is **memory-bandwidth bound**: all backends (Vulkan, HIP, CPU) converge at ~50 tok/s for Llama-2-7B-Q4_0 decode regardless of kernel.
- Prefill is **compute-bound**: Vulkan ~884 tok/s vs HIP ~349-986 tok/s (HIP requires rocWMMA build for parity); CPU only ~294 tok/s for pp512.

**Unified memory tricks (Linux)**:
- Boot params to push iGPU-accessible memory beyond default split:
  `iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856` (reserves min 4 GiB for OS, max 124 GiB for iGPU)
- GART (~8 GB framebuffer) vs GTT (~110+ GB, pageable from system RAM); models load entirely into GTT — Llama-4-Scout 109B/17B-active at 57.93 GiB loads with no transfers.
- **There is no explicit zero-copy**. The OS demand-paging plus `mmap` of GGUF is the only mechanism. No host-pinned / unified-virtual-addressing trick exists today in the public llama.cpp + amdgpu path.

**Known bug demonstrating the opportunity**: ggml-org/llama.cpp issue #18011 — *"ROCm model loading with Strix Halo (Ryzen 395) always dumps KV cache into shared memory"* — confirms that on `uma: 1` APUs the ROCm/HIP backend mis-allocates KV cache to GTT rather than VRAM, hurting perf at long context. Vulkan does not have this bug. This is precisely the kind of subtlety dotLLM's Vulkan backend can sidestep with explicit memory-domain selection.

**Backend recommendation per workload (community consensus)**:
- One-shot / short context → Vulkan + Flash Attention.
- Long sustained chat → HIP + rocWMMA + Flash Attention (needs custom rocWMMA build; standard ROCm packages lack gfx1151).
- Reality check: Vulkan is winning on Strix Halo today.

### 4.3 NPU usage for LLMs on Strix Halo: status

Strix Halo *has* an XDNA2 NPU (50 TOPS class). But:
- AMD's official Ryzen AI 1.7.1 OGA hybrid flow lists STX/KRK only — **not Strix Halo**.
- llm-tracker explicitly says NPU for LLMs is "unexplored territory" on Strix Halo.
- Lemonade's `RECIPE_DEFS` does include `flm` (FastFlowLM) and `ryzenai-llm` for `XDNA2` family without OS-specific gfx restrictions — so on a Strix Halo Windows machine you *can* run `flm` on the XDNA2 NPU, but it's a separate model path (FLM format), not a hybrid co-execution with the iGPU.

**Net**: On Strix Halo today, the practical play is **iGPU dominant, NPU secondary** — exactly the opposite of marketing implications. dotLLM should optimize for the iGPU path first.

### 4.4 Multi-node trillion-parameter Strix Halo cluster

AMD has published a Kimi-K2 / trillion-parameter LLM article on a Strix Halo cluster ("Trillion-Parameter LLM on an AMD Ryzen AI Max+ Cluster", 2026). The web fetch failed but the title confirms AMD is positioning Strix Halo as a desktop/edge MoE host. Combined with the vLLM ROCm experimental recipe gated to gfx1151, the picture is: **Strix Halo + MoE + unified 96+ GB memory is AMD's published flagship use-case** for this silicon.

### 4.5 Summary of techniques observed

| Technique | Where implemented | dotLLM-relevant |
|---|---|---|
| OpenAI-compatible REST routing | Lemonade `Router` | Already in dotLLM |
| Recipe-based recipe→backend→device mapping | Lemonade `system_info.cpp::RECIPE_DEFS` | dotLLM has implicit equivalent in `IBackend` selection |
| NPU exclusivity enforcement | Lemonade `Router::evict_all_npu_servers` | N/A — dotLLM has no NPU yet |
| Subprocess-per-backend isolation | Lemonade `WrappedServer` + `ProcessManager` | dotLLM is in-process by design; this is a difference, not a gap |
| LRU model eviction per model type | Lemonade `Router::find_lru_server_by_type` | dotLLM has paged KV; lacks multi-model eviction |
| OGA hybrid prefill-NPU / decode-iGPU | ORT-GenAI + VitisAI EP (Strix Point only) | Not applicable to Strix Halo today |
| GTT-resident weights via mmap | amdgpu kernel + llama.cpp `mmap` GGUF | dotLLM **already uses `MemoryMappedFile`** for GGUF — same trick |
| Vulkan-preferred over ROCm on Strix Halo | llama.cpp community + Lemonade `RECIPE_DEFS` order | dotLLM is already Vulkan-first on Strix Halo — aligned |
| Flash Attention on iGPU | llama.cpp Vulkan + HIP rocWMMA | dotLLM Vulkan does not yet ship FA |
| Continuous batching (vLLM ROCm) | vLLM subprocess | dotLLM has paged KV + scheduler; equivalent in spirit, less battle-tested |

---

## 5. dotLLM gap analysis

What GAIA/Lemonade/the AMD stack do that dotLLM does not (as of this branch `feature/qwen3.6`):

| Capability | GAIA/Lemonade/AMD | dotLLM today | Gap severity |
|---|---|---|---|
| OpenAI/Ollama/Anthropic REST | Yes (port 13305) | OpenAI compat in `DotLLM.Server` | Low — extend if needed |
| Multi-backend routing at the server level | Yes (8 backends, recipe table) | One in-process backend per build | **Medium** — affects multi-tenant story |
| Hardware probing (CPU arch, gfx ISA, NPU family) | Yes (`SystemInfo` with WMI/sysctl/sysfs + XRT) | None — manual backend choice | **Medium** — UX/auto-config |
| ROCm/Vulkan auto-selection on gfx1151 | Yes (prefers Vulkan) | Vulkan already shipped; no ROCm yet | Low |
| NPU support (XDNA2) | Yes via FLM and ryzenai-llm subprocesses | **None** | **Low ROI for now** (see §6) |
| OGA hybrid NPU+iGPU prefill/decode split | Yes (Strix Point/Krackan only) | None | **Out of scope** (Strix Halo not supported by AMD's hybrid path anyway) |
| GTT-resident weights via `mmap` | Yes (via llama.cpp) | **Yes** — already uses `MemoryMappedFile` | Aligned |
| Explicit VRAM vs GTT placement for KV cache | Partial (llama.cpp ROCm has bugs, Vulkan ok) | dotLLM Vulkan uses default heap selection | **High ROI** to be explicit |
| Flash Attention on iGPU | llama.cpp Vulkan + FA, HIP + rocWMMA + FA | Not yet in Vulkan backend | **High ROI** for Strix Halo |
| Continuous batching / paged KV | vLLM ROCm experimental | Already in `DotLLM.Engine` | Aligned |
| Multi-model concurrent serving with LRU | Yes | Not implemented | Medium |
| Per-layer placement / hybrid CPU+iGPU | No (each backend self-contained) | No | Neither has it — green field |
| Zero-copy CPU↔iGPU via shared mmap | No published mechanism | No | **High ROI green-field** |

**Key insight**: Lemonade does NOT have a zero-copy CPU↔iGPU buffer story or a hybrid CPU-prefill / iGPU-decode split. It treats each backend subprocess as opaque. dotLLM, being in-process pure C# orchestration with native Vulkan/CUDA on the side, can **actually be more sophisticated than Lemonade** here because we own the whole stack in one address space.

---

## 6. Prioritized recommendations

### High ROI (1-2 weeks each, concrete wins on Strix Halo)

#### H1. Explicit VRAM-vs-GTT placement for KV cache and weights in the Vulkan backend
**Why**: The ROCm-on-Strix-Halo KV-cache bug (llama.cpp #18011) shows that *implicit* memory selection on a `uma: 1` APU mis-routes critical buffers to slow GTT-pageable memory. Vulkan exposes memory heaps explicitly (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` vs `HOST_VISIBLE`) — dotLLM should be explicit and not rely on heuristic. On Strix Halo all heaps are physically the same DDR5, but driver pinning and cacheability differ.
**Acceptance**: At 8K context, `tg8K` does not degrade more than 5 % vs `tg128` on a Q4_K_M 7B model.
**Files**: `src/DotLLM.Vulkan/MemoryAllocator.cs` (or equivalent), `src/DotLLM.Engine/KvCache/*`.

#### H2. Vulkan Flash Attention shader
**Why**: llama.cpp's Vulkan FA is the difference between 884 tok/s pp512 (with FA) and the ROCm-no-WMMA fallback. dotLLM's Vulkan backend already ships F32/F16/BF16 + K-quants + IQ from this session's parallel agents; adding a FA kernel closes the prefill gap.
**Acceptance**: `pp512` ≥ 80 % of llama.cpp Vulkan FA on Llama-2-7B Q4_0 / Q4_K_M Strix Halo benchmark.
**Files**: `src/DotLLM.Vulkan/Shaders/flash_attention.comp` (new), `src/DotLLM.Vulkan/Attention/VulkanFlashAttention.cs` (new), wire into `IAttentionStrategy`.

#### H3. Memory-mapped GGUF weights shared between CPU SIMD and Vulkan via host-visible buffers
**Why**: dotLLM already uses `MemoryMappedFile` for GGUF (per CLAUDE.md). On a unified-memory APU, the same physical pages can back both a CPU `Span<byte>` and a Vulkan `VkBuffer` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT` (or `VK_KHR_external_memory_host`). Result: weight load is zero-copy on the iGPU path. llama.cpp does not do this today on the Vulkan path — this is **actual differentiation**.
**Acceptance**: Time-to-first-token for a 70B Q4_K_M model loaded from cold cache is dominated by disk I/O, not by an iGPU staging copy. Memory footprint shows weights counted once, not twice.
**Files**: `src/DotLLM.Models/GgufLoader.cs`, `src/DotLLM.Vulkan/HostVisibleBuffer.cs` (new), `src/DotLLM.Core/Tensors/MemoryDomain.cs` (new enum).
**Risk**: `VK_EXT_external_memory_host` driver support on amdvlk vs radv vs Windows AMD driver — needs feature probe + fallback.

#### H4. CPU prefill + iGPU decode coordinated handoff for short prompts
**Why**: For prompts < ~256 tokens, CPU prefill (with AVX-512 + Q4 dequant) is competitive with iGPU prefill *and* avoids the iGPU warm-up tax (Vulkan pipeline cache, descriptor binding). Then decode runs on iGPU where the memory bandwidth bias rewards it. Lemonade cannot do this because each backend is a separate subprocess. dotLLM can because all backends share an address space.
**Acceptance**: For 64-token prompts, end-to-end latency for first 32 generated tokens is ≥ 10 % lower than the pure-iGPU baseline.
**Files**: `src/DotLLM.Engine/Scheduling/HybridPrefillDecodeStrategy.cs` (new), `src/DotLLM.Core/Backends/IBackend.cs` (extend with `Capabilities.PrefersPrefill`/`PrefersDecode`).

### Medium ROI (2-4 weeks, architectural)

#### M1. `IBackend.PlacementHint` per-tensor or per-layer
Introduce explicit device placement decisions in `ModelConfig` / `IModelArchitecture`. Lets future work route the embedding+lm_head on iGPU, attention on CPU for batch=1, MoE experts on iGPU streamed from CPU mmap, etc. Lemonade lacks this entirely.

#### M2. Runtime micro-profiler that warms up each backend and picks
At startup, run a tiny GEMM + dequant + softmax probe on each available backend, store per-shape latency, and pick a placement plan. Lemonade's selection is purely static from `RECIPE_DEFS`; dotLLM can do adaptive selection in milliseconds.

#### M3. Multi-model LRU eviction in `DotLLM.Server`
Mirror Lemonade's `Router::find_lru_server_by_type` for embedding/reranking/LLM models loaded concurrently. Useful for hosting multiple models in a single server process (the current `DotLLM.Sample.Server` is single-model).

#### M4. Hardware probing via `SystemInfo` analog
Detect CPU AVX-512 / VNNI / AMX (Zen 5 has none of AMX), iGPU gfx ISA via Vulkan `VK_KHR_driver_properties` + `vendorID`/`deviceID`, NPU via `\\.\AmdNpu` Windows device + XRT availability. Build a dotLLM `HardwareCapabilities` record analogous to Lemonade's `system_info.cpp`.

### Low ROI / out of scope

- **XDNA2 NPU integration via VitisAI / IRON-MLIR**. Strix Halo NPU LLM story is unproven, AMD's own OGA hybrid does not list Strix Halo, FLM/RyzenAI-LLM are closed-source binaries from AMD (`lemonade-sdk/ryzenai-server` is a release-only repo, not source). Sinking weeks into a custom XDNA2 backend before AMD even ships a Linux-supported NPU LLM SDK for gfx1151 is wrong tradeoff.
- **Anthropic-API parity, Ollama-API parity**. Compatibility surface bloat. dotLLM should pick OpenAI and stop.
- **Whisper/Stable-Diffusion/Kokoro backends**. Out of scope per CLAUDE.md (dotLLM is transformer LLMs).
- **Cloning lemonade's subprocess-per-backend pattern**. Anti-pattern for an in-process .NET host where pluggable `IBackend` is cleaner.

---

## 7. Source list

All accessed on **2026-05-14**:

- AMD GAIA repository — `https://github.com/amd/gaia` (README, license MIT)
- lemonade-sdk repository — `https://github.com/lemonade-sdk/lemonade` (README, `AGENTS.md`, `DESIGN.md`)
- lemonade source files inspected via `gh api`:
  - `src/cpp/server/router.cpp`
  - `src/cpp/server/system_info.cpp`
  - `src/cpp/server/backend_manager.cpp`
  - `src/cpp/server/backends/llamacpp_server.cpp`
  - `src/cpp/server/backends/fastflowlm_server.cpp`
  - `src/cpp/server/backends/ryzenaiserver.cpp`
  - `src/cpp/include/lemon/model_types.h`
- AMD Ryzen AI 1.7.1 docs — `https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html` (OGA hybrid flow, "supports Strix and Krackan Point processors; Phoenix and Hawk not supported")
- AMD Ryzen AI 1.7.1 LLM overview — `https://ryzenai.docs.amd.com/en/latest/llm/overview.html` (deployment modes: NPU-only / Hybrid / GPU / CPU; supported models)
- AMD technical article — `https://www.amd.com/en/developer/resources/technical-articles/gaia-an-open-source-project-from-amd-for-running-local-llms-on-ryzen-ai.html`
- AMD technical article — `https://www.amd.com/en/developer/resources/technical-articles/2025/hybrid-npu-igpu-optimized-agent-on-amd-ryzen-ai-powered-pc-.html` (prefill on NPU 50 TOPS, decode on iGPU memory-bound)
- AMD technical article — `https://www.amd.com/en/developer/resources/technical-articles/2025/rethinking-local-ai-lemonade-servers-python-advantage.html`
- AMD technical article (2026) — `https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html` (trillion-parameter LLM on Strix Halo cluster, headline only — fetch failed)
- llm-tracker Strix Halo deep-dive — `https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance` (Vulkan vs HIP perf, GTT/GART config, rocWMMA gfx1151 PR #538)
- llm-tracker overview — `https://llm-tracker.info/_TOORG/Strix-Halo` (212 GB/s measured BW, backend recommendation matrix)
- llama.cpp Vulkan discussion — `https://github.com/ggml-org/llama.cpp/discussions/10879`
- llama.cpp ROCm discussion — `https://github.com/ggml-org/llama.cpp/discussions/15021`
- llama.cpp issue #18011 — `https://github.com/ggml-org/llama.cpp/issues/18011` (ROCm KV cache mis-allocation to shared memory on gfx1151)
- ollama issue #15601 — `https://github.com/ollama/ollama/issues/15601` (Vulkan Wave32 FA gap)
- kyuz0 strix halo toolboxes — `https://github.com/kyuz0/amd-strix-halo-toolboxes` and benchmark grid `https://kyuz0.github.io/amd-strix-halo-toolboxes/`
- InfoWorld Lemonade review — `https://www.infoworld.com/article/4169474/first-look-lemonade-serves-up-local-ai-with-limitations.html`

**Notes on source quality**: The primary sources here are the `lemonade-sdk/lemonade` C++ source (`router.cpp`, `system_info.cpp`, backend implementations, `AGENTS.md`) — these are load-bearing and authoritative. Secondary sources are AMD's own developer articles for the OGA hybrid story (which is Strix Point / Krackan, not Strix Halo). Tertiary are community benchmarks (llm-tracker, kyuz0) for the Strix-Halo-specific Vulkan-vs-ROCm performance reality. The AMD trillion-parameter-cluster article is cited by title only; its content could not be retrieved within the time budget but the title alone informs the strategic positioning of Strix Halo as MoE-host silicon.
