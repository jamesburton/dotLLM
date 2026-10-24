# CONTINUE_ON_CUDA.md — Pickup instructions for CUDA work

**Target audience:** an engineer with an NVIDIA GPU available, continuing
dotLLM's CUDA backend from where `feature/mamba-3` left off. The Ryzen dev
machine this branch was primarily developed on has only an AMD Radeon 8060S
iGPU — the CUDA path has the scaffolding in place but cannot be executed
there. All CUDA tests are present and skip cleanly without a device.

Branch: `feature/mamba-3`. Checkout commit: `576ea91` or later.

---

## What exists today

`src/DotLLM.Cuda/` is a full project with:

- **P/Invoke layer** against NVIDIA's CUDA Driver API (`libcuda.so` / `nvcuda.dll`) and cuBLAS — see `Interop/CudaDriverApi.cs`, `Interop/CublasApi.cs`. No custom C shared library; we call NVIDIA's system libs directly. License-clean (no ManagedCuda dep).
- **Model stack**: `CudaTransformerModel`, `CudaWeights`, `CudaForwardState`, `CudaKvCache`, `CudaQuantizedKvCache`.
- **Kernel launcher**: `CudaModule`, `CudaKernels`, `CudaGemm`.
- **PTX kernel catalog** (`native/kernels/*.cu`, compiled via `nvcc -ptx`): RMSNorm, RoPE (Norm + NeoX), Attention (naive), SwiGLU, Add, BiasAdd, Softmax, Embedding (F32/F16/Q8_0), Dequant (Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K), Quantized GEMV (Q8_0/Q4_K/Q6_K), Conversion (F16↔F32).
- **cuBLAS GEMM** for prefill path (FP16 with Tensor Core auto-dispatch when dims are ×8).
- **36 skipped unit tests** in `tests/DotLLM.Tests.Unit/Cuda/` — all runtime-probe CUDA availability and skip gracefully when no GPU is present.

See `docs/CUDA.md` (~700 lines, authoritative design doc) and `docs/GPU.md`
(data-flow overview).

## What almost-certainly works vs what needs validation

| Area | Status | Likely effort on an NVIDIA box |
|---|---|---|
| Project builds on machines without CUDA | ✅ verified (all dev so far) | — |
| Project builds with CUDA Toolkit installed | likely (no blockers) | minutes |
| `nvcc -ptx` on each `native/kernels/*.cu` | likely | minutes (`native/build.sh`) |
| `CudaModule.LoadFromFile` picks up the PTX | likely | — |
| 36 skipped tests become runnable | ✅ expected | — |
| Individual kernel correctness vs CPU reference (`CudaKernelComparisonTests`) | not exercised | hours (expected to pass first try for most; any failure is informative) |
| Full forward-pass on GGUF (SmolLM-135M, TinyLlama) | not exercised | day-ish — may hit rough edges |
| cuBLAS path with Tensor Cores | not exercised | hours |
| Quantized GEMV decode path | not exercised | hours |
| MLA on CUDA | **not started** | weeks — Phase A/B/C CPU path is the oracle |
| MoE on CUDA | **not started** | weeks |
| Mamba-3 SSM on CUDA | **not started** | weeks |

## Step-by-step pickup

### 1. Clone + build on the CUDA machine
```bash
git clone https://github.com/jamesburton/dotLLM.git
cd dotLLM
git checkout feature/mamba-3
dotnet build --configuration Debug
```
Expected: 0 warnings, 0 errors. The build succeeds whether or not CUDA is
present — all CUDA interop is runtime-resolved.

### 2. Verify CUDA is visible
```bash
nvidia-smi                    # confirm GPU + driver ≥ 525
nvcc --version                # confirm CUDA Toolkit 12.x for PTX build
```

### 3. Build the PTX kernels
```bash
cd native
./build.sh                    # or build.ps1 on Windows
# Expected output: one PTX file per .cu in native/kernels/
ls native/ptx/
```
If any `.cu` fails to compile, it's almost certainly a nvcc version mismatch
or a missing header. Fix one at a time; the kernels are independent.

### 4. Run CUDA kernel comparison tests

These are the 36 currently-skipped tests. They build reference outputs on
CPU (via `DotLLM.Cpu.Kernels`) and compare against CUDA results.

```bash
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj \
  --filter "FullyQualifiedName~Cuda" \
  --logger "console;verbosity=normal"
```

Expected: most pass, some may need tolerance tuning (`5e-4` is the common
threshold; Q4_K dequant-then-matmul might want looser on F16 paths).

### 5. Wire `ModelLoader.LoadFromGguf(... deviceId = 0)` end-to-end
```csharp
using var gguf = GgufFile.Open("path/to/SmolLM-135M.Q8_0.gguf");
var (model, _, config) = ModelLoader.LoadFromGguf(gguf, threading: ...,
                                                   deviceId: 0);   // CUDA
using ITensor logits = model.Forward(tokenIds, positions, deviceId: 0);
```

Check: does `ModelLoader.LoadFromGguf` actually dispatch to `CudaTransformerModel`
when `deviceId >= 0`? As of commit `576ea91` the dispatch exists in the
code path but has not been exercised. If it throws `NotSupportedException`
or similar, the dispatch arm is the first thing to wire up.

### 6. Real-weight sanity

Use the same fixtures the CPU tests use:

```bash
# On your CUDA machine, place (or env-var) the checkpoints:
# C:/temp/dotllm-tinyllama/                 (or $DOTLLM_TINYLLAMA_CHECKPOINT_PATH)
# C:/temp/dotllm-phi35-mini/                (or $DOTLLM_PHI35_CHECKPOINT_PATH)
# C:/temp/dotllm-granite31-moe/             (or $DOTLLM_GRANITE3_CHECKPOINT_PATH)

dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj \
  --filter "FullyQualifiedName~LoadsAndForwardsEndToEnd"
```

Currently these dispatch to CPU (`deviceId: -1`). For CUDA, introduce a
`_Cuda` sibling test per arch once the dispatch works, keeping the CPU
versions as oracles.

### 7. PyTorch reference gate

The CPU path already matches HF 5/5 argmax on 4 architectures (Qwen / Phi
/ Llama / DeepSeek-V2-Lite). The reference JSONs are at
`tests/DotLLM.Tests.Integration/Models/Loaders/references/`. Mirror the
`*_LogitsMatchPyTorchReference` pattern for CUDA — same JSON, run CUDA
forward, diff. Target: `DriftTolerances.Tight` or looser only if observed
BF16-on-GPU-vs-F32-on-CPU drift warrants it.

### 8. Benchmark

`benchmarks/DotLLM.Benchmarks/InferenceBenchmarks.cs` runs `prefill + decode`
on SmolLM-135M / Llama-3.2-1B / Llama-3.2-3B via the default CPU path. Add a
CUDA variant and compare. Baseline (CPU, this branch): see
`PLANS.md` → P1.3 pre-W2 section (~234 prefill / 17 decode tok/s on CPU
single-thread for SmolLM-135M Q8_0).

### 9. Hard architecture gaps to close on CUDA

These are structural — not bugs, just not-yet-done:

- **MLA on CUDA.** The CPU Phase A/B/C `MlaAttention` kernels are the
  numerical oracle (split-call matches single-call ≤ 1e-3 / 1e-4; F32+eager
  vs HF 5/5 argmax on DeepSeek-V2-Lite). Plan a port mirroring vLLM's
  MLA backend (prefill-expand / decode-absorbed split). The Phase B
  latent-cache layout (`MlaLatentKvState`) maps naturally to GPU HBM
  locality.
- **MoE on CUDA.** DeepSeek multi-shared-expert + Qwen-MoE + Granite-fused
  all need kernel ports. The Mixtral-style routing is the simplest first
  target.
- **Mamba-3 SSM on CUDA.** Selective scan + conv1d + MIMO — multi-month
  effort; low priority unless users ask.

### 10. Known secondary HF divergences (apply to both CPU and CUDA)

Tracked in our `P2.6` follow-ups, not blocking but worth fixing as long-context coverage grows:

1. **YaRN RoPE frequency rescaling** — parsed in `MlaConfig`, not applied.
   Matters for prompts > `original_max_position_embeddings` (4096 on V2-Lite).
   Active session at time of writing has an agent implementing this; check
   latest commits on `feature/mamba-3` for a `P2.6 fix: YaRN RoPE
   frequency rescaling` commit.
2. **`routed_scaling_factor`** — parsed, ignored. V2-Lite=1.0 no-op; V2-full
   and V3 use ≠1.0 with `norm_topk_prob=false`.
3. **`topk_method=group_limited_greedy`** — V3-only. V2 uses `greedy`.

### 11. Multi-GPU (NCCL)

`docs/MULTI_GPU.md` has the design. Not started.

## Contact points

- `docs/CUDA.md` — everything about the PTX architecture.
- `docs/GPU.md` — data-flow diagrams.
- `CLAUDE.md` — project coding conventions.
- `PLANS.md` — tracks outstanding work and past decisions.
- `docs/KV_CACHE.md` — Phase A/B/C MLA cache rationale (applies to GPU too).
- `docs/SUPPORTED_MODELS.md` — per-arch matrix with `Verified on` evidence.

## Reciprocal — what runs on this Ryzen box

### Backends verified on AMD Radeon 8060S iGPU (Strix Halo, gfx1151)

| Backend | Status on this iGPU | Notes |
|---|---|---|
| **CPU** (dotLLM.Cpu) | ✅ works | Default path; AVX-512F+CD+BW+DQ+VL+VBMI active at runtime. |
| **Vulkan** (dotLLM.Vulkan) | ✅ Add kernel verified end-to-end | Full LLM kernels not yet ported. Device reports `fp16: 1`, `bf16: 0`, `KHR_coopmat` matrix cores. |
| **HIP/ROCm** (dotLLM.Hip) | ⚠️ builds + initialises, `hipModuleLoadData` fails with `hipErrorInvalidKernelFile` | Known issue with gfx1151 iGPU on ROCm 7.1 Windows. Expected to work on discrete AMD GPUs (gfx1030/gfx1100). |
| **CUDA** (dotLLM.Cuda) | ❌ no NVIDIA GPU | That's why you exist. |

### Cross-library benchmarks (SmolLM-135M Q8_0, llama.cpp build d0a6dfeb2)

Reference numbers captured on the Ryzen + Radeon 8060S box for
cross-validation when the CUDA port is ready. Prompt sizes in tokens:
`pp` = prompt processing (prefill), `tg` = token generation (decode).

| Backend              | pp128 (tok/s) | pp512 (tok/s) | tg128 (tok/s) |
|----------------------|--------------:|--------------:|--------------:|
| **dotLLM CPU** (PLANS.md P1.3, SmolLM Q8_0, 20-tok decode) | ~234 | — | **~17** |
| llama.cpp `-ngl 0`   | 1465 ± 155 | 2897 ± 640 | 109 ± 10 |
| llama.cpp Vulkan `-ngl 99` (iGPU) | 15096 ± 689 | 25205 ± 3040 | **392 ± 45** |

Key gaps for the CPU path:
- dotLLM CPU decode is **~6.4× slower** than llama.cpp CPU decode.
- dotLLM CPU decode is **~23× slower** than llama.cpp Vulkan iGPU decode.

Candidate causes on our CPU path (tracked, not yet root-caused):
1. Single-thread by default; `ThreadingConfig.Parallel` exists but the
   `InferenceBenchmarks` run single-thread for stability.
2. No fused quantized-matmul (we dequantize → GEMM; llama.cpp MMQ fuses).
3. No SIMD in MLA (fixed in `c3f1c21` for MLA but not yet for the full
   GQA attention — though `Attention.cs` already uses `TensorPrimitives`).
4. Sampler overhead — llama.cpp's simple sampler is sub-microsecond;
   ours has more abstraction layers.

A CUDA comparison on an NVIDIA box should target **≥ llama.cpp's Vulkan
iGPU numbers** as a floor. Real CUDA on discrete Ampere/Ada/Hopper
should substantially exceed it.

### What's happening on the Ryzen box while you work

- **Vulkan backend** (`src/DotLLM.Vulkan/`) — working end-to-end on AMD
  Radeon 8060S iGPU (`AddKernel_ProducesElementwiseSum` passes). Scope
  is proof-of-pipeline; LLM-kernel port is the next task here.
- **HIP backend** (`src/DotLLM.Hip/`) — builds and initialises on Strix Halo
  iGPU (gfx1151) but `hipModuleLoadData` returns `hipErrorInvalidKernelFile`
  on ROCm 7.1 + iGPU; test skips cleanly. Expected to work on discrete AMD
  GPUs (gfx1030 / gfx1100) — validation left for a machine with one.
- **CPU kernel work** — MLA vectorisation, long-context tests, benchmark
  comparisons vs llama.cpp / Ollama (which already has a harness at
  `C:/Development/OllamaBenchmarks/` per the parent-dir scan).

## When you sync back

The two tracks diverge cleanly. The Ryzen box touches
`src/DotLLM.Vulkan/`, `src/DotLLM.Hip/`, `src/DotLLM.Cpu/Kernels/*`,
`src/DotLLM.Models/Architectures/Mla*.cs`, `tests/DotLLM.Tests.Unit/**`,
`tests/scripts/compare_logits_py_reference.py`. The CUDA box touches
`src/DotLLM.Cuda/`, `native/kernels/*.cu`, `native/build.*`,
`tests/DotLLM.Tests.Unit/Cuda/*`, `docs/CUDA.md`, `docs/GPU.md`. Minor
rebase expected on `dotLLM.slnx` project list; other than that the trees
should merge clean.
