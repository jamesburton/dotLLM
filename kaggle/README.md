# dotLLM dual-CUDA validation on Kaggle (issue #361)

Validate the cross-device KV handoff (`StagedKvHandoffTransfer`, #360) on **two real GPUs**
using a free Kaggle "GPU T4 ×2" session — the only dual-device hardware available to the project
(the Strix Halo dev box is a single iGPU; T5500 is a single RTX 3060).

## Why Kaggle fits
- **PTX, not a shared lib.** The CUDA backend compiles `.cu` → PTX (`compute_75`) loaded via the
  driver API. The **T4 is `sm_75`**, so the PTX runs natively — no JIT-arch mismatch.
- **.NET 10 installs without root** via `dotnet-install.sh --channel 10.0` into `$HOME`.
- **No P2P between the T4s** (no NVLink, PCIe topology) → a direct device↔device copy is unavailable,
  so the **device→host→device** staging path is the one you *must* use. That's exactly what
  `StagedKvHandoffTransfer` does, so this is a faithful test of the real mechanism.

## One-time Kaggle setup
1. New Notebook → **Settings → Accelerator → "GPU T4 ×2"**.
2. **Settings → Internet → On** (needed to fetch the SDK + clone the repo).
3. Open `dotllm-dual-cuda-validation.ipynb` (or paste the cells) and run top-to-bottom.

## What the notebook does (via `setup.sh`)
| Step | Command | Proves |
|------|---------|--------|
| `env` | versions + `nvidia-smi` | 2× T4 present, `nvcc` available |
| `dotnet` | install .NET 10 SDK into `$HOME` | runtime works headless/no-root |
| `ptx` | `native/build.sh` | kernels compile to PTX on the image |
| `build` | `dotnet build` | managed solution builds on Linux/.NET 10 |
| `test-cpu` | `DisaggregatedKvTransferTests` | **the staged seam is correct + the toolchain is green** |
| `test-cuda` | `CudaCrossDeviceKvTransferTests` | **prefill GPU0 → decode GPU1 token-parity** (needs the #361 CUDA impl) |

`test-cpu` already passes today — running it on Kaggle is the toolchain proof. `test-cuda` is the
goal and depends on the remaining #361 work below.

## Remaining #361 implementation (so `test-cuda` has something to run)
The multi-device primitives already exist (`CudaContext.Create(int deviceId)` + `MakeCurrent()`,
`CudaDevice.GetDeviceCount()`), so this is wiring, not new infrastructure:
1. `CudaKvCache : IHostStagedKvCache` — `DownloadLayer` (`cuMemcpyDtoH` the FP16 buffer → host,
   convert FP16→FP32) and `UploadLayer` (FP32→FP16 → `cuMemcpyHtoD`), each making the cache's own
   `CudaContext` current first (the cache must hold its context for cross-device correctness).
2. A `CudaCrossDeviceKvTransferTests` that creates two contexts (device 0 and 1), prefills on 0,
   hands off via `StagedKvHandoffTransfer`, decodes on 1, and asserts token-identical output to a
   single-device run. Skips when `CudaDevice.GetDeviceCount() < 2`.

## Dual-CUDA pipeline-parallel / layer-spanning (issue #367)

`dotllm-cuda-pipeline-spanning.ipynb` validates `CudaPipelineTransformerModel` — a transformer **split
across two CUDA devices** (stage-0 = layers `[0..K)` on GPU 0; stage-1 = layers `[K..L)` + final norm +
LM head on GPU 1), with the hidden state handed off **device0 → host (FP32) → device1**. It is the CUDA
mirror of the shipped Vulkan dual-device pipeline (#366), and lets a model span more VRAM than either T4
holds alone.

| Step | Command | Proves |
|------|---------|--------|
| `env`/`dotnet`/`ptx`/`build` | (as above) | toolchain green, driver-matched PTX |
| `test-pipeline` | `CudaPipelineParityTests` | split model == un-split `CudaTransformerModel` (prefill + decode) |

- The notebook clones **`issue/367-cuda-pipeline-parallel`** (override with `DOTLLM_BRANCH`).
- `CudaWeights.LoadFromGguf(..., firstLayer, numGpuLayers)` already windows each stage's device upload, and
  skips the output norm + LM head for the non-final (stage-0) window — so stage-0 holds embedding + its
  layers and stage-1 holds its layers + head, exactly the Vulkan split.
- **Single-device fallback:** the `PipelineModel_*` theories run both stages on GPU 0 (two contexts), so the
  core machinery is provable on one GPU; the `CrossDevicePipelineModel_*` theories need 2 GPUs and skip on a
  single-GPU box (`CudaDevice.GetDeviceCount() < 2`).
- Parity is FP16-precision (same as a single-device CUDA run); the FP32 hidden-state round-trip at the
  boundary is lossless, so the band is tight (abs 5e-3 / rel 5e-3).
- M-scope: dense / GQA causal (no MLA / MoE / gemma4 / graph-capture), matching #366.

## Notes / gotchas
- Kaggle sessions are time-boxed (~9–12 h) and idle-timeout; the build is incremental so re-runs are fast.
- The notebook clones the **fork** `jamesburton/dotLLM` by default (that's where the dev-track / #361
  branch lives — upstream `kkokosa/dotLLM` does not have it). Override with `DOTLLM_REPO` / `DOTLLM_BRANCH`.
  The fork must be reachable from Kaggle (public, or supply a token); set `DOTLLM_BRANCH=dev` to run only
  the CPU proof.
- FP16 round-trip: the CUDA cache is FP16, so cross-device parity is to FP16 precision (same as a
  single-device CUDA run) — compare against a CUDA single-device baseline, not the CPU FP32 oracle.
