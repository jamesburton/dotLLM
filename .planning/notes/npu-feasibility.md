# NPU feasibility: pure-native routes on XDNA2 (Strix Halo) and NPU 3720 (Meteor Lake)

**Issue**: #1626 · **Date**: 2026-11-04 · **Status**: research and PoC only. No product code.
**Builds on**: #1116 (design review), #1238 / #1261 / #1262 (Intel track, OpenVINO-based), #1285 (XDNA2 2-bit GEMV spike),
#1321 (Tev1 background classifier), #1476 (background embedder), and `gaia-lemonade-research.md`.

## Question

Can dotLLM drive each NPU from its **own** code: `[LibraryImport]` plus at most a thin C shim, with **no DirectML, no ONNX
Runtime and no OpenVINO runtime** in the product path? And what is each NPU plausible for?

The suggested table row "NPU | DirectML / ONNX Runtime | Planar 2-bit fixed-function tensor mapping" came from an agent
session, not from a measurement. It is **wrong on both counts**:

- **A pure route exists on both NPUs.** Neither needs DirectML or ORT.
- **Neither NPU has 2-bit fixed-function hardware.**
  - XDNA2 runs 2-bit as packed DMA plus an in-tile unpack to int8 inside a user-authored kernel (#1285).
  - The Intel NPU 3720 has no 2-bit or int4 compute at all (#1262: INT8 11.5 TOPS, FP16 5.7).

## Verdict

| | AMD XDNA2 (Strix Halo) | Intel NPU 3720 (Core Ultra 7 155H, `framework`) |
|---|---|---|
| Device node | `NPU Compute Accelerator Device`, class ComputeAccelerator, `PCI\VEN_1022&DEV_17F0`, service `IpuMcdmDriver`, driver **32.0.20102.4480** (`kipudrv.inf`), FW 1.1.2.65, 8 columns | `Intel(R) AI Boost`, `PCI\VEN_8086&DEV_7D1D`, driver **32.0.100.5540** (`npu.inf`) |
| Userspace shipped by the driver | `C:\Windows\System32\xrt_coreutil.dll` (XRT **2.21.0**, commit `15e6319b`), `xrt-smi.exe`, `xrt_core.dll` and validate artefacts in the DriverStore | `ze_loader.dll` (System32, 1.32), `npu_level_zero_umd.dll`, and **`openvino_intel_npu_compiler.dll` 2026.4.0.1, the compiler-in-driver** |
| Flat C API, P/Invoke-able | **Yes** (`xrtDeviceOpen`, `xrtBOAlloc`, `xrtBOAllocUserPtr`, `xrtBOMap`, `xrtBOSync`, `xrtXclbin*`, `xrtRun*`) | **Yes** (Level Zero `ze*`). Extensions: `ZE_extension_graph` v1.18, `ZE_extension_external_memmap_sysmem`, `ZE_extension_command_queue_npu`, `ZE_experimental_mutable_command_list` |
| Lowest route to run a custom kernel | **C++ only**: `xrt::hw_context(device, xclbin_uuid)` → `xrt::elf` / `xrt::module` → `xrt::ext::kernel(ctx, mod, "DPU")` → `xrt::run`. The legacy C `xrtDeviceLoadXclbin*` fails with `load_axlf: not supported`. Needs a ~60-line flat-C shim. | **No user kernels.** Compute runs only as a *graph*: `pfnCreate` on the graph extension with serialized OpenVINO IR (or a precompiled native blob), compiled by the in-driver compiler, then `pfnAppendGraphExecute` on a command list |
| Kernel authoring | IRON / mlir-aie + Peano (llvm-aie). **Native Windows is supported** (mlir-aie `buildHostWinNative`, Python 3.13, `xrt_windows_sdk.zip`), and #1285 used this route. Compile offline and ship the xclbin/ELF as checked-in artefacts, like our `.spv` | We would emit OpenVINO IR v11 (XML + weights) ourselves and hand it to the driver compiler. That is the only lever, and it is graph-level, not kernel-level |
| Verdict | **GO** for the #1285 track: proven from .NET today | **Conditional and low priority.** Pure L0 is possible but needs an in-house IR serializer to replace `openvino_c.dll` (#1261) |

The AMD route is exactly AMD's documented Windows host flow: a `.def` generated from the driver's `xrt_coreutil.dll`, then
`lib /def` for an import lib, plus the public XRT headers. Nothing extra has to be installed to *run* on the NPU. Only
kernel *authoring* needs the IRON toolchain.

## PoC results

Throwaway code, not committed. Scratch projects: `xrtprobe` (C API), `npushim` + `npurun` (C++ shim), and `zeprobe` (L0).

### Strix Halo XDNA2: .NET → `xrt_coreutil.dll` C API, no shim

| step | result |
|---|---|
| `xrtVersion*` | 2.21.0 |
| `xrtDeviceOpen(0)` | OK, 15–38 ms |
| `xrtBOAlloc(16 MiB, XRT_BO_FLAGS_HOST_ONLY, grp 0)` | OK, 1.1 ms. `flags=0` → "unsupported buffer type". `CACHEABLE` without a context → **access violation**, so don't |
| `xrtBOSync` to / from device, 16 MiB | 0.15–0.20 ms (a cache flush, not a copy), readback byte-exact |
| `xrtBOAllocUserPtr` over `NativeMemory.AlignedAlloc(4 KiB-aligned)` | OK, 1.9 ms. **`xrtBOMap` returns the same host pointer**, so the BO is zero-copy over our own memory |
| `xrtDeviceLoadXclbinHandle` | `load_axlf: not supported`. The NPU needs `hw_context` |

### Strix Halo XDNA2: .NET → C shim → XRT C++ (`hw_context` + ELF)

The driver's own `xrt-smi validate` artefacts drive this test. They are in `DriverStore\...\Runner\xrt_smi_strx.a`, a
Unix `ar` archive holding `validate.xclbin`, `nop.elf`, `gemm.xclbin` and `gemm.elf`.

| kernel | result |
|---|---|
| nop DPU kernel (`validate.xclbin` + `nop.elf`, 5 × 20-byte BOs) | setup (device + ctx + kernel + BOs) **46.6 ms**, first run 0.21 ms, **steady 295.5 µs per synchronous `start()`+`wait2()`** over 2000 runs |
| AMD int8 GEMM ELF (`gemm.xclbin` + `gemm.elf`) | **7.7 ms per run**, 200 runs (shape not published) |
| `xrt-smi validate` reference | GEMM **51.3 TOPS** (in-kernel cycle count), latency 376 µs, throughput 13.5k op/s (cmd-chained) |

The shim links against an import lib built from the driver DLL (541 exports) and compiles against XRT headers at the
driver's commit. It needs `/Zc:__cplusplus`, and a hand-written `xrt/detail/version-slim.h` because that header is
generated at build time. `xrt-smi validate` switches power mode to `performance` while it runs and restores `Default`
afterwards (checked).

**The dispatch floor is about 0.3 ms per synchronous call.** #1285 reported ~80–100 µs fixed overhead per call; that
figure likely amortised over queued runs. The 13.5k op/s cmd-chain figure (≈74 µs) agrees. Design for **one call per
decode step**, or per layer group, never per GEMV (the #1285 plan already says this).

### Meteor Lake NPU 3720 (`framework`): .NET → `ze_loader.dll`, no OpenVINO

| step | result |
|---|---|
| `zeInit(ZE_INIT_FLAG_VPU_ONLY)` | OK, 198 ms. One driver, one device of type VPU (5): `Intel(R) AI Boost`, id 0x7D1D, maxMemAlloc 48 GiB |
| `zeContextCreate`, `zeMemAllocHost` / `Device` / `Shared` (16 MiB) | all OK. Device alloc returns an NPU VA (`0x202000000`) |
| 16 MiB host → device → host copy on queue group 0 (compute + copy) | first submit **6.5 s** (NPU power-up / FW boot), then **3.5–5.6 ms**, byte-exact |
| empty command list (barrier only), submit + `zeCommandQueueSynchronize` | **1.69 ms per submit**. The OpenVINO path in #1238 measured ~0.4 ms per call, so raw sync submission is not free; async or mutable lists are the follow-up |

## What each workload fits

Shared constraint: the NPU, iGPU and CPU sit on one DRAM bus (Strix ~256 GB/s; MTL ~90 GB/s). The NPU's own read ceiling
is **~52–56 GB/s on XDNA2** (#1285) and **~13–35 GB/s on the 3720** (#1238). Running it alongside the iGPU therefore
**adds compute and frees the iGPU, but adds no bandwidth.**

| workload | XDNA2 | NPU 3720 |
|---|---|---|
| **Ternary residual adapter** (`TernaryAdapterKernel.cs`, rank 32: Down `R×D`, Up `D×R`, about 2·32·D MACs per token, ~16 KB of 2-bit weights at D=4096) | **No.** The work is microseconds on the CPU and below the 0.1–0.3 ms dispatch floor. Only viable *inside* a fused NPU decode step whose host layer is already on the NPU | **No.** Same reason, plus no 2-bit compute: unpacking to i4/i8 would be an expanded copy, which breaks rule 2 |
| **Planar 2-bit / PQ2 GEMV** (`MatMul.PQ2Planar.cs` layout) | **Yes. This is the #1285 result:** 2-bit packed weights stream at ~52 GB/s with a free in-tile unpack to int8 (16 MB GEMV in ~320 µs). No expansion. A planar layout maps naturally onto per-plane DMA descriptors | No native 2-bit |
| **Prefill GEMMs** | Possible, but not a speed win. 51 int8 TOPS peak vs the 8060S iGPU's ~59 FP16 TFLOPS WMMA peak (`gaia-lemonade-research.md`), on the same bus. The case is a *concurrent* prefill tier, or power. Needs M=64+ tilings (#1285 step 6) | No. Block-32 prefill has no good NPU form (#1238: 180–1500 ms split-per-group vs 12.9 ms per-channel) |
| **Background embedder / classifier** (#1476 BERT-class, #1321 Tev1 single-prefill) | **Best fit.** Fixed shapes, batchable, latency-tolerant, and it frees the iGPU. Costs one setup (~47 ms per context) plus one call per batch | Plausible via the OpenVINO-IR graph route (#1238 measured a ~0.35 J/token marginal cost, about half of Arc). Pure-L0 needs our own IR emitter |

### Zero-copy / UMA notes

- **XDNA2**:
  - `xrtBOAllocUserPtr` over our own 4 KiB-aligned host memory is zero-copy: the mapped pointer equals the host pointer,
    and a 16 MiB sync costs ~0.15 ms.
  - **Unverified**: user-ptr BOs over a *file-backed* GGUF mmap region (the rule 3 route), and whether the NPU's DMA
    reads those pages at full rate. Intel's equivalent (`external_memmap_sysmem`) ran at only 8.4 GB/s vs 35 GB/s from
    L0 host memory (#1238).
- **NPU 3720**: the zero-copy mmap import works, but it is slow. #1238 signed off one group-major copy in L0 host memory
  as a rule 3 deviation.
- **Neither NPU shares a buffer handle with Vulkan directly.** An NPU↔iGPU hand-off goes through host memory that both
  map: the same physical pages, but no single shared allocation API. So:
  - Plan hand-offs at **step granularity**, such as KV written by prefill and read by decode.
  - Never plan per-op ping-pong between the two.

## Recommended next steps

1. **AMD (#1285 integration)**:
   - Productise the shim as `native/xdna/`: flat C over `xrt::hw_context` / `xrt::ext::kernel` / `xrt::run`, opaque
     handles, coarse calls.
   - Generate the import lib from the installed `xrt_coreutil.dll` at build time, or vendor a `.def`.
   - Skip cleanly when `xrt_coreutil.dll` or the device is absent.
2. **AMD**: the #1285 spike notes and kernels (`.docs/npu/STRIX_XDNA2_SPIKE.md`, `.docs/npu/strix/`) are **no longer on
   Strix's disk**. Recover or rebuild the 2-bit GEMV xclbin/ELF. Then drive it through the shim from .NET and confirm
   parity plus the ~320 µs figure.
3. **AMD**: measure (a) `xrt::runlist` / cmd-chain per-call cost from the shim, and (b) a user-ptr BO over a GGUF mmap
   region: does it work, and how fast is the DMA read?
4. **Intel**: decide whether #1261 keeps `openvino_c.dll` or moves to raw L0 + `ZE_extension_graph` with an in-house IR v11
   emitter. Pure L0 removes a ~100 MB runtime dependency but makes dotLLM own IR serialization against a compiler that
   versions with the driver. Low priority: the 3720's 11.5 int8 TOPS and 1.7 ms raw submit floor limit it to the
   background-model role already scoped in #1238.
5. **Not worth pursuing**: running ternary adapters on either NPU as a standalone call, and NPU prefill on the 3720.

## Repro pointers

- Device and driver: `Get-PnpDevice -PresentOnly | ? FriendlyName -match 'NPU|AI Boost'`, and the
  `Get-CimInstance Win32_PnPSignedDriver` driver version.
- XRT: `& "<DriverStore>\kipudrv.inf_amd64_*\xrt-smi.exe" --batch examine` and `... validate --verbose`.
- Validate artefacts: copy `Runner\xrt_smi_strx.a` with PowerShell (a Git-bash `cp` truncated it to 8 KB), then extract
  with any `ar` reader. Ignore the nested member of the same name.
- Shim build:
  1. `dumpbin /exports xrt_coreutil.dll` → `.def` → `lib /def:xrt_coreutil.def /machine:x64`.
  2. `cl /std:c++17 /Zc:__cplusplus /EHsc /MD /LD`, with `-I` pointing at XRT `src/runtime_src/core/include` at commit
     `15e6319b`.
