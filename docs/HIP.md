# Experimental HIP / ROCm backend

`run`, `chat` and `serve` accept `--device hip` (HIP device 0) or `--device hip:N`.
The implementation supports dense Llama GGUF checkpoints with F16, Q4_0 or Q8_0 matrices,
and local HuggingFace SafeTensors checkpoints with F16, BF16, validated AWQ GEMM4 or
legacy GPTQ4 matrices. HF normalization weights may be F16, BF16 or F32; F32 norms
are narrowed on the GPU during load. BF16 weights retain their original bits. Quantized matrices
stay packed, activations stay F16, and reductions accumulate in registers at F32.
The final logit row widens to the existing `IModel` CPU-sampling F32 result contract
(`4 * vocabulary` bytes, 196,608 bytes for the qualified SmolLM2 checkpoint).

HIP is **explicit-only**. `--device auto` continues to try CUDA, Vulkan, then CPU.
An explicit HIP request fails if the runtime, selected device, code objects, model
format or direct-loading path is unavailable. It never falls back to CPU.

For reproducible build, kernel, model, application and Linux checks, use the
[integration testing handbook](HIP_TESTING.md). A self-contained
[adversarial review prompt](HIP_ADVERSARIAL_REVIEW.md) is available for an independent agent.

## Experimental Q8 KV opt-in

F16 remains the default. Selecting `q8_0` for either K or V additionally requires
`DOTLLM_HIP_ALLOW_Q8_KV=1` in the process environment. This applies to `run`, `chat`,
`serve` and direct `HipTransformerModel.CreateKvCache` calls. Other values, including
`true`, do not opt in. Without acknowledgement, the request fails; it never silently
substitutes F16. The flag does not select a cache type or affect Q8 model weights.

```powershell
$env:DOTLLM_HIP_ALLOW_Q8_KV = '1'
dotllm run /path/to/model.gguf --device hip --cache-type-k q8_0 --cache-type-v q8_0 -p "Hello"
```

```bash
DOTLLM_HIP_ALLOW_Q8_KV=1 dotllm run /path/to/model.gguf --device hip --cache-type-k q8_0 --cache-type-v q8_0 -p "Hello"
```

Q8 compresses each 32-value KV block into 32 signed bytes and a half-precision scale.
Rounding loses information that subsequent attention layers cannot recover. The
measured 1K/2K/4K/8K histories failed the original relative-L2 and KL gates with Q8
on every window; F16 passed. Across 512 scored predictions, Q8 changed 8 top-1
choices and increased combined perplexity by about 0.155%, while reducing owned
8K cache bytes from 188,989,440 to 100,515,840. These small aggregate differences
do not make Q8 equivalent to F16. CPU Q8 also failed; the evidence does not isolate
a HIP-specific defect. Mixed F16/Q8 caches require the same acknowledgement and
have no long-context quality qualification. See the [measurements and limits](PRECISION.md#actual-1k8k-histories).

## Current boundary

- Windows x64 is hardware-qualified with AMD HIP SDK 6.4.2 and a compatible display driver.
  Linux x64 has an AMD Infinity Storage loading implementation; see the prerequisites below.
  Linux compilation and software checks do not establish physical GPU/storage qualification.
- Native kernels and direct loading have been exercised on an RX 6900 XT (`gfx1030`).
  The distributed code objects also contain `gfx1100`; that target is not hardware-qualified here.
  Enumeration alone does not establish inference support: for example, `gfx1036` has no bundled image.
- Dense Llama, causal GQA/MHA, RMSNorm, SiLU and standard or linear RoPE.
  Unsupported geometry or architecture features fail before weight allocation.
- GGUF matrices support F16, Q4_0 and Q8_0. KV pages independently support F16 or Q8_0
  for keys and values; Q8_0 requires head dimensions divisible by 32. Q4_0 KV, BF16 GGUF,
  nonzero bias tensors, MoE, recurrent/hybrid architectures, LoRA and MTP are unsupported.
- All weights stay on the selected GPU. Partial `--gpu-layers` and `--gpu-layers 0`
  are rejected. Omit this flag or use `-1` for all layers.
- HF SafeTensors uses the existing single-file/sharded reader and HF config/tokenizer
  extraction. It accepts F16/BF16 matrices and F16/BF16/F32 norms. Raw HF Q/K rows use NeoX
  rotate-half RoPE; GGUF-converted Llama Q/K rows normally use adjacent pairs.
  Packed AWQ/GPTQ follows the same validated subset as CPU: separate dense Llama projections,
  four-bit producer layouts, F16 scales, supported group sizes and validated GPTQ group indices.
  Original qweight/qzeros/scales/g_idx file ranges load directly and stay packed on device.
- `bench`, `perplexity` and `embed` have separate execution paths and do not accept HIP yet.
- Linux requires the AIS direct-read extension, a working amdgpu/KFD P2PDMA path and
  a compatible model volume; no POSIX read-then-upload fallback is provided.

This is a correctness-oriented initial path, not a throughput claim. Prefill runs
tokens sequentially, with a preferred cancellation chunk of 16 tokens. Only the
last logit row is returned for multi-token prefill. Forward calls share scratch
and are serialized; each request owns independent paged KV. The server uses its
single-request generator, not the continuous batch scheduler.

## Run it

Keep checkpoints outside the repository. Point the process at the installed SDK
when its libraries are not already on the search path:

```powershell
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\6.4'
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"
dotllm run E:\models\llama-f16.gguf --device hip:0 -p 'Hello' -n 32
dotllm chat E:\models\llama-f16.gguf --device hip:0
dotllm serve E:\models\llama-f16.gguf --device hip:0
$env:DOTLLM_HIP_ALLOW_Q8_KV = '1' # Required for the following Q8 KV request.
dotllm run E:\models\llama-q4_0.gguf --device hip:0 --cache-type-k q8_0 --cache-type-v q8_0 -p 'Hello'
dotllm serve E:\models\hf-llama-f16 --device hip:0
```

K/V defaults to F16. Each side accepts `f16` or `q8_0`, including mixed K/V selections.
F32, Q4_0, TurboQuant and a mixed-precision window are rejected. HIP always uses
its own pages allocated as the sequence grows, regardless of generic CPU cache flags.
Actual 1K/2K/4K/8K probes pass the original logit-error gates with F16 KV; Q8 fails
those gates at every length. Q8 saves memory but is not qualified as F16-equivalent
quality. See the measured scoring windows and memory results in [PRECISION.md](PRECISION.md#actual-1k8k-histories).

HF directories must contain `config.json`, tokenizer assets and either a SafeTensors
file or `model.safetensors.index.json`. `auto` chooses CPU for HF; HIP requires an
explicit device. Declared HF chat templates are validated before weights load, and
named `tool_use` templates are selected when a request supplies tools. HF application
loading rejects unsupported overlays, RoPE/context overrides, expert offload and
separate MTP/DFlash heads. There is no runtime BF16-to-F16 checkpoint conversion.

`GET /v1/devices` distinguishes runtime availability from inference readiness.
HIP entries report `servable` and a `note` per device. Readiness checks load the
complete dense kernel manifest on that device and initialize its platform loading path.
Missing assets or an unsupported device remain visible but unselectable.
This preflight does not promise that a particular model will fit in free memory.
The WebUI offers only servable devices and leaves HIP as an explicit choice.

## Direct disk-to-device loading

On Windows, `HipDirectWeightStore` routes file ranges through DirectStorage into a shared
D3D12 default heap, then imports that heap with `hipImportExternalMemory` and maps
the buffer. The HIP device LUID selects the matching DXGI adapter. There is no
application host weight buffer and no host-copy fallback.

Each returned `HipDirectWeightBuffer` owns its D3D12 resource/heap and HIP external
memory import. Its `ByteLength` is payload size; `AllocatedBytes` includes 64 KiB
heap alignment. A buffer can outlive the loading session and is independently
disposable. Temporary F32 norm imports are released after conversion to F16.

DirectStorage itself uses internal staging (32 MiB by default); “no application
host staging” does not mean that the OS or driver allocates no host memory.
Loads validate file ranges before allocating, wait for completion, inspect the
DirectStorage error record, and clean up failed imports and sessions.

The DirectStorage DLLs are **not redistributed in releases or NuGet packages**.
Use the existing interactive installer described in [GPU.md](GPU.md#windows-directstorage-weight-load-vulkan-1304),
or supply `dstorage.dll` and `dstoragecore.dll` beside the executable, in
`DOTLLM_DIRECTSTORAGE_DIR`, or in `~/.dotllm/native/directstorage/1.3.0/x64`.
HIP direct loading is required and does not use the Vulkan opt-in environment flag.

### Linux AMD Infinity Storage

The Linux x64 path resolves `hipAmdFileRead` through `hipGetProcAddress`, following
the [AMD hipFile fastpath implementation](https://github.com/ROCm/rocm-systems/blob/328c911328af1720fa2567a922775456f72fe9d5/projects/hipfile/src/amd_detail/hip.cpp).
It calls the direct extension itself, so hipFile's automatic host-buffer fallback
cannot run. No extra native shim or libhipfile dependency is required for inference.

Model files are opened read-only with `O_DIRECT` on ext4/XFS. `statx` supplies direct-I/O
alignment; reads also respect older AIS drivers' 4 KiB requirement. An aligned device
allocation covers each original file range, and the tensor view starts at its exact
byte offset. Padding remains on device; files are not modified, duplicated or transcoded.
`AllocatedBytes` includes padding. Reads are bounded to 32 MiB, and a successful short
EOF read is accepted only when it contains every requested tensor byte. Driver/storage
errors, missing bytes and incompatible topology abort loading and free the allocation.
Buffers outlive the loader and free their original allocation, not the shifted view.

Prerequisites are more specific than a ROCm installation:

- Native Linux x64 with HIP GPU access and a runtime exposing `hipAmdFileRead` (the
  ABI is present in the installed ROCm 7.2.4 headers).
- An amdgpu/KFD driver with AIS support and kernel PCI P2PDMA support.
- A storage-to-GPU topology accepted by that driver, and a model file on a compatible
  ext4/XFS volume. Use a direct NVMe partition for initial qualification; encrypted,
  LVM, RAID, virtual disks and network mounts are not assumed to work.
- Sufficient dedicated VRAM for the model, aligned ranges, activations and live KV.

Run AMD's [ais-check](https://rocm.docs.amd.com/projects/hipFile/en/latest/how-to/checking-system-compatibility.html)
from a hipFile installation to check the kernel, runtime, amdgpu and model volume.
Then run `rocminfo` and the actual direct-read test. The library being present or
`hipcc` compiling successfully is insufficient. Older drivers that reject EOF-tail
reads fail explicitly; the application does not enable compatibility I/O to hide this.

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
bash native/hip/build.sh
dotnet build src/DotLLM.Hip -c Release -p:RequireHipKernels=true
dotllm run "$HOME/.dotllm/models/ORG/MODEL/model.gguf" --device hip:0 -p Hello -n 16
# Hardware qualification creates a temporary byte fixture on the selected model volume:
DOTLLM_TEST_HIP_LINUX_DIR=/path/on/ais-volume dotnet test tests/DotLLM.Tests.Unit -c Release \
  --filter FullyQualifiedName~LinuxDirectRead_ExactTailAndIndependentBufferLifetime
```

### Hyper-V build VM

A build/CPU-test VM on a 64 GB Windows host can use Generation 2, Ubuntu 24.04 LTS,
8 virtual CPUs, 16 GB RAM, a 120 GB dynamically expanding VHDX, networking and OpenSSH.
Use the Microsoft UEFI Certificate Authority Secure Boot template for Ubuntu.
Nested virtualization is unnecessary; leave it off. A normal virtual display device
does not expose an AMD compute GPU or a direct NVMe-to-GPU path.

Microsoft does [not support DDA/GPU-P on Windows 11 Pro](https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/troubleshoot-hyper-v-gpu-assignment-partitioning-passthrough-issues).
Therefore that VM is useful for Linux builds and CPU tests, but does not qualify HIP
inference on the RX 6900 XT. GPU tests need native Linux or a separately qualified
passthrough host/GPU/storage combination. Do not enable nested virtualization as a
substitute for GPU passthrough. The RX 6900 XT also is not listed in AMD's pinned
[ROCm 7.2 Radeon Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html);
its operation on a particular Linux stack still needs measurement.

## Native code and packaging

The managed backend calls the HIP runtime through source-generated P/Invoke.
On Windows HIP 6.4 uses `amdhip64_6.dll`; the resolver also supports the unversioned
runtime name and Linux SONAMEs. The dense path uses its own matrix/vector kernels;
it does not require hipBLAS. Separate hipBLAS bindings are available and tested for
future GEMM work, but missing hipBLAS does not prevent this dense path from running.

`native/hip/kernels/dense_f16.hip` provides embedding, F32-to-F16 conversion, RMSNorm,
linear projection, residual addition, SwiGLU, RoPE, KV page writes and paged attention.
`rmsnorm.hip` retains the standalone RMSNorm pipeline. `hipcc --genco` compiles fat
code objects with `gfx1030` and `gfx1100` images. There is no custom native shim.

```powershell
# Run from a VS x64 C++ developer environment with the HIP SDK on PATH.
pwsh native/hip/build.ps1
dotnet build src/DotLLM.Hip
```

`native/hip/build.sh` supplies the equivalent build on a ROCm development host.
Set `HIP_ARCHS` for the scripts, or `HipOffloadArchs` for MSBuild, to change targets.
`RequireHipKernels=true` makes native validation fail when compilation is required
but the compiler is missing. Ordinary consumers use the precompiled repository
code objects without installing a compiler.

Code objects stay outside single-file executables in `co/` beside the binary.
Native archives and the dotnet-tool package must contain `dense_f16.co` and
`rmsnorm.co`; release checks fail if either payload is absent. AMD runtime libraries
remain system prerequisites.

## Validation

The HIP test collection runs serially. Native tests compare results with independent
CPU formulas, including noncontiguous paged attention and all hipBLAS transpose
combinations. Direct-loading tests compare exact random file bytes at unaligned
offsets and across a 32 MiB request boundary, plus range errors and independent
buffer/session lifetimes. A deliberate one-byte source-offset mutation fails the
byte-identity tests; restoring it passes.

Device-selection tests cover missing runtime/assets, unsupported devices, invalid
ordinals, memory admission, load failures without CPU fallback, and the F16/Q8_0 cache
boundary. Application tests cover HF source ownership through replacement and unload,
declared templates, selected-device forwarding and cancellation releasing device pages.
End-to-end model and application qualification is tracked separately
in [issue #976](https://github.com/jamesburton/dotLLM/issues/976).

Linux software qualification on Ubuntu with ROCm 7.2.4 includes both compiled GPU
targets, installed-header ABI checks, a published Linux CLI and a CPU model smoke run.
`bash tests/scripts/hip-linux-direct-io/run.sh` executes 54 assertions through the real
Linux loader using a process-local simulated HIP runtime and actual `O_DIRECT` file reads.
It checks byte identity, padding, chunking, ownership and injected failures. Deliberately
ignoring the tensor displacement or storage errors fails the checks. This does not
exercise physical AIS transfers or Linux GPU model numerics; use the hardware test above
on a compatible storage/GPU system before treating that path as qualified.
