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
- Linux runtime probing and native compilation exist, but model loading currently
  requires the Windows DirectStorage route and fails explicitly on other platforms.

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
complete dense kernel manifest on that device and initialize its DirectStorage session.
Missing assets or an unsupported device remain visible but unselectable.
This preflight does not promise that a particular model will fit in free memory.
The WebUI offers only servable devices and leaves HIP as an explicit choice.

## Direct disk-to-device loading

`HipDirectWeightStore` routes file ranges through DirectStorage into a shared
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
