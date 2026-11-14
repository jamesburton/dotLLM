# HIP native kernels

The managed backend loads HIP code objects through `hipModuleLoadData`. Keep these sources and their build scripts alongside the packaged `co/*.co` files. The donor BlueCurve tree did not contain the referenced native sources, so `dense_f16.hip` implements a bounded dense primitive contract with F16 activations and direct packed storage.

## Build

Windows requires the AMD HIP SDK and the Visual Studio C++ build tools. Start **Developer PowerShell for VS 2022**, set `HIP_PATH` to the SDK root if needed, then run:

```powershell
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\6.4'
$env:Path = "$env:HIP_PATH\bin;$env:Path"
.\native\hip\build.ps1 -Architectures gfx1030
```

`-Architectures gfx1030,gfx1100` builds both targets. `HIP_ARCHS` (space-separated) also selects targets; the default is `gfx1030 gfx1100`. `-HipCompiler` accepts an explicit compiler path. The script reports the compiler version and fails if compilation or source discovery fails.

On Linux, with the matching ROCm toolchain on `PATH`:

```bash
HIP_ARCHS="gfx1030 gfx1100" ./native/hip/build.sh
```

The .NET project also rebuilds stale code objects when `hipcc` is available. `HipCompiler` and `HipOffloadArchs` are MSBuild overrides. Use `-p:RequireHipKernels=true` for native qualification: unavailable compilation of stale/missing kernels is an error. Run the native build script explicitly when changing compiler or target architectures; source timestamps alone do not identify a different toolchain.

## Kernel contract

`DenseF16Kernels` resolves every entry point below when constructed. Missing code objects or entry points fail before inference.

| Code object | Entry points |
| --- | --- |
| `dense_f16.co` | `dense_embed_f16`, `dense_convert_f32_f16`, `dense_rmsnorm_f16`, `dense_linear_f16`, `dense_add_f16`, `dense_swiglu_f16`, `dense_rope_f16`, `dense_write_kv_f16`, `dense_attention_f16`, `dense_embed_packed`, `dense_linear_packed`, `dense_write_kv_packed`, `dense_attention_packed` |
| `rmsnorm.co` | Existing `rmsnorm_f32` and `rmsnorm_f16` |

The dense kernels store F16 activations, F16 or packed weights, and independently typed F16/Q8_0 K/V. Projection, RMSNorm and online attention use F32 register accumulation. The conversion kernel is for narrowing file-backed F32 tensors during loading; callers release the source after the queued conversion completes.

- Linear weights are row-major `[outputDimension,inputDimension]`; this initial path processes one token at a time.
- RoPE supports standard adjacent-pair and NeoX split-half layouts, including a partial rotary dimension. Its scale multiplies the position.
- KV pointer tables contain one device pointer per page. Each page is `[pageTokens,kvHeads*headDimension]`, with independent K and V tables per layer/request.
- Attention reads exactly `length` visible tokens from those pages. Query head `h` uses KV head `h / (queryHeads / kvHeads)`. It allocates no contiguous history buffer or score tensor. Head dimensions are bounded to 256; larger dimensions fail explicitly.
- The pointers must belong to the context supplied to `DenseF16Kernels`, and allocations must cover the declared dimensions. The caller owns page-table capacity and model/token bounds.

Managed model/cache creation requires `DOTLLM_HIP_ALLOW_Q8_KV=1` whenever either KV plane uses Q8_0. F16 is the default; the flag acknowledges measured precision loss rather than changing the quantizer. See [precision findings](../../docs/PRECISION.md#actual-1k8k-histories).

The packed overloads accept `QuantizationType` for embedding/projection and independent `KvCacheDType` values for K/V. `KvCacheDType.F32` means F16 on the GPU, following the core enum convention. The original F16 methods retain their behavior.

| Resident format | Bytes per row of N elements | Shape boundary |
| --- | --- | --- |
| F16 weights | `2 * N` | Positive row width |
| GGUF Q4_0 weights | `18 * N / 32` | Row width divisible by 32 |
| GGUF Q8_0 weights | `34 * N / 32` | Row width divisible by 32 |
| F16 KV | `2 * N` | Head dimension up to 256 |
| Q8_0 KV | `34 * N / 32` | Head dimension divisible by 32 and up to 256 |

Packed embedding reads only the selected row into the F16 activation. Projection decodes individual packed values in registers, including during sequential prefill; it never materializes a wider matrix. KV pages hold `[pageTokens, kvHeads * headDimension]` rows in each plane's own format. K and V tables may name separate allocations or separate regions of a combined allocation; their row strides are independent. Attention reads those rows directly without dequantized history or score scratch. Q8 writes use max-absolute-value scales, ties-to-even rounding and symmetric [-127,127] clamping, matching the target CPU quantizer. Packed block layouts follow the target CPU scalar dequantizers.

This primitive set does not implement Q4 KV, other quantized formats, BF16 weights, AWQ/GPTQ, hybrid/recurrent layers, or fused multi-token prefill. Unsupported format/shape overloads fail before launch.

## Qualification

Run hardware tests with skip prevention:

```powershell
$env:DOTLLM_TEST_HIP_REQUIRED = '1'
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --filter 'FullyQualifiedName~HipDenseF16KernelTests|FullyQualifiedName~HipRmsNormKernelTests|FullyQualifiedName~HipRuntimeCandidateTests'
```

On 2026-11-14, all 18 selected tests passed without skips on Windows, RX 6900 XT (`gfx1030`, 16 GB), HIP SDK 6.4.2 (`hipcc` 6.4.50101), driver 32.0.21045.16010. Both `gfx1030` and `gfx1100` code-object targets compiled; only `gfx1030` was executed. This validates primitives and the hipBLAS ABI, not complete model quality or performance.

The matrix tests cover nonsquare shapes, padded leading dimensions and all transpose combinations. RoPE and projection tests reject deliberately substituted layouts; paged attention compares multiple grouped heads and partial pages against a scalar reference. The HIP attribute constants and legacy `hipblasGemmEx` datatype arguments were checked against the installed SDK headers. The legacy symbol takes datatype value 151 for F32 computation; it does not take the v2 compute enum.

Changing the native GQA mapping to `head % kvHeads` caused both attention cases (head dimensions 40 and 256) to fail numerically; restoring the division mapping passed both. The `gfx1030` code-object metadata reports zero shared/private segment bytes and zero register spills for all dense kernels, including attention.

The packed extension adds 22 cases: Q4_0/Q8_0 nonsquare projection and nonzero embedding rows at widths 32/64/128/256; independent F16/Q8_0 K/V combinations; byte-exact Q8 writes including zero blocks and rounding ties; reversed page tables, partially filled pages, guard bytes and unsupported geometry. With the 18 existing checks, **40/40 passed with zero skips** on the same RX 6900 XT. Swapping the native Q4 nibble halves failed all four Q4 cases while Q8 cases passed. Replacing the packed attention head grouping with modulo failed all twelve mixed/quantized cases while F16/F16 passed. The restored binary passed all forty again. Both compiled architectures report zero shared/private segment bytes and zero register spills in all thirteen kernels; only gfx1030 was executed. These are primitive checks; real-checkpoint quantized quality and throughput are separate qualification.
