# Experimental HIP / ROCm backend

`HipTransformerModel` provides a Windows x64 dense Llama inference path with direct
weight loading and independent paged KV caches. GGUF matrices may be F16, Q4_0 or
Q8_0. HF SafeTensors matrices must be F16; F16/F32 normalization weights are accepted,
and F32 norms narrow on the GPU during loading. Packed matrices stay packed,
activations stay F16, and reductions accumulate in F32 registers. The final logit row
uses the existing `IModel` CPU-sampling F32 result contract (`4 * vocabulary` bytes).

## Current boundary

- Windows x64 with AMD HIP SDK 6.4.2 and a compatible display driver. Native kernels
  and direct loading were exercised on RX 6900 XT (`gfx1030`). Bundled `gfx1100`
  code objects are compiled but not hardware-qualified here.
- Dense Llama, causal GQA/MHA, RMSNorm, SiLU and standard or linear RoPE.
  Unsupported architecture features and geometry fail before weight allocation.
- GGUF F16/Q4_0/Q8_0 weights; HF F16 weights and F16/F32 norms. Raw HF Q/K rows use
  NeoX rotate-half RoPE, while converted GGUF Llama rows normally use adjacent pairs.
- Independent F16 or Q8_0 K/V pages. Q8 requires head dimensions divisible by 32.
  Q4 KV, BF16 weights, AWQ/GPTQ GPU execution, bias tensors, MoE, recurrent/hybrid
  models, partial offload, LoRA and MTP are unsupported in this slice.
- Linux runtime probing and native compilation exist, but full model loading
  requires Windows DirectStorage and fails explicitly on other platforms.

This is a correctness-oriented initial path, not a throughput claim. Prefill runs
tokens sequentially and returns only the last logit row. Forward calls share scratch
and are serialized; each request owns an independent paged KV cache. CLI/server
routing and release integration are supplied by the next application slice.

## Experimental Q8 KV opt-in

F16 is the default. Selecting Q8 for either cache plane through
`HipTransformerModel.CreateKvCache` requires the process environment variable
`DOTLLM_HIP_ALLOW_Q8_KV=1`. Only the exact value `1` acknowledges the precision loss.
Unacknowledged Q8 fails before native allocation; it never silently substitutes F16.
The flag does not select a cache type and does not affect Q8 model weights.

Each Q8 block stores 32 signed bytes and a half-precision scale. Rounding loses
information that subsequent attention cannot recover. Actual 1K/2K/4K/8K histories
failed the original relative-L2 and KL gates in every Q8 window; F16 passed. Across
512 scored predictions, Q8 changed eight top-1 choices and increased combined
perplexity by about 0.155%, while reducing owned 8K cache bytes from 188,989,440 to
100,515,840. CPU Q8 also failed, so these results do not isolate a HIP-specific
defect. Mixed F16/Q8 caches require the same acknowledgement and have no long-context
quality qualification. See [measurements and limits](PRECISION.md#actual-1k8k-histories).

## Direct disk-to-device loading

`HipDirectWeightStore` submits file ranges through DirectStorage into a shared
D3D12 default heap, imports that heap with `hipImportExternalMemory`, and maps its
buffer. The HIP device LUID selects the matching DXGI adapter. There is no application
host weight buffer or host-copy fallback. DirectStorage has its own internal staging,
so this does not claim that the OS/driver allocates no host memory.

Each `HipDirectWeightBuffer` owns its resource, heap and HIP import independently
of the loading session. `ByteLength` is payload size and `AllocatedBytes` includes
64 KiB heap alignment. Loading validates ranges, waits for completion, inspects
storage errors and cleans up failed imports. F32 norm imports are freed after narrowing.

Supply `dstorage.dll` and `dstoragecore.dll` beside the executable, through
`DOTLLM_DIRECTSTORAGE_DIR`, or in `~/.dotllm/native/directstorage/1.3.0/x64`.
HIP requires direct loading and does not use the Vulkan opt-in flag.

## Build and validation

See [native build and numerical qualification](../native/hip/README.md).
The managed backend uses source-generated HIP P/Invoke and precompiled `co/` resources.
The dense path does not require hipBLAS; its separate hipBLAS bindings are also tested.

The HIP test collection runs serially. Native tests compare independent CPU formulas,
paged layouts and grouped attention. Direct-loading tests verify exact bytes across
unaligned offsets and 32 MiB boundaries, errors and independent buffer lifetimes.
Full-model checks cover prefill, decode, rollback and packed KV. Historical measured
precision, memory costs and negative controls are recorded in [PRECISION.md](PRECISION.md#9-hip-dense-integration-rx-6900-xt).
