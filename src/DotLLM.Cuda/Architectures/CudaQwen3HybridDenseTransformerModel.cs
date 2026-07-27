using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the Qwen3HybridDense (<c>qwen35</c>) model — e.g. PrismML's
/// Bonsai-27B. F32 activations throughout — mirrors
/// <c>DotLLM.Models.Architectures.Qwen3HybridDenseTransformerModel</c> (CPU) on the GPU.
/// Adapted from <see cref="CudaQwen3MoeHybridTransformerModel"/>: the GDN and
/// full-attention token-mixing sub-layers are byte-for-byte identical (shared
/// <see cref="HybridLayerLayout"/>/<see cref="GatedDeltaNetConfig"/> infrastructure); the
/// only structural difference is the FFN sub-layer — dense SwiGLU (this class) instead
/// of sparse MoE routing.
/// </summary>
/// <remarks>
/// <para>
/// Each of the <c>numLayers</c> layers has a token-mixing sub-layer (GDN recurrence or
/// full GQA attention) followed by a dense SwiGLU FFN. Layer kind for every index comes
/// from <see cref="ModelConfig.HybridLayout"/>; full-attention layers are placed every
/// <see cref="GatedDeltaNetConfig.FullAttnInterval"/> steps (Bonsai-27B: interval = 4 over
/// 64 layers → 16 full-attention layers, 48 GDN layers).
/// </para>
/// <para>
/// Unlike <see cref="CudaQwen3MoeHybridTransformerModel"/>'s <c>Gemm</c> dispatcher, this
/// class's <see cref="Gemm"/> has explicit I2_S / PQ2_0 branches — Bonsai-27B ships PQ2_0
/// ternary weights, so the ternary GEMV kernels (<see cref="CudaKernels.LaunchPQ2_0GemvF32Native"/>
/// / <see cref="CudaKernels.LaunchDequantPQ2_0ToF16"/>) must be reachable from every
/// projection site (GDN, attention, and dense FFN). The decode-time GEMV/fused-GEMV2 path is
/// F32-native (issue #161) — no F32↔F16 activation-conversion launches around it; only the
/// prefill dequant-then-cuBLAS-HGEMM path (seqLen &gt; 1) still stages through F16.
/// </para>
/// </remarks>
public sealed unsafe class CudaQwen3HybridDenseTransformerModel : IModel
{
    private readonly CudaQwen3HybridDenseForwardState _state;
    private readonly CudaGdnStateCache _gdnCache;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly GgufFile? _gguf;
    private readonly int _deviceId;

    // Per-layer device-side weight pointers — loaded once, alive for model lifetime.
    private readonly DeviceLayer[] _layers;

    // Output stage: token embedding (shared with lm_head when output.weight is missing) and
    // the final RMSNorm gain + lm_head projection.
    private readonly nint _tokenEmbedDevice;
    private readonly QuantizationType _tokenEmbedQt;
    private readonly nint _outputNormDevice; // F32 [hiddenSize]
    private readonly nint _outputDevice;     // lm_head raw quant bytes (may alias _tokenEmbedDevice)
    private readonly QuantizationType _outputQt;
    private readonly int _outputOutputDim;   // vocab size
    private readonly int _outputInputDim;    // hidden size
    private readonly bool _ownsOutputDevice; // false when aliased to embed

    private readonly HybridLayerLayout _layout;
    private readonly GatedDeltaNetConfig _gdn;
    private readonly int _intermediateSize;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _gdnLayerOrdinal;

    private readonly float _ropeTheta;
    private readonly int _ropeDim;

    // Model-owned device F16 scratch for on-the-fly weight dequant in the prefill path
    // (seqLen > 1). See CudaQwen3MoeHybridTransformerModel's field doc for the full
    // rationale — identical convention here.
    private nint _dequantScratchF16Weight;

    // Lazily allocated F16 activation staging buffers for the decode/prefill F16 GEMV/GEMM
    // path. Activations live in F32; the quantised GEMV kernels and cuBLAS HGEMM consume F16.
    private nint _activF16InScratch;
    private long _activF16InScratchElems;
    private nint _activF16OutScratch;
    private long _activF16OutScratchElems;

    // Host-side per-row embedding lookup (NOT a full-table GPU pre-dequant — see the
    // LoadFromGguf remarks for why). Points at the mmap'd GGUF data region backing
    // token_embd.weight; each Forward call dequantizes only its `seqLen` rows on the CPU
    // and H2D-copies the tiny result.
    private readonly nint _embedDataBase;
    private readonly ulong _embedDataOffset;
    private readonly long _embedRowBytes;

    // Per-attention-layer F16 KV cache. Sized lazily on first kvCache-enabled Forward call.
    // See CudaQwen3MoeHybridTransformerModel's field doc for the full rationale.
    private nint[]? _f16KCache;
    private nint[]? _f16VCache;
    private int _f16CacheMaxSeqLen;
    private int _f16CacheCurrentLength;

    // Identity of the IKvCache instance last seen by ForwardFullAttnBody (issue #185). A fresh
    // Forward call whose kvCache is a DIFFERENT instance than last time means "new/unrelated
    // sequence" even when its MaxLength happens to match the previous cache's -- EnsureF16KvCache's
    // "maxSeqLen <= _f16CacheMaxSeqLen" guard only resets _f16CacheCurrentLength/_f32KvValidLength
    // on a capacity-driven reallocation, so a same-size reused-instance-free sequence (e.g. every
    // rep after the first in `dotllm bench -r N>1`, or a scheduler session rebuild that lands on an
    // identical cacheSize) would otherwise leave the previous sequence's final depth "stuck" for the
    // new sequence's entire duration. Reference identity is a reliable signal here because every
    // caller (BenchRunner, ContinuousBatchScheduler, TextGenerator) allocates a brand-new IKvCache
    // instance per logical sequence and only reuses the SAME instance across calls that belong to
    // that one sequence (including legitimate mid-sequence speculative-decoding rollback, which must
    // NOT reset this state).
    private IKvCache? _lastForwardKvCache;

    private nint _f16KvWriteStaging;
    private long _f16KvWriteStagingElems;

    // Per-attention-layer-slot F32 KV read-staging buffers (issue #182). ONE pair per slot
    // (not shared across the 16 full-attention layers, unlike the old single shared-buffer
    // design) so each slot's incrementally-converted history survives from one layer's call
    // to the next layer's call within the same decode step, and across decode steps.
    // _f32KvValidLength[slot] tracks how many leading KV positions are already validly
    // converted into _f32KvReadStagingK/V[slot] as of the last call for that slot -- see
    // ForwardFullAttnBody's "incremental KV F16->F32 staging" block and EnsureF32KvReadStaging.
    private nint[]? _f32KvReadStagingK;
    private nint[]? _f32KvReadStagingV;
    private long[]? _f32KvReadStagingElems;
    private int[]? _f32KvValidLength;

    // Test-only escape hatch (issue #182): forces every call to take the full-range
    // reconversion path (pre-fix behavior) instead of the incremental append fast path. Used by
    // CudaQwen3HybridDenseIncrementalKvDequantTest to assert the fast path is bit-exact against
    // the old behavior across many consecutive decode steps and buffer growths. Never set on the
    // production Forward path.
    internal bool ForceFullKvReconvertForTest { get; set; }

    /// <summary>
    /// Test-only accessor for issue #185's regression test — exposes the F16 KV cache's current
    /// -length cursor directly, since correctness (attention logits) is unaffected by the bug this
    /// cursor's reset fixes (causal masking hides any stale over-length KV range regardless), so a
    /// black-box logits comparison cannot discriminate broken vs. fixed here.
    /// </summary>
    internal int DebugF16CacheCurrentLengthForTest => _f16CacheCurrentLength;

    // Opt-in split-KV attention (issue #183) scratch: partial (max, sum, out) per (head, split).
    // Sized once for the model's fixed (numHeads, headDim) shape and reused every decode step.
    private nint _attnSplitKvPartialMax;
    private nint _attnSplitKvPartialSum;
    private nint _attnSplitKvPartialOut;
    private long _attnSplitKvPartialHeadsAllocated;
    // Combined GQA-group + split-KV scratch (issues #197 + #198) -- separate buffers from the
    // #183 split-KV scratch above so this new, still-experimental tier cannot disturb the
    // already-tested #183 path's allocation lifecycle.
    private nint _attnGqaSplitPartialMax;
    private nint _attnGqaSplitPartialSum;
    private nint _attnGqaSplitPartialOut;
    private long _attnGqaSplitPartialHeadsAllocated;
    private int _attnGqaSplitMaxSplitAllocated;

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _gdnCache.AllocatedBytes;

    /// <summary>Number of full-attention layers — matches the sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>
    /// Creates a length-only <see cref="IKvCache"/> handle sized to <paramref name="maxSeqLen"/>.
    /// K/V storage is owned internally by this model (a per-attention-layer F16 device
    /// cache) — the returned handle only communicates the capacity to
    /// <see cref="Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, IKvCache?)"/>.
    /// </summary>
    public CudaHybridKvCacheHandle CreateKvCache(int maxSeqLen) => new(maxSeqLen);

    private CudaQwen3HybridDenseTransformerModel(
        ModelConfig config,
        GgufFile? gguf,
        DeviceLayer[] layers,
        nint tokenEmbedDevice, QuantizationType tokenEmbedQt,
        nint embedDataBase, ulong embedDataOffset, long embedRowBytes,
        nint outputNormDevice,
        nint outputDevice, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        bool ownsOutputDevice,
        int[] kvSlotForLayer, int attentionLayerCount,
        float ropeTheta, int ropeDim,
        CudaQwen3HybridDenseForwardState state, CudaGdnStateCache gdnCache,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context, CudaKernels kernels,
        int deviceId,
        nint dequantScratchDevice)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _tokenEmbedDevice = tokenEmbedDevice;
        _tokenEmbedQt = tokenEmbedQt;
        _embedDataBase = embedDataBase;
        _embedDataOffset = embedDataOffset;
        _embedRowBytes = embedRowBytes;
        _outputNormDevice = outputNormDevice;
        _outputDevice = outputDevice;
        _outputQt = outputQt;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _ownsOutputDevice = ownsOutputDevice;
        _layout = config.HybridLayout!;
        _gdn = config.GdnConfig!.Value;
        _intermediateSize = config.IntermediateSize;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _state = state;
        _gdnCache = gdnCache;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _deviceId = deviceId;
        _dequantScratchF16Weight = dequantScratchDevice;

        _gdnLayerOrdinal = new int[config.NumLayers];
        int gdnOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _gdnLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet
                ? gdnOrdinal++
                : -1;
        }
    }

    /// <summary>
    /// Loads a Qwen3HybridDense model from an opened GGUF file onto the given CUDA device.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for the model's lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX. Null auto-detects.</param>
    public static CudaQwen3HybridDenseTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"CudaQwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3HybridDense config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3HybridDense config must have GdnConfig populated.", nameof(config));

        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        // Force a fresh load of UploadRawTensor's process-wide cached PQ2_0 repack module/
        // function for THIS model load. Comparing a raw CUDA context handle across calls (see
        // EnsurePq2_0RepackFunc) is not fully reliable on its own: cuCtxDestroy_v2 (from a
        // previous model's Dispose) can free a context whose handle value the driver later
        // reissues to a brand-new cuCtxCreate_v2 call (a classic ABA hazard) — observed in
        // practice when many CUDA tests load/dispose Qwen3HybridDense models across one process
        // (issue #162 follow-up: "CUDA error 400: invalid resource handle" resurfaced in the
        // full suite despite EnsurePq2_0RepackFunc's context check passing). Resetting
        // unconditionally here guarantees every LoadFromGguf call gets a module freshly loaded
        // into ITS OWN just-created context, regardless of any handle-value coincidence.
        s_pq2_0RepackModule = null;
        s_pq2_0RepackFunc = 0;
        s_pq2_0RepackContext = 0;

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout!;
        int hiddenSize = config.HiddenSize;

        // ── Token embedding ──
        // Raw quant bytes are uploaded (small — e.g. ~340 MB at Bonsai's PQ2_0 packing for a
        // 248320-vocab table) for the lm_head-tied case. Unlike
        // CudaQwen3MoeHybridTransformerModel (built for A6000/H100-class VRAM budgets), this
        // model targets a 12 GB consumer card, and Bonsai's huge vocab makes a full-table F32
        // pre-dequant (~5 GB) plus the prefill dequant scratch (~2.5 GB, sized to the largest
        // single tile — the lm_head) push total VRAM demand past 12 GB. That silently trips
        // WDDM's host-RAM paging fallback (confirmed via
        // `Get-Counter '\GPU Process Memory(*)\Shared Usage'` showing multi-GB "shared usage"
        // for the process) — a hang-that-isn't-a-hang, not a deadlock: every GPU memory access
        // becomes a PCIe round-trip once oversubscribed. So the embedding LOOKUP (as opposed to
        // the lm_head projection) is done as a per-call host-side row dequant in Forward()
        // instead — only `seqLen` rows are ever needed, never all 248320.
        var embDesc = tensors["token_embd.weight"];
        long embRowBytes = Dequantize.RowByteSize(hiddenSize, embDesc.QuantizationType);
        // Uploaded via UploadRawTensor (not a hand-rolled alloc+copy) so PQ2_0 tensors get the
        // same load-time interleaved->split repack every other PQ2_0 weight in the model gets
        // (see UploadRawTensor's own doc comment) — see issue #162: this tensor previously
        // bypassed the repack, leaving it in the on-disk interleaved layout while every PQ2_0
        // GEMV/dequant kernel that might read it (via the tied-embedding lm_head fallback below)
        // unconditionally assumes split layout. `embRowBytes` (computed above, from
        // `hiddenSize`/`config.VocabSize`) is used only for the HOST-side per-token embedding
        // dequant in Forward() — reads directly from the mmap'd GGUF bytes via `dataBase`, not
        // from this GPU buffer — so it is unaffected by the on-device byte layout either way.
        nint tokenEmbedDevice = UploadRawTensor(dataBase, embDesc);

        // ── Output norm (always F32 [hiddenSize], dequant on host then H2D) ──
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormHost = new float[hiddenSize];
        Dequantize.ToFloat32(dataBase + (nint)outNormDesc.DataOffset, hiddenSize,
            outNormDesc.QuantizationType, outputNormHost);
        nint outputNormDevice = AllocDevice((long)hiddenSize * sizeof(float));
        fixed (float* p = outputNormHost)
        {
            CopyHtoD(outputNormDevice, (nint)p, (long)hiddenSize * sizeof(float));
        }

        // ── lm_head (tied to token embedding when output.weight is absent) ──
        nint outputDevice;
        QuantizationType outputQt;
        int outputOutputDim;
        int outputInputDim;
        bool ownsOutputDevice;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            // Uploaded via UploadRawTensor (issue #162 fix) — see tokenEmbedDevice's identical
            // fix above for the full rationale. This tensor drives the LM head projection
            // directly (Gemm() -> LaunchDequantPQ2_0ToF16/LaunchPQ2_0GemvF16In/
            // LaunchPQ2_0GemvF32Native, all split-layout readers), so the missing repack here
            // was the actual root cause of #162's -Inf prefill logits: the LM head decoded raw
            // interleaved on-disk bytes as if they were split-layout, producing effectively
            // random (garbage-magnitude, sign-uncorrelated vs. the CPU F32 reference) logit
            // values — most conspicuously ones landing outside FP16's finite range once the
            // prefill HGEMM path's F16 output store rounds them.
            outputDevice = UploadRawTensor(dataBase, outDesc);
            outputQt = outDesc.QuantizationType;
            outputInputDim = outDesc.Shape[0];
            outputOutputDim = outDesc.Shape[1];
            ownsOutputDevice = true;
        }
        else
        {
            outputDevice = tokenEmbedDevice;
            outputQt = embDesc.QuantizationType;
            outputInputDim = embDesc.Shape[0];
            outputOutputDim = embDesc.Shape[1];
            ownsOutputDevice = false;
        }

        // ── RoPE config ──
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new InvalidDataException(
                $"Qwen3HybridDense rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > config.HeadDim)
            throw new InvalidDataException(
                $"Qwen3HybridDense rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;

        // ── Per-layer load ──
        var layers = new DeviceLayer[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        long maxTileFloats = 0;

        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayerDevice(i, dataBase, tensors, config, ref maxTileFloats);
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        maxTileFloats = Math.Max(maxTileFloats, (long)outputOutputDim * outputInputDim);
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        var gdn = config.GdnConfig!.Value;
        var state = new CudaQwen3HybridDenseForwardState(
            hiddenSize: hiddenSize,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * gdn.NKHead + gdn.NVHead) * gdn.DState,
            dConv: gdn.DConv,
            nVHead: gdn.NVHead,
            nKHead: gdn.NKHead,
            dState: gdn.DState,
            intermediateSize: config.IntermediateSize);

        int gdnLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
            if (layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet) gdnLayerCount++;
        var gdnCache = new CudaGdnStateCache(gdn, gdnLayerCount);

        return new CudaQwen3HybridDenseTransformerModel(
            config, gguf, layers,
            tokenEmbedDevice, embDesc.QuantizationType,
            dataBase, embDesc.DataOffset, embRowBytes,
            outputNormDevice,
            outputDevice, outputQt, outputOutputDim, outputInputDim, ownsOutputDevice,
            kvSlotForLayer, attentionLayerCount,
            ropeTheta, ropeDim,
            state, gdnCache, stream, cublas, context, kernels, deviceId,
            dequantScratchDevice);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-layer loaders (host → device upload of raw quant bytes)
    // ──────────────────────────────────────────────────────────────────────

    private static DeviceLayer LoadLayerDevice(
        int layerIdx, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, ref long maxTileFloats)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        var layout = config.HybridLayout!;

        // Norms — F32 [hiddenSize].
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        nint attnNormDevice = UploadF32Tensor(dataBase, attnNormDesc, hiddenSize);
        var postNormDesc = tensors[$"{prefix}.post_attention_norm.weight"];
        nint postAttnNormDevice = UploadF32Tensor(dataBase, postNormDesc, hiddenSize);

        DeviceGdn? gdnDev = null;
        DeviceFullAttn? attnDev = null;
        switch (layout.LayerKind[layerIdx])
        {
            case HybridLayerKind.GatedDeltaNet:
                gdnDev = LoadGdnLayerDevice(prefix, dataBase, tensors, config, ref maxTileFloats);
                break;
            case HybridLayerKind.Attention:
                attnDev = LoadFullAttnLayerDevice(prefix, dataBase, tensors, config,
                    layout.HeadCountKv[layerIdx], ref maxTileFloats);
                break;
            default:
                throw new InvalidOperationException(
                    $"Unexpected HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx} in Qwen3HybridDense.");
        }

        // Dense SwiGLU FFN — ffn_gate.weight, ffn_up.weight, ffn_down.weight (no MoE
        // routing; no "_exps" suffix, confirmed against the real Ternary-Bonsai-27B-Q2_0.gguf).
        var gateDesc = tensors[$"{prefix}.ffn_gate.weight"];
        var upDesc = tensors[$"{prefix}.ffn_up.weight"];
        var downDesc = tensors[$"{prefix}.ffn_down.weight"];
        nint gateDevice = UploadRawTensor(dataBase, gateDesc);
        nint upDevice = UploadRawTensor(dataBase, upDesc);
        nint downDevice = UploadRawTensor(dataBase, downDesc);
        UpdateMaxTile(ref maxTileFloats, (long)gateDesc.Shape[0] * gateDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)upDesc.Shape[0] * upDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)downDesc.Shape[0] * downDesc.Shape[1]);

        return new DeviceLayer
        {
            AttnNormWeightDevice = attnNormDevice,
            PostAttnNormWeightDevice = postAttnNormDevice,
            Gdn = gdnDev,
            FullAttn = attnDev,

            GateWeight = gateDevice, GateQt = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0], GateOutputDim = gateDesc.Shape[1],

            UpWeight = upDevice, UpQt = upDesc.QuantizationType,
            UpInputDim = upDesc.Shape[0], UpOutputDim = upDesc.Shape[1],

            DownWeight = downDevice, DownQt = downDesc.QuantizationType,
            DownInputDim = downDesc.Shape[0], DownOutputDim = downDesc.Shape[1],
        };
    }

    private static DeviceGdn LoadGdnLayerDevice(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, ref long maxTileFloats)
    {
        var gdn = config.GdnConfig!.Value;
        int convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState;

        var qkvDesc = tensors[$"{prefix}.attn_qkv.weight"];
        var gateDesc = tensors[$"{prefix}.attn_gate.weight"];
        var alphaDesc = tensors[$"{prefix}.ssm_alpha.weight"];
        var betaDesc = tensors[$"{prefix}.ssm_beta.weight"];
        var conv1dWDesc = tensors[$"{prefix}.ssm_conv1d.weight"];
        var aDesc = tensors[$"{prefix}.ssm_a"];
        var dtBDesc = tensors[$"{prefix}.ssm_dt.bias"];
        var ssmNormDesc = tensors[$"{prefix}.ssm_norm.weight"];
        var outDesc = tensors[$"{prefix}.ssm_out.weight"];

        nint qkvDevice = UploadRawTensor(dataBase, qkvDesc);
        nint gateDevice = UploadRawTensor(dataBase, gateDesc);
        nint alphaDevice = UploadRawTensor(dataBase, alphaDesc);
        nint betaDevice = UploadRawTensor(dataBase, betaDesc);
        nint outDevice = UploadRawTensor(dataBase, outDesc);

        nint conv1dWeightDevice = UploadF32Tensor(dataBase, conv1dWDesc, gdn.DConv * convDim);
        nint conv1dBiasDevice = AllocDevice((long)convDim * sizeof(float));
        CudaDriverApi.cuMemsetD8_v2(conv1dBiasDevice, 0, (nuint)((long)convDim * sizeof(float)))
            .ThrowOnError();

        nint aDevice = UploadF32Tensor(dataBase, aDesc, gdn.NVHead);
        nint dtBiasDevice = UploadF32Tensor(dataBase, dtBDesc, gdn.NVHead);
        nint ssmNormDevice = UploadF32Tensor(dataBase, ssmNormDesc, gdn.DState);

        UpdateMaxTile(ref maxTileFloats, (long)qkvDesc.Shape[0] * qkvDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)gateDesc.Shape[0] * gateDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)alphaDesc.Shape[0] * alphaDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)betaDesc.Shape[0] * betaDesc.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)outDesc.Shape[0] * outDesc.Shape[1]);

        return new DeviceGdn
        {
            QkvDevice = qkvDevice, QkvQt = qkvDesc.QuantizationType,
            QkvInputDim = qkvDesc.Shape[0], QkvOutputDim = qkvDesc.Shape[1],

            GateDevice = gateDevice, GateQt = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0], GateOutputDim = gateDesc.Shape[1],

            AlphaDevice = alphaDevice, AlphaQt = alphaDesc.QuantizationType,
            AlphaInputDim = alphaDesc.Shape[0], AlphaOutputDim = alphaDesc.Shape[1],

            BetaDevice = betaDevice, BetaQt = betaDesc.QuantizationType,
            BetaInputDim = betaDesc.Shape[0], BetaOutputDim = betaDesc.Shape[1],

            Conv1dWeightDevice = conv1dWeightDevice,
            Conv1dBiasDevice = conv1dBiasDevice,
            ADevice = aDevice,
            DtBiasDevice = dtBiasDevice,
            SsmNormDevice = ssmNormDevice,

            OutDevice = outDevice, OutQt = outDesc.QuantizationType,
            OutInputDim = outDesc.Shape[0], OutOutputDim = outDesc.Shape[1],
        };
    }

    private static DeviceFullAttn LoadFullAttnLayerDevice(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, int numKvHeads, ref long maxTileFloats)
    {
        var q = tensors[$"{prefix}.attn_q.weight"];
        var k = tensors[$"{prefix}.attn_k.weight"];
        var v = tensors[$"{prefix}.attn_v.weight"];
        var o = tensors[$"{prefix}.attn_output.weight"];

        int expectedQGateOut = 2 * config.NumAttentionHeads * config.HeadDim;
        if (q.Shape[1] != expectedQGateOut)
        {
            throw new InvalidDataException(
                $"{prefix}.attn_q.weight has output dim {q.Shape[1]} but qwen35 expects " +
                $"{expectedQGateOut} = 2 * {config.NumAttentionHeads} * {config.HeadDim} (Q+Gate fused).");
        }

        nint qDevice = UploadRawTensor(dataBase, q);
        nint kDevice = UploadRawTensor(dataBase, k);
        nint vDevice = UploadRawTensor(dataBase, v);
        nint oDevice = UploadRawTensor(dataBase, o);

        nint qNormDevice = UploadF32Tensor(dataBase, tensors[$"{prefix}.attn_q_norm.weight"], config.HeadDim);
        nint kNormDevice = UploadF32Tensor(dataBase, tensors[$"{prefix}.attn_k_norm.weight"], config.HeadDim);

        UpdateMaxTile(ref maxTileFloats, (long)q.Shape[0] * q.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)k.Shape[0] * k.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)v.Shape[0] * v.Shape[1]);
        UpdateMaxTile(ref maxTileFloats, (long)o.Shape[0] * o.Shape[1]);

        return new DeviceFullAttn
        {
            QDevice = qDevice, QQt = q.QuantizationType,
            QInputDim = q.Shape[0], QOutputDim = q.Shape[1],

            KDevice = kDevice, KQt = k.QuantizationType,
            KInputDim = k.Shape[0], KOutputDim = k.Shape[1],

            VDevice = vDevice, VQt = v.QuantizationType,
            VInputDim = v.Shape[0], VOutputDim = v.Shape[1],

            ODevice = oDevice, OQt = o.QuantizationType,
            OInputDim = o.Shape[0], OOutputDim = o.Shape[1],

            NumKvHeads = numKvHeads,
            QNormDevice = qNormDevice,
            KNormDevice = kNormDevice,
        };
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Forward dispatch
    // ──────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
        => Forward(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly: false);

    /// <inheritdoc/>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        _profileActiveForThisCall = seqLen == 1;

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;

        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        // Category profiler bracket (issue #168): MakeCurrent + EnsureCapacity + H2D
        // token/position copy + host embed-lookup dequant + H2D embed copy. Confirmed via a
        // fresh dedicated bracket that this region is a genuinely small share of decode time
        // (~0.2% profiled) — not the dark matter some earlier hypotheses suspected.
        ProfStart();
        _context.MakeCurrent();
        if (DebugTrace) LogVram($"before EnsureCapacity(seqLen={seqLen})");
        _state.EnsureCapacity(seqLen);
        if (DebugTrace) LogVram($"after EnsureCapacity (state.AllocatedBytes={_state.AllocatedBytes:N0})");

        nint streamH = _stream.Handle;

        fixed (int* tokenPtr = tokenIds)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int)), streamH).ThrowOnError();
        }
        fixed (int* posPtr = positions)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int)), streamH).ThrowOnError();
        }

        // Host-side per-row embedding lookup — see LoadFromGguf's remarks on why this isn't
        // a GPU-resident full-table dequant. Dequantize each token's row (any quant type)
        // into a small host buffer, then a single bulk H2D copy into HiddenState.
        float[] embedHost = new float[(long)seqLen * hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            nint rowSrc = _embedDataBase + (nint)(_embedDataOffset + (ulong)tokenIds[t] * (ulong)_embedRowBytes);
            Dequantize.ToFloat32(rowSrc, hiddenSize, _tokenEmbedQt,
                embedHost.AsSpan(t * hiddenSize, hiddenSize));
        }
        fixed (float* pEmbedHost = embedHost)
        {
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.HiddenState, (nint)pEmbedHost,
                (nuint)((long)seqLen * hiddenSize * sizeof(float)), streamH).ThrowOnError();
        }
        if (ProfileTrace) _stream.Synchronize();
        ProfMark("setup-embed");

        for (int layer = 0; layer < _layers.Length; layer++)
        {
            RunSingleLayerBody(layer, seqLen, positions, hiddenSize,
                numHeads, numKvHeads, headDim, eps, kvCache);
        }

        if (DebugTrace) { _stream.Synchronize(); Console.Error.WriteLine("[hybrid-debug] all layers done, starting lm_head"); Console.Error.Flush(); LogVram("before lm-head"); }
        ProfStart();

        // Issue #185: only compute/copy the LAST token's logits when the caller has explicitly
        // opted in via lastTokenLogitsOnly (e.g. BenchRunner's untimed prefill / --depth context
        // extension, which only ever reads the last row via argmax). This must NOT be inferred
        // from kvCache alone -- speculative-decoding verification also submits multiple tokens
        // through a kvCache-bearing Forward call but genuinely needs EVERY position's logits to
        // accept/reject each draft token, so it correctly leaves the hint false (see IModel's XML
        // doc on this overload). Restricting the final RmsNorm/LM-head-GEMM/D2H-copy to one row
        // eliminates the single largest VRAM consumer in this Forward call when honored: the old
        // code always sized and computed a full [seqLen, vocabSize] Logits tensor (e.g. ~970 MiB
        // of scratch alone at seqLen=1024 for this model's 248,320-token vocabulary), which was
        // the dominant term in the VRAM-ceiling hang `dotllm bench --depth` hit beyond ~650-768 on
        // a 12GB card (confirmed via cuMemGetInfo instrumentation: EnsureCapacity's cap=1024
        // allocation landed on EXACTLY 0 MiB free). Every caller that hasn't opted in (including
        // ordinary seqLen==1 decode, where this is a no-op either way) keeps returning full
        // per-position logits, unchanged from before this fix.
        int logitsRows = lastTokenLogitsOnly ? 1 : seqLen;
        _state.EnsureLogitsCapacity(logitsRows);
        nint lmHeadInput = logitsRows == seqLen
            ? _state.HiddenState
            : _state.HiddenState + (nint)((long)(seqLen - 1) * hiddenSize * sizeof(float));
        _kernels.LaunchRmsNormF32(lmHeadInput, _outputNormDevice, lmHeadInput,
            hiddenSize, eps, logitsRows, streamH);
        Gemm(_outputDevice, _outputQt, lmHeadInput, _state.Logits,
             _outputOutputDim, _outputInputDim, logitsRows);
        ProfMark("lm-head");
        if (DebugTrace) { _stream.Synchronize(); Console.Error.WriteLine("[hybrid-debug] lm_head done"); Console.Error.Flush(); LogVram("after lm-head"); }

        _stream.Synchronize();
        ProfStart(); // measures result alloc + D2H logits copy

        var shape = new TensorShape(logitsRows, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.Logits,
            (nuint)((long)logitsRows * vocabSize * sizeof(float))).ThrowOnError();
        ProfMark("output-copy");
        if (DebugTrace) LogVram("after D2H logits copy");

        return result;
    }

    // Issue #178 (2026-07-25 profiling round): re-verified that HasCopyRmsNormF32 is actually
    // true at runtime for both "layer-pre-norm1" and "layer-resid1-norm2" below (a real xUnit
    // pass on CudaCopyRmsNormF32Test, not just the property returning true — confirms PTX isn't
    // stale) — the fused single-launch path IS the one executing in production, not the 2-launch
    // fallback. cuobjdump --dump-sass on copy_rmsnorm_f32 (sm_86) shows 24 registers/thread, 0
    // spills — a lean kernel with nothing left to trim internally. The remaining cost is
    // structural, not launch-count: decode always has seqLen==1, so blockIdx.x (=row) only ever
    // takes value 0 — each call launches exactly ONE block, occupying 1 of 28 SMs. This is the
    // same "grid too small to fill the device" shape the parallel gdn_scan_step_f32 ncu
    // investigation found (see .docs/handoff.md), but here it's structural rather than tunable:
    // there is only one row to normalize per decode step, so there is no more per-row parallelism
    // to spread across blocks. Splitting the 5120-wide reduction itself across multiple blocks
    // to raise occupancy would need either (a) a second kernel launch for the cross-block
    // reduction — which gives back exactly the launch this fusion exists to remove — or (b) a
    // cooperative-groups grid-wide sync (cuLaunchCooperativeKernel), which is not used anywhere
    // else in this codebase and adds real portability/occupancy-oversubscription risk for a
    // single-digit-percent theoretical upside. Concluded: no further action here without a
    // materially different (riskier, cooperative-launch) redesign; not attempted this session.
    private void RunSingleLayerBody(int layerIdx, int seqLen, ReadOnlySpan<int> positions,
        int hiddenSize, int numHeads, int numKvHeads, int headDim, float eps, IKvCache? kvCache)
    {
        nint streamH = _stream.Handle;
        long hiddenBytes = (long)seqLen * hiddenSize * sizeof(float);
        var kinds = _layout.LayerKind;
        ref readonly DeviceLayer lw = ref _layers[layerIdx];

        bool debug = DebugTrace;
        if (debug) { _stream.Synchronize(); Console.Error.WriteLine($"[hybrid-debug] layer {layerIdx} start ({kinds[layerIdx]})"); Console.Error.Flush(); }

        ProfStart();
        // 1. Token mixing — residual = hidden; normOut = RmsNorm(hidden, attn_norm).
        if (_kernels.HasCopyRmsNormF32)
        {
            _kernels.LaunchCopyRmsNormF32(_state.HiddenState, _state.Residual, lw.AttnNormWeightDevice,
                _state.NormOutput, hiddenSize, eps, seqLen, streamH);
        }
        else
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
                (nuint)hiddenBytes, streamH).ThrowOnError();
            _kernels.LaunchRmsNormF32(_state.HiddenState, lw.AttnNormWeightDevice, _state.NormOutput,
                hiddenSize, eps, seqLen, streamH);
        }
        ProfMark("layer-pre-norm1");

        if (kinds[layerIdx] == HybridLayerKind.GatedDeltaNet)
        {
            ForwardGdnBody(lw.Gdn!.Value, layerIdx, seqLen, hiddenSize, eps);
        }
        else
        {
            ForwardFullAttnBody(lw.FullAttn!.Value, layerIdx, seqLen, positions,
                numHeads, numKvHeads, headDim, eps, kvCache);
        }
        if (debug) { _stream.Synchronize(); Console.Error.WriteLine($"[hybrid-debug] layer {layerIdx} token-mixing done"); Console.Error.Flush(); }

        ProfStart();
        // 2. First residual add: hidden = residual + normOut.
        _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize, streamH);

        // 3. Dense FFN — residual = hidden; normOut = RmsNorm(hidden, post_attn_norm).
        if (_kernels.HasCopyRmsNormF32)
        {
            _kernels.LaunchCopyRmsNormF32(_state.HiddenState, _state.Residual, lw.PostAttnNormWeightDevice,
                _state.NormOutput, hiddenSize, eps, seqLen, streamH);
        }
        else
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
                (nuint)hiddenBytes, streamH).ThrowOnError();
            _kernels.LaunchRmsNormF32(_state.HiddenState, lw.PostAttnNormWeightDevice, _state.NormOutput,
                hiddenSize, eps, seqLen, streamH);
        }
        ProfMark("layer-resid1-norm2");

        ForwardDenseFfnBody(lw, seqLen, hiddenSize);
        if (debug) { _stream.Synchronize(); Console.Error.WriteLine($"[hybrid-debug] layer {layerIdx} ffn done"); Console.Error.Flush(); }

        ProfStart();
        // 4. Second residual add: hidden = residual + normOut.
        _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
            seqLen * hiddenSize, streamH);
        ProfMark("layer-resid2");
    }

    private static readonly bool DebugTrace =
        Environment.GetEnvironmentVariable("DOTLLM_HYBRID_DEBUG") == "1";

    /// <summary>
    /// Issue #185 diagnostic: logs free/total device VRAM (via <c>cuMemGetInfo</c>) under the same
    /// <see cref="DebugTrace"/> gate used by the existing hybrid-debug traces above — zero cost when
    /// unset. Added to find the real allocation source of the VRAM-ceiling hang that
    /// <c>dotllm bench --depth</c> hits beyond ~650-768 on a 12GB card (see the issue for the
    /// original symptom description).
    /// </summary>
    private static void LogVram(string label)
    {
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out nuint total).ThrowOnError();
        long usedMiB = (long)(total - free) / (1024 * 1024);
        long freeMiB = (long)free / (1024 * 1024);
        long totalMiB = (long)total / (1024 * 1024);
        Console.Error.WriteLine($"[hybrid-debug][vram] {label}: used={usedMiB}MiB free={freeMiB}MiB total={totalMiB}MiB");
        Console.Error.Flush();
    }

    // ──────────────────────────────────────────────────────────────────────
    //  One-off category profiler (DOTLLM_HYBRID_PROFILE=1). Coarse (Stopwatch +
    //  per-mark Synchronize — the sync itself perturbs true pipelined timing, so
    //  this is for RELATIVE category comparison only, not absolute steady-state
    //  throughput). Not used by production code paths; zero cost when unset.
    // ──────────────────────────────────────────────────────────────────────
    internal static readonly bool ProfileTrace =
        Environment.GetEnvironmentVariable("DOTLLM_HYBRID_PROFILE") == "1";
    internal static readonly System.Collections.Generic.Dictionary<string, double> ProfileTotalsMs = new();
    internal static readonly System.Collections.Generic.Dictionary<string, int> ProfileCounts = new();
    private readonly System.Diagnostics.Stopwatch _profSw = new();

    // Only accumulate for seqLen==1 (decode) calls — prefill (seqLen>1) routes every
    // projection through the much heavier dequant-then-cuBLAS path, which would dominate
    // and mask the decode-path (GEMV) picture this profiler exists to show.
    private bool _profileActiveForThisCall;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ProfStart()
    {
        if (!ProfileTrace || !_profileActiveForThisCall) return;
        _stream.Synchronize();
        _profSw.Restart();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ProfMark(string category)
    {
        if (!ProfileTrace || !_profileActiveForThisCall) return;
        _stream.Synchronize();
        double ms = _profSw.Elapsed.TotalMilliseconds;
        ProfileTotalsMs[category] = ProfileTotalsMs.GetValueOrDefault(category) + ms;
        ProfileCounts[category] = ProfileCounts.GetValueOrDefault(category) + 1;
        _profSw.Restart();
    }

    /// <summary>Prints accumulated category totals (ms, and ms/call) to stderr and clears them.</summary>
    internal static void ProfileReportAndReset()
    {
        if (!ProfileTrace) return;
        Console.Error.WriteLine("[hybrid-profile] category totals:");
        foreach (var kv in ProfileTotalsMs)
        {
            int n = ProfileCounts.GetValueOrDefault(kv.Key, 1);
            Console.Error.WriteLine($"  {kv.Key,-20} total={kv.Value,9:F2}ms  calls={n,6}  avg={kv.Value / n,7:F4}ms");
        }
        Console.Error.Flush();
        ProfileTotalsMs.Clear();
        ProfileCounts.Clear();
    }

    // ──────────────────────────────────────────────────────────────────────
    //  GDN token-mixing body — verbatim from CudaQwen3MoeHybridTransformerModel
    //  (architecture-agnostic to the FFN kind).
    // ──────────────────────────────────────────────────────────────────────

    private void ForwardGdnBody(
        in DeviceGdn gdnW, int absoluteLayerIdx, int seqLen, int hiddenSize, float eps)
    {
        nint streamH = _stream.Handle;
        int nVHead = _gdn.NVHead;
        int nKHead = _gdn.NKHead;
        int dState = _gdn.DState;
        int dConv = _gdn.DConv;
        int convDim = (2 * nKHead + nVHead) * dState;
        int vDim = nVHead * dState;
        int kDim = nKHead * dState;
        int gdnOrdinal = _gdnLayerOrdinal[absoluteLayerIdx];

        nint normOut = _state.NormOutput;
        nint qkvBuf = _state.GdnQkvBuf;
        nint zBuf = _state.GdnZBuf;
        nint alphaBuf = _state.GdnAlphaBuf;
        nint betaBuf = _state.GdnBetaBuf;
        nint qBuf = _state.GdnQBuf;
        nint kBuf = _state.GdnKBuf;
        nint vBuf = _state.GdnVBuf;
        nint gdnOut = _state.GdnOut;
        nint convInput = _state.GdnConvInput;

        ProfStart();
        // ── 1. Projections from the normed input ──
        Gemm(gdnW.QkvDevice, gdnW.QkvQt, normOut, qkvBuf,
             gdnW.QkvOutputDim, gdnW.QkvInputDim, seqLen);
        Gemm(gdnW.GateDevice, gdnW.GateQt, normOut, zBuf,
             gdnW.GateOutputDim, gdnW.GateInputDim, seqLen);
        // Alpha/Beta project to tiny output dims (NVHead each) — their decode-time GEMV cost is
        // dominated by the fixed shared-x staging overhead, not compute, so fusing them (unlike
        // gate+up/K+V above, which showed no measurable win — compute already dominates there)
        // avoids paying that staging cost twice.
        if (!TryFusedPQ2_0Gemm2(gdnW.AlphaDevice, gdnW.AlphaQt, gdnW.BetaDevice, gdnW.BetaQt,
                normOut, alphaBuf, betaBuf, gdnW.AlphaOutputDim, gdnW.BetaOutputDim,
                gdnW.AlphaInputDim, gdnW.BetaInputDim, seqLen))
        {
            Gemm(gdnW.AlphaDevice, gdnW.AlphaQt, normOut, alphaBuf,
                 gdnW.AlphaOutputDim, gdnW.AlphaInputDim, seqLen);
            Gemm(gdnW.BetaDevice, gdnW.BetaQt, normOut, betaBuf,
                 gdnW.BetaOutputDim, gdnW.BetaInputDim, seqLen);
        }
        ProfMark("gdn-1-proj");

        // ── 2. Decay g and write-gate beta ──
        if (_kernels.HasGdnDecaySigmoidF32)
        {
            _kernels.LaunchGdnDecaySigmoidF32(alphaBuf, betaBuf, gdnW.DtBiasDevice, gdnW.ADevice,
                seqLen, nVHead, streamH);
        }
        else
        {
            if (_kernels.HasGdnDecayF32)
            {
                _kernels.LaunchGdnDecayF32(alphaBuf, gdnW.DtBiasDevice, gdnW.ADevice,
                    seqLen, nVHead, streamH);
            }
            else
            {
                LaunchGdnDecayHostFallback(alphaBuf, gdnW.DtBiasDevice, gdnW.ADevice, seqLen, nVHead);
            }
            if (_kernels.HasElementwiseF32)
            {
                _kernels.LaunchSigmoidF32(betaBuf, (long)seqLen * nVHead, streamH);
            }
            else
            {
                LaunchSigmoidHostFallback(betaBuf, seqLen * nVHead);
            }
        }
        ProfMark("gdn-2-decaygate");

        // ── 3. Conv1d on QKV concat ──
        nint convStateDev = _gdnCache.GetConvStatePtr(gdnOrdinal);
        long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);

        // Issue #168: decode (seqLen==1) is the overwhelming majority of real GDN-body calls
        // and needs no physical [state; qkv] concat — the fused kernel reads conv_state and
        // the new qkv row directly and writes the shifted trailing state back in place,
        // folding what was 3 memcpy launches + conv1d + silu (5 launches) into 1. Prefill
        // (seqLen>1) keeps the general path unchanged (concat buffer, arbitrary window count).
        if (seqLen == 1 && _kernels.HasGdnConv1dCausalDecodeF32)
        {
            _kernels.LaunchGdnConv1dCausalDecodeF32(convStateDev, qkvBuf,
                gdnW.Conv1dWeightDevice, gdnW.Conv1dBiasDevice, qkvBuf, dConv, convDim, streamH);
        }
        else
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(convInput, convStateDev,
                (nuint)convStateBytes, streamH).ThrowOnError();
            long qkvRowsBytes = (long)seqLen * convDim * sizeof(float);
            nint convInputQkvOff = convInput + (nint)convStateBytes;
            CudaDriverApi.cuMemcpyDtoDAsync_v2(convInputQkvOff, qkvBuf,
                (nuint)qkvRowsBytes, streamH).ThrowOnError();

            _kernels.LaunchConv1dCausalF32(convInput, gdnW.Conv1dWeightDevice, gdnW.Conv1dBiasDevice,
                qkvBuf, dConv, convDim, seqLen, streamH);
            if (_kernels.HasElementwiseF32)
            {
                _kernels.LaunchSiluF32(qkvBuf, (long)seqLen * convDim, streamH);
            }
            else
            {
                LaunchSiluHostFallback(qkvBuf, (long)seqLen * convDim);
            }

            nint trailRowsSrc = convInput + (nint)((long)seqLen * convDim * sizeof(float));
            CudaDriverApi.cuMemcpyDtoDAsync_v2(convStateDev, trailRowsSrc,
                (nuint)convStateBytes, streamH).ThrowOnError();
        }
        ProfMark("gdn-3-conv1d");

        // ── 4. De-interleave Q/K/V from conv output, L2-normalise Q and K per head ──
        // Issue #170: decode (seqLen==1) fuses the deinterleave gather and both Q/K
        // L2-normalize launches into one — also drops the runtime integer division
        // deinterleave_gdn_qkv_f32 pays per element in the general path (SASS-confirmed
        // via cuobjdump; unnecessary here since seqLen==1 always makes the token index 0).
        if (seqLen == 1 && _kernels.HasGdnDeinterleaveL2NormDecodeF32)
        {
            _kernels.LaunchGdnDeinterleaveL2NormDecodeF32(
                qkvBuf, qBuf, kBuf, vBuf, nKHead, nVHead, dState, 1e-6f, streamH);
        }
        else
        {
            if (_kernels.HasDeinterleaveF32)
            {
                _kernels.LaunchDeinterleaveGdnQkvF32(qkvBuf, qBuf, kBuf, vBuf, kDim, vDim, seqLen, streamH);
            }
            else
            {
                long rowBytes = (long)convDim * sizeof(float);
                long kDimBytes = (long)kDim * sizeof(float);
                long vDimBytes = (long)vDim * sizeof(float);
                for (int t = 0; t < seqLen; t++)
                {
                    nint srcRow = qkvBuf + (nint)(t * rowBytes);
                    nint qDst = qBuf + (nint)(t * kDimBytes);
                    nint kDst = kBuf + (nint)(t * kDimBytes);
                    nint vDst = vBuf + (nint)(t * vDimBytes);
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(qDst, srcRow, (nuint)kDimBytes, streamH).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, srcRow + (nint)kDimBytes, (nuint)kDimBytes, streamH).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, srcRow + (nint)(2 * kDimBytes), (nuint)vDimBytes, streamH).ThrowOnError();
                }
            }

            _kernels.LaunchL2NormalizeHeadsF32(qBuf, seqLen * nKHead, dState, 1e-6f, streamH);
            _kernels.LaunchL2NormalizeHeadsF32(kBuf, seqLen * nKHead, dState, 1e-6f, streamH);
        }
        ProfMark("gdn-4-deinterleave");

        // ── 5. GDN scan — single-token kernel driven by host loop ──
        nint gdnStateDev = _gdnCache.GetGdnStatePtr(gdnOrdinal);
        long qStepBytes = (long)kDim * sizeof(float);
        long kStepBytes = qStepBytes;
        long vStepBytes = (long)vDim * sizeof(float);
        long gStepBytes = (long)nVHead * sizeof(float);
        long betaStepBytes = gStepBytes;
        long outStepBytes = vStepBytes;
        // Opt-in, default-OFF row-split cooperative-groups scan (issue #180,
        // DOTLLM_GDN_SCAN_APPROX_SPLIT4=1) — ~26-27% faster kernel time, ~1.5-2% end-to-end decode,
        // but NOT bit-exact vs the CPU oracle (see gated_delta_net_scan.cu's header for the full
        // measured-speedup-vs-bit-parity tradeoff writeup — this is why it defaults OFF). Checked
        // once per shape via IsGdnScanCoopSplit4Safe (cooperative-launch co-residency is a hard
        // per-GPU/per-shape ceiling, not a soft limit) with a safe fallback to the exact kernel.
        bool useCoopSplit4 = CudaKernels.EnableGdnScanApproxSplit4 &&
            _kernels.IsGdnScanCoopSplit4Safe(nVHead, dState);
        for (int t = 0; t < seqLen; t++)
        {
            nint qT = qBuf + (nint)(t * qStepBytes);
            nint kT = kBuf + (nint)(t * kStepBytes);
            nint vT = vBuf + (nint)(t * vStepBytes);
            nint gT = alphaBuf + (nint)(t * gStepBytes);
            nint betaT = betaBuf + (nint)(t * betaStepBytes);
            nint outT = gdnOut + (nint)(t * outStepBytes);
            if (useCoopSplit4)
            {
                _kernels.LaunchGdnScanStepF32CoopSplit4(gdnStateDev, qT, kT, vT, gT, betaT, outT,
                    _state.GdnScanPartialTmp, _state.GdnScanPartialOut, nVHead, nKHead, dState, streamH);
            }
            else
            {
                _kernels.LaunchGdnScanStepF32(gdnStateDev, qT, kT, vT, gT, betaT, outT,
                    nVHead, nKHead, dState, streamH);
            }
        }
        ProfMark("gdn-5-scan");

        // ── 6. Per-head RMSNorm(out, ssm_norm) * silu(z) gating ──
        // Investigated fusing these two launches (issue #172): implemented, SASS-verified,
        // correctness-tested, but showed no reproducible real-bench decode win (within this
        // machine's thermal-drift noise floor) — see rmsnorm_f32.cu's header for the full
        // writeup. Left as two separate calls.
        _kernels.LaunchRmsNormF32(gdnOut, gdnW.SsmNormDevice, gdnOut,
            dState, eps, seqLen * nVHead, streamH);
        _kernels.LaunchSwiGLUF32(zBuf, gdnOut, gdnOut, vDim, seqLen, streamH);
        ProfMark("gdn-6-normgate");

        // ── 7. ssm_out projection into NormOutput ──
        Gemm(gdnW.OutDevice, gdnW.OutQt, gdnOut, normOut,
             gdnW.OutOutputDim, gdnW.OutInputDim, seqLen);
        ProfMark("gdn-7-outproj");
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Full GQA attention body — verbatim from CudaQwen3MoeHybridTransformerModel.
    // ──────────────────────────────────────────────────────────────────────

    private void ForwardFullAttnBody(
        in DeviceFullAttn attn, int layer, int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, float eps, IKvCache? kvCache)
    {
        nint streamH = _stream.Handle;
        int qElems = numHeads * headDim;
        int qgElems = 2 * qElems;
        int kvElems = numKvHeads * headDim;

        nint normOut = _state.NormOutput;
        nint qgBuf = _state.QGateScratch;
        nint q = _state.QScratch;
        nint k = _state.KScratch;
        nint v = _state.VScratch;
        nint gate = _state.GateScratch;
        nint attnOut = _state.AttnOutput;

        ProfStart();
        // ── 1. Fused Q+Gate projection ──
        Gemm(attn.QDevice, attn.QQt, normOut, qgBuf, attn.QOutputDim, attn.QInputDim, seqLen);
        DumpDevice2D($"blk.{layer}.fa_qg", qgBuf, seqLen, qgElems);
        ProfMark("attn-1-qgproj");

        // ── 2. De-interleave QG → Q and Gate ──
        // Single gather-kernel launch (see elementwise_f32.cu's deinterleave_qgate_f32) —
        // replaces a numHeads-iteration host loop of paired cuMemcpyDtoDAsync calls, which
        // profiled as ~12% of total decode time (bigger than the attention kernel itself).
        if (_kernels.HasDeinterleaveF32)
        {
            _kernels.LaunchDeinterleaveQGateF32(qgBuf, q, gate, numHeads, headDim, seqLen, streamH);
        }
        else
        {
            long perTokenQgBytes = (long)qgElems * sizeof(float);
            long perTokenQBytes = (long)qElems * sizeof(float);
            long perHeadBytes = (long)headDim * sizeof(float);
            for (int t = 0; t < seqLen; t++)
            {
                nint qgRow = qgBuf + (nint)(t * perTokenQgBytes);
                nint qRow = q + (nint)(t * perTokenQBytes);
                nint gRow = gate + (nint)(t * perTokenQBytes);
                for (int h = 0; h < numHeads; h++)
                {
                    nint qgHead = qgRow + (nint)(h * 2 * perHeadBytes);
                    nint qHead = qRow + (nint)(h * perHeadBytes);
                    nint gHead = gRow + (nint)(h * perHeadBytes);
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(qHead, qgHead, (nuint)perHeadBytes, streamH).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(gHead, qgHead + (nint)perHeadBytes,
                        (nuint)perHeadBytes, streamH).ThrowOnError();
                }
            }
        }
        DumpDevice2D($"blk.{layer}.fa_q_split", q, seqLen, numHeads * headDim);
        DumpDevice2D($"blk.{layer}.fa_gate_split", gate, seqLen, numHeads * headDim);
        ProfMark("attn-2-deinterleave");

        // ── 3. K and V projections ──
        if (!TryFusedPQ2_0Gemm2(attn.KDevice, attn.KQt, attn.VDevice, attn.VQt,
                normOut, k, v, attn.KOutputDim, attn.VOutputDim,
                attn.KInputDim, attn.VInputDim, seqLen))
        {
            Gemm(attn.KDevice, attn.KQt, normOut, k, attn.KOutputDim, attn.KInputDim, seqLen);
            Gemm(attn.VDevice, attn.VQt, normOut, v, attn.VOutputDim, attn.VInputDim, seqLen);
        }
        DumpDevice2D($"blk.{layer}.fa_k", k, seqLen, numKvHeads * headDim);
        DumpDevice2D($"blk.{layer}.fa_v", v, seqLen, numKvHeads * headDim);
        ProfMark("attn-3-kvproj");

        // ── 4. Per-head QK-norm ──
        _kernels.LaunchRmsNormF32(q, attn.QNormDevice, q,
            headDim, eps, seqLen * numHeads, streamH);
        _kernels.LaunchRmsNormF32(k, attn.KNormDevice, k,
            headDim, eps, seqLen * numKvHeads, streamH);
        DumpDevice2D($"blk.{layer}.fa_q_postnorm", q, seqLen, qElems);
        DumpDevice2D($"blk.{layer}.fa_k_postnorm", k, seqLen, numKvHeads * headDim);
        ProfMark("attn-4-qknorm");

        // ── 5. RoPE — partial-rotary NeoX ──
        _kernels.LaunchRoPEF32(q, k, _state.PositionsDevice,
            seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, 1, streamH);
        DumpDevice2D($"blk.{layer}.fa_q_postrope", q, seqLen, qElems);
        DumpDevice2D($"blk.{layer}.fa_k_postrope", k, seqLen, numKvHeads * headDim);
        ProfMark("attn-5-rope");

        // ── 6. Attention (GQA with causal mask) ──
        if (kvCache is not null)
        {
            if (!ReferenceEquals(kvCache, _lastForwardKvCache))
            {
                _lastForwardKvCache = kvCache;
                _f16CacheCurrentLength = 0;
                if (_f32KvValidLength is not null) Array.Clear(_f32KvValidLength);
            }

            EnsureF16KvCache(kvCache.MaxLength, numKvHeads, headDim);
            int slot = _kvSlotForLayer[layer];
            if (slot < 0)
                throw new InvalidOperationException(
                    $"Layer {layer} is not a full-attention layer but ForwardFullAttnBody was invoked.");
            WriteF16KvRows(slot, k, v, positions, numKvHeads, headDim);
            ProfMark("attn-6a-kvwrite");

            int positionOffset = positions[0];
            int seqKv = _f16CacheCurrentLength;
            int kvLiveElems = seqKv * kvElems;

            // EnsureF32KvReadStaging may grow (and reset _f32KvValidLength[slot] to 0) --
            // always read the valid-length AFTER this call so a fresh grow is observed below.
            EnsureF32KvReadStaging(slot, seqKv, kvElems);
            nint kStage = _f32KvReadStagingK![slot];
            nint vStage = _f32KvReadStagingV![slot];

            // Incremental KV F16->F32 staging (issue #182): the old code unconditionally
            // reconverted the ENTIRE [0, seqKv) live range every call, even though ordinary
            // prefill/decode only ever APPENDS rows (one row per decode step). Convert just the
            // newly-written range when it is a plain contiguous append starting exactly where
            // this slot's staging buffer left off; otherwise fall back to the old full-range
            // reconversion, which is always correct (just not fast) and re-synchronizes
            // _f32KvValidLength. The fallback also covers non-contiguous position batches, a
            // freshly-grown/reallocated buffer (content lost, valid length reset to 0 above), and
            // a KV-cache handle reused for an unrelated sequence without a growth-triggered reset
            // (that case cannot be told apart here from ordinary appends by length alone, but ANY
            // deviation from "starts exactly at the recorded valid length" -- including the
            // shrink case the position range would otherwise imply -- is treated as untrusted and
            // triggers the safe full reconversion).
            //
            // Result (issue #182, RTX 3060, real Bonsai-27B, single continuous decode sequence --
            // NOT `dotllm bench -r N>1`, which was found during this work to be unsuitable for
            // measuring this specific change: all reps but the discarded warmup shared one model
            // instance whose _f16CacheCurrentLength never reset when a same-size IKvCache was
            // reused, so every REPORTED rep ran "stuck" at the previous rep's final depth for its
            // entire duration -- a separate bench-methodology gap, not a correctness bug in this
            // fix, tracked and FIXED by issue #185's reference-identity reset above): bit-exact vs. the old full-range
            // reconversion (dedicated correctness test, many consecutive steps incl. a mid-run F32
            // staging buffer growth). End-to-end wall-clock across several full single-sequence
            // decode runs (2048-4096 steps, single-token-decode-loop depth-building to avoid an
            // unrelated pre-existing VRAM-ceiling issue in the batched --depth path -- also found
            // this session, reproduces identically on unmodified pre-#182 code): a small, directionally
            // positive but noise-comparable full-run win (roughly 0 to +4.5% across independent
            // rounds, averaging ~+1.5%), consistent with the theoretical model (this eliminates a
            // per-decode-step O(depth) cost, i.e. an O(depth^2) cumulative cost over a generation,
            // replacing it with O(1) per step / O(depth) cumulative -- but attn-6b-kvdequant was
            // always a small slice of total decode time next to the O(depth) naive-attention kernel
            // itself, `attn-6c-core`, which this fix does not touch). Kept: zero added
            // synchronization/round-trips, provably less work than before in every case, and no
            // observed regression in any round (unlike this file's several genuine negative results,
            // which all showed consistent, large regressions from added sync overhead).
            int prevValid = _f32KvValidLength![slot];
            bool contiguousAppend = !ForceFullKvReconvertForTest
                && IsContiguousAscendingRun(positions) && positions[0] == prevValid;

            if (contiguousAppend)
            {
                int newCount = positions.Length;
                if (newCount > 0)
                {
                    long newFirstElems = (long)positions[0] * kvElems;
                    int newElems = newCount * kvElems;
                    nint kSrc = _f16KCache![slot] + (nint)(newFirstElems * sizeof(ushort));
                    nint vSrc = _f16VCache![slot] + (nint)(newFirstElems * sizeof(ushort));
                    nint kDst = kStage + (nint)(newFirstElems * sizeof(float));
                    nint vDst = vStage + (nint)(newFirstElems * sizeof(float));
                    _kernels.LaunchConvertF16ToF32(kSrc, kDst, newElems, streamH);
                    _kernels.LaunchConvertF16ToF32(vSrc, vDst, newElems, streamH);
                }
                _f32KvValidLength[slot] = Math.Max(prevValid, positions[^1] + 1);
            }
            else
            {
                _kernels.LaunchConvertF16ToF32(_f16KCache![slot], kStage, kvLiveElems, streamH);
                _kernels.LaunchConvertF16ToF32(_f16VCache![slot], vStage, kvLiveElems, streamH);
                _f32KvValidLength[slot] = seqKv;
            }
            ProfMark("attn-6b-kvdequant");

            // Opt-in split-KV ("Flash-Decoding") attention (issue #183): attention_f32's grid is
            // seqQ*numHeads, so decode's seqQ==1 underfills the GPU (numHeads=24 < 28 SMs for
            // Bonsai-27B) and the KV-tile loop runs fully sequentially within one block per head —
            // a cost that grows with context depth (`.docs/handoff.md`'s "Depth-dependent
            // attention finding": +151% 0.043->0.108ms/call, depth 0->256). Decode-only (seqLen==1)
            // scope; gated by a minimum seqKv (splitting a shallow KV range isn't worth the
            // grid.sync + combine overhead) and a per-shape cooperative-launch co-residency safety
            // check. Off by default (DOTLLM_ATTN_SPLIT_KV=1 to enable) — see CudaKernels.cs's
            // EnableAttentionSplitKv doc and attention_f32.cu's header for the full tradeoff
            // (reassociated, not bit-exact, float reduction across the cross-block combine).
            // kStage/vStage are this SLOT's per-layer F32 KV staging buffers (issue #182 made
            // these per-slot arrays, not one shared buffer) — the split-KV kernel reads the same
            // staged data the exact kernel would, just via more blocks.
            // Opt-in combined GQA-group + split-KV attention (issues #197 + #198): register-blocks
            // the QK/PV loops across the numHeads/numKvHeads query-head group sharing each KV
            // head (grid = numKvHeads instead of numHeads), composed with a runtime-tuned KV
            // split so the grid doesn't collapse below the #183 baseline for small numKvHeads
            // (Bonsai-27B: numKvHeads=4 alone would be 6x worse than today's grid=24 -- see
            // attention_f32.cu's combined-kernel header). Preferred over the plain #183 split-KV
            // tier below whenever the shape has real GQA (numKvHeads < numHeads) and the shape
            // fits this GPU's cooperative-launch co-residency ceiling; falls back to the #183
            // tier, then to the exact LaunchAttentionF32, on any ineligibility. Off by default
            // (DOTLLM_ATTN_GQA_SPLIT=1 to enable) -- see CudaKernels.cs's EnableAttentionGqaSplit
            // doc for the full tradeoff.
            bool gqaGroupEligible = CudaKernels.IsGqaGroupShapeSupported(numHeads, numKvHeads)
                && numKvHeads < numHeads;
            int gqaMaxSafeSplit = gqaGroupEligible
                ? _kernels.MaxSafeAttentionGqaSplit(numKvHeads, headDim, numHeads / numKvHeads)
                : 0;
            bool useGqaSplitAttn = seqLen == 1
                && CudaKernels.EnableAttentionGqaSplit
                && seqKv >= CudaKernels.AttentionGqaSplitMinSeqKv
                && gqaGroupEligible
                && _kernels.HasAttentionF32GqaSplitKv
                && gqaMaxSafeSplit >= 1;

            bool useSplitKvAttn = !useGqaSplitAttn
                && seqLen == 1
                && CudaKernels.EnableAttentionSplitKv
                && seqKv >= CudaKernels.AttentionSplitKvMinSeqKv
                && _kernels.IsAttentionSplitKvSafe(numHeads, headDim);

            if (useGqaSplitAttn)
            {
                int kvSplit = CudaKernels.ComputeAttentionKvSplit(seqKv, numKvHeads, gqaMaxSafeSplit);
                EnsureAttentionGqaSplitScratch(numHeads, headDim, gqaMaxSafeSplit);
                _kernels.LaunchAttentionF32GqaSplit(q, kStage, vStage, attnOut,
                    seqKv, numHeads, numKvHeads, headDim,
                    positionOffset: positionOffset, slidingWindow: 0, kvSplit,
                    _attnGqaSplitPartialMax, _attnGqaSplitPartialSum, _attnGqaSplitPartialOut, streamH);
            }
            else if (useSplitKvAttn)
            {
                EnsureAttentionSplitKvScratch(numHeads, headDim);
                _kernels.LaunchAttentionF32SplitKv(q, kStage, vStage, attnOut,
                    seqKv, numHeads, numKvHeads, headDim,
                    positionOffset: positionOffset, slidingWindow: 0,
                    _attnSplitKvPartialMax, _attnSplitKvPartialSum, _attnSplitKvPartialOut, streamH);
            }
            else
            {
                _kernels.LaunchAttentionF32(q, kStage, vStage, attnOut,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positionOffset: positionOffset, slidingWindow: 0, streamH);
            }
        }
        else
        {
            _kernels.LaunchAttentionF32(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeads, headDim,
                positionOffset: 0, slidingWindow: 0, streamH);
        }
        DumpDevice2D($"blk.{layer}.fa_attnout_pregate", attnOut, seqLen, qElems);
        ProfMark("attn-6c-core");

        // ── 7. attnOut *= sigmoid(gate). ──
        if (_kernels.HasElementwiseF32)
        {
            _kernels.LaunchSigmoidMulF32(attnOut, gate, (long)seqLen * qElems, streamH);
        }
        else
        {
            LaunchSigmoidMulHostFallback(attnOut, gate, (long)seqLen * qElems);
        }
        DumpDevice2D($"blk.{layer}.fa_attnout_postgate", attnOut, seqLen, qElems);
        ProfMark("attn-7-gate");

        // ── 8. Output projection ──
        Gemm(attn.ODevice, attn.OQt, attnOut, _state.NormOutput,
             attn.OOutputDim, attn.OInputDim, seqLen);
        ProfMark("attn-8-outproj");
    }

    /// <summary>
    /// Debug helper: D2H-copy a contiguous F32 device buffer and forward it to TensorDump.
    /// Compiled away to a single env-var check when DOTLLM_TENSOR_DUMP is unset.
    /// </summary>
    private void DumpDevice2D(string name, nint devPtr, int d0, int d1)
    {
        if (!DotLLM.Models.Architectures.TensorDump.Enabled) return;
        long n = (long)d0 * d1;
        if (n <= 0) return;
        _stream.Synchronize();
        float[] host = new float[n];
        fixed (float* pHost = host)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pHost, devPtr, (nuint)(n * sizeof(float))).ThrowOnError();
            DotLLM.Models.Architectures.TensorDump.Dump2D(name, pHost, d0, d1);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-attention-layer F16 KV cache (model-private) — verbatim.
    // ──────────────────────────────────────────────────────────────────────

    private void EnsureF16KvCache(int maxSeqLen, int numKvHeads, int headDim)
    {
        if (_f16KCache is not null && maxSeqLen <= _f16CacheMaxSeqLen) return;

        if (_f16KCache is not null)
        {
            for (int i = 0; i < _f16KCache.Length; i++)
            {
                if (_f16KCache[i] != 0) CudaDriverApi.cuMemFree_v2(_f16KCache[i]);
                if (_f16VCache![i] != 0) CudaDriverApi.cuMemFree_v2(_f16VCache[i]);
            }
        }

        _f16KCache = new nint[_attentionLayerCount];
        _f16VCache = new nint[_attentionLayerCount];
        long bytesPerLayer = (long)maxSeqLen * numKvHeads * headDim * sizeof(ushort);
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            _f16KCache[i] = AllocDevice(bytesPerLayer);
            _f16VCache[i] = AllocDevice(bytesPerLayer);
        }
        _f16CacheMaxSeqLen = maxSeqLen;
        _f16CacheCurrentLength = 0;

        // Per-slot F32 KV read-staging (issue #182): reset alongside the F16 cache reallocation
        // above. Freeing any previously-allocated staging buffers and re-zeroing the valid-length
        // array here mirrors _f16CacheCurrentLength's own reset -- the next ForwardFullAttnBody
        // call for every slot will see prevValid==0, take the "not a matching contiguous append"
        // branch only if positions[0] != 0 (freshly-sized buffers still correctly fast-path a
        // from-scratch prefill starting at position 0), and otherwise safely fall back to a full
        // reconversion.
        if (_f32KvReadStagingK is not null)
        {
            for (int i = 0; i < _f32KvReadStagingK.Length; i++)
            {
                if (_f32KvReadStagingK[i] != 0) CudaDriverApi.cuMemFree_v2(_f32KvReadStagingK[i]);
                if (_f32KvReadStagingV![i] != 0) CudaDriverApi.cuMemFree_v2(_f32KvReadStagingV[i]);
            }
        }
        _f32KvReadStagingK = new nint[_attentionLayerCount];
        _f32KvReadStagingV = new nint[_attentionLayerCount];
        _f32KvReadStagingElems = new long[_attentionLayerCount];
        _f32KvValidLength = new int[_attentionLayerCount];
    }

    private void EnsureF16KvWriteStaging(int seqLen, int kvElems)
    {
        long needed = (long)seqLen * kvElems;
        if (needed <= _f16KvWriteStagingElems) return;

        long grown = _f16KvWriteStagingElems == 0 ? 256L : _f16KvWriteStagingElems;
        while (grown < needed) grown *= 2;

        FreeIfNonZero(ref _f16KvWriteStaging);
        _f16KvWriteStaging = AllocDevice(grown * sizeof(ushort));
        _f16KvWriteStagingElems = grown;
    }

    /// <summary>
    /// Ensures <paramref name="slot"/>'s F32 KV read-staging buffer pair can hold
    /// <c>seqKv * kvElems</c> floats, growing (doubling) if needed. A grow reallocates the
    /// buffer at a new address with unspecified content -- any previously-converted prefix for
    /// this slot is gone, so <see cref="_f32KvValidLength"/>[<paramref name="slot"/>] is reset to
    /// 0, forcing the caller (see the "incremental KV F16->F32 staging" block in
    /// <c>ForwardFullAttnBody</c>) to fully reconvert on its next use of this slot.
    /// </summary>
    private void EnsureF32KvReadStaging(int slot, int seqKv, int kvElems)
    {
        long needed = (long)seqKv * kvElems;
        if (needed <= _f32KvReadStagingElems![slot]) return;

        long grown = _f32KvReadStagingElems[slot] == 0 ? 256L : _f32KvReadStagingElems[slot];
        while (grown < needed) grown *= 2;

        FreeIfNonZero(ref _f32KvReadStagingK![slot]);
        FreeIfNonZero(ref _f32KvReadStagingV![slot]);
        _f32KvReadStagingK[slot] = AllocDevice(grown * sizeof(float));
        _f32KvReadStagingV![slot] = AllocDevice(grown * sizeof(float));
        _f32KvReadStagingElems[slot] = grown;
        _f32KvValidLength![slot] = 0;
    }

    /// <summary>True if <paramref name="positions"/> is a strictly-ascending run of consecutive
    /// integers (e.g. <c>[5]</c>, <c>[5,6,7]</c>). Used to gate the incremental KV F16->F32
    /// staging fast path -- see the call site in <c>ForwardFullAttnBody</c>.</summary>
    private static bool IsContiguousAscendingRun(ReadOnlySpan<int> positions)
    {
        for (int i = 1; i < positions.Length; i++)
            if (positions[i] != positions[i - 1] + 1) return false;
        return true;
    }

    /// <summary>
    /// Ensures the opt-in split-KV attention (issue #183) partial scratch buffers can hold
    /// <c>numHeads * AttentionKvSplit</c> (max, sum) scalars and
    /// <c>numHeads * AttentionKvSplit * headDim</c> output floats. Sized once for the model's
    /// fixed (numHeads, headDim) shape (both are load-time constants for a given GGUF) — the
    /// "grown" check is a defensive no-op in practice, not a hot-path realloc.
    /// </summary>
    private void EnsureAttentionSplitKvScratch(int numHeads, int headDim)
    {
        long neededHeads = numHeads;
        if (neededHeads <= _attnSplitKvPartialHeadsAllocated) return;

        FreeIfNonZero(ref _attnSplitKvPartialMax);
        FreeIfNonZero(ref _attnSplitKvPartialSum);
        FreeIfNonZero(ref _attnSplitKvPartialOut);

        long scalarCount = neededHeads * CudaKernels.AttentionKvSplit;
        long outCount = scalarCount * headDim;
        _attnSplitKvPartialMax = AllocDevice(scalarCount * sizeof(float));
        _attnSplitKvPartialSum = AllocDevice(scalarCount * sizeof(float));
        _attnSplitKvPartialOut = AllocDevice(outCount * sizeof(float));
        _attnSplitKvPartialHeadsAllocated = neededHeads;
    }

    /// <summary>
    /// Ensures the opt-in combined GQA-group + split-KV attention (issues #197 + #198) partial
    /// scratch buffers can hold <c>numHeads * maxSplit</c> (max, sum) scalars and
    /// <c>numHeads * maxSplit * headDim</c> output floats. Sized against <paramref name="maxSplit"/>
    /// (the real co-residency CEILING from <see cref="CudaKernels.MaxSafeAttentionGqaSplit"/>),
    /// not the per-call <c>kvSplit</c> value the heuristic picks each step -- (numHeads, headDim,
    /// maxSplit) are all load-time constants for a given GGUF+GPU pair, so this is sized once and
    /// reused for every decode step even though the per-call split varies with seqKv. Mirrors
    /// <see cref="EnsureAttentionSplitKvScratch"/> exactly, generalized for a variable ceiling.
    /// </summary>
    private void EnsureAttentionGqaSplitScratch(int numHeads, int headDim, int maxSplit)
    {
        long neededHeads = numHeads;
        if (neededHeads <= _attnGqaSplitPartialHeadsAllocated && maxSplit <= _attnGqaSplitMaxSplitAllocated) return;

        FreeIfNonZero(ref _attnGqaSplitPartialMax);
        FreeIfNonZero(ref _attnGqaSplitPartialSum);
        FreeIfNonZero(ref _attnGqaSplitPartialOut);

        long scalarCount = neededHeads * maxSplit;
        long outCount = scalarCount * headDim;
        _attnGqaSplitPartialMax = AllocDevice(scalarCount * sizeof(float));
        _attnGqaSplitPartialSum = AllocDevice(scalarCount * sizeof(float));
        _attnGqaSplitPartialOut = AllocDevice(outCount * sizeof(float));
        _attnGqaSplitPartialHeadsAllocated = neededHeads;
        _attnGqaSplitMaxSplitAllocated = maxSplit;
    }

    private void WriteF16KvRows(int layerSlot, nint kSrcF32, nint vSrcF32,
                                 ReadOnlySpan<int> positions, int numKvHeads, int headDim)
    {
        nint streamH = _stream.Handle;
        int seqLen = positions.Length;
        int kvElems = numKvHeads * headDim;
        long rowBytes = (long)kvElems * sizeof(ushort);
        int totalElems = seqLen * kvElems;

        bool contiguous = seqLen > 0;
        int maxPos = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            int p = positions[i];
            if ((uint)p >= (uint)_f16CacheMaxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {p} at index {i} exceeds F16 KV cache capacity {_f16CacheMaxSeqLen}.");
            if (p > maxPos) maxPos = p;
            if (i > 0 && positions[i] != positions[i - 1] + 1) contiguous = false;
        }

        EnsureF16KvWriteStaging(seqLen, kvElems);
        nint stagingF16 = _f16KvWriteStaging;
        nint kBase = _f16KCache![layerSlot];
        nint vBase = _f16VCache![layerSlot];

        _kernels.LaunchConvertF32ToF16(kSrcF32, stagingF16, totalElems, streamH);
        if (contiguous && seqLen > 1)
        {
            long bulkBytes = (long)seqLen * rowBytes;
            nint kDst = kBase + (nint)((long)positions[0] * rowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, stagingF16, (nuint)bulkBytes, streamH).ThrowOnError();
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                nint kDst = kBase + (nint)((long)positions[i] * rowBytes);
                nint kS = stagingF16 + (nint)((long)i * rowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kS, (nuint)rowBytes, streamH).ThrowOnError();
            }
        }

        _kernels.LaunchConvertF32ToF16(vSrcF32, stagingF16, totalElems, streamH);
        if (contiguous && seqLen > 1)
        {
            long bulkBytes = (long)seqLen * rowBytes;
            nint vDst = vBase + (nint)((long)positions[0] * rowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, stagingF16, (nuint)bulkBytes, streamH).ThrowOnError();
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                nint vDst = vBase + (nint)((long)positions[i] * rowBytes);
                nint vS = stagingF16 + (nint)((long)i * rowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vS, (nuint)rowBytes, streamH).ThrowOnError();
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _f16CacheCurrentLength)
            _f16CacheCurrentLength = newLength;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Dense SwiGLU FFN body (replaces CudaQwen3MoeHybridTransformerModel's
    //  ForwardMoeBody).
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dense SwiGLU FFN forward. Reads pre-normed activations from
    /// <see cref="CudaQwen3HybridDenseForwardState.NormOutput"/> and writes the FFN output
    /// back to the same buffer. Entirely on-device — no host round-trip.
    /// </summary>
    private void ForwardDenseFfnBody(in DeviceLayer lw, int seqLen, int hiddenSize)
    {
        nint streamH = _stream.Handle;
        nint normOut = _state.NormOutput;
        nint ffnGate = _state.FfnGate;
        nint ffnUp = _state.FfnUp;
        nint siluOut = _state.SiluOutput;

        ProfStart();
        if (!TryFusedPQ2_0Gemm2(lw.GateWeight, lw.GateQt, lw.UpWeight, lw.UpQt,
                normOut, ffnGate, ffnUp, lw.GateOutputDim, lw.UpOutputDim,
                lw.GateInputDim, lw.UpInputDim, seqLen))
        {
            Gemm(lw.GateWeight, lw.GateQt, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
            Gemm(lw.UpWeight, lw.UpQt, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);
        }
        ProfMark("ffn-1-gateup");
        _kernels.LaunchSwiGLUF32(ffnGate, ffnUp, siluOut, _intermediateSize, seqLen, streamH);
        ProfMark("ffn-2-swiglu");
        Gemm(lw.DownWeight, lw.DownQt, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen);
        ProfMark("ffn-3-down");
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Gemm dispatcher — quantised-direct GEMV (decode) / HGEMM-after-F16-dequant (prefill)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-layer F32-in / F32-out projection dispatcher. Adapted from
    /// <see cref="CudaQwen3MoeHybridTransformerModel.Gemm"/> with explicit I2_S / PQ2_0
    /// branches added — Bonsai-27B ships PQ2_0 ternary weights end-to-end (GDN projections,
    /// attention projections, and the dense FFN), so the ternary GEMV/dequant kernels must
    /// be reachable from every call site. The MoE hybrid's Gemm lacks these branches; that
    /// is a pre-existing, separate gap (tracked, not fixed here — out of scope for this
    /// dense-architecture addition).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weight, QuantizationType qt, nint x, nint y, int m, int k, int seqLen)
    {
        nint streamH = _stream.Handle;

        if (qt == QuantizationType.F32)
        {
            CudaGemm.LinearF32(_cublas.Handle, x, weight, y, seqLen, k, m, streamH);
            return;
        }

        if (seqLen == 1)
        {
            if (qt == QuantizationType.Q8_0)
            {
                _kernels.LaunchQuantizedGemvF32In(weight, x, y, m, k, streamH);
                return;
            }

            if (qt == QuantizationType.I2_S)
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);
                _kernels.LaunchI2_SGemvF16In(weight, _activF16InScratch, _activF16OutScratch, m, k, streamH);
                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }

            if (qt == QuantizationType.PQ2_0)
            {
                // F32-native GEMV (#161) — converts F32<->F16 inline in the kernel's own
                // vectorized stage/store steps, so no surrounding LaunchConvertF32ToF16/
                // LaunchConvertF16ToF32 launches or _activF16InScratch/_activF16OutScratch
                // round-trip are needed here (unlike the I2_S branch above, still on the old
                // convert-launch-bracketed path — see native/kernels/pq2_0_gemv.cu's
                // "F32-native activations" file-header section for the full rationale and the
                // note on why I2_S wasn't also converted in this pass).
                _kernels.LaunchPQ2_0GemvF32Native(weight, x, y, m, k, streamH);
                return;
            }

            if (qt == QuantizationType.F16
                || _kernels.HasMmq(qt)
                || _kernels.HasQuantizedGemvKernel(qt))
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);

                if (qt == QuantizationType.F16)
                {
                    CudaGemm.GemvF16(_cublas.Handle, weight, _activF16InScratch,
                        _activF16OutScratch, m, k, streamH);
                }
                else if (_kernels.HasMmq(qt) && !CudaKernels.ForceDirectGemv)
                {
                    _kernels.LaunchQuantizedGemvMmq(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, preqScratch: 0, streamH);
                }
                else
                {
                    _kernels.LaunchQuantizedGemv(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, streamH);
                }

                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }
        }

        // ── Prefill (seqLen > 1) and decode fallback ──
        long totalElems = (long)m * k;
        int totalElemsI = checked((int)totalElems);
        int activInElems = checked((int)((long)seqLen * k));
        int activOutElems = checked((int)((long)seqLen * m));
        EnsureActivF16InScratch(activInElems);
        EnsureActivF16OutScratch(activOutElems);

        if (qt == QuantizationType.I2_S)
            _kernels.LaunchDequantI2_SToF16(weight, _dequantScratchF16Weight, m, k, streamH);
        else if (qt == QuantizationType.PQ2_0)
            _kernels.LaunchDequantPQ2_0ToF16(weight, _dequantScratchF16Weight, m, k, streamH);
        else
            _kernels.LaunchDequantToF16(weight, qt, _dequantScratchF16Weight, totalElemsI, streamH);

        _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, activInElems, streamH);
        CudaGemm.LinearF16(_cublas.Handle, _activF16InScratch, _dequantScratchF16Weight,
            _activF16OutScratch, seqLen, k, m, streamH);
        _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, activOutElems, streamH);
    }

    private void EnsureActivF16InScratch(long halfs)
    {
        if (halfs <= _activF16InScratchElems) return;
        FreeIfNonZero(ref _activF16InScratch);
        _activF16InScratch = AllocDevice(halfs * sizeof(ushort));
        _activF16InScratchElems = halfs;
    }

    private void EnsureActivF16OutScratch(long halfs)
    {
        if (halfs <= _activF16OutScratchElems) return;
        FreeIfNonZero(ref _activF16OutScratch);
        _activF16OutScratch = AllocDevice(halfs * sizeof(ushort));
        _activF16OutScratchElems = halfs;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Fused 2-way PQ2_0 decode dispatch — dense FFN gate+up, full-attention K+V.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dispatches a fused decode-time PQ2_0 GEMV for a projection pair sharing one input
    /// (e.g. dense FFN gate+up, full-attention K+V) via
    /// <see cref="CudaKernels.LaunchPQ2_0Gemv2F32Native"/> — F32-native in/out (#161), so unlike
    /// the pre-#161 version this needs no F32→F16 activation staging launch, no F16→F32 output
    /// conversion launches, and no <c>_activF16InScratch</c>/<c>_activF16OutScratch</c>/
    /// (former) <c>_fusedOut1F16Scratch</c> round-trip at all — <paramref name="x"/>/
    /// <paramref name="y0"/>/<paramref name="y1"/> are passed straight through to the kernel. Falls
    /// back to <see langword="false"/> (caller issues two separate <see cref="Gemm"/> calls) for
    /// prefill, mixed/non-PQ2_0 quant types, or unequal input dims.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool TryFusedPQ2_0Gemm2(
        nint weight0, QuantizationType qt0, nint weight1, QuantizationType qt1,
        nint x, nint y0, nint y1, int m0, int m1, int k0, int k1, int seqLen)
    {
        if (seqLen != 1 || k0 != k1
            || qt0 != QuantizationType.PQ2_0 || qt1 != QuantizationType.PQ2_0)
            return false;

        nint streamH = _stream.Handle;
        int k = k0;
        _kernels.LaunchPQ2_0Gemv2F32Native(weight0, weight1, x, y0, y1, m0, m1, k, streamH);
        return true;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Host fallbacks — temporary CPU paths used while waiting on CUDA kernels.
    //  Verbatim from CudaQwen3MoeHybridTransformerModel.
    // ──────────────────────────────────────────────────────────────────────

    private void LaunchGdnDecayHostFallback(nint alphaBufDev, nint dtBiasDev, nint aDev,
        int seqLen, int nVHead)
    {
        _stream.Synchronize();
        float[] alpha = new float[seqLen * nVHead];
        float[] dtBias = new float[nVHead];
        float[] a = new float[nVHead];
        fixed (float* pAlpha = alpha)
        fixed (float* pDtBias = dtBias)
        fixed (float* pA = a)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pAlpha, alphaBufDev, (nuint)(alpha.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pDtBias, dtBiasDev, (nuint)(dtBias.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pA, aDev, (nuint)(a.Length * sizeof(float))).ThrowOnError();

            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < nVHead; h++)
                {
                    int idx = t * nVHead + h;
                    float x = alpha[idx] + dtBias[h];
                    float softplus = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
                    alpha[idx] = MathF.Exp(softplus * a[h]);
                }
            }

            CudaDriverApi.cuMemcpyHtoD_v2(alphaBufDev, (nint)pAlpha, (nuint)(alpha.Length * sizeof(float))).ThrowOnError();
        }
    }

    private void LaunchSigmoidHostFallback(nint bufDev, long elems)
    {
        _stream.Synchronize();
        float[] buf = new float[elems];
        fixed (float* pBuf = buf)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pBuf, bufDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            for (long i = 0; i < elems; i++)
                buf[i] = 1f / (1f + MathF.Exp(-buf[i]));
            CudaDriverApi.cuMemcpyHtoD_v2(bufDev, (nint)pBuf, (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    private void LaunchSiluHostFallback(nint bufDev, long elems)
    {
        _stream.Synchronize();
        float[] buf = new float[elems];
        fixed (float* pBuf = buf)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pBuf, bufDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            for (long i = 0; i < elems; i++)
                buf[i] = buf[i] / (1f + MathF.Exp(-buf[i]));
            CudaDriverApi.cuMemcpyHtoD_v2(bufDev, (nint)pBuf, (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    private void LaunchSigmoidMulHostFallback(nint aDev, nint bDev, long elems)
    {
        _stream.Synchronize();
        float[] a = new float[elems];
        float[] b = new float[elems];
        fixed (float* pA = a)
        fixed (float* pB = b)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pA, aDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)pB, bDev, (nuint)(elems * sizeof(float))).ThrowOnError();
            for (long i = 0; i < elems; i++)
                a[i] *= 1f / (1f + MathF.Exp(-b[i]));
            CudaDriverApi.cuMemcpyHtoD_v2(aDev, (nint)pA, (nuint)(elems * sizeof(float))).ThrowOnError();
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Disposal
    // ──────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _layers.Length; i++)
        {
            FreeLayer(ref _layers[i]);
        }

        FreeIfNonZero(ref _dequantScratchF16Weight);
        FreeIfNonZero(ref _activF16InScratch);
        FreeIfNonZero(ref _activF16OutScratch);

        nint outNormPtr = _outputNormDevice;
        if (outNormPtr != 0) CudaDriverApi.cuMemFree_v2(outNormPtr);
        if (_ownsOutputDevice)
        {
            nint outPtr = _outputDevice;
            if (outPtr != 0) CudaDriverApi.cuMemFree_v2(outPtr);
        }
        nint embPtr = _tokenEmbedDevice;
        if (embPtr != 0) CudaDriverApi.cuMemFree_v2(embPtr);

        if (_f16KCache is not null)
        {
            for (int i = 0; i < _f16KCache.Length; i++)
            {
                if (_f16KCache[i] != 0) CudaDriverApi.cuMemFree_v2(_f16KCache[i]);
                if (_f16VCache![i] != 0) CudaDriverApi.cuMemFree_v2(_f16VCache[i]);
            }
            _f16KCache = null;
            _f16VCache = null;
        }
        FreeIfNonZero(ref _f16KvWriteStaging);
        if (_f32KvReadStagingK is not null)
        {
            for (int i = 0; i < _f32KvReadStagingK.Length; i++)
            {
                if (_f32KvReadStagingK[i] != 0) CudaDriverApi.cuMemFree_v2(_f32KvReadStagingK[i]);
                if (_f32KvReadStagingV![i] != 0) CudaDriverApi.cuMemFree_v2(_f32KvReadStagingV[i]);
            }
            _f32KvReadStagingK = null;
            _f32KvReadStagingV = null;
        }
        FreeIfNonZero(ref _attnSplitKvPartialMax);
        FreeIfNonZero(ref _attnSplitKvPartialSum);
        FreeIfNonZero(ref _attnSplitKvPartialOut);
        FreeIfNonZero(ref _attnGqaSplitPartialMax);
        FreeIfNonZero(ref _attnGqaSplitPartialSum);
        FreeIfNonZero(ref _attnGqaSplitPartialOut);

        _state.Dispose();
        _gdnCache.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();

        GC.SuppressFinalize(this);
    }

    private static void FreeLayer(ref DeviceLayer layer)
    {
        FreeIfNonZero(ref layer.AttnNormWeightDevice);
        FreeIfNonZero(ref layer.PostAttnNormWeightDevice);

        if (layer.Gdn is { } gdn)
        {
            FreeIfNonZero(ref gdn.QkvDevice);
            FreeIfNonZero(ref gdn.GateDevice);
            FreeIfNonZero(ref gdn.AlphaDevice);
            FreeIfNonZero(ref gdn.BetaDevice);
            FreeIfNonZero(ref gdn.Conv1dWeightDevice);
            FreeIfNonZero(ref gdn.Conv1dBiasDevice);
            FreeIfNonZero(ref gdn.ADevice);
            FreeIfNonZero(ref gdn.DtBiasDevice);
            FreeIfNonZero(ref gdn.SsmNormDevice);
            FreeIfNonZero(ref gdn.OutDevice);
            layer.Gdn = gdn;
        }
        if (layer.FullAttn is { } attn)
        {
            FreeIfNonZero(ref attn.QDevice);
            FreeIfNonZero(ref attn.KDevice);
            FreeIfNonZero(ref attn.VDevice);
            FreeIfNonZero(ref attn.ODevice);
            FreeIfNonZero(ref attn.QNormDevice);
            FreeIfNonZero(ref attn.KNormDevice);
            layer.FullAttn = attn;
        }
        FreeIfNonZero(ref layer.GateWeight);
        FreeIfNonZero(ref layer.UpWeight);
        FreeIfNonZero(ref layer.DownWeight);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Static helpers
    // ──────────────────────────────────────────────────────────────────────

    private static nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        return ptr;
    }

    private static void CopyHtoD(nint dst, nint src, long bytes)
    {
        CudaDriverApi.cuMemcpyHtoD_v2(dst, src, (nuint)bytes).ThrowOnError();
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            CudaDriverApi.cuMemFree_v2(ptr);
            ptr = 0;
        }
    }

    private static nint UploadF32Tensor(nint dataBase, GgufTensorDescriptor desc, int expectedElems)
    {
        float[] host = new float[expectedElems];
        Dequantize.ToFloat32(dataBase + (nint)desc.DataOffset, expectedElems,
            desc.QuantizationType, host);
        nint device = AllocDevice((long)expectedElems * sizeof(float));
        fixed (float* p = host)
        {
            CopyHtoD(device, (nint)p, (long)expectedElems * sizeof(float));
        }
        return device;
    }

    // Lazily-loaded, process-wide singleton for the PQ2_0 weight-repack kernel. Kept independent
    // of the per-model CudaKernels instance (`kernels` in LoadFromGguf) rather than threaded
    // through as a parameter: UploadRawTensor and its dozens of call sites (LoadLayerDevice,
    // LoadGdnLayerDevice, LoadFullAttnLayerDevice — all `private static`, running before this
    // model instance or its CudaKernels even exist) stay completely unchanged, which is the
    // entire point of doing the repack at this one choke point instead of threading split-layout
    // awareness through every layer's weight struct and every Gemm/TryFusedPQ2_0Gemm2 call site.
    // Mirrors CudaKernels.LaunchPQ2_0RepackSplitF16's launch shape exactly (same PTX, same grid
    // formula) — that instance-based wrapper is what tests use; this static path exists solely
    // because this call site structurally cannot hold a CudaKernels instance without a much
    // larger, invasive restructuring of the (static, pre-construction) loading pipeline.
    private static CudaModule? s_pq2_0RepackModule;
    private static nint s_pq2_0RepackFunc;
    // The CUDA context s_pq2_0RepackModule was loaded into (cuModuleLoad binds a module to
    // whichever context is current at load time). A CudaModule/function handle from a since-
    // destroyed context is invalid in any other context ("CUDA error 400: invalid resource
    // handle") — loading a second Qwen3HybridDense model (a fresh CudaContext.Create per
    // LoadFromGguf call, see that method) after the first model's Dispose() reuses this
    // process-wide cache across two different (one now-dead) contexts otherwise. Track the
    // context the cached module belongs to and reload whenever the CURRENT context differs.
    private static nint s_pq2_0RepackContext;

    private static nint EnsurePq2_0RepackFunc()
    {
        CudaDriverApi.cuCtxGetCurrent(out nint currentCtx).ThrowOnError();
        if (s_pq2_0RepackModule is null || currentCtx != s_pq2_0RepackContext)
        {
            string ptxDir = Path.Combine(AppContext.BaseDirectory, "ptx");
            s_pq2_0RepackModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "pq2_0_repack.ptx"));
            s_pq2_0RepackFunc = s_pq2_0RepackModule.GetFunction("pq2_0_repack_split_f16");
            s_pq2_0RepackContext = currentCtx;
        }
        return s_pq2_0RepackFunc;
    }

    private static nint UploadRawTensor(nint dataBase, GgufTensorDescriptor desc)
    {
        int innerDim = desc.Shape[0];
        long outerDim = desc.Shape.ElementCount / innerDim;
        long bytes = Dequantize.RowByteSize(innerDim, desc.QuantizationType) * outerDim;
        nint device = AllocDevice(bytes);
        CopyHtoD(device, dataBase + (nint)desc.DataOffset, bytes);

        if (desc.QuantizationType != QuantizationType.PQ2_0)
            return device;

        // One-time load-time repack: reorder from dotLLM's interleaved on-disk PQ2_0 layout
        // (per-group scale immediately followed by that group's 32 code bytes — never
        // 32-byte-aligned, PQ2_0_GROUP_BYTES=34) into the split layout pq2_0_gemv.cu /
        // dequant_pq2_0.cu now read (all scales, then all codes — see those files'
        // "Split-layout addressing" notes and pq2_0_repack.cu's file header). This makes every
        // group's hot-path code read unconditionally 32-byte-aligned with zero added
        // synchronization in the decode kernel — see native/kernels/pq2_0_gemv.cu's file header
        // for why this is preferred over an in-kernel batched-staging fix (measured regression
        // from added __syncwarp() barriers on a parallel investigation branch). Synchronous:
        // this runs once per tensor at load time, never on the inference hot path. The temporary
        // interleaved buffer is freed immediately after, so this is not a steady-state VRAM
        // increase — only a transient extra allocation (same size as the split buffer) during
        // the repack itself.
        int n = (int)outerDim;
        int k = innerDim;
        long splitBytes = CudaKernels.PQ2_0SplitLayoutBytes(n, k);
        nint splitDevice = AllocDevice(splitBytes);

        nint repackFunc = EnsurePq2_0RepackFunc();
        long totalGroups = (long)n * (k / 128);
        const int blockSize = 256;       // must match pq2_0_repack.cu's __launch_bounds__(256)
        const int warpsPerBlock = blockSize / 32;
        const int maxGridSize = 256;      // mirrors CudaKernels.MaxDequantGridSize
        uint gridDim = (uint)Math.Min((totalGroups + warpsPerBlock - 1) / warpsPerBlock, maxGridSize);
        if (gridDim == 0) gridDim = 1;

        nint srcArg = device, dstArg = splitDevice;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &srcArg, &dstArg, &nArg, &kArg };
        CudaDriverApi.cuLaunchKernel(repackFunc,
                gridDim, 1, 1, blockSize, 1, 1,
                0, 0, (nint)args, 0).ThrowOnError();
        CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();   // synchronous — one-time load-time cost, not hot path

        FreeIfNonZero(ref device);
        return splitDevice;
    }

    private static void UpdateMaxTile(ref long max, long candidate)
    {
        if (candidate > max) max = candidate;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Per-layer device-side bundles
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Per-layer device pointers (norms + token-mixing + dense FFN).</summary>
    internal struct DeviceLayer
    {
        public nint AttnNormWeightDevice;
        public nint PostAttnNormWeightDevice;
        public DeviceGdn? Gdn;
        public DeviceFullAttn? FullAttn;

        public nint GateWeight;
        public QuantizationType GateQt;
        public int GateInputDim;
        public int GateOutputDim;

        public nint UpWeight;
        public QuantizationType UpQt;
        public int UpInputDim;
        public int UpOutputDim;

        public nint DownWeight;
        public QuantizationType DownQt;
        public int DownInputDim;
        public int DownOutputDim;
    }

    /// <summary>Device-side GDN token-mixing weights.</summary>
    internal struct DeviceGdn
    {
        public nint QkvDevice;
        public QuantizationType QkvQt;
        public int QkvInputDim;
        public int QkvOutputDim;

        public nint GateDevice;
        public QuantizationType GateQt;
        public int GateInputDim;
        public int GateOutputDim;

        public nint AlphaDevice;
        public QuantizationType AlphaQt;
        public int AlphaInputDim;
        public int AlphaOutputDim;

        public nint BetaDevice;
        public QuantizationType BetaQt;
        public int BetaInputDim;
        public int BetaOutputDim;

        public nint Conv1dWeightDevice;
        public nint Conv1dBiasDevice;
        public nint ADevice;
        public nint DtBiasDevice;
        public nint SsmNormDevice;

        public nint OutDevice;
        public QuantizationType OutQt;
        public int OutInputDim;
        public int OutOutputDim;
    }

    /// <summary>Device-side full-attention weights.</summary>
    internal struct DeviceFullAttn
    {
        public nint QDevice;
        public QuantizationType QQt;
        public int QInputDim;
        public int QOutputDim;

        public nint KDevice;
        public QuantizationType KQt;
        public int KInputDim;
        public int KOutputDim;

        public nint VDevice;
        public QuantizationType VQt;
        public int VInputDim;
        public int VOutputDim;

        public nint ODevice;
        public QuantizationType OQt;
        public int OInputDim;
        public int OOutputDim;

        public int NumKvHeads;
        public nint QNormDevice;
        public nint KNormDevice;
    }
}
