using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the Mamba-3 pure-SSM architecture (issue #346). F32
/// activations and weights throughout, mirroring
/// <see cref="DotLLM.Models.Architectures.Mamba3TransformerModel"/> (CPU) and
/// <see cref="DotLLM.Vulkan.VulkanMamba3TransformerModel"/> on the GPU. Per-token
/// preprocessing (softplus/sigmoid/RMSNorm+bias/qk_pre_dot/scale) runs host-side,
/// mirroring the Vulkan port's proven design decision (see that class's own remarks
/// for the rationale) rather than fusing it into a device kernel on day one.
/// </summary>
/// <remarks>
/// <para>
/// <b>Homogeneous, not hybrid.</b> Every layer is <c>{RMSNorm, Mamba3Block, residual
/// add}</c> — no attention layers, no MoE, no LoRA path (CPU's own
/// <c>Mamba3TransformerModel.ForwardBatch</c> rejects LoRA adapters outright). This
/// class is therefore structurally simpler than
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>: no <c>HybridLayerLayout</c>
/// dispatch, no KV-cache, no MTP head.
/// </para>
/// <para>
/// <b>Safetensors only.</b> Mamba-3 has no GGUF tensor-naming convention on any
/// dotLLM backend — see this plan's "Deviations from the issue text" section. There is
/// no <c>LoadFromGguf</c> on this class.
/// </para>
/// <para>
/// <b>Weight loading strategy.</b> Reuses the existing CPU
/// <see cref="Mamba3WeightLoader.Load"/> to resolve tensor names/shapes/diagnostics
/// against a <see cref="Mamba3Weights"/> (host-side, mmap-backed handles), then
/// uploads each populated handle to a device buffer. This avoids duplicating
/// tensor-name/shape validation in CUDA-specific code — the same "load CPU-side, then
/// upload" strategy <c>VulkanMamba3TransformerModel.LoadFromSafetensors</c> already
/// uses.
/// </para>
/// <para>
/// <b>Not yet a complete <see cref="IModel"/>.</b> This task (Task 8 of issue #346)
/// delivers only the skeleton, weight upload, and <see cref="LoadFromSafetensors"/> —
/// the <c>Forward</c> overloads throw <see cref="NotImplementedException"/> until
/// Task 9 adds the real forward pass and <c>EnsureScratchCapacity</c>.
/// </para>
/// </remarks>
public sealed unsafe class CudaMamba3TransformerModel : IModel
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaKernels _kernels;
    private readonly Mamba3Config _m3;
    private readonly int _numLayers;

    private readonly DeviceLayer[] _layers;
    private readonly nint _tokenEmbedDevice;   // [vocab, hidden]
    private readonly nint _finalNormDevice;    // [hidden]
    private readonly nint _lmHeadDevice;       // [vocab, hidden] — aliases _tokenEmbedDevice when tied
    private readonly bool _lmHeadOwnsDevice;

    // Forward scratch — device buffers sized to the widest seqLen seen so far,
    // grown power-of-two on demand by EnsureScratchCapacity (Task 9). Mirrors
    // Mamba3ForwardScratch (CPU) / VulkanMamba3ForwardScratch (Vulkan).
    // Explicitly zero-initialized (rather than relying on the implicit default) so the
    // compiler's CS0649 ("field is never assigned") does not trip TreatWarningsAsErrors —
    // Task 9's EnsureScratchCapacity is the real assignor; until then these are legitimately
    // always 0 (== FreeIfNonZero's "not yet allocated" sentinel).
    private int _scratchCapacity;
    private nint _hidden = 0, _residual = 0, _normOut = 0, _blockOut = 0;      // [cap, hidden]
    private nint _projDevice = 0;                                   // [cap, dInProj] (in_proj GEMM output)
    private nint _xDevice = 0, _zDevice = 0, _yScanDevice = 0;               // [cap, dInner]
    private nint _dtDevice = 0, _adtDevice = 0, _trapDevice = 0, _gammaDevice = 0, _scaleDevice = 0, _qkPreDotDevice = 0; // [cap, nHead]
    private nint _anglesRawDevice = 0;                               // [cap, numRopeAngles]
    private nint _bDevice = 0, _cDevice = 0;                             // [cap, effRank, nHead, dState]
    private nint _coefDevice = 0;                                    // [nHead] — chunk-boundary coefficients
    private nint _logitsDevice;                                  // [vocab] — last-token logits (allocated once)

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => ScratchAllocatedBytes();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState()
    {
        _context.MakeCurrent();
        return new CudaMamba3StateCache(_m3, _numLayers);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Deliberately a no-op, matching CPU's <c>Mamba3TransformerModel.ResetSequenceState</c>:
    /// this model owns no persistent recurrent state of its own — every forward that is
    /// not given a caller-supplied <see cref="CudaMamba3StateCache"/> allocates and
    /// disposes a fresh ephemeral one for that call (see Task 9), so consecutive
    /// uncached forwards are already independent sequences.
    /// </remarks>
    public void ResetSequenceState() { }

    /// <inheritdoc/>
    /// <remarks>
    /// Stub — implemented in Task 9 alongside <c>ForwardBatch</c> and
    /// <c>EnsureScratchCapacity</c>. Declared now so this class satisfies
    /// <see cref="IModel"/> and the rest of the codebase can reference
    /// <see cref="CudaMamba3TransformerModel"/> (e.g. Task 10's
    /// <c>CudaModelLoader.LoadMamba3FromSafetensors</c>) ahead of the forward pass
    /// landing.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => throw new NotImplementedException(
            $"{nameof(CudaMamba3TransformerModel)}.Forward is implemented in Task 9 (issue #346).");

    /// <inheritdoc/>
    /// <remarks>See remarks on the 3-argument <c>Forward</c> overload.</remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => throw new NotImplementedException(
            $"{nameof(CudaMamba3TransformerModel)}.Forward is implemented in Task 9 (issue #346).");

    private readonly record struct DeviceLayer(
        nint Norm, nint InProj, nint OutProj, nint BNorm, nint CNorm,
        nint BBias, nint CBias, nint D, nint DtBias,
        nint MimoZ, nint MimoO);

    private CudaMamba3TransformerModel(
        ModelConfig config, CudaContext context, CudaStream stream, CudaCublasHandle cublas,
        CudaKernels kernels, DeviceLayer[] layers, nint tokenEmbedDevice, nint finalNormDevice,
        nint lmHeadDevice, bool lmHeadOwnsDevice)
    {
        Config = config;
        _m3 = config.Mamba3Config!;
        _numLayers = config.NumLayers;
        _context = context;
        _stream = stream;
        _cublas = cublas;
        _kernels = kernels;
        _layers = layers;
        _tokenEmbedDevice = tokenEmbedDevice;
        _finalNormDevice = finalNormDevice;
        _lmHeadDevice = lmHeadDevice;
        _lmHeadOwnsDevice = lmHeadOwnsDevice;

        cublas.SetStream(stream);

        long vocabBytes = (long)config.VocabSize * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _logitsDevice, (nuint)vocabBytes).ThrowOnError();
    }

    /// <summary>
    /// Loads a Mamba-3 model from an opened HF-convention safetensors source onto the
    /// specified GPU. Mirrors
    /// <see cref="DotLLM.Models.Architectures.Mamba3TransformerModel.LoadFromSafetensors"/>
    /// (CPU) and <c>VulkanMamba3TransformerModel.LoadFromSafetensors</c>'s "resolve on
    /// CPU, upload to device" strategy.
    /// </summary>
    /// <param name="file">An opened safetensors source positioned at a Mamba-3 checkpoint. Must outlive the returned model's load (not required to outlive the model itself — weights are uploaded, then the CPU-side handles are released).</param>
    /// <param name="config">Model config with <see cref="ModelConfig.Mamba3Config"/> populated and <see cref="ModelConfig.Architecture"/> == <see cref="Architecture.Mamba3"/>.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect (<c>AppContext.BaseDirectory/ptx</c>).</param>
    /// <exception cref="InvalidDataException">One or more required Mamba-3 tensors are missing/malformed — see <see cref="Mamba3Weights.Report"/> via the thrown message.</exception>
    public static CudaMamba3TransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Mamba3)
            throw new ArgumentException(
                $"CudaMamba3TransformerModel requires Architecture.Mamba3, got {config.Architecture}.",
                nameof(config));
        if (config.Mamba3Config is null)
            throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated for CudaMamba3TransformerModel.",
                nameof(config));

        // Reuse the CPU loader for tensor resolution/shape validation/diagnostics —
        // see the class doc for why this is preferable to a CUDA-specific re-implementation.
        Mamba3Weights weights = Mamba3WeightLoader.Load(config, file);
        try
        {
            if (weights.Report.HasMissingRequired)
                throw new InvalidDataException(
                    $"Mamba-3 weights are incomplete ({weights.Report.MissingRequiredCount} required tensors "
                    + "missing). Inspect Mamba3Weights.Report.Problems before attempting a CUDA load.");

            var context = CudaContext.Create(deviceId);
            var stream = CudaStream.Create();
            var cublas = CudaCublasHandle.Create();
            ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
            var kernels = new CudaKernels(ptxDir);

            var m3 = config.Mamba3Config;
            int hidden = config.HiddenSize;
            int vocab = config.VocabSize;
            int dInner = m3.DInner;
            int nHead = m3.NumHeads;
            int dState = m3.StateSize;
            int effRank = m3.IsMimo ? m3.MimoRank : 1;
            int bcBiasElems = nHead * effRank * dState;
            int mimoElems = m3.IsMimo ? nHead * m3.MimoRank * m3.HeadDim : 0;

            nint tokenEmbedDevice = UploadF32(weights.TokenEmbedding, (long)vocab * hidden, stream.Handle);
            nint finalNormDevice = UploadF32(weights.FinalNorm, hidden, stream.Handle);

            bool tied = weights.LmHead.Pointer == weights.TokenEmbedding.Pointer;
            nint lmHeadDevice = tied ? tokenEmbedDevice : UploadF32(weights.LmHead, (long)vocab * hidden, stream.Handle);

            var layers = new DeviceLayer[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
            {
                ref readonly var lw = ref weights.Layers[i];
                layers[i] = new DeviceLayer(
                    Norm: UploadF32(lw.Norm, hidden, stream.Handle),
                    InProj: UploadF32(lw.InProj, (long)m3.InputProjectionDim * hidden, stream.Handle),
                    OutProj: UploadF32(lw.OutProj, (long)hidden * dInner, stream.Handle),
                    BNorm: UploadF32(lw.BNorm, dState, stream.Handle),
                    CNorm: UploadF32(lw.CNorm, dState, stream.Handle),
                    BBias: UploadF32(lw.BBias, bcBiasElems, stream.Handle),
                    CBias: UploadF32(lw.CBias, bcBiasElems, stream.Handle),
                    D: UploadF32(lw.D, nHead, stream.Handle),
                    DtBias: UploadF32(lw.DtBias, nHead, stream.Handle),
                    MimoZ: m3.IsMimo ? UploadF32(lw.MimoZ, mimoElems, stream.Handle) : 0,
                    MimoO: m3.IsMimo ? UploadF32(lw.MimoO, mimoElems, stream.Handle) : 0);
            }

            // All H2D copies above were issued async on `stream` — synchronize before
            // releasing the CPU-side (mmap-backed) weight handles.
            stream.Synchronize();

            return new CudaMamba3TransformerModel(
                config, context, stream, cublas, kernels, layers,
                tokenEmbedDevice, finalNormDevice, lmHeadDevice, lmHeadOwnsDevice: !tied);
        }
        finally
        {
            // weights.Dispose() is a no-op for mmap-backed (OwnsMemory=false) handles —
            // the safetensors file itself is the lifetime anchor, and this method does
            // not retain a reference to it after the H2D copies above complete.
            weights.Dispose();
        }
    }

    /// <summary>
    /// Uploads a populated F32 <see cref="Mamba3TensorHandle"/> to a freshly-allocated
    /// device buffer via an async H2D copy on <paramref name="stream"/>. Returns 0 (no
    /// allocation) for an unpopulated handle — e.g. <c>MimoZ</c>/<c>MimoO</c> on a SISO
    /// checkpoint.
    /// </summary>
    private static nint UploadF32(Mamba3TensorHandle handle, long expectedElements, nint stream)
    {
        if (!handle.IsPopulated) return 0;
        if (handle.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"CudaMamba3TransformerModel requires F32 tensors; got {handle.SourceDType}. "
                + "Quantized/F16 Mamba-3 weights are not yet supported on CUDA (CPU-parity scope, issue #346).");

        long bytes = expectedElements * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoDAsync_v2(devPtr, handle.Pointer, (nuint)bytes, stream).ThrowOnError();
        return devPtr;
    }

    private long ScratchAllocatedBytes()
    {
        if (_scratchCapacity == 0) return 0;
        long cap = _scratchCapacity;
        int hidden = Config.HiddenSize, dInner = _m3.DInner, nHead = _m3.NumHeads;
        int dState = _m3.StateSize, numRopeAngles = _m3.NumRopeAngles;
        int effRank = _m3.IsMimo ? _m3.MimoRank : 1;
        long floats = cap * hidden * 4L                       // hidden/residual/normOut/blockOut
                    + cap * _m3.InputProjectionDim             // proj
                    + cap * dInner * 3L                        // x/z/yScan
                    + cap * nHead * 6L                         // dt/adt/trap/gamma/scale/qkPreDot
                    + cap * numRopeAngles                      // anglesRaw
                    + cap * effRank * nHead * dState * 2L;     // b/c
        return floats * sizeof(float);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var l in _layers)
        {
            FreeIfNonZero(l.Norm); FreeIfNonZero(l.InProj); FreeIfNonZero(l.OutProj);
            FreeIfNonZero(l.BNorm); FreeIfNonZero(l.CNorm); FreeIfNonZero(l.BBias); FreeIfNonZero(l.CBias);
            FreeIfNonZero(l.D); FreeIfNonZero(l.DtBias); FreeIfNonZero(l.MimoZ); FreeIfNonZero(l.MimoO);
        }
        FreeIfNonZero(_tokenEmbedDevice);
        FreeIfNonZero(_finalNormDevice);
        if (_lmHeadOwnsDevice) FreeIfNonZero(_lmHeadDevice);
        FreeIfNonZero(_logitsDevice);
        FreeScratch();

        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
        GC.SuppressFinalize(this);
    }

    private static void FreeIfNonZero(nint ptr)
    {
        if (ptr != 0) CudaDriverApi.cuMemFree_v2(ptr);
    }

    private void FreeScratch()
    {
        FreeIfNonZero(_hidden); FreeIfNonZero(_residual); FreeIfNonZero(_normOut); FreeIfNonZero(_blockOut);
        FreeIfNonZero(_projDevice); FreeIfNonZero(_xDevice); FreeIfNonZero(_zDevice); FreeIfNonZero(_yScanDevice);
        FreeIfNonZero(_dtDevice); FreeIfNonZero(_adtDevice); FreeIfNonZero(_trapDevice);
        FreeIfNonZero(_gammaDevice); FreeIfNonZero(_scaleDevice); FreeIfNonZero(_qkPreDotDevice);
        FreeIfNonZero(_anglesRawDevice); FreeIfNonZero(_bDevice); FreeIfNonZero(_cDevice); FreeIfNonZero(_coefDevice);
        _scratchCapacity = 0;
    }
}
