using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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
/// <b>SISO + MIMO complete <see cref="IModel"/>.</b> Task 8 delivered the skeleton,
/// weight upload, and <see cref="LoadFromSafetensors"/>; Task 9 added the real forward
/// pass (<c>EnsureScratchCapacity</c>, per-layer dispatch, host-side per-token prep) for
/// SISO checkpoints; Task 14 wired MIMO checkpoints (<see cref="Mamba3Config.IsMimo"/>)
/// into the same <c>ForwardCore</c> via a per-layer branch (<see cref="HostPrepareMimo"/>,
/// halved-RoPE, rank-parameterized chunk-boundary correction,
/// <c>LaunchMamba3SsdScanMimoF32</c>). No public MIMO checkpoint exists — MIMO
/// correctness evidence is the Task 13 kernel-level unit test plus this task's synthetic
/// end-to-end fixture test (mirrors CPU/Vulkan's own MIMO coverage; see
/// <c>docs/ROADMAP.md</c> step 60f).
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
    private nint _dtDevice = 0, _adtDevice = 0, _gammaDevice = 0, _scaleDevice = 0, _qkPreDotDevice = 0; // [cap, nHead]
    private nint _anglesRawDevice = 0;                               // [cap, numRopeAngles]
    private nint _bDevice = 0, _cDevice = 0;                             // [cap, effRank, nHead, dState]
    private nint _coefDevice = 0;                                    // [nHead] — chunk-boundary coefficients
    private nint _logitsDevice;                                  // [vocab] — last-token logits (allocated once)
    // #382 item 4: the final RMSNorm only needs to touch the last token — this is a fixed
    // [hidden]-sized device scratch for that single row (allocated once, like _logitsDevice,
    // since hiddenSize never changes for a loaded model), NOT part of the seqLen-scaled
    // scratch capacity growth path below.
    private readonly nint _lastHiddenDevice;

    // #382 item 1: reusable host-side per-forward scratch for HostPrepareSiso/HostPrepareMimo
    // — grown alongside the device scratch in EnsureScratchCapacity instead of `new float[...]`
    // per layer per forward (the previous version allocated ~12 arrays per layer per forward;
    // `proj` alone is seqLen*dInProj floats, LOH-sized on the real 370M checkpoint at any
    // realistic prefill length). Each array's *capacity* is cap-scaled; callers must still slice
    // to the current seqLen-scaled length before use (see HostPrepareSiso/Mimo) — using the full
    // array length after a capacity grow would upload/read stale or out-of-range data on a
    // smaller subsequent forward.
    private float[] _hostProj = [];
    private float[] _hostX = [];
    private float[] _hostZ = [];
    private float[] _hostDt = [];
    private float[] _hostAdt = [];
    private float[] _hostTrap = [];   // host-only — never uploaded to device (#382 item 3).
    private float[] _hostGamma = [];
    private float[] _hostScale = [];
    private float[] _hostAnglesRaw = [];
    private float[] _hostB = [];
    private float[] _hostC = [];
    private float[] _hostQkPreDot = [];
    // Chunk-boundary coefficient — [nHead], fixed size (no seqLen dependence), allocated once
    // in the constructor like _coefDevice rather than grown in EnsureScratchCapacity.
    private readonly float[] _hostCoef;

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

    /// <remarks>
    /// The <c>*Host</c> fields (<c>DtBiasHost</c>/<c>BNormHost</c>/<c>CNormHost</c>/
    /// <c>BBiasHost</c>/<c>CBiasHost</c>) are #382 item 2's fix: host-side copies of the small
    /// per-layer bias/norm-weight tensors that <see cref="HostPrepareSiso"/>/
    /// <see cref="HostPrepareMimo"/> need on the CPU for the softplus/sigmoid/RMSNorm+bias math.
    /// Populated once in <see cref="LoadFromSafetensors"/> (mirrors
    /// <c>VulkanMamba3Weights.LayerBuffers</c>'s <c>*Host</c> fields) so the per-forward path no
    /// longer re-downloads them from the matching device fields via <c>cuMemcpyDtoH_v2</c> on
    /// every layer of every call. The device-side <c>DtBias</c>/<c>BNorm</c>/<c>CNorm</c>/
    /// <c>BBias</c>/<c>CBias</c> buffers are still uploaded and kept (VRAM cost is trivial —
    /// these are nHead/dState-sized) for the documented future fused device-kernel host-prep
    /// replacement (see this class's doc remarks); only <c>D</c> is actually read as a device
    /// pointer by a kernel today (the SSD scan).
    /// </remarks>
    private readonly record struct DeviceLayer(
        nint Norm, nint InProj, nint OutProj, nint BNorm, nint CNorm,
        nint BBias, nint CBias, nint D, nint DtBias,
        nint MimoZ, nint MimoO,
        float[] DtBiasHost, float[] BNormHost, float[] CNormHost, float[] BBiasHost, float[] CBiasHost);

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
        // #382 item 4: fixed [hidden]-sized scratch for the final RMSNorm's last-row-only input
        // (see ForwardCore's tail). Allocated once here, not cap-scaled — hiddenSize is constant
        // for the lifetime of a loaded model.
        long lastHiddenBytes = (long)config.HiddenSize * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _lastHiddenDevice, (nuint)lastHiddenBytes).ThrowOnError();
        _hostCoef = new float[_m3.NumHeads];
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

            // #383: context creation is the one call in this sequence that cannot leak on its
            // own throw (nothing has been allocated yet), so it stays outside the try/catch
            // below — everything created from here on (stream/cublas/kernels/device buffers)
            // is disposed on any failure before rethrowing.
            var context = CudaContext.Create(deviceId);
            CudaStream? stream = null;
            CudaCublasHandle? cublas = null;
            CudaKernels? kernels = null;
            // Ledger of every device buffer allocated below. UploadF32 appends to this the
            // instant cuMemAlloc succeeds (before the H2D copy), so a throw mid-upload — e.g.
            // a BF16 tensor tripping the F32-only guard, or a later buffer hitting device OOM —
            // still leaves every earlier buffer (including the ones inside a partially-built
            // DeviceLayer that was never assigned into `layers`) recoverable. UploadF32 is
            // skipped entirely for the tied lm_head case (see below), so the alias is never
            // double-added/double-freed.
            var allocs = new List<nint>();
            try
            {
                stream = CudaStream.Create();
                cublas = CudaCublasHandle.Create();
                ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
                kernels = new CudaKernels(ptxDir);

                var m3 = config.Mamba3Config;
                int hidden = config.HiddenSize;
                int vocab = config.VocabSize;
                int dInner = m3.DInner;
                int nHead = m3.NumHeads;
                int dState = m3.StateSize;
                int effRank = m3.IsMimo ? m3.MimoRank : 1;
                int bcBiasElems = nHead * effRank * dState;
                int mimoElems = m3.IsMimo ? nHead * m3.MimoRank * m3.HeadDim : 0;

                nint tokenEmbedDevice = UploadF32(weights.TokenEmbedding, (long)vocab * hidden, stream.Handle, allocs);
                nint finalNormDevice = UploadF32(weights.FinalNorm, hidden, stream.Handle, allocs);

                bool tied = weights.LmHead.Pointer == weights.TokenEmbedding.Pointer;
                nint lmHeadDevice = tied ? tokenEmbedDevice : UploadF32(weights.LmHead, (long)vocab * hidden, stream.Handle, allocs);

                var layers = new DeviceLayer[config.NumLayers];
                for (int i = 0; i < config.NumLayers; i++)
                {
                    ref readonly var lw = ref weights.Layers[i];
                    layers[i] = new DeviceLayer(
                        Norm: UploadF32(lw.Norm, hidden, stream.Handle, allocs),
                        InProj: UploadF32(lw.InProj, (long)m3.InputProjectionDim * hidden, stream.Handle, allocs),
                        OutProj: UploadF32(lw.OutProj, (long)hidden * dInner, stream.Handle, allocs),
                        BNorm: UploadF32(lw.BNorm, dState, stream.Handle, allocs),
                        CNorm: UploadF32(lw.CNorm, dState, stream.Handle, allocs),
                        BBias: UploadF32(lw.BBias, bcBiasElems, stream.Handle, allocs),
                        CBias: UploadF32(lw.CBias, bcBiasElems, stream.Handle, allocs),
                        D: UploadF32(lw.D, nHead, stream.Handle, allocs),
                        DtBias: UploadF32(lw.DtBias, nHead, stream.Handle, allocs),
                        MimoZ: m3.IsMimo ? UploadF32(lw.MimoZ, mimoElems, stream.Handle, allocs) : 0,
                        MimoO: m3.IsMimo ? UploadF32(lw.MimoO, mimoElems, stream.Handle, allocs) : 0,
                        // #382 item 2: host-side copies of the same handles, taken while the
                        // mmap-backed CPU pointers are still valid (before `weights.Dispose()`
                        // in the outer finally / before the caller's `file` may be closed) — see
                        // DeviceLayer's doc comment for why HostPrepareSiso/Mimo need these.
                        DtBiasHost: CopyHostF32(lw.DtBias, nHead),
                        BNormHost: CopyHostF32(lw.BNorm, dState),
                        CNormHost: CopyHostF32(lw.CNorm, dState),
                        BBiasHost: CopyHostF32(lw.BBias, bcBiasElems),
                        CBiasHost: CopyHostF32(lw.CBias, bcBiasElems));
                }

                // All H2D copies above were issued async on `stream` — synchronize before
                // releasing the CPU-side (mmap-backed) weight handles.
                stream.Synchronize();

                return new CudaMamba3TransformerModel(
                    config, context, stream, cublas, kernels, layers,
                    tokenEmbedDevice, finalNormDevice, lmHeadDevice, lmHeadOwnsDevice: !tied);
            }
            catch
            {
                // Best-effort drain of any pending async H2D copies before freeing the buffers
                // they target — cheap, and removes any question of a copy racing the frees below.
                try { stream?.Synchronize(); } catch { /* already failing; nothing more to report */ }

                for (int i = allocs.Count - 1; i >= 0; i--)
                    FreeIfNonZero(allocs[i]);

                kernels?.Dispose();
                cublas?.Dispose();
                stream?.Dispose();
                // CudaContext.Create (above) makes the context current on THIS thread, and this
                // catch runs synchronously on the same thread — no MakeCurrent() call is needed
                // here (matches #368's convention: only cross-thread entry points need an
                // explicit rebind before touching CUDA state).
                context.Dispose();
                throw;
            }
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
    /// Copies a populated F32 <see cref="Mamba3TensorHandle"/> straight into a managed
    /// <c>float[]</c> — the host-side companion to <see cref="UploadF32"/> for the small
    /// per-layer tensors <see cref="HostPrepareSiso"/>/<see cref="HostPrepareMimo"/> need
    /// on the CPU (#382 item 2). Runs synchronously (unlike <see cref="UploadF32"/>'s async
    /// H2D copy) so it is safe to call from within the same load-time loop that also stages
    /// the device upload — the source pointer is guaranteed valid at this point regardless of
    /// when the async H2D copy above it actually completes. Returns a zero-filled array (not
    /// <see cref="Array.Empty{T}"/>) for an unpopulated handle, matching the zero-bias
    /// semantics the removed <c>DownloadF32</c> helper had for a null device pointer — callers
    /// index these arrays unconditionally.
    /// </summary>
    private static unsafe float[] CopyHostF32(Mamba3TensorHandle handle, long expectedElements)
    {
        var host = new float[expectedElements];
        if (!handle.IsPopulated) return host;
        fixed (float* p = host)
            Buffer.MemoryCopy((void*)handle.Pointer, p, host.Length * sizeof(float), expectedElements * sizeof(float));
        return host;
    }

    /// <summary>
    /// Uploads a populated F32 <see cref="Mamba3TensorHandle"/> to a freshly-allocated
    /// device buffer via an async H2D copy on <paramref name="stream"/>. Returns 0 (no
    /// allocation) for an unpopulated handle — e.g. <c>MimoZ</c>/<c>MimoO</c> on a SISO
    /// checkpoint. The allocated pointer is appended to <paramref name="allocs"/> immediately
    /// after <c>cuMemAlloc</c> succeeds (before the H2D copy), so callers can free everything
    /// allocated so far from that ledger if a later call in the same sequence throws (#383).
    /// </summary>
    private static nint UploadF32(Mamba3TensorHandle handle, long expectedElements, nint stream, List<nint> allocs)
    {
        if (!handle.IsPopulated) return 0;
        if (handle.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"CudaMamba3TransformerModel requires F32 tensors; got {handle.SourceDType}. "
                + "Quantized/F16 Mamba-3 weights are not yet supported on CUDA (CPU-parity scope, issue #346).");

        long bytes = expectedElements * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        CudaDriverApi.cuMemcpyHtoDAsync_v2(devPtr, handle.Pointer, (nuint)bytes, stream).ThrowOnError();
        return devPtr;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Forward
    // ────────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        // #347 lesson: MakeCurrent must be the first CUDA-touching statement in every
        // public entry point — the ephemeral CudaMamba3StateCache constructed below
        // allocates device memory before ForwardCore would otherwise set the context.
        _context.MakeCurrent();
        _ = kvCache; // Mamba-3 uses SSM state, not KV-cache — matches CPU's Forward(..., IKvCache?) contract.
        using var ephemeral = new CudaMamba3StateCache(_m3, _numLayers);
        return ForwardCore(tokenIds, positions, deviceId, ephemeral, runChunkBoundary: false);
    }

    /// <summary>
    /// Runs a forward pass that reads and writes a persistent
    /// <see cref="CudaMamba3StateCache"/>, enabling prefill-then-decode sequences.
    /// Mirrors CPU <c>Mamba3TransformerModel.Forward(..., Mamba3State)</c>.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
        CudaMamba3StateCache state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.NumLayers != _numLayers)
            throw new ArgumentException(
                $"CudaMamba3StateCache has {state.NumLayers} layers but model has {_numLayers}.", nameof(state));
        // Mirrors Mamba3TransformerModel.Forward(..., Mamba3State)'s element-count guards
        // (Mamba3TransformerModel.cs:217-226) — NumLayers alone does not catch a cache built
        // from a different Mamba3Config with the same layer count, which would otherwise hand
        // the kernels undersized buffers and produce out-of-bounds device writes with no
        // diagnostic. Mamba3State also exposes KState/VState element counts
        // (Mamba3State.cs:105,108) but CPU's own guard doesn't check them — CUDA validates
        // stronger and checks all four buffers here.
        int expectedSsm = _m3.NumHeads * _m3.HeadDim * _m3.StateSize;
        int expectedCum = _m3.NumHeads * _m3.NumRopeAngles;
        int expectedKRank = _m3.IsMimo ? _m3.MimoRank : 1;
        int expectedK = expectedKRank * _m3.NumHeads * _m3.StateSize;
        int expectedV = _m3.NumHeads * _m3.HeadDim;
        if (state.SsmStateElementsPerLayer != expectedSsm)
            throw new ArgumentException(
                $"CudaMamba3StateCache SSM layout mismatch: state has {state.SsmStateElementsPerLayer} "
                + $"elements/layer, model expects {expectedSsm}.", nameof(state));
        if (state.CumAngleElementsPerLayer != expectedCum)
            throw new ArgumentException(
                $"CudaMamba3StateCache cum_angle layout mismatch: state has {state.CumAngleElementsPerLayer} "
                + $"elements/layer, model expects {expectedCum}.", nameof(state));
        if (state.KStateElementsPerLayer != expectedK)
            throw new ArgumentException(
                $"CudaMamba3StateCache k_state layout mismatch: state has {state.KStateElementsPerLayer} "
                + $"elements/layer, model expects {expectedK}.", nameof(state));
        if (state.VStateElementsPerLayer != expectedV)
            throw new ArgumentException(
                $"CudaMamba3StateCache v_state layout mismatch: state has {state.VStateElementsPerLayer} "
                + $"elements/layer, model expects {expectedV}.", nameof(state));
        _context.MakeCurrent();
        return ForwardCore(tokenIds, positions, deviceId, state, runChunkBoundary: true);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Mirrors CPU <c>Mamba3TransformerModel.ForwardBatch</c>: rejects LoRA adapters
    /// (no Mamba-3 LoRA path), requires every request to carry a per-seq
    /// <see cref="CudaMamba3StateCache"/> (via <see cref="SequenceForwardRequest.MambaState"/>)
    /// once 2+ requests are batched together, and otherwise loops per request — no
    /// fused-GEMM batching in this v1 (see this plan's biggest-risk note for why).
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        _context.MakeCurrent();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "CudaMamba3TransformerModel.ForwardBatch does not support LoRA adapters "
                    + "(no Mamba-3 LoRA path today, matching the CPU host).");
        }

        if (requests.Count >= 2)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                if (requests[i].MambaState is null)
                    throw new ArgumentException(
                        $"CudaMamba3TransformerModel.ForwardBatch with {requests.Count} requests requires "
                        + $"every request to supply a per-seq MambaState; request[{i}] has none.",
                        nameof(requests));
            }
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            if (r.MambaState is null)
            {
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache);
            }
            else if (r.MambaState is CudaMamba3StateCache cudaState)
            {
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, cudaState);
            }
            else
            {
                throw new ArgumentException(
                    $"CudaMamba3TransformerModel requires a CudaMamba3StateCache; got {r.MambaState.GetType().Name}.",
                    nameof(requests));
            }
        }
        return results;
    }

    /// <summary>
    /// Grows every device scratch buffer so at least <paramref name="seqLen"/> tokens
    /// can be served without further allocation. Power-of-two growth, mirroring
    /// <c>Mamba3ForwardScratch.EnsureCapacity</c> (CPU) /
    /// <c>VulkanMamba3ForwardScratch.EnsureCapacity</c> (Vulkan).
    /// </summary>
    private void EnsureScratchCapacity(int seqLen)
    {
        if (seqLen <= _scratchCapacity) return;

        int cap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeScratch();
        _scratchCapacity = 0; // FreeScratch already resets this; explicit for clarity before re-set below.

        int hidden = Config.HiddenSize;
        int dInProj = _m3.InputProjectionDim;
        int dInner = _m3.DInner;
        int nHead = _m3.NumHeads;
        int dState = _m3.StateSize;
        int numRopeAngles = _m3.NumRopeAngles;
        int effRank = _m3.IsMimo ? _m3.MimoRank : 1;

        _hidden = AllocF32((long)cap * hidden);
        _residual = AllocF32((long)cap * hidden);
        _normOut = AllocF32((long)cap * hidden);
        _blockOut = AllocF32((long)cap * hidden);
        _projDevice = AllocF32((long)cap * dInProj);
        _xDevice = AllocF32((long)cap * dInner);
        _zDevice = AllocF32((long)cap * dInner);
        _yScanDevice = AllocF32((long)cap * dInner);
        _dtDevice = AllocF32((long)cap * nHead);
        _adtDevice = AllocF32((long)cap * nHead);
        _gammaDevice = AllocF32((long)cap * nHead);
        _scaleDevice = AllocF32((long)cap * nHead);
        _qkPreDotDevice = AllocF32((long)cap * nHead);
        _anglesRawDevice = AllocF32((long)cap * numRopeAngles);
        _bDevice = AllocF32((long)cap * effRank * nHead * dState);
        _cDevice = AllocF32((long)cap * effRank * nHead * dState);
        _coefDevice = AllocF32(nHead);

        // #382 item 1: reusable host-side per-forward scratch, grown alongside the device
        // buffers above instead of `new float[...]` per layer per forward in
        // HostPrepareSiso/HostPrepareMimo. Sized identically to their device counterparts
        // (bcWidth = effRank * nHead * dState covers both the SISO [T, H, N] and MIMO
        // [T, R, H, N] layouts, matching _bDevice/_cDevice's own sizing).
        _hostProj = new float[(long)cap * dInProj];
        _hostX = new float[(long)cap * dInner];
        _hostZ = new float[(long)cap * dInner];
        _hostDt = new float[(long)cap * nHead];
        _hostAdt = new float[(long)cap * nHead];
        _hostTrap = new float[(long)cap * nHead];
        _hostGamma = new float[(long)cap * nHead];
        _hostScale = new float[(long)cap * nHead];
        _hostAnglesRaw = new float[(long)cap * numRopeAngles];
        _hostB = new float[(long)cap * effRank * nHead * dState];
        _hostC = new float[(long)cap * effRank * nHead * dState];
        _hostQkPreDot = new float[(long)cap * nHead];

        _scratchCapacity = cap;
    }

    private static nint AllocF32(long elementCount)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        return ptr;
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
        CudaMamba3StateCache state, bool runChunkBoundary)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        // Positions are not consumed by the block forward on any backend (Mamba-3 encodes
        // position implicitly through cum_angle accumulation — see CPU's
        // Mamba3TransformerModel.Forward(..., IKvCache?) doc), but CPU's ForwardCore still
        // validates them against MaxSequenceLength for API-parity error behaviour — mirror
        // that here rather than silently accepting an out-of-range position that CPU would
        // reject.
        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }
        // deviceId controls the RETURNED tensor's placement, not where compute runs (compute
        // always runs on the GPU this model was loaded onto via LoadFromSafetensors' own
        // deviceId). This v1 only implements host-resident output (mirrors every CPU/Vulkan
        // Forward call site, which always passes -1) — matches CPU's
        // Mamba3TransformerModel.ForwardCore threading `deviceId` into UnmanagedTensor.Allocate,
        // but a device-resident (deviceId >= 0) result would need a D2D copy instead of the D2H
        // copy below, which is out of scope here.
        if (deviceId >= 0)
            throw new NotSupportedException(
                "CudaMamba3TransformerModel.Forward only supports deviceId=-1 (host-resident output "
                + "tensor) today. Device-resident output tensors are a future optimization.");
        if (_m3.NumGroups != 1)
            throw new NotSupportedException(
                $"CudaMamba3TransformerModel.Forward assumes NumGroups (n_groups) == 1 for its B/C "
                + $"split-offset math; got {_m3.NumGroups}. HostPrepareSiso's ofsC = ofsB + dState "
                + "hardcodes bcPerToken == dState, which is only correct for G=1 — every known real "
                + "checkpoint has n_groups=1 (matches CPU/Vulkan's own documented assumption), but a "
                + "G>1 checkpoint would silently corrupt every offset from ofsC onward instead of "
                + "failing loudly, so this guard rejects it explicitly rather than the alternative.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int nHead = _m3.NumHeads;
        int headDim = _m3.HeadDim;
        int dState = _m3.StateSize;
        int dInner = _m3.DInner;
        int dInProj = _m3.InputProjectionDim;
        int numRopeAngles = _m3.NumRopeAngles;
        float aFloor = _m3.AFloor;
        float eps = Config.NormEpsilon;
        nint s = _stream.Handle;
        int effRank = _m3.IsMimo ? _m3.MimoRank : 1;

        EnsureScratchCapacity(seqLen);

        // 1. Token upload + embedding lookup (device).
        int[] tokenIdsArr = tokenIds.ToArray();
        // #347 lesson: explicit bounds guard before the H2D upload feeds these IDs into a
        // fixed-size [vocab, hidden] embedding table lookup — an out-of-range token id would
        // otherwise read out of bounds on the device with no indirect per-element check to
        // catch it. Mirrors VulkanMamba3TransformerModel.ValidateTokenIds.
        for (int i = 0; i < tokenIdsArr.Length; i++)
        {
            if ((uint)tokenIdsArr[i] >= (uint)vocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenIdsArr[i]} at index {i} is out of range [0, {vocabSize}).");
        }
        nint tokenIdsDevice = 0;
        try
        {
            long tokenBytes = (long)seqLen * sizeof(int);
            CudaDriverApi.cuMemAlloc_v2(out tokenIdsDevice, (nuint)tokenBytes).ThrowOnError();
            fixed (int* p = tokenIdsArr)
                CudaDriverApi.cuMemcpyHtoD_v2(tokenIdsDevice, (nint)p, (nuint)tokenBytes).ThrowOnError();

            _kernels.LaunchEmbeddingLookupF32(_tokenEmbedDevice, QuantizationType.F32,
                tokenIdsDevice, _hidden, seqLen, hiddenSize, s);

            // 2. Layers.
            for (int layer = 0; layer < _numLayers; layer++)
            {
                var lw = _layers[layer];

                // Snapshot residual (D2D) + pre-norm (device).
                CudaDriverApi.cuMemcpyDtoDAsync_v2(_residual, _hidden, (nuint)((long)seqLen * hiddenSize * sizeof(float)), s).ThrowOnError();
                _kernels.LaunchRmsNormF32(_hidden, lw.Norm, _normOut, hiddenSize, eps, seqLen, s);

                // in_proj GEMM (device): proj[seqLen, dInProj] = normOut[seqLen, hidden] @ inProj[dInProj, hidden]^T.
                CudaGemm.LinearF32(_cublas.Handle, _normOut, lw.InProj, _projDevice, seqLen, hiddenSize, dInProj, s);
                _stream.Synchronize();

                // 3-6.5. Host prep + data-RoPE + chunk-boundary correction + SSD scan +
                // K/V-state persist — branches on IsMimo, mirroring CPU
                // Mamba3TransformerModel.ForwardCore's if(isMimo){...}else{...} structure
                // (Mamba3TransformerModel.cs:414-488). effRank threads through both branches
                // so EnsureScratchCapacity's already-effRank-aware _bDevice/_cDevice sizing
                // needs no change.
                if (_m3.IsMimo)
                {
                    HostPrepareMimo(seqLen, dInProj, dInner, nHead, dState, numRopeAngles, effRank, aFloor, eps,
                        lw, s);

                    // Data-RoPE (device) — mutates _bDevice/_cDevice in place, threads cum_angle.
                    // Halved mode for MIMO (Mamba3Block.ForwardMimo Step 5), nRank=effRank.
                    _kernels.LaunchMamba3DataRopeF32(_bDevice, _cDevice, _anglesRawDevice, _dtDevice,
                        state.GetCumAnglePtr(layer), state.GetCumAnglePtr(layer),
                        seqLen, effRank, nHead, dState, numRopeAngles, mode: 1 /* Halved */,
                        hasCumPrev: true, writeCumOut: true, s);

                    // Chunk-boundary correction (device) — BEFORE the scan, only for the
                    // state-threaded overload. nRank=effRank sums k_state over rank
                    // (Mamba3CanonicalSsd.ExecuteMimoStreaming's rank-summed boundary term).
                    if (runChunkBoundary)
                    {
                        _kernels.LaunchMamba3ChunkBoundaryF32(
                            state.GetSsmStatePtr(layer), state.GetVStatePtr(layer), state.GetKStatePtr(layer),
                            _coefDevice, nHead, headDim, dState, effRank, s);
                    }

                    // MIMO SSD scan (device) — mutates ssm_state in place, writes _yScanDevice.
                    // qRoped=_cDevice (C), kRoped=_bDevice (B) — matches
                    // Mamba3CanonicalSsd.ExecuteMimoStreaming(state, v, qRoped: cRHN, kRoped: bRHN, ...).
                    _kernels.LaunchMamba3SsdScanMimoF32(
                        state.GetSsmStatePtr(layer), _xDevice, _cDevice, _bDevice,
                        _qkPreDotDevice, _scaleDevice, _gammaDevice, _adtDevice, lw.D, _zDevice,
                        lw.MimoZ, lw.MimoO, _yScanDevice,
                        seqLen, effRank, nHead, headDim, dState, hasZ: true, s);

                    // Persist this chunk's last-token post-RoPE K (per rank) / raw V for
                    // the NEXT call's chunk-boundary correction. kState stores the SSD
                    // scan's "kRoped" argument — _bDevice/B, NOT _cDevice/C — confirmed
                    // against Mamba3CanonicalSsd.ExecuteMimoStreaming's persist step
                    // (kRoped param bound to bRHN at the ForwardMimo call site,
                    // Mamba3Block.cs:680) and against this file's own SISO branch below,
                    // which persists _bDevice for the identical reason (Task 9's original
                    // C-vs-B transcription bug — see progress.md — reproduced here in this
                    // task's brief and corrected before commit). kRoped layout [T, R, H, N]
                    // — the whole last-token [R, H, N] slice (all ranks) is contiguous.
                    if (runChunkBoundary)
                    {
                        long kBytes = (long)effRank * nHead * dState * sizeof(float);
                        long vBytes = (long)nHead * headDim * sizeof(float);
                        nint lastKSrc = _bDevice + (nint)((long)(seqLen - 1) * effRank * nHead * dState * sizeof(float));
                        nint lastVSrc = _xDevice + (nint)((long)(seqLen - 1) * nHead * headDim * sizeof(float));
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetKStatePtr(layer), lastKSrc, (nuint)kBytes, s).ThrowOnError();
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetVStatePtr(layer), lastVSrc, (nuint)vBytes, s).ThrowOnError();
                    }
                }
                else
                {
                    // Host prep — D2H the in_proj output, run the per-token
                    // softplus/sigmoid/RMSNorm+bias/qk_pre_dot/scale math on CPU (mirrors
                    // Mamba3Block.Forward Steps 2-4 exactly), H2D the results back.
                    HostPrepareSiso(seqLen, dInProj, dInner, nHead, dState, numRopeAngles, aFloor, eps,
                        lw, s);

                    // Data-RoPE (device) — mutates _bDevice/_cDevice in place, threads cum_angle.
                    _kernels.LaunchMamba3DataRopeF32(_bDevice, _cDevice, _anglesRawDevice, _dtDevice,
                        state.GetCumAnglePtr(layer), state.GetCumAnglePtr(layer),
                        seqLen, nRank: 1, nHead, dState, numRopeAngles, mode: 0,
                        hasCumPrev: true, writeCumOut: true, s);

                    // Chunk-boundary correction (device) — BEFORE the scan, only for the
                    // state-threaded overload (see this task's design note above).
                    if (runChunkBoundary)
                    {
                        _kernels.LaunchMamba3ChunkBoundaryF32(
                            state.GetSsmStatePtr(layer), state.GetVStatePtr(layer), state.GetKStatePtr(layer),
                            _coefDevice, nHead, headDim, dState, nRank: 1, s);
                    }

                    // SISO SSD scan (device) — mutates ssm_state in place, writes _yScanDevice.
                    _kernels.LaunchMamba3SsdScanSisoF32(
                        state.GetSsmStatePtr(layer), _xDevice, _cDevice, _bDevice,
                        _qkPreDotDevice, _scaleDevice, _gammaDevice, _adtDevice, lw.D, _zDevice, _yScanDevice,
                        seqLen, nHead, headDim, dState, hasZ: true, s);

                    // Persist this chunk's last-token post-RoPE K / raw V for the NEXT
                    // call's chunk-boundary correction (D2D — matches CPU's bHRN/xBuf slice
                    // copy at Mamba3Block.cs Step 6.5). kState stores bHRN (the "kRoped"
                    // argument to the SSD scan, i.e. _bDevice/B) not cHRN — confirmed against
                    // both Mamba3Block.cs:401 (bHRN -> kState) and
                    // VulkanMamba3TransformerModel.cs's matching RecordCopyBufferRange(_state.B -> kState)
                    // (CPU's C-tensor equivalent is _cDevice/"qRoped", which is never copied here).
                    if (runChunkBoundary)
                    {
                        long kBytes = (long)nHead * dState * sizeof(float);
                        long vBytes = (long)nHead * headDim * sizeof(float);
                        nint lastKSrc = _bDevice + (nint)((long)(seqLen - 1) * nHead * dState * sizeof(float));
                        nint lastVSrc = _xDevice + (nint)((long)(seqLen - 1) * nHead * headDim * sizeof(float));
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetKStatePtr(layer), lastKSrc, (nuint)kBytes, s).ThrowOnError();
                        CudaDriverApi.cuMemcpyDtoDAsync_v2(state.GetVStatePtr(layer), lastVSrc, (nuint)vBytes, s).ThrowOnError();
                    }
                }

                // 7. out_proj GEMM (device): blockOut[seqLen, hidden] = yScan[seqLen, dInner] @ outProj[hidden, dInner]^T.
                CudaGemm.LinearF32(_cublas.Handle, _yScanDevice, lw.OutProj, _blockOut, seqLen, dInner, hiddenSize, s);

                // Residual add (device): hidden = residual + blockOut.
                _kernels.LaunchAddF32(_residual, _blockOut, _hidden, seqLen * hiddenSize, s);
            }

            // 8. Final RMSNorm + lm_head GEMM (device, last token only).
            // #382 item 4: only the last row feeds the lm_head GEMV — normalizing all seqLen
            // rows (as the previous version did) is wasted prefill work RMSNorm being
            // row-independent makes provably unnecessary. Mirrors
            // VulkanMamba3TransformerModel.RunFinalNormAndLmHead: D2D-copy the last row into the
            // fixed-size _lastHiddenDevice scratch, then norm with rowCount:1.
            long hiddenRowBytes = (long)hiddenSize * sizeof(float);
            nint lastHiddenSrc = _hidden + (nint)((long)(seqLen - 1) * hiddenRowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_lastHiddenDevice, lastHiddenSrc, (nuint)hiddenRowBytes, s).ThrowOnError();
            _kernels.LaunchRmsNormF32(_lastHiddenDevice, _finalNormDevice, _lastHiddenDevice, hiddenSize, eps, rows: 1, s);

            CudaGemm.GemvF32(_cublas.Handle, _lmHeadDevice, _lastHiddenDevice, _logitsDevice, vocabSize, hiddenSize, s);

            var shape = new TensorShape(1, vocabSize);
            var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
            _stream.Synchronize();
            CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _logitsDevice, (nuint)((long)vocabSize * sizeof(float))).ThrowOnError();
            return result;
        }
        finally
        {
            if (tokenIdsDevice != 0) CudaDriverApi.cuMemFree_v2(tokenIdsDevice);
        }
    }

    /// <summary>
    /// Host-side per-token preprocessing for one SISO layer: downloads the in_proj GEMM
    /// output, replicates <c>Mamba3Block.Forward</c>'s Steps 2-4 (split, softplus/sigmoid
    /// DT/A/trap/gamma, B/C RMSNorm+bias, qk_pre_dot, shifted-gamma/scale) on the CPU
    /// exactly as written there, then uploads the per-token tables the device kernels
    /// need. Mirrors Vulkan's <c>ComputeHostTables</c> design decision (see this class's
    /// doc comment) — a fused on-device version is a documented future optimization, not
    /// attempted in this plan (see the biggest-risk note).
    /// </summary>
    private void HostPrepareSiso(int seqLen, int dInProj, int dInner, int nHead, int dState,
        int numRopeAngles, float aFloor, float normEps, DeviceLayer lw, nint stream)
    {
        // #382 item 1: reuse the cap-sized host scratch fields instead of `new float[...]` —
        // only the first seqLen-scaled prefix of each is valid/written this call; the D2H copy
        // below and every UploadF32Array call at the bottom are careful to size themselves off
        // `seqLen`, not the (possibly larger) field's own Length, matching the analogous care
        // EnsureScratchCapacity's device buffers already require.
        float[] proj = _hostProj;
        fixed (float* p = proj)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, _projDevice, (nuint)((long)seqLen * dInProj * sizeof(float))).ThrowOnError();

        // #382 item 2: cached once at load (DeviceLayer.*Host) — no more per-forward
        // cuMemcpyDtoH_v2 re-download of these static per-layer tensors.
        float[] dtBias = lw.DtBiasHost;
        float[] bNormW = lw.BNormHost;
        float[] cNormW = lw.CNormHost;
        float[] bBias = lw.BBiasHost;   // numBcHeads=1 SISO, so [nHead, dState]
        float[] cBias = lw.CBiasHost;

        int ofsZ = 0, ofsX = dInner, ofsB = 2 * dInner, ofsC = ofsB + dState;
        int ofsDdDt = ofsC + dState, ofsDdA = ofsDdDt + nHead, ofsTrap = ofsDdA + nHead, ofsAngles = ofsTrap + nHead;

        float[] x = _hostX, z = _hostZ, dt = _hostDt, adt = _hostAdt, trap = _hostTrap,
            gamma = _hostGamma, scale = _hostScale, anglesRaw = _hostAnglesRaw,
            bHRN = _hostB, cHRN = _hostC, qkPreDot = _hostQkPreDot;

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;
            Array.Copy(proj, src + ofsZ, z, t * dInner, dInner);
            Array.Copy(proj, src + ofsX, x, t * dInner, dInner);

            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];

                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;

                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;
            }

            Array.Copy(proj, src + ofsAngles, anglesRaw, t * numRopeAngles, numRopeAngles);

            // B/C RMSNorm + bias (numBcHeads=1 broadcasts to every head — matches every
            // real checkpoint's n_groups=1; multi-group is a Mamba3Block-documented
            // future extension, not implemented on any backend today).
            int bSrcBase = src + ofsB, cSrcBase = src + ofsC;
            RmsNormFactor(proj, bSrcBase, dState, normEps, out float bInvRms);
            RmsNormFactor(proj, cSrcBase, dState, normEps, out float cInvRms);
            for (int h = 0; h < nHead; h++)
            {
                int biasBase = h * dState;
                int dstBase = (t * nHead + h) * dState;
                for (int n = 0; n < dState; n++)
                {
                    bHRN[dstBase + n] = proj[bSrcBase + n] * bInvRms * bNormW[n] + bBias[biasBase + n];
                    cHRN[dstBase + n] = proj[cSrcBase + n] * cInvRms * cNormW[n] + cBias[biasBase + n];
                }
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            int baseT = t * nHead * dState;
            for (int h = 0; h < nHead; h++)
            {
                int off = baseT + h * dState;
                // TensorPrimitives.Dot, not a naive scalar accumulation loop — matches CPU's
                // Mamba3Block.cs:339 and Vulkan's identical choice exactly, avoiding a gratuitous
                // F32 reassociation against the oracle Task 11's parity tests will diff against.
                qkPreDot[t * nHead + h] = TensorPrimitives.Dot(
                    cHRN.AsSpan(off, dState), bHRN.AsSpan(off, dState));
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }

        // Chunk-boundary coefficient: coef[h] = dt[0,h] * (1 - trap[0,h]) — only the
        // FIRST token's dt/trap matter (Mamba3Block.ApplyChunkBoundaryAdjustment reads
        // dt[0,:]/trap[0,:] only).
        float[] coef = _hostCoef;
        for (int h = 0; h < nHead; h++) coef[h] = dt[h] * (1f - trap[h]);

        UploadF32Array(x.AsSpan(0, seqLen * dInner), _xDevice, stream);
        UploadF32Array(z.AsSpan(0, seqLen * dInner), _zDevice, stream);
        UploadF32Array(dt.AsSpan(0, seqLen * nHead), _dtDevice, stream);
        UploadF32Array(adt.AsSpan(0, seqLen * nHead), _adtDevice, stream);
        // trap is never uploaded — #382 item 3: no kernel consumes a device trap buffer
        // (the removed _trapDevice was allocated + uploaded every layer for nothing).
        UploadF32Array(gamma.AsSpan(0, seqLen * nHead), _gammaDevice, stream);
        UploadF32Array(scale.AsSpan(0, seqLen * nHead), _scaleDevice, stream);
        UploadF32Array(anglesRaw.AsSpan(0, seqLen * numRopeAngles), _anglesRawDevice, stream);
        UploadF32Array(bHRN.AsSpan(0, seqLen * nHead * dState), _bDevice, stream);
        UploadF32Array(cHRN.AsSpan(0, seqLen * nHead * dState), _cDevice, stream);
        UploadF32Array(qkPreDot.AsSpan(0, seqLen * nHead), _qkPreDotDevice, stream);
        UploadF32Array(coef, _coefDevice, stream);
        _stream.Synchronize();
    }

    /// <summary>
    /// MIMO analog of <see cref="HostPrepareSiso"/> — line-for-line port of
    /// <c>Mamba3Block.ForwardMimo</c>'s Steps 2-4. Differs from SISO in: <c>bcPerToken</c>
    /// includes the rank factor, B/C are laid out <c>[T, R, H, N]</c> (RmsNorm+bias applied
    /// per <c>(r, g)</c> slice with bias shape <c>[H, R, N]</c>), and <c>qkPreDot</c> sums
    /// the pre-rotation dot over rank (<c>qkPreDotSum</c>).
    /// </summary>
    private void HostPrepareMimo(int seqLen, int dInProj, int dInner, int nHead, int dState,
        int numRopeAngles, int mimoRank, float aFloor, float normEps, DeviceLayer lw, nint stream)
    {
        int r_ = mimoRank;
        int bcPerToken = dState * r_; // numBcHeads=1 on every known checkpoint

        // #382 items 1+2 — see HostPrepareSiso's matching comments; identical rationale.
        float[] proj = _hostProj;
        fixed (float* p = proj)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, _projDevice, (nuint)((long)seqLen * dInProj * sizeof(float))).ThrowOnError();

        float[] dtBias = lw.DtBiasHost;
        float[] bNormW = lw.BNormHost;
        float[] cNormW = lw.CNormHost;
        float[] bBias = lw.BBiasHost;   // [H, R, N]
        float[] cBias = lw.CBiasHost;

        int ofsZ = 0, ofsX = dInner, ofsB = 2 * dInner, ofsC = ofsB + bcPerToken;
        int ofsDdDt = ofsC + bcPerToken, ofsDdA = ofsDdDt + nHead, ofsTrap = ofsDdA + nHead, ofsAngles = ofsTrap + nHead;

        float[] x = _hostX, z = _hostZ, dt = _hostDt, adt = _hostAdt, trap = _hostTrap,
            gamma = _hostGamma, scale = _hostScale, anglesRaw = _hostAnglesRaw,
            bRHN = _hostB, cRHN = _hostC, qkPreDotSum = _hostQkPreDot;

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;
            Array.Copy(proj, src + ofsZ, z, t * dInner, dInner);
            Array.Copy(proj, src + ofsX, x, t * dInner, dInner);

            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];
                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;
                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;
            }

            Array.Copy(proj, src + ofsAngles, anglesRaw, t * numRopeAngles, numRopeAngles);

            for (int rr = 0; rr < r_; rr++)
            {
                int bSrcBase = src + ofsB + rr * dState;
                int cSrcBase = src + ofsC + rr * dState;
                RmsNormFactor(proj, bSrcBase, dState, normEps, out float bInvRms);
                RmsNormFactor(proj, cSrcBase, dState, normEps, out float cInvRms);
                for (int h = 0; h < nHead; h++)
                {
                    int biasBase = (h * r_ + rr) * dState;
                    int dstBase = ((t * r_ + rr) * nHead + h) * dState;
                    for (int n = 0; n < dState; n++)
                    {
                        bRHN[dstBase + n] = proj[bSrcBase + n] * bInvRms * bNormW[n] + bBias[biasBase + n];
                        cRHN[dstBase + n] = proj[cSrcBase + n] * cInvRms * cNormW[n] + cBias[biasBase + n];
                    }
                }
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sum = 0f;
                for (int rr = 0; rr < r_; rr++)
                {
                    int baseIdx = ((t * r_ + rr) * nHead + h) * dState;
                    // TensorPrimitives.Dot, not a naive scalar accumulation loop — matches
                    // CPU's Mamba3Block.cs:646 and the SISO branch above (HostPrepareSiso)
                    // exactly, avoiding a gratuitous F32 reassociation against the oracle
                    // this task's parity tests diff against (same rationale as Task 9's
                    // fix-round change to HostPrepareSiso).
                    sum += TensorPrimitives.Dot(cRHN.AsSpan(baseIdx, dState), bRHN.AsSpan(baseIdx, dState));
                }
                qkPreDotSum[t * nHead + h] = sum;
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }

        float[] coef = _hostCoef;
        for (int h = 0; h < nHead; h++) coef[h] = dt[h] * (1f - trap[h]);

        UploadF32Array(x.AsSpan(0, seqLen * dInner), _xDevice, stream);
        UploadF32Array(z.AsSpan(0, seqLen * dInner), _zDevice, stream);
        UploadF32Array(dt.AsSpan(0, seqLen * nHead), _dtDevice, stream);
        UploadF32Array(adt.AsSpan(0, seqLen * nHead), _adtDevice, stream);
        // trap is never uploaded — see HostPrepareSiso's matching comment (#382 item 3).
        UploadF32Array(gamma.AsSpan(0, seqLen * nHead), _gammaDevice, stream);
        UploadF32Array(scale.AsSpan(0, seqLen * nHead), _scaleDevice, stream);
        UploadF32Array(anglesRaw.AsSpan(0, seqLen * numRopeAngles), _anglesRawDevice, stream);
        UploadF32Array(bRHN.AsSpan(0, seqLen * r_ * nHead * dState), _bDevice, stream);
        UploadF32Array(cRHN.AsSpan(0, seqLen * r_ * nHead * dState), _cDevice, stream);
        UploadF32Array(qkPreDotSum.AsSpan(0, seqLen * nHead), _qkPreDotDevice, stream);
        UploadF32Array(coef, _coefDevice, stream);
        _stream.Synchronize();
    }

    private static void UploadF32Array(ReadOnlySpan<float> host, nint devicePtr, nint stream)
    {
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyHtoDAsync_v2(devicePtr, (nint)p, (nuint)(host.Length * sizeof(float)), stream).ThrowOnError();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RmsNormFactor(float[] proj, int offset, int n, float normEps, out float invRms)
    {
        // F32 accumulator upcast to double — matches Mamba3Block.RmsNormInto's accumulator
        // precision exactly (bit-parity requirement, not just "close enough").
        double acc = 0.0;
        for (int i = 0; i < n; i++) { double v = proj[offset + i]; acc += v * v; }
        float mean = (float)(acc / n);
        invRms = 1f / MathF.Sqrt(mean + normEps);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftPlus(float x)
    {
        if (x > 20f) return x;
        if (x < -20f) return MathF.Exp(x);
        return MathF.Log(1f + MathF.Exp(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private long ScratchAllocatedBytes()
    {
        // _logitsDevice and _lastHiddenDevice are allocated once in the constructor,
        // independent of _scratchCapacity — always counted.
        long logitsBytes = (long)Config.VocabSize * sizeof(float);
        long lastHiddenBytes = (long)Config.HiddenSize * sizeof(float);
        long fixedBytes = logitsBytes + lastHiddenBytes;
        if (_scratchCapacity == 0) return fixedBytes;
        long cap = _scratchCapacity;
        int hidden = Config.HiddenSize, dInner = _m3.DInner, nHead = _m3.NumHeads;
        int dState = _m3.StateSize, numRopeAngles = _m3.NumRopeAngles;
        int effRank = _m3.IsMimo ? _m3.MimoRank : 1;
        long floats = cap * hidden * 4L                       // hidden/residual/normOut/blockOut
                    + cap * _m3.InputProjectionDim             // proj
                    + cap * dInner * 3L                        // x/z/yScan
                    + cap * nHead * 5L                         // dt/adt/gamma/scale/qkPreDot (#382 item 3: trap removed)
                    + cap * numRopeAngles                      // anglesRaw
                    + cap * effRank * nHead * dState * 2L      // b/c
                    + nHead;                                   // _coefDevice — [nHead], not cap-scaled
        return floats * sizeof(float) + fixedBytes;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _context.MakeCurrent();

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
        FreeIfNonZero(_lastHiddenDevice);
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

    /// <summary>
    /// Frees a scratch pointer (if allocated) and zeroes the field in place. Used instead of
    /// <see cref="FreeIfNonZero"/> for every field in <see cref="FreeScratch"/> so that if
    /// <see cref="EnsureScratchCapacity"/>'s realloc sequence throws mid-way (e.g. device OOM
    /// on a later buffer), the fields already freed here are left as 0 rather than dangling —
    /// a subsequent re-entry (with <c>_scratchCapacity == 0</c>) that calls <c>FreeScratch</c>
    /// again would otherwise re-free stale pointers that may by then alias live driver
    /// re-handed allocations (free-of-live-buffer). Calling this twice on the same field is a
    /// safe no-op (second call sees <c>p == 0</c>).
    /// </summary>
    private static void FreeAndClear(ref nint p)
    {
        FreeIfNonZero(p);
        p = 0;
    }

    private void FreeScratch()
    {
        FreeAndClear(ref _hidden); FreeAndClear(ref _residual); FreeAndClear(ref _normOut); FreeAndClear(ref _blockOut);
        FreeAndClear(ref _projDevice); FreeAndClear(ref _xDevice); FreeAndClear(ref _zDevice); FreeAndClear(ref _yScanDevice);
        FreeAndClear(ref _dtDevice); FreeAndClear(ref _adtDevice);
        FreeAndClear(ref _gammaDevice); FreeAndClear(ref _scaleDevice); FreeAndClear(ref _qkPreDotDevice);
        FreeAndClear(ref _anglesRawDevice); FreeAndClear(ref _bDevice); FreeAndClear(ref _cDevice); FreeAndClear(ref _coefDevice);
        _scratchCapacity = 0;
    }
}
