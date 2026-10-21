// src/DotLLM.Cuda/Architectures/CudaNemotronHMoe.cs
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// The routed-MoE FFN of <c>nemotron_h_moe</c> (Nemotron 3.5 Bolt and Lightning) on CUDA, all experts resident in VRAM
/// (issue #403; design: #1063 section 8a). Mirrors <c>NemotronHTransformerModel.ForwardMoeFfnBody</c> /
/// <c>ForwardMoeFfnBlockGrouped</c> (CPU):
/// <list type="number">
///   <item>Router GEMM on the full normed <c>h</c>, async D2H of the logits into pinned memory, then an event.</item>
///   <item>While the host waits on that event the GPU runs <c>latent_down(h)</c> and the shared expert (enqueued first).</item>
///   <item>The host routes every token with the CPU model's own <c>RouteToken</c> (sigmoid, selection-only bias, top-k with
///   ties to the lower id, renorm with the F16-min clamp, scale), so the selection is the CPU oracle's by construction, and
///   builds a CSR schedule grouped by expert, sent in one async H2D.</item>
///   <item>The expert-grouped GEMV runs up (+relu²) and down for every used expert, then the slot-ordered weighted sum,
///   <c>latent_up</c>, and the shared output is added unweighted.</item>
/// </list>
/// Non-latent layers (Lightning, <c>LatentSize == 0</c>) run the experts on <c>h</c> and skip both latent projections.
/// </summary>
public sealed unsafe partial class CudaNemotronHTransformerModel
{
    /// <summary>
    /// Token block of the MoE body; bounds the routed scratch to O(block · topK · width), like the CPU's
    /// <c>GroupedMoeBlockTokens</c>. Settable from tests (a small value forces several blocks per prefill).
    /// </summary>
    internal static int MoeBlockTokens { get; set; } = 512;

    /// <summary>
    /// VRAM kept free when deciding whether a <c>nemotron_h_moe</c> model fits (issue #1063 addendum): the CUDA
    /// context, cuBLAS workspace and driver allocations live outside the estimate. Windows WDDM spills over-committed
    /// VRAM into shared system memory instead of failing allocations, so the fit is checked up front.
    /// </summary>
    internal const long MoeVramReserveBytes = 512L * 1024 * 1024;

    /// <summary>Device bytes a Nemotron-H model will allocate, by category (see <see cref="EstimateDeviceBytes"/>).</summary>
    /// <param name="Weights">Every uploaded weight, including all expert banks.</param>
    /// <param name="ForwardState">Forward activations at the planning token count, plus the SSM recurrent state.</param>
    /// <param name="MoeScratch">Routed-MoE scratch and schedule buffer for one token block.</param>
    /// <param name="GemmScratch">The F16 dequant tile and the F16 activation in/out scratch of the GEMM dispatcher.</param>
    internal readonly record struct DeviceBudget(long Weights, long ForwardState, long MoeScratch, long GemmScratch)
    {
        /// <summary>Sum of every category.</summary>
        public long Total => Weights + ForwardState + MoeScratch + GemmScratch;
    }

    /// <summary>
    /// Pure estimate of the device memory <see cref="BuildFromPrebuiltWeights"/> and a forward of
    /// <paramref name="planTokens"/> tokens allocate: the weights (as the upload sizes them), the forward state and SSM
    /// cache (as <c>CudaNemotronHForwardState</c> / <c>CudaNemotronHSsmStateCache</c> size them, including the state's
    /// power-of-two row rounding), the routed-MoE scratch for a block of <c>min(planTokens, moeBlockTokens)</c> tokens, and
    /// the GEMM dispatcher's F16 scratch. Longer prompts grow the forward state beyond this.
    /// </summary>
    /// <param name="config">Model configuration (hybrid layout and SSM config populated).</param>
    /// <param name="cpuLayers">Per-layer CPU weights (only shapes and types are read).</param>
    /// <param name="outputQt">LM-head quantization type.</param>
    /// <param name="outputM">LM-head rows (vocab).</param>
    /// <param name="outputK">LM-head row length (hidden).</param>
    /// <param name="planTokens">Tokens per forward to plan for.</param>
    /// <param name="moeBlockTokens">MoE token block.</param>
    /// <returns>The estimate, by category.</returns>
    internal static DeviceBudget EstimateDeviceBytes(ModelConfig config, NemotronHLayerWeights[] cpuLayers,
        QuantizationType outputQt, int outputM, int outputK, int planTokens, int moeBlockTokens)
    {
        int hidden = config.HiddenSize;
        var ssm = config.SsmConfig!.Value;
        const long F = sizeof(float);

        long weights = hidden * F + Raw(outputQt, outputM, outputK);
        long maxTile = (long)outputM * outputK;
        long maxK = outputK, maxM = outputM;
        int maxInter = 0, numSsm = 0;
        var moeSpec = default(MoeScratchSpec);

        foreach (var l in cpuLayers)
        {
            weights += hidden * F;
            if (l.Ssm is { } s)
            {
                numSsm++;
                weights += Raw(s.InQuantType, s.InOutputDim, s.InInputDim) + Raw(s.OutQuantType, s.OutOutputDim, s.OutInputDim);
                weights += F * ((long)s.Conv1dWeight.Length + s.Conv1dBias.Length + s.A.Length + s.D.Length
                    + s.DtBias.Length + s.NormWeight.Length);
                Tile(s.InOutputDim, s.InInputDim);
                Tile(s.OutOutputDim, s.OutInputDim);
            }
            if (l.Attention is { } a)
            {
                weights += Raw(a.QQuantType, a.QOutputDim, a.QInputDim) + Raw(a.KQuantType, a.KOutputDim, a.KInputDim)
                         + Raw(a.VQuantType, a.VOutputDim, a.VInputDim) + Raw(a.OQuantType, a.OOutputDim, a.OInputDim);
                Tile(a.QOutputDim, a.QInputDim);
                Tile(a.KOutputDim, a.KInputDim);
                Tile(a.VOutputDim, a.VInputDim);
                Tile(a.OOutputDim, a.OInputDim);
            }
            if (l.Ffn is { } f)
            {
                weights += Raw(f.UpQuantType, f.UpOutputDim, f.UpInputDim) + Raw(f.DownQuantType, f.DownOutputDim, f.DownInputDim);
                Tile(f.UpOutputDim, f.UpInputDim);
                Tile(f.DownOutputDim, f.DownInputDim);
                maxInter = Math.Max(maxInter, f.UpOutputDim);
            }
            if (l.Moe is { } m)
            {
                weights += Raw(m.GateInpQuantType, m.NumExperts, hidden)
                         + Raw(m.UpShexpQuantType, m.SharedIntermediateSize, hidden)
                         + Raw(m.DownShexpQuantType, hidden, m.SharedIntermediateSize)
                         + m.UpPerExpertBytes * m.NumExperts + m.DownPerExpertBytes * m.NumExperts;
                Tile(m.NumExperts, hidden);
                Tile(m.SharedIntermediateSize, hidden);
                Tile(hidden, m.SharedIntermediateSize);
                if (m.HasLatent)
                {
                    weights += Raw(m.LatentDownQuantType, m.LatentSize, hidden) + Raw(m.LatentUpQuantType, hidden, m.LatentSize);
                    Tile(m.LatentSize, hidden);
                    Tile(hidden, m.LatentSize);
                }
                moeSpec = MoeScratchSpec.Widen(moeSpec, m, hidden);
            }
        }
        if (maxInter == 0) maxInter = hidden;
        maxTile = Math.Max(maxTile, maxInter);

        // CudaNemotronHForwardState: rows rounded up to a power of two, ConvInput carries dConv-1 extra rows, and two
        // int[maxSeqLen] buffers (token ids, positions). Then the SSM cache's conv + recurrent state per SSM layer.
        long rows = (long)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(planTokens, 1));
        long q = (long)config.NumAttentionHeads * config.HeadDim, kv = (long)config.NumKvHeads * config.HeadDim;
        long bc = (long)ssm.NGroup * ssm.DState;
        long perRow = 3L * hidden + maxInter + config.VocabSize + 2 * q + 2 * kv
                    + ssm.InputProjectionDim + 2L * ssm.ConvDim + ssm.NHead + 3L * ssm.DInner + 2 * bc;
        long state = F * (rows * perRow + (long)(ssm.DConv - 1) * ssm.ConvDim) + 2L * sizeof(int) * config.MaxSequenceLength;
        state += F * numSsm * ((long)ssm.ConvStateElements + ssm.SsmStateElements);

        long moe = moeSpec.NumExperts > 0
            ? MoeScratchBytes(moeSpec, hidden, Math.Min(Math.Max(planTokens, 1), Math.Max(moeBlockTokens, 1)))
            : 0;

        // Dequant tile (F16) plus the F16 activation in / out scratch of a planTokens-row GEMM.
        long gemm = 2L * maxTile + 2L * planTokens * (maxK + maxM);
        return new DeviceBudget(weights, state, moe, gemm);

        static long Raw(QuantizationType qt, long rowCount, long cols) => Dequantize.RowByteSize(cols, qt) * rowCount;

        void Tile(long m, long k)
        {
            maxTile = Math.Max(maxTile, m * k);
            maxM = Math.Max(maxM, m);
            maxK = Math.Max(maxK, k);
        }
    }

    /// <summary>Device bytes of the routed-MoE scratch for <paramref name="tokens"/> tokens per block (mirrors <see cref="AllocateMoeScratch"/>).</summary>
    private static long MoeScratchBytes(MoeScratchSpec s, int hidden, int tokens)
    {
        long slots = (long)tokens * s.TopK;
        long floats = Math.Max((long)tokens * s.NumExperts, 1)
                    + (s.LatentSize > 0 ? Math.Max((long)tokens * s.LatentSize, 1) : 0)
                    + Math.Max((long)tokens * Math.Max(s.SharedIntermediate, 1), 1)
                    + Math.Max((long)tokens * hidden, 1)
                    + Math.Max(slots * s.MoeIntermediate, 1)
                    + Math.Max(slots * s.ExpertDim, 1)
                    + Math.Max((long)tokens * s.ExpertDim, 1);
        return floats * sizeof(float) + MoeScheduleBytes(s.NumExperts, checked((int)slots));
    }

    /// <summary>
    /// Throws when <paramref name="budget"/> exceeds free VRAM minus <see cref="MoeVramReserveBytes"/>, before anything is
    /// allocated (WDDM would otherwise spill silently into shared system memory rather than fail an allocation).
    /// </summary>
    private static void ThrowIfMoeModelDoesNotFit(DeviceBudget budget)
    {
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out nuint total).ThrowOnError();
        if (budget.Total <= (long)free - MoeVramReserveBytes) return;
        const double GiB = 1024.0 * 1024 * 1024;
        throw new NotSupportedException(
            $"This nemotron_h_moe model needs {budget.Total / GiB:F2} GiB of VRAM (weights {budget.Weights / GiB:F2}, " +
            $"forward state {budget.ForwardState / GiB:F2}, MoE scratch {budget.MoeScratch / GiB:F2}, GEMM scratch " +
            $"{budget.GemmScratch / GiB:F2}), {(long)free / GiB:F2} GiB free of {(long)total / GiB:F2} GiB " +
            $"({MoeVramReserveBytes / (1024 * 1024)} MiB kept in reserve). The CUDA backend keeps every expert resident; " +
            "expert offload is tracked in #1063. Use --device cpu or vulkan meanwhile.");
    }

    /// <summary>Which MoE kernels ran; read by tests to prove the intended path executed (issue #403).</summary>
    internal static class MoePathCounters
    {
        /// <summary>MoE layer bodies run, counted once per token block.</summary>
        public static long LayerCalls;
        /// <summary>Expert-grouped GEMV launches (up and down) per bank type.</summary>
        public static long Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS;
        /// <summary>Expert groups (used experts) dispatched, summed over every launch.</summary>
        public static long Groups;

        /// <summary>Zeroes every counter.</summary>
        public static void Reset()
        {
            Interlocked.Exchange(ref LayerCalls, 0); Interlocked.Exchange(ref Groups, 0);
            Interlocked.Exchange(ref Q4_K, 0); Interlocked.Exchange(ref Q5_K, 0); Interlocked.Exchange(ref Q6_K, 0);
            Interlocked.Exchange(ref Q8_0, 0); Interlocked.Exchange(ref IQ2_XXS, 0);
        }

        /// <summary>Records one expert-grouped launch over <paramref name="groups"/> experts.</summary>
        /// <param name="qt">Bank type of the launch.</param>
        /// <param name="groups">Used experts in the launch.</param>
        internal static void CountLaunch(QuantizationType qt, int groups)
        {
            Interlocked.Add(ref Groups, groups);
            switch (qt)
            {
                case QuantizationType.Q4_K: Interlocked.Increment(ref Q4_K); break;
                case QuantizationType.Q5_K: Interlocked.Increment(ref Q5_K); break;
                case QuantizationType.Q6_K: Interlocked.Increment(ref Q6_K); break;
                case QuantizationType.Q8_0: Interlocked.Increment(ref Q8_0); break;
                case QuantizationType.IQ2_XXS: Interlocked.Increment(ref IQ2_XXS); break;
            }
        }
    }

    /// <summary>
    /// Device-side weights of one MoE layer. The routed banks are one device allocation each, addressed through the
    /// per-expert pointer arrays <see cref="Up"/> / <see cref="Down"/> (the hook an expert cache swaps, #1063 PR B).
    /// </summary>
    private sealed class DeviceMoe
    {
        /// <summary>The CPU layer weights: routing reads their bias, top-k, renorm and scale on the host.</summary>
        public required NemotronHMoeWeights Host { get; init; }
        public required nint GateInp { get; init; }
        public required QuantizationType GateInpQt { get; init; }
        public required nint UpBank { get; init; }
        public required nint DownBank { get; init; }
        /// <summary>Device pointer of expert e's up matrix (<c>UpBank + e · UpPerExpertBytes</c>).</summary>
        public required nint[] Up { get; init; }
        /// <summary>Device pointer of expert e's down matrix (<c>DownBank + e · DownPerExpertBytes</c>).</summary>
        public required nint[] Down { get; init; }
        public required QuantizationType UpQt { get; init; }
        public required QuantizationType DownQt { get; init; }
        public required long UpPerExpertBytes { get; init; }
        public required long DownPerExpertBytes { get; init; }
        public required nint UpShexp { get; init; }
        public required QuantizationType UpShexpQt { get; init; }
        public required nint DownShexp { get; init; }
        public required QuantizationType DownShexpQt { get; init; }
        public required nint LatentDown { get; init; }
        public required QuantizationType LatentDownQt { get; init; }
        public required nint LatentUp { get; init; }
        public required QuantizationType LatentUpQt { get; init; }
        public required int NumExperts { get; init; }
        public required int NumExpertsPerTok { get; init; }
        public required int MoeIntermediateSize { get; init; }
        public required int LatentSize { get; init; }
        public required int SharedIntermediateSize { get; init; }
        public required bool NormalizeWeights { get; init; }
        public required float WeightsScale { get; init; }

        /// <summary>Width the routed experts read and write: the latent, or the hidden size when there is none.</summary>
        public int ExpertDim(int hiddenSize) => LatentSize > 0 ? LatentSize : hiddenSize;
    }

    /// <summary>Per-layer maxima of the MoE geometry; sizes the shared routed scratch (zero when the model has no MoE layer).</summary>
    private readonly record struct MoeScratchSpec(
        int NumExperts, int TopK, int MoeIntermediate, int ExpertDim, int LatentSize, int SharedIntermediate)
    {
        public static MoeScratchSpec Widen(MoeScratchSpec c, NemotronHMoeWeights m, int hiddenSize) => new(
            Math.Max(c.NumExperts, m.NumExperts),
            Math.Max(c.TopK, m.NumExpertsPerTok),
            Math.Max(c.MoeIntermediate, m.MoeIntermediateSize),
            Math.Max(c.ExpertDim, m.HasLatent ? m.LatentSize : hiddenSize),
            Math.Max(c.LatentSize, m.LatentSize),
            Math.Max(c.SharedIntermediate, m.SharedIntermediateSize));
    }

    private MoeScratchSpec _moeSpec;

    // Routed scratch, allocated once at load for blocks of _moeScratchTokens tokens (no lazy growth, #1063 addendum).
    private int _moeScratchTokens;
    private nint _moeRouterLogits;   // [n, maxExperts]
    private nint _moeXLatent;        // [n, maxLatent]
    private nint _moeShMid;          // [n, maxSharedInter]
    private nint _moeShOut;          // [n, hidden]
    private nint _moeMid;            // [n · maxTopK, maxMoeInter]
    private nint _moeSlotOut;        // [n · maxTopK, maxExpertDim]
    private nint _moeAcc;            // [n, maxExpertDim]
    private nint _moeSchedDevice;    // CSR schedule (see BuildMoeSchedule)
    private nint _moePinned;         // pinned host: router logits, then the schedule staging
    private long _moePinnedScheduleOffset;
    private nint _moeEvent;
    private float[] _moeSelScratch = [];
    private int[] _moeTopIdx = [];
    private float[] _moeWeights = [];
    private int[] _moeCounts = [];
    private int[] _moeFill = [];

    /// <summary>
    /// Uploads one MoE layer. Throws <see cref="NotSupportedException"/> naming the type for a bank the expert kernel
    /// cannot read, and an <see cref="InvalidOperationException"/> pointing at #1063 when the banks do not fit in VRAM.
    /// </summary>
    private static DeviceMoe UploadDeviceMoeLayer(NemotronHMoeWeights w, int layerIdx, int hiddenSize,
        ref long maxTileFloats, List<nint> allocs)
    {
        int expertDim = w.HasLatent ? w.LatentSize : hiddenSize;
        ValidateBank(layerIdx, "ffn_up_exps", w.UpExpsQuantType, expertDim);
        ValidateBank(layerIdx, "ffn_down_exps", w.DownExpsQuantType, w.MoeIntermediateSize);

        int n = w.NumExperts;
        nint gate = UploadRawTensorFromHost(w.GateInpWeight, w.GateInpQuantType, n, hiddenSize, allocs);
        UpdateMaxTile(ref maxTileFloats, (long)n * hiddenSize);

        nint upSh = UploadRawTensorFromHost(w.UpShexpWeight, w.UpShexpQuantType, w.SharedIntermediateSize, hiddenSize, allocs);
        nint downSh = UploadRawTensorFromHost(w.DownShexpWeight, w.DownShexpQuantType, hiddenSize, w.SharedIntermediateSize, allocs);
        UpdateMaxTile(ref maxTileFloats, (long)w.SharedIntermediateSize * hiddenSize);

        nint latDown = 0, latUp = 0;
        if (w.HasLatent)
        {
            latDown = UploadRawTensorFromHost(w.LatentDownWeight, w.LatentDownQuantType, w.LatentSize, hiddenSize, allocs);
            latUp = UploadRawTensorFromHost(w.LatentUpWeight, w.LatentUpQuantType, hiddenSize, w.LatentSize, allocs);
            UpdateMaxTile(ref maxTileFloats, (long)w.LatentSize * hiddenSize);
        }

        nint upBank = UploadBank(layerIdx, "ffn_up_exps", w.UpExpsWeight, w.UpPerExpertBytes * n, allocs);
        nint downBank = UploadBank(layerIdx, "ffn_down_exps", w.DownExpsWeight, w.DownPerExpertBytes * n, allocs);
        var up = new nint[n];
        var down = new nint[n];
        for (int e = 0; e < n; e++)
        {
            up[e] = upBank + (nint)(e * w.UpPerExpertBytes);
            down[e] = downBank + (nint)(e * w.DownPerExpertBytes);
        }

        return new DeviceMoe
        {
            Host = w,
            GateInp = gate, GateInpQt = w.GateInpQuantType,
            UpBank = upBank, DownBank = downBank, Up = up, Down = down,
            UpQt = w.UpExpsQuantType, DownQt = w.DownExpsQuantType,
            UpPerExpertBytes = w.UpPerExpertBytes, DownPerExpertBytes = w.DownPerExpertBytes,
            UpShexp = upSh, UpShexpQt = w.UpShexpQuantType, DownShexp = downSh, DownShexpQt = w.DownShexpQuantType,
            LatentDown = latDown, LatentDownQt = w.LatentDownQuantType, LatentUp = latUp, LatentUpQt = w.LatentUpQuantType,
            NumExperts = n, NumExpertsPerTok = w.NumExpertsPerTok, MoeIntermediateSize = w.MoeIntermediateSize,
            LatentSize = w.LatentSize, SharedIntermediateSize = w.SharedIntermediateSize,
            NormalizeWeights = w.NormalizeWeights, WeightsScale = w.WeightsScale,
        };
    }

    private static void ValidateBank(int layerIdx, string name, QuantizationType qt, int k)
    {
        if (!CudaKernels.SupportsMoeExpertGroupedGemv(qt))
            throw new NotSupportedException(
                $"blk.{layerIdx}.{name}.weight is {qt}: the CUDA nemotron_h_moe expert kernels read Q4_K, Q5_K, Q6_K, Q8_0 " +
                "and IQ2_XXS banks only (issue #403). Use --device cpu or vulkan for this file.");
        int mul = CudaKernels.MoeExpertGroupedKMultiple(qt);
        if (k % mul != 0)
            throw new NotSupportedException(
                $"blk.{layerIdx}.{name}.weight ({qt}) has a row length of {k}, not a multiple of {mul}; the CUDA expert kernel " +
                "cannot read it (issue #403).");
    }

    private static nint UploadBank(int layerIdx, string name, nint host, long bytes, List<nint> allocs)
    {
        nint device;
        try
        {
            device = AllocDevice(bytes, allocs);
        }
        catch (CudaException ex) when (ex.ErrorCode == 2) // CUDA_ERROR_OUT_OF_MEMORY
        {
            throw new InvalidOperationException(
                $"blk.{layerIdx}.{name}.weight ({bytes / (1024 * 1024)} MiB) does not fit in free VRAM: this CUDA build runs " +
                "nemotron_h_moe with every expert resident. Expert offload for GPUs smaller than the model is tracked in " +
                "issue #1063; until then use --device cpu or vulkan, or a smaller quantization.", ex);
        }
        CopyHtoD(device, host, bytes);
        return device;
    }

    private static void FreeMoe(DeviceMoe m)
    {
        FreeIfNonZeroValue(m.GateInp);
        FreeIfNonZeroValue(m.UpBank);
        FreeIfNonZeroValue(m.DownBank);
        FreeIfNonZeroValue(m.UpShexp);
        FreeIfNonZeroValue(m.DownShexp);
        FreeIfNonZeroValue(m.LatentDown);
        FreeIfNonZeroValue(m.LatentUp);
    }

    /// <summary>The widest MoE geometry over <paramref name="layers"/>; <c>default</c> when there is no MoE layer.</summary>
    private static MoeScratchSpec ComputeMoeSpec(DeviceLayer[] layers, int hiddenSize)
    {
        var spec = default(MoeScratchSpec);
        foreach (var l in layers)
            if (l.Moe is { } m) spec = MoeScratchSpec.Widen(spec, m.Host, hiddenSize);
        return spec;
    }

    /// <summary>Schedule bytes for up to <paramref name="experts"/> groups and <paramref name="slots"/> slots.</summary>
    private static long MoeScheduleBytes(int experts, int slots)
        => 2L * experts * sizeof(long) + (experts + 1L) * sizeof(int) + 3L * slots * sizeof(int);

    /// <summary>
    /// Allocates the routed-MoE scratch once, at load, for blocks of <paramref name="tokens"/> tokens, sized from the
    /// per-layer maxima (no lazy growth). Frees whatever it allocated if an allocation fails.
    /// </summary>
    private void AllocateMoeScratch(int tokens)
    {
        try
        {
            AllocateMoeScratchCore(tokens);
        }
        catch
        {
            DisposeMoe();
            throw;
        }
    }

    private void AllocateMoeScratchCore(int tokens)
    {
        var s = _moeSpec;
        int hidden = Config.HiddenSize;
        long slots = (long)tokens * s.TopK;
        _moeRouterLogits = AllocScratch((long)tokens * s.NumExperts);
        _moeXLatent = s.LatentSize > 0 ? AllocScratch((long)tokens * s.LatentSize) : 0;
        _moeShMid = AllocScratch((long)tokens * Math.Max(s.SharedIntermediate, 1));
        _moeShOut = AllocScratch((long)tokens * hidden);
        _moeMid = AllocScratch(slots * s.MoeIntermediate);
        _moeSlotOut = AllocScratch(slots * s.ExpertDim);
        _moeAcc = AllocScratch((long)tokens * s.ExpertDim);

        long schedBytes = MoeScheduleBytes(s.NumExperts, checked((int)slots));
        CudaDriverApi.cuMemAlloc_v2(out _moeSchedDevice, (nuint)schedBytes).ThrowOnError();
        _moePinnedScheduleOffset = ((long)tokens * s.NumExperts * sizeof(float) + 15) & ~15L;
        CudaDriverApi.cuMemHostAlloc(out _moePinned, (nuint)(_moePinnedScheduleOffset + schedBytes), 0).ThrowOnError();
        if (_moeEvent == 0)
            CudaDriverApi.cuEventCreate(out _moeEvent, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();

        _moeSelScratch = new float[s.NumExperts];
        _moeTopIdx = new int[slots];
        _moeWeights = new float[slots];
        _moeCounts = new int[s.NumExperts + 1];
        _moeFill = new int[s.NumExperts];
        _moeScratchTokens = tokens;

        static nint AllocScratch(long floats)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)(Math.Max(floats, 1) * sizeof(float))).ThrowOnError();
            return p;
        }
    }

    private void FreeMoeScratch()
    {
        FreeIfNonZero(ref _moeRouterLogits);
        FreeIfNonZero(ref _moeXLatent);
        FreeIfNonZero(ref _moeShMid);
        FreeIfNonZero(ref _moeShOut);
        FreeIfNonZero(ref _moeMid);
        FreeIfNonZero(ref _moeSlotOut);
        FreeIfNonZero(ref _moeAcc);
        FreeIfNonZero(ref _moeSchedDevice);
        if (_moePinned != 0) { CudaDriverApi.cuMemFreeHost(_moePinned); _moePinned = 0; }
        _moeScratchTokens = 0;
    }

    /// <summary>Frees the routed scratch, the pinned staging and the routing event (called from <see cref="Dispose"/>).</summary>
    private void DisposeMoe()
    {
        FreeMoeScratch();
        if (_moeEvent != 0) { CudaDriverApi.cuEventDestroy_v2(_moeEvent); _moeEvent = 0; }
    }

    /// <summary>
    /// MoE FFN sub-layer: reads the normed activations from <c>_state.NormOutput</c> and writes the layer output back into
    /// it (the contract of every sub-layer body). Token blocks of at most <see cref="MoeBlockTokens"/>.
    /// </summary>
    private void ForwardMoeBody(DeviceMoe moe, int seqLen, int hiddenSize)
    {
        // The scratch was sized at load for blocks of _moeScratchTokens tokens.
        int block = _moeScratchTokens;
        for (int start = 0; start < seqLen; start += block)
        {
            int n = Math.Min(block, seqLen - start);
            ForwardMoeBlock(moe, n, hiddenSize, _state.NormOutput + (nint)((long)start * hiddenSize * sizeof(float)));
        }
    }

    private void ForwardMoeBlock(DeviceMoe moe, int n, int hiddenSize, nint h)
    {
        nint streamH = _stream.Handle;
        int nExpert = moe.NumExperts;
        int topK = moe.NumExpertsPerTok;
        int expertDim = moe.ExpertDim(hiddenSize);
        Interlocked.Increment(ref MoePathCounters.LayerCalls);

        // 1-2. Router on the full h, logits to pinned host memory, then the event the host waits on.
        Gemm(moe.GateInp, moe.GateInpQt, h, _moeRouterLogits, nExpert, hiddenSize, n);
        CudaDriverApi.cuMemcpyDtoHAsync_v2(_moePinned, _moeRouterLogits,
            (nuint)((long)n * nExpert * sizeof(float)), streamH).ThrowOnError();
        CudaDriverApi.cuEventRecord(_moeEvent, streamH).ThrowOnError();

        // 3. Work the GPU runs while the host waits: latent_down(h) and the shared expert (up -> relu^2 -> down).
        if (moe.LatentSize > 0)
            Gemm(moe.LatentDown, moe.LatentDownQt, h, _moeXLatent, moe.LatentSize, hiddenSize, n);
        Gemm(moe.UpShexp, moe.UpShexpQt, h, _moeShMid, moe.SharedIntermediateSize, hiddenSize, n);
        _kernels.LaunchReluSquaredInplaceF32(_moeShMid, n * moe.SharedIntermediateSize, streamH);
        Gemm(moe.DownShexp, moe.DownShexpQt, _moeShMid, _moeShOut, hiddenSize, moe.SharedIntermediateSize, n);

        // 4-5. Route on the host (the CPU model's RouteToken) and send the CSR schedule.
        CudaDriverApi.cuEventSynchronize(_moeEvent).ThrowOnError();
        var sched = BuildMoeSchedule(moe, n);
        CudaDriverApi.cuMemcpyHtoDAsync_v2(_moeSchedDevice, _moePinned + (nint)_moePinnedScheduleOffset,
            (nuint)sched.TotalBytes, streamH).ThrowOnError();

        nint upPtrs = _moeSchedDevice + (nint)sched.UpPtrsOffset;
        nint downPtrs = _moeSchedDevice + (nint)sched.DownPtrsOffset;
        nint groupStart = _moeSchedDevice + (nint)sched.GroupStartOffset;
        nint slotIds = _moeSchedDevice + (nint)sched.SlotIdsOffset;
        nint slotToken = _moeSchedDevice + (nint)sched.SlotTokenOffset;
        nint weights = _moeSchedDevice + (nint)sched.WeightsOffset;

        // 6. Routed experts: up (+relu^2) reads the token's row of the expert input, down reads the slot's own row.
        nint expertIn = moe.LatentSize > 0 ? _moeXLatent : h;
        _kernels.LaunchMoeExpertGroupedGemvF32(moe.UpQt, upPtrs, groupStart, slotIds, slotToken,
            expertIn, _moeMid, moe.MoeIntermediateSize, expertDim, sched.Groups, relu2: true, streamH);
        MoePathCounters.CountLaunch(moe.UpQt, sched.Groups);
        _kernels.LaunchMoeExpertGroupedGemvF32(moe.DownQt, downPtrs, groupStart, slotIds, xRow: 0,
            _moeMid, _moeSlotOut, expertDim, moe.MoeIntermediateSize, sched.Groups, relu2: false, streamH);
        MoePathCounters.CountLaunch(moe.DownQt, sched.Groups);

        // 7. Weighted sum in slot order, back to the hidden size (latent_up), then + the shared expert.
        if (moe.LatentSize > 0)
        {
            _kernels.LaunchMoeWeightedSlotSumF32(_moeSlotOut, weights, _moeAcc, n, topK, expertDim, streamH);
            Gemm(moe.LatentUp, moe.LatentUpQt, _moeAcc, h, hiddenSize, moe.LatentSize, n);
        }
        else
        {
            // Every reader of h (router, shared expert, up projection) is already queued ahead of this write.
            _kernels.LaunchMoeWeightedSlotSumF32(_moeSlotOut, weights, h, n, topK, hiddenSize, streamH);
        }
        _kernels.LaunchAddF32(h, _moeShOut, h, n * hiddenSize, streamH);
    }

    /// <summary>Byte offsets of the schedule arrays inside the pinned staging / device schedule buffer.</summary>
    private readonly record struct MoeSchedule(int Groups, long UpPtrsOffset, long DownPtrsOffset, long GroupStartOffset,
        long SlotIdsOffset, long SlotTokenOffset, long WeightsOffset, long TotalBytes);

    /// <summary>
    /// Routes the <paramref name="n"/> tokens whose router logits sit at the start of the pinned buffer and writes the CSR
    /// schedule into the pinned staging area: per used expert (ascending id) its up and down device pointers and its slot
    /// range; the slot ids grouped by expert (a counting sort, ascending slot within an expert, as the CPU grouped body);
    /// per slot its token; per slot its gating weight. Slot id = <c>token · topK + k</c>.
    /// </summary>
    private MoeSchedule BuildMoeSchedule(DeviceMoe moe, int n)
    {
        int nExpert = moe.NumExperts;
        int topK = moe.NumExpertsPerTok;
        int slots = n * topK;
        float* logits = (float*)_moePinned;
        byte* stage = (byte*)_moePinned + _moePinnedScheduleOffset;

        // The group count is not known until every token is routed, so route into managed arrays first.
        int[] topIdx = _moeTopIdx;
        float[] wSlot = _moeWeights;
        int[] counts = _moeCounts;
        Array.Clear(counts, 0, nExpert + 1);
        var sel = _moeSelScratch.AsSpan(0, nExpert);
        for (int t = 0; t < n; t++)
        {
            NemotronHTransformerModel.RouteToken(moe.Host,
                new Span<float>(logits + (long)t * nExpert, nExpert), sel,
                topIdx.AsSpan(t * topK, topK), wSlot.AsSpan(t * topK, topK));
            for (int k = 0; k < topK; k++) counts[topIdx[t * topK + k] + 1]++;
        }

        int groups = 0;
        for (int e = 0; e < nExpert; e++) if (counts[e + 1] > 0) groups++;
        for (int e = 0; e < nExpert; e++) counts[e + 1] += counts[e];   // counts[e] = first slot of expert e

        long upOff = 0;
        long downOff = upOff + (long)groups * sizeof(long);
        long gsOff = downOff + (long)groups * sizeof(long);
        long slotOff = gsOff + (groups + 1L) * sizeof(int);
        long tokOff = slotOff + (long)slots * sizeof(int);
        long wOff = tokOff + (long)slots * sizeof(int);
        long total = wOff + (long)slots * sizeof(float);

        long* upPtrs = (long*)(stage + upOff);
        long* downPtrs = (long*)(stage + downOff);
        int* groupStart = (int*)(stage + gsOff);
        int* slotIds = (int*)(stage + slotOff);
        int* slotToken = (int*)(stage + tokOff);
        float* weights = (float*)(stage + wOff);

        wSlot.AsSpan(0, slots).CopyTo(new Span<float>(weights, slots));

        int g = 0;
        for (int e = 0; e < nExpert; e++)
        {
            if (counts[e + 1] == counts[e]) continue;
            upPtrs[g] = moe.Up[e];
            downPtrs[g] = moe.Down[e];
            groupStart[g] = counts[e];
            g++;
        }
        groupStart[groups] = slots;

        int[] fill = _moeFill;
        Array.Copy(counts, fill, nExpert);
        for (int s = 0; s < slots; s++) slotIds[fill[topIdx[s]]++] = s;
        for (int s = 0; s < slots; s++) slotToken[s] = s / topK;

        return new MoeSchedule(groups, upOff, downOff, gsOff, slotOff, tokOff, wOff, total);
    }
}
