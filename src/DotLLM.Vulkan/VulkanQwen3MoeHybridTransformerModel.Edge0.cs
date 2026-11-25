using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    private Edge0LatentKernels? _edge0Kernels;
    private VulkanDevice.Buffer? _edge0Xl, _edge0ZAll;
    private VulkanDevice.Buffer[]? _edge0Z, _edge0Uo;
    private int _edge0Rows;

    /// <summary>Single-row decode skips the D / U groups no selected expert uses (<c>edge0_d_mmvq</c> / <c>edge0_u_mmvq</c>): identical numerics, fewer bytes. <c>DOTLLM_EDGE0_SKIP_GROUPS=0</c> (or setting this) restores the generic path (A/B).</summary>
    internal static bool Edge0SkipUnusedGroups { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_SKIP_GROUPS") != "0";

    private static int ParseStore(string name) => Environment.GetEnvironmentVariable(name)?.ToLowerInvariant() switch { "f16" => 1, "bf16" => 2, "q8" => 3, _ => 0 };
    /// <summary>Experiment (#1981): simulated storage format of the D output, the per-group z and the per-group U output (0 = F32, 1 = F16, 2 = BF16, 3 = Q8_0).</summary>
    internal static int Edge0SimXl { get; set; } = ParseStore("DOTLLM_EDGE0_STORE_XL");
    internal static int Edge0SimZ { get; set; } = ParseStore("DOTLLM_EDGE0_STORE_Z");
    internal static int Edge0SimUo { get; set; } = ParseStore("DOTLLM_EDGE0_STORE_UO");

    /// <summary>Test hook: layers recorded on the group-skipping decode path.</summary>
    internal long Edge0SkipLayers { get; private set; }

    internal long Edge0SkipLayersCount => Edge0SkipLayers;

    /// <summary>Test hook: Edge0 latent-MoE layers recorded.</summary>
    internal long Edge0LayersRecorded { get; private set; }

    /// <summary>
    /// Attaches the Edge0 latent-MoE glue kernels and allocates their per-forward scratch for up to <paramref name="rows"/> rows (D output for all
    /// groups, eight per-group z and U-output slabs). Call once, after the final planned row count is known; the scratch is never grown while recording.
    /// </summary>
    internal void Q4AttachEdge0(Edge0LatentKernels kernels, int groups, int latent, int hidden, int rows)
    {
        if (_kernels.QuantizeQ8_1RowsActivations is null || kernels.IndexedQ3K is null || kernels.IndexedIq4Nl is null)
            throw new NotSupportedException("The Edge0 Vulkan path needs integer-dot-product support (the Q3_K / IQ4_NL expert banks are read through the dp4a MMVQ kernels).");
        FreeEdge0Scratch();
        _edge0Kernels = kernels;
        _edge0Rows = rows;
        _edge0Xl = _device.AllocateDeviceLocal((long)rows * groups * latent * sizeof(float));
        _edge0ZAll = _device.AllocateDeviceLocal((long)groups * latent * sizeof(float));
        _edge0Z = new VulkanDevice.Buffer[groups];
        _edge0Uo = new VulkanDevice.Buffer[groups];
        for (int g = 0; g < groups; g++)
        {
            _edge0Z[g] = _device.AllocateDeviceLocal((long)rows * latent * sizeof(float));
            _edge0Uo[g] = _device.AllocateDeviceLocal((long)rows * hidden * sizeof(float));
        }
        kernels.InvalidateDescriptorCache();
    }

    /// <summary>Device bytes of the Edge0 per-forward scratch (0 when not attached).</summary>
    internal long Edge0ScratchBytes => _edge0Xl is null ? 0 : _edge0Xl.Size + _edge0Z!.Sum(b => b.Size) + _edge0Uo!.Sum(b => b.Size);

    private void FreeEdge0Scratch()
    {
        _edge0Xl?.Dispose(); _edge0Xl = null;
        _edge0ZAll?.Dispose(); _edge0ZAll = null;
        if (_edge0Z is not null) foreach (var b in _edge0Z) b.Dispose();
        if (_edge0Uo is not null) foreach (var b in _edge0Uo) b.Dispose();
        _edge0Z = _edge0Uo = null;
    }

    /// <summary>
    /// The MoE half of one Edge0 layer (the block input is already in <c>NormOutput</c>): router on the BF16 matrix, top-k, D for every group, the
    /// per-(token, expert) gather with the 1/s column scale, the Q3_K / IQ4_NL experts (indexed MMVQ below the grouped threshold, grouped coopmat
    /// above) at width <c>latent</c>, the per-group routing-weighted sum, U per group, the sum of the group outputs, then the sigmoid-gated shared
    /// expert. Semantics are those of <c>Edge0LatentMoe</c> on the CPU.
    /// </summary>
    private void RecordEdge0MoeLayer(nint cmd, VulkanQwen3MoeMoeUpload.LayerBundle moeW, int T)
    {
        var e0 = moeW.Edge0!;
        var k = _edge0Kernels ?? throw new InvalidOperationException("Edge0 kernels were not attached (Q4AttachEdge0).");
        if (T > _edge0Rows) throw new InvalidOperationException($"Edge0 scratch holds {_edge0Rows} rows, the forward has {T}.");
        int H = e0.Hidden, L = e0.Latent, G = e0.Groups, topK = moeW.NumExpertsPerTok, E = moeW.NumExperts, interm = moeW.IntermediateSize;
        int rows = T * topK;
        MoeStageBegin();

        // The routed result overwrites NormOutput; the shared expert reads the untouched block input from MoeSharedInput.
        RecordCopyBufferRange(cmd, _state.NormOutput, _state.MoeSharedInput, 0, 0, (ulong)((long)T * H * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);

        // 1. router (BF16 read in place) + top-k
        RecordMatmul(cmd, moeW.Gate, QuantizationType.BF16, _state.NormOutput, _state.MoeRouterLogits, outputDim: E, inputDim: H, seqLen: T);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.MoeTopkSoftmax.Record(cmd, _state.MoeRouterLogits, _state.MoeTopkIndices, _state.MoeTopkWeights,
            seqLen: T, numExperts: E, k: topK, normTopKProb: moeW.NormTopKProb);
        KernelSupport.ComputeToComputeBarrier(cmd);

        // 2. D for every group, then gather the routed group row of each (token, expert) with the 1/s column scale
        bool skip = T == 1 && Edge0SkipUnusedGroups && k.HasDecodeKernels && e0.LatentDenseType == QuantizationType.Q8_0 && (H % 32) == 0 && (L % 32) == 0
            && _kernels.QuantizeQ8_1RowsActivations is not null && TryPrepareQ8Activations(cmd, _state.NormOutput, H);
        if (skip)
            k.RecordDecodeD(cmd, e0.D, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _edge0Xl!, _state.MoeTopkIndices, e0.Labels, G, L, H, topK, E);
        else
            RecordMatmul(cmd, e0.D, e0.LatentDenseType, _state.NormOutput, _edge0Xl!, outputDim: G * L, inputDim: H, seqLen: T);
        KernelSupport.ComputeToComputeBarrier(cmd);
        if (Edge0SimXl != 0) { k.RecordRoundTrip(cmd, _edge0Xl!, T * G * L, Edge0SimXl); KernelSupport.ComputeToComputeBarrier(cmd); }
        k.RecordGather(cmd, _edge0Xl!, _state.MoeTopkIndices, e0.Labels, e0.InverseScale, _state.MoeExpandedInput, L, G, topK, rows, E);
        KernelSupport.ComputeToComputeBarrier(cmd);
        MoeStage("edge0_d_gather");

        // 3. experts at width L (gate/up K = L, down M = L); the result rows land in MoeDownRows [rows, L]
        long stride = e0.RecordBytes, upOff = e0.GateUpBytes, downOff = 2 * e0.GateUpBytes;
        var bank = e0.Records;
        bool grouped = T >= GroupedMinTokens && k.GroupedQ3K is not null && k.GroupedIq4Nl is not null && _kernels.MoeBuildTileList is not null
            && _kernels.MoeExpertOffsets is not null && _kernels.MoeExpandGroupByExpert is not null && _kernels.MoeUngroupScatter is not null && (L % 256) == 0;
        if (grouped)
        {
            var gateK = k.GroupedQ3K!; var downK = k.GroupedIq4Nl!;
            _kernels.MoeExpertOffsets!.Record(cmd, _state.MoeTopkIndices, _state.MoeGroupCounts, _state.MoeGroupOffsets, _state.MoeGroupCounters, rows: rows, numExperts: E);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.MoeExpandGroupByExpert!.Record(cmd, _state.MoeExpandedInput, _state.MoeTopkIndices, _state.MoeGroupOffsets,
                _state.MoeGroupCounters, _state.MoeDownRows, _state.MoeGroupPerm, rows: rows, hidden: L, numExperts: E);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.MoeBuildTileList!.Record(cmd, _state.MoeGroupOffsets, _state.MoeGroupDispatchArgs, E, gateK.MTiles(interm), downK.MTiles(L), tileRows: gateK.RowTile);
            KernelSupport.ComputeToIndirectAndComputeBarrier(cmd);
            gateK.RecordIndirect(cmd, bank, _state.MoeDownRows, _state.MoeGroupOffsets, _state.MoeGateInter, _state.MoeGroupDispatchArgs, 0, interm, L, rows, E, stride, 0);
            gateK.RecordIndirect(cmd, bank, _state.MoeDownRows, _state.MoeGroupOffsets, _state.MoeUpInter, _state.MoeGroupDispatchArgs, 0, interm, L, rows, E, stride, upOff);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.SwiGlu.Record(cmd, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter, n: rows * interm);
            KernelSupport.ComputeToComputeBarrier(cmd);
            downK.RecordIndirect(cmd, bank, _state.MoeSiluInter, _state.MoeGroupOffsets, _state.MoeExpandedInput, _state.MoeGroupDispatchArgs,
                MoeBuildTileListKernel.ArgsStrideBytes, L, interm, rows, E, stride, downOff);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.MoeUngroupScatter!.Record(cmd, _state.MoeExpandedInput, _state.MoeGroupPerm, _state.MoeDownRows, rows: rows, hidden: L);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        else
        {
            _kernels.QuantizeQ8_1RowsActivations!.Record(cmd, _state.MoeExpandedInput, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: rows, k: L);
            KernelSupport.ComputeToComputeBarrier(cmd);
            k.IndexedQ3K!.Record(cmd, bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _state.MoeTopkIndices, _state.MoeGateInter, interm, L, rows, E, stride, 0);
            k.IndexedQ3K.Record(cmd, bank, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _state.MoeTopkIndices, _state.MoeUpInter, interm, L, rows, E, stride, upOff);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.SwiGlu.Record(cmd, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter, n: rows * interm);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.QuantizeQ8_1RowsActivations.Record(cmd, _state.MoeSiluInter, _state.MoeSiluInterXq, _state.MoeSiluInterXds, n: rows, k: interm);
            KernelSupport.ComputeToComputeBarrier(cmd);
            k.IndexedIq4Nl!.Record(cmd, bank, _state.MoeSiluInterXq, _state.MoeSiluInterXds, _state.MoeTopkIndices, _state.MoeDownRows, L, interm, rows, E, stride, downOff);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        MoeStage("edge0_experts");

        // 4. per-group weighted sum of the expert outputs, U per group, sum of the group outputs -> NormOutput
        if (skip)
        {
            Edge0SkipLayers++;
            k.RecordCombineAll(cmd, _state.MoeDownRows, _state.MoeTopkIndices, _state.MoeTopkWeights, e0.Labels, _edge0ZAll!, L, topK, E);
            KernelSupport.ComputeToComputeBarrier(cmd);
            if (Edge0SimZ != 0) { k.RecordRoundTrip(cmd, _edge0ZAll!, G * L, Edge0SimZ); KernelSupport.ComputeToComputeBarrier(cmd); }
            _kernels.QuantizeQ8_1RowsActivations!.Record(cmd, _edge0ZAll!, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: G, k: L);
            KernelSupport.ComputeToComputeBarrier(cmd);
            k.RecordDecodeU(cmd, e0.U, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _state.NormOutput, _state.MoeTopkIndices, e0.Labels, H, L, topK, E);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        else
        {
            for (int g = 0; g < G; g++)
                k.RecordGroupCombine(cmd, _state.MoeDownRows, _state.MoeTopkIndices, _state.MoeTopkWeights, e0.Labels, _edge0Z![g], L, topK, g, T, E);
            KernelSupport.ComputeToComputeBarrier(cmd);
            if (Edge0SimZ != 0) { for (int g = 0; g < G; g++) k.RecordRoundTrip(cmd, _edge0Z![g], T * L, Edge0SimZ); KernelSupport.ComputeToComputeBarrier(cmd); }
            for (int g = 0; g < G; g++)
            {
                RecordMatmul(cmd, e0.U[g], e0.LatentDenseType, _edge0Z![g], _edge0Uo![g], outputDim: H, inputDim: L, seqLen: T);
                KernelSupport.ComputeToComputeBarrier(cmd);
                if (Edge0SimUo != 0) { k.RecordRoundTrip(cmd, _edge0Uo![g], T * H, Edge0SimUo); KernelSupport.ComputeToComputeBarrier(cmd); }   // a single-row Q8_0 / Q4_0 GEMV quantizes its input into the shared Q8_1 scratch
            }
            k.RecordSum(cmd, _edge0Uo!, _state.NormOutput, T * H);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        MoeStage("edge0_u");

        // 5. sigmoid-gated shared expert, added into NormOutput
        if (moeW.HasSharedExpert) RecordSharedExpert(cmd, moeW, postAttnNormWeight: null, T, H, Config.NormEpsilon);
        Edge0LayersRecorded++;
    }
}
