using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    private Edge0LatentKernels? _edge0Kernels;
    // The ONLY per-forward Edge0 scratch (James's rule-1 rulings, #1981): the D output for every group as F16, ONE F16 slab of z (one group, more than 8 rows) and ONE F16 slab of U
    // partials. No F32 tensor is stored: z for up to 8 rows is rounded to F16 and packed as Q8_1 into the existing MoeExpandedInputXq/Xds, the U partials are summed into NormOutput
    // (F32 accumulation) in ascending group order.
    private VulkanDevice.Buffer? _edge0Xl, _edge0Z, _edge0Uo;
    private int _edge0Rows;
    private Kernels.MoeIndexedPlanarQ2MmvqKernel? _mtpPlanar;
    private Kernels.MatMulQ8_0MmvqRowsKernel? _draftRows;

    /// <summary>Single-row decode skips the D / U groups no selected expert uses (<c>edge0_d_mmvq</c> / <c>edge0_u_mmvq</c>): identical numerics, fewer bytes. <c>DOTLLM_EDGE0_SKIP_GROUPS=0</c> (or setting this) visits every group (A/B).</summary>
    internal static bool Edge0SkipUnusedGroups { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_SKIP_GROUPS") != "0";

    /// <summary>Attaches the planar Q2_0 kernel the Edge0 MTP head's MoE layer is recorded with (G6, #1990). Owned by this core.</summary>
    internal void Q4AttachMtpPlanar(Kernels.MoeIndexedPlanarQ2MmvqKernel kernel) { _mtpPlanar?.Dispose(); _mtpPlanar = kernel; }

    /// <summary>Attaches the Q8_0 row-subset GEMV the draft-vocab head is scored with (#2013). Owned by this core; false when the device lacks integer dot product or the SPIR-V.</summary>
    internal bool Q4AttachDraftRows(string spvDir)
    {
        _draftRows ??= Kernels.MatMulQ8_0MmvqRowsKernel.TryCreate(_device, spvDir);
        return _draftRows is not null && _kernels.MatMulQ8Mmvq is not null && _kernels.QuantizeQ8_1RowsActivations is not null;
    }

    /// <summary>
    /// Records <c>y[i] = W_q8[rows[i]] . x</c> for the single decode row in <paramref name="input"/> (quantized to Q8_1 here): a Q8_0 matrix read in place at
    /// the listed rows only (#2013). False when the kernels or scratch are unavailable (nothing recorded).
    /// </summary>
    internal bool Q4TryRecordMatmulRows(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output, VulkanDevice.Buffer rows,
        int m, int k, int weightRows)
    {
        if (_draftRows is null || !TryPrepareQ8Activations(cmdBuf, input, k)) return false;
        _draftRows.Record(cmdBuf, weights, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, output, rows, m, k, weightRows);
        return true;
    }

    /// <summary>Test hook: layers recorded on the group-skipping decode path.</summary>
    internal long Edge0SkipLayers { get; private set; }

    internal long Edge0SkipLayersCount => Edge0SkipLayers;

    /// <summary>
    /// The F16-output wide GEMVs serve D and U at 9..16 rows (#2004). Off (<c>DOTLLM_EDGE0_WIDE_GEMV=0</c>) sends those rows to the blocked coopmat GEMM, the
    /// pre-#2004 behaviour (the A/B arm).
    /// </summary>
    internal static bool Edge0WideGemvEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_WIDE_GEMV") != "0";

    /// <summary>Test hook: D / U dispatches that ran on the F16 wide GEMV.</summary>
    internal long Edge0WideDispatches { get; private set; }

    /// <summary>Test hook: Edge0 latent-MoE layers recorded.</summary>
    internal long Edge0LayersRecorded { get; private set; }

    /// <summary>
    /// Attaches the Edge0 latent-MoE glue kernels and allocates their per-forward scratch for up to <paramref name="rows"/> rows: the D output for all groups
    /// (F16, <c>rows * groups * latent * 2</c> bytes) one group's z (F16, <c>rows * latent * 2</c> bytes) and one group's U output (F16, <c>rows * hidden * 2</c> bytes). Call once, after the final planned row count is known;
    /// the scratch is never grown while recording.
    /// </summary>
    internal void Q4AttachEdge0(Edge0LatentKernels kernels, int groups, int latent, int hidden, int rows)
    {
        if (_kernels.QuantizeQ8_1RowsActivations is null || kernels.IndexedQ3K is null || kernels.IndexedIq4Nl is null)
            throw new NotSupportedException("The Edge0 Vulkan path needs integer-dot-product support (the Q3_K / IQ4_NL expert banks are read through the dp4a MMVQ kernels).");
        if (!kernels.HasNarrowPath)
            throw new NotSupportedException("The Edge0 Vulkan path needs the F16-output D / U kernels: integer dot product with wave32 (single-row), and cooperative matrices with a wave64 device (the blocked 128x128 Q8_0 GEMMs for more than a few rows).");
        if (groups != 8) throw new NotSupportedException($"The Edge0 Vulkan path is written for 8 D/U groups, not {groups}.");
        if ((latent & 63) != 0 || (hidden & 63) != 0) throw new NotSupportedException($"Edge0 latent {latent} / hidden {hidden} must be multiples of 64 for the Q8_0 GEMMs.");
        FreeEdge0Scratch();
        _edge0Kernels = kernels;
        _edge0Rows = rows;
        _edge0Xl = _device.AllocateDeviceLocal((long)rows * groups * latent * sizeof(ushort));
        _edge0Z = _device.AllocateDeviceLocal((long)rows * latent * sizeof(ushort));
        _edge0Uo = _device.AllocateDeviceLocal((long)rows * hidden * sizeof(ushort));
        kernels.InvalidateDescriptorCache();
    }

    /// <summary>Device bytes of the Edge0 per-forward scratch (0 when not attached): the F16 D output, one F16 z slab and one F16 U slab. Nothing else.</summary>
    internal long Edge0ScratchBytes => (_edge0Xl?.Size ?? 0) + (_edge0Z?.Size ?? 0) + (_edge0Uo?.Size ?? 0);

    /// <summary>
    /// The MoE half of the Edge0 MTP head's block (G6, #1990): the generic routed-expert path (router, top-k softmax, broadcast, gate/up, SwiGLU, down, weighted scatter, sigmoid-gated
    /// shared expert) with the three expert projections read in place from the planar Q2_0 blobs by <see cref="Kernels.MoeIndexedPlanarQ2MmvqKernel"/> (indexed for every row count:
    /// the head runs one row per draft step and AbsorbChunk-row chunks for the prompt).
    /// </summary>
    private void RecordEdge0MtpMoeLayer(nint cmd, VulkanQwen3MoeMoeUpload.LayerBundle moeW, int T)
    {
        var k = _mtpPlanar ?? throw new InvalidOperationException("The planar Q2_0 kernel was not attached (Q4AttachMtpPlanar).");
        int H = Config.HiddenSize, topK = moeW.NumExpertsPerTok, E = moeW.NumExperts, interm = moeW.IntermediateSize, rows = T * topK;
        MoeStageBegin();
        RecordCopyBufferRange(cmd, _state.NormOutput, _state.MoeSharedInput, 0, 0, (ulong)((long)T * H * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
        RecordMatmul(cmd, moeW.Gate, QuantizationType.F32, _state.NormOutput, _state.MoeRouterLogits, outputDim: E, inputDim: H, seqLen: T);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.MoeTopkSoftmax.Record(cmd, _state.MoeRouterLogits, _state.MoeTopkIndices, _state.MoeTopkWeights, seqLen: T, numExperts: E, k: topK, normTopKProb: moeW.NormTopKProb);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.MoeBroadcast.Record(cmd, _state.NormOutput, _state.MoeExpandedInput, seqLen: T, topK: topK, hidden: H);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.QuantizeQ8_1RowsActivations!.Record(cmd, _state.MoeExpandedInput, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: rows, k: H);
        KernelSupport.ComputeToComputeBarrier(cmd);
        var blobs = moeW.Edge0Mtp!.Blobs;
        k.Record(cmd, blobs, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _state.MoeTopkIndices, _state.MoeGateInter, Kernels.PlanarQ2Projection.Gate, interm, H, rows, E);
        k.Record(cmd, blobs, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _state.MoeTopkIndices, _state.MoeUpInter, Kernels.PlanarQ2Projection.Up, interm, H, rows, E);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.SwiGlu.Record(cmd, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter, n: rows * interm);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.QuantizeQ8_1RowsActivations.Record(cmd, _state.MoeSiluInter, _state.MoeSiluInterXq, _state.MoeSiluInterXds, n: rows, k: interm);
        KernelSupport.ComputeToComputeBarrier(cmd);
        k.Record(cmd, blobs, _state.MoeSiluInterXq, _state.MoeSiluInterXds, _state.MoeTopkIndices, _state.MoeDownRows, Kernels.PlanarQ2Projection.Down, H, interm, rows, E);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.MoeWeightedScatter.Record(cmd, _state.MoeDownRows, _state.MoeTopkWeights, _state.NormOutput, seqLen: T, topK: topK, hiddenSize: H);
        KernelSupport.ComputeToComputeBarrier(cmd);
        if (moeW.HasSharedExpert) RecordSharedExpert(cmd, moeW, postAttnNormWeight: null, T, H, Config.NormEpsilon);
    }

    private void FreeEdge0Scratch()
    {
        _edge0Xl?.Dispose(); _edge0Xl = null;
        _edge0Z?.Dispose(); _edge0Z = null;
        _edge0Uo?.Dispose(); _edge0Uo = null;
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

        // 2. D for every group (F16 output), then gather the routed group row of each (token, expert) with the 1/s column scale
        if (e0.LatentDenseType != QuantizationType.Q8_0)
            throw new NotSupportedException($"The Vulkan Edge0 path stores no F32 scratch and has F16-output kernels for a Q8_0 latent only, not {e0.LatentDenseType} (DOTLLM_EDGE0_LATENT=q4 packs run on the CPU).");
        if (_state.MoeExpandedInputXq.Size < QuantizeQ8_1RowsKernel.PackedBytes(T, H) || _state.MoeExpandedInputXds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(T, H))
            throw new InvalidOperationException("The shared Q8_1 scratch is too small for the Edge0 D input.");
        bool skip = Edge0SkipUnusedGroups;
        var xqOps = _kernels.QuantizeQ8_1RowsActivations!;
        if (T == 1)
        {
            xqOps.Record(cmd, _state.NormOutput, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: 1, k: H);
            KernelSupport.ComputeToComputeBarrier(cmd);
            k.RecordDecodeD(cmd, e0.D, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _edge0Xl!, _state.MoeTopkIndices, e0.Labels, G, L, H, topK, E, skip);
        }
        else if (k.MultiC16 is { } multi && SmallRowGemvEnabled && MatMulQ8_0MmvqMultiKernel.Accepts(T, H))
        {
            xqOps.Record(cmd, _state.NormOutput, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: T, k: H);
            KernelSupport.ComputeToComputeBarrier(cmd);
            multi.Record(cmd, e0.D, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _edge0Xl!, m: G * L, k: H, n: T);
        }
        else if (k.WideD is { } wideD && Edge0WideGemvEnabled && SmallRowGemvEnabled && SmallRowWideEnabled && Edge0WideGemvKernel.Accepts(T, H))
        {
            Edge0WideDispatches++;
            wideD.Record(cmd, e0.D, _state.NormOutput, _edge0Xl!, m: G * L, k: H, n: T);   // 9..16 rows (#2004): F32 in, F16 out, no coopmat tile
        }
        else
            k.RecordGemmD(cmd, e0.D, _state.NormOutput, _edge0Xl!, G * L, H, T);
        KernelSupport.ComputeToComputeBarrier(cmd);
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

        // 4. per-group weighted sum of the expert outputs (z, F16), U per group (F16 output), summed into NormOutput in ascending group order
        if (T == 1)
        {
            if (skip) Edge0SkipLayers++;
            k.RecordCombineAll(cmd, _state.MoeDownRows, _state.MoeTopkIndices, _state.MoeTopkWeights, e0.Labels, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, L, topK, E);
            KernelSupport.ComputeToComputeBarrier(cmd);
            k.RecordDecodeU(cmd, e0.U, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _state.NormOutput, _state.MoeTopkIndices, e0.Labels, H, L, topK, E, skip);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        else
        {
            bool small = k.MultiC16 is { } && SmallRowGemvEnabled && MatMulQ8_0MmvqMultiKernel.Accepts(T, L);
            var wideU = Edge0WideGemvEnabled && SmallRowGemvEnabled && SmallRowWideEnabled && !small && Edge0WideGemvKernel.Accepts(T, L) ? k.WideU : null;
            for (int g = 0; g < G; g++)
            {
                if (small)
                {
                    // 2..8 rows: z (rounded to F16) as Q8_1 blocks for the dp4a small-row MMVQ
                    k.RecordGroupCombine(cmd, _state.MoeDownRows, _state.MoeTopkIndices, _state.MoeTopkWeights, e0.Labels, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, L, topK, g, T, E);
                    KernelSupport.ComputeToComputeBarrier(cmd);
                    k.MultiC16!.Record(cmd, e0.U[g], _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, _edge0Uo!, m: H, k: L, n: T);
                }
                else
                {
                    k.RecordGroupCombineF16(cmd, _state.MoeDownRows, _state.MoeTopkIndices, _state.MoeTopkWeights, e0.Labels, _edge0Z!, L, topK, g, T, E);
                    KernelSupport.ComputeToComputeBarrier(cmd);
                    if (wideU is not null) { Edge0WideDispatches++; wideU.Record(cmd, e0.U[g], _edge0Z!, _edge0Uo!, m: H, k: L, n: T); }   // 9..16 rows (#2004): F16 z in, F16 out
                    else k.RecordGemmU(cmd, e0.U[g], _edge0Z!, _edge0Uo!, H, L, T);
                }
                KernelSupport.ComputeToComputeBarrier(cmd);
                k.RecordUAcc(cmd, _edge0Uo!, _state.NormOutput, T * H, first: g == 0);
                KernelSupport.ComputeToComputeBarrier(cmd);   // the next group's combine overwrites the z scratch, its U overwrites the F16 slab
            }
        }
        MoeStage("edge0_u");

        // 5. sigmoid-gated shared expert, added into NormOutput
        if (moeW.HasSharedExpert) RecordSharedExpert(cmd, moeW, postAttnNormWeight: null, T, H, Config.NormEpsilon);
        Edge0LayersRecorded++;
    }
}
