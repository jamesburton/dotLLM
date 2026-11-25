using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    private Edge0LatentKernels? _edge0Kernels;
    private VulkanDevice.Buffer? _edge0Xl;
    private VulkanDevice.Buffer[]? _edge0Z, _edge0Uo;
    private int _edge0Rows;

    /// <summary>Test hook: Edge0 latent-MoE layers recorded.</summary>
    internal long Edge0LayersRecorded { get; private set; }

    /// <summary>
    /// Attaches the Edge0 latent-MoE glue kernels and allocates their per-forward scratch for up to <paramref name="rows"/> rows (D output for all
    /// groups, eight per-group z and U-output slabs). Call once, after the final planned row count is known; the scratch is never grown while recording.
    /// </summary>
    internal void Q4AttachEdge0(Edge0LatentKernels kernels, int groups, int latent, int hidden, int rows)
    {
        if (_kernels.QuantizeQ8_1RowsActivations is null || _kernels.MoeMmvqIq(QuantizationType.Q3_K) is null)
            throw new NotSupportedException("The Edge0 Vulkan path needs integer-dot-product support (the Q3_K / IQ4_NL expert banks are read through the dp4a MMVQ kernels).");
        FreeEdge0Scratch();
        _edge0Kernels = kernels;
        _edge0Rows = rows;
        _edge0Xl = _device.AllocateDeviceLocal((long)rows * groups * latent * sizeof(float));
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
        RecordMatmul(cmd, e0.D, e0.LatentDenseType, _state.NormOutput, _edge0Xl!, outputDim: G * L, inputDim: H, seqLen: T);
        KernelSupport.ComputeToComputeBarrier(cmd);
        k.RecordGather(cmd, _edge0Xl!, _state.MoeTopkIndices, e0.Labels, e0.InverseScale, _state.MoeExpandedInput, L, G, topK, rows, E);
        KernelSupport.ComputeToComputeBarrier(cmd);
        MoeStage("edge0_d_gather");

        // 3. experts at width L (gate/up K = L, down M = L); the result rows land in MoeDownRows [rows, L]
        bool grouped = T >= GroupedMinTokens && GroupedKernelsFit(moeW, L, interm)
            && _kernels.MoeExpertOffsets is not null && _kernels.MoeExpandGroupByExpert is not null && _kernels.MoeUngroupScatter is not null && (L % 256) == 0;
        if (grouped)
            RecordGroupedExperts(cmd, moeW, T, L, interm, E, rows, fusedGlue: false, topK);
        else
        {
            _kernels.QuantizeQ8_1RowsActivations!.Record(cmd, _state.MoeExpandedInput, _state.MoeExpandedInputXq, _state.MoeExpandedInputXds, n: rows, k: L);
            KernelSupport.ComputeToComputeBarrier(cmd);
            RecordGateUpBank(cmd, moeW.W1QuantType, moeW.W1Bank, _state.MoeGateInter, interm, L, rows, E);
            RecordGateUpBank(cmd, moeW.W3QuantType, moeW.W3Bank, _state.MoeUpInter, interm, L, rows, E);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.SwiGlu.Record(cmd, _state.MoeGateInter, _state.MoeUpInter, _state.MoeSiluInter, n: rows * interm);
            KernelSupport.ComputeToComputeBarrier(cmd);
            _kernels.QuantizeQ8_1RowsActivations.Record(cmd, _state.MoeSiluInter, _state.MoeSiluInterXq, _state.MoeSiluInterXds, n: rows, k: interm);
            KernelSupport.ComputeToComputeBarrier(cmd);
            RecordDownMmvq(cmd, moeW.W2QuantType, moeW, L, interm, rows, E);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        MoeStage("edge0_experts");

        // 4. per-group weighted sum of the expert outputs, U per group, sum of the group outputs -> NormOutput
        for (int g = 0; g < G; g++)
            k.RecordGroupCombine(cmd, _state.MoeDownRows, _state.MoeTopkIndices, _state.MoeTopkWeights, e0.Labels, _edge0Z![g], L, topK, g, T, E);
        KernelSupport.ComputeToComputeBarrier(cmd);
        for (int g = 0; g < G; g++)
        {
            RecordMatmul(cmd, e0.U[g], e0.LatentDenseType, _edge0Z![g], _edge0Uo![g], outputDim: H, inputDim: L, seqLen: T);
            KernelSupport.ComputeToComputeBarrier(cmd);   // a single-row Q8_0 / Q4_0 GEMV quantizes its input into the shared Q8_1 scratch
        }
        k.RecordSum(cmd, _edge0Uo!, _state.NormOutput, T * H);
        KernelSupport.ComputeToComputeBarrier(cmd);
        MoeStage("edge0_u");

        // 5. sigmoid-gated shared expert, added into NormOutput
        if (moeW.HasSharedExpert) RecordSharedExpert(cmd, moeW, postAttnNormWeight: null, T, H, Config.NormEpsilon);
        Edge0LayersRecorded++;
    }
}
