using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Narrow internal surface through which <see cref="VulkanQwen4ExpTransformerModel"/> drives this class's already-validated
/// building blocks (Gated-DeltaNet layer, full-GQA attention layer, routed+shared MoE layer, quant-aware matmul dispatcher) on its
/// own residual plumbing. Qwen4-Exp is a Qwen3.5-style GDN/attention + MoE hybrid whose only structural differences are the
/// 4-stream gated residual around each block (which replaces the pre-norms and residual adds) and the sigmoid GDN output gate,
/// so it composes this model instead of forking ~1500 lines of kernels.
/// </summary>
public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    internal VulkanDevice Q4Device => _device;
    internal VulkanQwen3MoeHybridForwardState Q4State => _state;
    internal VulkanQwen3MoeHybridWeights Q4Weights => _weights;
    internal VulkanQwen3MoeHybridKernels Q4Kernels => _kernels;
    internal VulkanDevice.SubmitContext Q4Submit => _submit;

    /// <summary>Grows the per-forward scratch to <paramref name="seqLen"/> rows; true when buffers were re-created (descriptor caches were dropped).</summary>
    internal bool Q4EnsureCapacity(int seqLen)
    {
        bool resized;
        try { resized = _state.EnsureCapacity(seqLen); }
        catch (Interop.VulkanException e) when (e.ErrorCode is -2 or -1)   // out of device / host memory (#880)
        {
            throw new InvalidOperationException(Qwen4ExpResidencyPlan.ScratchGrowthMessage(seqLen, e.Message), e);
        }
        if (resized) { _kernels.InvalidateAll(); _iqF16Prefill?.InvalidateDescriptorCache(); }
        return resized;
    }

    /// <summary>Drops every cached descriptor set (call after re-creating buffers the caller feeds into this model's kernels).</summary>
    internal void Q4InvalidateCaches() { _kernels.InvalidateAll(); _iqF16Prefill?.InvalidateDescriptorCache(); }

    internal void Q4UploadPositions(ReadOnlySpan<int> positions) => UploadPositions(positions);

    internal void Q4RecordEmbedding(nint cmdBuf, ReadOnlySpan<int> tokenIds) => RecordEmbeddingGather(cmdBuf, tokenIds);

    internal void Q4RecordGdn(nint cmdBuf, int layer, int seqLen, float eps, VulkanGdnStateCache gdnCache, GdnPostScanGateF32Kernel sigmoidGate,
                              Q4GdnRowSnapshots? snap = null)
        => RecordGdnLayer(cmdBuf, layer, _weights.Layers[layer].Gdn!.Value, seqLen, eps, gdnCache, sigmoidGate, snap);

    /// <summary>GDN ordinal of absolute layer <paramref name="layer"/> (-1 for attention layers).</summary>
    internal int Q4GdnOrdinal(int layer) => _gdnLayerOrdinal[layer];

    internal void Q4RecordAttention(nint cmdBuf, int layer, int seqLen, ReadOnlySpan<int> positions, VulkanNemotronHKvCache kvCache)
        => RecordFullAttnLayer(cmdBuf, layer, _weights.Layers[layer].Attention!.Value, seqLen, positions,
            Config.NumAttentionHeads, _weights.Layers[layer].Attention!.Value.NumKvHeads, Config.HeadDim, kvCache);

    /// <summary>MoE on the block input already in <c>NormOutput</c>; the caller staged a copy of it in <c>MoeSharedInput</c> (no post-attention norm exists). Result lands in <c>NormOutput</c>.</summary>
    internal void Q4RecordMoe(nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle bundle, int seqLen)
        => RecordMoeLayer(cmdBuf, bundle, postAttnNormWeight: null, seqLen, Config.HiddenSize, Config.NormEpsilon);

    internal void Q4RecordMatmul(nint cmdBuf, VulkanDevice.Buffer weights, QuantizationType qt, VulkanDevice.Buffer input,
        VulkanDevice.Buffer output, int outputDim, int inputDim, int seqLen)
        => RecordMatmul(cmdBuf, weights, qt, input, output, outputDim, inputDim, seqLen);

    internal static void Q4Copy(nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, ulong srcOffset, ulong dstOffset, ulong size)
        => RecordCopyBufferRange(cmdBuf, src, dst, srcOffset, dstOffset, size);

    /// <summary>Per-sequence GDN state sized for this model's GDN layers.</summary>
    internal VulkanGdnStateCache Q4CreateGdnState() => CreateGdnStateCache();

    /// <summary>Sparse per-attention-layer KV cache sized for <paramref name="maxSeqLen"/> positions.</summary>
    internal VulkanNemotronHKvCache Q4CreateKvCache(int maxSeqLen) => CreateKvCache(maxSeqLen);
}

/// <summary>
/// Per-row recurrent-state snapshots a speculative verify forward asks the Gated-DeltaNet layers to record (#820): the state after each
/// of the first <see cref="Rows"/> rows (scan twin kernel, owned by the model, bit-identical to the shipping scan) and the conv window after each of them.
/// Buffers are indexed by GDN ordinal: <c>Gdn[l]</c> holds <c>Rows x stateElements</c> floats, <c>Conv[l]</c> <c>Rows x convStateElements</c>.
/// </summary>
internal sealed class Q4GdnRowSnapshots : IDisposable
{
    public required GdnScanMultiTokenSnapshotF32Kernel Kernel { get; init; }
    public required VulkanDevice.Buffer[] Gdn { get; init; }
    public required VulkanDevice.Buffer[] Conv { get; init; }
    public int Rows { get; set; }

    public void Dispose()
    {
        foreach (var b in Gdn) b?.Dispose();
        foreach (var b in Conv) b?.Dispose();
    }
}
