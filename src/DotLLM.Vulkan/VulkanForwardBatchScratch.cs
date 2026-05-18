namespace DotLLM.Vulkan;

/// <summary>
/// Per-batch scratch buffers used by <c>VulkanTransformerModel.ForwardBatch</c>
/// to slice the batched [Σ N_i, ...] activation buffers into per-seq attention
/// inputs / outputs. Grow-only — sized for the largest seqLen ever passed to
/// any single sequence in the batched path and for the largest sequence count
/// for the lm_head fan-out.
/// </summary>
/// <remarks>
/// <para>
/// Path-1 fusion in <c>ForwardBatch</c>:
/// <list type="bullet">
///   <item>Intra-block matmuls (RMSNorm, Q/K/V/O, gate/up/down, lm_head) are dispatched at
///         <c>seqLen = Σ N_i</c> against the existing <see cref="VulkanForwardState"/> scratch
///         (HiddenState, NormOutput, Q, K, V, FfnGate, FfnUp, SiluOutput, AttnOutput).</item>
///   <item>Attention is per-seq — each sequence has its own <c>VulkanKvCache</c> and its
///         own positionOffset. The existing <c>AttentionF32Kernel.Record</c> reads its Q
///         input from offset 0 of the bound Q buffer, so we copy that seq's Q slice from
///         the batched Q buffer into <see cref="PerSeqQ"/>, run attention into
///         <see cref="PerSeqAttn"/>, then copy the result back into the batched
///         <see cref="VulkanForwardState.AttnOutput"/> at the matching offset.</item>
///   <item>The lm_head runs on the LAST hidden row per simple sequence (Vulkan convention —
///         see <c>VulkanTransformerModel.Forward</c>'s lm_head block). Each seq's last
///         hidden row is copied into <see cref="LastRowHidden"/> at slot i, then a single
///         batched RMSNorm + matmul produces <see cref="BatchedLogits"/> of shape
///         <c>[N_simple, vocab]</c>, which is split back into per-seq host tensors.</item>
/// </list>
/// </para>
/// <para>
/// All buffers are device-local — host never touches them between the per-forward submit
/// and the post-submit download of <see cref="BatchedLogits"/>. Allocation lifetime matches
/// the parent <c>VulkanTransformerModel</c> via <see cref="IDisposable"/>.
/// </para>
/// </remarks>
internal sealed class VulkanForwardBatchScratch : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _qDim;            // numHeads * headDim
    private readonly int _vocabSize;

    private int _perSeqCapacityTokens;     // largest single-seq seqLen seen so far
    private int _batchCapacitySeqs;        // largest batch-seq count seen so far
    private bool _disposed;

    /// <summary>FP32 staging for one sequence's Q rows: shape <c>[maxSingleSeq, numHeads * headDim]</c>.</summary>
    public VulkanDevice.Buffer? PerSeqQ { get; private set; }

    /// <summary>FP32 staging for one sequence's attention output: shape <c>[maxSingleSeq, numHeads * headDim]</c>.</summary>
    public VulkanDevice.Buffer? PerSeqAttn { get; private set; }

    /// <summary>FP32 stacked last-token hidden rows, one per simple sequence: shape <c>[maxBatchSeqs, hidden]</c>.</summary>
    public VulkanDevice.Buffer? LastRowHidden { get; private set; }

    /// <summary>FP32 batched lm_head output: shape <c>[maxBatchSeqs, vocab]</c>.</summary>
    public VulkanDevice.Buffer? BatchedLogits { get; private set; }

    public long AllocatedBytes { get; private set; }

    public VulkanForwardBatchScratch(VulkanDevice device, int hiddenSize, int qDim, int vocabSize)
    {
        _device = device;
        _hiddenSize = hiddenSize;
        _qDim = qDim;
        _vocabSize = vocabSize;
    }

    /// <summary>
    /// Ensures both per-seq and batched-output buffers are large enough.
    /// </summary>
    /// <param name="maxSingleSeqTokens">Largest single sequence token count in the batch (= max N_i).</param>
    /// <param name="batchSeqs">Number of sequences contributing to the batched lm_head (= simple count).</param>
    /// <returns><c>true</c> when any buffer was re-allocated (so descriptor caches keyed on these handles are stale).</returns>
    public bool EnsureCapacity(int maxSingleSeqTokens, int batchSeqs)
    {
        if (maxSingleSeqTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxSingleSeqTokens));
        if (batchSeqs <= 0) throw new ArgumentOutOfRangeException(nameof(batchSeqs));

        bool resized = false;

        if (PerSeqQ is null || PerSeqAttn is null || maxSingleSeqTokens > _perSeqCapacityTokens)
        {
            PerSeqQ?.Dispose();
            PerSeqAttn?.Dispose();
            long perSeqBytes = (long)maxSingleSeqTokens * _qDim * sizeof(float);
            PerSeqQ = _device.AllocateDeviceLocal(perSeqBytes);
            PerSeqAttn = _device.AllocateDeviceLocal(perSeqBytes);
            _perSeqCapacityTokens = maxSingleSeqTokens;
            resized = true;
        }

        if (LastRowHidden is null || BatchedLogits is null || batchSeqs > _batchCapacitySeqs)
        {
            LastRowHidden?.Dispose();
            BatchedLogits?.Dispose();
            long lastRowBytes = (long)batchSeqs * _hiddenSize * sizeof(float);
            long batchedLogitsBytes = (long)batchSeqs * _vocabSize * sizeof(float);
            LastRowHidden = _device.AllocateDeviceLocal(lastRowBytes);
            BatchedLogits = _device.Allocate(batchedLogitsBytes); // host-readable for download
            _batchCapacitySeqs = batchSeqs;
            resized = true;
        }

        if (resized)
        {
            long perSeqBytes = (long)_perSeqCapacityTokens * _qDim * sizeof(float);
            long lastRowBytes = (long)_batchCapacitySeqs * _hiddenSize * sizeof(float);
            long batchedLogitsBytes = (long)_batchCapacitySeqs * _vocabSize * sizeof(float);
            AllocatedBytes = perSeqBytes * 2 + lastRowBytes + batchedLogitsBytes;
        }

        return resized;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        PerSeqQ?.Dispose(); PerSeqQ = null;
        PerSeqAttn?.Dispose(); PerSeqAttn = null;
        LastRowHidden?.Dispose(); LastRowHidden = null;
        BatchedLogits?.Dispose(); BatchedLogits = null;
    }
}
