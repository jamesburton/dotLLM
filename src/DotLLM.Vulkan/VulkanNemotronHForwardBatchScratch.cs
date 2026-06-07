namespace DotLLM.Vulkan;

/// <summary>
/// Per-batch scratch buffers used by <c>VulkanNemotronHTransformerModel.ForwardBatch</c>
/// to fan out the final RMSNorm + lm_head across <c>N_simple</c> in-flight sequences as
/// a single batched matmul.
/// </summary>
/// <remarks>
/// <para>
/// Phase 5f mirror — NemotronH variant. Unlike the dense
/// <see cref="VulkanForwardBatchScratch"/>, this scratch class does NOT include per-seq
/// Q / attention-output staging buffers: the NemotronH hybrid mixes Mamba2-style SSM
/// layers (per-token recurrent — state thread is per-seq) with GQA full-attention
/// layers, so the per-layer loop cannot be batched at <c>seqLen = Σ N_i</c>. The
/// batched path therefore runs the full per-seq Forward layer loop and only fuses the
/// terminal RMSNorm + lm_head into a single dispatch over a stacked
/// <c>[N_simple, hidden]</c> buffer — the same stacked-buffer pattern as the Phase 5a
/// CPU lm_head fusion.
/// </para>
/// <para>
/// Grow-only — sized for the largest sequence count ever seen so far. Allocation
/// lifetime matches the parent <c>VulkanNemotronHTransformerModel</c> via
/// <see cref="IDisposable"/>. Zero VRAM cost when only <c>Forward</c> is used (lazy).
/// </para>
/// </remarks>
internal sealed class VulkanNemotronHForwardBatchScratch : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _vocabSize;

    private int _batchCapacitySeqs;
    private bool _disposed;

    /// <summary>FP32 stacked last-token hidden rows, one per simple sequence: shape <c>[maxBatchSeqs, hidden]</c>.</summary>
    public VulkanDevice.Buffer? LastRowHidden { get; private set; }

    /// <summary>FP32 batched lm_head output: shape <c>[maxBatchSeqs, vocab]</c>. Host-visible for the post-submit download.</summary>
    public VulkanDevice.Buffer? BatchedLogits { get; private set; }

    public long AllocatedBytes { get; private set; }

    public VulkanNemotronHForwardBatchScratch(VulkanDevice device, int hiddenSize, int vocabSize)
    {
        _device = device;
        _hiddenSize = hiddenSize;
        _vocabSize = vocabSize;
    }

    /// <summary>
    /// Ensures the per-batch buffers are sized for at least <paramref name="batchSeqs"/>
    /// simple sequences. Grows monotonically; never shrinks.
    /// </summary>
    /// <returns><c>true</c> when any buffer was re-allocated (so descriptor caches keyed
    /// on these handles are stale and the host must invalidate them).</returns>
    public bool EnsureCapacity(int batchSeqs)
    {
        if (batchSeqs <= 0) throw new ArgumentOutOfRangeException(nameof(batchSeqs));
        if (LastRowHidden is not null && batchSeqs <= _batchCapacitySeqs) return false;

        LastRowHidden?.Dispose();
        BatchedLogits?.Dispose();

        long lastRowBytes = (long)batchSeqs * _hiddenSize * sizeof(float);
        long batchedLogitsBytes = (long)batchSeqs * _vocabSize * sizeof(float);
        // LastRowHidden is touched only by the in-submit copy + compute matmul — device-local.
        LastRowHidden = _device.AllocateDeviceLocal(lastRowBytes);
        // BatchedLogits is host-readable for the post-submit download path.
        BatchedLogits = _device.Allocate(batchedLogitsBytes);
        _batchCapacitySeqs = batchSeqs;
        AllocatedBytes = lastRowBytes + batchedLogitsBytes;
        return true;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        LastRowHidden?.Dispose(); LastRowHidden = null;
        BatchedLogits?.Dispose(); BatchedLogits = null;
    }
}
