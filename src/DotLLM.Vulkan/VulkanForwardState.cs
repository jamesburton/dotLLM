namespace DotLLM.Vulkan;

/// <summary>
/// Owns all per-forward-pass scratch buffers on the Vulkan device.
/// Sized for the maximum <c>seqLen</c> the caller has used so far; grows monotonically.
/// Mirrors <c>DotLLM.Cuda.CudaForwardState</c> but for FP32 storage on a Vulkan device.
/// </summary>
/// <remarks>
/// All buffers are host-visible host-coherent. The Vulkan scaffold does not
/// have a staging path yet; <see cref="VulkanDevice.Allocate"/> returns memory
/// that is mappable from both host and GPU, which is slower than device-local
/// memory for real kernels but keeps weight upload / result download trivial.
/// Optimising this is explicitly out of scope for the end-to-end wave.
/// </remarks>
internal sealed class VulkanForwardState : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;
    private int _capacitySeqLen;

    // ── Transformer layer scratch (all FP32) ──────────────────────────
    public VulkanDevice.Buffer HiddenState { get; private set; } = null!;
    public VulkanDevice.Buffer Residual { get; private set; } = null!;
    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;
    public VulkanDevice.Buffer AddScratch { get; private set; } = null!;
    public VulkanDevice.Buffer Q { get; private set; } = null!;
    public VulkanDevice.Buffer K { get; private set; } = null!;
    public VulkanDevice.Buffer V { get; private set; } = null!;
    public VulkanDevice.Buffer AttnOutput { get; private set; } = null!;
    public VulkanDevice.Buffer FfnGate { get; private set; } = null!;
    public VulkanDevice.Buffer FfnUp { get; private set; } = null!;
    public VulkanDevice.Buffer SiluOutput { get; private set; } = null!;

    // ── Logits (last token only) ──────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; private set; }

    // ── Host → device transfer scratch (tokens + positions) ──────────
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    public VulkanForwardState(
        VulkanDevice device,
        int hiddenSize, int numHeads, int numKvHeads, int headDim,
        int intermediateSize, int vocabSize, int initialSeqLen)
    {
        _device = device;
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;

        // LM-head logits are always one token (last). Positions buffer sized for some reasonable
        // default; grows with EnsureCapacity.
        Logits = device.Allocate((long)vocabSize * sizeof(float));
        PositionsBuffer = device.Allocate(Math.Max(1, initialSeqLen) * sizeof(int));

        AllocateForCapacity(Math.Max(1, initialSeqLen));
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough to host <paramref name="seqLen"/> tokens.
    /// Grows monotonically; never shrinks.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _capacitySeqLen) return;

        ReleaseLayerScratch();
        AllocateForCapacity(seqLen);
    }

    private void AllocateForCapacity(int seqLen)
    {
        long hiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);
        long qBytes = (long)seqLen * _numHeads * _headDim * sizeof(float);
        long kvBytes = (long)seqLen * _numKvHeads * _headDim * sizeof(float);
        long ffnBytes = (long)seqLen * _intermediateSize * sizeof(float);

        HiddenState = _device.Allocate(hiddenBytes);
        Residual = _device.Allocate(hiddenBytes);
        NormOutput = _device.Allocate(hiddenBytes);
        AddScratch = _device.Allocate(hiddenBytes);

        Q = _device.Allocate(qBytes);
        K = _device.Allocate(kvBytes);
        V = _device.Allocate(kvBytes);
        AttnOutput = _device.Allocate(qBytes);

        FfnGate = _device.Allocate(ffnBytes);
        FfnUp = _device.Allocate(ffnBytes);
        SiluOutput = _device.Allocate(ffnBytes);

        // Resize positions buffer.
        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;

        AllocatedBytes = hiddenBytes * 4 + qBytes * 2 + kvBytes * 2 + ffnBytes * 3
                       + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int);
    }

    private void ReleaseLayerScratch()
    {
        HiddenState?.Dispose();
        Residual?.Dispose();
        NormOutput?.Dispose();
        AddScratch?.Dispose();
        Q?.Dispose();
        K?.Dispose();
        V?.Dispose();
        AttnOutput?.Dispose();
        FfnGate?.Dispose();
        FfnUp?.Dispose();
        SiluOutput?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        ReleaseLayerScratch();
        Logits?.Dispose();
        PositionsBuffer?.Dispose();
    }
}
