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
    //
    // Hidden / AddScratch form a 2-slot pair. The "current hidden" sits in
    // one slot; the "scratch that the residual-add writes into" sits in the
    // other. After each residual add we swap the slot indices — the buffer
    // that was just written becomes the new HiddenState, and the previous
    // HiddenState becomes the new AddScratch. This replaces two vkCmdCopyBuffer
    // dispatches per residual cycle (HiddenState→Residual snapshot copy and
    // AddScratch→HiddenState writeback copy) with pure pointer relabelling —
    // Residual is simply an alias for the current HiddenState, because the
    // snapshot read and the norm read come from the same buffer.
    private readonly VulkanDevice.Buffer[] _hiddenSlots = new VulkanDevice.Buffer[2];
    private int _hiddenIdx; // index into _hiddenSlots of the current HiddenState

    /// <summary>Currently active hidden-state buffer (rotates on each residual add).</summary>
    public VulkanDevice.Buffer HiddenState => _hiddenSlots[_hiddenIdx];

    /// <summary>
    /// Residual snapshot for the current pre-norm → post-add cycle. By design
    /// it aliases <see cref="HiddenState"/>: the snapshot is always the
    /// hidden state as-of the start of the cycle, and rotating the hidden
    /// slot only happens after the residual add finishes reading from here.
    /// </summary>
    public VulkanDevice.Buffer Residual => _hiddenSlots[_hiddenIdx];

    /// <summary>Alternate hidden slot — residual-add writes here, then <see cref="RotateHiddenSlot"/> promotes it.</summary>
    public VulkanDevice.Buffer AddScratch => _hiddenSlots[_hiddenIdx ^ 1];

    /// <summary>
    /// Flips <see cref="HiddenState"/> and <see cref="AddScratch"/>. Call after
    /// the residual add has been dispatched and the post-add barrier is in
    /// place — the old <see cref="AddScratch"/> now carries the new hidden
    /// state and the old <see cref="HiddenState"/> becomes free scratch.
    /// </summary>
    public void RotateHiddenSlot() => _hiddenIdx ^= 1;

    /// <summary>
    /// Resets the hidden-slot rotation so the next forward starts with the
    /// canonical slot 0 as <see cref="HiddenState"/>. Called at the start of
    /// every forward — without this, the forward pass would alternate the
    /// physical buffer it writes the initial embedding into, defeating any
    /// per-buffer descriptor-set caching that depends on the buffer handle
    /// staying stable across forwards.
    /// </summary>
    public void ResetHiddenSlot() => _hiddenIdx = 0;

    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;
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
    /// <returns><c>true</c> when scratch was re-allocated (so every cached VkBuffer handle
    /// is now stale); <c>false</c> when existing capacity was already sufficient.</returns>
    public bool EnsureCapacity(int seqLen)
    {
        if (seqLen <= _capacitySeqLen) return false;

        ReleaseLayerScratch();
        AllocateForCapacity(seqLen);
        return true;
    }

    private void AllocateForCapacity(int seqLen)
    {
        long hiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);
        long qBytes = (long)seqLen * _numHeads * _headDim * sizeof(float);
        long kvBytes = (long)seqLen * _numKvHeads * _headDim * sizeof(float);
        long ffnBytes = (long)seqLen * _intermediateSize * sizeof(float);

        // Buffers never read or written from the host go device-local — the
        // driver can pick its tiled / swizzled native layout instead of the
        // host-coherent linear layout that Allocate() returns. On UMA (Strix
        // Halo iGPU) this lets the compute path use the GPU's preferred
        // memory access pattern; on dGPU it keeps the bytes off the
        // PCIe-host-coherent path.
        //
        // Bias-add receiver buffers (Q, K, V, NormOutput, FfnGate, FfnUp)
        // stay host-visible because AddBiasRows host-maps them when a bias
        // tensor is present. SmolLM-135M has no biases so the host-map
        // never fires, but Phi-3 / Qwen3 / DeepSeek-V2 do — moving them
        // requires a bias_add_f32 compute kernel (issue #7).
        _hiddenSlots[0] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenSlots[1] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenIdx = 0;
        NormOutput = _device.Allocate(hiddenBytes);

        Q = _device.Allocate(qBytes);
        K = _device.Allocate(kvBytes);
        V = _device.Allocate(kvBytes);
        AttnOutput = _device.AllocateDeviceLocal(qBytes);

        FfnGate = _device.Allocate(ffnBytes);
        FfnUp = _device.Allocate(ffnBytes);
        SiluOutput = _device.AllocateDeviceLocal(ffnBytes);

        // Resize positions buffer — host writes positions per forward.
        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;

        // hiddenBytes × 3: two hidden slots (HiddenState/AddScratch rotate) + NormOutput.
        AllocatedBytes = hiddenBytes * 3 + qBytes * 2 + kvBytes * 2 + ffnBytes * 3
                       + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int);
    }

    private void ReleaseLayerScratch()
    {
        _hiddenSlots[0]?.Dispose();
        _hiddenSlots[1]?.Dispose();
        _hiddenSlots[0] = null!;
        _hiddenSlots[1] = null!;
        NormOutput?.Dispose();
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
