using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-forward scratch buffers for the Qwen3HybridDense Vulkan path. Mirrors
/// <see cref="VulkanQwen3MoeHybridForwardState"/> for the shared hidden-state and
/// token-mixing slots (only the GDN or the full-attention set is touched per
/// layer), with the routed-MoE scratch replaced by the three dense SwiGLU FFN
/// buffers. Grows monotonically with <see cref="EnsureCapacity"/>.
/// </summary>
internal sealed class VulkanQwen3HybridDenseForwardState : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _vocabSize;
    private readonly int _intermediateSize;
    private readonly int _qElems;          // numHeads * headDim
    private readonly int _kvElems;         // numKvHeads * headDim
    private readonly int _convDim;         // (2*NKHead + NVHead) * DState
    private readonly int _dConv;
    private readonly int _gdnVDim;         // NVHead * DState
    private readonly int _gdnKDim;         // NKHead * DState
    private readonly int _nVHead;

    private int _capacitySeqLen;

    // ── Hidden / residual / scratch ──────────────────────────────────────────
    public VulkanDevice.Buffer HiddenState { get; private set; } = null!;
    public VulkanDevice.Buffer Residual { get; private set; } = null!;
    public VulkanDevice.Buffer AddScratch { get; private set; } = null!;
    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;

    // ── Full attention ───────────────────────────────────────────────────────
    public VulkanDevice.Buffer QGateScratch { get; private set; } = null!;  // [seqLen, 2*qElems]
    public VulkanDevice.Buffer Q { get; private set; } = null!;
    public VulkanDevice.Buffer GateScratch { get; private set; } = null!;
    public VulkanDevice.Buffer K { get; private set; } = null!;
    public VulkanDevice.Buffer V { get; private set; } = null!;
    public VulkanDevice.Buffer AttnOutput { get; private set; } = null!;

    // ── GDN ──────────────────────────────────────────────────────────────────
    public VulkanDevice.Buffer GdnConvInput { get; private set; } = null!;   // [(dConv-1+seqLen), convDim]
    public VulkanDevice.Buffer GdnQkvBuf { get; private set; } = null!;      // [seqLen, convDim] (also conv output)
    public VulkanDevice.Buffer GdnZBuf { get; private set; } = null!;        // [seqLen, vDim]
    public VulkanDevice.Buffer GdnAlphaBuf { get; private set; } = null!;    // [seqLen, nVHead]
    public VulkanDevice.Buffer GdnBetaBuf { get; private set; } = null!;     // [seqLen, nVHead]
    public VulkanDevice.Buffer GdnQBuf { get; private set; } = null!;        // [seqLen, kDim]
    public VulkanDevice.Buffer GdnKBuf { get; private set; } = null!;        // [seqLen, kDim]
    public VulkanDevice.Buffer GdnVBuf { get; private set; } = null!;        // [seqLen, vDim]
    public VulkanDevice.Buffer GdnOut { get; private set; } = null!;         // [seqLen, vDim]

    // ── Dense SwiGLU FFN ─────────────────────────────────────────────────────
    public VulkanDevice.Buffer FfnGate { get; private set; } = null!;        // [seqLen, intermediate]
    public VulkanDevice.Buffer FfnUp { get; private set; } = null!;          // [seqLen, intermediate]
    public VulkanDevice.Buffer FfnSilu { get; private set; } = null!;        // [seqLen, intermediate]

    // ── Logits + positions ───────────────────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; }
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    private bool _disposed;
    public long AllocatedBytes { get; private set; }

    public VulkanQwen3HybridDenseForwardState(
        VulkanDevice device, ModelConfig config, GatedDeltaNetConfig gdn, int initialSeqLen)
    {
        _device = device;
        _hiddenSize = config.HiddenSize;
        _vocabSize = config.VocabSize;
        _intermediateSize = config.IntermediateSize;
        _qElems = config.NumAttentionHeads * config.HeadDim;
        _kvElems = config.NumKvHeads * config.HeadDim;
        _convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState;
        _dConv = gdn.DConv;
        _gdnVDim = gdn.NVHead * gdn.DState;
        _gdnKDim = gdn.NKHead * gdn.DState;
        _nVHead = gdn.NVHead;

        Logits = device.Allocate((long)_vocabSize * sizeof(float));
        PositionsBuffer = device.Allocate(Math.Max(1, initialSeqLen) * sizeof(int));

        AllocateForCapacity(Math.Max(1, initialSeqLen));
    }

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
        long qBytes = (long)seqLen * _qElems * sizeof(float);
        long kvBytes = (long)seqLen * _kvElems * sizeof(float);
        long qgBytes = (long)seqLen * 2 * _qElems * sizeof(float);

        long convInputBytes = (long)(_dConv - 1 + seqLen) * _convDim * sizeof(float);
        long convBytes = (long)seqLen * _convDim * sizeof(float);
        long vDimBytes = (long)seqLen * _gdnVDim * sizeof(float);
        long kDimBytes = (long)seqLen * _gdnKDim * sizeof(float);
        long alphaBytes = (long)seqLen * _nVHead * sizeof(float);

        long ffnBytes = (long)seqLen * _intermediateSize * sizeof(float);

        HiddenState = _device.AllocateDeviceLocal(hiddenBytes);
        Residual = _device.AllocateDeviceLocal(hiddenBytes);
        AddScratch = _device.AllocateDeviceLocal(hiddenBytes);
        NormOutput = _device.AllocateDeviceLocal(hiddenBytes);

        QGateScratch = _device.AllocateDeviceLocal(qgBytes);
        Q = _device.AllocateDeviceLocal(qBytes);
        GateScratch = _device.AllocateDeviceLocal(qBytes);
        K = _device.AllocateDeviceLocal(kvBytes);
        V = _device.AllocateDeviceLocal(kvBytes);
        AttnOutput = _device.AllocateDeviceLocal(qBytes);

        GdnConvInput = _device.AllocateDeviceLocal(convInputBytes);
        GdnQkvBuf = _device.AllocateDeviceLocal(convBytes);
        GdnZBuf = _device.AllocateDeviceLocal(vDimBytes);
        GdnAlphaBuf = _device.AllocateDeviceLocal(alphaBytes);
        GdnBetaBuf = _device.AllocateDeviceLocal(alphaBytes);
        GdnQBuf = _device.AllocateDeviceLocal(kDimBytes);
        GdnKBuf = _device.AllocateDeviceLocal(kDimBytes);
        GdnVBuf = _device.AllocateDeviceLocal(vDimBytes);
        GdnOut = _device.AllocateDeviceLocal(vDimBytes);

        FfnGate = _device.AllocateDeviceLocal(ffnBytes);
        FfnUp = _device.AllocateDeviceLocal(ffnBytes);
        FfnSilu = _device.AllocateDeviceLocal(ffnBytes);

        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;
        AllocatedBytes = hiddenBytes * 4 + qBytes * 3 + kvBytes * 2 + qgBytes
            + convInputBytes + convBytes + vDimBytes * 3 + kDimBytes * 2 + alphaBytes * 2
            + ffnBytes * 3
            + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int);
    }

    private void ReleaseLayerScratch()
    {
        HiddenState?.Dispose(); Residual?.Dispose(); AddScratch?.Dispose(); NormOutput?.Dispose();
        QGateScratch?.Dispose(); Q?.Dispose(); GateScratch?.Dispose();
        K?.Dispose(); V?.Dispose(); AttnOutput?.Dispose();
        GdnConvInput?.Dispose(); GdnQkvBuf?.Dispose(); GdnZBuf?.Dispose();
        GdnAlphaBuf?.Dispose(); GdnBetaBuf?.Dispose();
        GdnQBuf?.Dispose(); GdnKBuf?.Dispose(); GdnVBuf?.Dispose(); GdnOut?.Dispose();
        FfnGate?.Dispose(); FfnUp?.Dispose(); FfnSilu?.Dispose();
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
