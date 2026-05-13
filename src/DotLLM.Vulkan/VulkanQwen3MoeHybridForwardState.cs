using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-forward scratch buffers for the Qwen3MoeHybrid Vulkan path. Mirrors
/// <see cref="DotLLM.Models.Architectures.Qwen3MoeHybridForwardState"/> in
/// layout — one shared hidden-state slot, the token-mixing scratch for both
/// GDN and full-attention paths (only the relevant set is touched per layer),
/// and the routed-MoE per-call scratch. Grows monotonically with
/// <see cref="EnsureCapacity"/>.
/// </summary>
internal sealed class VulkanQwen3MoeHybridForwardState : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _vocabSize;
    private readonly int _qElems;          // numHeads * headDim
    private readonly int _kvElems;         // numKvHeads * headDim
    private readonly int _convDim;         // (2*NKHead + NVHead) * DState
    private readonly int _dConv;
    private readonly int _gdnVDim;         // NVHead * DState
    private readonly int _gdnKDim;         // NKHead * DState
    private readonly int _nVHead;

    // MoE shapes
    private readonly int _moeIntermediate;
    private readonly int _moeNumExperts;
    private readonly int _moeTopK;
    private readonly int _moeSharedIntermediate;

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

    // Single-token slice buffers used by the per-token GDN scan dispatch.
    // The scan shader takes whole-buffer descriptor bindings; copying the
    // current token's row into these compact buffers per iteration sidesteps
    // adding a per-dispatch offset push constant to the shader. Sizes are
    // fixed (don't depend on seqLen) and survive across forwards.
    public VulkanDevice.Buffer GdnTokenQ { get; private set; } = null!;      // [kDim]
    public VulkanDevice.Buffer GdnTokenK { get; private set; } = null!;      // [kDim]
    public VulkanDevice.Buffer GdnTokenV { get; private set; } = null!;      // [vDim]
    public VulkanDevice.Buffer GdnTokenG { get; private set; } = null!;      // [nVHead]
    public VulkanDevice.Buffer GdnTokenBeta { get; private set; } = null!;   // [nVHead]
    public VulkanDevice.Buffer GdnTokenOut { get; private set; } = null!;    // [vDim]

    // ── MoE per-call scratch ─────────────────────────────────────────────────
    public VulkanDevice.Buffer MoeRouterLogits { get; private set; } = null!;
    public VulkanDevice.Buffer MoeTopkIndices { get; private set; } = null!;
    public VulkanDevice.Buffer MoeTopkWeights { get; private set; } = null!;
    public VulkanDevice.Buffer MoeExpandedInput { get; private set; } = null!;
    public VulkanDevice.Buffer MoeGateInter { get; private set; } = null!;
    public VulkanDevice.Buffer MoeUpInter { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSiluInter { get; private set; } = null!;
    public VulkanDevice.Buffer MoeDownRows { get; private set; } = null!;

    public VulkanDevice.Buffer MoeSharedInput { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSharedGate { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSharedUp { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSharedSilu { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSharedSumA { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSharedSumB { get; private set; } = null!;
    public VulkanDevice.Buffer MoeSharedGateLogits { get; private set; } = null!;

    // ── Logits + positions ───────────────────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; }
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    private bool _disposed;
    public long AllocatedBytes { get; private set; }

    public VulkanQwen3MoeHybridForwardState(
        VulkanDevice device, ModelConfig config, GatedDeltaNetConfig gdn, int initialSeqLen)
    {
        _device = device;
        _hiddenSize = config.HiddenSize;
        _vocabSize = config.VocabSize;
        _qElems = config.NumAttentionHeads * config.HeadDim;
        _kvElems = config.NumKvHeads * config.HeadDim;
        _convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState;
        _dConv = gdn.DConv;
        _gdnVDim = gdn.NVHead * gdn.DState;
        _gdnKDim = gdn.NKHead * gdn.DState;
        _nVHead = gdn.NVHead;

        var moe = config.Moe!;
        _moeIntermediate = moe.MoeIntermediateSize;
        _moeNumExperts = moe.NumExperts;
        _moeTopK = moe.NumExpertsPerTok;
        _moeSharedIntermediate = moe.SharedExpertIntermediateSize ?? 0;

        Logits = device.Allocate((long)_vocabSize * sizeof(float));
        PositionsBuffer = device.Allocate(Math.Max(1, initialSeqLen) * sizeof(int));

        // Per-token GDN scan slice buffers are sized once and reused.
        GdnTokenQ = device.AllocateDeviceLocal((long)_gdnKDim * sizeof(float));
        GdnTokenK = device.AllocateDeviceLocal((long)_gdnKDim * sizeof(float));
        GdnTokenV = device.AllocateDeviceLocal((long)_gdnVDim * sizeof(float));
        GdnTokenG = device.AllocateDeviceLocal((long)_nVHead * sizeof(float));
        GdnTokenBeta = device.AllocateDeviceLocal((long)_nVHead * sizeof(float));
        GdnTokenOut = device.AllocateDeviceLocal((long)_gdnVDim * sizeof(float));

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

        long convInputBytes = (long)(_dConv - 1 + seqLen) * _convDim * sizeof(float);
        long convBytes = (long)seqLen * _convDim * sizeof(float);
        long vDimBytes = (long)seqLen * _gdnVDim * sizeof(float);
        long kDimBytes = (long)seqLen * _gdnKDim * sizeof(float);
        long alphaBytes = (long)seqLen * _nVHead * sizeof(float);

        long routerBytes = (long)seqLen * _moeNumExperts * sizeof(float);
        long topkIdxBytes = (long)seqLen * _moeTopK * sizeof(int);
        long topkWBytes = (long)seqLen * _moeTopK * sizeof(float);
        long expandBytes = (long)seqLen * _moeTopK * _hiddenSize * sizeof(float);
        long interBytes = (long)seqLen * _moeTopK * _moeIntermediate * sizeof(float);
        long downRowsBytes = expandBytes;

        long sharedInterBytes = _moeSharedIntermediate > 0
            ? (long)seqLen * _moeSharedIntermediate * sizeof(float)
            : 1L * sizeof(float);
        long sharedGateLogitBytes = (long)seqLen * sizeof(float);

        HiddenState = _device.AllocateDeviceLocal(hiddenBytes);
        Residual = _device.AllocateDeviceLocal(hiddenBytes);
        AddScratch = _device.AllocateDeviceLocal(hiddenBytes);
        NormOutput = _device.AllocateDeviceLocal(hiddenBytes);

        QGateScratch = _device.AllocateDeviceLocal((long)seqLen * 2 * _qElems * sizeof(float));
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

        MoeRouterLogits = _device.AllocateDeviceLocal(routerBytes);
        MoeTopkIndices = _device.AllocateDeviceLocal(topkIdxBytes);
        MoeTopkWeights = _device.AllocateDeviceLocal(topkWBytes);
        MoeExpandedInput = _device.AllocateDeviceLocal(expandBytes);
        MoeGateInter = _device.AllocateDeviceLocal(interBytes);
        MoeUpInter = _device.AllocateDeviceLocal(interBytes);
        MoeSiluInter = _device.AllocateDeviceLocal(interBytes);
        MoeDownRows = _device.AllocateDeviceLocal(downRowsBytes);

        MoeSharedInput = _device.AllocateDeviceLocal(hiddenBytes);
        MoeSharedGate = _device.AllocateDeviceLocal(sharedInterBytes);
        MoeSharedUp = _device.AllocateDeviceLocal(sharedInterBytes);
        MoeSharedSilu = _device.AllocateDeviceLocal(sharedInterBytes);
        MoeSharedSumA = _device.AllocateDeviceLocal(hiddenBytes);
        MoeSharedSumB = _device.AllocateDeviceLocal(hiddenBytes);
        MoeSharedGateLogits = _device.AllocateDeviceLocal(sharedGateLogitBytes);

        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;
        AllocatedBytes = hiddenBytes * 4 + qBytes * 3 + kvBytes * 2 + 2 * (long)seqLen * _qElems * sizeof(float)
            + convInputBytes + convBytes + vDimBytes * 3 + kDimBytes * 2 + alphaBytes * 2
            + routerBytes + topkIdxBytes + topkWBytes + expandBytes + interBytes * 3 + downRowsBytes
            + hiddenBytes * 3 + sharedInterBytes * 3 + sharedGateLogitBytes
            + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int)
            + (long)(_gdnKDim * 2 + _gdnVDim * 2 + _nVHead * 2) * sizeof(float);
    }

    private void ReleaseLayerScratch()
    {
        HiddenState?.Dispose(); Residual?.Dispose(); AddScratch?.Dispose(); NormOutput?.Dispose();
        QGateScratch?.Dispose(); Q?.Dispose(); GateScratch?.Dispose();
        K?.Dispose(); V?.Dispose(); AttnOutput?.Dispose();
        GdnConvInput?.Dispose(); GdnQkvBuf?.Dispose(); GdnZBuf?.Dispose();
        GdnAlphaBuf?.Dispose(); GdnBetaBuf?.Dispose();
        GdnQBuf?.Dispose(); GdnKBuf?.Dispose(); GdnVBuf?.Dispose(); GdnOut?.Dispose();
        MoeRouterLogits?.Dispose(); MoeTopkIndices?.Dispose(); MoeTopkWeights?.Dispose();
        MoeExpandedInput?.Dispose();
        MoeGateInter?.Dispose(); MoeUpInter?.Dispose(); MoeSiluInter?.Dispose(); MoeDownRows?.Dispose();
        MoeSharedInput?.Dispose(); MoeSharedGate?.Dispose(); MoeSharedUp?.Dispose();
        MoeSharedSilu?.Dispose(); MoeSharedSumA?.Dispose(); MoeSharedSumB?.Dispose();
        MoeSharedGateLogits?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        ReleaseLayerScratch();
        GdnTokenQ?.Dispose(); GdnTokenK?.Dispose(); GdnTokenV?.Dispose();
        GdnTokenG?.Dispose(); GdnTokenBeta?.Dispose(); GdnTokenOut?.Dispose();
        Logits?.Dispose();
        PositionsBuffer?.Dispose();
    }
}
