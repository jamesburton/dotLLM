using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-forward scratch buffers for the NemotronH Vulkan path. Sized for the maximum
/// <c>seqLen</c> seen so far; grows monotonically with <see cref="EnsureCapacity"/>.
/// Mirrors <see cref="VulkanForwardState"/> in slot management but adds the SSM-specific
/// scratch (Zxbcdt, ConvInput, XBC, DtBuf, SsmX, SsmB, SsmC, SsmY, SsmZ) and the FFN
/// intermediate.
/// </summary>
internal sealed class VulkanNemotronHForwardState : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _maxIntermediateSize;
    private readonly int _vocabSize;
    private readonly int _qElems;     // numHeads * headDim
    private readonly int _kvElems;    // numKvHeads * headDim
    private readonly int _inputProjectionDim;
    private readonly int _convDim;
    private readonly int _dConv;
    private readonly int _dInner;
    private readonly int _nHead;
    private readonly int _bcDim;      // n_group * d_state — width of B and width of C

    private int _capacitySeqLen;

    // ── Hidden / residual rotation pair (mirrors VulkanForwardState) ─────
    private readonly VulkanDevice.Buffer[] _hiddenSlots = new VulkanDevice.Buffer[2];
    private int _hiddenIdx;
    public VulkanDevice.Buffer HiddenState => _hiddenSlots[_hiddenIdx];
    public VulkanDevice.Buffer Residual => _hiddenSlots[_hiddenIdx];
    public VulkanDevice.Buffer AddScratch => _hiddenSlots[_hiddenIdx ^ 1];
    public void RotateHiddenSlot() => _hiddenIdx ^= 1;
    public void ResetHiddenSlot() => _hiddenIdx = 0;

    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;

    // ── Attention scratch ─────────────────────────────────────────────────
    public VulkanDevice.Buffer Q { get; private set; } = null!;
    public VulkanDevice.Buffer K { get; private set; } = null!;
    public VulkanDevice.Buffer V { get; private set; } = null!;
    public VulkanDevice.Buffer AttnOutput { get; private set; } = null!;

    // ── SSM scratch ───────────────────────────────────────────────────────
    public VulkanDevice.Buffer Zxbcdt { get; private set; } = null!;
    public VulkanDevice.Buffer ConvInput { get; private set; } = null!;   // [(d_conv-1+seqLen) * conv_dim]
    public VulkanDevice.Buffer XBC { get; private set; } = null!;          // [seqLen, conv_dim]
    public VulkanDevice.Buffer DtBuf { get; private set; } = null!;        // [seqLen, n_head]
    public VulkanDevice.Buffer SsmX { get; private set; } = null!;         // [seqLen, d_inner]
    public VulkanDevice.Buffer SsmB { get; private set; } = null!;         // [seqLen, n_group * d_state]
    public VulkanDevice.Buffer SsmC { get; private set; } = null!;         // [seqLen, n_group * d_state]
    public VulkanDevice.Buffer SsmY { get; private set; } = null!;         // [seqLen, d_inner] (in-place GroupRMSNorm + SwiGLU output)
    public VulkanDevice.Buffer SsmZ { get; private set; } = null!;         // [seqLen, d_inner] (extracted z slice for the silu(z)*y gate)

    // ── FFN scratch ───────────────────────────────────────────────────────
    public VulkanDevice.Buffer FfnIntermediate { get; private set; } = null!;

    // ── MoE scratch (nemotron_h_moe, #402); null when the model has no routed layers ─────────────────────────
    // Sized from the per-layer MAXIMA (#1045): Nemotron 3.5 Bolt varies experts / top-k / width per layer, and its LatentMoE
    // experts read a [seqLen, latent] input instead of the hidden state. ExpertDim is the widest expert input (latent or hidden).
    private readonly NemotronHMoeScratchSpec? _moe;
    public VulkanDevice.Buffer? MoeRouterLogits { get; private set; }      // [seqLen, numExperts]
    public VulkanDevice.Buffer? MoeTopkIndices { get; private set; }       // [seqLen, topK] int32
    public VulkanDevice.Buffer? MoeTopkWeights { get; private set; }       // [seqLen, topK]
    public VulkanDevice.Buffer? MoeExpandedInput { get; private set; }     // [seqLen*topK, expertDim]  (token row repeated per slot)
    public VulkanDevice.Buffer? MoeExpandedInputXq { get; private set; }   // Q8_1 packed int8 of MoeExpandedInput (MMVQ decode path)
    public VulkanDevice.Buffer? MoeExpandedInputXds { get; private set; }
    public VulkanDevice.Buffer? MoeUpInter { get; private set; }           // [seqLen*topK, moeIntermediate]
    public VulkanDevice.Buffer? MoeUpInterXq { get; private set; }         // Q8_1 packed int8 of the relu^2 output (MMVQ down)
    public VulkanDevice.Buffer? MoeUpInterXds { get; private set; }
    public VulkanDevice.Buffer? MoeDownRows { get; private set; }          // [seqLen*topK, expertDim]
    public VulkanDevice.Buffer? MoeSharedOut { get; private set; }         // [seqLen, hidden] shared-expert output
    public VulkanDevice.Buffer? MoeLatentIn { get; private set; }          // [seqLen, latent] ffn_latent_down(x); null without LatentMoE
    public VulkanDevice.Buffer? MoeLatentOut { get; private set; }         // [seqLen, latent] weighted expert sum before ffn_latent_up

    // ── Logits + positions ────────────────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; private set; }
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    public VulkanNemotronHForwardState(
        VulkanDevice device,
        int hiddenSize, int maxIntermediateSize, int vocabSize,
        int qElems, int kvElems,
        MambaSsmConfig ssm,
        int initialSeqLen,
        NemotronHMoeScratchSpec? moe = null)
    {
        _device = device;
        _moe = moe;
        _hiddenSize = hiddenSize;
        _maxIntermediateSize = maxIntermediateSize;
        _vocabSize = vocabSize;
        _qElems = qElems;
        _kvElems = kvElems;
        _inputProjectionDim = ssm.InputProjectionDim;
        _convDim = ssm.ConvDim;
        _dConv = ssm.DConv;
        _dInner = ssm.DInner;
        _nHead = ssm.NHead;
        _bcDim = ssm.NGroup * ssm.DState;

        // Read back by the host every decoded token: HOST_CACHED, not write-combined (#143, #471).
        Logits = device.AllocateHostReadback((long)vocabSize * sizeof(float));
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
        long ffnBytes = (long)seqLen * _maxIntermediateSize * sizeof(float);
        long zxbcdtBytes = (long)seqLen * _inputProjectionDim * sizeof(float);
        long convInputBytes = (long)(_dConv - 1 + seqLen) * _convDim * sizeof(float);
        long xbcBytes = (long)seqLen * _convDim * sizeof(float);
        long dtBytes = (long)seqLen * _nHead * sizeof(float);
        long ssmXBytes = (long)seqLen * _dInner * sizeof(float);
        long ssmBcBytes = (long)seqLen * _bcDim * sizeof(float);
        long ssmYBytes = ssmXBytes;
        long ssmZBytes = ssmXBytes;

        _hiddenSlots[0] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenSlots[1] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenIdx = 0;
        NormOutput = _device.AllocateDeviceLocal(hiddenBytes);

        Q = _device.AllocateDeviceLocal(qBytes);
        K = _device.AllocateDeviceLocal(kvBytes);
        V = _device.AllocateDeviceLocal(kvBytes);
        AttnOutput = _device.AllocateDeviceLocal(qBytes);

        Zxbcdt = _device.AllocateDeviceLocal(zxbcdtBytes);
        ConvInput = _device.AllocateDeviceLocal(convInputBytes);
        XBC = _device.AllocateDeviceLocal(xbcBytes);
        DtBuf = _device.AllocateDeviceLocal(dtBytes);
        SsmX = _device.AllocateDeviceLocal(ssmXBytes);
        SsmB = _device.AllocateDeviceLocal(ssmBcBytes);
        SsmC = _device.AllocateDeviceLocal(ssmBcBytes);
        SsmY = _device.AllocateDeviceLocal(ssmYBytes);
        SsmZ = _device.AllocateDeviceLocal(ssmZBytes);

        FfnIntermediate = _device.AllocateDeviceLocal(ffnBytes);

        long moeBytes = 0;
        if (_moe is { } moe)
        {
            long rows = (long)seqLen * moe.TopK;
            int expertDim = moe.ExpertDim;
            MoeRouterLogits = _device.AllocateDeviceLocal((long)seqLen * moe.NumExperts * sizeof(float));
            MoeTopkIndices = _device.AllocateDeviceLocal(rows * sizeof(int));
            MoeTopkWeights = _device.AllocateDeviceLocal(rows * sizeof(float));
            MoeExpandedInput = _device.AllocateDeviceLocal(rows * expertDim * sizeof(float));
            MoeExpandedInputXq = _device.AllocateDeviceLocal(Kernels.QuantizeQ8_1RowsKernel.PackedBytes((int)rows, expertDim));
            MoeExpandedInputXds = _device.AllocateDeviceLocal(Kernels.QuantizeQ8_1RowsKernel.ScaleBytes((int)rows, expertDim));
            MoeUpInter = _device.AllocateDeviceLocal(rows * moe.MoeIntermediate * sizeof(float));
            MoeUpInterXq = _device.AllocateDeviceLocal(Kernels.QuantizeQ8_1RowsKernel.PackedBytes((int)rows, moe.MoeIntermediate));
            MoeUpInterXds = _device.AllocateDeviceLocal(Kernels.QuantizeQ8_1RowsKernel.ScaleBytes((int)rows, moe.MoeIntermediate));
            MoeDownRows = _device.AllocateDeviceLocal(rows * expertDim * sizeof(float));
            MoeSharedOut = _device.AllocateDeviceLocal(hiddenBytes);
            moeBytes = MoeRouterLogits.Size + MoeTopkIndices.Size + MoeTopkWeights.Size + MoeExpandedInput.Size
                     + MoeExpandedInputXq.Size + MoeExpandedInputXds.Size + MoeUpInter.Size + MoeUpInterXq.Size
                     + MoeUpInterXds.Size + MoeDownRows.Size + MoeSharedOut.Size;
            if (moe.LatentSize > 0)
            {
                MoeLatentIn = _device.AllocateDeviceLocal((long)seqLen * moe.LatentSize * sizeof(float));
                MoeLatentOut = _device.AllocateDeviceLocal((long)seqLen * moe.LatentSize * sizeof(float));
                moeBytes += MoeLatentIn.Size + MoeLatentOut.Size;
            }
        }

        // Resize positions buffer (host writes positions per forward).
        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;
        AllocatedBytes = hiddenBytes * 3 + qBytes * 2 + kvBytes * 2 + ffnBytes
                       + zxbcdtBytes + convInputBytes + xbcBytes + dtBytes
                       + ssmXBytes + ssmBcBytes * 2 + ssmYBytes + ssmZBytes
                       + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int) + moeBytes;
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
        Zxbcdt?.Dispose();
        ConvInput?.Dispose();
        XBC?.Dispose();
        DtBuf?.Dispose();
        SsmX?.Dispose();
        SsmB?.Dispose();
        SsmC?.Dispose();
        SsmY?.Dispose();
        SsmZ?.Dispose();
        FfnIntermediate?.Dispose();
        MoeRouterLogits?.Dispose(); MoeTopkIndices?.Dispose(); MoeTopkWeights?.Dispose();
        MoeExpandedInput?.Dispose(); MoeExpandedInputXq?.Dispose(); MoeExpandedInputXds?.Dispose();
        MoeUpInter?.Dispose(); MoeUpInterXq?.Dispose(); MoeUpInterXds?.Dispose();
        MoeDownRows?.Dispose(); MoeSharedOut?.Dispose();
        MoeLatentIn?.Dispose(); MoeLatentOut?.Dispose();
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

/// <summary>
/// Routed-MoE scratch geometry of a nemotron_h_moe model: the MAXIMUM over its MoE layers of each dimension, since Nemotron 3.5 Bolt
/// varies them per layer (#1045). Every layer dispatches with its own geometry inside these buffers.
/// </summary>
/// <param name="NumExperts">Largest routed-expert count of any layer (router-logit row width).</param>
/// <param name="TopK">Largest experts-per-token of any layer.</param>
/// <param name="MoeIntermediate">Widest routed-expert intermediate of any layer.</param>
/// <param name="ExpertDim">Widest routed-expert input: the LatentMoE width for a latent layer, the hidden size otherwise.</param>
/// <param name="LatentSize">Largest LatentMoE width of any layer, or 0 when no layer is latent.</param>
internal readonly record struct NemotronHMoeScratchSpec(int NumExperts, int TopK, int MoeIntermediate, int ExpertDim, int LatentSize);
