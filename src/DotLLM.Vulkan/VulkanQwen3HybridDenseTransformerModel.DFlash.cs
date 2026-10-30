using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures.DFlash;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// DFlash on the Vulkan <c>qwen35</c> trunk (issues #1175, #1193), all on the GPU: the verify forward copies the tapped layer
/// inputs into a host-readback buffer (F16 in the state), and the drafter (<see cref="VulkanDFlashBackend"/>) runs on the device
/// with its weights in their stored quantisation (imported zero-copy on UMA), its context ring in device F16, and the target's
/// own embedding table and lm-head. The injection of a verify's taps and the next draft record into one command buffer that is
/// submitted once, when the draft's logits are read.
/// </summary>
public sealed partial class VulkanQwen3HybridDenseTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps, IVulkanDFlashHost
{
    private DFlashDrafter? _dflash;
    private VulkanDFlashBackend? _dflashGpu;
    private string _spvDir = string.Empty;
    private int[] _dflashSlotForLayerInput = [];
    private VulkanDevice.Buffer? _dflashTapBuffer;   // [slot][row][hidden] F32, host readback, transient
    private long _dflashTapBufferBytes;
    private float[] _dflashTapHost = [];

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dflash is not null;

    /// <summary>Device bytes held by the attached drafter's weights and norm vectors (0 without one).</summary>
    public long DFlashWeightBytes => _dflashGpu?.WeightBytes ?? 0;

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dflash is not null) throw new InvalidOperationException("A DFlash drafter is already attached.");
        var drafter = DFlashDrafter.Open(path, Config.HiddenSize, Config.NumLayers, Config.VocabSize, CreateDFlashBackend());
        int[] slots;
        try { slots = drafter.SlotForLayerInput(Config.NumLayers); }
        catch { drafter.Dispose(); throw; }
        _dflashSlotForLayerInput = slots;
        _dflashGpu = (VulkanDFlashBackend)drafter.Backend;
        _dflash = drafter;
        // Sized now so a draft never grows them mid-recording (a grow flushes the recorded draft early).
        if (_state.EnsureCapacity(drafter.BlockSize)) InvalidateDescriptorCaches();
        EnsureMultiRowLogits(drafter.BlockSize, Config.VocabSize);
    }

    /// <summary>A drafter backend on this model's device (also the tests' handle on the Vulkan drafter kernels).</summary>
    /// <returns>The backend.</returns>
    internal IDFlashBackend CreateDFlashBackend() => new VulkanDFlashBackend(this, VulkanDFlashKernels.Create(_device, _spvDir));

    /// <summary>Draft tokens per round (see the CPU host).</summary>
    public int MtpPreferredDraftTokens => _dflash is null ? 0 : _dflash.DraftTokensPerRound;

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dflash is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(VulkanQwen3HybridDenseTransformerModel)} has a native draft argmax only for DFlash states.");
        return dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
    }

    private ITensor ForwardDFlashDraft(DFlashMtpState state, int tokenId, int position)
    {
        state.NextDraft(this, tokenId, position, state.Drafter.DraftTokensPerRound);
        var row = state.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        unsafe { row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length)); }
        return result;
    }

    /// <summary>Allocates the tap buffer for <paramref name="rows"/> before any recording (a new buffer invalidates descriptor caches).</summary>
    private void EnsureDFlashTapBuffer(int rows)
    {
        long bytes = (long)_dflash!.TargetLayers.Length * rows * Config.HiddenSize * sizeof(float);
        if (_dflashTapBuffer is not null && _dflashTapBufferBytes >= bytes) return;
        _dflashTapBuffer?.Dispose();
        _dflashTapBuffer = _device.AllocateHostReadback(bytes);
        _dflashTapBufferBytes = bytes;
        InvalidateDescriptorCaches();
    }

    /// <summary>Records the copy of <c>HiddenState</c> rows <c>[first, first + rows)</c> into tap slot of <paramref name="layerInput"/>.</summary>
    private void RecordDFlashTap(nint cmdBuf, int layerInput, int first, int rows)
    {
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        long rowBytes = (long)Config.HiddenSize * sizeof(float);
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _dflashTapBuffer!,
            srcOffset: (ulong)(first * rowBytes), dstOffset: (ulong)((long)slot * rows * rowBytes), size: (ulong)(rows * rowBytes));
    }

    /// <summary>Hands the downloaded taps to the state and injects them (after the forward's last submit).</summary>
    private unsafe void FinishDFlashTaps(DFlashMtpState state, ReadOnlySpan<int> positions, int rows)
    {
        int h = Config.HiddenSize, slots = _dflash!.TargetLayers.Length;
        int n = slots * rows * h;
        if (_dflashTapHost.Length < n) _dflashTapHost = new float[n];
        var host = _dflashTapHost.AsSpan(0, n);
        _device.Download(_dflashTapBuffer!, host);
        state.EnsureTapRows(rows);
        for (int s = 0; s < slots; s++)
            for (int r = 0; r < rows; r++)
                state.CaptureTap(r, s, host.Slice((s * rows + r) * h, h));
        state.InjectCaptured(positions);   // recorded on the drafter's command buffer; the next draft (or any invalidation) submits it
        state.TrimTapRows();
        if (rows > 4 * _dflash.BlockSize)
        {
            // A long prefill grew it: give the transient buffer back.
            _dflashTapBuffer!.Dispose();
            _dflashTapBuffer = null;
            _dflashTapBufferBytes = 0;
            _dflashTapHost = [];
            InvalidateDescriptorCaches();
        }
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, nint dst)
    {
        var g = _dflashGpu!;
        if (_state.EnsureCapacity(tokens.Length)) InvalidateDescriptorCaches();
        nint cmd = g.Command();
        RecordEmbeddingGather(cmd, tokens);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        var (buf, off) = g.Resolve(dst);
        RecordCopyBufferRange(cmd, _state.HiddenState, buf, 0, (ulong)off, (ulong)((long)tokens.Length * Config.HiddenSize * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
    }

    /// <inheritdoc/>
    unsafe void IDFlashTargetOps.ProjectForDraft(nint hidden, int rows, float* logits)
    {
        var g = _dflashGpu!;
        int h = Config.HiddenSize, vocab = Config.VocabSize;
        bool resized = _state.EnsureCapacity(rows);
        var logitsBuf = rows == 1 ? SingleRowLogits : EnsureMultiRowLogits(rows, vocab);
        if (resized) InvalidateDescriptorCaches();
        nint cmd = g.Command();
        var (buf, off) = g.Resolve(hidden);
        RecordCopyBufferRange(cmd, buf, _state.NormOutput, (ulong)off, 0, (ulong)((long)rows * h * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
        var headIn = _state.NormOutput;
        if (_hadamard is { } headRot)
        {
            headIn = _state.HadamardScratch!;
            headRot.RecordForward(cmd, _state.NormOutput, headIn, rows, _weights.OutputInputDim);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        RecordMatmul(cmd, _weights.OutputWeight, _weights.OutputDeviceQuantType, headIn, logitsBuf,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: rows);
        g.Flush();
        _device.Download(logitsBuf, new Span<float>(logits, rows * vocab));
    }

    private void DisposeDFlash()
    {
        _dflashTapBuffer?.Dispose();
        _dflash?.Dispose();
        _dflashGpu = null;
    }

    // ── IVulkanDFlashHost ───────────────────────────────────────────

    VulkanDevice IVulkanDFlashHost.Device => _device;

    RmsNormF32Kernel IVulkanDFlashHost.DFlashRmsNorm => _kernels.RmsNorm;

    SwiGluF32Kernel IVulkanDFlashHost.DFlashSwiGlu => _kernels.SwiGlu;

    AddKernel IVulkanDFlashHost.DFlashAdd => _kernels.Add;

    long IVulkanDFlashHost.DFlashUploadBytes(in DFlashDrafter.Matrix w)
        => VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(w.Out, w.In, w.Qt);

    VulkanDevice.Buffer IVulkanDFlashHost.UploadDFlashMatrix(VulkanStagingBuffer stage, in DFlashDrafter.Matrix w,
        out QuantizationType deviceQt, out long bytes)
        => VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(_device, stage, w.Ptr, w.Qt, w.Out, w.In, forceF32: false, out deviceQt, out bytes);

    void IVulkanDFlashHost.RecordDFlashMatmul(nint cmd, VulkanDevice.Buffer w, QuantizationType qt, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int outDim, int inDim, int rows, float scale)
    {
        if (scale != 1f) throw new NotSupportedException("The qwen35 Vulkan host does not apply per-tensor weight scales (X.scale) to DFlash projections.");
        RecordMatmul(cmd, w, qt, x, y, outputDim: outDim, inputDim: inDim, seqLen: rows);
    }

    void IVulkanDFlashHost.OnDFlashBackendDisposed(VulkanDFlashBackend backend)
    {
        if (ReferenceEquals(_dflashGpu, backend)) _dflashGpu = null;
    }
}
