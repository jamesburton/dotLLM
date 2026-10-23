using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Architectures.DFlash;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// DFlash on the Vulkan <c>qwen35</c> trunk (issue #1175): the target verifies on the GPU, the drafter runs on the CPU.
/// The verify forward copies the tapped layer inputs into a host-readback buffer; the drafter's block embedding and its
/// lm-head run on the device with the target's own weights (nothing is copied to the host).
/// </summary>
public sealed partial class VulkanQwen3HybridDenseTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps
{
    private DFlashDrafter? _dflash;
    private ComputeThreadPool? _dflashPool;
    private int[] _dflashSlotForLayerInput = [];
    private VulkanDevice.Buffer? _dflashTapBuffer;   // [slot][row][hidden] F32, host readback, transient
    private long _dflashTapBufferBytes;
    private float[] _dflashTapHost = [];

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dflash is not null;

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dflash is not null) throw new InvalidOperationException("A DFlash drafter is already attached.");
        var drafter = DFlashDrafter.Open(path, Config.HiddenSize, Config.NumLayers, Config.VocabSize);
        int[] slots;
        try { slots = drafter.SlotForLayerInput(Config.NumLayers); }
        catch { drafter.Dispose(); throw; }
        int threads = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_DFLASH_THREADS"), out int t) && t > 0
            ? t : Math.Max(1, Environment.ProcessorCount / 2);
        _dflashPool = threads > 1 ? new ComputeThreadPool(threads) : null;
        _dflashSlotForLayerInput = slots;
        _dflash = drafter;
    }

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
        state.InjectCaptured(positions, _dflashPool);
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
    unsafe void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, float* dst)
    {
        if (_state.EnsureCapacity(tokens.Length)) InvalidateDescriptorCaches();
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordEmbeddingGather(cmdBuf, tokens);
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        KernelSupport.TransferToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();
        _device.Download(_state.HiddenState, new Span<float>(dst, tokens.Length * Config.HiddenSize));
    }

    /// <inheritdoc/>
    unsafe void IDFlashTargetOps.ProjectForDraft(float* hidden, int rows, float* logits)
    {
        int h = Config.HiddenSize, vocab = Config.VocabSize;
        bool resized = _state.EnsureCapacity(rows);
        var logitsBuf = rows == 1 ? SingleRowLogits : EnsureMultiRowLogits(rows, vocab);
        if (resized) InvalidateDescriptorCaches();
        _device.Upload(new ReadOnlySpan<float>(hidden, rows * h), _state.NormOutput);
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        var headIn = _state.NormOutput;
        if (_hadamard is { } headRot)
        {
            headIn = _state.HadamardScratch!;
            headRot.RecordForward(cmdBuf, _state.NormOutput, headIn, rows, _weights.OutputInputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType, headIn, logitsBuf,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: rows);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();
        _device.Download(logitsBuf, new Span<float>(logits, rows * vocab));
    }

    /// <inheritdoc/>
    ComputeThreadPool? IDFlashTargetOps.DraftPool => _dflashPool;

    private void DisposeDFlash()
    {
        _dflashTapBuffer?.Dispose();
        _dflash?.Dispose();
        _dflashPool?.Dispose();
    }
}
