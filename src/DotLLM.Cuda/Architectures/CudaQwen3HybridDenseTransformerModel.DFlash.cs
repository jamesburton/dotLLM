using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures.DFlash;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// DFlash on the CUDA <c>qwen35</c> trunk (issue #1175): the target verifies on the GPU, the drafter runs on the CPU. The
/// verify forward copies the tapped layer inputs asynchronously into pinned host memory; the drafter's block embedding uses
/// the host embedding path the trunk already uses, and its lm-head runs on the device with the target's own head.
/// </summary>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps
{
    private DFlashDrafter? _dflash;
    private ComputeThreadPool? _dflashPool;
    private int[] _dflashSlotForLayerInput = [];
    private DFlashMtpState? _dflashCall;     // the DFlash state of the forward in flight, null otherwise
    private nint _dflashTapPinned;           // [slot][row][hidden] F32, page-locked
    private long _dflashTapPinnedBytes;

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
    public int MtpPreferredDraftTokens => _dflash?.DraftTokensPerRound ?? 0;

    private int ForwardDFlashArgMax(DFlashMtpState state, int tokenId, int position)
        => state.NextDraft(this, tokenId, position, state.Drafter.DraftTokensPerRound);

    private ITensor ForwardDFlashDraft(DFlashMtpState state, int tokenId, int position)
    {
        ForwardDFlashArgMax(state, tokenId, position);
        var row = state.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    /// <summary>Rows of this tile the drafter's ring can hold: <c>[first, seqLen)</c>.</summary>
    private int DFlashTapFirst(int seqLen) => Math.Max(0, seqLen - _dflashCall!.RingCapacity);

    private void EnsureDFlashTapPinned(int rows)
    {
        long bytes = (long)_dflash!.TargetLayers.Length * rows * Config.HiddenSize * sizeof(float);
        if (_dflashTapPinned != 0 && _dflashTapPinnedBytes >= bytes) return;
        FreeDFlashTapPinned();
        CudaDriverApi.cuMemHostAlloc(out _dflashTapPinned, (nuint)bytes, 0).ThrowOnError();
        _dflashTapPinnedBytes = bytes;
    }

    private void FreeDFlashTapPinned()
    {
        if (_dflashTapPinned != 0) CudaDriverApi.cuMemFreeHost(_dflashTapPinned);
        _dflashTapPinned = 0;
        _dflashTapPinnedBytes = 0;
    }

    /// <summary>Queues the D2H copy of <c>HiddenState</c> rows <c>[first, seqLen)</c> into the tap slot of <paramref name="layerInput"/>.</summary>
    private void QueueDFlashTap(int layerInput, int seqLen)
    {
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        int first = DFlashTapFirst(seqLen), rows = seqLen - first;
        long rowBytes = (long)Config.HiddenSize * sizeof(float);
        CudaDriverApi.cuMemcpyDtoHAsync_v2(_dflashTapPinned + (nint)((long)slot * rows * rowBytes),
            _state.HiddenState + (nint)(first * rowBytes), (nuint)(rows * rowBytes), _stream.Handle).ThrowOnError();
    }

    /// <summary>After the tile's stream synchronize: hands the taps to the state and injects them.</summary>
    private void FinishDFlashTaps(ReadOnlySpan<int> positions)
    {
        var state = _dflashCall!;
        int first = DFlashTapFirst(positions.Length), rows = positions.Length - first, h = Config.HiddenSize;
        state.EnsureTapRows(rows);
        float* host = (float*)_dflashTapPinned;
        for (int s = 0; s < _dflash!.TargetLayers.Length; s++)
            for (int r = 0; r < rows; r++)
                state.CaptureTap(r, s, new ReadOnlySpan<float>(host + ((long)s * rows + r) * h, h));
        state.InjectCaptured(positions.Slice(first), _dflashPool);
        state.TrimTapRows();
        if (rows > 4 * _dflash.BlockSize) FreeDFlashTapPinned();   // a long prefill tile grew it
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, float* dst)
    {
        int h = Config.HiddenSize;
        for (int t = 0; t < tokens.Length; t++)
        {
            nint rowSrc = _embedDataBase + (nint)(_embedDataOffset + (ulong)tokens[t] * (ulong)_embedRowBytes);
            Dequantize.ToFloat32(rowSrc, h, _tokenEmbedQt, new Span<float>(dst + (long)t * h, h));
        }
        _hadamard?.RotateInverseInPlaceHost(dst, tokens.Length, h);
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.ProjectForDraft(float* hidden, int rows, float* logits)
    {
        _context.MakeCurrent();
        _state.EnsureCapacity(rows);
        _state.EnsureLogitsCapacity(rows);
        nint streamH = _stream.Handle;
        CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.HiddenState, (nint)hidden,
            (nuint)((long)rows * Config.HiddenSize * sizeof(float)), streamH).ThrowOnError();
        nint headIn = _state.HiddenState;
        if (_hadamard is { } headRot)
        {
            headIn = _state.HadamardScratch;
            headRot.RotateForward(_state.HiddenState, headIn, rows, _outputInputDim, permuteGdnValueHeads: false, streamH);
        }
        GemmLmHead(_outputDevice, _outputQt, headIn, _state.Logits, _outputOutputDim, _outputInputDim, rows);
        _stream.Synchronize();
        CudaDriverApi.cuMemcpyDtoH_v2((nint)logits, _state.Logits,
            (nuint)((long)rows * Config.VocabSize * sizeof(float))).ThrowOnError();
    }

    /// <inheritdoc/>
    ComputeThreadPool? IDFlashTargetOps.DraftPool => _dflashPool;

    private void DisposeDFlash()
    {
        FreeDFlashTapPinned();
        _dflash?.Dispose();
        _dflashPool?.Dispose();
    }
}
