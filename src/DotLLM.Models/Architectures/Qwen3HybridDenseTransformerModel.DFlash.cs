using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Architectures.DFlash;

namespace DotLLM.Models.Architectures;

/// <summary>DFlash block-diffusion drafting on the CPU <c>qwen35</c> trunk (issue #1175).</summary>
public sealed unsafe partial class Qwen3HybridDenseTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps
{
    private DFlashDrafter? _dflash;
    private int[] _dflashSlotForLayerInput = [];   // [layer input 0..NumLayers] -> tap slot, -1 = not tapped
    private nint _dflashHeadScratch;               // Hadamard-rotated head input for the drafted rows

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
        _dflashSlotForLayerInput = slots;
        _dflashHeadScratch = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
            (nuint)((long)drafter.BlockSize * Config.HiddenSize * sizeof(float)), 64);
        _dflash = drafter;
    }

    /// <summary>Draft tokens per round: the block's drafts, capped by <c>DOTLLM_DFLASH_DRAFT</c> when set.</summary>
    public int MtpPreferredDraftTokens => _dflash is null ? 0 : _dflash.DraftTokensPerRound;

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dflash is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(Qwen3HybridDenseTransformerModel)} has a native draft argmax only for DFlash states.");
        return dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
    }

    private ITensor ForwardDFlashDraft(DFlashMtpState state, int tokenId, int position)
    {
        state.NextDraft(this, tokenId, position, state.Drafter.DraftTokensPerRound);
        var row = state.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    private void CaptureDFlashTaps(DFlashMtpState state, int layerInput, float* hidden, int first, int seqLen, int hiddenSize)
    {
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        for (int t = first; t < seqLen; t++)
            state.CaptureTap(t - first, slot, new ReadOnlySpan<float>(hidden + (long)t * hiddenSize, hiddenSize));
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, float* dst) => EmbedTokens(tokens, dst, Config.HiddenSize);

    /// <inheritdoc/>
    void IDFlashTargetOps.ProjectForDraft(float* hidden, int rows, float* logits)
    {
        float* headIn = hidden;
        if (_hadamard is { } headRot)
        {
            headIn = (float*)_dflashHeadScratch;
            headRot.RotateForward(hidden, headIn, rows, _outputInputDim);
        }
        Gemm(_outputWeight, _outputQuantType, headIn, logits, _outputOutputDim, _outputInputDim, rows);
    }

    /// <inheritdoc/>
    ComputeThreadPool? IDFlashTargetOps.DraftPool => _threadPool;
}
