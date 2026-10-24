using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures.DFlash;

namespace DotLLM.Models.Architectures;

/// <summary>
/// DFlash / LiLiCorr block drafting on the CPU <c>nemotron_h</c> / <c>nemotron_h_moe</c> trunk (issue #1237), the oracle for the
/// Vulkan host. The drafter takes the MTP slot exactly as on <c>qwen35</c> (#1175): a forward carrying its state captures the
/// tapped layer inputs and injects them into the drafter's context, and the MTP speculative decoder verifies the drafts.
/// Verify and replay batches are at most <see cref="RowByRowMaxTokens"/> rows, so they run row by row through the decode path
/// and the recurrent rollback is a checkpoint of the Mamba-2 conv/SSM state plus the decoder's replay of the accepted rows.
/// </summary>
public sealed unsafe partial class NemotronHTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps
{
    private DFlashDrafter? _dflash;
    private int[] _dflashSlotForLayerInput = [];   // [layer input 0..NumLayers] -> tap slot, -1 = not tapped

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dflash is not null;

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dflash is not null) throw new InvalidOperationException("A DFlash drafter is already attached.");
        var drafter = DFlashDrafter.Open(path, Config.HiddenSize, Config.NumLayers, Config.VocabSize,
            new DFlashCpuBackend(_threadPool), _threadPool);
        int[] slots;
        try { slots = drafter.SlotForLayerInput(Config.NumLayers); }
        catch { drafter.Dispose(); throw; }
        _dflashSlotForLayerInput = slots;
        _dflash = drafter;
    }

    /// <inheritdoc/>
    public bool SupportsMtp => _dflash is not null;

    /// <summary>Draft tokens per round: the block's drafts, capped by <c>DOTLLM_DFLASH_DRAFT</c> when set.</summary>
    public int MtpPreferredDraftTokens => _dflash is null ? 0 : _dflash.DraftTokensPerRound;

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => _dflash is null ? null : new DFlashMtpState(_dflash);

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dflash is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(NemotronHTransformerModel)} drafts only through an attached DFlash drafter.");
        return dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
    }

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(NemotronHTransformerModel)} drafts only through an attached DFlash drafter.");
        dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
        var row = dflash.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
    {
        if (adapter is not null)
            throw new NotSupportedException($"{nameof(NemotronHTransformerModel)} does not support LoRA adapters.");
        var taps = _dflash is not null ? mtpState as DFlashMtpState : null;
        if (taps is null) return Forward(tokenIds, positions, deviceId, kvCache);
        taps.EnsureTapRows(tokenIds.Length);
        var result = ForwardWithTaps(tokenIds, positions, deviceId, kvCache, taps);
        try
        {
            taps.InjectCaptured(positions);
            taps.TrimTapRows();
        }
        catch
        {
            result.Dispose();
            throw;
        }
        return result;
    }

    private void CaptureDFlashTaps(DFlashMtpState state, int layerInput, float* hidden, int seqLen, int row0)
    {
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        int h = Config.HiddenSize;
        for (int t = 0; t < seqLen; t++)
            state.CaptureTap(row0 + t, slot, new ReadOnlySpan<float>(hidden + (long)t * h, h));
    }

    // ── Recurrent checkpoint (the decoder's restore + replay rollback) ──

    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    /// <inheritdoc/>
    public object? CheckpointRecurrentState() => new SsmCheckpoint(_activeSsm ?? _ssmCache);

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        if (checkpoint is null) return;
        if (checkpoint is not SsmCheckpoint c)
            throw new ArgumentException($"Not a {nameof(NemotronHTransformerModel)} checkpoint.", nameof(checkpoint));
        c.RestoreInto(_activeSsm ?? _ssmCache);
    }

    /// <summary>A copy of every SSM layer's conv and SSM state (F32, the state's own type: a rollback must be bit-exact).</summary>
    private sealed class SsmCheckpoint : IDisposable
    {
        private readonly SsmStateCache _source;
        private float* _data;
        private readonly long _conv, _ssm;

        public SsmCheckpoint(SsmStateCache cache)
        {
            _source = cache;
            _conv = cache.ConvStateElements;
            _ssm = cache.SsmStateElements;
            long total = Math.Max(1, cache.NumSsmLayers * (_conv + _ssm));
            _data = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
            for (int i = 0; i < cache.NumSsmLayers; i++)
            {
                cache.GetConvState(i).CopyTo(new Span<float>(_data + i * (_conv + _ssm), (int)_conv));
                cache.GetSsmState(i).CopyTo(new Span<float>(_data + i * (_conv + _ssm) + _conv, (int)_ssm));
            }
        }

        public void RestoreInto(SsmStateCache cache)
        {
            ObjectDisposedException.ThrowIf(_data == null, this);
            if (!ReferenceEquals(cache, _source))
                throw new InvalidOperationException("The checkpoint belongs to a different SSM state.");
            for (int i = 0; i < cache.NumSsmLayers; i++)
            {
                new ReadOnlySpan<float>(_data + i * (_conv + _ssm), (int)_conv).CopyTo(cache.GetConvState(i));
                new ReadOnlySpan<float>(_data + i * (_conv + _ssm) + _conv, (int)_ssm).CopyTo(cache.GetSsmState(i));
            }
        }

        public void Dispose()
        {
            if (_data != null) NativeMemory.AlignedFree(_data);
            _data = null;
        }
    }

    // ── IDFlashTargetOps ──

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, nint dst) => EmbedTokens(tokens, (float*)dst, Config.HiddenSize);

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedHost(ReadOnlySpan<int> tokens, float* dst) => EmbedTokens(tokens, dst, Config.HiddenSize);

    /// <inheritdoc/>
    void IDFlashTargetOps.ProjectForDraft(nint hiddenRows, int rows, float* logits)
        => Gemm(_outputWeight, _outputQuantType, (float*)hiddenRows, logits, _outputOutputDim, _outputInputDim, rows, preQuantizedInput: null);
}
