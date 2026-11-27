using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures.DSpark;

namespace DotLLM.Models.Architectures;

/// <summary>
/// The DSpark speculative drafter on the CPU Inkling 2 target (issue #2039). The drafter takes the model's MTP slot (<see cref="IDFlashDraftAttachable"/>),
/// so the unmodified <c>MtpSpeculativeDecoder</c> verifies its drafts. The verify forward is row-count invariant (bit-identical to single-row decoding)
/// and, with <see cref="ForwardWithRecurrentSnapshots"/>, records the short-conv windows so a rejection rewinds the recurrent state without a replay.
/// </summary>
public sealed unsafe partial class Inkling2TransformerModel : IDFlashDraftAttachable, IInkling2DSparkTarget
{
    private Inkling2DSparkDrafter? _dspark;
    private int _recordedPos0;

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dspark is not null;

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dspark is not null) throw new InvalidOperationException("A DSpark drafter is already attached.");
        int visits = _c.LoopPasses > 1 ? _nPhys + (_c.LoopPasses - 1) * (_c.LoopEnd - _c.LoopStart) : _nPhys;
        var drafter = Inkling2DSparkDrafter.Open(path, _hidden, visits, _vocab, _pool);
        _dspark = drafter;
    }

    /// <summary>Layer visits per token (48 for Inkling-2-Mini: 24 physical layers plus two extra passes over the 12-layer core).</summary>
    internal int LayerVisits => _c.LoopPasses > 1 ? _nPhys + (_c.LoopPasses - 1) * (_c.LoopEnd - _c.LoopStart) : _nPhys;

    /// <summary>The attached drafter (tests: mutants and probes), or null.</summary>
    internal Inkling2DSparkDrafter? DSparkDrafter => _dspark;

    /// <summary>The model's own sequence state (the one the interface <c>Forward</c> overloads use).</summary>
    internal Inkling2SequenceState DefaultState => _defaultState;

    /// <summary>Opens a DSpark state for the attached drafter (a test handle; the engine goes through <see cref="CreateMtpState(int)"/>).</summary>
    internal Inkling2DSparkState CreateDSparkState() => new(_dspark ?? throw new InvalidOperationException("No DSpark drafter attached."), LayerVisits);

    /// <inheritdoc/>
    public bool SupportsMtp => _dspark is not null;

    /// <inheritdoc/>
    public int MtpPreferredDraftTokens => _dspark?.DraftTokensPerRound ?? 0;

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => _dspark is null ? null : CreateDSparkState();

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength) => CreateMtpState();

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dspark is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        if (state is not Inkling2DSparkState s) throw new NotSupportedException($"{nameof(Inkling2TransformerModel)} drafts only with DSpark states.");
        return s.NextDraft(this, tokenId, position, s.Drafter.DraftTokensPerRound);
    }

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        if (state is not Inkling2DSparkState s) throw new NotSupportedException($"{nameof(Inkling2TransformerModel)} drafts only with DSpark states.");
        s.NextDraft(this, tokenId, position, s.Drafter.DraftTokensPerRound);
        var row = s.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
        => ForwardWithMtp(tokenIds, positions, deviceId, mtpState, lastTokenLogitsOnly: false, recordRows: false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState,
                           bool lastTokenLogitsOnly)
        => ForwardWithMtp(tokenIds, positions, deviceId, mtpState, lastTokenLogitsOnly, recordRows: false);

    /// <inheritdoc/>
    public bool SupportsRecurrentRowSnapshots => true;

    /// <inheritdoc/>
    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, IMtpState? mtpState)
        => ForwardWithMtp(tokenIds, positions, deviceId, mtpState, lastTokenLogitsOnly: false, recordRows: true);

    /// <inheritdoc/>
    public void RestoreRecurrentStateToRow(int row)
    {
        if (Mutant == Inkling2Mutant.SkipRestore) return;
        _defaultState.RestoreToRow(row, _recordedPos0);
    }

    private ITensor ForwardWithMtp(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IMtpState? mtpState, bool lastTokenLogitsOnly, bool recordRows)
    {
        if (mtpState is not Inkling2DSparkState s)
            return ForwardCore(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly, images: null, tap: null, recordRows: recordRows);
        s.EnsureTapRows(tokenIds.Length);
        var logits = ForwardCore(tokenIds, positions, deviceId, _defaultState, lastTokenLogitsOnly, images: null, tap: s, recordRows: recordRows);
        s.InjectCaptured(positions);
        s.TrimTapRows();
        return logits;
    }

    // ── IInkling2DSparkTarget ──

    void IInkling2DSparkTarget.EmbedRaw(int token, float* dst)
    {
        if ((uint)token >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(token));
        var embType = _gguf!.TensorsByName[Inkling2Tensors.TokenEmbd].QuantizationType;
        long rowBytes = Dequantize.RowByteSize(_hidden, embType);
        Dequantize.ToFloat32(_tokenEmbed + (nint)(token * rowBytes), _hidden, embType, new Span<float>(dst, _hidden));
    }

    void IInkling2DSparkTarget.ProjectHead(float* hidden, int rows, float* logits)
        => Gemm(_output.Ptr, _output.Qt, hidden, logits, _vocab, _hidden, rows, _pool);
}
