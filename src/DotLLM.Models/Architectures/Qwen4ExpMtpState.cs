using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence state of the qwen4exp MTP draft head (issue #820): the head layer's dense K/V cells, the captured trunk residual
/// rows of the last forward, and the "pending" residual that seeds the next draft step.
/// </summary>
/// <remarks>
/// <para><b>Rows are the 4-stream residual.</b> The head reads the trunk's residual BEFORE the head mixer, so a captured row is
/// <c>hc * hidden</c> floats wide (<see cref="HiddenSize"/> reports that width), not the mixed hidden state the dense hybrids hand over.</para>
/// <para><b>Cell numbering.</b> The head predicts the token at position <c>p + 1</c> from the pair <c>(R_{p-1}, token_p)</c> (the trunk
/// residual that PREDICTED token <c>p</c>, plus token <c>p</c>; the #469 pairing). Strata's "cell i = (R_i, token_{i+1})" is the same
/// thing shifted by one, and it is the numbering used here for the K/V cells and their RoPE position: cell <c>c = p - 1</c>. Token 0 has
/// no preceding residual, so it owns no cell and a sequence of <c>n</c> absorbed tokens holds <c>max(n - 1, 0)</c> cells.
/// <see cref="CurrentLength"/> counts tokens (positions), like every <see cref="IMtpState"/>.</para>
/// <para>Attention is dense over all cells (no indexer): exact below 2051 positions, a draft-only approximation beyond, and never a
/// correctness matter because the verify forward decides every emitted token.</para>
/// </remarks>
public sealed class Qwen4ExpMtpState : IMtpState
{
    private readonly int _rowWidth;
    private readonly int _maxLength;
    private readonly float[] _pending;
    private readonly float[] _carry;
    private float[] _captured = [];
    private int _capturedCount;
    private int _covered;

    internal Qwen4ExpDenseKv Kv { get; }

    /// <summary>Creates an empty state.</summary>
    /// <param name="rowWidth">Residual row width, <c>hcCount * hiddenSize</c>.</param>
    /// <param name="kvStride">KV row width of the head's attention layer.</param>
    /// <param name="maxLength">Longest sequence (positions) the state must cover.</param>
    public Qwen4ExpMtpState(int rowWidth, int kvStride, int maxLength)
    {
        if (rowWidth <= 0) throw new ArgumentOutOfRangeException(nameof(rowWidth));
        _rowWidth = rowWidth;
        _maxLength = maxLength;
        _pending = new float[rowWidth];
        _carry = new float[rowWidth];
        Kv = new Qwen4ExpDenseKv(kvStride);
    }

    /// <inheritdoc/>
    public int CurrentLength => _covered;

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => _captured.AsSpan(0, _capturedCount * _rowWidth);

    /// <inheritdoc/>
    public int CapturedRowCount => _capturedCount;

    /// <inheritdoc/>
    public int HiddenSize => _rowWidth;

    /// <summary>Head cells held (<c>max(CurrentLength - 1, 0)</c>).</summary>
    public int CellCount => Kv.Length;

    /// <summary>Resident bytes of the head's K/V cells.</summary>
    public long Bytes => Kv.Bytes + (_captured.Length + _pending.Length + _carry.Length) * sizeof(float);

    internal ReadOnlySpan<float> Pending => _pending;
    internal ReadOnlySpan<float> Carry => _carry;
    internal Span<float> PendingMutable => _pending;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if (length < 0 || length > _covered) throw new ArgumentOutOfRangeException(nameof(length));
        _covered = length;
        Kv.Truncate(Math.Max(length - 1, 0));
    }

    internal void SetCapturedRows(ReadOnlySpan<float> rows, int rowCount)
    {
        int needed = rowCount * _rowWidth;
        if (_captured.Length < needed) _captured = new float[needed];
        rows.Slice(0, needed).CopyTo(_captured);
        _capturedCount = rowCount;
    }

    /// <summary>Prepares an absorb of <paramref name="count"/> contiguous tokens from <paramref name="firstPosition"/>: drops speculative cells past it, rejects a gap.</summary>
    internal void BeginAbsorb(int firstPosition, int count)
    {
        if (_covered > firstPosition) Rollback(firstPosition);
        else if (_covered < firstPosition)
            throw new InvalidOperationException(
                $"MTP absorb at position {firstPosition} but the head only covers {_covered} positions. Every trunk forward of the " +
                "sequence must pass the MTP state so the head absorbs it (prefill included).");
        if ((long)firstPosition + count > _maxLength)
            throw new InvalidOperationException(
                $"Qwen4ExpMtpState covers {_maxLength} positions; absorbing [{firstPosition}, {firstPosition + count}) exceeds it. " +
                "Create the state with CreateMtpState(maxSequenceLength) covering the whole sequence.");
    }

    internal void EndAbsorb(int length) => _covered = length;

    /// <summary>Copies the residual row token <paramref name="i"/> of the captured batch pairs with: the carried row for <c>i == 0</c>, else captured row <c>i - 1</c>.</summary>
    internal void CopyPairingRow(int i, Span<float> dest)
    {
        if (i == 0) _carry.CopyTo(dest);
        else _captured.AsSpan((i - 1) * _rowWidth, _rowWidth).CopyTo(dest);
    }

    /// <summary>Marks position <paramref name="position"/> covered after a speculative draft step wrote its cell.</summary>
    internal void MarkCovered(int position) => _covered = position + 1;

    /// <inheritdoc/>
    public void SeedFromCapturedRow(int rowIndex)
    {
        if ((uint)rowIndex >= (uint)_capturedCount)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"rowIndex {rowIndex} out of range [0, {_capturedCount}): CapturedHiddenRows was not populated by a trunk forward carrying this state.");
        var row = _captured.AsSpan(rowIndex * _rowWidth, _rowWidth);
        row.CopyTo(_pending);
        row.CopyTo(_carry);
    }

    /// <inheritdoc/>
    public void Dispose() => Kv.Dispose();
}
