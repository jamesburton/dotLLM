using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-sequence MTP state for Gemma 4 MTP speculative decoding on the Vulkan backend (issue #959).
/// Unlike Qwen4Exp MTP, the Gemma 4 drafter has NO separate KV-cache — it reads K/V directly from the
/// trunk's <see cref="VulkanKvCache"/> via Q-only cross-attention (<c>shared_kv_layers = block_count</c>).
/// This state carries only the captured trunk hidden rows (for seeding <c>pre_projection</c>) and
/// the pending drafter hidden (256-dim) that chains between draft steps via <c>post_projection</c>.
/// </summary>
internal sealed class VulkanGemma4MtpState : IMtpState
{
    private readonly int _trunkHiddenSize;
    private readonly int _drafterHiddenSize;
    private readonly float[] _capturedRows;

    /// <summary>Maximum number of trunk hidden rows that can be captured in a single forward.</summary>
    public const int MaxCaptureRows = 32;

    private int _capturedCount;
    private int _coveredLength;

    /// <summary>Pending drafter hidden state (256-dim) — the <c>post_projection</c> output from the last draft step, seeding the next.</summary>
    internal readonly float[] PendingHidden;

    /// <summary>Pending trunk hidden state (2560-dim) — captured from the trunk, used for <c>pre_projection</c> input.</summary>
    internal readonly float[] PendingTrunkHidden;

    /// <summary>True when <see cref="PendingTrunkHidden"/> has been seeded by <see cref="SeedFromCapturedRow"/>.</summary>
    internal bool HasSeed { get; private set; }

    /// <summary>
    /// Reference to the trunk's KV-cache, set during the trunk forward (the overload with <c>mtpState</c>).
    /// The drafter's Q-only cross-attention reads K/V from this cache.
    /// </summary>
    internal VulkanKvCache? TrunkKvCache { get; set; }

    public VulkanGemma4MtpState(int trunkHiddenSize, int drafterHiddenSize)
    {
        _trunkHiddenSize = trunkHiddenSize;
        _drafterHiddenSize = drafterHiddenSize;
        _capturedRows = new float[MaxCaptureRows * trunkHiddenSize];
        PendingHidden = new float[drafterHiddenSize];
        PendingTrunkHidden = new float[trunkHiddenSize];
    }

    /// <inheritdoc/>
    public int CurrentLength => _coveredLength;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if (length < _coveredLength)
            _coveredLength = length;
    }

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => _capturedRows.AsSpan(0, _capturedCount * _trunkHiddenSize);

    /// <inheritdoc/>
    public int CapturedRowCount => _capturedCount;

    /// <inheritdoc/>
    public int HiddenSize => _trunkHiddenSize;

    /// <inheritdoc/>
    public void SeedFromCapturedRow(int rowIndex)
    {
        if ((uint)rowIndex >= (uint)_capturedCount)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"Row {rowIndex} is out of range; only {_capturedCount} rows captured.");
        _capturedRows.AsSpan(rowIndex * _trunkHiddenSize, _trunkHiddenSize)
            .CopyTo(PendingTrunkHidden);
        HasSeed = true;
    }

    /// <summary>
    /// Captures trunk hidden rows from a GPU download. Called after the trunk forward
    /// downloads the post-final-norm hidden state for each position.
    /// </summary>
    /// <param name="rows">Row-major hidden state, <c>[count, trunkHiddenSize]</c>.</param>
    /// <param name="count">Number of rows to capture (clamped to <see cref="CaptureRows"/>).</param>
    /// <param name="firstPosition">First position in the batch.</param>
    internal void CaptureRows(ReadOnlySpan<float> rows, int count, int firstPosition)
    {
        int n = Math.Min(count, MaxCaptureRows);
        int srcOffset = (count - n) * _trunkHiddenSize;
        rows.Slice(srcOffset, n * _trunkHiddenSize).CopyTo(_capturedRows);
        _capturedCount = n;
        _coveredLength = firstPosition + count;
    }

    /// <summary>
    /// Marks positions as covered without re-absorbing (the Gemma 4 drafter has no separate KV cache to fill).
    /// </summary>
    internal void SetCovered(int length) => _coveredLength = length;

    /// <summary>
    /// Updates the pending drafter hidden state with the <c>post_projection</c> output of the last draft step.
    /// </summary>
    internal void SetPendingDrafterHidden(ReadOnlySpan<float> hidden)
    {
        hidden.Slice(0, _drafterHiddenSize).CopyTo(PendingHidden);
    }

    public void Dispose() { }
}
