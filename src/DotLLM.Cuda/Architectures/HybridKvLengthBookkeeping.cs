namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Host-side length bookkeeping for the CUDA hybrid models' model-owned F16 KV-cache and its
/// per-slot F32 read-staging mirrors (issues #182, #478). Pure functions, so the rules are unit
/// testable without a GPU.
/// </summary>
/// <remarks>
/// <para>
/// The models track two lengths: the F16 cache's live length (the <c>seqKv</c> attention reads) and,
/// per attention slot, how many leading rows of the F32 staging copy are already converted. Both
/// used to be grow-only. Speculative decoding rolls the length-only <see cref="CudaHybridKvCacheHandle"/>
/// back to a committed position after a rejected round (#476), but the model never heard about it:
/// every later append landed below the recorded valid length and took the full-range F16→F32
/// reconversion until the sequence outgrew the old maximum. Correctness survived (the causal mask
/// works by position), the cost did not.
/// </para>
/// <para>
/// <see cref="SyncToCommitted"/> runs once per <c>Forward</c>, with the handle's length before this
/// call advances it — the committed prefix. Rows at or beyond it are either about to be rewritten
/// or are rejected speculation, so both lengths shrink to it. F32 rows below it still mirror the
/// F16 rows exactly: every F16 write is converted for its slot in the same call.
/// </para>
/// </remarks>
internal static class HybridKvLengthBookkeeping
{
    /// <summary>
    /// Shrinks the F16 live length and every per-slot F32 valid length to
    /// <paramref name="committedLength"/> when the caller rolled the KV-cache back below them.
    /// </summary>
    /// <param name="f16Length">The F16 cache's current live length.</param>
    /// <param name="validLengths">Per-slot F32 staging valid lengths (mutated in place); may be empty.</param>
    /// <param name="committedLength">The KV handle's length before this forward advances it.</param>
    /// <returns>The new F16 live length.</returns>
    public static int SyncToCommitted(int f16Length, Span<int> validLengths, int committedLength)
    {
        if (committedLength < 0) throw new ArgumentOutOfRangeException(nameof(committedLength));
        for (int i = 0; i < validLengths.Length; i++)
            if (validLengths[i] > committedLength)
                validLengths[i] = committedLength;
        return Math.Min(f16Length, committedLength);
    }

    /// <summary>
    /// The F16 live length after writing rows at <paramref name="positions"/>: grows to cover the
    /// highest written position, never shrinks (shrinking is <see cref="SyncToCommitted"/>'s job).
    /// </summary>
    public static int LengthAfterWrite(int f16Length, ReadOnlySpan<int> positions)
    {
        int maxPos = -1;
        for (int i = 0; i < positions.Length; i++)
            if (positions[i] > maxPos) maxPos = positions[i];
        return Math.Max(f16Length, maxPos + 1);
    }

    /// <summary>
    /// <see langword="true"/> when a slot's F32 staging can be extended by converting only the rows
    /// just written (issue #182): the batch is one contiguous ascending run that starts exactly at
    /// the slot's valid length. Anything else takes the always-correct full-range reconversion.
    /// </summary>
    public static bool IsIncrementalAppend(ReadOnlySpan<int> positions, int prevValidLength, bool forceFull)
    {
        if (forceFull || positions.Length == 0 || positions[0] != prevValidLength)
            return false;
        for (int i = 1; i < positions.Length; i++)
            if (positions[i] != positions[i - 1] + 1)
                return false;
        return true;
    }
}
