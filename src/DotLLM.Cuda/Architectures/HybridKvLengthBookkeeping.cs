namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Host-side length bookkeeping for the CUDA hybrid models' model-owned F16 KV-cache
/// (issues #476, #478). Pure functions, so the rules are unit testable without a GPU.
/// </summary>
/// <remarks>
/// The models track the F16 cache's live length (the <c>seqKv</c> attention reads). It used to be
/// grow-only. Speculative decoding rolls the length-only <see cref="CudaHybridKvCacheHandle"/>
/// back to a committed position after a rejected round (#476), but the model never heard about it:
/// every later append converted and attended over the rejected rows until the sequence outgrew the
/// old maximum. Correctness survived (the causal mask is positional), the cost did not.
/// <see cref="SyncToCommitted"/> runs once per <c>Forward</c>, with the handle's length before this
/// call advances it — the committed prefix. The F16 cursor shrinks to it so the next forward
/// converts only the live range.
/// </remarks>
internal static class HybridKvLengthBookkeeping
{
    /// <summary>
    /// Shrinks the F16 live length to <paramref name="committedLength"/> when the caller rolled the
    /// KV-cache back below it.
    /// </summary>
    /// <param name="f16Length">The F16 cache's current live length.</param>
    /// <param name="committedLength">The KV handle's length before this forward advances it.</param>
    /// <returns>The new F16 live length.</returns>
    public static int SyncToCommitted(int f16Length, int committedLength)
    {
        if (committedLength < 0) throw new ArgumentOutOfRangeException(nameof(committedLength));
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
}
