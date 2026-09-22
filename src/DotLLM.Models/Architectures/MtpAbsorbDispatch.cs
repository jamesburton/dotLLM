namespace DotLLM.Models.Architectures;

/// <summary>
/// Selects how an MTP-carrying trunk <c>Forward</c> absorbs its batch into the MTP head's KV-cache
/// (issue #472): one batched, KV-only head pass over all S rows (the default), or the #469
/// reference loop that runs the full single-token head step once per token.
/// </summary>
/// <remarks>
/// <para>
/// An absorb only needs the K/V rows the head writes at each position. Everything downstream of the
/// K/V write in a head step — attention, the gate, the O-projection, both residual adds, the whole
/// FFN and the LM head — produces a hidden state the absorb discards: the next draft step is seeded
/// from a trunk row, never from an absorbed step's output. The batched path therefore computes only
/// <c>embed → enorm/hnorm → eh_proj → attn_norm → K/V projections → K-norm → RoPE</c> for S rows at
/// once and writes them to slots <c>[positions[0], positions[0] + S)</c>.
/// </para>
/// <para>
/// <c>DOTLLM_MTP_ABSORB_PER_TOKEN=1</c> restores the per-token loop. Tests and in-process A/B
/// benchmarks set <see cref="PerTokenOverride"/> instead.
/// </para>
/// </remarks>
internal static class MtpAbsorbDispatch
{
    private static readonly bool PerTokenFromEnv =
        Environment.GetEnvironmentVariable("DOTLLM_MTP_ABSORB_PER_TOKEN") is "1" or "true" or "TRUE";

    /// <summary>
    /// When set, overrides the environment on the CURRENT thread: <see langword="true"/> forces the
    /// per-token loop. Thread-static so parallel test classes cannot flip each other's arm — the
    /// absorb runs synchronously on the thread that called <c>Forward</c>.
    /// </summary>
    [ThreadStatic]
    internal static bool? PerTokenOverride;

    /// <summary><see langword="true"/> when the absorb should use the #469 per-token loop.</summary>
    internal static bool UsePerToken => PerTokenOverride ?? PerTokenFromEnv;

    /// <summary>
    /// <see langword="true"/> when <paramref name="positions"/> is one contiguous ascending run — the
    /// only shape the batched path writes (a single slab of KV slots). Prefill and verify batches
    /// always are; anything else falls back to the per-token loop.
    /// </summary>
    internal static bool IsContiguous(ReadOnlySpan<int> positions)
    {
        for (int i = 1; i < positions.Length; i++)
            if (positions[i] != positions[0] + i)
                return false;
        return true;
    }
}
