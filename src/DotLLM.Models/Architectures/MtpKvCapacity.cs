namespace DotLLM.Models.Architectures;

/// <summary>
/// Capacity bookkeeping for the MTP head's position-indexed KV-cache (issue #966). The cache must be
/// able to cover the whole sequence (<c>ceiling</c> = the trunk KV-cache's <c>MaxLength</c>), but a
/// 64k-position F32 head cache is 512 MiB, so backends allocate <see cref="Initial"/> rows up front and
/// grow in <see cref="GrowthStep"/>-row steps as positions are actually reached. Pure arithmetic —
/// device-agnostic, shared by the CPU and CUDA states and testable without a GPU.
/// </summary>
public static class MtpKvCapacity
{
    /// <summary>Growth granularity, in sequence positions.</summary>
    public const int GrowthStep = 1024;

    /// <summary>
    /// Rows to allocate at construction: <c>min(ceiling, GrowthStep)</c>. Never more than the eager
    /// allocation the state used before (<c>ceiling</c> rows), so short contexts do not grow.
    /// </summary>
    public static int Initial(int ceiling)
    {
        if (ceiling <= 0) throw new ArgumentOutOfRangeException(nameof(ceiling));
        return Math.Min(ceiling, GrowthStep);
    }

    /// <summary>
    /// Capacity required to hold <paramref name="needed"/> rows given the current allocation:
    /// <paramref name="current"/> when it already suffices, otherwise <paramref name="needed"/> rounded
    /// up to a multiple of <see cref="GrowthStep"/> and clamped to <paramref name="ceiling"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException"><paramref name="needed"/> exceeds <paramref name="ceiling"/>.</exception>
    public static int Next(int current, long needed, int ceiling)
    {
        if (needed > ceiling)
            throw new InvalidOperationException(
                $"MTP KV-cache exhausted: {needed} positions needed against a ceiling of {ceiling}. " +
                "Create the state with CreateMtpState(maxSequenceLength) covering the whole sequence.");
        if (needed <= current)
            return current;
        long rounded = (needed + GrowthStep - 1) / GrowthStep * GrowthStep;
        return (int)Math.Min(rounded, ceiling);
    }
}
