namespace DotLLM.Core.Sampling;

/// <summary>
/// Result of one <see cref="IDiffusionUnmaskSampler"/> step: which canvas positions to commit
/// (unmask) this step, the token each commits to, and the entropy of every masked position.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="UnmaskedPositions"/> and <see cref="TokenIds"/> are parallel arrays of the same
/// length (the number of positions committed this step). A position appearing here MUST be set to
/// the corresponding token on the canvas and is thereafter frozen under the absorbing-state policy.
/// </para>
/// <para>
/// <see cref="PerPositionEntropy"/> is parallel to the sampler's input rows (one entry per
/// currently-masked position, in the same order), so the scheduler can compute the average canvas
/// entropy for its early-stop check without recomputing softmaxes.
/// </para>
/// </remarks>
public sealed record UnmaskDecision
{
    /// <summary>Canvas indices committed (unmasked) this step. Parallel to <see cref="TokenIds"/>.</summary>
    public required IReadOnlyList<int> UnmaskedPositions { get; init; }

    /// <summary>Token id sampled for each committed position. Parallel to <see cref="UnmaskedPositions"/>.</summary>
    public required IReadOnlyList<int> TokenIds { get; init; }

    /// <summary>
    /// Predictive entropy (nats) of every masked position considered this step, in the same order
    /// as the sampler's input rows. Read by the scheduler for the confidence early-stop check.
    /// </summary>
    public required IReadOnlyList<float> PerPositionEntropy { get; init; }

    /// <summary>Mean of <see cref="PerPositionEntropy"/>, or 0 when there were no masked positions.</summary>
    public float AverageEntropy
    {
        get
        {
            int n = PerPositionEntropy.Count;
            if (n == 0)
                return 0f;
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += PerPositionEntropy[i];
            return (float)(sum / n);
        }
    }
}
