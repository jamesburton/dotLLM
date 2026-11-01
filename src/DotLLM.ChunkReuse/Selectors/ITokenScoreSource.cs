namespace DotLLM.ChunkReuse.Selectors;

/// <summary>A per-token importance score over a context (higher = recompute first).</summary>
public interface ITokenScoreSource
{
    /// <summary>One finite score per context token, <c>[context.Length]</c>.</summary>
    /// <param name="context">The context.</param>
    float[] Score(ChunkedContext context);
}
