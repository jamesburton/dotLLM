namespace DotLLM.Engine;

/// <summary>
/// Outcome of one <see cref="GenerationSession.GenerateUntilBoundaryAsync"/> segment.
/// </summary>
/// <param name="Text">Decoded text of <paramref name="TokenIds"/>, with a matched stop-string suffix trimmed (as <see cref="TextGenerator"/> does).</param>
/// <param name="TokenIds">The tokens this segment committed to the session (a stop-string match keeps its tokens; an end-of-generation / <c>Stop</c> token is not committed).</param>
/// <param name="FinishReason"><see cref="FinishReason.Stop"/> for a <c>Stop</c> result or end-of-generation token; <see cref="FinishReason.Length"/> for a <c>StopInclude</c> result, the per-segment token limit or a full KV-cache (the same mapping <see cref="TextGenerator"/> uses).</param>
/// <param name="StartPosition">The session position when the segment began.</param>
/// <param name="EndPosition">The session position when it ended; KV length and next-token logits are valid here.</param>
public sealed record GenerationSegmentResult(
    string Text,
    IReadOnlyList<int> TokenIds,
    FinishReason FinishReason,
    int StartPosition,
    int EndPosition);
