namespace DotLLM.Tokenizers.Reasoning;

/// <summary>
/// Incremental splitter of model output into reasoning and answer. Implemented by the single-block tag
/// splitter (<see cref="ReasoningSplitter"/>: <c>&lt;think&gt;</c>, Gemma-4 channel) and by
/// <see cref="HarmonySplitter"/> (gpt-oss channel messages). Streaming and one-shot use are the same machine:
/// <see cref="Feed"/> pieces, then <see cref="Finish"/>; the concatenation of the streamed chunks must equal
/// the one-shot result.
/// </summary>
public interface IReasoningSplitter
{
    /// <summary>True while the splitter is inside a reasoning block (as of the last <see cref="Feed"/>).</summary>
    bool InReasoning { get; }

    /// <summary>True if any text has been, or is being, treated as reasoning.</summary>
    bool SawReasoning { get; }

    /// <summary>
    /// Number of raw characters fed that belong to the reasoning part including its closing tag (all of them
    /// when it was never closed); <c>-1</c> if there was no reasoning or it is still open mid-stream.
    /// </summary>
    long ReasoningRawLength { get; }

    /// <summary>Feeds the next piece of decoded text and returns whatever is now safe to emit.</summary>
    ReasoningChunk Feed(string text);

    /// <summary>Signals end of generation and returns everything still held back.</summary>
    ReasoningChunk Finish();
}
