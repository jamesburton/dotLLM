using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Result of a completed inference request.
/// </summary>
public record InferenceResponse
{
    /// <summary>Generated token IDs.</summary>
    public required int[] GeneratedTokenIds { get; init; }

    /// <summary>Decoded output text.</summary>
    public required string Text { get; init; }

    /// <summary>Reason generation stopped.</summary>
    public required FinishReason FinishReason { get; init; }

    /// <summary>Number of prompt tokens processed.</summary>
    public int PromptTokenCount { get; init; }

    /// <summary>Number of tokens generated.</summary>
    public int GeneratedTokenCount { get; init; }

    /// <summary>Timing measurements from the inference run.</summary>
    public InferenceTimings Timings { get; init; }

    /// <summary>Per-token log-probability info for each generated token. Null when logprobs not requested.</summary>
    public TokenLogprobInfo[]? Logprobs { get; init; }

    /// <summary>Parsed tool calls from the generated text. Null if no tool calls were detected.</summary>
    public ToolCall[]? ToolCalls { get; init; }

    /// <summary>
    /// The stop string that ended generation, or <see langword="null"/> when generation ended for
    /// any other reason (EOS, max-tokens, cancellation).
    /// </summary>
    /// <remarks>
    /// Reported explicitly because <see cref="Text"/> has already had the match trimmed off, so a
    /// caller cannot recover it by testing the text — which is precisely what the Anthropic
    /// <c>stop_sequence</c> mapping used to do, silently and correctly, back when stop strings
    /// never fired at all (#459).
    /// </remarks>
    public string? MatchedStopSequence { get; init; }
}
