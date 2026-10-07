namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses IBM Granite 3.x tool calls (#796): the model answers a tool request with the
/// <c>&lt;|tool_call|&gt;</c> special token followed by a JSON list of calls,
/// <c>&lt;|tool_call|&gt;[{"name": "get_weather", "arguments": {"city": "Paris"}}]</c>.
/// </summary>
/// <remarks>
/// Mirrors llama.cpp's <c>common_chat_params_init_granite</c> (a <c>&lt;|tool_call|&gt;</c> prefix then a JSON
/// array; the turn then ends with <c>&lt;|end_of_text|&gt;</c>). Before this parser existed Granite fell through to
/// <see cref="GenericToolCallParser"/>: the call itself parsed (the JSON is bare), but the server's streaming
/// suppressor is deliberately off for the heuristic generic parser, so <c>&lt;|tool_call|&gt;</c> and the raw JSON
/// leaked into <c>delta.content</c> as well as arriving as <c>tool_calls</c>. A marker-based parser lets the
/// suppressor hold the call back from the moment the marker appears.
/// </remarks>
public sealed class GraniteToolCallParser : IToolCallParser
{
    private const string Marker = "<|tool_call|>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        int m = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (m < 0)
            return null;

        // The model may repeat the marker per call (<|tool_call|>{...}<|tool_call|>{...}); every JSON value
        // after the first marker counts, in order.
        string after = generatedText[(m + Marker.Length)..].Replace(Marker, " ", StringComparison.Ordinal);
        return ToolCallJsonHelper.ParseAll(after, requireWholeText: false);
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);
}
