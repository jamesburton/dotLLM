namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Fallback tool call parser that detects bare JSON with <c>name</c> and
/// <c>arguments</c>/<c>parameters</c> fields. No special markers required.
/// </summary>
public sealed class GenericToolCallParser : IToolCallParser
{
    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
        => ToolCallJsonHelper.ExtractAndParse(generatedText, "call");

    /// <inheritdoc/>
    /// <remarks>A bare JSON call opens with <c>{</c>. Prose that merely contains a brace is delayed, not lost: the streaming
    /// endpoints deliver held text that never parses into a call at the end of the turn.</remarks>
    public IReadOnlyList<string> StartMarkers { get; } = ["{"];

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
    {
        // Heuristic: text contains a JSON-like pattern with "name" key
        int braceIndex = text.IndexOf('{');
        if (braceIndex < 0)
            return false;

        // Check if there's a "name" key after the brace
        int nameIndex = text.IndexOf("\"name\"", braceIndex, StringComparison.Ordinal);
        return nameIndex > braceIndex;
    }
}
