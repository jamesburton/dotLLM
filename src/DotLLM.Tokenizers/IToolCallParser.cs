namespace DotLLM.Tokenizers;

/// <summary>
/// Parses tool call invocations from model-generated text.
/// </summary>
public interface IToolCallParser
{
    /// <summary>
    /// Attempts to parse tool calls from the generated text.
    /// </summary>
    /// <param name="generatedText">Model output text.</param>
    /// <returns>Parsed tool calls, or null if no tool calls were found.</returns>
    ToolCall[]? TryParse(string generatedText);

    /// <summary>
    /// Attempts to parse tool calls and coerces each call's arguments to the types declared by the
    /// matching entry of <paramref name="tools"/> (see <see cref="ToolCallParsers.ToolArgumentCoercer"/>).
    /// Parsers whose wire format is untyped (the XML family) override this to use the schema while parsing.
    /// </summary>
    /// <param name="generatedText">Model output text.</param>
    /// <param name="tools">The request's tool definitions, or null when unknown.</param>
    /// <returns>Parsed, schema-coerced tool calls, or null if no tool calls were found.</returns>
    ToolCall[]? TryParse(string generatedText, IReadOnlyList<ToolDefinition>? tools)
    {
        var calls = TryParse(generatedText);
        return calls is null ? null : ToolCallParsers.ToolArgumentCoercer.Coerce(calls, tools);
    }

    /// <summary>
    /// Checks whether the text begins with a tool call marker.
    /// Used during streaming to detect partial tool calls early.
    /// </summary>
    /// <param name="text">Text to check.</param>
    /// <returns>True if the text appears to start a tool call.</returns>
    bool IsToolCallStart(string text);
}
