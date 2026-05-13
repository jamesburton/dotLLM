namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls from the Hermes-compatible XML-wrapped JSON format
/// emitted by SmolLM3 and the Qwen3 / NousResearch Hermes family:
/// <c>&lt;tool_call&gt;{"name": "...", "arguments": {...}}&lt;/tool_call&gt;</c>.
/// </summary>
/// <remarks>
/// <para>
/// Bytewise identical to <see cref="HermesToolCallParser"/> at the parse
/// level — Hermes itself ships this exact tag pair, and SmolLM3's
/// <c>xml_tools</c> chat-template branch produces the same wire format.
/// Surfacing it under the architecture-neutral name <c>XmlToolCallParser</c>
/// matches the SmolLM3 roadmap entry (step 56) and lets the factory dispatch
/// against the XML-vs-Pythonic format choice without coupling to the
/// originating model family.
/// </para>
/// <para>
/// Streaming detection (<see cref="IsToolCallStart"/>) checks for the
/// opening <c>&lt;tool_call&gt;</c> tag verbatim.
/// </para>
/// </remarks>
public sealed class XmlToolCallParser : IToolCallParser
{
    private readonly HermesToolCallParser _inner = new();

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText) => _inner.TryParse(generatedText);

    /// <inheritdoc/>
    public bool IsToolCallStart(string text) => _inner.IsToolCallStart(text);
}
