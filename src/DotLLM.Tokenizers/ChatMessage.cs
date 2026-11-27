namespace DotLLM.Tokenizers;

/// <summary>
/// A single message in a chat conversation.
/// </summary>
public record ChatMessage
{
    /// <summary>Role of the message sender: "system", "user", "assistant", or "tool".</summary>
    public required string Role { get; init; }

    /// <summary>Text content of the message.</summary>
    public required string Content { get; init; }

    /// <summary>Tool calls made by the assistant. Null if not a tool-calling response.</summary>
    public ToolCall[]? ToolCalls { get; init; }

    /// <summary>ID of the tool call this message is responding to. Null if not a tool result.</summary>
    public string? ToolCallId { get; init; }

    /// <summary>
    /// Reasoning ("thinking") the assistant produced before <see cref="Content"/>. Exposed to the chat
    /// template as <c>message.reasoning_content</c>; templates that render history reasoning
    /// (Qwen3.x <c>preserve_thinking</c>) use it. Null when the turn carried none.
    /// </summary>
    public string? ReasoningContent { get; init; }

    /// <summary>
    /// Structured content (OpenAI "content parts": text and images, in order) for multimodal turns (issue #2035). When set, the chat template sees
    /// <c>message.content</c> as the list of <c>{type, text}</c> / <c>{type: "image_url", image_url: {url}}</c> objects instead of a string, and
    /// <see cref="Content"/> should carry the concatenated text (used by code that only handles plain text). Null for ordinary text messages.
    /// </summary>
    public IReadOnlyList<ChatContentPart>? ContentParts { get; init; }
}

/// <summary>One part of a multimodal <see cref="ChatMessage"/>: <c>Type</c> is <c>"text"</c> (uses <see cref="Text"/>) or <c>"image_url"</c> (uses <see cref="ImageUrl"/>).</summary>
public sealed record ChatContentPart
{
    /// <summary><c>"text"</c> or <c>"image_url"</c>.</summary>
    public required string Type { get; init; }

    /// <summary>The text of a <c>"text"</c> part.</summary>
    public string? Text { get; init; }

    /// <summary>The URL of an <c>"image_url"</c> part (a <c>data:</c> URI, or any string the caller resolves itself; templates only test the part type).</summary>
    public string? ImageUrl { get; init; }
}
