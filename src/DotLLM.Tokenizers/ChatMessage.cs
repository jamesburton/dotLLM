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
}
