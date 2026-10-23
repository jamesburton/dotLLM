using DotLLM.Tokenizers;

namespace DotLLM.Server;

/// <summary>Applies the active model profile's system prompt to a chat request (#716).</summary>
public static class ProfileSystemPrompt
{
    /// <summary>
    /// Prepends the profile's system prompt when there is one and the request carries no system message of its own; a request that brings its
    /// own system message always wins. Returns the input unchanged otherwise.
    /// </summary>
    public static ChatMessage[] Apply(string? profileSystem, ChatMessage[] messages)
    {
        if (string.IsNullOrWhiteSpace(profileSystem)) return messages;
        foreach (var m in messages)
            if (string.Equals(m.Role, "system", StringComparison.OrdinalIgnoreCase)) return messages;
        var result = new ChatMessage[messages.Length + 1];
        result[0] = new ChatMessage { Role = "system", Content = profileSystem };
        Array.Copy(messages, 0, result, 1, messages.Length);
        return result;
    }
}
