using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server;

/// <summary>
/// Validates incoming inference requests and clamps parameters to model limits.
/// </summary>
public static class RequestValidator
{
    /// <summary>Default maximum number of messages in a chat request (see <see cref="ServerOptions.MaxMessages"/>).</summary>
    public const int DefaultMaxMessages = ServerOptions.DefaultMaxMessages;

    /// <summary>
    /// Builds the "too many messages" error tail naming the effective limit and how to change it.
    /// Returns null when <paramref name="count"/> is within <paramref name="maxMessages"/> (0 = unlimited).
    /// </summary>
    public static string? CheckMessageCount(int count, int maxMessages) =>
        maxMessages > 0 && count > maxMessages
            ? $"exceeds maximum of {maxMessages} messages (got {count}); raise it with --max-messages or " +
              $"{ServerOptions.MaxMessagesEnvVar} (0 = unlimited)"
            : null;

    /// <summary>
    /// Validates a chat completion request before inference.
    /// Returns an error message if invalid, or null if valid.
    /// </summary>
    /// <param name="request">The request.</param>
    /// <param name="maxMessages">Message-count cap; 0 = unlimited. Pass <see cref="ServerOptions.EffectiveMaxMessages"/>.</param>
    public static string? ValidateChatRequest(ChatCompletionRequest request, int maxMessages = DefaultMaxMessages)
    {
        if (request.Messages is null || request.Messages.Length == 0)
            return "messages array must not be empty";

        if (CheckMessageCount(request.Messages.Length, maxMessages) is { } tooMany)
            return "messages array " + tooMany;

        if (request.MaxTokens.HasValue && request.MaxTokens.Value <= 0)
            return "max_tokens must be a positive integer";

        // n used to be accepted and never read, so n:3 silently returned one choice (#460).
        // Range-check it here so an unsupportable value is refused rather than quietly reinterpreted.
        if (request.N.HasValue)
        {
            if (request.N.Value < 1)
                return "n must be a positive integer";
            if (request.N.Value > MaxChoices)
                return $"n exceeds the maximum of {MaxChoices} supported by this server";
        }

        return null;
    }

    /// <summary>
    /// Upper bound on <c>n</c>. Each choice is an independent generation over the same prompt, so
    /// the cost is linear in <c>n</c>; the cap keeps one request from monopolising the batch.
    /// </summary>
    public const int MaxChoices = 8;

    /// <summary>
    /// Validates a raw completion request before inference.
    /// Returns an error message if invalid, or null if valid.
    /// </summary>
    public static string? ValidateCompletionRequest(CompletionRequest request)
    {
        if (string.IsNullOrEmpty(request.Prompt))
            return "prompt must not be empty";

        if (request.MaxTokens.HasValue && request.MaxTokens.Value <= 0)
            return "max_tokens must be a positive integer";

        return null;
    }

    /// <summary>
    /// Validates prompt length against the model's context window and clamps max_tokens.
    /// Returns an error message if the prompt alone exceeds context, or null if valid.
    /// When valid, <paramref name="effectiveMaxTokens"/> is clamped to remaining context.
    /// </summary>
    public static string? ValidatePromptLength(
        string prompt, ITokenizer tokenizer, int maxSequenceLength,
        int requestedMaxTokens, out int effectiveMaxTokens, out int promptTokenCount)
    {
        promptTokenCount = tokenizer.CountTokens(prompt);

        if (promptTokenCount >= maxSequenceLength)
        {
            effectiveMaxTokens = 0;
            return $"prompt ({promptTokenCount} tokens) exceeds model context length ({maxSequenceLength})";
        }

        int remaining = maxSequenceLength - promptTokenCount;
        effectiveMaxTokens = Math.Min(requestedMaxTokens, remaining);
        return null;
    }
}
