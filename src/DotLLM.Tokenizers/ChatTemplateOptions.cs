using System.Text.Json;

namespace DotLLM.Tokenizers;

/// <summary>
/// Options for applying a chat template.
/// </summary>
/// <remarks>
/// Every reasoning-related member is nullable on purpose: a <see langword="null"/> value means
/// "the client did not say", and the corresponding Jinja variable is then left <b>undefined</b> so the
/// template applies its own default (e.g. Qwen3.x tests <c>enable_thinking is undefined</c>). A
/// variable that is present-but-null is not undefined, so nothing here is ever inserted as null.
/// </remarks>
public record ChatTemplateOptions
{
    /// <summary>Whether to append the assistant turn prefix for generation.</summary>
    public bool AddGenerationPrompt { get; init; } = true;

    /// <summary>Tool definitions available to the model. Null if tool calling is not enabled.</summary>
    public ToolDefinition[]? Tools { get; init; }

    /// <summary>
    /// Exposed to the template as <c>enable_thinking</c> (Qwen3.x / Bonsai / GLM style switch).
    /// Null leaves the variable undefined so the template default applies.
    /// </summary>
    public bool? EnableThinking { get; init; }

    /// <summary>
    /// Exposed to the template as <c>reasoning_effort</c> (e.g. <c>low</c>/<c>medium</c>/<c>xhigh</c> for
    /// Qwen3.x). Null leaves it undefined. Templates may reject values they do not know.
    /// </summary>
    public string? ReasoningEffort { get; init; }

    /// <summary>
    /// Exposed to the template as <c>preserve_thinking</c>: whether earlier assistant turns keep their
    /// reasoning in the rendered history. Null leaves it undefined (the template default).
    /// </summary>
    public bool? PreserveThinking { get; init; }

    /// <summary>
    /// Arbitrary extra variables merged into the Jinja context (OpenAI-compatible
    /// <c>chat_template_kwargs</c>). Applied first, so the explicit members above win over a
    /// same-named kwarg, and the structural variables (<c>messages</c>, <c>tools</c>,
    /// <c>add_generation_prompt</c>, <c>bos_token</c>, <c>eos_token</c>) can never be overridden.
    /// </summary>
    public IReadOnlyDictionary<string, JsonElement>? TemplateKwargs { get; init; }
}
