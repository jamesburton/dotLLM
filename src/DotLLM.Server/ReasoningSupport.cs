using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;

namespace DotLLM.Server;

/// <summary>
/// The per-request reasoning decision (#767): whether this generation's output is split into
/// <c>reasoning_content</c> + <c>content</c>, and with which splitter.
/// </summary>
internal sealed class ReasoningPlan
{
    /// <summary>No splitting: raw output is the answer.</summary>
    public static readonly ReasoningPlan Disabled = new(ReasoningFormat.None, promptOpened: false);

    private readonly ReasoningFormat _format;

    internal ReasoningPlan(ReasoningFormat format, bool promptOpened)
    {
        _format = format;
        PromptOpened = promptOpened;
    }

    /// <summary>True when output is split.</summary>
    public bool Enabled => _format != ReasoningFormat.None;

    /// <summary>True when the rendered prompt already ended inside an open think block.</summary>
    public bool PromptOpened { get; }

    /// <summary>A fresh splitter for one generated sequence, or null when <see cref="Enabled"/> is false.</summary>
    public ReasoningSplitter? NewSplitter()
        => Enabled ? new ReasoningSplitter(PromptOpened, detectAnywhere: _format == ReasoningFormat.Deepseek) : null;

    /// <summary>
    /// Applies the stop-sequence gate: when the prompt opened a think block, stop strings (other than the
    /// template's control tokens) arm only after <c>&lt;/think&gt;</c>, so they cannot cut the thinking short.
    /// </summary>
    public InferenceOptions Gate(InferenceOptions options, IReadOnlyList<string> ungatedStops)
        => Enabled && PromptOpened
            ? options with { StopSequencesArmedAfter = ReasoningFormats.CloseTag, StopSequencesUngated = ungatedStops }
            : options;

    /// <summary>
    /// Splits one complete generation. The token count is 0 when there was no reasoning.
    /// </summary>
    public (string? Reasoning, string Content, int ReasoningTokens) SplitComplete(string text, ITokenizer? tokenizer)
    {
        if (!Enabled)
            return (null, text, 0);

        var (reasoning, content, state) = ReasoningSplitter.Split(text, PromptOpened, _format == ReasoningFormat.Deepseek);
        if (!state.SawReasoning)
            return (null, content, 0);

        int tokens = 0;
        if (tokenizer is not null && state.ReasoningRawLength > 0)
            tokens = tokenizer.CountTokens(text.Substring(0, (int)Math.Min(state.ReasoningRawLength, text.Length)));
        return (reasoning.Length == 0 ? null : reasoning, content, tokens);
    }
}

/// <summary>Request-side helpers shared by the OpenAI, Anthropic and Ollama surfaces (#767).</summary>
internal static class ReasoningSupport
{
    /// <summary>Stop strings that are template control tokens and must end the turn even inside reasoning.</summary>
    internal static readonly string[] UngatedStops =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>"];

    /// <summary>
    /// Builds the template options for a request. <paramref name="constrained"/> (a <c>response_format</c>
    /// / grammar / forced <c>tool_choice</c> will constrain decoding from the first token) defaults
    /// thinking OFF unless the client asked for it explicitly: a constraint cannot coexist with an open
    /// reasoning block.
    /// </summary>
    internal static ChatTemplateOptions BuildTemplateOptions(
        ToolDefinition[]? tools,
        bool? enableThinking,
        string? reasoningEffort,
        IReadOnlyDictionary<string, JsonElement>? kwargs,
        bool constrained)
    {
        bool? thinking = enableThinking;
        if (thinking is null && kwargs is not null
            && kwargs.TryGetValue("enable_thinking", out var kw)
            && kw.ValueKind is JsonValueKind.True or JsonValueKind.False)
        {
            thinking = kw.GetBoolean();
        }

        string? effort = reasoningEffort;
        if (string.Equals(effort, "none", StringComparison.OrdinalIgnoreCase))
        {
            thinking ??= false;     // OpenAI's "do not reason"
            effort = null;          // and the template has no such level
        }

        if (thinking is null && constrained)
            thinking = false;

        return new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = tools,
            EnableThinking = thinking,
            ReasoningEffort = effort,
            TemplateKwargs = kwargs,
        };
    }

    /// <summary>
    /// Applies the template, turning a template-raised error (e.g. an unsupported <c>reasoning_effort</c>)
    /// into a client-facing message instead of a 500.
    /// </summary>
    internal static bool TryApply(
        IChatTemplate template, IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options,
        out string prompt, out string? error, out string? param)
    {
        try
        {
            prompt = template.Apply(messages, options);
            error = null;
            param = null;
            return true;
        }
        catch (JinjaException ex)
        {
            prompt = "";
            error = ex.Message;
            param = options.ReasoningEffort is not null && ex.Message.Contains("reasoning", StringComparison.OrdinalIgnoreCase)
                ? "reasoning_effort"
                : "chat_template_kwargs";
            return false;
        }
    }

    /// <summary>Whether tool_choice will install a decoding constraint (mirrors <c>ToolChoiceBinder.Apply</c>).</summary>
    internal static bool ToolChoiceConstrains(
        ToolChoice toolChoice, ToolDefinition[]? tools, IToolCallParser? modelParser)
        => tools is { Length: > 0 } && modelParser is not null
           && toolChoice switch
           {
               ToolChoice.Required => true,
               ToolChoice.Function fn => Array.Exists(tools, t => t.Name == fn.Name),
               _ => false,
           };

    /// <summary>
    /// Resolves the effective format and builds the plan for a rendered prompt. A constrained request, or
    /// one whose template did not actually open a block, still splits a model-emitted <c>&lt;think&gt;</c>
    /// only when not constrained: constrained output is the answer from its first token.
    /// </summary>
    internal static ReasoningPlan Plan(ReasoningFormat serverFormat, string? requestFormat, bool constrained, string prompt, out string? error)
    {
        error = null;
        var format = serverFormat;
        if (requestFormat is not null)
        {
            if (!ReasoningFormats.TryParse(requestFormat, out format))
            {
                error = $"reasoning_format: unknown value '{requestFormat}'. Expected: none, auto, deepseek.";
                return ReasoningPlan.Disabled;
            }
        }

        if (constrained || format == ReasoningFormat.None)
            return ReasoningPlan.Disabled;
        return new ReasoningPlan(format, ReasoningFormats.PromptOpensThinking(prompt));
    }

    /// <summary>
    /// Counts one streamed token toward reasoning: true when the splitter was inside a block before the
    /// token or is inside one after it (so both the opening and the closing tag tokens count).
    /// </summary>
    internal static bool IsReasoningToken(bool wasInReasoning, ReasoningSplitter splitter)
        => wasInReasoning || splitter.InReasoning;

    /// <summary>Usage details, or null when no reasoning was produced.</summary>
    internal static Models.CompletionTokensDetailsDto? Details(int reasoningTokens)
        => reasoningTokens > 0 ? new Models.CompletionTokensDetailsDto { ReasoningTokens = reasoningTokens } : null;
}
