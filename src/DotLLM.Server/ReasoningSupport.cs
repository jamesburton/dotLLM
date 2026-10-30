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

    internal ReasoningPlan(ReasoningFormat format, bool promptOpened, ReasoningMarkup markup = ReasoningMarkup.Think)
    {
        _format = format;
        PromptOpened = promptOpened;
        Markup = markup;
    }

    /// <summary>The model's reasoning markup (<c>&lt;think&gt;</c>, Gemma-4 channel, Harmony). Set even when splitting is off.</summary>
    public ReasoningMarkup Markup { get; }

    /// <summary>
    /// Drops the stop strings this model's markup forbids (Harmony's <c>&lt;|end|&gt;</c> closes the analysis
    /// message, it does not end the turn). Applies even when reasoning splitting is disabled.
    /// </summary>
    public IReadOnlyList<string> FilterStops(IReadOnlyList<string> stops) => Markup.FilterStops(stops);

    /// <summary>True when output is split.</summary>
    public bool Enabled => _format != ReasoningFormat.None;

    /// <summary>True when the rendered prompt already ended inside an open think block.</summary>
    public bool PromptOpened { get; }

    /// <summary>A fresh splitter for one generated sequence, or null when <see cref="Enabled"/> is false.</summary>
    public IReasoningSplitter? NewSplitter()
        => !Enabled ? null
            : Markup == ReasoningMarkup.Harmony
                ? new HarmonySplitter()
                : new ReasoningSplitter(PromptOpened, _format == ReasoningFormat.Deepseek, Markup.OpenTag(), Markup.CloseTag());

    /// <summary>
    /// Applies the stop-sequence gate: stop strings (other than the template's control tokens) are suspended
    /// while the model is inside a think block, so a stop string in its thinking cannot cut the turn short
    /// before it has answered. Covers both a block the template opened and one the model opens itself.
    /// </summary>
    public InferenceOptions Gate(InferenceOptions options, IReadOnlyList<string> ungatedStops)
        => Enabled
            ? options with
            {
                ReasoningStopGate = new StopGate(
                    Markup.OpenTag(), Markup.CloseTag(), PromptOpened,
                    OpenOnlyAtStart: _format != ReasoningFormat.Deepseek),
                StopSequencesUngated = Markup.FilterStops(ungatedStops),
            }
            : options;

    /// <summary>
    /// The effective thinking budget (#1292): the request's value, else the server default; null when unlimited
    /// (negative), when output is not split (thinking off, constrained, <c>reasoning_format=none</c>), or for Harmony,
    /// whose analysis channel has no single close tag to force.
    /// </summary>
    /// <param name="requested">The request's budget (<c>thinking_budget_tokens</c> / Anthropic <c>budget_tokens</c>), or null.</param>
    /// <param name="serverDefault">The server's <c>--reasoning-budget</c> (-1 = unlimited).</param>
    /// <returns>The budget in tokens (0 = close at once), or null for none.</returns>
    public int? ThinkingBudget(int? requested, int serverDefault)
    {
        int budget = requested ?? serverDefault;
        return Enabled && Markup != ReasoningMarkup.Harmony && budget >= 0 ? budget : null;
    }

    /// <summary>
    /// Installs the thinking budget on <paramref name="options"/> (#1292); unchanged when <paramref name="budget"/> is null.
    /// </summary>
    /// <param name="options">The request's options.</param>
    /// <param name="budget">From <see cref="ThinkingBudget(int?, int)"/>.</param>
    /// <param name="message">Text forced before the close tag when the budget runs out, or null.</param>
    /// <returns>The options carrying the budget.</returns>
    public InferenceOptions WithThinkingBudget(InferenceOptions options, int? budget, string? message)
        => budget is int b
            ? options with { ThinkingBudget = new ThinkingBudget(b, Markup.OpenTag(), Markup.CloseTag(), PromptOpened, message) }
            : options;

    /// <summary>
    /// Splits one complete generation. The token count is 0 when there was no reasoning.
    /// </summary>
    public (string? Reasoning, string Content, int ReasoningTokens) SplitComplete(string text, ITokenizer? tokenizer)
    {
        if (!Enabled)
            return (null, text, 0);

        string reasoning, content;
        IReasoningSplitter state;
        if (Markup == ReasoningMarkup.Harmony)
        {
            var h = HarmonySplitter.Split(text);
            (reasoning, content, state) = (h.Reasoning, h.Content, h.State);
        }
        else
        {
            var t = ReasoningSplitter.Split(text, PromptOpened, _format == ReasoningFormat.Deepseek, Markup.OpenTag(), Markup.CloseTag());
            (reasoning, content, state) = (t.Reasoning, t.Content, t.State);
        }
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
    /// What the loaded chat template does with the thinking controls, for <c>/props</c> (#1106): whether it reads
    /// <c>enable_thinking</c> / <c>reasoning_effort</c> and whether it emits a splittable think block.
    /// Null when no Jinja template is loaded.
    /// </summary>
    internal static Models.ReasoningCapabilitiesDto? Capabilities(IChatTemplate? template)
    {
        if (template is not JinjaChatTemplate jinja)
            return null;
        string src = jinja.TemplateSource;
        return new Models.ReasoningCapabilitiesDto
        {
            EnableThinking = src.Contains("enable_thinking", StringComparison.Ordinal),
            ReasoningEffort = src.Contains("reasoning_effort", StringComparison.Ordinal),
            ThinkingBlock = jinja.ReasoningMarkup != ReasoningMarkup.Harmony
                && src.Contains(jinja.ReasoningMarkup.OpenTag(), StringComparison.Ordinal),
        };
    }

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

    /// <summary>
    /// The text the chat template appends after the last message when the generation prompt is on
    /// (e.g. <c>&lt;|im_start|&gt;assistant\n&lt;think&gt;\n</c>), or <see langword="null"/> when it cannot be
    /// determined. The engine places its recurrent-prefix snapshot just before this suffix so the next
    /// turn's prompt continues from it (#1093).
    /// </summary>
    /// <param name="template">The chat template.</param>
    /// <param name="messages">The messages that produced <paramref name="prompt"/>.</param>
    /// <param name="options">The options that produced <paramref name="prompt"/> (generation prompt on).</param>
    /// <param name="prompt">The prompt rendered with the generation prompt on.</param>
    /// <returns>
    /// <paramref name="prompt"/> minus the same render with the generation prompt off, when that render is
    /// a strict prefix of it; otherwise <see langword="null"/> (also when the second render throws).
    /// </returns>
    internal static string? GenerationPromptSuffix(
        IChatTemplate template, IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options, string prompt)
    {
        try
        {
            string bare = template.Apply(messages, options with { AddGenerationPrompt = false });
            return prompt.Length > bare.Length && prompt.StartsWith(bare, StringComparison.Ordinal)
                ? prompt[bare.Length..]
                : null;
        }
        catch (JinjaException)
        {
            return null;
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
    internal static ReasoningPlan Plan(ReasoningFormat serverFormat, string? requestFormat, bool constrained, string prompt, out string? error,
        IChatTemplate? template = null)
    {
        error = null;
        var markup = (template as JinjaChatTemplate)?.ReasoningMarkup ?? ReasoningMarkup.Think;
        var format = serverFormat;
        if (requestFormat is not null)
        {
            if (!ReasoningFormats.TryParse(requestFormat, out format))
            {
                error = $"reasoning_format: unknown value '{requestFormat}'. Expected: none, auto, deepseek.";
                return new ReasoningPlan(ReasoningFormat.None, false, markup);
            }
        }

        if (constrained || format == ReasoningFormat.None)
            return markup == ReasoningMarkup.Think ? ReasoningPlan.Disabled : new ReasoningPlan(ReasoningFormat.None, false, markup);
        bool opened = markup != ReasoningMarkup.Harmony
            && ReasoningFormats.PromptOpensThinking(prompt, markup.OpenTag(), markup.CloseTag());
        return new ReasoningPlan(format, opened, markup);
    }

    /// <summary>
    /// Counts one streamed token toward reasoning: true when the splitter was inside a block before the
    /// token or is inside one after it (so both the opening and the closing tag tokens count).
    /// </summary>
    /// <summary>
    /// <c>max_tokens</c> with room for the answer after a thinking budget (#1292): at least budget + reserve, so a think
    /// block that uses its whole budget cannot leave the answer truncated. Unchanged without a positive budget or reserve.
    /// The result is still clamped to the context window by prompt validation.
    /// </summary>
    /// <param name="maxTokens">The requested (or default) <c>max_tokens</c>.</param>
    /// <param name="budget">The effective thinking budget, or null.</param>
    /// <param name="answerReserve">The server's <c>--reasoning-answer-reserve</c> (0 = llama.cpp: the budget counts inside max_tokens).</param>
    /// <returns>The effective <c>max_tokens</c>.</returns>
    internal static int WithAnswerReserve(int maxTokens, int? budget, int answerReserve)
        => budget is > 0 and int b && answerReserve > 0
            ? (int)Math.Max(maxTokens, Math.Min(int.MaxValue, (long)b + answerReserve))
            : maxTokens;

    internal static bool IsReasoningToken(bool wasInReasoning, IReasoningSplitter splitter)
        => wasInReasoning || splitter.InReasoning;

    /// <summary>Usage details, or null when no reasoning was produced.</summary>
    internal static Models.CompletionTokensDetailsDto? Details(int reasoningTokens)
        => reasoningTokens > 0 ? new Models.CompletionTokensDetailsDto { ReasoningTokens = reasoningTokens } : null;
}
