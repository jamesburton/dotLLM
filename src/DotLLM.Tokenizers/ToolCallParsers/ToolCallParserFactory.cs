using DotLLM.Core.Configuration;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Selects the appropriate <see cref="IToolCallParser"/> based on model architecture
/// and chat template content.
/// </summary>
public static class ToolCallParserFactory
{
    /// <summary>
    /// Creates a tool call parser for the given model.
    /// </summary>
    /// <param name="architecture">Model architecture enum.</param>
    /// <param name="chatTemplate">Raw chat template string (for heuristic detection). May be null.</param>
    /// <returns>The best-matching parser for this model.</returns>
    public static IToolCallParser Create(Architecture architecture, string? chatTemplate = null)
    {
        // 1. Template content heuristics (highest priority — template is the source of truth)
        if (!string.IsNullOrEmpty(chatTemplate))
        {
            // gpt-oss / OpenAI Harmony: channel messages; a tool call is a commentary message with a
            // `to=functions.NAME` recipient. Its own token set (<|channel|>/<|message|>/<|call|>).
            if (Reasoning.ReasoningMarkups.Detect(chatTemplate) == Reasoning.ReasoningMarkup.Harmony)
                return new HarmonyToolCallParser();

            // Gemma-4: the template renders tool calls as <|tool_call>call:NAME{...}<tool_call|>.
            // Checked first — its own token pair, no overlap with the formats below.
            if (chatTemplate.Contains("<|tool_call>", StringComparison.Ordinal))
                return new Gemma4ToolCallParser();

            // Qwen3-Coder XML (Qwen3.5/3.6/3.8, Ornith, Qwen3-Coder, Nemotron-3): the template shows
            // <tool_call><function=NAME><parameter=KEY>. It ALSO contains the bare "<tool_call>" the
            // Hermes check below keys on, so it must be tested before it. The parser is a superset
            // that falls back to Hermes JSON inside <tool_call>.
            if (chatTemplate.Contains("<function=", StringComparison.Ordinal)
                && chatTemplate.Contains("<parameter=", StringComparison.Ordinal))
                return new QwenXmlToolCallParser();

            // SmolLM3 ships TWO tool-calling branches in a single template
            // (chosen at render time by a `xml_tools` vs `python_tools`
            // variable). When the template has only the Pythonic branch
            // wired we route to the Pythonic parser; the XML branch shares
            // the Hermes-compatible <tool_call> envelope with everyone else.
            if (chatTemplate.Contains("python_tools", StringComparison.Ordinal)
                && !chatTemplate.Contains("xml_tools", StringComparison.Ordinal))
                return new PythonicToolCallParser();

            if (chatTemplate.Contains("<tool_call>", StringComparison.Ordinal))
                return new XmlToolCallParser();

            if (chatTemplate.Contains("python_tag", StringComparison.Ordinal) ||
                chatTemplate.Contains("<|python_tag|>", StringComparison.Ordinal))
                return new LlamaToolCallParser();

            if (chatTemplate.Contains("[TOOL_CALLS]", StringComparison.Ordinal))
                return new MistralToolCallParser();
        }

        // 2. Architecture-based fallback
        return architecture switch
        {
            Architecture.Llama => new LlamaToolCallParser(),
            Architecture.Mistral => new MistralToolCallParser(),
            Architecture.Qwen or Architecture.QwenMoe => new HermesToolCallParser(),
            // Qwen3.5+ hybrids speak the Qwen3-Coder XML form; the parser also accepts Hermes JSON.
            Architecture.Qwen3MoeHybrid or Architecture.Qwen3HybridDense => new QwenXmlToolCallParser(),
            Architecture.Gemma4 => new Gemma4ToolCallParser(),
            Architecture.GptOss => new HarmonyToolCallParser(),
            // SmolLM3 defaults to the XML (Hermes-compatible) format; the
            // Pythonic variant is template-gated above.
            Architecture.SmolLM3 => new XmlToolCallParser(),
            // BitNet uses a tool-aware template that emits <tool_call> blocks
            // (Hermes-compatible). Route to HermesToolCallParser regardless of
            // what the GGUF metadata declares (the template is substituted).
            Architecture.BitNet => new HermesToolCallParser(),
            _ => new GenericToolCallParser()
        };
    }

    /// <summary>
    /// Selects the parser to use for *output* given the effective <see cref="ToolChoice"/>.
    /// </summary>
    /// <param name="toolChoice">The request's tool choice.</param>
    /// <param name="modelParser">The model-family parser produced by <see cref="Create"/>.</param>
    /// <returns>
    /// A <see cref="GenericToolCallParser"/> for <see cref="ToolChoice.Required"/> and
    /// <see cref="ToolChoice.Function"/>; otherwise <paramref name="modelParser"/> unchanged.
    /// </returns>
    /// <remarks>
    /// When <c>tool_choice</c> is <c>required</c> or a specific function, the output is produced by
    /// <c>ToolCallSchemaBuilder</c> + <c>JsonSchemaConstraint</c>, which emit a <b>bare JSON object</b>
    /// (<c>{"name":…,"arguments":…}</c>) — the JSON-schema constraint machinery cannot emit a model's
    /// literal envelope tokens (<c>&lt;tool_call&gt;</c>, <c>&lt;|python_tag|&gt;</c>, <c>[TOOL_CALLS]</c>).
    /// Parsing constrained output with a marker-based parser therefore always yields <c>null</c>, and
    /// the tool call is silently lost. Every constrained call site must route through this method so the
    /// rule cannot fork again (see issue #325).
    /// <para>
    /// Note this is about *parsing*; the <c>argumentsKey</c> passed to <c>ToolCallSchemaBuilder</c> must
    /// still be derived from the <i>model</i> parser (Llama uses <c>"parameters"</c>), so callers should
    /// compute it before swapping.
    /// </para>
    /// </remarks>
    public static IToolCallParser ForToolChoice(ToolChoice toolChoice, IToolCallParser modelParser)
        => toolChoice is ToolChoice.Required or ToolChoice.Function
            ? new GenericToolCallParser()
            : modelParser;
}
