using DotLLM.Core.Configuration;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Installs a decoding constraint for <c>tool_choice</c>, shared by the OpenAI
/// (<c>/v1/chat/completions</c>) and Anthropic (<c>/v1/messages</c>) surfaces.
/// </summary>
/// <remarks>
/// <para>
/// Extracted from <c>MessagesEndpoint</c> for #456. The Anthropic path honoured
/// <c>tool_choice</c> from #449; the OpenAI path parsed it into a local and then
/// <b>discarded it</b> — <c>required</c>, <c>none</c> and a named function all had no effect on
/// generation. The SDK conformance matrix caught this as <c>openai/tool.single</c> and
/// <c>openai/tool.parallel</c> failing while the equivalent Anthropic rows passed, which is what
/// identified the fix as a sharing problem rather than a missing feature.
/// </para>
/// <para>
/// The logic is backend-agnostic by construction: it takes an already-parsed
/// <see cref="ToolChoice"/>, so each surface keeps its own wire parsing.
/// </para>
/// </remarks>
internal static class ToolChoiceBinder
{
    /// <summary>
    /// Applies Anthropic <c>tool_choice</c> semantics and returns the tool-call parser this
    /// request should use (null when tool calls must not be produced).
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><c>auto</c> — the model's own parser, unconstrained.</item>
    /// <item><c>any</c>/<c>tool</c> — decoding is constrained to a tool-call JSON schema, and the
    /// markerless parser is used, because the constraint emits a bare JSON object rather than the
    /// model's <c>&lt;tool_call&gt;</c> envelope. Same construction as the CLI's forced path.</item>
    /// <item><c>none</c> — no parser, so nothing the model emits is reported as <c>tool_use</c>.</item>
    /// </list>
    /// A caller-supplied <c>response_format</c> does not exist on this surface, so the constraint
    /// slot is always free.
    /// </remarks>
    /// <param name="toolChoice">The parsed Anthropic <c>tool_choice</c>.</param>
    /// <param name="tools">The tool definitions the request supplied, if any.</param>
    /// <param name="modelParser">The model's own tool-call parser, if the model has one.</param>
    /// <param name="options">Inference options; a decoding constraint is installed on them.</param>
    /// <param name="forcedToolCall">
    /// True when decoding was constrained, i.e. the whole completion IS the tool call and none
    /// of it is assistant text.
    /// </param>
    internal static IToolCallParser? Apply(
        ToolChoice toolChoice,
        ToolDefinition[]? tools,
        IToolCallParser? modelParser,
        ref DotLLM.Core.Configuration.InferenceOptions options,
        out bool forcedToolCall)
    {
        forcedToolCall = false;
        if (tools is not { Length: > 0 })
            return null;
        if (toolChoice is ToolChoice.None)
            return null;
        if (modelParser is null)
            return null;

        string argumentsKey = modelParser is LlamaToolCallParser ? "parameters" : "arguments";
        var schema = toolChoice switch
        {
            ToolChoice.Required => ToolCallSchemaBuilder.BuildForRequired(tools, argumentsKey),
            ToolChoice.Function fn when Array.Find(tools, t => t.Name == fn.Name) is { } target =>
                ToolCallSchemaBuilder.BuildForFunction(target, argumentsKey),
            _ => null,
        };
        if (schema is null)
            return modelParser;

        options = options with
        {
            ResponseFormat = new DotLLM.Core.Configuration.ResponseFormat.JsonSchema
            {
                Schema = schema,
                Name = "tool_call",
            },
        };
        forcedToolCall = true;
        return ToolCallParserFactory.ForToolChoice(toolChoice, modelParser);
    }
}
