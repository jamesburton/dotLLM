using System.Text.Json;

namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// IChatTemplate implementation backed by a Jinja2-subset interpreter.
/// Parses the template once at construction and evaluates per Apply() call.
/// </summary>
public sealed class JinjaChatTemplate : IChatTemplate
{
    private readonly JinjaTemplate _ast;
    private readonly string _bosToken;
    private readonly string _eosToken;
    private readonly bool _aliasXmlTools;

    private static readonly System.Text.RegularExpressions.Regex GenerationTag = new(
        @"\{%(-?)\s*(?:end)?generation\s*(-?)%\}", System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>
    /// Creates a new Jinja2 chat template.
    /// </summary>
    /// <param name="templateSource">Jinja2 template string (from GGUF metadata or HuggingFace config).</param>
    /// <param name="bosToken">Beginning-of-sequence token string.</param>
    /// <param name="eosToken">End-of-sequence token string.</param>
    public JinjaChatTemplate(string templateSource, string bosToken, string eosToken)
    {
        _bosToken = bosToken;
        _eosToken = eosToken;

        // Jinja2's default is keep_trailing_newline=False: ONE trailing newline of the template source is
        // dropped (HF transformers and llama.cpp's minja both do this). Llama-3.x's template ends
        // "{%- endif %}\n", so without this every prompt ended "<|start_header_id|>assistant<|end_header_id|>\n\n\n"
        // — an extra blank line the model never saw in training (found while diagnosing #771).
        if (templateSource.EndsWith("\r\n", StringComparison.Ordinal))
            templateSource = templateSource[..^2];
        else if (templateSource.EndsWith('\n'))
            templateSource = templateSource[..^1];

        // HF's `{% generation %}` / `{% endgeneration %}` mark the span an assistant-token mask covers (training
        // only); at render time they emit nothing. SmolLM3's template wraps every assistant turn in them, the
        // parser rejected the unknown statement keyword, and the server then fell back to the plain transcript
        // template: no tools, no /think switch, the wrong chat format (#797). Rewritten as comments, which keep
        // the `-` whitespace-control markers and emit nothing, exactly like minja's no-op treatment.
        templateSource = GenerationTag.Replace(templateSource, m => "{#" + m.Groups[1].Value + " generation " + m.Groups[2].Value + "#}");

        // SmolLM3 reads its tool list from `xml_tools` / `python_tools` and never from `tools`.
        _aliasXmlTools = templateSource.Contains("xml_tools", StringComparison.Ordinal)
            && !templateSource.Contains("tools is defined", StringComparison.Ordinal)
            && !templateSource.Contains("if tools", StringComparison.Ordinal);

        var lexer = new JinjaLexer(templateSource);
        var tokens = lexer.Tokenize();
        var parser = new JinjaParser(tokens);
        _ast = parser.Parse();
    }

    /// <inheritdoc/>
    public string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
    {
        var context = BuildContext(messages, options);
        var evaluator = new JinjaEvaluator(context);
        return evaluator.Evaluate(_ast);
    }

    private Dictionary<string, object?> BuildContext(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
    {
        // Convert ChatMessage[] to List<Dict> matching HuggingFace Jinja template convention
        var messageList = new List<object?>();
        foreach (var msg in messages)
        {
            var dict = new Dictionary<string, object?>
            {
                ["role"] = msg.Role,
                ["content"] = msg.Content,
            };

            if (msg.ToolCalls is { Length: > 0 })
            {
                var toolCalls = new List<object?>();
                foreach (var tc in msg.ToolCalls)
                {
                    var tcDict = new Dictionary<string, object?>
                    {
                        ["id"] = tc.Id,
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object?>
                        {
                            ["name"] = tc.FunctionName,
                            // Parse arguments JSON into dict so tojson in templates
                            // produces correct output (not double-serialized).
                            ["arguments"] = ParseJsonToDict(tc.Arguments) ?? tc.Arguments,
                        }
                    };
                    toolCalls.Add(tcDict);
                }
                dict["tool_calls"] = toolCalls;
            }

            if (msg.ToolCallId is not null)
                dict["tool_call_id"] = msg.ToolCallId;

            // Only when present: templates test `message.reasoning_content is string`.
            if (msg.ReasoningContent is not null)
                dict["reasoning_content"] = msg.ReasoningContent;

            messageList.Add(dict);
        }

        var context = new Dictionary<string, object?>();

        // chat_template_kwargs first so every explicit field below, and the structural variables,
        // take precedence over a same-named kwarg.
        if (options.TemplateKwargs is { Count: > 0 })
        {
            foreach (var (key, value) in options.TemplateKwargs)
                context[key] = ConvertJsonElement(value);
        }

        // Reasoning switches: inserted only when the client set them, so an unset value stays
        // *undefined* in the template (a null-valued key is not undefined).
        if (options.EnableThinking is { } enableThinking)
            context["enable_thinking"] = enableThinking;
        if (options.ReasoningEffort is { Length: > 0 } effort)
            context["reasoning_effort"] = effort;
        if (options.PreserveThinking is { } preserve)
            context["preserve_thinking"] = preserve;

        context["messages"] = messageList;
        context["add_generation_prompt"] = options.AddGenerationPrompt;
        context["bos_token"] = _bosToken;
        context["eos_token"] = _eosToken;

        // Add tool definitions if present
        if (options.Tools is { Length: > 0 })
        {
            var tools = new List<object?>();
            foreach (var tool in options.Tools)
            {
                var toolDict = new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["function"] = new Dictionary<string, object?>
                    {
                        ["name"] = tool.Name,
                        ["description"] = tool.Description,
                        ["parameters"] = ParseJsonToDict(tool.ParametersSchema),
                    }
                };
                tools.Add(toolDict);
            }
            context["tools"] = tools;

            // SmolLM3 (#797): the template's tool section is gated on `xml_tools or python_tools` (HF callers
            // pass xml_tools=...), so an OpenAI-style `tools` request rendered NO tool declarations and the model
            // answered that it had no tools. Alias `tools` to `xml_tools` unless the client chose a branch itself.
            if (_aliasXmlTools && !context.ContainsKey("xml_tools") && !context.ContainsKey("python_tools"))
                context["xml_tools"] = tools;
        }

        return context;
    }

    /// <summary>
    /// Parses a JSON string into nested Dictionary/List structures
    /// that the Jinja evaluator can traverse.
    /// </summary>
    private static object? ParseJsonToDict(string json)
    {
        if (string.IsNullOrEmpty(json))
            return null;

        try
        {
            using var doc = JsonDocument.Parse(json);
            return ConvertJsonElement(doc.RootElement);
        }
        catch
        {
            return json; // fallback to raw string
        }
    }

    private static object? ConvertJsonElement(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Object => ConvertJsonObject(element),
        JsonValueKind.Array => ConvertJsonArray(element),
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number => element.TryGetInt32(out int i) ? i : element.GetDouble(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null => null,
        _ => element.ToString()
    };

    private static Dictionary<string, object?> ConvertJsonObject(JsonElement element)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var prop in element.EnumerateObject())
            dict[prop.Name] = ConvertJsonElement(prop.Value);
        return dict;
    }

    private static List<object?> ConvertJsonArray(JsonElement element)
    {
        var list = new List<object?>();
        foreach (var item in element.EnumerateArray())
            list.Add(ConvertJsonElement(item));
        return list;
    }
}
