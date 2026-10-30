using System.Text.Json;
using DotLLM.Tokenizers.Reasoning;

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
    private readonly ITokenizer? _guard;
    private readonly string[] _assistantExempt;

    private static readonly System.Text.RegularExpressions.Regex GenerationTag = new(
        @"\{%(-?)\s*(?:end)?generation\s*(-?)%\}", System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>
    /// Creates a new Jinja2 chat template.
    /// </summary>
    /// <param name="templateSource">Jinja2 template string (from GGUF metadata or HuggingFace config).</param>
    /// <param name="bosToken">Beginning-of-sequence token string.</param>
    /// <param name="eosToken">End-of-sequence token string.</param>
    /// <param name="guardTokenizer">
    /// When set, every caller-supplied string (message content, tool results and arguments, tool definitions,
    /// template kwargs) is passed through <see cref="ITokenizer.MarkUntrustedSpecials"/> before rendering, so the
    /// rendered prompt tokenizes such text as plain text and never as control tokens (#1339,
    /// <see cref="ControlTokenGuard"/>). Null = no protection (tests, tooling that renders text only).
    /// </param>
    public JinjaChatTemplate(string templateSource, string bosToken, string eosToken, ITokenizer? guardTokenizer = null)
    {
        _guard = guardTokenizer;
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

        TemplateSource = templateSource;
        ReasoningMarkup = Reasoning.ReasoningMarkups.Detect(templateSource);

        // An assistant turn replayed from history legitimately carries its own reasoning tags, and the template
        // (e.g. Qwen3: content.split('</think>')) parses them out of the content; those spellings stay live there.
        _assistantExempt = ReasoningMarkup is Reasoning.ReasoningMarkup.Think or Reasoning.ReasoningMarkup.Gemma4Channel
            ? [ReasoningMarkup.OpenTag(), ReasoningMarkup.CloseTag()]
            : [];

        var lexer = new JinjaLexer(templateSource);
        var tokens = lexer.Tokenize();
        var parser = new JinjaParser(tokens);
        _ast = parser.Parse();
    }

    /// <summary>The Jinja source this template was built from (trailing newline normalised, <c>{% generation %}</c> markers rewritten).</summary>
    public string TemplateSource { get; }

    /// <summary>The reasoning markup this template's model uses, detected from the source (#798).</summary>
    public Reasoning.ReasoningMarkup ReasoningMarkup { get; }

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
                ["role"] = Guard(msg.Role),
                ["content"] = Guard(msg.Content, msg.Role == "assistant" ? _assistantExempt : null),
            };

            if (msg.ToolCalls is { Length: > 0 })
            {
                var toolCalls = new List<object?>();
                foreach (var tc in msg.ToolCalls)
                {
                    var tcDict = new Dictionary<string, object?>
                    {
                        ["id"] = Guard(tc.Id),
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object?>
                        {
                            ["name"] = Guard(tc.FunctionName),
                            // Parse arguments JSON into dict so tojson in templates
                            // produces correct output (not double-serialized).
                            ["arguments"] = Guard(ParseJsonToDict(tc.Arguments) ?? tc.Arguments),
                        }
                    };
                    toolCalls.Add(tcDict);
                }
                dict["tool_calls"] = toolCalls;
            }

            if (msg.ToolCallId is not null)
                dict["tool_call_id"] = Guard(msg.ToolCallId);

            // Only when present: templates test `message.reasoning_content is string`.
            if (msg.ReasoningContent is not null)
                dict["reasoning_content"] = Guard(msg.ReasoningContent);

            messageList.Add(dict);
        }

        var context = new Dictionary<string, object?>();

        // chat_template_kwargs first so every explicit field below, and the structural variables,
        // take precedence over a same-named kwarg.
        if (options.TemplateKwargs is { Count: > 0 })
        {
            foreach (var (key, value) in options.TemplateKwargs)
                context[key] = Guard(ConvertJsonElement(value));
        }

        // Reasoning switches: inserted only when the client set them, so an unset value stays
        // *undefined* in the template (a null-valued key is not undefined).
        if (options.EnableThinking is { } enableThinking)
            context["enable_thinking"] = enableThinking;
        if (options.ReasoningEffort is { Length: > 0 } effort)
            context["reasoning_effort"] = Guard(effort);
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
                        ["name"] = Guard(tool.Name),
                        ["description"] = Guard(tool.Description),
                        ["parameters"] = Guard(ParseJsonToDict(tool.ParametersSchema)),
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

    /// <summary>Marks special-token spellings in untrusted caller text (see <see cref="ControlTokenGuard"/>).</summary>
    private string? Guard(string? text, IReadOnlyList<string>? exempt = null)
        => _guard is null || text is null ? text : _guard.MarkUntrustedSpecials(text, exempt);

    /// <summary>Recursive <see cref="Guard(string?, IReadOnlyList{string}?)"/> over parsed JSON (strings and dictionary keys).</summary>
    private object? Guard(object? value)
    {
        if (_guard is null)
            return value;
        switch (value)
        {
            case string s:
                return _guard.MarkUntrustedSpecials(s);
            case Dictionary<string, object?> d:
                var copy = new Dictionary<string, object?>(d.Count);
                foreach (var (k, v) in d)
                    copy[_guard.MarkUntrustedSpecials(k)] = Guard(v);
                return copy;
            case List<object?> l:
                var list = new List<object?>(l.Count);
                foreach (var v in l)
                    list.Add(Guard(v));
                return list;
            default:
                return value;
        }
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
