using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses the Qwen3-Coder XML tool-call format used by Qwen3.5 / 3.6 / 3.8, Ornith, Qwen3-Coder and
/// Nemotron-style templates:
/// <code>
/// &lt;tool_call&gt;
/// &lt;function=get_weather&gt;
/// &lt;parameter=city&gt;
/// Paris
/// &lt;/parameter&gt;
/// &lt;/function&gt;
/// &lt;/tool_call&gt;
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// Behaviour follows llama.cpp's <c>common_chat_params_init_qwen3_coder</c>: the closing
/// <c>&lt;/tool_call&gt;</c> is optional (it is routinely consumed as a stop sequence), the opening
/// <c>&lt;tool_call&gt;</c> may be omitted before a <c>&lt;function=</c> block, several blocks are parallel
/// calls, and anything before the first call (a leading <c>&lt;/think&gt;</c>, reasoning, prose) is not part of
/// the call. A function block without its <c>&lt;/function&gt;</c> is a truncated generation and is not
/// reported, so a call cut off by <c>max_tokens</c> is never executed with half its arguments.
/// </para>
/// <para>
/// This parser is a superset of <see cref="HermesToolCallParser"/>: a <c>&lt;tool_call&gt;</c> body that
/// is JSON rather than <c>&lt;function=</c> XML is parsed as Hermes JSON, so a fine-tune that switches
/// formats is still understood.
/// </para>
/// <para>
/// Parameter values are text. A declared <c>string</c> parameter keeps the text verbatim (minus the single
/// newline the template wraps it in); declared integer/number/boolean/object/array parameters are parsed as
/// JSON; with no schema available the text is parsed as JSON when valid and kept as a string otherwise.
/// Use <see cref="TryParse(string, IReadOnlyList{ToolDefinition}?)"/> to supply the schema.
/// </para>
/// </remarks>
public sealed class QwenXmlToolCallParser : IToolCallParser
{
    private const string OpenTag = "<tool_call>";
    private const string CloseTag = "</tool_call>";
    private const string FunctionOpen = "<function=";
    private const string FunctionClose = "</function>";
    private const string ParamOpen = "<parameter=";
    private const string ParamClose = "</parameter>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText) => TryParse(generatedText, null);

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText, IReadOnlyList<ToolDefinition>? tools)
    {
        if (string.IsNullOrEmpty(generatedText))
            return null;

        var calls = new List<ToolCall>();
        int pos = 0;

        while (pos < generatedText.Length)
        {
            int open = generatedText.IndexOf(OpenTag, pos, StringComparison.Ordinal);
            int bare = generatedText.IndexOf(FunctionOpen, pos, StringComparison.Ordinal);

            if (open < 0 && bare < 0)
                break;

            if (open >= 0 && (bare < 0 || open < bare))
            {
                int bodyStart = open + OpenTag.Length;
                int bodyEnd = generatedText.IndexOf(CloseTag, bodyStart, StringComparison.Ordinal);
                // Look ahead to the NEXT open tag so a missing close tag does not swallow later calls.
                int nextOpen = generatedText.IndexOf(OpenTag, bodyStart, StringComparison.Ordinal);
                if (bodyEnd < 0 || (nextOpen >= 0 && nextOpen < bodyEnd))
                    bodyEnd = nextOpen >= 0 ? nextOpen : generatedText.Length;

                ParseBlock(generatedText.AsSpan(bodyStart, bodyEnd - bodyStart), tools, calls);
                pos = bodyEnd;
                if (pos < generatedText.Length && string.CompareOrdinal(generatedText, pos, CloseTag, 0, CloseTag.Length) == 0)
                    pos += CloseTag.Length;
            }
            else
            {
                // Bare <function=...> with no <tool_call> wrapper: Qwen3-Coder occasionally omits it.
                int end = ParseFunction(generatedText, bare, tools, calls);
                pos = end > bare ? end : bare + FunctionOpen.Length;
            }
        }

        if (calls.Count == 0)
            return null;

        // Sequential ids over the call list as a whole.
        var result = new ToolCall[calls.Count];
        for (int i = 0; i < result.Length; i++)
            result[i] = calls[i] with { Id = $"call_{i}" };
        return ToolArgumentCoercer.Coerce(result, tools);
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(OpenTag, StringComparison.Ordinal)
           || text.Contains(FunctionOpen, StringComparison.Ordinal);

    /// <summary>Parses the body of one <c>&lt;tool_call&gt;</c> block (XML, or Hermes JSON as a fallback).</summary>
    private static void ParseBlock(ReadOnlySpan<char> body, IReadOnlyList<ToolDefinition>? tools, List<ToolCall> calls)
    {
        string text = body.ToString();
        int p = 0;
        bool any = false;
        while (true)
        {
            int f = text.IndexOf(FunctionOpen, p, StringComparison.Ordinal);
            if (f < 0)
                break;
            int end = ParseFunction(text, f, tools, calls);
            any |= end > f;
            p = end > f ? end : f + FunctionOpen.Length;
        }

        if (any)
            return;

        // No XML function inside: Hermes-style JSON body.
        var json = ToolCallJsonHelper.ExtractAndParse(text.Trim(), "call");
        if (json is { Length: > 0 })
            calls.AddRange(json);
    }

    /// <summary>
    /// Parses one <c>&lt;function=NAME&gt;...&lt;/function&gt;</c> starting at <paramref name="start"/>.
    /// Returns the index just past <c>&lt;/function&gt;</c>, or <paramref name="start"/> when the block is
    /// malformed or unterminated (nothing is added).
    /// </summary>
    private static int ParseFunction(string text, int start, IReadOnlyList<ToolDefinition>? tools, List<ToolCall> calls)
    {
        int nameStart = start + FunctionOpen.Length;
        int nameEnd = text.IndexOf('>', nameStart);
        if (nameEnd < 0)
            return start;

        string name = text.Substring(nameStart, nameEnd - nameStart).Trim();
        if (name.Length == 0 || name.AsSpan().IndexOfAny('<', '\n') >= 0)
            return start;

        int close = text.IndexOf(FunctionClose, nameEnd, StringComparison.Ordinal);
        if (close < 0)
            return start; // truncated generation: never report half a call

        var tool = ToolArgumentCoercer.Find(tools, name);
        JsonDocument? schemaDoc = null;
        JsonElement? props = null;
        try
        {
            if (tool is not null && !string.IsNullOrWhiteSpace(tool.ParametersSchema))
            {
                try
                {
                    schemaDoc = JsonDocument.Parse(tool.ParametersSchema);
                    if (schemaDoc.RootElement.ValueKind == JsonValueKind.Object
                        && schemaDoc.RootElement.TryGetProperty("properties", out var pr)
                        && pr.ValueKind == JsonValueKind.Object)
                        props = pr;
                }
                catch (JsonException)
                {
                    schemaDoc?.Dispose();
                    schemaDoc = null;
                }
            }

            var buffer = new ArrayBufferWriter<byte>();
            using (var w = new Utf8JsonWriter(buffer))
            {
                w.WriteStartObject();
                int p = nameEnd + 1;
                while (p < close)
                {
                    int po = text.IndexOf(ParamOpen, p, close - p, StringComparison.Ordinal);
                    if (po < 0)
                        break;
                    int keyStart = po + ParamOpen.Length;
                    int keyEnd = text.IndexOf('>', keyStart, close - keyStart);
                    if (keyEnd < 0)
                        break;
                    string key = text.Substring(keyStart, keyEnd - keyStart).Trim();

                    int valStart = keyEnd + 1;
                    int valEnd = text.IndexOf(ParamClose, valStart, close - valStart, StringComparison.Ordinal);
                    int next;
                    if (valEnd < 0)
                    {
                        // Missing </parameter>: value runs to the next <parameter= or the function end.
                        int np = text.IndexOf(ParamOpen, valStart, close - valStart, StringComparison.Ordinal);
                        valEnd = np >= 0 ? np : close;
                        next = valEnd;
                    }
                    else
                    {
                        next = valEnd + ParamClose.Length;
                    }

                    if (key.Length > 0)
                    {
                        string raw = StripTemplateNewlines(text.AsSpan(valStart, valEnd - valStart));
                        JsonElement? schema = props is { } pp && pp.TryGetProperty(key, out var ps) ? ps : null;
                        w.WritePropertyName(key);
                        ToolArgumentCoercer.WriteTyped(w, raw, schema);
                    }

                    p = next;
                }
                w.WriteEndObject();
            }

            calls.Add(new ToolCall($"call_{calls.Count}", name, Encoding.UTF8.GetString(buffer.WrittenSpan)));
            return close + FunctionClose.Length;
        }
        finally
        {
            schemaDoc?.Dispose();
        }
    }

    /// <summary>
    /// The template wraps each value as <c>&lt;parameter=k&gt;\nVALUE\n&lt;/parameter&gt;</c>; remove exactly that
    /// one leading and one trailing newline (CRLF tolerated) and nothing else, so multi-line string values
    /// and deliberate leading/trailing spaces survive.
    /// </summary>
    internal static string StripTemplateNewlines(ReadOnlySpan<char> value)
    {
        if (value.StartsWith("\r\n"))
            value = value[2..];
        else if (value.StartsWith("\n"))
            value = value[1..];

        if (value.EndsWith("\r\n"))
            value = value[..^2];
        else if (value.EndsWith("\n"))
            value = value[..^1];

        return value.ToString();
    }
}
