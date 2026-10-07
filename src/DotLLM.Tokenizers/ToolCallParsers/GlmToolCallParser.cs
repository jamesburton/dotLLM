using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses GLM-4.5 / 4.6 / 4.7 tool calls (#797):
/// <c>&lt;tool_call&gt;NAME&lt;arg_key&gt;k&lt;/arg_key&gt;&lt;arg_value&gt;v&lt;/arg_value&gt;…&lt;/tool_call&gt;</c>.
/// </summary>
/// <remarks>
/// <para>Behaviour follows llama.cpp's <c>common_chat_params_init_glm_4_5</c>: several <c>&lt;tool_call&gt;</c> blocks
/// are parallel calls, the closing <c>&lt;/tool_call&gt;</c> is optional (the server's <c>&lt;/tool_call&gt;</c> stop
/// sequence trims it, and the turn then ends on the <c>&lt;|observation|&gt;</c> end-of-message token), and a block
/// whose last <c>&lt;arg_value&gt;</c> never closed is a truncated generation and is not reported (never execute
/// half a call). Anything before the first block (a closing <c>&lt;/think&gt;</c>, prose) is ignored.</para>
/// <para>Values are text. The template renders a non-string argument as JSON and a string verbatim
/// (<c>v | tojson if v is not string else v</c>), so a declared <c>string</c> parameter keeps its text and every
/// other declared type is parsed as JSON; with no schema the text is JSON when valid, else a string
/// (see <see cref="ToolArgumentCoercer.WriteTyped"/>). A value has at most one template newline stripped from each
/// end.</para>
/// </remarks>
public sealed class GlmToolCallParser : IToolCallParser
{
    private const string OpenTag = "<tool_call>";
    private const string CloseTag = "</tool_call>";
    private const string KeyOpen = "<arg_key>";
    private const string KeyClose = "</arg_key>";
    private const string ValueOpen = "<arg_value>";
    private const string ValueClose = "</arg_value>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText) => TryParse(generatedText, null);

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText, IReadOnlyList<ToolDefinition>? tools)
    {
        if (string.IsNullOrEmpty(generatedText))
            return null;

        var calls = new List<ToolCall>();
        int pos = 0;
        while (true)
        {
            int open = generatedText.IndexOf(OpenTag, pos, StringComparison.Ordinal);
            if (open < 0)
                break;
            int bodyStart = open + OpenTag.Length;
            int close = generatedText.IndexOf(CloseTag, bodyStart, StringComparison.Ordinal);
            int next = generatedText.IndexOf(OpenTag, bodyStart, StringComparison.Ordinal);
            int bodyEnd = close >= 0 && (next < 0 || close < next) ? close : next >= 0 ? next : generatedText.Length;

            if (TryParseBlock(generatedText.AsSpan(bodyStart, bodyEnd - bodyStart), tools, out var call))
                calls.Add(call with { Id = $"call_{calls.Count}" });

            pos = bodyEnd;
            if (pos < generatedText.Length && string.CompareOrdinal(generatedText, pos, CloseTag, 0, CloseTag.Length) == 0)
                pos += CloseTag.Length;
        }

        return calls.Count == 0 ? null : ToolArgumentCoercer.Coerce(calls.ToArray(), tools);
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(OpenTag, StringComparison.Ordinal);

    /// <summary>A function name is an identifier-like token (letters, digits and <c>_ - . : /</c>): rejects Hermes JSON bodies and prose.</summary>
    private static bool IsFunctionName(string name)
    {
        if (name.Length == 0)
            return false;
        foreach (char c in name)
            if (!char.IsLetterOrDigit(c) && c is not ('_' or '-' or '.' or ':' or '/'))
                return false;
        return true;
    }

    private static bool TryParseBlock(ReadOnlySpan<char> bodySpan, IReadOnlyList<ToolDefinition>? tools, out ToolCall call)
    {
        call = null!;
        string body = bodySpan.ToString();

        int firstKey = body.IndexOf(KeyOpen, StringComparison.Ordinal);
        string name = (firstKey >= 0 ? body[..firstKey] : body).Trim();
        if (!IsFunctionName(name))
            return false;

        JsonDocument? schemaDoc = null;
        try
        {
            JsonElement? props = null;
            if (ToolArgumentCoercer.Find(tools, name) is { } tool && !string.IsNullOrWhiteSpace(tool.ParametersSchema))
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
                int p = firstKey >= 0 ? firstKey : body.Length;
                while (p < body.Length)
                {
                    int ko = body.IndexOf(KeyOpen, p, StringComparison.Ordinal);
                    if (ko < 0)
                        break;
                    int kc = body.IndexOf(KeyClose, ko + KeyOpen.Length, StringComparison.Ordinal);
                    if (kc < 0)
                        return false;                       // truncated inside a key
                    string key = body.Substring(ko + KeyOpen.Length, kc - ko - KeyOpen.Length).Trim();

                    int vo = body.IndexOf(ValueOpen, kc + KeyClose.Length, StringComparison.Ordinal);
                    if (vo < 0)
                        return false;                       // a key without a value
                    int vs = vo + ValueOpen.Length;
                    int vc = body.IndexOf(ValueClose, vs, StringComparison.Ordinal);
                    if (vc < 0)
                    {
                        // Missing </arg_value>: acceptable only when the value is followed by the end of an
                        // explicitly closed block; a block cut off mid-value is a truncated generation.
                        return false;
                    }

                    if (key.Length > 0)
                    {
                        string raw = QwenXmlToolCallParser.StripTemplateNewlines(body.AsSpan(vs, vc - vs));
                        JsonElement? schema = props is { } pp && pp.TryGetProperty(key, out var ps) ? ps : null;
                        w.WritePropertyName(key);
                        ToolArgumentCoercer.WriteTyped(w, raw, schema);
                    }
                    p = vc + ValueClose.Length;
                }
                w.WriteEndObject();
            }

            call = new ToolCall("call_0", name, Encoding.UTF8.GetString(buffer.WrittenSpan));
            return true;
        }
        finally
        {
            schemaDoc?.Dispose();
        }
    }
}
