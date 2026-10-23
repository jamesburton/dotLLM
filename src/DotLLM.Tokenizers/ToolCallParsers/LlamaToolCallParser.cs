using System.Text.Json;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls from Llama 3.x output.
/// </summary>
/// <remarks>
/// <para>
/// Accepted shapes after <c>&lt;|python_tag|&gt;</c> (or bare, when the whole response is the call):
/// </para>
/// <list type="bullet">
/// <item><c>{"name": "f", "parameters": {...}}</c> — the documented Llama 3.1 form (also <c>arguments</c>).</item>
/// <item><c>{"type": "function", "function": "f", "parameters": {...}}</c> — Llama-3.2-1B's habit.</item>
/// <item><c>{"type": "function", "function": {"name": "f", "arguments": {...}}}</c> — OpenAI envelope.</item>
/// <item>Several of the above as a JSON array or separated by <c>;</c> / whitespace (parallel calls).</item>
/// <item><c>brave_search.call(query="...")</c> — the Llama 3.1 built-in-tool (pythonic) form.</item>
/// </list>
/// </remarks>
public sealed class LlamaToolCallParser : IToolCallParser
{
    private const string Marker = "<|python_tag|>";

    // name.call(k="v", n=1) — Llama 3.1 builtin tools (brave_search, wolfram_alpha, code_interpreter).
    private static readonly Regex BuiltinCall = new(
        @"^\s*(?<name>[A-Za-z_][A-Za-z0-9_]*)\.call\((?<args>.*)\)\s*(?:<\|eom_id\|>|<\|eot_id\|>)?\s*$",
        RegexOptions.Compiled | RegexOptions.Singleline);

    private static readonly Regex KwArg = new(
        @"(?<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:""(?<s>(?:[^""\\]|\\.)*)""|'(?<s2>(?:[^'\\]|\\.)*)'|(?<v>[^,\s)]+))",
        RegexOptions.Compiled);

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        // 1. Explicit <|python_tag|> marker — definitive tool call signal
        int markerIndex = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (markerIndex >= 0)
        {
            string afterMarker = generatedText[(markerIndex + Marker.Length)..];

            var builtin = TryParseBuiltin(afterMarker);
            if (builtin is not null)
                return builtin;

            return ToolCallJsonHelper.ParseAll(afterMarker, requireWholeText: false);
        }

        // 2. No marker — Llama 3.2 lightweight models may omit <|python_tag|>.
        //    If the ENTIRE response is a JSON tool call (starts with { or [),
        //    treat it as a tool call. If there's prose before the JSON,
        //    the model is quoting a schema in its text response — not a tool call.
        string trimmed = generatedText.Trim();
        if (trimmed.Length > 0 && trimmed[0] is '{' or '[')
            return ToolCallJsonHelper.ParseAll(trimmed, requireWholeText: true);

        return null;
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);

    private static ToolCall[]? TryParseBuiltin(string text)
    {
        var m = BuiltinCall.Match(text);
        if (!m.Success)
            return null;

        var buffer = new System.Buffers.ArrayBufferWriter<byte>();
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            foreach (Match a in KwArg.Matches(m.Groups["args"].Value))
            {
                w.WritePropertyName(a.Groups["k"].Value);
                if (a.Groups["s"].Success)
                    w.WriteStringValue(Regex.Unescape(a.Groups["s"].Value));
                else if (a.Groups["s2"].Success)
                    w.WriteStringValue(Regex.Unescape(a.Groups["s2"].Value));
                else
                {
                    string v = a.Groups["v"].Value;
                    if (long.TryParse(v, out long l)) w.WriteNumberValue(l);
                    else if (v is "True" or "true") w.WriteBooleanValue(true);
                    else if (v is "False" or "false") w.WriteBooleanValue(false);
                    else w.WriteStringValue(v);
                }
            }
            w.WriteEndObject();
        }

        return [new ToolCall("call_0", m.Groups["name"].Value, System.Text.Encoding.UTF8.GetString(buffer.WrittenSpan))];
    }
}
