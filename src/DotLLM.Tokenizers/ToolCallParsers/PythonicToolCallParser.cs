using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls written in Pythonic call syntax —
/// <c>function_name(arg=value, other=value)</c> — as emitted by SmolLM3's
/// <c>python_tools</c> chat-template branch and similar
/// Python-DSL-flavoured tool-calling formats.
/// </summary>
/// <remarks>
/// <para>
/// <b>Grammar (best-effort, model-output-tolerant).</b>
/// </para>
/// <code>
///   tool_calls   ::= call (separator call)*
///   separator    ::= ";" | newline | "," (at top level only)
///   call         ::= identifier "(" (kwarg ("," kwarg)*)? ")"
///   kwarg        ::= identifier "=" pyvalue
///   pyvalue      ::= str_literal | num_literal | true | false | None | list | dict | identifier
///   str_literal  ::= "..." | '...' (escapes recognised: \\ \" \' \n \r \t)
///   list         ::= "[" (pyvalue ("," pyvalue)*)? "]"
///   dict         ::= "{" (str_literal ":" pyvalue ("," str_literal ":" pyvalue)*)? "}"
/// </code>
/// <para>
/// <b>Surrounding text.</b> A canonical SmolLM3 emission wraps the calls
/// inside an open/close marker, but most checkpoints we have seen ship
/// the bare expression on its own line. The parser scans the text for
/// an identifier followed by '(' and parses from there; preceding prose
/// is ignored. Optional opening marker
/// <c>&lt;tool_call&gt;</c> / <c>&lt;|python_call|&gt;</c> is consumed when
/// present so the same parser handles wrapped and bare emissions.
/// </para>
/// <para>
/// <b>Argument serialisation.</b> Each kwarg's right-hand-side is
/// converted to its JSON form: strings → JSON strings, numbers → JSON
/// numbers (decimal-point preserved as in source for fidelity), booleans
/// → <c>true</c> / <c>false</c>, <c>None</c> → <c>null</c>, lists / dicts
/// recurse. The final per-call <see cref="ToolCall.Arguments"/> is a JSON
/// object whose keys are the kwarg names and whose values are the
/// converted forms.
/// </para>
/// </remarks>
public sealed class PythonicToolCallParser : IToolCallParser
{
    // SmolLM3's `python_tools` chat-template branch wraps the python-call
    // payload in `<tool_call>` tags (same outer envelope as Hermes/XML) but
    // emits a Pythonic call inside instead of JSON. Bare emissions skip the
    // wrapper. Both shapes are recognised — the wrapper is consumed if seen.
    private const string OpenTag = "<tool_call>";
    private const string CloseTag = "</tool_call>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        if (string.IsNullOrEmpty(generatedText))
            return null;

        string text = StripWrapperTags(generatedText);

        var calls = new List<ToolCall>();
        int pos = 0;
        int callIndex = 0;
        while (pos < text.Length)
        {
            SkipWhitespaceAndSeparators(text, ref pos);
            if (pos >= text.Length)
                break;

            int callStart = pos;
            if (TryParseCall(text, ref pos, out string? name, out string? args))
            {
                calls.Add(new ToolCall($"call_{callIndex++}", name!, args!));
            }
            else
            {
                // Could not parse a call here — advance one char so we don't
                // loop forever on noise. The IsToolCallStart heuristic should
                // have prevented entry on pure prose, but be defensive.
                if (pos == callStart)
                    pos++;
            }
        }

        return calls.Count > 0 ? calls.ToArray() : null;
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
    {
        if (string.IsNullOrEmpty(text))
            return false;

        // Wrapper present → it's a call regardless of payload shape.
        if (text.Contains(OpenTag, StringComparison.Ordinal))
            return true;

        // Otherwise look for an identifier followed directly by '(' in the
        // first line — this is the recognisable Pythonic call signature.
        // We need the identifier to look like a function name (letter / _
        // start, alnum / _ continuation) so plain words don't match.
        int n = text.Length;
        for (int i = 0; i < n; i++)
        {
            if (!IsIdentStart(text[i]))
                continue;
            int j = i;
            while (j < n && IsIdentChar(text[j])) j++;
            if (j < n && text[j] == '(')
                return true;
            // Skip the rest of this word to avoid scanning it character-by-character.
            i = j;
        }
        return false;
    }

    private static string StripWrapperTags(string s)
    {
        int open = s.IndexOf(OpenTag, StringComparison.Ordinal);
        if (open < 0)
            return s;
        int payloadStart = open + OpenTag.Length;
        int close = s.IndexOf(CloseTag, payloadStart, StringComparison.Ordinal);
        if (close < 0)
            return s[payloadStart..];
        return s[payloadStart..close];
    }

    // ─────────────────────────────────────────────────────────────────────
    // Parser
    // ─────────────────────────────────────────────────────────────────────

    private static bool TryParseCall(string text, ref int pos, out string? name, out string? jsonArgs)
    {
        name = null;
        jsonArgs = null;

        // Identifier
        SkipSpacesAndTabs(text, ref pos);
        int idStart = pos;
        if (pos >= text.Length || !IsIdentStart(text[pos]))
            return false;
        pos++;
        while (pos < text.Length && IsIdentChar(text[pos])) pos++;
        if (pos == idStart)
            return false;
        string ident = text[idStart..pos];

        // '('
        SkipSpacesAndTabs(text, ref pos);
        if (pos >= text.Length || text[pos] != '(')
            return false;
        pos++;

        // kwargs
        var sb = new StringBuilder();
        sb.Append('{');
        bool first = true;
        SkipSpacesAndTabs(text, ref pos);
        while (pos < text.Length && text[pos] != ')')
        {
            SkipSpacesAndTabs(text, ref pos);
            int kStart = pos;
            if (pos >= text.Length || !IsIdentStart(text[pos]))
                return false;
            pos++;
            while (pos < text.Length && IsIdentChar(text[pos])) pos++;
            if (pos == kStart)
                return false;
            string key = text[kStart..pos];

            SkipSpacesAndTabs(text, ref pos);
            if (pos >= text.Length || text[pos] != '=')
                return false;
            pos++;
            SkipSpacesAndTabs(text, ref pos);

            if (!TryParseValue(text, ref pos, out string? jsonValue))
                return false;

            if (!first) sb.Append(',');
            first = false;
            sb.Append('"').Append(JsonEscape(key)).Append("\":").Append(jsonValue);

            SkipSpacesAndTabs(text, ref pos);
            if (pos < text.Length && text[pos] == ',')
            {
                pos++;
                continue;
            }
        }

        if (pos >= text.Length || text[pos] != ')')
            return false;
        pos++; // consume ')'
        sb.Append('}');

        // Validate the JSON we just synthesised — the parser composes a
        // well-formed object from validated literals, but a final round-trip
        // confirms the model didn't sneak in a Python literal we silently
        // pass-through as garbage.
        try
        {
            using var _ = JsonDocument.Parse(sb.ToString());
        }
        catch (JsonException)
        {
            return false;
        }

        name = ident;
        jsonArgs = sb.ToString();
        return true;
    }

    private static bool TryParseValue(string text, ref int pos, out string? jsonValue)
    {
        jsonValue = null;
        if (pos >= text.Length) return false;
        char c = text[pos];
        if (c == '"' || c == '\'')
            return TryParseString(text, ref pos, c, out jsonValue);
        if (c == '[')
            return TryParseList(text, ref pos, out jsonValue);
        if (c == '{')
            return TryParseDict(text, ref pos, out jsonValue);
        if (c == '-' || c == '+' || (c >= '0' && c <= '9'))
            return TryParseNumber(text, ref pos, out jsonValue);
        // Bare keywords: True / False / None / true / false / null. SmolLM3's
        // `python_tools` template canonicalises to Python casing; we accept
        // the JSON spellings too because models drift.
        if (IsIdentStart(c))
            return TryParseKeyword(text, ref pos, out jsonValue);
        return false;
    }

    private static bool TryParseString(string text, ref int pos, char quote, out string? jsonValue)
    {
        jsonValue = null;
        pos++; // consume opening quote
        var sb = new StringBuilder();
        while (pos < text.Length)
        {
            char c = text[pos++];
            if (c == quote)
            {
                jsonValue = "\"" + JsonEscape(sb.ToString()) + "\"";
                return true;
            }
            if (c == '\\' && pos < text.Length)
            {
                char esc = text[pos++];
                sb.Append(esc switch
                {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    _ => esc,
                });
                continue;
            }
            sb.Append(c);
        }
        return false; // unterminated
    }

    private static bool TryParseNumber(string text, ref int pos, out string? jsonValue)
    {
        jsonValue = null;
        int start = pos;
        if (text[pos] == '+' || text[pos] == '-') pos++;
        bool sawDigit = false;
        while (pos < text.Length && text[pos] >= '0' && text[pos] <= '9') { pos++; sawDigit = true; }
        if (pos < text.Length && text[pos] == '.')
        {
            pos++;
            while (pos < text.Length && text[pos] >= '0' && text[pos] <= '9') { pos++; sawDigit = true; }
        }
        if (pos < text.Length && (text[pos] == 'e' || text[pos] == 'E'))
        {
            pos++;
            if (pos < text.Length && (text[pos] == '+' || text[pos] == '-')) pos++;
            while (pos < text.Length && text[pos] >= '0' && text[pos] <= '9') pos++;
        }
        if (!sawDigit) return false;

        string raw = text[start..pos];
        // Normalise leading '+' to nothing for JSON conformance.
        if (raw.StartsWith('+')) raw = raw[1..];
        // JSON forbids "1." style — promote to "1.0" so the round-trip parses.
        if (raw.EndsWith('.')) raw += "0";
        jsonValue = raw;
        return true;
    }

    private static bool TryParseList(string text, ref int pos, out string? jsonValue)
    {
        jsonValue = null;
        pos++; // [
        var sb = new StringBuilder();
        sb.Append('[');
        SkipSpacesAndTabs(text, ref pos);
        bool first = true;
        while (pos < text.Length && text[pos] != ']')
        {
            SkipSpacesAndTabs(text, ref pos);
            if (!TryParseValue(text, ref pos, out string? element))
                return false;
            if (!first) sb.Append(',');
            first = false;
            sb.Append(element);
            SkipSpacesAndTabs(text, ref pos);
            if (pos < text.Length && text[pos] == ',') { pos++; continue; }
        }
        if (pos >= text.Length || text[pos] != ']') return false;
        pos++;
        sb.Append(']');
        jsonValue = sb.ToString();
        return true;
    }

    private static bool TryParseDict(string text, ref int pos, out string? jsonValue)
    {
        jsonValue = null;
        pos++; // {
        var sb = new StringBuilder();
        sb.Append('{');
        SkipSpacesAndTabs(text, ref pos);
        bool first = true;
        while (pos < text.Length && text[pos] != '}')
        {
            SkipSpacesAndTabs(text, ref pos);
            if (pos >= text.Length) return false;
            char qc = text[pos];
            if (qc != '"' && qc != '\'') return false;
            if (!TryParseString(text, ref pos, qc, out string? keyJson))
                return false;
            SkipSpacesAndTabs(text, ref pos);
            if (pos >= text.Length || text[pos] != ':') return false;
            pos++;
            SkipSpacesAndTabs(text, ref pos);
            if (!TryParseValue(text, ref pos, out string? valJson))
                return false;

            if (!first) sb.Append(',');
            first = false;
            sb.Append(keyJson).Append(':').Append(valJson);

            SkipSpacesAndTabs(text, ref pos);
            if (pos < text.Length && text[pos] == ',') { pos++; continue; }
        }
        if (pos >= text.Length || text[pos] != '}') return false;
        pos++;
        sb.Append('}');
        jsonValue = sb.ToString();
        return true;
    }

    private static bool TryParseKeyword(string text, ref int pos, out string? jsonValue)
    {
        int start = pos;
        while (pos < text.Length && IsIdentChar(text[pos])) pos++;
        string kw = text[start..pos];
        jsonValue = kw switch
        {
            "True" or "true" => "true",
            "False" or "false" => "false",
            "None" or "null" => "null",
            _ => null,
        };
        return jsonValue is not null;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────

    private static bool IsIdentStart(char c) => (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';

    private static bool IsIdentChar(char c) =>
        IsIdentStart(c) || (c >= '0' && c <= '9');

    private static void SkipSpacesAndTabs(string text, ref int pos)
    {
        while (pos < text.Length && (text[pos] == ' ' || text[pos] == '\t')) pos++;
    }

    /// <summary>
    /// Skips whitespace (incl. newlines) and inter-call separators (';',
    /// newline, top-level ','). Newlines after a call are the most common
    /// SmolLM3 separator; ';' is the Python-list canonical separator the
    /// HF template uses; ',' is tolerated for permissive parsing.
    /// </summary>
    private static void SkipWhitespaceAndSeparators(string text, ref int pos)
    {
        while (pos < text.Length)
        {
            char c = text[pos];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ';' || c == ',')
                pos++;
            else
                break;
        }
    }

    private static string JsonEscape(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20)
                        sb.Append("\\u").Append(((int)c).ToString("X4", CultureInfo.InvariantCulture));
                    else
                        sb.Append(c);
                    break;
            }
        }
        return sb.ToString();
    }
}
