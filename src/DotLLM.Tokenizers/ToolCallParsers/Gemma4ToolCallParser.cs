using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses Gemma-4 tool calls:
/// <c>&lt;|tool_call&gt;call:get_weather{city:&lt;|"|&gt;Paris&lt;|"|&gt;,days:3}&lt;tool_call|&gt;</c>.
/// </summary>
/// <remarks>
/// <para>
/// The argument syntax is Gemma's own, not JSON: bare (unquoted) keys, strings delimited by the
/// <c>&lt;|"|&gt;</c> token, and bare numbers / <c>true</c> / <c>false</c> / <c>null</c>, with nested
/// <c>{...}</c> objects and <c>[...]</c> arrays. It mirrors llama.cpp's <c>common_chat_params_init_gemma4</c>
/// grammar. Strings are scanned for their closing delimiter, so values may freely contain
/// <c>{ } , :</c>.
/// </para>
/// <para>
/// Repeated <c>&lt;|tool_call&gt;</c> blocks are parallel calls. A missing <c>&lt;tool_call|&gt;</c> is tolerated
/// once the argument object has closed; an unterminated argument object (truncated generation) is not
/// reported. Whatever follows the last call (<c>&lt;eos&gt;</c>, <c>&lt;turn|&gt;</c>, stray special-token text)
/// is ignored.
/// </para>
/// </remarks>
public sealed class Gemma4ToolCallParser : IToolCallParser
{
    private const string Open = "<|tool_call>";
    private const string Close = "<tool_call|>";
    private const string CallPrefix = "call:";
    private const string Quote = "<|\"|>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        if (string.IsNullOrEmpty(generatedText))
            return null;

        var calls = new List<ToolCall>();
        int pos = 0;
        while (pos < generatedText.Length)
        {
            int open = generatedText.IndexOf(Open, pos, StringComparison.Ordinal);
            if (open < 0)
                break;

            int p = open + Open.Length;
            while (p < generatedText.Length && char.IsWhiteSpace(generatedText[p])) p++;
            if (string.CompareOrdinal(generatedText, p, CallPrefix, 0, CallPrefix.Length) != 0)
            {
                pos = open + Open.Length;
                continue;
            }

            p += CallPrefix.Length;
            int brace = generatedText.IndexOf('{', p);
            if (brace < 0)
                break;
            string name = generatedText.Substring(p, brace - p).Trim();
            if (name.Length == 0 || name.AsSpan().IndexOfAny('<', '\n', ' ') >= 0)
            {
                pos = brace;
                continue;
            }

            var buffer = new ArrayBufferWriter<byte>();
            bool ok;
            int end = brace;
            using (var w = new Utf8JsonWriter(buffer))
            {
                var r = new Reader(generatedText, brace);
                ok = r.TryWriteObject(w);
                end = r.Pos;
            }

            if (!ok)
            {
                pos = brace + 1;
                continue;
            }

            calls.Add(new ToolCall($"call_{calls.Count}", name, Encoding.UTF8.GetString(buffer.WrittenSpan)));
            pos = end;
            if (string.CompareOrdinal(generatedText, pos, Close, 0, Close.Length) == 0)
                pos += Close.Length;
        }

        return calls.Count > 0 ? calls.ToArray() : null;
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Open, StringComparison.Ordinal);

    /// <summary>Recursive-descent reader for Gemma's argument syntax, writing JSON as it goes.</summary>
    private ref struct Reader
    {
        private readonly string _s;
        public int Pos;
        private int _depth;

        public Reader(string s, int pos)
        {
            _s = s;
            Pos = pos;
            _depth = 0;
        }

        private void SkipWs()
        {
            while (Pos < _s.Length && char.IsWhiteSpace(_s[Pos])) Pos++;
        }

        public bool TryWriteObject(Utf8JsonWriter w)
        {
            if (++_depth > 32 || Pos >= _s.Length || _s[Pos] != '{')
                return false;
            Pos++;
            w.WriteStartObject();
            SkipWs();
            if (Pos < _s.Length && _s[Pos] == '}')
            {
                Pos++;
                w.WriteEndObject();
                _depth--;
                return true;
            }

            while (true)
            {
                SkipWs();
                int colon = _s.IndexOf(':', Pos);
                if (colon < 0)
                    return false;
                string key = _s.Substring(Pos, colon - Pos).Trim();
                // Keys never contain a closing brace or the string delimiter.
                if (key.Length == 0 || key.AsSpan().IndexOfAny('}', '<') >= 0)
                    return false;
                Pos = colon + 1;
                SkipWs();
                w.WritePropertyName(key);
                if (!TryWriteValue(w))
                    return false;
                SkipWs();
                if (Pos >= _s.Length)
                    return false;
                if (_s[Pos] == ',')
                {
                    Pos++;
                    continue;
                }
                if (_s[Pos] == '}')
                {
                    Pos++;
                    w.WriteEndObject();
                    _depth--;
                    return true;
                }
                return false;
            }
        }

        private bool TryWriteArray(Utf8JsonWriter w)
        {
            if (++_depth > 32)
                return false;
            Pos++; // [
            w.WriteStartArray();
            SkipWs();
            if (Pos < _s.Length && _s[Pos] == ']')
            {
                Pos++;
                w.WriteEndArray();
                _depth--;
                return true;
            }

            while (true)
            {
                SkipWs();
                if (!TryWriteValue(w))
                    return false;
                SkipWs();
                if (Pos >= _s.Length)
                    return false;
                if (_s[Pos] == ',')
                {
                    Pos++;
                    continue;
                }
                if (_s[Pos] == ']')
                {
                    Pos++;
                    w.WriteEndArray();
                    _depth--;
                    return true;
                }
                return false;
            }
        }

        private bool TryWriteValue(Utf8JsonWriter w)
        {
            if (Pos >= _s.Length)
                return false;

            if (string.CompareOrdinal(_s, Pos, Quote, 0, Quote.Length) == 0)
            {
                int start = Pos + Quote.Length;
                int end = _s.IndexOf(Quote, start, StringComparison.Ordinal);
                if (end < 0)
                    return false;
                w.WriteStringValue(_s.AsSpan(start, end - start));
                Pos = end + Quote.Length;
                return true;
            }

            char c = _s[Pos];
            if (c == '{')
            {
                return TryWriteObject(w);
            }
            if (c == '[')
                return TryWriteArray(w);

            // Bare token: number / true / false / null, else treated as a string.
            int s = Pos;
            while (Pos < _s.Length && _s[Pos] is not (',' or '}' or ']'))
                Pos++;
            string tok = _s.Substring(s, Pos - s).Trim();
            if (tok.Length == 0)
                return false;
            if (tok == "true") w.WriteBooleanValue(true);
            else if (tok == "false") w.WriteBooleanValue(false);
            else if (tok == "null") w.WriteNullValue();
            else if (double.TryParse(tok, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && double.IsFinite(d))
            {
                if (long.TryParse(tok, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long l))
                    w.WriteNumberValue(l);
                else
                    w.WriteNumberValue(d);
            }
            else
                w.WriteStringValue(tok);
            return true;
        }
    }
}
