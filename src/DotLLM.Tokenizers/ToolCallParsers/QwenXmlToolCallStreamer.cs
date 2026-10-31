using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>One streamed fragment of a tool call (#1342): the first carries <see cref="Id"/> and <see cref="Name"/>.</summary>
/// <param name="Index">Position in the assistant's tool-call list.</param>
/// <param name="Id">Call id; non-null on the first delta only.</param>
/// <param name="Name">Function name; non-null on the first delta only.</param>
/// <param name="ArgumentsFragment">Next piece of the arguments JSON (empty on the first delta).</param>
public readonly record struct ToolCallStreamDelta(int Index, string? Id, string? Name, string ArgumentsFragment);

/// <summary>
/// Incrementally streams the FIRST call of the Qwen3-Coder XML tool-call format (#1342, ds4 #1251): the call's id and name
/// as soon as <c>&lt;function=NAME&gt;</c> is complete, then the arguments JSON in fragments. Parameters the tool's schema
/// declares as strings stream their text as it arrives, escaped exactly as <see cref="QwenXmlToolCallParser"/> escapes
/// them (framing newlines stripped); every other parameter is buffered to its closing tag so it gets the right JSON
/// representation. The concatenated fragments equal <see cref="QwenXmlToolCallParser"/>'s arguments for the same text.
/// </summary>
/// <remarks>
/// <para>
/// Only call 0 streams; later calls are the caller's to emit once complete (with indexes from 1). A call whose body is
/// Hermes JSON, whose name is not a declared tool, or that is malformed before the header completes is left alone
/// (<see cref="Disabled"/>): the caller's whole-text path handles it as before.
/// </para>
/// <para>
/// A call cut off mid-arguments (<c>max_tokens</c>, client abort) has already shown the client a prefix of the arguments.
/// <see cref="Started"/> without <see cref="Completed"/> tells the caller that happened: it must finish with the real
/// finish reason (never <c>tool_calls</c>) and must not also deliver the call text as content.
/// </para>
/// </remarks>
public sealed class QwenXmlToolCallStreamer
{
    private const string OpenTag = "<tool_call>";
    private const string FunctionOpen = "<function=";
    private const string FunctionClose = "</function>";
    private const string ParamOpen = "<parameter=";
    private const string ParamClose = "</parameter>";

    private enum State { Seek, AfterOpen, Header, Between, ParamKey, StringValue, BufferedValue, Done, Disabled }

    private readonly IReadOnlyList<ToolDefinition>? _tools;
    private State _state = State.Seek;
    private string _rest = "";
    private JsonElement? _props;
    private string _key = "";
    private JsonElement? _keySchema;
    private bool _wroteParam;
    private bool _leadingDone;
    private string _held = "";

    /// <summary>Creates a streamer for one response.</summary>
    /// <param name="tools">The request's tools; the first call is streamed only if it names one of them.</param>
    public QwenXmlToolCallStreamer(IReadOnlyList<ToolDefinition>? tools) => _tools = tools;

    /// <summary>True once the first delta (id and name) has been returned.</summary>
    public bool Started { get; private set; }

    /// <summary>True once the call's closing <c>&lt;/function&gt;</c> arrived and the arguments object was closed.</summary>
    public bool Completed => _state == State.Done;

    /// <summary>True when this response's first call is not streamable; the caller's whole-text path owns it.</summary>
    public bool Disabled => _state == State.Disabled;

    /// <summary>Feeds the next piece of answer text (prose included) and returns the deltas it completes.</summary>
    /// <param name="text">Incremental answer text.</param>
    /// <returns>Zero or more deltas, in order.</returns>
    public List<ToolCallStreamDelta> Feed(string text)
    {
        var output = new List<ToolCallStreamDelta>();
        if (_state is State.Done or State.Disabled || text.Length == 0)
            return output;
        _rest += text;
        while (Step(output)) { }
        return output;
    }

    private void Emit(List<ToolCallStreamDelta> output, string fragment)
    {
        if (fragment.Length > 0)
            output.Add(new ToolCallStreamDelta(0, null, null, fragment));
    }

    /// <summary>Advances the state machine once; false when it needs more text.</summary>
    private bool Step(List<ToolCallStreamDelta> output)
    {
        switch (_state)
        {
            case State.Seek:
            {
                int open = _rest.IndexOf(OpenTag, StringComparison.Ordinal);
                int bare = _rest.IndexOf(FunctionOpen, StringComparison.Ordinal);
                if (open >= 0 && (bare < 0 || open < bare))
                {
                    _rest = _rest[(open + OpenTag.Length)..];
                    _state = State.AfterOpen;
                    return true;
                }
                if (bare >= 0)
                {
                    _rest = _rest[bare..];
                    _state = State.Header;
                    return true;
                }
                // No marker yet: keep only a tail that could still grow into one.
                int keep = Math.Min(_rest.Length, OpenTag.Length - 1);
                _rest = _rest[^keep..];
                return false;
            }

            case State.AfterOpen:
            {
                _rest = _rest.TrimStart();
                if (_rest.Length == 0)
                    return false;
                if (_rest.StartsWith(FunctionOpen, StringComparison.Ordinal))
                {
                    _state = State.Header;
                    return true;
                }
                if (FunctionOpen.StartsWith(_rest, StringComparison.Ordinal))
                    return false;
                _state = State.Disabled;   // a Hermes JSON body: not ours
                return false;
            }

            case State.Header:
            {
                int nameEnd = _rest.IndexOf('>', FunctionOpen.Length);
                if (nameEnd < 0)
                {
                    if (_rest.AsSpan(FunctionOpen.Length).IndexOfAny('<', '\n') >= 0)
                        _state = State.Disabled;
                    return false;
                }
                string name = _rest.Substring(FunctionOpen.Length, nameEnd - FunctionOpen.Length).Trim();
                var tool = ToolArgumentCoercer.Find(_tools, name);
                if (name.Length == 0 || name.AsSpan().IndexOfAny('<', '\n') >= 0 || tool is null)
                {
                    _state = State.Disabled;
                    return false;
                }

                _props = null;
                if (!string.IsNullOrWhiteSpace(tool.ParametersSchema))
                {
                    try
                    {
                        using var doc = JsonDocument.Parse(tool.ParametersSchema);
                        if (doc.RootElement.ValueKind == JsonValueKind.Object
                            && doc.RootElement.TryGetProperty("properties", out var pr) && pr.ValueKind == JsonValueKind.Object)
                            _props = pr.Clone();
                    }
                    catch (JsonException)
                    {
                    }
                }

                _rest = _rest[(nameEnd + 1)..];
                Started = true;
                output.Add(new ToolCallStreamDelta(0, "call_0", name, ""));
                Emit(output, "{");
                _state = State.Between;
                return true;
            }

            case State.Between:
            {
                int param = _rest.IndexOf(ParamOpen, StringComparison.Ordinal);
                int close = _rest.IndexOf(FunctionClose, StringComparison.Ordinal);
                if (close >= 0 && (param < 0 || close < param))
                {
                    Emit(output, "}");
                    _rest = _rest[(close + FunctionClose.Length)..];
                    _state = State.Done;
                    return false;
                }
                if (param >= 0)
                {
                    _rest = _rest[(param + ParamOpen.Length)..];
                    _state = State.ParamKey;
                    return true;
                }
                // Between parameters only whitespace is expected; keep a tail that could grow into a tag.
                int keep = Math.Min(_rest.Length, FunctionClose.Length - 1);
                _rest = _rest[^keep..];
                return false;
            }

            case State.ParamKey:
            {
                int keyEnd = _rest.IndexOf('>');
                if (keyEnd < 0)
                    return false;
                _key = _rest[..keyEnd].Trim();
                _rest = _rest[(keyEnd + 1)..];
                _keySchema = _props is { } p && _key.Length > 0 && p.TryGetProperty(_key, out var ks) ? ks : null;
                _leadingDone = false;
                _held = "";
                if (_key.Length > 0 && _keySchema is { } s && ToolArgumentCoercer.DeclaresString(s))
                {
                    Emit(output, (_wroteParam ? "," : "") + "\"" + JsonEncodedText.Encode(_key) + "\":\"");
                    _wroteParam = true;
                    _state = State.StringValue;
                }
                else
                {
                    _state = State.BufferedValue;
                }
                return true;
            }

            case State.StringValue:
                return StepStringValue(output);

            case State.BufferedValue:
            {
                int end = ValueEnd(out bool isParamClose);
                if (end < 0)
                    return false;
                if (_key.Length > 0)
                {
                    string raw = QwenXmlToolCallParser.StripTemplateNewlines(_rest.AsSpan(0, end));
                    Emit(output, (_wroteParam ? "," : "") + TypedProperty(_key, raw, _keySchema));
                    _wroteParam = true;
                }
                _rest = _rest[(end + (isParamClose ? ParamClose.Length : 0))..];
                _state = State.Between;
                return true;
            }

            default:
                return false;
        }
    }

    /// <summary>Index of the first <c>&lt;/parameter&gt;</c> (or, failing that, <c>&lt;/function&gt;</c>) in the unread text; -1 when neither is there yet.</summary>
    private int ValueEnd(out bool isParamClose)
    {
        int pc = _rest.IndexOf(ParamClose, StringComparison.Ordinal);
        int fc = _rest.IndexOf(FunctionClose, StringComparison.Ordinal);
        isParamClose = pc >= 0 && (fc < 0 || pc < fc);
        return isParamClose ? pc : fc;
    }

    private bool StepStringValue(List<ToolCallStreamDelta> output)
    {
        if (!_leadingDone)
        {
            // Remove exactly one template newline (CRLF tolerated) from the front, as the parser does.
            if (_rest.Length == 0)
                return false;
            if (_rest[0] == '\n')
                _rest = _rest[1..];
            else if (_rest[0] == '\r')
            {
                if (_rest.Length < 2)
                    return false;
                if (_rest[1] == '\n')
                    _rest = _rest[2..];
            }
            _leadingDone = true;
        }

        int end = ValueEnd(out bool isParamClose);
        if (end >= 0)
        {
            string total = _held + _rest[..end];
            if (total.EndsWith("\r\n", StringComparison.Ordinal))
                total = total[..^2];
            else if (total.EndsWith('\n'))
                total = total[..^1];
            Emit(output, JsonEncodedText.Encode(total).ToString());
            Emit(output, "\"");
            _held = "";
            _rest = _rest[(end + (isParamClose ? ParamClose.Length : 0))..];
            _state = State.Between;
            return true;
        }

        // No terminator yet: release everything except a tail that could be the start of one, a trailing newline run (it may
        // be the framing newline) and a lone high surrogate (its pair has not arrived).
        int safe = _rest.Length - TagPrefixTail(_rest);
        string text = _held + _rest[..safe];
        _rest = _rest[safe..];
        int hold = 0;
        while (hold < text.Length && (text[text.Length - 1 - hold] is '\n' or '\r'))
            hold++;
        if (hold == 0 && text.Length > 0 && char.IsHighSurrogate(text[^1]))
            hold = 1;
        _held = text[(text.Length - hold)..];
        Emit(output, JsonEncodedText.Encode(text[..(text.Length - hold)]).ToString());
        return false;
    }

    /// <summary>Length of the longest suffix of <paramref name="s"/> that is a proper prefix of a value terminator.</summary>
    private static int TagPrefixTail(string s)
    {
        int best = 0;
        foreach (string tag in (ReadOnlySpan<string>)[ParamClose, FunctionClose])
            for (int len = Math.Min(tag.Length - 1, s.Length); len > best; len--)
                if (s.AsSpan(s.Length - len).SequenceEqual(tag.AsSpan(0, len)))
                {
                    best = len;
                    break;
                }
        return best;
    }

    /// <summary>Renders <c>"key":value</c> exactly as the parser's writer would.</summary>
    private static string TypedProperty(string key, string raw, JsonElement? schema)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WritePropertyName(key);
            ToolArgumentCoercer.WriteTyped(w, raw, schema);
            w.WriteEndObject();
        }
        string json = Encoding.UTF8.GetString(buffer.WrittenSpan);
        return json[1..^1];
    }
}
