using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Reasoning;

/// <summary>
/// Incremental splitter for OpenAI Harmony output (gpt-oss, #798). The model emits a sequence of channel
/// messages; this routes them:
/// <list type="bullet">
/// <item><c>&lt;|channel|&gt;analysis&lt;|message|&gt;…&lt;|end|&gt;</c> to the reasoning stream;</item>
/// <item><c>&lt;|channel|&gt;final&lt;|message|&gt;…</c> (and a commentary <i>preamble</i> with no recipient,
/// and any unknown channel) to the content stream;</item>
/// <item>a message with a recipient (<c>&lt;|channel|&gt;commentary to=functions.NAME &lt;|constrain|&gt;json&lt;|message|&gt;{…}&lt;|call|&gt;</c>,
/// or the recipient in the role header: <c>&lt;|start|&gt;assistant to=functions.NAME&lt;|channel|&gt;commentary json&lt;|message|&gt;</c>)
/// is passed through <b>raw</b> to the content stream, header and terminator included, so the tool-call
/// parser (<see cref="ToolCallParsers.HarmonyToolCallParser"/>) and the streaming suppressor see exactly what
/// they would on an unsplit output. The header is buffered until <c>&lt;|message|&gt;</c> so the marker is never
/// split across chunks.</item>
/// </list>
/// Like <see cref="ReasoningSplitter"/> it never emits part of a marker, drops whitespace at the edges of the
/// reasoning text, and is deterministic under any chunking: feeding one character at a time yields the same
/// concatenation as feeding everything at once. Consecutive messages of the same stream are joined with a blank
/// line. Mirrors llama.cpp's gpt-oss handling (analysis to <c>reasoning_content</c>, final to <c>content</c>,
/// recipient messages to tool calls).
/// </summary>
public sealed class HarmonySplitter : IReasoningSplitter
{
    /// <summary>Harmony <c>&lt;|channel|&gt;</c> token text.</summary>
    internal const string Channel = "<|channel|>";
    /// <summary>Harmony <c>&lt;|message|&gt;</c> token text.</summary>
    internal const string Message = "<|message|>";
    /// <summary>Harmony <c>&lt;|start|&gt;</c> token text.</summary>
    internal const string Start = "<|start|>";
    /// <summary>Harmony <c>&lt;|end|&gt;</c> (message terminator; NOT end of turn).</summary>
    internal const string End = "<|end|>";
    /// <summary>Harmony <c>&lt;|call|&gt;</c> (tool-call terminator; end of generation).</summary>
    internal const string Call = "<|call|>";
    /// <summary>Harmony <c>&lt;|return|&gt;</c> (final terminator; end of generation).</summary>
    internal const string Return = "<|return|>";

    /// <summary>The header that opens an analysis message, usable as a stop-gate open tag.</summary>
    public const string AnalysisOpen = Channel + "analysis" + Message;

    private static readonly string[] Consumed = [End, Call, Return];
    private static readonly string[] Boundaries = [End, Call, Return, Start, Channel];
    private static readonly string[] HeaderStarts = [Start, Channel, "to="];

    private static readonly Regex ChannelName = new(@"<\|channel\|>\s*([A-Za-z_]\w*)", RegexOptions.Compiled);
    private static readonly Regex Recipient = new(@"\bto=([^\s<]+)", RegexOptions.Compiled);

    private enum Mode { Idle, Header, Body }
    private enum Kind { Content, Reasoning, Tool }

    private readonly StringBuilder _buf = new();
    private readonly StringBuilder _reasoningOut = new();
    private readonly StringBuilder _contentOut = new();
    private Mode _mode = Mode.Idle;
    private Kind _kind;
    private bool _skipWs;
    private bool _sepReasoning, _sepContent;
    private bool _emittedReasoning, _emittedContent;
    private bool _anyOutput;
    private long _fed;
    private long _reasoningRawEnd = -1;
    private bool _everInReasoning;

    /// <inheritdoc/>
    public bool InReasoning
        => (_mode == Mode.Body && _kind == Kind.Reasoning)
           || (_mode == Mode.Header && _buf.ToString().Contains(Channel + "analysis", StringComparison.Ordinal));

    /// <inheritdoc/>
    public bool SawReasoning => _everInReasoning;

    /// <inheritdoc/>
    public long ReasoningRawLength => _reasoningRawEnd;

    /// <inheritdoc/>
    public ReasoningChunk Feed(string text)
    {
        if (text.Length == 0)
            return default;
        _fed += text.Length;
        _buf.Append(text);
        Process(final: false);
        return Drain();
    }

    /// <inheritdoc/>
    public ReasoningChunk Finish()
    {
        Process(final: true);
        return Drain();
    }

    /// <summary>One-shot split of a complete output.</summary>
    public static (string Reasoning, string Content, HarmonySplitter State) Split(string text)
    {
        var s = new HarmonySplitter();
        var a = s.Feed(text);
        var b = s.Finish();
        return (a.Reasoning + b.Reasoning, a.Content + b.Content, s);
    }

    private ReasoningChunk Drain()
    {
        string r = _reasoningOut.Length == 0 ? "" : _reasoningOut.ToString();
        string c = _contentOut.Length == 0 ? "" : _contentOut.ToString();
        _reasoningOut.Clear();
        _contentOut.Clear();
        return new ReasoningChunk(r, c);
    }

    private void Process(bool final)
    {
        while (true)
        {
            bool again = _mode switch
            {
                Mode.Idle => ProcessIdle(final),
                Mode.Header => ProcessHeader(final),
                _ => ProcessBody(final),
            };
            if (!again)
                return;
        }
    }

    // Between messages (and before the first): what comes next?
    private bool ProcessIdle(bool final)
    {
        string s = _buf.ToString();
        string t = s.TrimStart();
        if (t.Length == 0)
        {
            if (!final)
                return false;                       // undecided: may still become a header
            if (!_anyOutput)
                _contentOut.Append(s);              // whitespace-only generation: keep it, as the think splitter does
            _buf.Clear();
            return false;
        }

        foreach (var h in HeaderStarts)
        {
            if (t.StartsWith(h, StringComparison.Ordinal))
            {
                _buf.Remove(0, s.Length - t.Length);
                _mode = Mode.Header;
                return true;
            }
        }

        if (!final)
        {
            foreach (var h in HeaderStarts)
                if (t.Length < h.Length && h.StartsWith(t, StringComparison.Ordinal))
                    return false;                   // could still become a header marker
        }

        // Not Harmony markup at all (a model/template that does not use channels): plain answer text.
        _mode = Mode.Body;
        _kind = Kind.Content;
        StartMessage(Kind.Content);
        return true;
    }

    private bool ProcessHeader(bool final)
    {
        string s = _buf.ToString();
        int msg = s.IndexOf(Message, StringComparison.Ordinal);

        // A terminator before <|message|>: an aborted / empty header. Drop it.
        int term = IndexOfAny(s, Consumed, out int termLen);
        if (term >= 0 && (msg < 0 || term < msg))
        {
            _buf.Remove(0, term + termLen);
            _mode = Mode.Idle;
            return true;
        }

        if (msg < 0)
        {
            if (final)
                _buf.Clear();                       // header never completed (cut off): nothing to route
            return false;
        }

        string header = s.Substring(0, msg);
        string rest = s.Substring(msg + Message.Length);

        string? channel = ChannelName.Match(header) is { Success: true } cm ? cm.Groups[1].Value : null;
        string? recipient = Recipient.Match(header) is { Success: true } rm ? rm.Groups[1].Value : null;
        bool toolRecipient = recipient is not null
            && !recipient.Equals("assistant", StringComparison.OrdinalIgnoreCase)
            && !recipient.Equals("user", StringComparison.OrdinalIgnoreCase);

        Kind kind = toolRecipient ? Kind.Tool
            : string.Equals(channel, "analysis", StringComparison.Ordinal) ? Kind.Reasoning
            : Kind.Content;

        _buf.Clear().Append(rest);
        _mode = Mode.Body;
        _kind = kind;
        StartMessage(kind);
        if (kind == Kind.Tool)
        {
            _contentOut.Append(header).Append(Message);
            _anyOutput = true;
        }
        return true;
    }

    private bool ProcessBody(bool final)
    {
        if (_kind == Kind.Reasoning && _skipWs)
        {
            int ws = 0;
            while (ws < _buf.Length && char.IsWhiteSpace(_buf[ws]))
                ws++;
            if (ws > 0)
                _buf.Remove(0, ws);
            if (_buf.Length == 0)
            {
                if (final)
                    _reasoningRawEnd = _fed;
                return false;
            }
            _skipWs = false;
        }

        string s = _buf.ToString();
        int idx = IndexOfAny(s, Boundaries, out int len);
        if (idx >= 0)
        {
            EmitBody(s.Substring(0, idx), closing: true);
            bool consumed = Array.IndexOf(Consumed, s.Substring(idx, len)) >= 0;
            if (consumed && _kind == Kind.Tool)
                _contentOut.Append(s, idx, len);            // keep the terminator in the raw tool segment
            int restFrom = consumed ? idx + len : idx;
            _buf.Clear().Append(s, restFrom, s.Length - restFrom);
            if (_kind == Kind.Reasoning)
                _reasoningRawEnd = _fed - _buf.Length;
            _mode = Mode.Idle;
            return true;
        }

        if (final)
        {
            EmitBody(s, closing: true);
            _buf.Clear();
            if (_kind == Kind.Reasoning)
                _reasoningRawEnd = _fed;
            return false;
        }

        int hold = 0;
        foreach (var m in Boundaries)
            hold = Math.Max(hold, PartialTagSuffix(s, m));
        int safe = s.Length - hold;
        if (_kind == Kind.Reasoning)
            safe = s.AsSpan(0, safe).TrimEnd().Length;     // trailing whitespace is held until text or the end follows
        if (safe > 0)
        {
            EmitBody(s.Substring(0, safe), closing: false);
            _buf.Remove(0, safe);
        }
        return false;
    }

    private void StartMessage(Kind kind)
    {
        switch (kind)
        {
            case Kind.Reasoning:
                _sepReasoning = true;
                _skipWs = true;
                _everInReasoning = true;
                _reasoningRawEnd = -1;
                break;
            case Kind.Content:
                _sepContent = true;
                break;
        }
    }

    private void EmitBody(string text, bool closing)
    {
        if (text.Length == 0)
            return;
        switch (_kind)
        {
            case Kind.Reasoning:
                if (closing)
                    text = text.TrimEnd();
                if (text.Length == 0)
                    return;
                if (_sepReasoning && _emittedReasoning)
                    _reasoningOut.Append("\n\n");
                _sepReasoning = false;
                _reasoningOut.Append(text);
                _emittedReasoning = true;
                break;
            case Kind.Content:
                if (_sepContent && _emittedContent)
                    _contentOut.Append("\n\n");
                _sepContent = false;
                _contentOut.Append(text);
                _emittedContent = true;
                _anyOutput = true;
                break;
            default:
                _contentOut.Append(text);                   // tool segment: raw
                _anyOutput = true;
                break;
        }
    }

    private static int IndexOfAny(string s, string[] markers, out int length)
    {
        int best = -1;
        length = 0;
        foreach (var m in markers)
        {
            int i = s.IndexOf(m, StringComparison.Ordinal);
            if (i >= 0 && (best < 0 || i < best))
            {
                best = i;
                length = m.Length;
            }
        }
        return best;
    }

    private static int PartialTagSuffix(string s, string tag)
    {
        int max = Math.Min(tag.Length - 1, s.Length);
        for (int len = max; len > 0; len--)
        {
            if (string.CompareOrdinal(s, s.Length - len, tag, 0, len) == 0)
                return len;
        }
        return 0;
    }
}
