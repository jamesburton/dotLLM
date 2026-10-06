using System.Text;

namespace DotLLM.Tokenizers.Reasoning;

/// <summary>One increment of splitter output: text for <c>reasoning_content</c> and text for <c>content</c>.</summary>
/// <param name="Reasoning">Reasoning text (empty when none).</param>
/// <param name="Content">Answer text (empty when none).</param>
public readonly record struct ReasoningChunk(string Reasoning, string Content)
{
    /// <summary>True when neither stream produced text.</summary>
    public bool IsEmpty => Reasoning.Length == 0 && Content.Length == 0;
}

/// <summary>
/// Incremental splitter of model output into reasoning and answer around
/// <c>&lt;think&gt;…&lt;/think&gt;</c> (#767). Feed it decoded token text as it streams; it never emits
/// part of a tag, so a tag split across tokens (<c>"&lt;/th"</c> + <c>"ink&gt;"</c>) is invisible to the
/// consumer. Non-streaming use is the same machine: <see cref="Feed"/> the whole text, then
/// <see cref="Finish"/>.
/// </summary>
/// <remarks>
/// <para>Whitespace policy, chosen so the streamed concatenation equals the non-streamed result:
/// whitespace right after <c>&lt;think&gt;</c> and right before <c>&lt;/think&gt;</c> is dropped, and
/// whitespace right after <c>&lt;/think&gt;</c> is dropped from the answer (the template's
/// <c>\n\n</c> separator). Trailing whitespace is <i>held back</i> in a stream until a non-whitespace
/// character or the closing tag arrives, because it cannot be un-sent.</para>
/// <para>An unclosed block (generation hit <c>max_tokens</c> mid-thought) leaves everything as
/// reasoning and <c>content</c> empty.</para>
/// </remarks>
public sealed class ReasoningSplitter
{
    private enum Mode { Detect, Reasoning, Content }

    private const string Open = ReasoningFormats.OpenTag;
    private const string Close = ReasoningFormats.CloseTag;

    private readonly bool _detectAnywhere;
    private readonly StringBuilder _buf = new();
    private readonly StringBuilder _reasoningOut = new();
    private readonly StringBuilder _contentOut = new();
    private Mode _mode;
    private bool _skipReasoningWs;
    private bool _skipContentWs;
    private long _fed;
    private long _reasoningRawEnd = -1;
    private bool _everInReasoning;

    /// <summary>Creates a splitter.</summary>
    /// <param name="startInReasoning">The prompt already opened a think block, so output starts as reasoning.</param>
    /// <param name="detectAnywhere">Recognise <c>&lt;think&gt;</c> at any position, not only the start of the output.</param>
    public ReasoningSplitter(bool startInReasoning, bool detectAnywhere = false)
    {
        _detectAnywhere = detectAnywhere;
        _mode = startInReasoning ? Mode.Reasoning : Mode.Detect;
        _skipReasoningWs = startInReasoning;
        _everInReasoning = startInReasoning;
    }

    /// <summary>True while the splitter is inside a think block (as of the last <see cref="Feed"/>).</summary>
    public bool InReasoning => _mode == Mode.Reasoning;

    /// <summary>True if any text has been, or is being, treated as reasoning.</summary>
    public bool SawReasoning => _everInReasoning;

    /// <summary>
    /// Number of raw characters fed that belong to the reasoning part including the closing tag (all of
    /// them when the block was never closed); <c>-1</c> if there was no reasoning or it is still open
    /// mid-stream. Lets a caller count reasoning tokens by re-tokenising the raw prefix.
    /// </summary>
    public long ReasoningRawLength => _reasoningRawEnd;

    /// <summary>Feeds the next piece of decoded text and returns whatever is now safe to emit.</summary>
    public ReasoningChunk Feed(string text)
    {
        if (text.Length == 0)
            return default;
        _fed += text.Length;
        _buf.Append(text);
        Process(final: false);
        return Drain();
    }

    /// <summary>Signals end of generation and returns everything still held back.</summary>
    public ReasoningChunk Finish()
    {
        Process(final: true);
        return Drain();
    }

    /// <summary>One-shot split of a complete output.</summary>
    public static (string Reasoning, string Content, ReasoningSplitter State) Split(
        string text, bool startInReasoning, bool detectAnywhere = false)
    {
        var s = new ReasoningSplitter(startInReasoning, detectAnywhere);
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
                Mode.Detect => ProcessDetect(final),
                Mode.Reasoning => ProcessReasoning(final),
                _ => ProcessContent(final),
            };
            if (!again)
                return;
        }
    }

    // Before the first non-whitespace character: is the output opening a think block itself?
    private bool ProcessDetect(bool final)
    {
        string s = _buf.ToString();
        string t = s.TrimStart();
        if (t.Length == 0)
        {
            if (!final)
                return false;                  // all whitespace so far: undecided
            _contentOut.Append(s);
            _buf.Clear();
            _mode = Mode.Content;
            return false;
        }
        if (t.StartsWith(Open, StringComparison.Ordinal))
        {
            _buf.Clear().Append(t, Open.Length, t.Length - Open.Length);
            EnterReasoning();
            return true;
        }
        if (!final && t.Length < Open.Length && Open.StartsWith(t, StringComparison.Ordinal))
            return false;                      // could still become "<think>"
        _mode = Mode.Content;                  // not thinking: whitespace and all belongs to the answer
        return true;
    }

    private bool ProcessReasoning(bool final)
    {
        if (_skipReasoningWs)
        {
            int ws = LeadingWs(_buf);
            if (ws > 0)
                _buf.Remove(0, ws);
            if (_buf.Length == 0)
            {
                if (final)
                    _reasoningRawEnd = _fed;
                return false;
            }
            _skipReasoningWs = false;
        }

        string s = _buf.ToString();
        int idx = s.IndexOf(Close, StringComparison.Ordinal);
        if (idx >= 0)
        {
            _reasoningOut.Append(s, 0, idx);
            TrimEnd(_reasoningOut);
            _buf.Clear().Append(s, idx + Close.Length, s.Length - idx - Close.Length);
            _reasoningRawEnd = _fed - _buf.Length;
            _mode = Mode.Content;
            _skipContentWs = true;
            return true;
        }

        if (final)
        {
            // Unclosed: whatever is left (even a dangling "</thi") is reasoning; trailing ws dropped.
            _reasoningOut.Append(s);
            TrimEnd(_reasoningOut);
            _buf.Clear();
            _reasoningRawEnd = _fed;
            return false;
        }

        // Emit everything that is neither a possible partial closing tag nor trailing whitespace.
        int safe = s.Length - PartialTagSuffix(s, Close);
        int emit = s.AsSpan(0, safe).TrimEnd().Length;
        if (emit > 0)
        {
            _reasoningOut.Append(s, 0, emit);
            _buf.Remove(0, emit);
        }
        return false;
    }

    private bool ProcessContent(bool final)
    {
        if (_skipContentWs)
        {
            int ws = LeadingWs(_buf);
            if (ws > 0)
                _buf.Remove(0, ws);
            if (_buf.Length == 0)
                return false;
            _skipContentWs = false;
        }

        if (!_detectAnywhere)
        {
            _contentOut.Append(_buf);
            _buf.Clear();
            return false;
        }

        string s = _buf.ToString();
        int idx = s.IndexOf(Open, StringComparison.Ordinal);
        if (idx >= 0)
        {
            _contentOut.Append(s, 0, idx);
            _buf.Clear().Append(s, idx + Open.Length, s.Length - idx - Open.Length);
            EnterReasoning();
            return true;
        }
        int hold = final ? 0 : PartialTagSuffix(s, Open);
        _contentOut.Append(s, 0, s.Length - hold);
        _buf.Remove(0, s.Length - hold);
        return false;
    }

    private void EnterReasoning()
    {
        _mode = Mode.Reasoning;
        _skipReasoningWs = true;
        _everInReasoning = true;
        _reasoningRawEnd = -1;
    }

    /// <summary>Length of the longest proper prefix of <paramref name="tag"/> that ends <paramref name="s"/>.</summary>
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

    private static int LeadingWs(StringBuilder sb)
    {
        int i = 0;
        while (i < sb.Length && char.IsWhiteSpace(sb[i]))
            i++;
        return i;
    }

    private static void TrimEnd(StringBuilder sb)
    {
        int n = sb.Length;
        while (n > 0 && char.IsWhiteSpace(sb[n - 1]))
            n--;
        sb.Length = n;
    }
}
