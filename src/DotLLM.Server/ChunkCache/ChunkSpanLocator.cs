using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Server.ChunkCache;

/// <summary>A span located in the rendered prompt.</summary>
/// <param name="Span">The accepted annotation (from <see cref="ChunkAnnotationParser"/>).</param>
/// <param name="RenderedStartChar">First UTF-16 char of the span in the rendered prompt string.</param>
/// <param name="RenderedEndChar">One past its last char. Stripping the U+FDD0 markers (#1339) from
/// <c>rendered[start..end]</c> gives the span text, and the range starts and ends on span characters. The markers inside
/// follow the marking of the whole message, so the range is not always <c>MarkUntrustedSpecials(text)</c>: a special
/// spelling that crosses the span's end leaves its marker inside the range. Equal characters are not equal tokens at
/// the edges; the token check is the caller's.</param>
internal sealed record LocatedSpan(ChunkSpanRef Span, int RenderedStartChar, int RenderedEndChar);

/// <summary>The result of <see cref="ChunkSpanLocator.Locate(IChatTemplate, IReadOnlyList{ChatMessage}, ChatTemplateOptions, string, IReadOnlyList{ChunkSpanRef}, int)"/>.</summary>
/// <param name="Located">The spans found in the rendered prompt, in input order.</param>
/// <param name="Warnings">Why the others were ignored: <c>chunk_template_altered</c> (the template changed or moved the
/// span's text), or <c>chunk_span_invalid</c> (the span is not where its engine position says: a caller bug).</param>
internal sealed record LocatedSpans(IReadOnlyList<LocatedSpan> Located, IReadOnlyList<ChunkWarning> Warnings)
{
    /// <summary>
    /// Per input span (same order as the <c>spans</c> argument): null when located, else its warning code. Lets the caller
    /// pick the strict-mode error in document order (ruling 4). Empty when there were no spans.
    /// </summary>
    public IReadOnlyList<string?> Codes { get; init; } = [];
}

/// <summary>
/// Locates accepted chunk spans in the rendered chat prompt (spec §5.3 item 5; contract ruling 20). The template is
/// rendered again with every span replaced by a sentinel; a span is located only when its sentinel comes out intact,
/// exactly once, and substituting the span texts back into that render reproduces the real prompt around it exactly.
/// Anything else is <c>chunk_template_altered</c> and the span stays plain text.
/// </summary>
/// <remarks>
/// <para>
/// <b>Sentinels.</b> <c>U+E000</c>, a fresh GUID (per request), the span's index, the probe characters <c>"</c>,
/// <c>\</c> and <c>LF</c>, then <c>U+E001</c>. Caller text cannot forge one (the GUID is unguessable; a forged copy would
/// make the sentinel appear twice, which ignores the span). The probe characters break the sentinel under any escaping
/// filter, so a template that JSON-encodes content (Llama-3.1 <c>tool</c> messages: <c>| tojson</c>) alters every span
/// in it, whatever the span's own characters. A trimming template (Llama-3.1 user / assistant / system <c>| trim</c>) is
/// caught by the comparison: the sentinel is not whitespace, so the trim happens in the real prompt only.
/// </para>
/// <para>
/// <b>Comparison.</b> The real prompt holds caller text marked as a whole message (U+FDD0 after the first character of
/// each special-token spelling, #1339), so U+FDD0 is skipped on both sides (ruling 20); stripping the markers from a
/// located range gives the span text. Assistant content keeps the template's reasoning tags live (unmarked): such a span
/// is located by characters while its tokens differ, so the caller's token check decides reuse. A span whose own text contains U+FDD0 is ignored: the marker is
/// dropped at encode (Task 9, F10), so the span's text is not what the prompt's tokens encode.
/// </para>
/// <para>
/// <b>Attribution and cost.</b> The comparison walks the substituted render and the real prompt once from the start and
/// once from the end. A span inside the common prefix or suffix is located: everything rendered before (or after) it is
/// identical, so its offsets are exact. When the two differ, the first and last spans not yet proven are ignored
/// (conservatively: the difference lies between them) and the remaining ones are tried in another render with those two
/// restored; spans whose sentinel broke are dropped from the next render too. At most <see cref="MaxSentinelRenders"/>
/// renders per request, whatever the number of spans; spans still unproven then are ignored. Work is
/// O(<see cref="MaxSentinelRenders"/> × (render + prompt length) + spans × log spans): never one render per span.
/// </para>
/// <para>
/// <b>Never throws on request content.</b> A template that throws on the sentinel render ignores every span still
/// being tried; a span whose engine position is out of range or does not hold its text is <c>chunk_span_invalid</c>.
/// </para>
/// </remarks>
internal static class ChunkSpanLocator
{
    /// <summary>Most sentinel renders per request (the real prompt is rendered by the caller, not here).</summary>
    internal const int MaxSentinelRenders = 4;

    private const char Open = '';
    private const char Close = '';
    private const string Probe = "\"\\\n";
    private const char Marker = ControlTokenGuard.Marker;

    private static readonly LocatedSpans None = new([], []);

    private enum State : byte { Candidate, Located, Altered, Invalid }

    /// <summary>Locates <paramref name="spans"/> in <paramref name="rendered"/>.</summary>
    /// <param name="template">The template that rendered the prompt.</param>
    /// <param name="messages">The messages exactly as rendered (after a profile system prompt was prepended).</param>
    /// <param name="options">The template options exactly as rendered.</param>
    /// <param name="rendered">The rendered prompt: <c>template.Apply(messages, options)</c>.</param>
    /// <param name="spans">The parser's accepted spans; their <see cref="ChunkSpanRef.Engine"/> positions index the
    /// converter's messages, before the profile prompt.</param>
    /// <param name="engineMessageOffset">Added to every engine message index (ruling 31): 1 when
    /// <c>ProfileSystemPrompt.Apply</c> prepended a system message, else 0. Required on purpose.</param>
    /// <returns>The located spans (input order) and the warnings for the rest.</returns>
    public static LocatedSpans Locate(IChatTemplate template, IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options,
        string rendered, IReadOnlyList<ChunkSpanRef> spans, int engineMessageOffset)
        => Locate(template, messages, options, rendered, spans, engineMessageOffset, Guid.NewGuid().ToString("N"));

    /// <summary>As the public overload, with a fixed sentinel nonce (tests).</summary>
    internal static LocatedSpans Locate(IChatTemplate template, IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options,
        string rendered, IReadOnlyList<ChunkSpanRef> spans, int engineMessageOffset, string nonce)
    {
        ArgumentNullException.ThrowIfNull(template);
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(rendered);
        ArgumentNullException.ThrowIfNull(spans);
        ArgumentException.ThrowIfNullOrEmpty(nonce);
        int n = spans.Count;
        if (n == 0)
            return None;

        var state = new State[n];
        var msgOf = new int[n];
        var startR = new int[n];
        var endR = new int[n];

        // 1. Each span must sit in its engine message where the parser said (offset by the profile prompt), and must not
        //    carry a caller U+FDD0. Overlaps inside one message (impossible from the parser) keep the earlier input.
        var byMessage = new Dictionary<int, List<int>>();
        for (int k = 0; k < n; k++)
        {
            var span = spans[k];
            int m = span.Engine.EngineMessage + engineMessageOffset;
            string? content = m >= 0 && m < messages.Count ? messages[m].Content : null;
            if (content is null || span.Engine.StartChar < 0 || span.Engine.StartChar >= span.Engine.EndChar
                || span.Engine.EndChar > content.Length
                || !content.AsSpan(span.Engine.StartChar, span.Engine.EndChar - span.Engine.StartChar).SequenceEqual(span.Text))
            {
                state[k] = State.Invalid;
                continue;
            }
            if (span.Text.Contains(Marker))
            {
                state[k] = State.Altered;
                continue;
            }
            msgOf[k] = m;
            if (!byMessage.TryGetValue(m, out var list))
                byMessage[m] = list = [];
            list.Add(k);
        }
        foreach (var list in byMessage.Values)
        {
            list.Sort((a, b) => spans[a].Engine.StartChar != spans[b].Engine.StartChar
                ? spans[a].Engine.StartChar.CompareTo(spans[b].Engine.StartChar)
                : a.CompareTo(b));
            int end = -1, kept = -1;
            foreach (int k in list)
            {
                if (spans[k].Engine.StartChar < end)
                {
                    state[Math.Max(k, kept)] = State.Invalid; // the later one in input order
                    if (k < kept)
                    {
                        kept = k;
                        end = spans[k].Engine.EndChar;
                    }
                    continue;
                }
                kept = k;
                end = spans[k].Engine.EndChar;
            }
            list.RemoveAll(k => state[k] != State.Candidate);
        }

        // 2. Sentinel renders.
        string prefix = Open + nonce + ":";
        var pos = new int[n];
        for (int pass = 0; pass < MaxSentinelRenders; pass++)
        {
            if (Array.IndexOf(state, State.Candidate) < 0)
                break;

            string s;
            try
            {
                s = template.Apply(WithSentinels(messages, spans, byMessage, state, prefix), options);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                break; // every span still being tried is altered (below)
            }

            // Find every sentinel: once, intact. Others broke (escaped, dropped, duplicated): altered.
            Array.Fill(pos, -1);
            FindSentinels(s, prefix, state, pos);
            bool anyBroken = false;
            for (int k = 0; k < n; k++)
            {
                if (state[k] == State.Candidate && pos[k] < 0)
                {
                    state[k] = State.Altered;
                    anyBroken = true;
                }
            }

            var order = new List<int>();
            for (int k = 0; k < n; k++)
                if (state[k] == State.Candidate)
                    order.Add(k);
            if (order.Count == 0)
                continue;
            order.Sort((a, b) => pos[a].CompareTo(pos[b]));

            // The render with the span texts substituted back, and each span's range in it.
            var aS = new int[n];
            var bS = new int[n];
            string substituted = Substitute(s, spans, order, pos, prefix, aS, bS);

            var proven = new bool[n];
            bool equal = Walk(rendered, substituted, order, aS, bS, startR, endR, proven);
            var unproven = new List<int>();
            foreach (int k in order)
            {
                if (equal || proven[k])
                    state[k] = State.Located;
                else
                    unproven.Add(k);
            }

            // A broken sentinel explains the difference; otherwise blame the spans at its two ends.
            if (!anyBroken && unproven.Count > 0)
            {
                state[unproven[0]] = State.Altered;
                state[unproven[^1]] = State.Altered;
            }
        }

        // 3. Result.
        var located = new List<LocatedSpan>();
        var codes = new string?[n];
        int altered = 0, invalid = 0;
        for (int k = 0; k < n; k++)
        {
            switch (state[k])
            {
                case State.Located:
                    located.Add(new LocatedSpan(spans[k], startR[k], endR[k]));
                    break;
                case State.Invalid:
                    invalid++;
                    codes[k] = ChunkCacheCodes.SpanInvalid;
                    break;
                default:
                    altered++; // Altered, or still a candidate after the last render
                    codes[k] = ChunkCacheCodes.TemplateAltered;
                    break;
            }
        }
        var warnings = new List<ChunkWarning>();
        if (altered > 0)
            warnings.Add(new ChunkWarning(ChunkCacheCodes.TemplateAltered, altered));
        if (invalid > 0)
            warnings.Add(new ChunkWarning(ChunkCacheCodes.SpanInvalid, invalid));
        return new LocatedSpans(located, warnings) { Codes = codes };
    }

    private static string Sentinel(string prefix, int k) =>
        prefix + k.ToString(CultureInfo.InvariantCulture) + Probe + Close;

    /// <summary>The messages with every candidate span replaced by its sentinel (other messages are the same objects).</summary>
    private static ChatMessage[] WithSentinels(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ChunkSpanRef> spans,
        Dictionary<int, List<int>> byMessage, State[] state, string prefix)
    {
        var copy = new ChatMessage[messages.Count];
        for (int m = 0; m < copy.Length; m++)
            copy[m] = messages[m];
        foreach (var (m, list) in byMessage)
        {
            string content = messages[m].Content;
            StringBuilder? sb = null;
            int at = 0;
            foreach (int k in list)
            {
                if (state[k] != State.Candidate)
                    continue;
                sb ??= new StringBuilder(content.Length + 64);
                sb.Append(content, at, spans[k].Engine.StartChar - at).Append(Sentinel(prefix, k));
                at = spans[k].Engine.EndChar;
            }
            if (sb is not null)
                copy[m] = messages[m] with { Content = sb.Append(content, at, content.Length - at).ToString() };
        }
        return copy;
    }

    /// <summary>Records each candidate's sentinel position in <paramref name="s"/>; a sentinel seen twice is -2 (broken).</summary>
    private static void FindSentinels(string s, string prefix, State[] state, int[] pos)
    {
        int i = 0;
        while ((i = s.IndexOf(Open, i)) >= 0)
        {
            if (!s.AsSpan(i).StartsWith(prefix, StringComparison.Ordinal))
            {
                i++;
                continue;
            }
            int d = i + prefix.Length, k = 0, digits = 0;
            while (d < s.Length && s[d] is >= '0' and <= '9' && digits < 9)
            {
                k = k * 10 + (s[d] - '0');
                d++;
                digits++;
            }
            if (digits == 0 || k < 0 || k >= state.Length || state[k] != State.Candidate
                || !s.AsSpan(d).StartsWith(Probe + Close, StringComparison.Ordinal))
            {
                i++;
                continue;
            }
            pos[k] = pos[k] == -1 ? i : -2;
            i = d + Probe.Length + 1;
        }
        for (int k = 0; k < pos.Length; k++)
            if (pos[k] == -2)
                pos[k] = -1;
    }

    /// <summary>The sentinel render with each span text in place of its sentinel; <paramref name="aS"/> / <paramref name="bS"/>
    /// receive each span's range in the result.</summary>
    private static string Substitute(string s, IReadOnlyList<ChunkSpanRef> spans, List<int> order, int[] pos, string prefix,
        int[] aS, int[] bS)
    {
        var sb = new StringBuilder(s.Length);
        int at = 0;
        foreach (int k in order)
        {
            sb.Append(s, at, pos[k] - at);
            Debug.Assert(!spans[k].Text.Contains(Marker), "a span with U+FDD0 must be ignored before Walk (its proof relies on it)");
            aS[k] = sb.Length;
            sb.Append(spans[k].Text);
            bS[k] = sb.Length;
            at = pos[k] + Sentinel(prefix, k).Length;
        }
        return sb.Append(s, at, s.Length - at).ToString();
    }

    /// <summary>
    /// Compares the real prompt <paramref name="r"/> with the substituted render <paramref name="t"/>, U+FDD0 skipped on
    /// both sides, from the start and then from the end. A span whose every char is matched in one walk is proven, with
    /// its range in <paramref name="r"/>. Returns whether the two are equal.
    /// </summary>
    private static bool Walk(string r, string t, List<int> order, int[] aS, int[] bS, int[] startR, int[] endR, bool[] proven)
    {
        // From the start. CORRECTNESS DEPENDS ON span texts carrying no U+FDD0 (spans with one are ignored before any
        // render): every span char of t is then matched, never skipped, so the first and last span chars are always
        // visited at aS and bS - 1, and a span is proven only when all its chars matched. Markers in r are skipped
        // before a match, so a recorded start is never a marker.
        int i = 0, j = 0, ps = 0, pe = 0;
        while (true)
        {
            while (i < r.Length && r[i] == Marker)
                i++;
            while (j < t.Length && t[j] == Marker)
                j++;
            if (i == r.Length || j == t.Length || r[i] != t[j])
                break;
            if (ps < order.Count && aS[order[ps]] == j)
                startR[order[ps++]] = i;
            if (pe < order.Count && bS[order[pe]] - 1 == j)
            {
                endR[order[pe]] = i + 1;
                proven[order[pe++]] = true;
            }
            i++;
            j++;
        }
        if (i == r.Length && j == t.Length)
            return true;

        // From the end, never into what the first walk matched.
        int iLow = i, jLow = j;
        i = r.Length - 1;
        j = t.Length - 1;
        int qs = order.Count - 1, qe = order.Count - 1;
        while (true)
        {
            while (i >= iLow && r[i] == Marker)
                i--;
            while (j >= jLow && t[j] == Marker)
                j--;
            if (i < iLow || j < jLow || r[i] != t[j])
                break;
            if (qe >= 0 && bS[order[qe]] - 1 == j)
                endR[order[qe--]] = i + 1;
            if (qs >= 0 && aS[order[qs]] == j)
            {
                startR[order[qs]] = i;
                proven[order[qs--]] = true;
            }
            i--;
            j--;
        }
        return false;
    }
}
