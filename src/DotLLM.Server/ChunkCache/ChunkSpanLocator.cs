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
internal sealed record LocatedSpans(IReadOnlyList<LocatedSpan> Located, IReadOnlyList<ChunkWarning> Warnings);

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


    private static readonly LocatedSpans None = new([], []);

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

        // Stub (test-first commit): nothing is located yet; every span is reported as template-altered.
        return new LocatedSpans([], [new ChunkWarning(ChunkCacheCodes.TemplateAltered, n)]);
    }
}
