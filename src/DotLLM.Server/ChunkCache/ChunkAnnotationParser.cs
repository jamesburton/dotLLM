using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Parses and validates the <c>chunk_cache</c> annotations of a chat request (spec §5.1–5.3, §5.8; contract
/// <c>docs/contracts/chunk-cache-v1</c>, rulings 1–35 and the Task 10 review amendments). It reads only the request: no
/// store, no tokenizer, no reuse. Every accepted span is checked against the text the server received (UTF-8 byte range
/// on character boundaries, hash recomputed over the decoded text) and mapped to the engine message it lands in.
/// </summary>
/// <remarks>
/// <para>
/// <b>Surfaces (ruling 19).</b> OpenAI: string <c>content</c> only, through the manifest (<c>part: 0</c>); array
/// content never reaches the parser (model binding refuses it). Anthropic: an inline <c>chunk</c> on a <c>text</c> block
/// (also on a text block inside a <c>tool_result</c>'s content array) and the manifest (a top-level <c>text</c> block,
/// or a <c>tool_result</c> with string content). Engine positions come from
/// <see cref="AnthropicConverter"/> itself (<see cref="AnthropicTextOrigins"/>), so the <c>'\n'</c> joins and the
/// <c>tool_result</c> split are the converter's, not a copy.
/// </para>
/// <para>
/// <b>Defects.</b> One defect per annotation, the first in the check order (ruling 33): structure, version, id,
/// wire eligibility of the named chunker, range/boundary, overlap, then hash (an ineligible span reserves no range).
/// Request options with a wrong type or an unknown value (<c>report</c>, <c>on_miss</c>, an out-of-range
/// <c>query.message</c>) are malformed and fall back to the default (ruling 34). Non-strict: a warning per code with a count, and the span is plain
/// text. Strict: <see cref="ParsedChunkRequest.Error"/> is the first defect in document order (ruling 4 as amended:
/// request-level first, then inline annotations in (message, part) order, then manifest entries in array order).
/// </para>
/// <para>
/// <b>Never throws on body content.</b> Every string the parser itself reads goes through a guarded read: a lone-surrogate
/// escape (which System.Text.Json parses but cannot return as a string) makes the annotation
/// <c>chunk_annotation_malformed</c>, never an exception.
/// </para>
/// <para>
/// <b>Bounded work.</b> The manifest is capped by its array length before any entry is read; each annotated part is
/// encoded to UTF-8 once, with a checkpoint table for byte-to-char offsets; overlapping spans are rejected before they are
/// hashed, so the bytes hashed per request never exceed the request's own text.
/// </para>
/// </remarks>
internal static class ChunkAnnotationParser
{
    /// <summary>SHA-256 computations on this thread (test hook for the cost bound).</summary>
    [ThreadStatic]
    internal static int HashComputations;

    private const int MaxEchoChars = 64;

    /// <summary>Parses an OpenAI <c>/v1/chat/completions</c> request.</summary>
    /// <param name="request">The bound request.</param>
    /// <param name="options">The server's chunk-cache options.</param>
    /// <param name="ns">
    /// The caller's store namespace, always server-assigned (API key, else <see cref="ChunkCacheWire.DefaultNamespace"/>).
    /// Required on purpose: the authentication step must pass it. A body <c>chunk_cache.namespace</c> other than this one is
    /// ignored with <c>chunk_namespace_ignored</c>. Its format is validated where it is assigned, not here.
    /// </param>
    /// <returns>Null when the flag is off, the request carries no annotation, or its messages are missing (the request
    /// validator rejects those); otherwise the parse.</returns>
    public static ParsedChunkRequest? Parse(ChatCompletionRequest request, ChunkCacheOptions options, string ns)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(ns);
        if (!options.Enabled || IsAbsent(request.ChunkCache))
            return null; // an inline `chunk` cannot occur on this surface (string content only)
        if (request.Messages is null || Array.IndexOf(request.Messages, null) >= 0)
            return null;

        var wire = new WireMessage[request.Messages.Length];
        for (int i = 0; i < wire.Length; i++)
        {
            var dto = request.Messages[i];
            wire[i] = new WireMessage(dto.Role ?? "",
                dto.Content is null ? [] : [new WirePart(true, dto.Content, null, i, 0)]);
        }
        return ParseCore(wire, RequestConverter.ToMessages(request.Messages), request.ChunkCache, options, ns);
    }

    /// <summary>Parses an Anthropic <c>/v1/messages</c> request.</summary>
    /// <param name="request">The bound request.</param>
    /// <param name="options">The server's chunk-cache options.</param>
    /// <param name="ns">The caller's store namespace (see the OpenAI overload).</param>
    /// <returns>Null when the flag is off, the request carries no annotation, or its messages are missing; otherwise the
    /// parse.</returns>
    public static ParsedChunkRequest? Parse(AnthropicMessagesRequest request, ChunkCacheOptions options, string ns)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(ns);
        if (!options.Enabled || request.Messages is null || Array.IndexOf(request.Messages, null) >= 0)
            return null;

        // Cheap pre-scan: an unannotated request never pays for the conversion or the wire view.
        if (IsAbsent(request.ChunkCache) && !HasInlineChunk(request.Messages))
            return null;

        var origins = new AnthropicTextOrigins();
        var engine = AnthropicConverter.ToMessages(request, origins);
        var wire = new WireMessage[request.Messages.Length];
        for (int m = 0; m < wire.Length; m++)
            wire[m] = AnthropicMessage(request.Messages[m], m, origins);
        return ParseCore(wire, engine, request.ChunkCache, options, ns);
    }

    private static bool HasInlineChunk(AnthropicMessageDto[] messages)
    {
        foreach (var msg in messages)
        {
            if (msg.Content.ValueKind != JsonValueKind.Array)
                continue;
            foreach (var block in msg.Content.EnumerateArray())
            {
                if (InlineOf(block) is not null)
                    return true;
                if (block.ValueKind == JsonValueKind.Object && block.TryGetProperty("content", out var inner)
                    && inner.ValueKind == JsonValueKind.Array)
                    foreach (var item in inner.EnumerateArray())
                        if (InlineOf(item) is not null)
                            return true;
            }
        }
        return false;
    }

    private static WireMessage AnthropicMessage(AnthropicMessageDto msg, int m, AnthropicTextOrigins origins)
    {
        // The converter has already read every string below (it would have thrown first), so plain reads are safe here.
        var content = msg.Content;
        if (content.ValueKind == JsonValueKind.String)
            return new WireMessage(msg.Role ?? "", [TextPart(content.GetString() ?? "", null, origins, m, 0, -1)]);
        if (content.ValueKind != JsonValueKind.Array)
            return new WireMessage(msg.Role ?? "", []);

        var parts = new List<WirePart>();
        int b = 0;
        foreach (var block in content.EnumerateArray())
        {
            JsonElement? inline = InlineOf(block);
            bool isText = block.ValueKind == JsonValueKind.Object && block.TryGetProperty("type", out var t)
                && t.ValueKind == JsonValueKind.String && t.ValueEquals("text");
            bool isToolResult = block.ValueKind == JsonValueKind.Object && block.TryGetProperty("type", out var t2)
                && t2.ValueKind == JsonValueKind.String && t2.ValueEquals("tool_result");

            if (isText && block.TryGetProperty("text", out var txt) && txt.ValueKind == JsonValueKind.String)
            {
                parts.Add(TextPart(txt.GetString()!, inline, origins, m, b, -1));
            }
            else if (isToolResult && block.TryGetProperty("content", out var rc) && rc.ValueKind == JsonValueKind.String)
            {
                parts.Add(TextPart(rc.GetString()!, inline, origins, m, b, -1));
            }
            else if (isToolResult && block.TryGetProperty("content", out var arr) && arr.ValueKind == JsonValueKind.Array)
            {
                // Inner blocks are reachable inline only: the manifest's `at` has no inner index (ruling 30).
                var inner = new List<WirePart>();
                int k = 0;
                foreach (var item in arr.EnumerateArray())
                {
                    JsonElement? innerInline = InlineOf(item);
                    inner.Add(item.ValueKind == JsonValueKind.Object && item.TryGetProperty("text", out var it)
                              && it.ValueKind == JsonValueKind.String
                        ? TextPart(it.GetString()!, innerInline, origins, m, b, k)
                        : new WirePart(false, null, innerInline, -1, 0));
                    k++;
                }
                parts.Add(new WirePart(false, null, inline, -1, 0, inner));
            }
            else
            {
                parts.Add(new WirePart(false, null, inline, -1, 0));
            }
            b++;
        }
        return new WireMessage(msg.Role ?? "", parts);
    }

    private static WirePart TextPart(string text, JsonElement? inline, AnthropicTextOrigins origins, int m, int b, int k) =>
        origins.TryGet(m, b, k, out int engineMessage, out int offset)
            ? new WirePart(true, text, inline, engineMessage, offset)
            : new WirePart(true, text, inline, -1, 0);

    private static JsonElement? InlineOf(JsonElement block) =>
        block.ValueKind == JsonValueKind.Object && block.TryGetProperty("chunk", out var c) && c.ValueKind != JsonValueKind.Null
            ? c
            : null;

    private static bool IsAbsent(JsonElement? e) => e is null || e.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined;

    // ------------------------------------------------------------------------------------------------ core

    /// <summary>One content part of a wire message, as the parser sees it.</summary>
    /// <param name="IsText">Whether a span may address this part.</param>
    /// <param name="Text">The decoded text (text parts only).</param>
    /// <param name="Inline">The part's inline <c>chunk</c> object, if any.</param>
    /// <param name="EngineMessage">The engine message holding the text; -1 when the converter does not carry it.</param>
    /// <param name="EngineOffset">Char offset of the text's first char in that engine message.</param>
    /// <param name="Inner">Anthropic <c>tool_result</c> array content: its blocks (inline placement only).</param>
    internal sealed record WirePart(bool IsText, string? Text, JsonElement? Inline, int EngineMessage, int EngineOffset,
        IReadOnlyList<WirePart>? Inner = null);

    /// <summary>A wire message: its role and content parts (string content is one part).</summary>
    internal sealed record WireMessage(string Role, IReadOnlyList<WirePart> Parts);

    private readonly record struct Annotation(JsonElement Chunk, bool Inline, int Message, int Part, int Inner, WirePart? Target);

    /// <summary>The surface-independent parse over a wire view, its engine messages and the raw <c>chunk_cache</c>.</summary>
    internal static ParsedChunkRequest? ParseCore(IReadOnlyList<WireMessage> messages, IReadOnlyList<ChatMessage> engine,
        JsonElement? chunkCache, ChunkCacheOptions options, string ns)
    {
        if (!options.Enabled)
            return null;

        var inline = new List<Annotation>();
        for (int m = 0; m < messages.Count; m++)
        {
            var parts = messages[m].Parts;
            for (int p = 0; p < parts.Count; p++)
            {
                if (parts[p].Inline is { } c)
                    inline.Add(new Annotation(c, true, m, p, -1, parts[p]));
                if (parts[p].Inner is { } inner)
                    for (int k = 0; k < inner.Count; k++)
                        if (inner[k].Inline is { } ic)
                            inline.Add(new Annotation(ic, true, m, p, k, inner[k]));
            }
        }

        bool hasCc = !IsAbsent(chunkCache);
        if (!hasCc && inline.Count == 0)
            return null;

        var acc = new Accumulator();
        JsonElement cc = hasCc ? chunkCache!.Value : default;
        bool ccObject = hasCc && cc.ValueKind == JsonValueKind.Object;

        // Request-level options. `strict` is read first: it decides whether defects are errors.
        bool strict = false;
        var policy = ChunkPolicy.Exact;
        string report = "summary";
        int minChunkTokens = ChunkCacheWire.DefaultMinChunkTokens;
        string onMiss = "prefill";
        string? query = null;
        bool dropAll = false;
        JsonElement manifest = default;
        int manifestLength = 0;

        if (hasCc && !ccObject)
        {
            acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache must be an object");
            dropAll = true;
        }
        else if (ccObject)
        {
            if (Present(cc, "strict", out var st))
            {
                if (st.ValueKind is JsonValueKind.True or JsonValueKind.False)
                    strict = st.ValueKind == JsonValueKind.True;
                else
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.strict must be a boolean");
            }

            if (Present(cc, "v", out var v))
            {
                if (v.ValueKind != JsonValueKind.Number)
                {
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.v must be a number");
                    dropAll = true;
                }
                else if (!(v.TryGetInt32(out int vi) && vi == ChunkCacheWire.Version))
                {
                    acc.Request(ChunkCacheCodes.VersionUnsupported, $"chunk_cache.v {Echo(v)} is not supported; this server speaks v1");
                    dropAll = true;
                }
            }

            if (Present(cc, "policy", out var pol))
            {
                if (!TryString(pol, out string? polName))
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.policy must be a string");
                else if (PolicyOf(polName) is { } pp)
                    policy = pp;
                else
                    acc.Request(ChunkCacheCodes.PolicyUnavailable, $"chunk_cache.policy {Echo(pol)} is not a known policy; exact applies");
            }

            // The namespace is server-assigned (§5.2); a body value is only ever compared with the caller's own.
            if (Present(cc, "namespace", out var bodyNs))
            {
                if (!TryString(bodyNs, out string? bodyNsValue))
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.namespace must be a string");
                else if (!string.Equals(bodyNsValue, ns, StringComparison.Ordinal))
                    acc.Request(ChunkCacheCodes.NamespaceIgnored, "chunk_cache.namespace is assigned by the server; the body value was ignored");
            }

            // Ruling 34: a wrong type or an unknown value is malformed; the default applies.
            if (Present(cc, "report", out var rep))
            {
                if (TryString(rep, out string? r) && r is "none" or "summary" or "per_chunk")
                    report = r;
                else
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.report must be \"none\", \"summary\" or \"per_chunk\"");
            }

            if (Present(cc, "min_chunk_tokens", out var mct))
            {
                if (mct.ValueKind == JsonValueKind.Number && mct.TryGetInt32(out int mctv) && mctv >= 0)
                    minChunkTokens = mctv;
                else
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.min_chunk_tokens must be a non-negative integer");
            }

            if (Present(cc, "on_miss", out var om))
            {
                if (TryString(om, out string? o) && o is "prefill" or "encode")
                    onMiss = o;
                else
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.on_miss must be \"prefill\" or \"encode\"");
            }

            if (Present(cc, "query", out var q) && !TryQuery(q, messages, out query))
                acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.query must be {text: string} or {message: index of a request message}");

            if (Present(cc, "chunks", out var chunks))
            {
                if (chunks.ValueKind == JsonValueKind.Array)
                {
                    manifest = chunks;
                    manifestLength = chunks.GetArrayLength(); // O(1): nothing is materialised before the cap
                }
                else if (!dropAll)
                {
                    acc.Request(ChunkCacheCodes.AnnotationMalformed, "chunk_cache.chunks must be an array");
                    dropAll = true;
                }
            }
        }

        // Both placements (ruling 2: an options-only chunk_cache is not a conflict): served as unannotated.
        long annotated = (long)inline.Count + manifestLength;
        if (!dropAll && inline.Count > 0 && manifestLength > 0)
        {
            acc.Request(ChunkCacheCodes.ManifestConflict, "inline chunk annotations and chunk_cache.chunks in one request");
            dropAll = true;
        }

        var spans = new List<ChunkSpanRef>();
        if (!dropAll)
        {
            int keep = (int)Math.Min(annotated, options.MaxChunks);
            if (annotated > options.MaxChunks)
                acc.Request(ChunkCacheCodes.TooMany, $"{annotated} chunk annotations; at most {options.MaxChunks} per request",
                    (int)Math.Min(int.MaxValue, annotated - options.MaxChunks));

            var state = new SpanState();
            int index = 0;
            foreach (var a in inline)
            {
                if (index >= keep)
                    break;
                if (Validate(index++, a, messages, engine, state, acc) is { } span)
                    spans.Add(span);
            }
            if (manifestLength > 0)
            {
                foreach (var entry in manifest.EnumerateArray())
                {
                    if (index >= keep)
                        break; // the rest are never read
                    if (Validate(index++, new Annotation(entry, false, -1, -1, -1, null), messages, engine, state, acc) is { } span)
                        spans.Add(span);
                }
            }
        }

        return new ParsedChunkRequest(policy, spans, query ?? LastUserText(engine), strict, report, ns, acc.Warnings,
            strict ? acc.FirstError : null)
        {
            Annotated = (int)Math.Min(int.MaxValue, annotated),
            MinChunkTokens = minChunkTokens,
            OnMiss = onMiss,
            Defects = acc.Defects,
        };
    }

    /// <summary>Per-request state: per-part UTF-8 encodings (built once) and the ranges taken for the overlap rule.</summary>
    private sealed class SpanState
    {
        public readonly Dictionary<WirePart, PartText> Parts = new(ReferenceEqualityComparer.Instance);
        public readonly Dictionary<(int, int, int), List<(int Start, int End)>> Taken = [];

        public PartText Of(WirePart part)
        {
            if (!Parts.TryGetValue(part, out var t))
                Parts[part] = t = new PartText(part.Text!);
            return t;
        }
    }

    /// <summary>
    /// A part's UTF-8 bytes (encoded once per request) and a byte-to-char checkpoint table every
    /// <see cref="Step"/> chars, so a byte offset maps to a char offset by scanning at most one step.
    /// </summary>
    private sealed class PartText
    {
        private const int Step = 1024;
        private readonly int[] _cpByte;
        private readonly int[] _cpChar;

        public PartText(string text)
        {
            Bytes = Encoding.UTF8.GetBytes(text); // a lone surrogate encodes as U+FFFD (ruling 17, in-process only)
            var bytesAt = new List<int> { 0 };
            var charsAt = new List<int> { 0 };
            int b = 0, c = 0;
            while (c < text.Length)
            {
                int next = Math.Min(text.Length, c + Step);
                if (next < text.Length && char.IsLowSurrogate(text[next]) && char.IsHighSurrogate(text[next - 1]))
                    next--; // never split a surrogate pair
                b += Encoding.UTF8.GetByteCount(text.AsSpan(c, next - c));
                c = next;
                bytesAt.Add(b);
                charsAt.Add(c);
            }
            _cpByte = [.. bytesAt];
            _cpChar = [.. charsAt];
        }

        public byte[] Bytes { get; }

        /// <summary>The char offset of a byte offset that lies on a UTF-8 character boundary.</summary>
        public int CharOffset(int byteOffset)
        {
            int i = Array.BinarySearch(_cpByte, byteOffset);
            if (i < 0)
                i = ~i - 1;
            return _cpChar[i] + Encoding.UTF8.GetCharCount(Bytes, _cpByte[i], byteOffset - _cpByte[i]);
        }
    }

    private static ChunkSpanRef? Validate(int index, Annotation a, IReadOnlyList<WireMessage> messages,
        IReadOnlyList<ChatMessage> engine, SpanState state, Accumulator acc)
    {
        var ch = a.Chunk;

        // 1. Structure: types and required fields (ruling 23). No string is read unguarded.
        if (ch.ValueKind != JsonValueKind.Object)
            return acc.Drop(index, ChunkCacheCodes.AnnotationMalformed, "malformed", "a chunk annotation must be an object");
        bool hasV = Present(ch, "v", out var v);
        if ((a.Inline && !hasV) || (hasV && v.ValueKind != JsonValueKind.Number))
            return acc.Drop(index, ChunkCacheCodes.AnnotationMalformed, "malformed", "chunk.v must be a number (required inline)");
        if (!Present(ch, "id", out var idEl) || !TryString(idEl, out string? id))
            return acc.Drop(index, ChunkCacheCodes.AnnotationMalformed, "malformed", "chunk.id must be a string");
        if (!Present(ch, "hash", out var hashEl) || !TryString(hashEl, out string? claimedHash))
            return acc.Drop(index, ChunkCacheCodes.AnnotationMalformed, "malformed", "chunk.hash must be a string");
        string? chunker = null, positioning = null, stability = null;
        if ((Present(ch, "chunker", out var ck) && !TryString(ck, out chunker))
            || (Present(ch, "positioning", out var pos) && !TryString(pos, out positioning))
            || (Present(ch, "stability", out var stab) && !TryString(stab, out stability))
            || (Present(ch, "no_store", out var nsEl) && nsEl.ValueKind is not (JsonValueKind.True or JsonValueKind.False)))
            return acc.Drop(index, ChunkCacheCodes.AnnotationMalformed, "malformed", "chunk.chunker / positioning / stability must be strings, no_store a boolean");

        int message = a.Message, part = a.Part, inner = a.Inner;
        WirePart? target = a.Target;
        int? start = null, end = null;
        if (!a.Inline)
        {
            if (!Present(ch, "at", out var at) || at.ValueKind != JsonValueKind.Object
                || !TryInt(at, "message", required: true, out int? mi) || !TryInt(at, "part", required: true, out int? pi)
                || !TryInt(at, "start", required: false, out start) || !TryInt(at, "end", required: false, out end))
                return acc.Drop(index, ChunkCacheCodes.AnnotationMalformed, "malformed", "at must be {message, part[, start, end]} with integer values");
            message = mi!.Value;
            part = pi!.Value;
        }

        // 2. Version.
        if (hasV && !(v.TryGetInt32(out int vi) && vi == ChunkCacheWire.Version))
            return acc.Drop(index, ChunkCacheCodes.VersionUnsupported, "version", "chunk.v must be 1");
        if (positioning is not null && positioning != "isolated")
            return acc.Drop(index, ChunkCacheCodes.VersionUnsupported, "positioning", "v1 accepts only positioning \"isolated\"");

        // 3. Id, and the chunker it names.
        if (id is not { Length: > 0 } || CountRunes(id) > ChunkCacheWire.MaxIdLength)
            return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "id_invalid", $"id must be a string of 1..{ChunkCacheWire.MaxIdLength} characters");
        if (chunker is not null && !ChunkCacheWire.WireChunkers.Contains(chunker))
            return acc.Drop(index, ChunkCacheCodes.NotWireEligible, "not_wire_eligible", $"chunker {Echo(ck)} is not wire-eligible (text-first)");

        // 4. Range and boundary. The part is encoded once per request, whatever the number of annotations on it.
        if (!a.Inline)
        {
            if (message < 0 || message >= messages.Count || part < 0 || part >= messages[message].Parts.Count)
                return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "part_out_of_range", $"at {{message {message}, part {part}}} is out of range");
            target = messages[message].Parts[part];
        }
        if (target is not { IsText: true, Text: { } text })
            return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "non_text", "a chunk span must lie in a text part");

        var partText = state.Of(target);
        byte[] bytes = partText.Bytes;
        int s = start ?? 0, e = end ?? bytes.Length;
        if (!(0 <= s && s < e && e <= bytes.Length))
            return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "range_invalid", $"byte range [{s}, {e}) is empty or outside the part's {bytes.Length} bytes");
        if (!OnBoundary(bytes, s) || !OnBoundary(bytes, e))
            return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "multibyte_boundary", $"byte range [{s}, {e}) is not on UTF-8 character boundaries");

        // 5. Overlap: the later span (document order) is dropped (ruling 1). A span that then fails its hash still holds
        //    its range, as in the contract's reference parse.
        var key = (message, part, inner);
        if (!state.Taken.TryGetValue(key, out var ranges))
            state.Taken[key] = ranges = [];
        foreach (var (rs, re) in ranges)
            if (s < re && rs < e)
                return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "overlap", $"byte range [{s}, {e}) overlaps an earlier span");
        ranges.Add((s, e));

        // 6. Hash, last: only spans that passed every cheap check are hashed, and accepted ranges never overlap.
        HashComputations++;
        string hash = ChunkCacheWire.HashPrefix + Convert.ToHexStringLower(SHA256.HashData(bytes.AsSpan(s, e - s)));
        if (!string.Equals(claimedHash, hash, StringComparison.Ordinal))
            return acc.Drop(index, ChunkCacheCodes.HashMismatch, "hash_mismatch", "hash does not match the received text");

        int startChar = partText.CharOffset(s);
        int endChar = partText.CharOffset(e);
        string spanText = text[startChar..endChar];

        // Defence in depth: the engine message must hold the span text where the mapping says.
        if (target.EngineMessage < 0 || target.EngineMessage >= engine.Count)
            return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "not_carried", "the part is not carried into the prompt as text");
        var engineSpan = new EngineSpan(target.EngineMessage, target.EngineOffset + startChar, target.EngineOffset + endChar);
        string engineContent = engine[engineSpan.EngineMessage].Content ?? "";
        if (engineSpan.StartChar < 0 || engineSpan.EndChar > engineContent.Length
            || !engineContent.AsSpan(engineSpan.StartChar, engineSpan.EndChar - engineSpan.StartChar).SequenceEqual(spanText))
            return acc.Drop(index, ChunkCacheCodes.SpanInvalid, "not_carried", "the span is not where the converter put the part's text");

        bool noStore = Present(ch, "no_store", out var nsv) && nsv.ValueKind == JsonValueKind.True;
        return new ChunkSpanRef(message, part, s, e, id, hash, chunker, stability == "volatile", noStore, spanText, engineSpan, inner);
    }

    /// <summary>A property that is present and not JSON <c>null</c> (null means "use the default").</summary>
    private static bool Present(JsonElement obj, string name, out JsonElement value) =>
        obj.TryGetProperty(name, out value) && value.ValueKind != JsonValueKind.Null;

    /// <summary>
    /// Reads a JSON string without throwing: a lone-surrogate escape (<c>"\ud800"</c>) parses, but System.Text.Json
    /// cannot return it as a string; that is a malformed annotation, not a server error.
    /// </summary>
    private static bool TryString(JsonElement e, out string? value)
    {
        value = null;
        if (e.ValueKind != JsonValueKind.String)
            return false;
        try
        {
            value = e.GetString();
            return value is not null;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static bool TryQuery(JsonElement q, IReadOnlyList<WireMessage> messages, out string? query)
    {
        query = null;
        if (q.ValueKind != JsonValueKind.Object)
            return false;
        if (Present(q, "text", out var t))
            return TryString(t, out query);
        if (Present(q, "message", out var m))
        {
            // Ruling 34: an index outside the request's messages is malformed (the caller sees the default query applied).
            if (m.ValueKind != JsonValueKind.Number || !m.TryGetInt32(out int mi) || mi < 0 || mi >= messages.Count)
                return false;
            query = string.Join('\n', messages[mi].Parts.Where(p => p.IsText).Select(p => p.Text));
            return true;
        }
        return true;
    }

    private static bool TryInt(JsonElement obj, string name, bool required, out int? value)
    {
        value = null;
        if (!Present(obj, name, out var p))
            return !required;
        if (p.ValueKind != JsonValueKind.Number || !p.TryGetInt32(out int v))
            return false;
        value = v;
        return true;
    }

    /// <summary>Client JSON echoed into an error message, clipped so a body cannot inflate the response or a log.</summary>
    private static string Echo(JsonElement e)
    {
        string raw = e.GetRawText();
        return raw.Length <= MaxEchoChars ? raw : raw[..MaxEchoChars] + "…";
    }

    private static bool OnBoundary(byte[] bytes, int i) => i == 0 || i == bytes.Length || (bytes[i] & 0xC0) != 0x80;

    private static int CountRunes(string s)
    {
        int n = 0;
        foreach (var _ in s.EnumerateRunes())
            n++;
        return n;
    }

    private static ChunkPolicy? PolicyOf(string? s) => s switch
    {
        "exact" => ChunkPolicy.Exact,
        "shadow" => ChunkPolicy.Shadow,
        "balanced" => ChunkPolicy.Balanced,
        "fast" => ChunkPolicy.Fast,
        "auto" => ChunkPolicy.Auto,
        _ => null,
    };

    /// <summary>The last engine message with role <c>user</c> (Anthropic tool results are <c>tool</c> messages there).</summary>
    private static string? LastUserText(IReadOnlyList<ChatMessage> engine)
    {
        for (int i = engine.Count - 1; i >= 0; i--)
            if (engine[i].Role == "user")
                return engine[i].Content;
        return null;
    }

    /// <summary>Warnings (code → count, first-occurrence order), defects, and the first error in document order.</summary>
    private sealed class Accumulator
    {
        private readonly List<ChunkWarning> _warnings = [];
        private readonly List<ChunkDefect> _defects = [];

        public IReadOnlyList<ChunkWarning> Warnings => _warnings;

        public IReadOnlyList<ChunkDefect> Defects => _defects;

        public ChunkCacheError? FirstError { get; private set; }

        public void Request(string code, string message, int count = 1) => Add(code, message, count);

        public ChunkSpanRef? Drop(int index, string code, string kind, string message)
        {
            _defects.Add(new ChunkDefect(index, code, kind));
            Add(code, $"chunk {index}: {message}", 1);
            return null;
        }

        private void Add(string code, string message, int count)
        {
            int i = _warnings.FindIndex(w => w.Code == code);
            if (i < 0)
                _warnings.Add(new ChunkWarning(code, count));
            else
                _warnings[i] = _warnings[i] with { Count = _warnings[i].Count + count };
            if (FirstError is null && !ChunkCacheCodes.IsWarningOnly(code))
                FirstError = ChunkCacheCodes.ErrorFor(code, message);
        }
    }
}
