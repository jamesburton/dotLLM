// SPDX-License-Identifier: MIT
using System.Buffers;
using System.Net;
using System.Text;
using System.Text.Json;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// The HTTP half of the annotating pipeline (spec 8.3): on the provider's <see cref="HttpClient"/>, it writes the
/// top-level <c>chunk_cache</c> manifest (placement B) into chat request bodies for the spans that
/// <see cref="ChunkAnnotatingChatClient"/> handed over through <see cref="ChunkManifestScope"/>, and reads the
/// response's <c>chunk_cache</c> report back into that scope.
/// </summary>
/// <remarks>
/// <para><b>When it writes.</b> Only for a POST to <c>.../chat/completions</c> or <c>.../messages</c> with a current scope
/// holding registered parts, a JSON body without <c>chunk_cache</c> or inline <c>chunk</c> annotations, capabilities from
/// <see cref="IChunkCapabilitySource"/> (ruling 16: no capabilities, no annotations), and at least one surviving span.
/// Otherwise the request goes out with the caller's content object untouched, so its bytes and headers are exactly the
/// caller's. Install it only on providers that are really dotLLM (gate on the provider kind, spec 8.3).</para>
/// <para><b>How it locates spans.</b> It walks the serialised <c>messages</c> (a string <c>content</c> is part 0; an
/// array's text parts keep their array index) and decodes each part's JSON string, so escapes such as <c>é</c> are
/// resolved before anything is compared or hashed: the hash is over the decoded text, never the wire bytes. A registered
/// text is found as the whole part, as its prefix (a suffix appended after registration), or inside it (a bridge that
/// joined texts). Every span is then re-hashed against the part text; a mismatch drops the span (a mutated part yields no
/// annotation, never a wrong one). Offsets are converted from UTF-16 to UTF-8 bytes; a span that starts or ends inside a
/// surrogate pair is dropped. A span registered "to the end of the part" ends where the registered text ends.</para>
/// <para><b>Manifest.</b> Document order (message, part, start); a span overlapping an earlier one of the same part is
/// dropped (the server would drop it, ruling 1); at most <c>max_chunks_per_request</c>; ids cut to 256 UTF-16 chars
/// (telemetry only). Request options at their defaults are omitted. The member is spliced in before the body's closing
/// brace, so every other byte of the body is unchanged.</para>
/// <para><b>Retries.</b> A resend of the same request message (a retrying handler above this one) starts again from the
/// caller's original content, so it carries the same manifest, or none if the capability was lost meanwhile.</para>
/// <para><b>Errors.</b> In strict mode a <c>chunk_*</c> error reaches the caller unchanged. In non-strict mode, a 400 or
/// 503 whose error code starts with <c>chunk_</c> (which a conforming server does not send) invalidates the capability
/// answer and the request is resent once with the caller's original body (<see cref="FallBackOnChunkError"/>).</para>
/// <para><b>Report.</b> Blocking: read from the buffered JSON body (which stays readable). Streaming
/// (<c>text/event-stream</c>): the body is wrapped in a <see cref="SseChunkCacheScanner"/>, so nothing is buffered and
/// the report is set on the scope when its event has been read.</para>
/// </remarks>
public sealed class ChunkCacheBodyHandler : DelegatingHandler
{
    /// <summary>The maximum id length on the wire (spec 5.2).</summary>
    public const int MaxIdLength = 256;

    private static readonly HttpRequestOptionsKey<HttpContent> s_original = new("DotLLM.ChunkCache.Client.OriginalContent");

    private readonly IChunkCapabilitySource _capabilities;

    /// <summary>With a fixed capability answer (<see langword="null"/> or a body without version 1 = never annotate).</summary>
    public ChunkCacheBodyHandler(ChunkCacheCapabilities? capabilities)
        : this(new FixedChunkCapabilities(capabilities is not null && capabilities.SupportsVersion(1) ? capabilities : null))
    {
    }

    /// <summary>With a capability source (usually a <see cref="ChunkCapabilityCache"/> shared with the decorator).</summary>
    public ChunkCacheBodyHandler(IChunkCapabilitySource capabilities)
    {
        ArgumentNullException.ThrowIfNull(capabilities);
        _capabilities = capabilities;
    }

    /// <summary>
    /// Resend once without annotations when a non-strict request is refused with a <c>chunk_*</c> 400 or 503. Default
    /// <see langword="true"/>.
    /// </summary>
    public bool FallBackOnChunkError { get; set; } = true;

    /// <inheritdoc />
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        // A resend of a message this handler rewrote: start again from the caller's own content.
        if (request.Options.TryGetValue(s_original, out HttpContent? original))
            request.Content = original;
        else
            original = request.Content;

        ChunkManifestScope? scope = ChunkManifestScope.Current;
        if (scope is null || scope.Parts.Count == 0 || original is null || request.Method != HttpMethod.Post || !IsChatPath(request.RequestUri))
            return await base.SendAsync(request, cancellationToken).ConfigureAwait(false);

        int registered = scope.Parts.Sum(p => p.Spans.Count);
        if (original.Headers.ContentEncoding.Count > 0)
            return await SendPlainAsync(request, scope, Skip(ChunkSkipReasons.NotJson, registered), false, cancellationToken).ConfigureAwait(false);

        byte[] body = await original.ReadAsByteArrayAsync(cancellationToken).ConfigureAwait(false);
        WireBody? wire = WireBody.Read(body);
        if (wire is null)
            return await SendPlainAsync(request, scope, Skip(ChunkSkipReasons.NotJson, registered), false, cancellationToken).ConfigureAwait(false);
        if (wire.AlreadyAnnotated)
            return await SendPlainAsync(request, scope, Skip(ChunkSkipReasons.AlreadyAnnotated, registered), true, cancellationToken).ConfigureAwait(false);

        ChunkCacheCapabilities? caps = await GetCapabilitiesAsync(wire.Model, cancellationToken).ConfigureAwait(false);
        if (caps is null)
            return await SendPlainAsync(request, scope, Skip(ChunkSkipReasons.NotCapable, registered), false, cancellationToken).ConfigureAwait(false);

        ManifestPlan plan = ManifestPlan.Build(wire, scope.Parts, caps.MaxChunksPerRequest);
        if (plan.Accepted.Count == 0)
            return await SendPlainAsync(request, scope, plan.Outcome(registered, annotated: false, fellBack: false), false, cancellationToken).ConfigureAwait(false);

        byte[] rewritten = Splice(body, plan.WriteManifest(scope.Options));
        var content = new ByteArrayContent(rewritten);
        foreach (KeyValuePair<string, IEnumerable<string>> h in original.Headers)
        {
            if (!string.Equals(h.Key, "Content-Length", StringComparison.OrdinalIgnoreCase))
                content.Headers.TryAddWithoutValidation(h.Key, h.Value);
        }

        request.Options.Set(s_original, original);
        request.Content = content;
        scope.SetOutcome(plan.Outcome(registered, annotated: true, fellBack: false));

        HttpResponseMessage response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
        if (FallBackOnChunkError && !scope.Options.Strict && await IsChunkErrorAsync(response, cancellationToken).ConfigureAwait(false))
        {
            // A conforming server never refuses a non-strict manifest; this one did. Forget its capabilities and send the
            // caller's request as it was.
            _capabilities.Invalidate();
            response.Dispose();
            request.Content = original;
            scope.SetOutcome(plan.Outcome(registered, annotated: false, fellBack: true));
            return await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
        }

        await AttachReportReaderAsync(response, scope, cancellationToken).ConfigureAwait(false);
        return response;
    }

    private async Task<HttpResponseMessage> SendPlainAsync(HttpRequestMessage request, ChunkManifestScope scope, ChunkManifestOutcome outcome,
        bool readReport, CancellationToken ct)
    {
        scope.SetOutcome(outcome);
        HttpResponseMessage response = await base.SendAsync(request, ct).ConfigureAwait(false);
        if (readReport)
            await AttachReportReaderAsync(response, scope, ct).ConfigureAwait(false);
        return response;
    }

    private async ValueTask<ChunkCacheCapabilities?> GetCapabilitiesAsync(string? model, CancellationToken ct)
    {
        try
        {
            ChunkCacheCapabilities? caps = await _capabilities.GetAsync(model, ct).ConfigureAwait(false);
            return caps is not null && caps.SupportsVersion(1) ? caps : null;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception)
        {
            return null; // ruling 16: a failing source means "not capable"
        }
    }

    private static ChunkManifestOutcome Skip(string reason, int registered) =>
        new(false, reason, registered, 0, 0, 0, 0, 0, 0, false);

    private static bool IsChatPath(Uri? uri)
    {
        if (uri is null) return false;
        string path = (uri.IsAbsoluteUri ? uri.AbsolutePath : uri.OriginalString.Split('?')[0]).TrimEnd('/');
        return path.EndsWith("/chat/completions", StringComparison.OrdinalIgnoreCase)
               || path.EndsWith("/messages", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Inserts <c>,"chunk_cache":manifest</c> before the closing brace of the top-level object.</summary>
    private static byte[] Splice(byte[] body, ReadOnlySpan<byte> manifest)
    {
        int close = body.Length - 1;
        while (close >= 0 && body[close] is (byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n')
            close--;
        ReadOnlySpan<byte> member = ",\"chunk_cache\":"u8;
        var result = new byte[body.Length + member.Length + manifest.Length];
        body.AsSpan(0, close).CopyTo(result);
        member.CopyTo(result.AsSpan(close));
        manifest.CopyTo(result.AsSpan(close + member.Length));
        body.AsSpan(close).CopyTo(result.AsSpan(close + member.Length + manifest.Length));
        return result;
    }

    private static async Task<bool> IsChunkErrorAsync(HttpResponseMessage response, CancellationToken ct)
    {
        if (response.StatusCode is not (HttpStatusCode.BadRequest or HttpStatusCode.ServiceUnavailable))
            return false;
        byte[] bytes = await response.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false); // buffered: still readable
        try
        {
            using var doc = JsonDocument.Parse(bytes);
            return doc.RootElement.ValueKind == JsonValueKind.Object
                   && doc.RootElement.TryGetProperty("error", out JsonElement error) && error.ValueKind == JsonValueKind.Object
                   && error.TryGetProperty("code", out JsonElement code) && code.ValueKind == JsonValueKind.String
                   && code.GetString()!.StartsWith("chunk_", StringComparison.Ordinal);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static async Task AttachReportReaderAsync(HttpResponseMessage response, ChunkManifestScope scope, CancellationToken ct)
    {
        if (!response.IsSuccessStatusCode) return;
        HttpContent content = response.Content;
        if (string.Equals(content.Headers.ContentType?.MediaType, "text/event-stream", StringComparison.OrdinalIgnoreCase))
        {
            Stream stream = await content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            var wrapped = new StreamContent(new SseChunkCacheScanner(stream, scope.SetReport));
            foreach (KeyValuePair<string, IEnumerable<string>> h in content.Headers)
                wrapped.Headers.TryAddWithoutValidation(h.Key, h.Value);
            response.Content = wrapped;
            return;
        }

        // Only a JSON body is buffered and read; anything else (an NDJSON stream, say) passes through untouched.
        string? mediaType = content.Headers.ContentType?.MediaType;
        if (mediaType is null || !(string.Equals(mediaType, "application/json", StringComparison.OrdinalIgnoreCase)
                                   || mediaType.EndsWith("+json", StringComparison.OrdinalIgnoreCase)))
            return;

        byte[] bytes = await content.ReadAsByteArrayAsync(ct).ConfigureAwait(false); // buffered: still readable
        if (ChunkCacheReport.FromJson(bytes.AsSpan(StartsWithBom(bytes) ? 3 : 0)) is { } report)
            scope.SetReport(report);
    }

    private static bool StartsWithBom(byte[] b) => b.Length >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF;

    /// <summary>One serialised text part: its indices and its decoded text.</summary>
    private sealed record WirePart(int Message, int Part, string Text);

    /// <summary>What the handler needs from a request body.</summary>
    private sealed class WireBody
    {
        private WireBody(string? model, bool alreadyAnnotated, List<WirePart> parts)
        {
            Model = model;
            AlreadyAnnotated = alreadyAnnotated;
            Parts = parts;
        }

        public string? Model { get; }

        public bool AlreadyAnnotated { get; }

        public List<WirePart> Parts { get; }

        /// <summary>Parses the body; <see langword="null"/> when it is not a JSON object with a <c>messages</c> array.</summary>
        public static WireBody? Read(byte[] body)
        {
            try
            {
                using var doc = JsonDocument.Parse(body.AsMemory(StartsWithBom(body) ? 3 : 0));
                JsonElement root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object
                    || !root.TryGetProperty("messages", out JsonElement messages) || messages.ValueKind != JsonValueKind.Array)
                    return null;

                string? model = root.TryGetProperty("model", out JsonElement m) && m.ValueKind == JsonValueKind.String ? m.GetString() : null;
                if (root.TryGetProperty(ChunkCacheReport.PropertyName, out _))
                    return new WireBody(model, true, []);

                var parts = new List<WirePart>();
                int i = 0;
                foreach (JsonElement message in messages.EnumerateArray())
                {
                    if (message.ValueKind == JsonValueKind.Object && message.TryGetProperty("content", out JsonElement content))
                    {
                        if (content.ValueKind == JsonValueKind.String)
                        {
                            AddPart(parts, i, 0, content);
                        }
                        else if (content.ValueKind == JsonValueKind.Array)
                        {
                            int j = 0;
                            foreach (JsonElement part in content.EnumerateArray())
                            {
                                if (HasInlineChunk(part))
                                    return new WireBody(model, true, []);
                                if (part.ValueKind == JsonValueKind.Object && part.TryGetProperty("text", out JsonElement text)
                                    && text.ValueKind == JsonValueKind.String
                                    && (!part.TryGetProperty("type", out JsonElement type) || type.ValueEquals("text"u8)))
                                    AddPart(parts, i, j, text);
                                j++;
                            }
                        }
                    }

                    i++;
                }

                return new WireBody(model, false, parts);
            }
            catch (JsonException)
            {
                return null;
            }
        }

        private static void AddPart(List<WirePart> parts, int message, int part, JsonElement text)
        {
            try
            {
                parts.Add(new WirePart(message, part, text.GetString()!));
            }
            catch (InvalidOperationException)
            {
                // Not decodable as UTF-16 (a lone-surrogate escape): the server cannot read it either; never annotate it.
            }
        }

        /// <summary>An inline <c>chunk</c> on a part, or on a part nested in it (an Anthropic <c>tool_result</c>).</summary>
        private static bool HasInlineChunk(JsonElement part)
        {
            if (part.ValueKind != JsonValueKind.Object) return false;
            if (part.TryGetProperty("chunk", out _)) return true;
            if (part.TryGetProperty("content", out JsonElement nested) && nested.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement child in nested.EnumerateArray())
                {
                    if (child.ValueKind == JsonValueKind.Object && child.TryGetProperty("chunk", out _))
                        return true;
                }
            }

            return false;
        }
    }

    /// <summary>One accepted annotation.</summary>
    private sealed record Accepted(WirePart Part, int Start16, int End16, int Start8, int End8, ChunkSpan Span);

    /// <summary>The spans located, verified and ordered for one body.</summary>
    private sealed class ManifestPlan
    {
        private int _notLocated, _hashMismatch, _invalid, _overlapping, _overLimit;

        public List<Accepted> Accepted { get; } = [];

        public static ManifestPlan Build(WireBody wire, IReadOnlyList<ChunkPart> candidates, int? maxChunks)
        {
            var plan = new ManifestPlan();
            List<WirePart> parts = wire.Parts;
            var located = new bool[candidates.Count];
            var exactPart = new bool[parts.Count];
            var matches = new List<(int Part, int Candidate, int Offset)>();

            // 1. Whole-part matches.
            var byText = new Dictionary<string, List<int>>(StringComparer.Ordinal);
            for (int c = 0; c < candidates.Count; c++)
            {
                if (!byText.TryGetValue(candidates[c].Text, out List<int>? list))
                    byText[candidates[c].Text] = list = [];
                list.Add(c);
            }

            for (int p = 0; p < parts.Count; p++)
            {
                if (byText.TryGetValue(parts[p].Text, out List<int>? cs))
                {
                    exactPart[p] = true;
                    foreach (int c in cs)
                    {
                        matches.Add((p, c, 0));
                        located[c] = true;
                    }
                }
            }

            // 2. A suffix appended after registration.
            for (int p = 0; p < parts.Count; p++)
            {
                if (exactPart[p]) continue;
                string text = parts[p].Text;
                for (int c = 0; c < candidates.Count; c++)
                {
                    string t = candidates[c].Text;
                    if (t.Length > 0 && text.Length > t.Length && text.StartsWith(t, StringComparison.Ordinal))
                    {
                        matches.Add((p, c, 0));
                        located[c] = true;
                    }
                }
            }

            // 3. Inside a part (a bridge that joined texts), only for texts not found otherwise.
            for (int c = 0; c < candidates.Count; c++)
            {
                if (located[c] || candidates[c].Text.Length == 0) continue;
                for (int p = 0; p < parts.Count; p++)
                {
                    int at = parts[p].Text.IndexOf(candidates[c].Text, StringComparison.Ordinal);
                    if (at > 0)
                    {
                        matches.Add((p, c, at));
                        located[c] = true;
                    }
                }
            }

            for (int c = 0; c < candidates.Count; c++)
            {
                if (!located[c]) plan._notLocated += candidates[c].Spans.Count;
            }

            var resolved = new List<Accepted>();
            foreach ((int p, int c, int offset) in matches)
            {
                WirePart part = parts[p];
                ChunkPart candidate = candidates[c];
                foreach (ChunkSpan span in candidate.Spans)
                {
                    int s16 = offset + span.Start;
                    int e16 = offset + (span.End ?? candidate.Text.Length);
                    // A boundary inside a surrogate pair leaves half a pair in the span text: the lone-surrogate check
                    // catches both ends.
                    if (span.Start < 0 || s16 >= e16 || e16 > part.Text.Length || HasLoneSurrogate(part.Text.AsSpan(s16, e16 - s16)))
                    {
                        plan._invalid++;
                        continue;
                    }

                    ReadOnlySpan<char> text = part.Text.AsSpan(s16, e16 - s16);
                    if (!string.Equals(ChunkHash.OfSpan(text), span.Hash, StringComparison.Ordinal))
                    {
                        plan._hashMismatch++;
                        continue;
                    }

                    int s8 = Encoding.UTF8.GetByteCount(part.Text.AsSpan(0, s16));
                    resolved.Add(new Accepted(part, s16, e16, s8, s8 + Encoding.UTF8.GetByteCount(text), span));
                }
            }

            // Document order; drop overlaps (and duplicates) within a part, keeping the earlier span.
            resolved.Sort((a, b) =>
            {
                int k = a.Part.Message.CompareTo(b.Part.Message);
                if (k == 0) k = a.Part.Part.CompareTo(b.Part.Part);
                if (k == 0) k = a.Start16.CompareTo(b.Start16);
                return k != 0 ? k : a.End16.CompareTo(b.End16);
            });
            WirePart? current = null;
            int lastEnd = 0;
            foreach (Accepted a in resolved)
            {
                if (!ReferenceEquals(a.Part, current))
                {
                    current = a.Part;
                    lastEnd = 0;
                }

                if (a.Start16 < lastEnd)
                {
                    plan._overlapping++;
                    continue;
                }

                lastEnd = a.End16;
                plan.Accepted.Add(a);
            }

            if (maxChunks is int max && max >= 0 && plan.Accepted.Count > max)
            {
                plan._overLimit = plan.Accepted.Count - max;
                plan.Accepted.RemoveRange(max, plan._overLimit);
            }

            return plan;
        }

        public ChunkManifestOutcome Outcome(int registered, bool annotated, bool fellBack) =>
            new(annotated, annotated ? null : fellBack ? null : ChunkSkipReasons.NoSpans, registered, annotated ? Accepted.Count : 0,
                _notLocated, _hashMismatch, _invalid, _overlapping, _overLimit, fellBack);

        /// <summary>The <c>chunk_cache</c> object (spec 5.2); request options at their defaults are omitted.</summary>
        public byte[] WriteManifest(ChunkRequestOptions options)
        {
            var buffer = new ArrayBufferWriter<byte>(256 + (Accepted.Count * 200));
            using (var w = new Utf8JsonWriter(buffer))
            {
                w.WriteStartObject();
                w.WriteNumber("v", 1);
                if (!string.Equals(options.Policy, "exact", StringComparison.Ordinal))
                    w.WriteString("policy", options.Policy);
                if (options.Selector is not null || options.Budget is not null)
                {
                    w.WriteStartObject("recompute");
                    if (options.Selector is not null) w.WriteString("selector", options.Selector);
                    if (options.Budget is double budget) w.WriteNumber("budget", budget);
                    w.WriteEndObject();
                }

                if (!string.Equals(options.OnMiss, "prefill", StringComparison.Ordinal))
                    w.WriteString("on_miss", options.OnMiss);
                if (options.Strict)
                    w.WriteBoolean("strict", true);
                if (!string.Equals(options.Report, "summary", StringComparison.Ordinal))
                    w.WriteString("report", options.Report);

                w.WriteStartArray("chunks");
                foreach (Accepted a in Accepted)
                {
                    ChunkSpan s = a.Span;
                    w.WriteStartObject();
                    w.WriteString("id", CutId(s.Id));
                    w.WriteString("hash", s.Hash);
                    w.WriteStartObject("at");
                    w.WriteNumber("message", a.Part.Message);
                    w.WriteNumber("part", a.Part.Part);
                    if (a.Start16 != 0 || a.End16 != a.Part.Text.Length)
                    {
                        w.WriteNumber("start", a.Start8);
                        w.WriteNumber("end", a.End8);
                    }

                    w.WriteEndObject();
                    if (s.Chunker is not null) w.WriteString("chunker", s.Chunker);
                    if (s.Source is { } src)
                    {
                        w.WriteStartObject("source");
                        w.WriteString("path", src.Path);
                        w.WriteNumber("start_line", src.StartLine);
                        w.WriteNumber("end_line", src.EndLine);
                        if (src.Revision is not null) w.WriteString("rev", src.Revision);
                        w.WriteEndObject();
                    }

                    if (s.Volatile) w.WriteString("stability", "volatile");
                    if (s.NoStore) w.WriteBoolean("no_store", true);
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
            }

            return buffer.WrittenSpan.ToArray();
        }

        private static string CutId(string id)
        {
            if (id.Length <= MaxIdLength) return id;
            int n = MaxIdLength;
            if (char.IsHighSurrogate(id[n - 1])) n--; // never split a surrogate pair
            return id[..n];
        }

        private static bool HasLoneSurrogate(ReadOnlySpan<char> text)
        {
            for (int i = 0; i < text.Length; i++)
            {
                if (char.IsHighSurrogate(text[i]))
                {
                    if (i + 1 >= text.Length || !char.IsLowSurrogate(text[i + 1])) return true;
                    i++;
                }
                else if (char.IsLowSurrogate(text[i]))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
