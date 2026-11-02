// SPDX-License-Identifier: MIT
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;

namespace DotLLM.ChunkCache.Client;

/// <summary>Per-call facts a <see cref="IRecomputePolicy"/> may consider that only the harness knows.</summary>
/// <param name="IsSubAgent">The call comes from a sub-agent.</param>
/// <param name="AfterCompaction">The context was just compacted.</param>
/// <param name="UserOptedIntoApproximation">The user or session explicitly allowed lossy reuse (spec 11 item 3).</param>
public readonly record struct ChunkSessionFlags(bool IsSubAgent, bool AfterCompaction, bool UserOptedIntoApproximation);

/// <summary>Optional settings of <see cref="ChunkAnnotatingChatClient"/>. Not a wire DTO.</summary>
public sealed class ChunkAnnotatingOptions
{
    /// <summary>
    /// The capability source shared with the <see cref="ChunkCacheBodyHandler"/>. When set, a call to a server that is
    /// not capable opens no scope at all, and the policy sees the capabilities. The handler gates on its own source
    /// either way.
    /// </summary>
    public IChunkCapabilitySource? Capabilities { get; set; }

    /// <summary>
    /// The harness's per-call flags (sub-agent, after compaction, approximation opt-in). Unset means all false. Never read
    /// from <see cref="ChatOptions.AdditionalProperties"/>, which some bridges put on the wire.
    /// </summary>
    public Func<ChatOptions?, ChunkSessionFlags>? SessionFlags { get; set; }
}

/// <summary>One completed call that went out with a manifest scope.</summary>
/// <param name="options">The request options (after the no-lossy-default guard).</param>
/// <param name="outcome">What the body handler did (last attempt); <see langword="null"/> if no request reached it.</param>
/// <param name="report">The server's <c>chunk_cache</c> report, if any.</param>
public sealed class ChunkCacheCallEventArgs(ChunkRequestOptions options, ChunkManifestOutcome? outcome, JsonObject? report) : EventArgs
{
    /// <summary>The request options sent.</summary>
    public ChunkRequestOptions Options { get; } = options;

    /// <summary>What the body handler did: spans sent and dropped, and why (the spec 12.5 "plumbing loss").</summary>
    public ChunkManifestOutcome? Outcome { get; } = outcome;

    /// <summary>The server's report (policy applied, counts, warnings); log it per call (spec 11 item 10).</summary>
    public JsonObject? Report { get; } = report;
}

/// <summary>
/// The MEAI decorator of spec 8.3: finds the chunk-bearing parts of the outgoing messages (through the registry, by part
/// <b>text</b>), decides the request options, and hands both to <see cref="ChunkCacheBodyHandler"/> through a
/// <see cref="ChunkManifestScope"/>; lifts the response's <c>chunk_cache</c> into
/// <see cref="ChatResponse.AdditionalProperties"/> (streaming: onto an update, so <c>ToChatResponse</c> keeps it) and
/// raises <see cref="CallCompleted"/>.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><b>Placement</b> (spec 12.1): below any retrying client, so every retry rebuilds the same manifest, and above the
/// provider. The body handler goes on the provider's <see cref="HttpClient"/>.</item>
/// <item><b>Nothing lossy by default.</b> The default policy is <see cref="ExactPolicy"/>; a policy other than
/// <c>exact</c> or <c>shadow</c> is sent as <c>exact</c> unless <see cref="ChunkSessionFlags.UserOptedIntoApproximation"/>
/// is set for the call.</item>
/// <item><b>Idempotent.</b> Inside another decorator's scope (two decorators stacked) it passes the call through, so the
/// outermost decorator's manifest is the only one; the body handler leaves a body that already carries annotations alone.</item>
/// <item>Texts read: <see cref="TextContent.Text"/> and a string <see cref="FunctionResultContent.Result"/>. With nothing
/// registered, or no capabilities, no scope is opened and the request is the plain one.</item>
/// </list>
/// </remarks>
public sealed class ChunkAnnotatingChatClient : DelegatingChatClient
{
    private readonly IChunkSpanRegistry _registry;
    private readonly IRecomputePolicy _policy;
    private readonly ChunkAnnotatingOptions _options;

    /// <summary>Creates the decorator (spec 8.3 signature).</summary>
    public ChunkAnnotatingChatClient(IChatClient inner, IChunkSpanRegistry registry, IRecomputePolicy policy)
        : this(inner, registry, policy, null)
    {
    }

    /// <summary>Creates the decorator with options.</summary>
    public ChunkAnnotatingChatClient(IChatClient inner, IChunkSpanRegistry registry, IRecomputePolicy policy, ChunkAnnotatingOptions? options)
        : base(inner)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(policy);
        _registry = registry;
        _policy = policy;
        _options = options ?? new ChunkAnnotatingOptions();
    }

    /// <summary>Raised after every call that went out with a manifest scope (blocking: after the response; streaming: after the last update).</summary>
    public event EventHandler<ChunkCacheCallEventArgs>? CallCompleted;

    /// <inheritdoc />
    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (ChunkManifestScope.Current is not null)
            return await base.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);

        IList<ChatMessage> list = messages as IList<ChatMessage> ?? messages.ToList();
        Prepared? prepared = await PrepareAsync(list, options, cancellationToken).ConfigureAwait(false);
        if (prepared is null)
            return await base.GetResponseAsync(list, options, cancellationToken).ConfigureAwait(false);

        using ChunkManifestScope scope = ChunkManifestScope.Begin(prepared.Parts, prepared.Options);
        ChatResponse response = await base.GetResponseAsync(list, options, cancellationToken).ConfigureAwait(false);
        ChunkCacheReport.Attach(response, scope.ResponseReport);
        CallCompleted?.Invoke(this, new ChunkCacheCallEventArgs(prepared.Options, scope.Outcome, scope.ResponseReport));
        return response;
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        IList<ChatMessage> list = messages as IList<ChatMessage> ?? messages.ToList();
        Prepared? prepared = ChunkManifestScope.Current is null
            ? await PrepareAsync(list, options, cancellationToken).ConfigureAwait(false)
            : null;
        if (prepared is null)
        {
            await foreach (ChatResponseUpdate u in base.GetStreamingResponseAsync(list, options, cancellationToken).ConfigureAwait(false))
                yield return u;
            yield break;
        }

        using ChunkManifestScope scope = ChunkManifestScope.Begin(prepared.Parts, prepared.Options);
        bool attached = false;
        ChatResponseUpdate? last = null;
        IAsyncEnumerator<ChatResponseUpdate> e = base.GetStreamingResponseAsync(list, options, cancellationToken).GetAsyncEnumerator(cancellationToken);
        try
        {
            while (true)
            {
                // An async iterator resumes on the caller's context, where the AsyncLocal is not set: re-enter the scope
                // so a request (or a retry) made during this step still sees it.
                ChunkManifestScope.Enter(scope);
                if (!await e.MoveNextAsync().ConfigureAwait(false))
                    break;
                ChatResponseUpdate update = e.Current;
                if (!attached && scope.ResponseReport is { } report)
                {
                    ChunkCacheReport.Attach(update, report);
                    attached = true;
                }

                last = update;
                yield return update;
            }
        }
        finally
        {
            await e.DisposeAsync().ConfigureAwait(false);
        }

        if (!attached && scope.ResponseReport is { } late)
        {
            // The report was read after the last update was produced: carry it on one more, content-free update.
            var carrier = new ChatResponseUpdate
            {
                ResponseId = last?.ResponseId, MessageId = last?.MessageId, ConversationId = last?.ConversationId,
                ModelId = last?.ModelId, CreatedAt = last?.CreatedAt, Role = last?.Role,
            };
            ChunkCacheReport.Attach(carrier, late);
            yield return carrier;
        }

        CallCompleted?.Invoke(this, new ChunkCacheCallEventArgs(prepared.Options, scope.Outcome, scope.ResponseReport));
    }

    private sealed record Prepared(IReadOnlyList<ChunkPart> Parts, ChunkRequestOptions Options);

    /// <summary>The registered parts and the request options, or <see langword="null"/> when the call goes out plain.</summary>
    private async ValueTask<Prepared?> PrepareAsync(IList<ChatMessage> messages, ChatOptions? options, CancellationToken ct)
    {
        var parts = new List<ChunkPart>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        long promptChars = 0, chunkChars = 0;
        foreach (ChatMessage message in messages)
        {
            foreach (AIContent content in message.Contents)
            {
                string? text = content switch
                {
                    TextContent t => t.Text,
                    FunctionResultContent { Result: string s } => s,
                    _ => null,
                };
                if (string.IsNullOrEmpty(text)) continue;
                promptChars += text.Length;
                if (_registry.TryFind(text, out string? registered, out IReadOnlyList<ChunkSpan> spans) && spans.Count > 0 && seen.Add(registered))
                {
                    parts.Add(new ChunkPart(registered, spans));
                    foreach (ChunkSpan s in spans)
                        chunkChars += (s.End ?? registered.Length) - s.Start;
                }
            }
        }

        if (parts.Count == 0)
            return null;

        ChunkCacheCapabilities? caps = null;
        if (_options.Capabilities is { } source)
        {
            try
            {
                caps = await source.GetAsync(options?.ModelId, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception)
            {
                caps = null; // ruling 16
            }

            if (caps is null || !caps.SupportsVersion(1))
                return null;
        }

        ChunkSessionFlags flags = _options.SessionFlags?.Invoke(options) ?? default;
        var context = new ChunkRequestContext(options?.ModelId ?? string.Empty, EstimateTokens(promptChars), EstimateTokens(chunkChars), caps,
            flags.IsSubAgent, flags.AfterCompaction, flags.UserOptedIntoApproximation);
        ChunkRequestOptions request = _policy.Decide(context) ?? new ChunkRequestOptions();
        if (!flags.UserOptedIntoApproximation && !IsNonLossy(request.Policy))
            request = request with { Policy = "exact", Selector = null, Budget = null };

        return new Prepared(parts, request);
    }

    private static bool IsNonLossy(string? policy) =>
        string.Equals(policy, "exact", StringComparison.Ordinal) || string.Equals(policy, "shadow", StringComparison.Ordinal);

    /// <summary>A rough token estimate (4 UTF-16 chars per token), for the policy only.</summary>
    private static int EstimateTokens(long chars) => (int)Math.Min(int.MaxValue, (chars + 3) / 4);
}
