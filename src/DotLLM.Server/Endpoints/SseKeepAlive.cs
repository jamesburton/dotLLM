namespace DotLLM.Server.Endpoints;

/// <summary>
/// SSE keep-alive for a streaming request that is waiting - for a model load, or for the execution lock (#1023).
/// </summary>
/// <remarks>
/// <para>
/// Every <see cref="DefaultInterval"/> while the request waits, an SSE <b>comment</b> line is written:
/// <c>: waiting for loading owner/repo/x.gguf (started 2026-10-16T14:12:54Z, 73s)</c> followed by a blank line. Comment lines (a
/// leading colon) are part of the SSE spec and are ignored by EventSource and the OpenAI / Anthropic SDKs, so a client that does not
/// know them is unaffected, while one that reads the raw stream can tell "queued and alive" from "dead". Nothing is written when the
/// wait is shorter than one interval, so the fast path's bytes are unchanged.
/// </para>
/// <para>
/// The first keep-alive commits the response (200, <c>text/event-stream</c>); any later error must then be reported in-stream.
/// <see cref="StopAsync"/> waits for an in-flight write, so the caller's own writes never interleave with a keep-alive.
/// </para>
/// </remarks>
internal sealed class SseKeepAlive : IAsyncDisposable
{
    /// <summary>Interval between keep-alive comments.</summary>
    public static readonly TimeSpan DefaultInterval = TimeSpan.FromSeconds(10);

    /// <summary>The <see cref="HttpContext.Items"/> key the running keep-alive of a request is registered under.</summary>
    private const string ItemKey = "dotllm.sse-keepalive";

    private readonly HttpContext _context;
    private readonly Func<string> _reason;
    private readonly CancellationTokenSource _stop;
    private readonly Task _loop;
    private volatile bool _started;

    private SseKeepAlive(HttpContext context, Func<string> reason, TimeSpan interval)
    {
        _context = context;
        _reason = reason;
        _stop = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);
        _loop = RunAsync(interval);
    }

    /// <summary>True once a keep-alive has been written: the response has started, so errors must go in-stream.</summary>
    public bool Started => _started;

    /// <summary>Starts sending keep-alives for <paramref name="context"/>.</summary>
    /// <param name="context">The streaming request.</param>
    /// <param name="reason">What the request is waiting for, re-read for every comment.</param>
    /// <param name="interval">Interval between comments; <see cref="DefaultInterval"/> when null.</param>
    /// <returns>The running keep-alive; stop it before writing the first real event.</returns>
    public static SseKeepAlive Start(HttpContext context, Func<string> reason, TimeSpan? interval = null)
    {
        var keepAlive = new SseKeepAlive(context, reason, interval ?? DefaultInterval);
        context.Items[ItemKey] = keepAlive;
        return keepAlive;
    }

    /// <summary>
    /// Stops the request's running keep-alive, if any, and waits for an in-flight comment (#1023 review). Call before touching the
    /// response headers or writing anything: a keep-alive commits the headers itself, so setting them concurrently threw
    /// "headers are read-only".
    /// </summary>
    /// <param name="context">The request.</param>
    /// <returns>A task that completes when no keep-alive can write any more.</returns>
    public static Task StopForAsync(HttpContext context) =>
        context.Items.TryGetValue(ItemKey, out var value) && value is SseKeepAlive keepAlive ? keepAlive.StopAsync() : Task.CompletedTask;

    /// <summary>
    /// Wraps a model gate so keep-alives run while <paramref name="execute"/> waits, and stop before the work starts.
    /// </summary>
    /// <param name="context">The streaming request.</param>
    /// <param name="execute">The gate (e.g. <see cref="ServerState.ExecuteAsync(Func{Task}, CancellationToken)"/>).</param>
    /// <param name="reason">What the request is waiting for.</param>
    /// <param name="interval">Interval between comments; <see cref="DefaultInterval"/> when null.</param>
    /// <param name="existing">A keep-alive already running (from the activation phase) to continue rather than start anew.</param>
    /// <returns>The wrapped gate.</returns>
    public static Func<Func<Task>, CancellationToken, Task> Wrap(
        HttpContext context, Func<Func<Task>, CancellationToken, Task> execute, Func<string> reason,
        TimeSpan? interval = null, SseKeepAlive? existing = null) =>
        async (work, ct) =>
        {
            if (existing is null) await StopForAsync(context);   // never two writers on one response
            var keepAlive = existing ?? Start(context, reason, interval);
            try
            {
                await execute(async () =>
                {
                    await keepAlive.StopAsync();
                    await work();
                }, ct);
            }
            finally
            {
                // One this wrapper started is disposed (its linked token source too); a caller's own is only stopped.
                if (existing is null) await keepAlive.DisposeAsync();
                else await keepAlive.StopAsync();
            }
        };

    /// <summary>
    /// Passes <paramref name="source"/> through and sends keep-alive comments until its first item arrives (issue #1025). The
    /// role chunk goes out when the request takes the gate, but the first token waits for the prompt prefill, which takes
    /// minutes on a long agent prompt. Without this the stream was silent all that time, so a client could not tell a
    /// long prefill from a dead server, and a read timeout dropped it.
    /// </summary>
    /// <typeparam name="T">Item type.</typeparam>
    /// <param name="context">The streaming request, whose response has already started.</param>
    /// <param name="source">The token source. Its first <c>MoveNextAsync</c> runs the prefill.</param>
    /// <param name="interval">Interval between comments; <see cref="DefaultInterval"/> when null.</param>
    /// <param name="ct">Cancels the enumeration.</param>
    /// <returns>The items of <paramref name="source"/>, unchanged.</returns>
    public static async IAsyncEnumerable<T> UntilFirstItem<T>(
        HttpContext context, IAsyncEnumerable<T> source, TimeSpan? interval = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        await StopForAsync(context);   // never two writers on one response
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        SseKeepAlive? keepAlive = Start(context,
            () => $"prompt processing ({System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalSeconds:F0}s)", interval);
        try
        {
            await foreach (var item in source.WithCancellation(ct).ConfigureAwait(false))
            {
                if (keepAlive is not null)
                {
                    // Waits for an in-flight comment, so the first real event never interleaves with one.
                    await keepAlive.DisposeAsync().ConfigureAwait(false);
                    keepAlive = null;
                }
                yield return item;
            }
        }
        finally
        {
            if (keepAlive is not null) await keepAlive.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>Stops the keep-alives and waits for any in-flight write. Idempotent.</summary>
    /// <returns>A task that completes when no keep-alive can be written any more.</returns>
    public async Task StopAsync()
    {
        try { _stop.Cancel(); } catch (ObjectDisposedException) { }
        try { await _loop.ConfigureAwait(false); } catch { /* the loop never throws; belt and braces */ }
    }

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        _stop.Dispose();
    }

    private async Task RunAsync(TimeSpan interval)
    {
        try
        {
            while (true)
            {
                await Task.Delay(interval, _stop.Token).ConfigureAwait(false);
                if (_stop.IsCancellationRequested) return;
                if (!_context.Response.HasStarted)
                    SseResponse.ApplyHeaders(_context);
                _started = true;
                string line = _reason().Replace('\r', ' ').Replace('\n', ' ');
                // Written with RequestAborted, not the stop token: a stop must never cut a comment in half.
                await _context.Response.WriteAsync($": waiting for {line}\n\n", _context.RequestAborted).ConfigureAwait(false);
                await _context.Response.Body.FlushAsync(_context.RequestAborted).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) { /* stopped, or the client went away */ }
        catch (Exception) { /* a failed keep-alive must never fail the request; the real write will surface it */ }
    }
}
