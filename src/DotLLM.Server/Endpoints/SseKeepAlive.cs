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
    public static SseKeepAlive Start(HttpContext context, Func<string> reason, TimeSpan? interval = null) =>
        new(context, reason, interval ?? DefaultInterval);

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
                await keepAlive.StopAsync();
            }
        };

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
