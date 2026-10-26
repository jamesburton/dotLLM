using System.Diagnostics;
using System.Globalization;

namespace DotLLM.Engine;

/// <summary>
/// llama.cpp <c>print_timing</c>-style per-request timing lines and a live prefill-progress pulse (#1288).
/// Off by default: with no <see cref="Sink"/> every entry point is a single null check, so the engine pays nothing.
/// The server installs a sink at startup (disable with <c>DOTLLM_LOG_TIMINGS=0</c>).
/// </summary>
/// <remarks>
/// Idea from xhinker/azllm (<c>tools/serve/session.hpp</c> <c>print_timing</c> / <c>TimingPulse</c>, Apache-2.0); no code
/// was copied. Prefill progress counts a chunk once <c>IModel.Forward</c> has returned for it. Every backend's
/// <c>Forward</c> returns host-side logits, so it has synchronised its device (CUDA <c>stream.Synchronize</c>, Vulkan fence
/// wait) before returning: the count is work the device completed, not work that was queued.
/// </remarks>
public static class RequestTimingLog
{
    /// <summary>Environment variable that disables the timing lines when set to <c>0</c>.</summary>
    public const string EnvVar = "DOTLLM_LOG_TIMINGS";

    /// <summary>Receives each formatted line. Null (the default) disables timing logs entirely.</summary>
    public static Action<string>? Sink { get; set; }

    /// <summary>How often a running prefill reports progress. Default 3 s.</summary>
    public static TimeSpan PulseInterval { get; set; } = TimeSpan.FromSeconds(3);

    /// <summary>
    /// Writes the per-request timing line for <paramref name="timings"/> when a sink is installed.
    /// </summary>
    /// <param name="timings">The finished request's timings.</param>
    public static void Report(in InferenceTimings timings)
    {
        if (Sink is { } sink)
            sink(Format(timings));
    }

    /// <summary>
    /// Writes the timing line inside <paramref name="context"/>, so a line produced on the scheduler's run loop still
    /// carries the request's ambient log scope (the server's <c>req#N</c> tag is an <see cref="AsyncLocal{T}"/>).
    /// </summary>
    /// <param name="timings">The finished request's timings.</param>
    /// <param name="context">The context captured at submission by <see cref="CaptureContext"/>, or null.</param>
    public static void Report(in InferenceTimings timings, ExecutionContext? context)
    {
        if (Sink is not { } sink)
            return;
        if (context is null)
        {
            sink(Format(timings));
            return;
        }

        ExecutionContext.Run(context, static s =>
        {
            var (k, line) = ((Action<string>, string))s!;
            k(line);
        }, (sink, Format(timings)));
    }

    /// <summary>Captures the caller's execution context for a later <see cref="Report(in InferenceTimings, ExecutionContext?)"/>; null (free) when logging is off.</summary>
    /// <returns>The captured context, or null.</returns>
    public static ExecutionContext? CaptureContext() => Sink is null ? null : ExecutionContext.Capture();

    /// <summary>
    /// Formats one request's timing line: prompt eval over <b>computed</b> tokens only (prefix-cache hits excluded),
    /// eval, the prompt-cache reuse count when a prefix was reused, and draft acceptance + mean accepted length when
    /// speculation ran.
    /// </summary>
    /// <param name="t">The finished request's timings.</param>
    /// <returns>The single-line summary.</returns>
    public static string Format(in InferenceTimings t)
    {
        int computed = Math.Max(0, t.PrefillTokenCount - t.CachedTokenCount);
        var ci = CultureInfo.InvariantCulture;
        string line = string.Format(ci,
            "timing: prompt eval {0:F1} ms / {1} tokens ({2:F2} ms/tok, {3:F1} tok/s) | eval {4:F1} ms / {5} tokens ({6:F2} ms/tok, {7:F1} tok/s)",
            t.PrefillTimeMs, computed, computed > 0 ? t.PrefillTimeMs / computed : 0, Rate(computed, t.PrefillTimeMs),
            t.DecodeTimeMs, t.DecodeTokenCount, t.DecodeTokenCount > 0 ? t.DecodeTimeMs / t.DecodeTokenCount : 0,
            Rate(t.DecodeTokenCount, t.DecodeTimeMs));
        if (t.CachedTokenCount > 0)
            line += string.Format(ci, " | prompt cache {0}/{1} tokens reused", t.CachedTokenCount, t.PrefillTokenCount);
        if (t.SpeculativeDraftTokens > 0)
        {
            line += string.Format(ci, " | draft acceptance {0:F3} ({1}/{2}), mean len {3:F2}",
                t.SpeculativeAcceptanceRate, Math.Min(t.SpeculativeAcceptedTokens, t.SpeculativeDraftTokens),
                t.SpeculativeDraftTokens, t.SpeculativeMeanAcceptedLength);
        }

        return line;
    }

    /// <summary>
    /// Starts a progress pulse for a prefill of <paramref name="total"/> computed tokens, or returns <c>null</c> (no
    /// allocation, no timer) when no sink is installed.
    /// </summary>
    /// <param name="total">Tokens this prefill computes (prefix-cache hits excluded).</param>
    /// <returns>The pulse to <see cref="PrefillPulse.Advance"/> after each completed chunk and dispose at the end; or null.</returns>
    public static PrefillPulse? BeginPrefill(int total) =>
        Sink is { } sink && total > 0 ? new PrefillPulse(sink, total, PulseInterval) : null;

    /// <summary>
    /// <see cref="BeginPrefill(int)"/> inside <paramref name="context"/>, so the pulse's timer (which captures the
    /// execution context) writes with the submitting request's log scope even when started on the scheduler loop.
    /// </summary>
    /// <param name="total">Tokens this prefill computes.</param>
    /// <param name="context">A context from <see cref="CaptureContext"/>, or null.</param>
    /// <returns>The pulse, or null when logging is off.</returns>
    public static PrefillPulse? BeginPrefill(int total, ExecutionContext? context)
    {
        if (context is null)
            return BeginPrefill(total);

        // A non-null context means logging was on at submission, so this closure is never paid when it is off.
        PrefillPulse? pulse = null;
        ExecutionContext.Run(context, _ => pulse = BeginPrefill(total), null);
        return pulse;
    }

    private static double Rate(int tokens, double ms) => tokens > 0 && ms > 0 ? tokens * 1000.0 / ms : 0;
}

/// <summary>
/// One running prefill's progress (#1288): the caller advances a completed-token counter after each chunk the device
/// finished; a timer reports it every <see cref="RequestTimingLog.PulseInterval"/>. A prefill shorter than one interval
/// prints nothing.
/// </summary>
public sealed class PrefillPulse : IDisposable
{
    private readonly Action<string> _sink;
    private readonly long _start = Stopwatch.GetTimestamp();
    private readonly Timer? _timer;
    private int _completed;

    /// <summary>Creates a pulse.</summary>
    /// <param name="sink">Receives the progress lines.</param>
    /// <param name="total">Tokens this prefill computes.</param>
    /// <param name="interval">Report interval; <see cref="Timeout.InfiniteTimeSpan"/> for none (tests tick <see cref="Pulse"/>).</param>
    internal PrefillPulse(Action<string> sink, int total, TimeSpan interval)
    {
        _sink = sink;
        Total = total;

        // The timer captures the execution context, so a server request tag (an AsyncLocal) still prefixes the line.
        if (interval > TimeSpan.Zero && interval != Timeout.InfiniteTimeSpan)
            _timer = new Timer(static s => ((PrefillPulse)s!).Pulse(), this, interval, interval);
    }

    /// <summary>Tokens this prefill computes.</summary>
    public int Total { get; }

    /// <summary>Tokens the device has completed so far.</summary>
    public int Completed => Volatile.Read(ref _completed);

    /// <summary>Records that the device completed <paramref name="tokens"/> more prompt tokens.</summary>
    /// <param name="tokens">Tokens in the chunk that just completed.</param>
    public void Advance(int tokens) => Interlocked.Add(ref _completed, tokens);

    /// <summary>Writes one progress line now.</summary>
    internal void Pulse()
    {
        int done = Completed;
        double seconds = Stopwatch.GetElapsedTime(_start).TotalSeconds;
        _sink(string.Format(CultureInfo.InvariantCulture,
            "prefill: {0}/{1} tokens ({2:F1}%), {3:F1} tok/s, {4:F1}s elapsed",
            done, Total, done * 100.0 / Total, seconds > 0 ? done / seconds : 0, seconds));
    }

    /// <inheritdoc/>
    public void Dispose() => _timer?.Dispose();
}
