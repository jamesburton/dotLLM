using System.Diagnostics.CodeAnalysis;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server;

/// <summary>
/// Timestamped server console log (#1021): the <c>[dotllm] ...</c> lines the server and its loaders write, plus a terse request
/// lifecycle (arrival, lock wait, model load, completion / cancellation). Without timestamps an incident such as "a load hung behind
/// a streaming request until the client timed out" cannot be reconstructed from the console afterwards.
/// </summary>
/// <remarks>
/// <para>
/// Timestamps are applied by a <see cref="TextWriter"/> installed over <see cref="Console.Out"/> / <see cref="Console.Error"/>
/// (<see cref="InstallTimestamps"/>) rather than at each of the ~40 existing <c>Console.WriteLine("[dotllm] ...")</c> sites, so lines
/// from the model loaders and warm-up (which live in other assemblies) are stamped too. Only lines that START with <c>[dotllm]</c> are
/// stamped: Spectre.Console banners, spinners and progress bars pass through untouched.
/// </para>
/// <para>Format: <c>2026-10-16 14:12:54.921 [dotllm] Loading model from ...</c> (local time).</para>
/// </remarks>
public static class ServerLog
{
    /// <summary>The prefix every server log line starts with; only lines starting with it are timestamped.</summary>
    public const string Prefix = "[dotllm]";

    private static readonly AsyncLocal<RequestScope?> CurrentRequestScope = new();
    private static long _requestCounter;
    private static int _installed;

    /// <summary>
    /// The current request's tag (<c>req#12</c>) when the caller runs inside a request logged by <see cref="UseRequestLifecycleLog"/>;
    /// null otherwise (startup, idle sweep, tests).
    /// </summary>
    public static string? RequestTag => CurrentRequestScope.Value?.Tag;

    /// <summary>The current request (its tag, and a task that completes when it ends), or null outside one.</summary>
    internal static RequestScope? CurrentRequest => CurrentRequestScope.Value;

    /// <summary>
    /// Enters a request scope for the current async flow: sets <see cref="RequestTag"/> and, on dispose, completes
    /// <see cref="RequestScope.Ended"/> and clears it. Used by <see cref="UseRequestLifecycleLog"/> and by tests.
    /// </summary>
    /// <param name="tag">The request's tag.</param>
    /// <returns>The scope; dispose it when the request ends.</returns>
    internal static RequestScope BeginRequest(string tag)
    {
        var scope = new RequestScope(tag);
        CurrentRequestScope.Value = scope;
        return scope;
    }

    /// <summary>One in-flight request: its tag and a task that completes when it has ended (#1023 activation hand-off).</summary>
    internal sealed class RequestScope(string tag) : IDisposable
    {
        private readonly TaskCompletionSource _ended = new(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>The request's tag (<c>req#12</c>).</summary>
        public string Tag { get; } = tag;

        /// <summary>Completes when the request has ended.</summary>
        public Task Ended => _ended.Task;

        /// <inheritdoc/>
        public void Dispose()
        {
            _ended.TrySetResult();
            if (ReferenceEquals(CurrentRequestScope.Value, this)) CurrentRequestScope.Value = null;
        }
    }

    /// <summary>
    /// Replaces <see cref="Console.Out"/> and <see cref="Console.Error"/> with writers that prefix each <c>[dotllm]</c> line with a
    /// local timestamp. Idempotent; call once at server start, before any model is loaded.
    /// </summary>
    public static void InstallTimestamps()
    {
        if (Interlocked.Exchange(ref _installed, 1) == 1) return;
        Console.SetOut(new TimestampingWriter(Console.Out));
        Console.SetError(new TimestampingWriter(Console.Error));
    }

    /// <summary>
    /// Routes the engine's per-request timing lines and prefill-progress pulse (#1288) to <see cref="Info"/>, unless
    /// <paramref name="envValue"/> (<c>DOTLLM_LOG_TIMINGS</c>) is <c>0</c>.
    /// </summary>
    /// <param name="envValue">The <c>DOTLLM_LOG_TIMINGS</c> value, or null when unset.</param>
    /// <returns>Whether timing lines are now on.</returns>
    public static bool InstallTimingLog(string? envValue)
    {
        if (envValue?.Trim() == "0")
            return false;
        DotLLM.Engine.RequestTimingLog.Sink = Info;
        return true;
    }

    /// <summary>
    /// Routes repeat-guard trips (#1291) to <see cref="Info"/>, with free memory of the serving device: NVML for a CUDA
    /// device, else the host's available physical memory.
    /// </summary>
    /// <param name="resolvedDevice">Returns the device currently serving (<c>cuda</c>, <c>cuda:1</c>, <c>cpu</c>, ...), read at trip time.</param>
    public static void InstallRepeatGuardLog(Func<string?> resolvedDevice)
    {
        DotLLM.Engine.Samplers.StopConditions.RepeatGuardStopCondition.Log = Info;
        DotLLM.Engine.Samplers.StopConditions.RepeatGuardStopCondition.MemoryProbe = () => FreeMemory(resolvedDevice());
    }

    /// <summary>Describes free memory for <paramref name="device"/>: <c>"cuda:0 free 812 MiB"</c> or <c>"host free 9123 MiB"</c>.</summary>
    /// <param name="device">The serving device, or null.</param>
    /// <returns>The description.</returns>
    internal static string FreeMemory(string? device)
    {
        if (device is not null && device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase))
        {
            int ordinal = device.IndexOf(':') is int c and >= 0 && int.TryParse(device.AsSpan(c + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out int o) ? o : 0;
            if (DotLLM.Cuda.CudaMemoryInfo.TryGet(ordinal, out long free, out _))
                return FormattableString.Invariant($"cuda:{ordinal} free {free >> 20} MiB");
        }

        var gc = GC.GetGCMemoryInfo();
        return FormattableString.Invariant($"host free {Math.Max(0, gc.TotalAvailableMemoryBytes - gc.MemoryLoadBytes) >> 20} MiB");
    }

    /// <summary>Writes <c>[dotllm] {req#N }{message}</c> to the console (timestamped once <see cref="InstallTimestamps"/> ran).</summary>
    /// <param name="message">The log message, without the prefix.</param>
    public static void Info(string message) =>
        Console.WriteLine(RequestTag is { } tag ? $"{Prefix} {tag} {message}" : $"{Prefix} {message}");

    /// <summary>Formats an elapsed time tersely: <c>850ms</c>, <c>12.3s</c>, <c>4m05s</c>.</summary>
    /// <param name="elapsed">The duration to format.</param>
    /// <returns>The formatted duration.</returns>
    public static string Duration(TimeSpan elapsed) =>
        elapsed.TotalSeconds < 1 ? $"{elapsed.TotalMilliseconds:F0}ms"
        : elapsed.TotalMinutes < 1 ? elapsed.TotalSeconds.ToString("F1", CultureInfo.InvariantCulture) + "s"
        : $"{(int)elapsed.TotalMinutes}m{elapsed.Seconds:D2}s";

    /// <summary>The timestamp prefix for <paramref name="time"/>: <c>yyyy-MM-dd HH:mm:ss.fff</c>.</summary>
    /// <param name="time">The instant to format (local time).</param>
    /// <returns>The formatted timestamp.</returns>
    public static string Timestamp(DateTime time) => time.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture);

    /// <summary>
    /// Logs each inference-shaped request (every <c>POST</c>): arrival with its endpoint, and completion with status and duration -
    /// or cancellation when the client disconnected first. Sets <see cref="RequestTag"/> for the request so the lock-wait and
    /// model-load lines <see cref="ServerState"/> writes can be attributed to it. <c>GET</c>s (health / props polling) are not logged.
    /// </summary>
    /// <param name="app">The application to add the middleware to.</param>
    /// <returns><paramref name="app"/>.</returns>
    public static IApplicationBuilder UseRequestLifecycleLog(this IApplicationBuilder app) =>
        app.Use(async (HttpContext context, Func<Task> next) =>
        {
            if (!HttpMethods.IsPost(context.Request.Method))
            {
                await next();
                return;
            }

            string tag = "req#" + Interlocked.Increment(ref _requestCounter).ToString(CultureInfo.InvariantCulture);
            using var scope = BeginRequest(tag);
            long start = Stopwatch.GetTimestamp();
            Info($"{context.Request.Method} {context.Request.Path}");
            try
            {
                await next();
            }
            finally
            {
                string elapsed = Duration(Stopwatch.GetElapsedTime(start));
                Info(context.RequestAborted.IsCancellationRequested
                    ? $"cancelled (client disconnected) after {elapsed}"
                    : $"done {context.Response.StatusCode} in {elapsed}");
            }
        });

    /// <summary>
    /// Prefixes each line that starts with <see cref="Prefix"/> with a timestamp. <see cref="Console.SetOut"/> wraps this in a
    /// synchronized writer, so the line-start state is never raced.
    /// </summary>
    internal sealed class TimestampingWriter(TextWriter inner, Func<DateTime>? clock = null) : TextWriter
    {
        private readonly Func<DateTime> _clock = clock ?? (() => DateTime.Now);
        private bool _atLineStart = true;

        public override Encoding Encoding => inner.Encoding;

        public override IFormatProvider FormatProvider => inner.FormatProvider;

        [AllowNull]
        public override string NewLine
        {
            get => inner.NewLine;
            set => inner.NewLine = value;
        }

        public override void Write(char value)
        {
            inner.Write(value);
            _atLineStart = value == '\n';
        }

        public override void Write(char[] buffer, int index, int count) => Write(new string(buffer, index, count));

        public override void Write(string? value)
        {
            if (string.IsNullOrEmpty(value)) return;
            if (_atLineStart && value.StartsWith(Prefix, StringComparison.Ordinal))
                inner.Write(Timestamp(_clock()) + " ");
            inner.Write(value);
            _atLineStart = value[^1] == '\n';
        }

        public override void WriteLine(string? value)
        {
            Write(value);
            inner.WriteLine();
            _atLineStart = true;
        }

        public override void WriteLine()
        {
            inner.WriteLine();
            _atLineStart = true;
        }

        public override void Flush() => inner.Flush();
    }
}
