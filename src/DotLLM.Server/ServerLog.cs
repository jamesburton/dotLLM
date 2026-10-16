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

    private static readonly AsyncLocal<string?> CurrentRequestTag = new();
    private static long _requestCounter;
    private static int _installed;

    /// <summary>
    /// The current request's tag (<c>req#12</c>) when the caller runs inside a request logged by <see cref="UseRequestLifecycleLog"/>;
    /// null otherwise (startup, idle sweep, tests).
    /// </summary>
    public static string? RequestTag => CurrentRequestTag.Value;

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
            CurrentRequestTag.Value = tag;
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
                CurrentRequestTag.Value = null;
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
