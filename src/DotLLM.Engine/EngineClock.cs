using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DotLLM.Engine;

/// <summary>
/// The timestamp source the decode loop and the MTP round timer read (#1314/#1373). In production this is
/// <see cref="Stopwatch.GetTimestamp"/>; tests install a per-thread virtual clock so gate decisions that compare measured
/// milliseconds do not depend on wall time or scheduler preemption. Ticks are in <see cref="Stopwatch.Frequency"/> units.
/// </summary>
internal static class EngineClock
{
    [ThreadStatic] private static Func<long>? t_override;

    /// <summary>Installs a clock for the calling thread only (parallel tests do not see each other's); dispose to restore.</summary>
    internal static IDisposable Override(Func<long> timestamp)
    {
        var previous = t_override;
        t_override = timestamp;
        return new Restore(previous);
    }

    /// <summary>Current timestamp in <see cref="Stopwatch.Frequency"/> ticks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static long GetTimestamp() => t_override is { } f ? f() : Stopwatch.GetTimestamp();

    private sealed class Restore(Func<long>? previous) : IDisposable
    {
        public void Dispose() => t_override = previous;
    }
}
