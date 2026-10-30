using System.Globalization;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Degenerate-output guard (#1291): ends a request after <c>N</c> identical tokens in a row, as <c>finish_reason=length</c>.
/// Under memory pressure a sampler can pin on one token until <c>max_tokens</c>, which looks like a hang from outside; the
/// guard cuts it short and logs once, with free memory when the host installs a probe.
/// </summary>
/// <remarks>
/// Idea from xhinker/azllm <c>tools/serve/session.hpp</c> <c>RepeatGuard</c> (Apache-2.0); no code was copied. Stateful
/// (the current run), so one instance serves exactly one sequence.
/// </remarks>
public sealed class RepeatGuardStopCondition : IStopCondition
{
    private readonly int _limit;
    private int _token = -1;
    private int _run;

    /// <summary>Creates a guard that trips on the <paramref name="limit"/>-th identical token in a row.</summary>
    /// <param name="limit">Run length that ends the request; must be positive.</param>
    public RepeatGuardStopCondition(int limit)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(limit);
        _limit = limit;
    }

    /// <summary>Run length used when a request does not set <c>InferenceOptions.RepeatGuardLimit</c>. Default 256; 0 disables.</summary>
    public static int DefaultLimit { get; set; } = 256;

    /// <summary>Receives the one-line trip message; null (default) logs nothing.</summary>
    public static Action<string>? Log { get; set; }

    /// <summary>Describes free memory for the trip message (e.g. <c>"device free 812 MiB"</c>); null when unknown.</summary>
    public static Func<string?>? MemoryProbe { get; set; }

    /// <summary>
    /// The guard for one sequence, or <c>null</c> when disabled.
    /// </summary>
    /// <param name="limit">The request's limit; null uses <see cref="DefaultLimit"/>; 0 (or negative) disables.</param>
    /// <returns>A fresh guard, or null.</returns>
    public static RepeatGuardStopCondition? Create(int? limit)
    {
        int n = limit ?? DefaultLimit;
        return n > 0 ? new RepeatGuardStopCondition(n) : null;
    }

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText) => Check(tokenId, generatedTokens.Count);

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, ReadOnlySpan<char> decodedTail) => Check(tokenId, generatedTokens.Count);

    private StopResult Check(int tokenId, int generated)
    {
        if (tokenId == _token)
        {
            _run++;
        }
        else
        {
            _token = tokenId;
            _run = 1;
        }

        if (_run < _limit)
            return StopResult.Continue;

        if (Log is { } log)
        {
            string? memory = MemoryProbe?.Invoke();
            log(string.Format(CultureInfo.InvariantCulture,
                "repeat guard: token {0} repeated {1} times after {2} generated; ending as length{3}",
                tokenId, _run, generated, memory is null ? "" : " (" + memory + ")"));
        }

        return StopResult.StopInclude; // StopInclude maps to finish_reason=length everywhere
    }
}
