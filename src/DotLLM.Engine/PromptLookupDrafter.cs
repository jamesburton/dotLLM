using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DotLLM.Engine;

/// <summary>
/// Settings for prompt-lookup (n-gram) drafting (issue #1293): the last tokens of prompt + output are matched against earlier
/// context and the tokens that followed the match are proposed as drafts, verified by the target exactly like a draft model's.
/// </summary>
/// <param name="DraftTokens">Maximum drafts proposed per round (K).</param>
/// <param name="MaxNgram">Longest suffix matched; a match of this length ends the search early.</param>
/// <param name="MinNgram">Shortest suffix accepted as a match. Also the key length of the index.</param>
public sealed record PromptLookupOptions(
    int DraftTokens = PromptLookupOptions.DefaultDraftTokens,
    int MaxNgram = PromptLookupOptions.DefaultMaxNgram,
    int MinNgram = PromptLookupOptions.DefaultMinNgram)
{
    /// <summary>Default drafts per round: a hit usually copies a run of text, so a longer block than a model drafter's pays off.</summary>
    public const int DefaultDraftTokens = 8;

    /// <summary>Default longest matched suffix (azllm's fixed n-gram).</summary>
    public const int DefaultMaxNgram = 6;

    /// <summary>Default shortest matched suffix.</summary>
    public const int DefaultMinNgram = 2;

    /// <summary>Settings from the CLI / server flags: the shortest match is lowered to <paramref name="maxNgram"/> when that is below the default.</summary>
    /// <param name="draftTokens">Maximum drafts per round.</param>
    /// <param name="maxNgram">Longest matched suffix.</param>
    /// <returns>The validated settings.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A value is below 1.</exception>
    public static PromptLookupOptions Create(int draftTokens, int maxNgram)
    {
        var options = new PromptLookupOptions(draftTokens, maxNgram, Math.Clamp(maxNgram, 1, DefaultMinNgram));
        options.Validate();
        return options;
    }

    /// <summary>Throws when the settings are out of range.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A value is below 1, or <see cref="MinNgram"/> exceeds <see cref="MaxNgram"/>.</exception>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(DraftTokens, 1, nameof(DraftTokens));
        ArgumentOutOfRangeException.ThrowIfLessThan(MinNgram, 1, nameof(MinNgram));
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxNgram, MinNgram, nameof(MaxNgram));
    }
}

/// <summary>
/// Prompt-lookup draft source (issue #1293): finds the latest earlier occurrence of the longest suffix (between
/// <see cref="PromptLookupOptions.MinNgram"/> and <see cref="PromptLookupOptions.MaxNgram"/> tokens) of the context and proposes
/// the tokens that followed it. Idea from xhinker/azllm <c>qwen35.cpp</c> <c>lookup()</c> (Apache-2.0) and llama.cpp's prompt lookup.
/// </summary>
/// <remarks>
/// The index is a hash chain keyed by the <see cref="PromptLookupOptions.MinNgram"/>-gram ending at each position: <c>_head[h]</c> is
/// the latest such position, <c>_prev[i]</c> the one before it with the same bucket. <see cref="Append"/> is O(1), the search walks at
/// most <see cref="MaxChainSteps"/> candidates newest first and confirms each by comparing tokens (bucket collisions never match
/// falsely). All buffers are pooled and sized once to the sequence capacity, so the decode loop allocates nothing.
/// </remarks>
public sealed class PromptLookupDrafter : IDisposable
{
    /// <summary>Candidates examined per lookup: bounds the cost on degenerate, highly repetitive context.</summary>
    public const int MaxChainSteps = 64;

    private readonly int _minN;
    private readonly int _maxN;
    private readonly int _capacity;
    private readonly int _mask;
    private int[] _tokens;
    private int[] _prev;
    private int[] _head;
    private int _count;

    /// <summary>Creates a drafter over <paramref name="prompt"/> followed by <paramref name="generated"/>.</summary>
    /// <param name="options">N-gram bounds.</param>
    /// <param name="prompt">Prompt tokens.</param>
    /// <param name="generated">Tokens generated so far (may be empty).</param>
    /// <param name="capacity">Maximum context length (the KV-cache size); tokens appended past it are ignored.</param>
    public PromptLookupDrafter(PromptLookupOptions options, ReadOnlySpan<int> prompt, ReadOnlySpan<int> generated, int capacity)
    {
        options.Validate();
        _minN = options.MinNgram;
        _maxN = options.MaxNgram;
        _capacity = Math.Max(capacity, prompt.Length + generated.Length);
        int buckets = (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1024, _capacity));
        _mask = buckets - 1;
        _tokens = ArrayPool<int>.Shared.Rent(_capacity);
        _prev = ArrayPool<int>.Shared.Rent(_capacity);
        _head = ArrayPool<int>.Shared.Rent(buckets);
        _head.AsSpan(0, buckets).Fill(-1);
        foreach (int t in prompt) Append(t);
        foreach (int t in generated) Append(t);
    }

    /// <summary>Tokens indexed so far.</summary>
    public int Count => _count;

    /// <summary>Adds the next context token to the index.</summary>
    /// <param name="token">The token.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(int token)
    {
        if (_count >= _capacity) return;
        int i = _count++;
        _tokens[i] = token;
        if (i + 1 < _minN) { _prev[i] = -1; return; }
        int h = Hash(i);
        _prev[i] = _head[h];
        _head[h] = i;
    }

    /// <summary>
    /// Writes the continuation of the best match of the current suffix into <paramref name="drafts"/>: the longest suffix match
    /// (up to <see cref="PromptLookupOptions.MaxNgram"/>), the latest occurrence on a tie.
    /// </summary>
    /// <param name="drafts">Destination; its length caps the number of drafts.</param>
    /// <returns>Drafts written; 0 when no suffix of at least <see cref="PromptLookupOptions.MinNgram"/> tokens recurs.</returns>
    public int Draft(Span<int> drafts) => Draft(drafts, out _);

    /// <summary>As <see cref="Draft(Span{int})"/>, also reporting the matched suffix length.</summary>
    /// <param name="drafts">Destination; its length caps the number of drafts.</param>
    /// <param name="matchLength">Length of the matched suffix, 0 when nothing matched.</param>
    /// <returns>Drafts written.</returns>
    public int Draft(Span<int> drafts, out int matchLength)
    {
        matchLength = 0;
        int last = _count - 1;
        if (drafts.IsEmpty || last < _minN) return 0;

        int best = -1;
        int steps = 0;
        for (int e = _head[Hash(last)]; e >= 0 && steps < MaxChainSteps; e = _prev[e], steps++)
        {
            if (e >= last) continue;   // the suffix itself
            int m = 0;
            while (m < _maxN && e - m >= 0 && _tokens[e - m] == _tokens[last - m]) m++;
            if (m > matchLength)
            {
                matchLength = m;
                best = e;
                if (m == _maxN) break;
            }
        }
        if (matchLength < _minN) { matchLength = 0; return 0; }

        // The continuation starts after the match. When the match overlaps the suffix (a periodic run) it runs past the end of the
        // context; it then continues the period from the drafts already written, LZ77-style.
        int period = last - best;
        int n = drafts.Length;
        for (int j = 0; j < n; j++)
        {
            int src = best + 1 + j;
            drafts[j] = src <= last ? _tokens[src] : drafts[j - period];
        }
        return n;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Hash(int end)
    {
        uint h = 2166136261u;
        for (int k = end - _minN + 1; k <= end; k++)
            h = (h ^ (uint)_tokens[k]) * 16777619u;
        return (int)((h ^ (h >> 15)) & (uint)_mask);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_tokens.Length == 0) return;
        ArrayPool<int>.Shared.Return(_tokens);
        ArrayPool<int>.Shared.Return(_prev);
        ArrayPool<int>.Shared.Return(_head);
        _tokens = _prev = _head = [];
    }
}
