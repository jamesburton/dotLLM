namespace DotLLM.Core.Moe;

/// <summary>
/// Decides which experts computed off the device (cold misses, #1071) are copied into the device <see cref="ExpertCache"/>.
/// Copy-on-miss (#1070, <see cref="MoeColdMode.Copy"/>) does not use an admission: every miss is admitted by the cache's own
/// replacement policy.
/// </summary>
public interface IExpertAdmission
{
    /// <summary>
    /// Starts a forward of <paramref name="tokens"/> tokens: advances the admission's token clock and sets the byte budget the
    /// promotions of this forward may copy.
    /// </summary>
    /// <param name="tokens">Tokens of the forward.</param>
    /// <param name="copyBudgetBytes">Bytes the promotions of this forward may copy in total.</param>
    void BeginForward(int tokens, long copyBudgetBytes);

    /// <summary>
    /// Records one layer call's routing (resident and cold experts alike), after its <see cref="ExpertCache"/> lookup and before
    /// any <see cref="TryPromote"/> of the same call.
    /// </summary>
    /// <param name="layer">Layer index.</param>
    /// <param name="experts">Distinct experts the call routed to.</param>
    /// <param name="tokens">Tokens routed to each expert (same order).</param>
    void Observe(int layer, ReadOnlySpan<int> experts, ReadOnlySpan<int> tokens);

    /// <summary>
    /// Offers a cold miss of the current call for admission. When admitted, the expert is placed in the cache (via
    /// <see cref="ExpertCache.Promote"/>) and its slot is returned; the caller copies its weights in.
    /// </summary>
    /// <param name="layer">Layer of the current call.</param>
    /// <param name="expert">The cold expert.</param>
    /// <param name="bytes">Bytes the copy moves.</param>
    /// <param name="evicted">True when the promotion displaced a resident expert.</param>
    /// <returns>The slot to copy into, or -1 when the expert stays cold.</returns>
    int TryPromote(int layer, int expert, long bytes, out bool evicted);
}

/// <summary>
/// Promote-only admission with a decayed per-expert score (design #1063 section 4, "window64"): every routed token adds 1 to its
/// expert's score, and at each <see cref="WindowTokens"/>-token boundary every score is multiplied by <see cref="Decay"/>. A cold
/// expert is copied into the cache only when a slot of its pool is free, or its score beats the weakest unpinned resident's,
/// and the forward's copy budget still covers it. That copies 3.6–28× fewer experts than LRU for 5–7 points of hit rate, which
/// is the right trade when the misses can be computed where their weights already live.
/// </summary>
public sealed class WindowedPromotionAdmission : IExpertAdmission
{
    private readonly ExpertCache _cache;
    private readonly double[] _score;   // per key, as of _epochOf[key]
    private readonly long[] _epochOf;
    private long _tokens;
    private long _budget;

    /// <summary>Creates the admission over <paramref name="cache"/>.</summary>
    /// <param name="cache">The device cache it promotes into.</param>
    /// <param name="windowTokens">Tokens per decay window (64 in the design).</param>
    /// <param name="decay">Score multiplier per elapsed window (0.5 in the design).</param>
    /// <exception cref="ArgumentOutOfRangeException">The window is not positive or the decay is outside (0, 1].</exception>
    public WindowedPromotionAdmission(ExpertCache cache, int windowTokens = 64, double decay = 0.5)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfLessThan(windowTokens, 1);
        if (decay is <= 0 or > 1) throw new ArgumentOutOfRangeException(nameof(decay), "must be in (0, 1].");
        _cache = cache;
        WindowTokens = windowTokens;
        Decay = decay;
        _score = new double[cache.KeyCount];
        _epochOf = new long[cache.KeyCount];
    }

    /// <summary>Tokens per decay window.</summary>
    public int WindowTokens { get; }

    /// <summary>Score multiplier per elapsed window.</summary>
    public double Decay { get; }

    /// <summary>Bytes the current forward may still copy.</summary>
    public long RemainingBudgetBytes => _budget;

    /// <summary>Tokens seen so far (the clock of the decay windows).</summary>
    public long Tokens => _tokens;

    /// <inheritdoc/>
    public void BeginForward(int tokens, long copyBudgetBytes)
    {
        _tokens += Math.Max(tokens, 0);
        _budget = Math.Max(copyBudgetBytes, 0);
    }

    /// <inheritdoc/>
    public void Observe(int layer, ReadOnlySpan<int> experts, ReadOnlySpan<int> tokens)
    {
        if (tokens.Length < experts.Length) throw new ArgumentException("tokens is shorter than experts.", nameof(tokens));
        for (int i = 0; i < experts.Length; i++)
        {
            int key = _cache.KeyOf(layer, experts[i]);
            _score[key] = ScoreOf(key) + tokens[i];
            _epochOf[key] = Epoch;
        }
    }

    /// <summary>The decayed score of <paramref name="expert"/> of <paramref name="layer"/> now.</summary>
    /// <param name="layer">Layer index.</param>
    /// <param name="expert">Expert index.</param>
    /// <returns>The score.</returns>
    public double Score(int layer, int expert) => ScoreOf(_cache.KeyOf(layer, expert));

    /// <inheritdoc/>
    public int TryPromote(int layer, int expert, long bytes, out bool evicted)
    {
        evicted = false;
        if (bytes > _budget) return -1;
        int pool = _cache.PoolOf(layer);
        if (pool < 0) return -1;

        // The weakest unpinned slot of the pool: a free slot wins outright, else the lowest decayed score (ties to the lower slot).
        int victim = -1;
        double weakest = double.PositiveInfinity;
        int slots = _cache.SlotsIn(pool);
        for (int s = 0; s < slots; s++)
        {
            if (_cache.IsPinned(pool, s)) continue;
            var (l, e) = _cache.ExpertAt(pool, s);
            if (l < 0) { victim = s; weakest = double.NegativeInfinity; break; }
            double sc = ScoreOf(_cache.KeyOf(l, e));
            if (sc < weakest) { weakest = sc; victim = s; }
        }
        if (victim < 0) return -1;
        if (!(ScoreOf(_cache.KeyOf(layer, expert)) > weakest)) return -1;

        _cache.Promote(layer, expert, victim, out evicted);
        _budget -= bytes;
        return victim;
    }

    private long Epoch => _tokens / WindowTokens;

    private double ScoreOf(int key)
    {
        long elapsed = Epoch - _epochOf[key];
        double s = _score[key];
        return elapsed <= 0 || s == 0 ? s : s * Math.Pow(Decay, elapsed);
    }
}
