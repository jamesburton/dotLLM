namespace DotLLM.Engine;

/// <summary>
/// Online cost/acceptance model of one model's MTP rounds (issue #900): per-position draft acceptance and the wall time of a round as a
/// function of its draft length K. From those it predicts the throughput (generated tokens per millisecond) of every K and picks the best,
/// and says whether even the best K is slower than plain decode.
/// </summary>
/// <remarks>
/// <para><b>Why K is a lever.</b> A round drafts K tokens (K sequential head steps), verifies K+1 rows in one trunk forward and commits
/// <c>1 + accepted</c> tokens. On a mixture-of-experts trunk every verify row reads its own experts, so round time grows roughly linearly with K
/// while the expected tokens per round flatten (acceptance multiplies down the chain): the K that is right for a bandwidth-bound 60 GiB file is
/// too long for a 37 GiB one, where measured K=4 rounds lost to plain decode at every acceptance seen. The model therefore measures instead of
/// assuming: acceptance per draft position (EMA of P(position i accepted | positions below accepted)) and round milliseconds per K (EMA).</para>
/// <para><b>Unmeasured K.</b> Interpolated/extrapolated linearly from the nearest measured Ks; with a single measured K a proportional guess
/// (<c>K + 1 + <see cref="FixedRows"/></c> rows) is used. The guess is only ever a reason to try a K once: the round that follows replaces it with a
/// measurement (see <see cref="MtpRoundController"/>, which also re-measures neighbouring Ks periodically).</para>
/// <para>Thread-safe; one instance lives in the <see cref="MtpAdaptiveGate"/> of a generator and so persists across requests (acceptance moves with
/// content, hence the decay).</para>
/// </remarks>
public sealed class MtpCostModel
{
    /// <summary>Largest draft length the model tracks.</summary>
    public const int MaxK = 8;

    /// <summary>EMA weight of a new observation.</summary>
    private const double Alpha = 0.25;

    /// <summary>Smallest EMA weight of an acceptance observation (about the last 30 rounds count).</summary>
    private const double AcceptanceAlphaFloor = 0.03;

    /// <summary>Acceptance assumed for a draft position that has never been observed (the K=4 mean seen on the released head is ~0.7).</summary>
    public const double PriorAcceptance = 0.7;

    /// <summary>Row-equivalents of fixed per-round cost (head absorb, state restore, argmax readbacks) assumed by the one-point extrapolation.</summary>
    public const double FixedRows = 1.5;

    private readonly object _lock = new();
    private readonly double[] _acc = new double[MaxK + 1];
    private readonly int[] _accN = new int[MaxK + 1];
    private readonly double[] _cost = new double[MaxK + 1];
    private readonly int[] _costN = new int[MaxK + 1];
    private bool _firstRoundSeen;

    /// <summary>Forgets every observation (see <see cref="MtpAdaptiveGate.Reset"/>).</summary>
    public void Reset()
    {
        lock (_lock)
        {
            Array.Clear(_acc); Array.Clear(_accN); Array.Clear(_cost); Array.Clear(_costN);
            _firstRoundSeen = false;
        }
    }

    /// <summary>Records one finished round: draft length <paramref name="k"/>, <paramref name="accepted"/> drafts accepted (0..k), wall milliseconds.</summary>
    public void RecordRound(int k, int accepted, double roundMs)
    {
        if (k < 1 || k > MaxK || roundMs <= 0) return;
        accepted = Math.Clamp(accepted, 0, k);
        lock (_lock)
        {
            // The very first round of a model pays for cold pages / clock ramp (1.3-2x slow on this box): keep its acceptance, drop its time.
            bool coldRound = !_firstRoundSeen;
            _firstRoundSeen = true;
            if (!coldRound)
            {
                _cost[k] = _costN[k] == 0 ? roundMs : _cost[k] + Alpha * (roundMs - _cost[k]);
                _costN[k]++;
            }
            // Position i was "reached" when positions 1..i-1 were all accepted, i.e. accepted >= i-1; it was accepted when accepted >= i.
            for (int i = 1; i <= k && accepted >= i - 1; i++)
            {
                double hit = accepted >= i ? 1.0 : 0.0;
                // Acceptance is a Bernoulli rate: a running mean first (1/n), then a slow EMA, so one lucky round does not move the best K.
                _accN[i]++;
                _acc[i] = _accN[i] == 1 ? hit : _acc[i] + Math.Max(1.0 / _accN[i], AcceptanceAlphaFloor) * (hit - _acc[i]);
            }
        }
    }

    /// <summary>Rounds recorded for draft length <paramref name="k"/>.</summary>
    public int RoundsAt(int k) { lock (_lock) return (uint)k <= MaxK ? _costN[k] : 0; }

    /// <summary>Total rounds recorded.</summary>
    public int TotalRounds { get { lock (_lock) { int n = 0; for (int k = 1; k <= MaxK; k++) n += _costN[k]; return n; } } }

    /// <summary>Expected tokens a round of draft length <paramref name="k"/> commits: 1 (the correction/bonus) plus the expected accepted drafts.</summary>
    public double ExpectedTokens(int k)
    {
        lock (_lock) return ExpectedTokensLocked(k);
    }

    private double ExpectedTokensLocked(int k)
    {
        double prior = PriorLocked();
        double chain = 1.0, tokens = 1.0;
        for (int i = 1; i <= Math.Min(k, MaxK); i++)
        {
            chain *= _accN[i] > 0 ? _acc[i] : prior;
            tokens += chain;
        }
        return tokens;
    }

    // A position never observed inherits the observed positions' mean, so a long K that has only run short rounds is not judged on a fixed prior.
    private double PriorLocked()
    {
        double sum = 0; int n = 0;
        for (int i = 1; i <= MaxK; i++) if (_accN[i] > 0) { sum += _acc[i]; n++; }
        return n > 0 ? sum / n : PriorAcceptance;
    }

    /// <summary>Predicted wall milliseconds of a round of draft length <paramref name="k"/>; 0 when nothing has been measured yet.</summary>
    public double PredictRoundMs(int k)
    {
        lock (_lock) return PredictRoundMsLocked(k);
    }

    private double PredictRoundMsLocked(int k)
    {
        if (_costN[k] > 0) return _cost[k];
        int lo = 0, hi = 0;
        for (int j = 1; j <= MaxK; j++)
        {
            if (_costN[j] == 0) continue;
            if (j < k) lo = j;                       // nearest measured below
            else if (j > k && hi == 0) hi = j;       // nearest measured above
        }
        if (lo > 0 && hi > 0)
            return _cost[lo] + (_cost[hi] - _cost[lo]) * (k - lo) / (hi - lo);
        if (lo > 0 || hi > 0)
        {
            int a = lo > 0 ? lo : hi;
            // A second measured point on the same side gives a slope; otherwise a proportional guess.
            int b = 0;
            for (int j = 1; j <= MaxK; j++) if (_costN[j] > 0 && j != a && (b == 0 || Math.Abs(j - a) < Math.Abs(b - a))) b = j;
            if (b > 0)
                return Math.Max(_cost[a] + (_cost[b] - _cost[a]) * (k - a) / (b - a), _cost[a] * 0.25);
            return _cost[a] * (k + 1 + FixedRows) / (a + 1 + FixedRows);
        }
        return 0;
    }

    /// <summary>
    /// The draft length in 1..<paramref name="maxK"/> with the lowest predicted milliseconds per generated token, and that figure.
    /// Returns <c>(maxK, 0)</c> while nothing has been measured.
    /// </summary>
    public (int K, double MsPerToken) BestK(int maxK)
    {
        maxK = Math.Clamp(maxK, 1, MaxK);
        lock (_lock)
        {
            int best = maxK; double bestMs = 0;
            for (int k = 1; k <= maxK; k++)
            {
                double c = PredictRoundMsLocked(k);
                if (c <= 0) continue;
                double perTok = c / ExpectedTokensLocked(k);
                if (bestMs == 0 || perTok < bestMs) { bestMs = perTok; best = k; }
            }
            return (best, bestMs);
        }
    }

    /// <summary>Predicted ms per generated token at draft length <paramref name="k"/>; 0 when unknown.</summary>
    public double MsPerToken(int k)
    {
        lock (_lock)
        {
            double c = PredictRoundMsLocked(Math.Clamp(k, 1, MaxK));
            return c <= 0 ? 0 : c / ExpectedTokensLocked(k);
        }
    }

    /// <summary>Observed acceptance of draft position <paramref name="i"/> (1-based), or the prior when never observed.</summary>
    public double Acceptance(int i)
    {
        lock (_lock) return (uint)(i - 1) < MaxK && _accN[i] > 0 ? _acc[i] : PriorLocked();
    }
}

/// <summary>
/// Per-request MTP round policy on top of an <see cref="MtpCostModel"/>: warms up at the configured K, then follows the model's best K, re-measures
/// a neighbouring K now and then so the model cannot go stale, and says when the request should drop to plain decode because MTP is measurably slower.
/// </summary>
public sealed class MtpRoundController
{
    /// <summary>Rounds at the configured K before the model is trusted (the first round also pays for clock ramp / cold pages).</summary>
    public const int WarmupRounds = 3;

    /// <summary>Every this many rounds one round runs a neighbouring K to refresh its cost estimate.</summary>
    public const int ExploreEvery = 16;

    /// <summary>Rounds before a fall-back to plain decode is considered.</summary>
    public const int MinRoundsForFallback = 5;

    /// <summary>MTP must be this much slower per token than plain before the request drops it (hysteresis against noise).</summary>
    public const double FallbackMargin = 0.04;

    /// <summary>
    /// Zero-draft calibration rounds a request runs when its gate has no plain-decode sample yet (#906). The first <see cref="CalibrationColdRounds"/> are cold
    /// (clock ramp, first-use pipelines, cold n-gram pages: 140-260 ms against 45 ms warm on UD-IQ4_XS in a fresh process) and are not counted; the median of
    /// the rest is the request's plain reference and the gate's first plain sample.
    /// </summary>
    public const int CalibrationRounds = 8;

    /// <summary>Leading calibration rounds that are discarded as cold.</summary>
    public const int CalibrationColdRounds = 2;

    /// <summary>
    /// A request calibrates only when it is long enough for the calibration to matter: the rounds run at plain speed, so a request must have enough tokens
    /// left for a win to repay them, and a short one could not amortise a speculative start anyway.
    /// </summary>
    public const int MinTokensToCalibrate = 48;

    private readonly MtpCostModel _model;
    private readonly int _configuredK;
    private int _round;
    private int _roundsThisRequest;
    private int _calibrationLeft;
    private int _calibrationSeen;
    private readonly List<double> _calibrationSamples = new();
    private readonly MtpAdaptiveGate? _gate;

    /// <param name="model">Shared cost model.</param>
    /// <param name="configuredK">The K the caller asked for; also the upper bound.</param>
    /// <param name="calibratePlain">Start with <see cref="CalibrationRounds"/> zero-draft rounds that measure plain decode inside this request. Set when the gate has
    /// no plain sample: without a reference a losing configuration cannot be detected, and the first request would run entirely on MTP.</param>
    /// <param name="gate">The gate that receives the calibrated plain reference as its first plain sample (null in unit tests of the controller alone).</param>
    public MtpRoundController(MtpCostModel model, int configuredK, bool calibratePlain = false, MtpAdaptiveGate? gate = null)
    {
        _gate = gate;
        _model = model;
        _configuredK = Math.Clamp(configuredK, 1, MtpCostModel.MaxK);
        _calibrationLeft = calibratePlain ? CalibrationRounds : 0;
    }

    /// <summary>True while zero-draft calibration rounds remain.</summary>
    public bool Calibrating => _calibrationLeft > 0;

    /// <summary>Plain-decode milliseconds per token measured by this request's calibration rounds (median of the counted ones); 0 until one counted round ran.</summary>
    public double CalibratedPlainMsPerToken
    {
        get
        {
            if (_calibrationSamples.Count == 0) return 0;
            var sorted = _calibrationSamples.OrderBy(x => x).ToList();
            return sorted[sorted.Count / 2];
        }
    }

    /// <summary>
    /// Draft length for the next round, never above <paramref name="remainingTokens"/> or the configured K. Returns 0 for a calibration round (no draft,
    /// one committed token; see <see cref="MtpSpeculativeDecoder"/>).
    /// </summary>
    public int NextK(int remainingTokens)
    {
        if (_calibrationLeft > 0)
            return 0;
        int cap = Math.Min(_configuredK, Math.Max(remainingTokens, 1));
        int k;
        if (_roundsThisRequest < WarmupRounds && _model.RoundsAt(_configuredK) < WarmupRounds)
            k = cap;
        else
        {
            k = _model.BestK(_configuredK).K;
            if (_round > 0 && _round % ExploreEvery == 0)
            {
                // Refresh the less-measured neighbour of the current best.
                int up = Math.Min(k + 1, _configuredK), down = Math.Max(k - 1, 1);
                k = _model.RoundsAt(up) <= _model.RoundsAt(down) ? up : down;
            }
        }
        _round++;
        return Math.Clamp(k, 1, cap);
    }

    /// <summary>Feeds back a finished round (<paramref name="emitted"/> = tokens committed, i.e. accepted drafts + 1). <paramref name="k"/> 0 is a calibration round.</summary>
    public void Record(int k, int emitted, double roundMs)
    {
        if (k == 0)
        {
            if (_calibrationLeft > 0) _calibrationLeft--;
            bool cold = _calibrationSeen++ < CalibrationColdRounds;
            if (MtpAdaptiveGate.TraceEnabled)
                Console.Error.WriteLine($"[mtp] calibration round {_calibrationSeen}/{CalibrationRounds}: {roundMs:F1} ms{(cold ? " (cold, not counted)" : "")}");
            if (!cold && roundMs > 0 && emitted > 0) _calibrationSamples.Add(roundMs / emitted);
            if (_calibrationLeft == 0 && _calibrationSamples.Count > 0)
            {
                _gate?.AdoptPlainReference(CalibratedPlainMsPerToken);
                if (MtpAdaptiveGate.TraceEnabled)
                    Console.Error.WriteLine($"[mtp] calibrated plain reference: {CalibratedPlainMsPerToken:F1} ms/token ({_calibrationSamples.Count} counted rounds)");
            }
            return;
        }
        _roundsThisRequest++;
        _model.RecordRound(k, emitted - 1, roundMs);
    }

    /// <summary>
    /// True when the model's best K is slower per generated token than plain decode (<paramref name="plainMsPerToken"/> &gt; 0, else this request's own
    /// calibration) by more than <see cref="FallbackMargin"/>, after enough rounds in this request to trust it.
    /// </summary>
    public bool ShouldFallBack(double plainMsPerToken)
    {
        if (plainMsPerToken <= 0) plainMsPerToken = CalibratedPlainMsPerToken;
        if (plainMsPerToken <= 0 || _roundsThisRequest < MinRoundsForFallback) return false;
        var (_, perTok) = _model.BestK(_configuredK);
        return perTok > 0 && perTok > plainMsPerToken * (1.0 + FallbackMargin);
    }
}
