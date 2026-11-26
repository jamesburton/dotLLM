namespace DotLLM.Engine;

/// <summary>
/// Per-generator, per-model decision of whether MTP self-speculation is actually faster than plain decode.
/// </summary>
/// <remarks>
/// <para><b>Why.</b> MTP is a win on bandwidth-bound giants (Bonsai 27B, 1.8-2.15x, #469) and a clear loss on small
/// models: on Tev1-4B Q4_K_M (Vulkan, Strix Halo) a draft-and-verify round costs ~105 ms against ~24 ms for a plain
/// decode forward, so even at 73% draft acceptance generation ran at 18-20 tok/s against 35-37 tok/s with MTP off, at
/// every length from 16 to 160 tokens. A fixed default cannot be right for both, and the break-even depends on backend,
/// quantization and model size, so the generator measures it instead.</para>
/// <para><b>Policy.</b> Requests shorter than <see cref="MinMaxTokens"/> never use MTP (it cannot amortise its prefill
/// absorb plus a draft round; a 1-token classifier answer paid 110 ms for a 25 ms job). Otherwise: try MTP once, try plain
/// once, then use whichever measured fewer ms per generated token, re-probing the other arm every
/// <see cref="ReprobeEvery"/> requests so a changed load or thermal state cannot pin a stale choice. A sample only counts
/// when the request generated at least <see cref="MinSampleTokens"/> decode tokens.</para>
/// <para>Opt-in (<c>mtpAdaptive</c> on <see cref="TextGenerator"/>): library callers that asked for MTP explicitly keep
/// exactly that. <c>DOTLLM_MTP_ADAPTIVE=0</c> disables it process-wide.</para>
/// </remarks>
public sealed class MtpAdaptiveGate
{
    /// <summary>Requests with a smaller <c>max_tokens</c> never engage MTP.</summary>
    public const int MinMaxTokens = 12;

    /// <summary>Minimum decode-phase tokens for a request to contribute a speed sample.</summary>
    public const int MinSampleTokens = 12;

    /// <summary>Re-measure the currently losing arm after this many decisions.</summary>
    public const int ReprobeEvery = 16;

    private const double Alpha = 0.3;

    /// <summary>Environment switch: <c>0</c> turns the adaptive policy off (MTP engages whenever enabled).</summary>
    public const string DisableEnvVar = "DOTLLM_MTP_ADAPTIVE";

    /// <summary>
    /// Resident model size (ComputeMemoryBytes, weights plus scratch) from which MTP is tried first: bandwidth-bound large models win (Bonsai 27B, ~7+ GiB, 1.8-2.15x), small
    /// ones lose (Tev1-4B Q4_K_M, 4.95 GiB resident on Vulkan, 0.5x). A prior only: measured speed replaces it after one sample per arm.
    /// </summary>
    public const long MtpPriorMinModelBytes = 6L << 30;

    /// <summary>Longest wait between re-probes of a clearly losing arm (decisions).</summary>
    public const int MaxReprobeEvery = 640;

    /// <summary>An arm slower than the winner by more than this ratio is "clearly losing": its re-probe interval doubles (a probe of a 0.8x arm costs a fifth of one request).</summary>
    public const double ClearLossRatio = 1.10;

    /// <summary>Prompt lengths up to this many tokens use the gate itself (band 0); longer prompts get their own band per doubling (see <see cref="ForContext"/>).</summary>
    public const int ShortContextTokens = 512;

    /// <summary>Number of context bands (band 0 plus one per doubling of the prompt length above <see cref="ShortContextTokens"/>, the last one open-ended).</summary>
    public const int ContextBands = 7;

    private readonly bool _tryMtpFirst;
    private readonly long _modelBytes;
    private readonly MtpAdaptiveGate?[] _bands = new MtpAdaptiveGate?[ContextBands];
    private readonly object _lock = new();
    private int _reprobeEvery = ReprobeEvery;

    /// <summary>
    /// Online model of this model's MTP rounds (acceptance per draft position, round cost per K). Shared by every request of the generator, so
    /// the draft length and the mid-request fall-back (<see cref="MtpRoundController"/>) learn once and keep learning.
    /// </summary>
    public MtpCostModel CostModel { get; } = new();

    /// <param name="modelBytes">Resident model size in bytes; selects which arm is explored first.</param>
    public MtpAdaptiveGate(long modelBytes = long.MaxValue)
    {
        _modelBytes = modelBytes;
        _tryMtpFirst = modelBytes >= MtpPriorMinModelBytes;
    }

    /// <summary>Band index of a prompt of <paramref name="contextTokens"/> tokens: 0 up to <see cref="ShortContextTokens"/>, then one per doubling.</summary>
    public static int BandOf(int contextTokens)
    {
        if (contextTokens <= ShortContextTokens) return 0;
        int band = 1;
        for (long limit = ShortContextTokens * 2L; contextTokens > limit && band < ContextBands - 1; limit *= 2) band++;
        return band;
    }

    /// <summary>
    /// The gate that owns the measurements for a prompt of <paramref name="contextTokens"/> tokens. MTP's break-even moves with context (Edge0-145B on
    /// Vulkan: 1.6-1.8x plain at 50-100 tokens, slower than plain at 2.4K because acceptance falls and verify rows cost more), so one verdict per model
    /// would be wrong at one end or the other. Band 0 (up to <see cref="ShortContextTokens"/> tokens) is this gate; every longer band is a lazily created
    /// sibling with its own arms, re-probe schedule and <see cref="CostModel"/>, so a loss measured at 2.4K is remembered at 2.4K and never charged to
    /// short prompts (and vice versa).
    /// </summary>
    public MtpAdaptiveGate ForContext(int contextTokens)
    {
        int band = BandOf(contextTokens);
        if (band == 0) return this;
        lock (_lock) return _bands[band] ??= new MtpAdaptiveGate(_modelBytes);
    }
    private double _plainMsPerToken, _mtpMsPerToken;
    private int _plainSamples, _mtpSamples;
    private int _sinceReprobe;

    /// <summary>
    /// Forgets every measurement (arms, re-probe schedule, cost model) and starts again from the size prior. The server calls this after its warm-up:
    /// warm-up requests are tiny and run cold (the first MTP request of a process measured 2x slower than steady state), and one such sample made
    /// the gate settle on plain decode for the first dozens of real requests.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _plainMsPerToken = _mtpMsPerToken = 0;
            _plainSamples = _mtpSamples = 0;
            _sinceReprobe = 0;
            _reprobeEvery = ReprobeEvery;
            foreach (var b in _bands) b?.Reset();
        }
        CostModel.Reset();
    }

    /// <summary>True unless <see cref="DisableEnvVar"/> is <c>0</c>.</summary>
    public static bool EnabledByEnvironment => Environment.GetEnvironmentVariable(DisableEnvVar) != "0";

    /// <summary>EMA of decode ms per token for plain decode; 0 until sampled.</summary>
    public double PlainMsPerToken { get { lock (_lock) return _plainMsPerToken; } }

    /// <summary>EMA of decode ms per token for MTP rounds; 0 until sampled.</summary>
    public double MtpMsPerToken { get { lock (_lock) return _mtpMsPerToken; } }

    /// <summary>Trace switch: <c>DOTLLM_MTP_TRACE=1</c> prints every gate decision and sample to stderr (diagnosis of "why did request N run plain?").</summary>
    public static bool TraceEnabled { get; } = Environment.GetEnvironmentVariable("DOTLLM_MTP_TRACE") == "1";

    /// <summary>One-line state of both arms, for the trace.</summary>
    public string Describe()
    {
        lock (_lock)
            return $"plain {_plainMsPerToken:F1} ms/tok (n={_plainSamples}), mtp {_mtpMsPerToken:F1} ms/tok (n={_mtpSamples}), since-reprobe {_sinceReprobe}/{_reprobeEvery}";
    }

    /// <summary>
    /// Forecasts, without counting a decision, whether the next request is likely to decode with MTP: MTP is unexplored or
    /// plain is unmeasured, MTP is the measured winner, or the losing arm's re-probe falls on the next decision. A plain
    /// request that takes the recurrent prefix snapshot reads this to decide whether to feed the MTP head through its
    /// prefill. The feed costs time to first token on every such turn, and it only pays when a later hit drafts from the
    /// head (#1088 review).
    /// </summary>
    /// <returns><see langword="true"/> when the next request is likely to use MTP.</returns>
    public bool LikelyUsesMtpNext()
    {
        lock (_lock)
        {
            if (_mtpSamples == 0 || _plainSamples == 0) return true;   // an arm still to explore: MTP may well be tried next
            return _mtpMsPerToken < _plainMsPerToken || _sinceReprobe + 1 >= _reprobeEvery;
        }
    }

    /// <summary>Decides whether the next request (with the given <c>max_tokens</c>) should use MTP.</summary>
    public bool ShouldUseMtp(int maxTokens)
    {
        if (maxTokens < MinMaxTokens)
            return false;

        lock (_lock)
        {
            if (_mtpSamples == 0 && _plainSamples == 0) return _tryMtpFirst;  // size prior picks the first arm
            if (_mtpSamples == 0) return true;      // explore the other arm once
            if (_plainSamples == 0) return false;

            bool mtpWins = _mtpMsPerToken < _plainMsPerToken;
            if (++_sinceReprobe >= _reprobeEvery)
            {
                _sinceReprobe = 0;
                // Re-measure the losing arm; back off when it loses by a clear margin so a hopeless arm costs ~nothing in the long run.
                double ratio = mtpWins ? _plainMsPerToken / _mtpMsPerToken : _mtpMsPerToken / _plainMsPerToken;
                _reprobeEvery = ratio > ClearLossRatio ? Math.Min(_reprobeEvery * 2, MaxReprobeEvery) : ReprobeEvery;
                return !mtpWins;
            }
            return mtpWins;
        }
    }

    /// <summary>
    /// True when a request of up to <paramref name="remainingTokens"/> decode tokens should open with zero-draft calibration rounds (see
    /// <see cref="MtpRoundController.CalibrationRounds"/>): there is no plain-decode sample yet, so neither this request nor the gate could notice a
    /// speculative configuration that loses. Costs about a dozen plain steps once; <c>DOTLLM_MTP_CALIBRATE=0</c> turns it off.
    /// </summary>
    public bool NeedsPlainReference(int remainingTokens)
    {
        if (remainingTokens < MtpRoundController.MinTokensToCalibrate || !CalibrationEnabled) return false;
        lock (_lock) return _plainSamples == 0;
    }

    /// <summary>
    /// Takes a request's own calibrated plain-decode figure as the first plain sample when the gate has none (the calibration rounds are too few for
    /// <see cref="RecordSegment"/>'s minimum). Later real plain samples replace it through the usual EMA.
    /// </summary>
    internal void AdoptPlainReference(double msPerToken)
    {
        if (msPerToken <= 0) return;
        lock (_lock)
        {
            if (_plainSamples != 0) return;
            _plainMsPerToken = msPerToken;
            _plainSamples = 1;
        }
    }

    /// <summary>
    /// Records the measured loss of a request that dropped MTP mid-way (<see cref="MtpRoundController.ShouldFallBack"/>). The fall-back fires after
    /// <see cref="MtpRoundController.MinRoundsForFallback"/> rounds, which commit about 9 tokens: fewer than <see cref="MinSampleTokens"/>, so
    /// <see cref="RecordSegment"/> discarded that segment and the MTP arm stayed unmeasured. Every request then retried MTP, fell back again and paid
    /// the rounds again, forever (found on Edge0, #2018). The figure the controller decided on (the cost model's best-K ms per generated token, which
    /// excludes the cold first round) is the MTP sample instead.
    /// </summary>
    /// <param name="msPerToken">The cost model's best predicted ms per generated token at the time of the fall-back.</param>
    internal void RecordFallback(double msPerToken) => RecordSample(usedMtp: true, msPerToken);

    /// <summary>True unless <c>DOTLLM_MTP_CALIBRATE</c> is <c>0</c>.</summary>
    public static bool CalibrationEnabled { get; } = Environment.GetEnvironmentVariable("DOTLLM_MTP_CALIBRATE") != "0";

    /// <summary>Feeds back a finished request: whether MTP was used, tokens generated, and decode-phase time.</summary>
    public void Record(bool usedMtp, int generatedTokens, double decodeMs)
        => RecordSegment(usedMtp, generatedTokens - 1, decodeMs); // the first token comes from prefill

    /// <summary>
    /// Feeds back a stretch of decode (<paramref name="decodeTokens"/> tokens in <paramref name="decodeMs"/>) done with one arm. A request that
    /// fell back from MTP to plain decode mid-way records its two stretches separately.
    /// </summary>
    public void RecordSegment(bool usedMtp, int decodeTokens, double decodeMs)
    {
        if (decodeTokens < MinSampleTokens || decodeMs <= 0)
            return;

        RecordSample(usedMtp, decodeMs / decodeTokens);
    }

    private void RecordSample(bool usedMtp, double perToken)
    {
        if (perToken <= 0) return;
        lock (_lock)
        {
            if (usedMtp)
            {
                _mtpMsPerToken = _mtpSamples == 0 ? perToken : _mtpMsPerToken + Alpha * (perToken - _mtpMsPerToken);
                _mtpSamples++;
            }
            else
            {
                _plainMsPerToken = _plainSamples == 0 ? perToken : _plainMsPerToken + Alpha * (perToken - _plainMsPerToken);
                _plainSamples++;
            }
        }
    }
}
