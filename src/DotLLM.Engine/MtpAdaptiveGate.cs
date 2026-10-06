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
    public const int MinSampleTokens = 6;

    /// <summary>Re-measure the currently losing arm after this many decisions.</summary>
    public const int ReprobeEvery = 40;

    private const double Alpha = 0.3;

    /// <summary>Environment switch: <c>0</c> turns the adaptive policy off (MTP engages whenever enabled).</summary>
    public const string DisableEnvVar = "DOTLLM_MTP_ADAPTIVE";

    /// <summary>
    /// Resident model size (ComputeMemoryBytes, weights plus scratch) from which MTP is tried first: bandwidth-bound large models win (Bonsai 27B, ~7+ GiB, 1.8-2.15x), small
    /// ones lose (Tev1-4B Q4_K_M, 4.95 GiB resident on Vulkan, 0.5x). A prior only: measured speed replaces it after one sample per arm.
    /// </summary>
    public const long MtpPriorMinModelBytes = 6L << 30;

    private readonly bool _tryMtpFirst;
    private readonly object _lock = new();

    /// <param name="modelBytes">Resident model size in bytes; selects which arm is explored first.</param>
    public MtpAdaptiveGate(long modelBytes = long.MaxValue) => _tryMtpFirst = modelBytes >= MtpPriorMinModelBytes;
    private double _plainMsPerToken, _mtpMsPerToken;
    private int _plainSamples, _mtpSamples;
    private int _sinceReprobe;

    /// <summary>True unless <see cref="DisableEnvVar"/> is <c>0</c>.</summary>
    public static bool EnabledByEnvironment => Environment.GetEnvironmentVariable(DisableEnvVar) != "0";

    /// <summary>EMA of decode ms per token for plain decode; 0 until sampled.</summary>
    public double PlainMsPerToken { get { lock (_lock) return _plainMsPerToken; } }

    /// <summary>EMA of decode ms per token for MTP rounds; 0 until sampled.</summary>
    public double MtpMsPerToken { get { lock (_lock) return _mtpMsPerToken; } }

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
            if (++_sinceReprobe >= ReprobeEvery)
            {
                _sinceReprobe = 0;
                return !mtpWins;
            }
            return mtpWins;
        }
    }

    /// <summary>Feeds back a finished request: whether MTP was used, tokens generated, and decode-phase time.</summary>
    public void Record(bool usedMtp, int generatedTokens, double decodeMs)
    {
        int decodeTokens = generatedTokens - 1; // the first token comes from prefill
        if (decodeTokens < MinSampleTokens || decodeMs <= 0)
            return;

        double perToken = decodeMs / decodeTokens;
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
