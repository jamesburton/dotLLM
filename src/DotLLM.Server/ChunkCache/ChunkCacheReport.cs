using DotLLM.Server.Models;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Builds the response <c>chunk_cache</c> object (spec §5.9; contract rulings 5–10) from a plan and the request's actual
/// exact prefix reuse.
/// </summary>
/// <remarks>
/// <para>
/// <b>Counts.</b> <c>annotated = prefix + hit + miss + ignored</c>: a verified chunk wholly inside the restored exact prefix
/// is <c>prefix</c>; otherwise it is <c>hit</c> or <c>miss</c> by the store lookup (ruling 5: a hit is counted under
/// <c>exact</c> too); everything else (dropped by the parser, altered, too short, unverified, unsupported model) is
/// <c>ignored</c>.
/// </para>
/// <para>
/// <b>Tokens.</b> v1 never applies reuse, so the actual partition is <c>prompt = prefix_reused + prefilled</c> with
/// <c>prefix_reused</c> = <c>cached_tokens</c> (exact rows only) and every chunk counter 0. Under <c>shadow</c>, <c>plan</c>
/// has its own partition (ruling 7): the hits outside the prefix become <c>chunk_reused</c>, of which the first
/// ⌈budget × n⌉ rows of each chunk are <c>chunk_recomputed</c> (the <see cref="PlanSelector"/> selector; budget = the
/// request's <c>recompute.budget</c>, else <see cref="DefaultPlanBudget"/>, ruling 50).
/// </para>
/// <para>
/// <b>Times.</b> <c>timings_ms.prefill</c> is the measured prefill. <c>full_prefill_estimate</c> is an estimate of the
/// unannotated request's prefill (ruling 48): its non-prefix rows at the request's own measured prefill rate (v1 applies no
/// chunk reuse, so that rate is the unannotated one and the estimate equals <c>prefill</c>), or at the backend's default
/// rate when nothing was measured. A future policy that applies reuse must not use its own (faster) measured rate here.
/// <c>predicted_ttft_ms</c> assembles the reused rows, recomputes the selected ones (cost per row × visible key) and
/// prefills the rest.
/// </para>
/// </remarks>
internal static class ChunkCacheReport
{
    /// <summary>Recompute selector the shadow plan assumes (the only one on every backend today).</summary>
    public const string PlanSelector = "chunk_starts";

    /// <summary>Recompute budget the shadow plan assumes when the request gives no <c>recompute.budget</c> (ruling 50).</summary>
    public const double DefaultPlanBudget = 0.05;

    /// <summary>Builds the report.</summary>
    /// <param name="plan">The pipeline's plan.</param>
    /// <param name="promptTokens">Prompt tokens of the response (<c>usage</c>).</param>
    /// <param name="cachedTokens">Exact rows the prompt cache restored (<c>cached_tokens</c>).</param>
    /// <param name="prefillMs">Measured prefill time.</param>
    /// <param name="cost">The backend's cost model.</param>
    /// <returns>The report.</returns>
    public static ChunkCacheReportDto Build(ChunkCachePlan plan, int promptTokens, int cachedTokens, double prefillMs, ChunkCostModel cost)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(cost);
        int prompt = Math.Max(0, promptTokens);
        int cached = Math.Clamp(cachedTokens, 0, prompt);

        int prefix = 0, hit = 0, miss = 0;
        long reused = 0, recomputed = 0;
        double recomputeMs = 0;
        foreach (var s in plan.Spans)
        {
            long end = (long)s.TokenStart + s.TokenCount;
            if (end <= cached)
            {
                prefix++;
                continue;
            }
            if (!s.Hit)
            {
                miss++;
                continue;
            }
            hit++;
            // A hit that starts inside the restored prefix is reused only beyond it.
            long first = Math.Max(s.TokenStart, cached);
            long rows = end - first;
            long selected = Math.Min(rows, (long)Math.Ceiling(rows * plan.Budget));
            reused += rows;
            recomputed += selected;
            recomputeMs += cost.RecomputeMs(first, selected);
        }
        int ignored = Math.Max(0, plan.Annotated - prefix - hit - miss);

        int prefilled = prompt - cached;
        double rate = prefilled > 0 && prefillMs > 0 && double.IsFinite(prefillMs)
            ? prefilled * 1000.0 / prefillMs
            : cost.PrefillTokensPerSec;
        long fullEstimate = Ms(ChunkCostModel.PrefillMs(prefilled, rate));

        ChunkCachePlanDto? planDto = null;
        if (plan.Shadow)
        {
            int chunkReused = (int)Math.Min(reused, prefilled);
            double predicted = cost.AssemblyMs(chunkReused) + recomputeMs + ChunkCostModel.PrefillMs(prefilled - chunkReused, rate);
            planDto = new ChunkCachePlanDto
            {
                Selector = PlanSelector,
                Budget = plan.Budget,
                Tokens = new ChunkCacheTokensDto
                {
                    Prompt = prompt,
                    PrefixReused = cached,
                    PrefixReusedApprox = 0,
                    ChunkReused = chunkReused,
                    ChunkRecomputed = (int)Math.Min(recomputed, chunkReused),
                    Prefilled = prefilled - chunkReused,
                    EncodeQueued = 0,
                },
                PredictedTtftMs = Ms(predicted),
                FullPrefillEstimateMs = fullEstimate,
            };
        }

        return new ChunkCacheReportDto
        {
            V = ChunkCacheWire.Version,
            PolicyRequested = PolicyName(plan.Requested),
            PolicyApplied = PolicyName(ChunkPolicy.Exact),
            Reason = plan.Reason,
            Chunks = new ChunkCacheChunksDto { Annotated = plan.Annotated, Prefix = prefix, Hit = hit, Miss = miss, Ignored = ignored },
            Tokens = new ChunkCacheTokensDto
            {
                Prompt = prompt,
                PrefixReused = cached,
                PrefixReusedApprox = 0,
                ChunkReused = 0,
                ChunkRecomputed = 0,
                Prefilled = prefilled,
                EncodeQueued = 0,
            },
            Plan = planDto,
            TimingsMs = new ChunkCacheTimingsDto
            {
                Assembly = 0,
                Probe = 0,
                Recompute = 0,
                Prefill = Ms(prefillMs),
                FullPrefillEstimate = fullEstimate,
            },
            Warnings = plan.Warnings.Select(w => new ChunkCacheWarningDto { Code = w.Code, Count = w.Count }).ToArray(),
        };
    }

    /// <summary>The wire name of a policy.</summary>
    public static string PolicyName(ChunkPolicy policy) => policy switch
    {
        ChunkPolicy.Exact => "exact",
        ChunkPolicy.Shadow => "shadow",
        ChunkPolicy.Balanced => "balanced",
        ChunkPolicy.Fast => "fast",
        ChunkPolicy.Auto => "auto",
        _ => "exact",
    };

    private static long Ms(double ms) => double.IsFinite(ms) && ms > 0 ? (long)Math.Round(ms) : 0;
}
