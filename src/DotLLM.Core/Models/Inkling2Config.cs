namespace DotLLM.Core.Models;

/// <summary>
/// Hyper-parameters of Thinking Machines Inkling 2 (GGUF <c>general.architecture</c> = <c>inkling2_loop</c> or <c>inkling2</c>)
/// on top of the generic <see cref="ModelConfig"/> fields. Layer-by-layer semantics and their llama.cpp sources: <c>docs/INKLING2.md</c>.
/// </summary>
/// <remarks>
/// A plain <c>inkling2</c> file (no loop) is the same graph with <see cref="LoopPasses"/> = 1; the CPU reference accepts it, but only the
/// looped Inkling-2-Mini has been validated against an oracle.
/// </remarks>
public sealed record Inkling2Config
{
    /// <summary>Per physical layer: true for a local (sliding-window) layer, false for a global one (<c>attention.sliding_window_pattern</c>).</summary>
    public required bool[] IsLocalLayer { get; init; }

    /// <summary>Sliding window of the local layers, in tokens (<c>attention.sliding_window</c>, 1024). A key at distance &gt;= this is masked.</summary>
    public required int SlidingWindow { get; init; }

    /// <summary>Width of the relative-position projection (<c>attention.relative_length</c>, 16).</summary>
    public required int RelativeLength { get; init; }

    /// <summary>Relative-position extent of the global layers (<c>attention.relative_extent</c>, 1024); distances beyond it get no bias.</summary>
    public required int RelativeExtent { get; init; }

    /// <summary>L2 clip threshold of Q and K (<c>attention.qk_clip</c>, 22.627417 = 16*sqrt(2)).</summary>
    public required float QkClip { get; init; }

    /// <summary>L2 clip threshold of the relative-position projection (<c>attention.relative_clip</c>, 8).</summary>
    public required float RelativeClip { get; init; }

    /// <summary>Log-scaling floor (<c>attention.log_scale_floor</c>, 8192).</summary>
    public required int LogScaleFloor { get; init; }

    /// <summary>Log-scaling alpha (<c>attention.log_scale_alpha</c>, 0.1).</summary>
    public required float LogScaleAlpha { get; init; }

    /// <summary>Taps of every short convolution (<c>shortconv.l_cache</c>, 4).</summary>
    public required int ConvKernel { get; init; }

    /// <summary>Layers per attention-residual block (<c>attn_res.block_size</c>, 6).</summary>
    public required int AttnResBlockSize { get; init; }

    /// <summary>Gate softcap beta of SITU (<c>activation.situ_beta</c>, 4).</summary>
    public required float SituBeta { get; init; }

    /// <summary>Up (linear) softcap beta of SITU (<c>activation.situ_linear_beta</c>, 25); 0 disables it.</summary>
    public required float SituLinearBeta { get; init; }

    /// <summary>Routed expert count (128).</summary>
    public required int ExpertCount { get; init; }

    /// <summary>Experts used per token (8).</summary>
    public required int ExpertUsedCount { get; init; }

    /// <summary>Per-expert intermediate width (1152).</summary>
    public required int ExpertFfLength { get; init; }

    /// <summary>Normalised expert weights are multiplied by this (<c>expert_weights_scale</c>, 8).</summary>
    public required float ExpertWeightsScale { get; init; }

    /// <summary>Final logit softcap (<c>final_logit_softcapping</c>, 30); 0 = none.</summary>
    public required float FinalLogitSoftcap { get; init; }

    /// <summary>Row count of the embedding and output matrices (201,024): padded past the 200,058-token vocabulary, which is <see cref="ModelConfig.VocabSize"/>.</summary>
    public required int PaddedVocabSize { get; init; }

    /// <summary>First layer of the looped core (<c>loop.start</c>, 6); 0 with <see cref="LoopPasses"/> = 1 when the file has no loop.</summary>
    public required int LoopStart { get; init; }

    /// <summary>One past the last layer of the looped core (<c>loop.end</c>, 18).</summary>
    public required int LoopEnd { get; init; }

    /// <summary>Number of passes over the core (<c>loop.count</c>, 3); 1 = no loop.</summary>
    public required int LoopPasses { get; init; }

    /// <summary>Rank of the cross-pass controllers (<c>loop.gate_rank</c>, 64).</summary>
    public required int LoopGateRank { get; init; }

    /// <summary>Layers in the looped core.</summary>
    public int CoreLayers => LoopEnd - LoopStart;

    /// <summary>KV / conv cache slots: every physical layer plus one extra copy of the core per extra pass (24 + 2 x 12 = 48).</summary>
    /// <param name="physicalLayers">Physical layer count (<c>block_count</c>).</param>
    /// <returns>The slot count.</returns>
    public int CacheSlots(int physicalLayers) => physicalLayers + (LoopPasses - 1) * CoreLayers;

    /// <summary>Actionable refusal for backends that cannot run <see cref="DotLLM.Core.Configuration.Architecture.Inkling2"/> yet.</summary>
    /// <param name="backend">Backend name shown to the user.</param>
    /// <returns>The refusal message.</returns>
    public static string UnsupportedMessage(string backend)
        => $"The inkling2 architecture (Thinking Machines Inkling 2: looped MoE with depth mixing and relative-position attention) has a CPU reference forward only; "
         + $"the {backend} backend is not implemented yet (epic https://github.com/jamesburton/dotLLM/issues/2025). Loading it through a generic loader would silently "
         + "run a different model, so it is refused. Use the CPU backend.";
}
