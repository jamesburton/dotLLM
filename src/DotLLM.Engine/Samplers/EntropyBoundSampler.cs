using System.Buffers;
using System.Numerics.Tensors;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Entropy-bounded parallel unmasking sampler for masked text-diffusion decoding
/// (DiffusionGemma's <c>EntropyBoundSamplerConfig</c>).
/// </summary>
/// <remarks>
/// <para>
/// For each currently-masked canvas position this computes the predictive entropy of the softmax
/// over its (optionally soft-capped, temperature-scaled) logits. Positions are then committed in
/// <i>ascending</i> entropy order — the most confident first — while their running entropy total
/// stays under <see cref="DiffusionStepContext.EntropyBound"/> and the
/// <see cref="DiffusionStepContext.UnmaskBudget"/> is not exhausted. This realises the intended
/// "confident positions unmask first" behaviour: a low-entropy (peaked) distribution carries little
/// information risk, so many such positions fit under the bound, whereas a single high-entropy
/// position can consume the whole budget by itself.
/// </para>
/// <para>
/// At least one position is always committed while masked positions remain (the lowest-entropy one),
/// so the canvas cannot stall even if every position's entropy exceeds the bound.
/// </para>
/// <para>
/// Token selection for the chosen positions reuses the ordinary per-position sampler steps
/// (temperature → top-k → top-p → …) via a <see cref="SamplerPipeline"/>; this sampler does not
/// reinvent categorical sampling. As temperature → 0 the pipeline is greedy, so selection and
/// tokens are deterministic (argmax).
/// </para>
/// </remarks>
public sealed class EntropyBoundSampler : IDiffusionUnmaskSampler
{
    private const float MinTemperature = 1e-4f;

    private readonly SamplerPipeline? _tokenSampler;

    /// <summary>
    /// Creates an entropy-bound sampler.
    /// </summary>
    /// <param name="tokenSampler">
    /// Per-position token sampler used to pick the committed positions' token ids. When null,
    /// committed positions take their argmax token — deterministic, matching the temperature → 0
    /// behaviour. Supply a configured pipeline to apply stochastic top-k / top-p / temperature
    /// token selection on top of the entropy-bound position selection.
    /// </param>
    public EntropyBoundSampler(SamplerPipeline? tokenSampler = null)
        => _tokenSampler = tokenSampler;

    /// <inheritdoc/>
    public UnmaskDecision SelectAndSample(
        ReadOnlySpan<float> logits,
        ReadOnlySpan<int> maskedPositions,
        int vocabSize,
        int maskedCount,
        DiffusionStepContext context)
    {
        if (vocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), vocabSize, "Vocab size must be positive.");
        if (maskedCount < 0)
            throw new ArgumentOutOfRangeException(nameof(maskedCount), maskedCount, "Masked count cannot be negative.");
        if (maskedPositions.Length != maskedCount)
            throw new ArgumentException("maskedPositions length must equal maskedCount.", nameof(maskedPositions));
        if (logits.Length != (long)maskedCount * vocabSize)
            throw new ArgumentException("logits length must equal maskedCount * vocabSize.", nameof(logits));

        if (maskedCount == 0)
        {
            return new UnmaskDecision
            {
                UnmaskedPositions = [],
                TokenIds = [],
                PerPositionEntropy = [],
            };
        }

        float temperature = MathF.Max(context.Temperature, MinTemperature);
        float softCap = context.LogitSoftCap;

        var entropy = new float[maskedCount];

        float[]? rented = null;
        Span<float> probs = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : (rented = ArrayPool<float>.Shared.Rent(vocabSize)).AsSpan(0, vocabSize);

        try
        {
            for (int r = 0; r < maskedCount; r++)
            {
                ReadOnlySpan<float> row = logits.Slice(r * vocabSize, vocabSize);
                entropy[r] = ComputeEntropy(row, temperature, softCap, probs);
            }
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }

        // Materialize positions so the comparison lambda can capture them (a ref-struct span cannot
        // be captured) and so the output arrays can index by row.
        int[] positionsByRow = maskedPositions.ToArray();

        // Rank rows by ascending entropy (most confident first). Ties broken by lower position
        // index for determinism.
        int[] order = new int[maskedCount];
        for (int i = 0; i < maskedCount; i++)
            order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int c = entropy[a].CompareTo(entropy[b]);
            return c != 0 ? c : positionsByRow[a].CompareTo(positionsByRow[b]);
        });

        int budget = context.UnmaskBudget > 0 ? context.UnmaskBudget : 1;
        budget = Math.Min(budget, maskedCount);
        float bound = context.EntropyBound;

        var chosenRows = new List<int>(budget);
        double runningEntropy = 0;
        for (int k = 0; k < maskedCount && chosenRows.Count < budget; k++)
        {
            int row = order[k];
            // Always admit the single most-confident position so the canvas never stalls; admit
            // further positions only while the cumulative entropy stays under the bound.
            if (chosenRows.Count > 0 && runningEntropy + entropy[row] > bound)
                break;
            runningEntropy += entropy[row];
            chosenRows.Add(row);
        }

        var positions = new int[chosenRows.Count];
        var tokens = new int[chosenRows.Count];
        Span<float> logitBuf = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : new float[vocabSize];

        for (int i = 0; i < chosenRows.Count; i++)
        {
            int row = chosenRows[i];
            logits.Slice(row * vocabSize, vocabSize).CopyTo(logitBuf);
            ApplySoftCap(logitBuf, softCap);
            positions[i] = positionsByRow[row];
            // Default: deterministic argmax (reproducible, == temperature → 0). When a pipeline is
            // supplied, reuse it for stochastic token sampling.
            tokens[i] = _tokenSampler is null
                ? TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)logitBuf)
                : _tokenSampler.Sample(logitBuf, []);
        }

        return new UnmaskDecision
        {
            UnmaskedPositions = positions,
            TokenIds = tokens,
            PerPositionEntropy = entropy,
        };
    }

    /// <summary>
    /// Computes the Shannon entropy (nats) of the softmax over a single position's logits, after the
    /// optional Gemma logit soft-cap and temperature scaling. <paramref name="probs"/> is scratch of
    /// length <c>vocabSize</c>.
    /// </summary>
    private static float ComputeEntropy(ReadOnlySpan<float> logits, float temperature, float softCap, Span<float> probs)
    {
        logits.CopyTo(probs);
        ApplySoftCap(probs, softCap);
        if (temperature != 1.0f)
            TensorPrimitives.Multiply(probs, 1f / temperature, probs);

        TensorPrimitives.SoftMax(probs, probs);

        // H = -Σ p·ln p. Skip non-positive p (0·ln0 = 0) for numerical safety.
        double h = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            float p = probs[i];
            if (p > 0f)
                h -= p * Math.Log(p);
        }
        return (float)h;
    }

    /// <summary>Applies the Gemma logit soft-cap <c>cap·tanh(x/cap)</c> in place. No-op when cap ≤ 0.</summary>
    private static void ApplySoftCap(Span<float> logits, float softCap)
    {
        if (softCap <= 0f)
            return;
        for (int i = 0; i < logits.Length; i++)
            logits[i] = softCap * MathF.Tanh(logits[i] / softCap);
    }
}
