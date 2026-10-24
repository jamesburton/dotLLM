using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Core.Lora;

/// <summary>
/// Composes an ordered stack of LoRA adapters into a single <see cref="LoraAdapter"/>
/// by rank-concatenation, so every backend's existing single-adapter apply path
/// computes the additive stack delta <c>Σᵢ wᵢ·(αᵢ/rᵢ)·(x·Bᵢ)·Aᵢ</c> unchanged.
/// </summary>
/// <remarks>
/// For each adapted <c>(layer, proj)</c> site the composite buffers are
/// <c>B = [B₁;…;Bₙ]</c> (rows stacked on the rank axis) and
/// <c>A = [A₁'|…|Aₙ']</c> (columns concatenated on the rank axis) where each
/// <c>Aᵢ' = (wᵢ·αᵢ/rᵢ)·Aᵢ</c>. The composite's <c>Alpha = Rank = Σrᵢ</c> so the
/// runtime scale (<c>Alpha/Rank</c>) is 1. Stacked adapters must have uniform site
/// coverage (every site targeted by one is targeted by all) and F32 weights.
/// </remarks>
public static unsafe class LoraComposer
{
    private static readonly string[] StandardProjections =
        { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" };

    /// <summary>
    /// Composes a stack of LoRA adapters into a single composite <see cref="LoraAdapter"/>
    /// via rank-concatenation. The composite has <c>Rank = Alpha = Σrᵢ</c> so that
    /// the runtime's <c>scale = Alpha/Rank = 1</c> and each adapter's contribution
    /// is pre-baked into the A columns as <c>wᵢ·αᵢ/rᵢ</c>.
    /// </summary>
    /// <param name="stack">
    /// Ordered list of (adapter, weight) pairs. Weight 1.0 means full contribution;
    /// weight 0.5 means half contribution. At least one element required.
    /// </param>
    /// <param name="cfg">
    /// Model configuration used to enumerate layers. Only <see cref="ModelConfig.NumLayers"/>
    /// is consumed; all adapters are expected to be compatible with this config.
    /// </param>
    /// <param name="maxRank">
    /// Maximum allowed composite rank. Mirrors <c>CudaForwardState.MaxLoraRank</c> (default 256).
    /// Throws <see cref="NotSupportedException"/> if the sum exceeds this cap.
    /// </param>
    /// <returns>
    /// A new <see cref="LoraAdapter"/> that is owned by the caller and must be disposed.
    /// The input adapters are NOT disposed — the caller continues to own them.
    /// </returns>
    /// <exception cref="ArgumentException">Stack is empty.</exception>
    /// <exception cref="NotSupportedException">
    /// Composite rank exceeds <paramref name="maxRank"/>, or an adapter has non-F32 weights.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Adapters have non-uniform coverage (some but not all target a given site), or
    /// incompatible dimensions at a site.
    /// </exception>
    public static LoraAdapter Compose(
        IReadOnlyList<(ILoraAdapter adapter, float weight)> stack,
        ModelConfig cfg,
        int maxRank = 256)
    {
        ArgumentNullException.ThrowIfNull(stack);
        ArgumentNullException.ThrowIfNull(cfg);
        if (stack.Count == 0)
            throw new ArgumentException("Stack must contain at least one adapter.", nameof(stack));

        int totalRank = 0;
        foreach (var (a, _) in stack)
            totalRank += a.Rank;

        if (totalRank > maxRank)
            throw new NotSupportedException(
                $"Composed LoRA rank {totalRank} exceeds the device delta rank cap ({maxRank}). " +
                "Use fewer or lower-rank adapters.");

        var projNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (a, _) in stack)
            foreach (var p in a.TargetModules) projNames.Add(p);
        if (projNames.Count == 0)
            foreach (var p in StandardProjections) projNames.Add(p);

        string name = "stack[" + string.Join("+", stack.Select(s => s.adapter.Name)) + "]";
        var composite = new LoraAdapter(name, totalRank, totalRank, projNames.ToArray());

        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                foreach (var proj in projNames)
                {
                    // Coverage at this site: either ALL or NONE of the stack must target it.
                    int covered = 0;
                    foreach (var (a, _) in stack)
                        if (a.GetLayerWeights(layer, proj) is not null) covered++;
                    if (covered == 0) continue;
                    if (covered != stack.Count)
                        throw new InvalidOperationException(
                            $"LoRA stack has non-uniform coverage at layer {layer} '{proj}': " +
                            $"{covered}/{stack.Count} adapters target it. Stacked adapters must target " +
                            "the same (layer, projection) sites.");

                    int inputDim = -1, outputDim = -1, rSum = 0;
                    foreach (var (a, _) in stack)
                    {
                        var w = a.GetLayerWeights(layer, proj)!.Value;
                        if (w.WeightDType != LoraWeightDType.F32 || w.ResolvedAWeightDType != LoraWeightDType.F32)
                            throw new NotSupportedException(
                                $"LoRA stacking is F32-only; adapter '{a.Name}' layer {layer} '{proj}' " +
                                $"is {w.WeightDType}.");
                        if (inputDim < 0) { inputDim = w.InputDim; outputDim = w.OutputDim; }
                        else if (w.InputDim != inputDim || w.OutputDim != outputDim)
                            throw new InvalidOperationException(
                                $"Incompatible dims at layer {layer} '{proj}': {w.InputDim}x{w.OutputDim} " +
                                $"vs {inputDim}x{outputDim}.");
                        rSum += a.Rank;
                    }

                    nint bConcat = LoraAdapter.AllocAligned((long)rSum * inputDim);   // [rSum, inputDim]
                    nint aConcat = LoraAdapter.AllocAligned((long)outputDim * rSum);  // [outputDim, rSum]
                    float* bDst = (float*)bConcat;
                    float* aDst = (float*)aConcat;

                    int rowOffset = 0;
                    foreach (var (a, weight) in stack)
                    {
                        var w = a.GetLayerWeights(layer, proj)!.Value;
                        int rank = a.Rank;
                        float scale = weight * (a.Alpha / a.Rank);

                        // B block: contiguous [rank, inputDim] copy at row offset.
                        Buffer.MemoryCopy((void*)w.BHandle, bDst + (long)rowOffset * inputDim,
                            (long)rank * inputDim * sizeof(float),
                            (long)rank * inputDim * sizeof(float));

                        // A block: per output row, copy [rank] cols at col offset, pre-scaled.
                        float* aSrc = (float*)w.AHandle; // [outputDim, rank]
                        for (int o = 0; o < outputDim; o++)
                        {
                            float* src = aSrc + (long)o * rank;
                            float* dst = aDst + (long)o * rSum + rowOffset;
                            for (int r = 0; r < rank; r++) dst[r] = src[r] * scale;
                        }
                        rowOffset += rank;
                    }

                    composite.AddLayerWeights(layer, proj,
                        new LoraLayerWeights(aConcat, bConcat, inputDim, outputDim));
                }
            }
        }
        catch
        {
            composite.Dispose(); // free any buffers added before the failure
            throw;
        }
        return composite;
    }
}
