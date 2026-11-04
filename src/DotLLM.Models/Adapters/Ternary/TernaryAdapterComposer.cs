using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Adapters.Ternary;

/// <summary>
/// Composes several ternary adapters into one by rank concatenation (the <c>LoraComposer</c> pattern):
/// per layer, <c>W_down = [W_down_1; ...; W_down_n]</c> (rows stacked), <c>W_up^T</c> likewise, and the
/// per-rank-row alpha vector carries each adapter's own gain, so a single pass over Σ K_i rank rows
/// computes <c>x + Σ_i alpha_i W_up_i sigma(W_down_i x)</c>. All adapters read the same input row (parallel
/// experts, NOT a sequential chain). Rows are bit-copied, so the composite owns its storage and the
/// sources may be disposed afterwards.
/// </summary>
public static unsafe class TernaryAdapterComposer
{
    /// <summary>
    /// Builds the composite. <paramref name="weights"/> optionally scales each adapter's alpha
    /// (like LoRA stack weights); null means 1 for all.
    /// </summary>
    /// <param name="stack">Adapters in concatenation order (order fixes the summation order).</param>
    /// <param name="weights">Optional per-adapter multiplier on alpha.</param>
    /// <param name="maxRank">Cap on the summed rank of any layer.</param>
    public static TernaryResidualAdapter Compose(IReadOnlyList<TernaryResidualAdapter> stack,
        IReadOnlyList<float>? weights = null, int maxRank = 256)
    {
        ArgumentNullException.ThrowIfNull(stack);
        if (stack.Count == 0) throw new ArgumentException("Stack must contain at least one adapter.", nameof(stack));
        if (weights is not null && weights.Count != stack.Count)
            throw new ArgumentException("weights.Count must equal stack.Count.", nameof(weights));
        int d = stack[0].DModel;
        bool relu = stack[0].Relu;
        foreach (var a in stack)
        {
            if (a.DModel != d) throw new InvalidOperationException($"Adapter '{a.Name}' d_model {a.DModel} != {d}.");
            if (a.Relu != relu) throw new InvalidOperationException($"Adapter '{a.Name}' sigma differs from '{stack[0].Name}'.");
        }

        int maxLayer = stack.Max(a => a.Layers.Length) - 1;
        int wpr = TernaryAdapterKernel.WordsPerRow(d);
        var layers = new List<TernaryLayerData>();
        var natives = new List<nint>();
        try
        {
            for (int layer = 0; layer <= maxLayer; layer++)
            {
                int total = 0;
                foreach (var a in stack) total += a.RankAt(layer);
                if (total == 0) continue;
                if (total > maxRank)
                    throw new NotSupportedException($"Composed ternary rank {total} at layer {layer} exceeds cap {maxRank}.");
                long words = (long)total * wpr;
                ulong* down = (ulong*)NativeMemory.AlignedAlloc((nuint)(words * 8), 64);
                natives.Add((nint)down);
                ulong* up = (ulong*)NativeMemory.AlignedAlloc((nuint)(words * 8), 64);
                natives.Add((nint)up);
                float* ra = (float*)NativeMemory.AlignedAlloc((nuint)total * sizeof(float), 64);
                natives.Add((nint)ra);
                int row = 0;
                for (int s = 0; s < stack.Count; s++)
                {
                    int r = stack[s].RankAt(layer);
                    if (r == 0) continue;
                    ref readonly TernaryLayerData src = ref stack[s].Layers[layer];
                    long n = (long)r * wpr;
                    Buffer.MemoryCopy(src.Down, down + (long)row * wpr, n * 8, n * 8);
                    Buffer.MemoryCopy(src.Up, up + (long)row * wpr, n * 8, n * 8);
                    float w = weights?[s] ?? 1f;
                    for (int j = 0; j < r; j++) ra[row + j] = src.RankAlpha[j] * w;
                    row += r;
                }
                layers.Add(new TernaryLayerData(layer, total, down, up, ra));
            }
            string name = "stack[" + string.Join("+", stack.Select(a => a.Name)) + "]";
            return new TernaryResidualAdapter(name, d, relu, layers.ToArray(), new List<IDisposable>(), natives);
        }
        catch
        {
            foreach (nint b in natives) NativeMemory.AlignedFree((void*)b);
            throw;
        }
    }
}
