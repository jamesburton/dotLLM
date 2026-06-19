using DotLLM.Core.Lora;
using DotLLM.Core.Models;

namespace DotLLM.Tests.Integration.Models.Lora;

internal static unsafe class SyntheticLoraAdapterFactory
{
    // Builds q_proj/v_proj adapters on every layer with deterministic weights.
    public static LoraAdapter ForConfig(ModelConfig cfg, int rank, float alpha, int seed)
    {
        var adapter = new LoraAdapter("synthetic", rank, alpha, new[] { "q_proj", "v_proj" });
        int dModel = cfg.HiddenSize;
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int vOut = cfg.NumKvHeads * cfg.HeadDim;
        var rng = new Random(seed);
        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            AddProj(adapter, layer, "q_proj", dModel, qOut, rank, rng);
            AddProj(adapter, layer, "v_proj", dModel, vOut, rank, rng);
        }
        return adapter;
    }

    private static void AddProj(LoraAdapter a, int layer, string proj, int dIn, int dOut, int r, Random rng)
    {
        nint bH = LoraAdapter.AllocAligned((long)r * dIn);     // B: [r, dIn]
        nint aH = LoraAdapter.AllocAligned((long)dOut * r);    // A: [dOut, r]
        Fill((float*)bH, (long)r * dIn, rng);
        Fill((float*)aH, (long)dOut * r, rng);
        a.AddLayerWeights(layer, proj, new LoraLayerWeights(aH, bH, dIn, dOut));
    }

    private static void Fill(float* p, long n, Random rng)
    {
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 0.02 - 0.01); // small deltas
    }
}
