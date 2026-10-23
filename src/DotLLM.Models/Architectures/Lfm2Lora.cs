using DotLLM.Core.Lora;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>Validates the runtime projections implemented by the LFM2 hybrid model.</summary>
internal static class Lfm2Lora
{
    public static void Validate(ILoraAdapter adapter, ModelConfig config)
    {
        if (adapter.Rank <= 0) throw new ArgumentException("LoRA rank must be positive.", nameof(adapter));
        if (adapter is not LoraAdapter concrete)
            throw new NotSupportedException("LFM2 LoRA requires an enumerable LoraAdapter so every target can be validated before state changes.");
        foreach (var ((layer, proj), w) in concrete.LayerWeights)
        {
            string site = $"LoRA '{adapter.Name}' layer {layer} projection '{proj}'";
            if ((uint)layer >= (uint)config.NumLayers) throw new ArgumentException($"{site}: layer is outside the model.");
            int h = config.HiddenSize, q = config.NumAttentionHeads * config.HeadDim;
            int kv = config.HybridLayout!.HeadCountKv[layer] * config.HeadDim;
            bool conv = config.HybridLayout.LayerKind[layer] == HybridLayerKind.ShortConv;
            bool moe = config.Moe!.IsMoeLayer(layer);
            (int In, int Out) expected;
            switch (proj)
            {
                case "in_proj" when conv: expected = (h, 3 * h); break;
                case "out_proj" when conv: expected = (h, h); break;
                case "q_proj" when !conv: expected = (h, q); break;
                case "k_proj" or "v_proj" when !conv: expected = (h, kv); break;
                case "o_proj" when !conv: expected = (q, h); break;
                case "gate_proj" or "up_proj" when !moe: expected = (h, config.IntermediateSize); break;
                case "down_proj" when !moe: expected = (config.IntermediateSize, h); break;
                default:
                    if (proj.StartsWith("mlp.experts.", StringComparison.Ordinal))
                    {
                        string[] parts = proj.Split('.');
                        if (moe && parts.Length == 4 && int.TryParse(parts[2], out int e) && (uint)e < (uint)config.Moe.NumExperts && parts[2] == e.ToString(System.Globalization.CultureInfo.InvariantCulture) && parts[3] is "gate_proj" or "up_proj" or "down_proj")
                        { expected = parts[3] == "down_proj" ? (config.Moe.MoeIntermediateSize, h) : (h, config.Moe.MoeIntermediateSize); break; }
                    }
                    if (proj is "in_proj" or "out_proj" or "q_proj" or "k_proj" or "v_proj" or "o_proj" or "gate_proj" or "up_proj" or "down_proj")
                        throw new ArgumentException($"{site}: projection does not exist on this layer kind.");
                    throw new NotSupportedException($"{site}: unsupported LFM2 LoRA target. Norms, convolution kernels, routers, embeddings and LM head cannot be adapted.");
            }
            if (w.InputDim != expected.In || w.OutputDim != expected.Out)
                throw new ArgumentException($"{site}: factor shape {w.InputDim}x{w.OutputDim} differs from projection {expected.In}x{expected.Out}.");
        }
    }
}
