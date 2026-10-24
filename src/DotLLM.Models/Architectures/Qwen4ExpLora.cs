using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Runtime LoRA state of one <see cref="Qwen4ExpTransformerModel"/> (#845): the adapter active for the current forward call and the shared
/// delta application the model's projections call after their base GEMM. Adapters are never merged into the weights (docs/LORA.md).
/// </summary>
internal sealed unsafe class Qwen4ExpLoraContext(ComputeThreadPool? pool)
{
    /// <summary>The adapter of the forward call in progress, or null.</summary>
    public ILoraAdapter? Adapter;

    /// <summary><c>y += (alpha / rank) * (x B) A</c> for the site, when <see cref="Adapter"/> targets it.</summary>
    public void Apply(int layer, string projName, ReadOnlySpan<float> x, Span<float> y, int tokens, int inDim, int outDim)
    {
        if (Adapter is not { } adapter) return;
        fixed (float* xp = x)
        fixed (float* yp = y)
            LoraProjection.Apply(adapter, layer, projName, xp, yp, tokens, inDim, outDim, pool);
    }

    /// <summary>
    /// Supported sites and their shapes, for <paramref name="config"/>:
    /// QSA layers <c>q_proj</c> (the FUSED <c>[q | gate]</c> projection, <c>2 * heads * headDim</c> wide), <c>k_proj</c>, <c>v_proj</c>, <c>o_proj</c>;
    /// Gated-DeltaNet layers <c>in_proj_qkv</c>, <c>in_proj_z</c>, <c>in_proj_a</c>, <c>in_proj_b</c>, <c>out_proj</c> (HF names; the factors must be
    /// expressed in the GGUF layout the model consumes - value heads TILED - exactly like the base weights); and the routed experts
    /// <c>mlp.experts.{j}.{gate,up,down}_proj</c> on every layer. Anything else is rejected with <see cref="NotSupportedException"/> (unsupported
    /// target) or <see cref="ArgumentException"/> (a supported name at the wrong layer or with the wrong shape).
    /// </summary>
    public static void Validate(ILoraAdapter adapter, ModelConfig config)
    {
        if (adapter is not LoraAdapter concrete) return;   // an adapter type that cannot be enumerated is checked site by site at apply time
        var layout = config.HybridLayout!;
        var gdn = config.GdnConfig!.Value;
        var moe = config.Moe!;
        int hidden = config.HiddenSize, nH = config.NumAttentionHeads, d = config.HeadDim;
        int convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState, vDim = gdn.NVHead * gdn.DState;

        foreach (var ((layer, proj), w) in concrete.LayerWeights)
        {
            string site = $"LoRA adapter '{adapter.Name}' layer {layer} projection '{proj}'";
            if (layer < 0 || layer >= config.NumLayers)
                throw new ArgumentException($"{site}: the model has {config.NumLayers} layers (the MTP block takes no adapter).");
            bool gdnLayer = layout.LayerKind[layer] == HybridLayerKind.GatedDeltaNet;

            (int In, int Out)? expected = null;
            string? family = null;                              // which layer kind owns the name
            switch (proj)
            {
                case "q_proj": family = "qsa"; expected = (hidden, 2 * nH * d); break;
                case "k_proj":
                case "v_proj": family = "qsa"; expected = (hidden, layout.HeadCountKv[layer] * d); break;
                case "o_proj": family = "qsa"; expected = (nH * d, hidden); break;
                case "in_proj_qkv": family = "gdn"; expected = (hidden, convDim); break;
                case "in_proj_z": family = "gdn"; expected = (hidden, vDim); break;
                case "in_proj_a":
                case "in_proj_b": family = "gdn"; expected = (hidden, gdn.NVHead); break;
                case "out_proj": family = "gdn"; expected = (vDim, hidden); break;
                default:
                    if (TryExpert(proj, moe, hidden, out var ex)) { expected = ex; break; }
                    throw new NotSupportedException(
                        $"{site} is not supported on qwen4exp. Supported targets: QSA q_proj/k_proj/v_proj/o_proj, Gated-DeltaNet in_proj_qkv/in_proj_z/" +
                        "in_proj_a/in_proj_b/out_proj, and routed-expert mlp.experts.{j}.{gate|up|down}_proj. The shared expert, the expert router, the " +
                        "hyper-connection mixers, the QSA indexer, the n-gram (PLE) branch, the embeddings and the LM head cannot be adapted.");
            }
            if (family == "qsa" && gdnLayer || family == "gdn" && !gdnLayer)
                throw new ArgumentException(
                    $"{site}: layer {layer} is a {(gdnLayer ? "Gated-DeltaNet" : "QSA attention")} layer, which has no '{proj}' projection.");
            if (w.InputDim != expected!.Value.In || w.OutputDim != expected.Value.Out)
                throw new ArgumentException(
                    $"{site}: factor shape {w.InputDim}x{w.OutputDim} does not match the base projection {expected.Value.In}x{expected.Value.Out}.");
        }
    }

    private static bool TryExpert(string proj, MoeConfig moe, int hidden, out (int In, int Out) shape)
    {
        shape = default;
        const string prefix = "mlp.experts.";
        if (!proj.StartsWith(prefix, StringComparison.Ordinal)) return false;
        var rest = proj.AsSpan(prefix.Length);
        int dot = rest.IndexOf('.');
        if (dot <= 0 || !int.TryParse(rest[..dot], out int expert) || (uint)expert >= (uint)moe.NumExperts) return false;
        int inter = moe.MoeIntermediateSize;
        switch (rest[(dot + 1)..].ToString())
        {
            case "gate_proj":
            case "up_proj": shape = (hidden, inter); return true;
            case "down_proj": shape = (inter, hidden); return true;
            default: return false;
        }
    }
}
