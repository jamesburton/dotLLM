using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.SafeTensors;

public static partial class HfConfigExtractor
{
    /// <summary>
    /// Builds the <see cref="ModelConfig"/> of a ricursiveAI Ripple <c>config.json</c> (<c>RicursiveForCausalLM</c>, issue #1643).
    /// </summary>
    /// <remarks>
    /// Field names and defaults are the vendor <c>configuration_ricursive.py</c> <c>RicursiveConfig</c>'s (Ripple-65B-Preview, transformers
    /// 4.57): <c>layer_types</c> (<c>"sliding_attention"</c> = window + RoPE, <c>"full_attention"</c> = NoPE; default every 5th layer full),
    /// <c>sliding_window</c> (1024), <c>use_gated_attention</c> (true), <c>gate_full_precision</c> (true), <c>use_valres</c> (<b>false</b>),
    /// <c>ffn_clamp_threshold</c> (null; the reference tests <c>if clamp_threshold:</c>, so null and 0 both mean no clamp),
    /// <c>embed_scale</c> (null = folded into <c>embed_tokens.weight</c>), <c>head_dim</c> (128), <c>rope_theta</c>, <c>rms_norm_eps</c>,
    /// <c>tie_word_embeddings</c> (false). <c>hybrid_layer_pattern</c> / <c>is_hybrid_swa</c> are derived fields the vendor recomputes from
    /// <c>layer_types</c>; a stored <c>hybrid_layer_pattern</c> that disagrees with <c>layer_types</c> is rejected rather than ignored.
    /// </remarks>
    private static ModelConfig ExtractRicursive(JsonElement root)
    {
        int hidden = GetInt32(root, "hidden_size");
        int layers = GetInt32(root, "num_hidden_layers");
        int heads = GetInt32(root, "num_attention_heads");
        int kvHeads = GetInt32OrDefault(root, "num_key_value_heads", heads);
        int ffn = GetInt32(root, "intermediate_size");
        int vocab = GetInt32(root, "vocab_size");
        int headDim = GetInt32OrDefault(root, "head_dim", 128);
        if (hidden <= 0 || layers <= 0 || heads <= 0 || kvHeads <= 0 || ffn <= 0 || vocab <= 0 || headDim <= 0 || heads % kvHeads != 0)
            throw new InvalidDataException(
                $"ricursive config.json has an invalid shape (hidden {hidden}, layers {layers}, heads {heads}/{kvHeads}, ffn {ffn}, vocab {vocab}, head_dim {headDim}).");

        string act = GetStringOrDefault(root, "hidden_act", "silu")!;
        if (!string.Equals(act, "silu", StringComparison.Ordinal))
            throw new NotSupportedException($"ricursive config.json hidden_act '{act}' is not supported (only silu).");

        int window = GetInt32OrDefault(root, "sliding_window", 1024);
        if (window <= 0)
            throw new InvalidDataException($"ricursive config.json sliding_window must be > 0, got {window}.");

        bool[] sliding = ReadRicursiveLayerTypes(root, layers);
        if (Array.IndexOf(sliding, true) < 0)
            throw new InvalidDataException("ricursive config.json layer_types has no sliding_attention layer; the model would have no positional encoding.");

        var perLayer = new int?[layers];
        var noRope = new List<int>();
        for (int l = 0; l < layers; l++)
        {
            perLayer[l] = sliding[l] ? window : null;
            if (!sliding[l]) noRope.Add(l);
        }

        float? embedScale = null;
        if (root.TryGetProperty("embed_scale", out var es) && es.ValueKind != JsonValueKind.Null)
        {
            if (es.ValueKind != JsonValueKind.Number || !es.TryGetDouble(out double esv) || !double.IsFinite(esv) || esv <= 0)
                throw new InvalidDataException($"ricursive config.json embed_scale must be null (folded) or a finite positive number, got {es}.");
            embedScale = (float)esv;
        }

        bool clampDeclared = false;
        float clampValue = 0f;
        if (root.TryGetProperty("ffn_clamp_threshold", out var ct) && ct.ValueKind != JsonValueKind.Null)
        {
            if (ct.ValueKind != JsonValueKind.Number || !ct.TryGetDouble(out double ctv) || !double.IsFinite(ctv) || ctv < 0)
                throw new InvalidDataException($"ricursive config.json ffn_clamp_threshold must be null or a finite non-negative number, got {ct}.");
            clampDeclared = true;
            clampValue = (float)ctv;
        }
        var clamp = new float[layers];
        Array.Fill(clamp, clampValue);

        float theta = GetFloatOrDefault(root, "rope_theta", 10000.0f);
        return new ModelConfig
        {
            Architecture = Architecture.Ricursive,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = ffn,
            NumLayers = layers,
            NumAttentionHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            MaxSequenceLength = GetInt32OrDefault(root, "max_position_embeddings", 2048),
            NormEpsilon = GetFloatOrDefault(root, "rms_norm_eps", 1e-6f),
            AttentionType = AttentionType.GQA,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            EmbeddingScale = embedScale,
            TiedEmbeddings = GetBoolOrDefault(root, "tie_word_embeddings", false),
            RoPEConfig = new RoPEConfig(Theta: theta, DimensionCount: headDim, Type: RoPEType.NeoX),
            PositionEncodingType = PositionEncodingType.RoPE,
            SlidingWindowSize = window,
            PerLayerSlidingWindow = perLayer,
            NoRopeLayers = noRope,
            UseQkNorm = true,
            Ricursive = new RicursiveConfig
            {
                UseGatedAttention = GetBoolOrDefault(root, "use_gated_attention", true),
                GateFullPrecision = GetBoolOrDefault(root, "gate_full_precision", true),
                UseValueResidual = GetBoolOrDefault(root, "use_valres", false),
                FfnClamp = clamp,
                FfnClampDeclared = clampDeclared,
                EmbeddingScaleFolded = embedScale is null,
            },
        };
    }

    /// <summary>Per-layer sliding flags from <c>layer_types</c> (default: full iff <c>i % 5 == 4</c>), cross-checked with a stored <c>hybrid_layer_pattern</c>.</summary>
    private static bool[] ReadRicursiveLayerTypes(JsonElement root, int layers)
    {
        var sliding = new bool[layers];
        if (root.TryGetProperty("layer_types", out var lt) && lt.ValueKind == JsonValueKind.Array)
        {
            if (lt.GetArrayLength() != layers)
                throw new InvalidDataException($"ricursive config.json layer_types has {lt.GetArrayLength()} entries, expected {layers}.");
            int i = 0;
            foreach (var e in lt.EnumerateArray())
            {
                sliding[i] = e.ValueKind == JsonValueKind.String ? e.GetString() switch
                {
                    "sliding_attention" => true,
                    "full_attention" => false,
                    var other => throw new InvalidDataException($"ricursive config.json layer_types[{i}] = '{other}' is not supported."),
                } : throw new InvalidDataException($"ricursive config.json layer_types[{i}] must be a string.");
                i++;
            }
        }
        else
        {
            for (int l = 0; l < layers; l++)
                sliding[l] = l % RicursiveConfig.DefaultFullAttentionPeriod != RicursiveConfig.DefaultFullAttentionPeriod - 1;
        }

        if (root.TryGetProperty("hybrid_layer_pattern", out var hp) && hp.ValueKind == JsonValueKind.Array)
        {
            int i = 0;
            bool agrees = hp.GetArrayLength() == layers;
            if (agrees)
                foreach (var e in hp.EnumerateArray())
                {
                    if (e.ValueKind != JsonValueKind.Number || !e.TryGetInt32(out int v) || (v != 0) != sliding[i]) { agrees = false; break; }
                    i++;
                }
            if (!agrees)
                throw new InvalidDataException(
                    "ricursive config.json hybrid_layer_pattern disagrees with layer_types (1 = sliding_attention, 0 = full_attention); the config is inconsistent.");
        }
        return sliding;
    }
}
