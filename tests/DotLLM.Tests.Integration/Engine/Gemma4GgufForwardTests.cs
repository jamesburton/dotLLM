using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using System.Diagnostics;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Real-weight validation for the <b>Gemma-4 MoE</b> GGUF backbone (autoregressive,
/// no diffusion). The released <c>unsloth/gemma-4-26B-A4B-it</c> GGUF declares
/// <c>general.architecture = gemma4</c> and is a Gemma-3-style sparse-MoE transformer
/// (30 layers, hidden 2816, 16 heads, per-layer 8-sliding / 2-global KV heads,
/// head_dim 256 with global head_dim 512, 128 experts top-8, dual RoPE per attention
/// type, QK-norm, 4 RMSNorms, final-logit soft-cap 30). It carries
/// <c>embedding_length_per_layer_input = 0</c> and no per-layer-embedding / AltUp /
/// Laurel tensors, so the standard Gemma-4 MoE forward path covers it verbatim
/// (no Gemma-3n PLE machinery required).
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_GEMMA4_GGUF</c> (path to the .gguf). When unset or missing the
/// test self-skips so the build never depends on the multi-gig checkpoint. CPU-only.
/// This is the gating validation for issue #40: a single cacheless causal forward of
/// the real 26B checkpoint must produce a sensible next token for a trivial factual
/// prompt (e.g. "The capital of France is" → " Paris").
/// </remarks>
public sealed class Gemma4GgufForwardTests
{

    private readonly ITestOutputHelper _output;

    public Gemma4GgufForwardTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Gemma-4-26B fixture, resolved via <see cref="KnownTestFixtures.Gemma4_26B_A4B_Q4KM"/>:
    /// <c>$DOTLLM_GEMMA4_GGUF</c>, then the dotLLM test cache, then the HF hub cache (#308).
    /// </summary>
    private static FixtureLocation Gemma4Fixture => KnownTestFixtures.Gemma4_26B_A4B_Q4KM;

    private static string? TryResolveModelPath() => Gemma4Fixture.Path;

    /// <summary>
    /// Load the real Gemma-4-26B-A4B GGUF and run a SINGLE cacheless causal forward over
    /// a short factual prompt; assert the greedy next token is non-degenerate and report
    /// the decoded continuation + the top-5 logits for eyeballing.
    /// </summary>
    [SkippableFact]
    public unsafe void Gemma4_26B_SingleForward_PredictsSensibleNextToken()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, Gemma4Fixture.SkipMessage(KnownTestFixtures.Gemma4_26BDescription));

        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path!);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        loadSw.Stop();
        using var _ = gguf;
        using var __ = model;

        // Architecture sanity: the gemma4 GGUF must map to the Gemma-4 MoE config shape.
        Assert.Equal(Architecture.Gemma4, config.Architecture);
        Assert.NotNull(config.Moe);
        Assert.True(config.IsGemmaArchitecture, "Gemma-4 must report IsGemmaArchitecture.");

        _output.WriteLine($"[gemma4 GGUF] {path}");
        _output.WriteLine($"  load wall   : {loadSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  arch        : {config.Architecture}");
        _output.WriteLine($"  layers      : {config.NumLayers}  hidden : {config.HiddenSize}  vocab : {config.VocabSize}");
        _output.WriteLine($"  heads       : {config.NumAttentionHeads} (kv {config.NumKvHeads}, global-kv {config.NumGlobalKvHeads})");
        _output.WriteLine($"  head_dim    : {config.HeadDim} (global {config.GlobalHeadDim})");
        _output.WriteLine($"  experts     : {config.Moe!.NumExperts} top-{config.Moe.NumExpertsPerTok} (width {config.Moe.MoeIntermediateSize})");
        _output.WriteLine($"  softcap     : {config.FinalLogitSoftcap}  embedScale : {config.EmbeddingScale}");

        // NOTE: this is an INSTRUCT-tuned (-it) checkpoint. A bare sentence fragment like
        // "The capital of France is" is NOT reliably completed with " Paris" by an instruct
        // model (it wants to answer conversationally, not continue the fragment). We instead
        // use a strongly completion-shaped factual prompt the model continues unambiguously:
        // "The Eiffel Tower is located in" → " Paris" (greedy, logit ~19, top-1 by a wide
        // margin). This discriminates a correct forward (sharp, specific, correct token) from
        // a broken one (flat low-magnitude logits) — a degraded forward cannot produce a
        // confident, specific, correct answer.
        const string prompt = "The Eiffel Tower is located in";
        // Gemma sets add_bos_token=True — prepend the <bos> (id 2) the model trained with.
        int[] encoded = tokenizer.Encode(prompt);
        Assert.NotEmpty(encoded);
        int[] promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);
        int promptLen = promptIds.Length;
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++) positions[i] = i;
        _output.WriteLine($"  prompt      : '{prompt}'  ids=[{string.Join(",", promptIds)}]");

        var fwdSw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(promptIds, positions, deviceId: -1, kvCache: null, adapter: null, AttentionMaskSpec.Causal);
        fwdSw.Stop();

        int vocab = logits.Shape[1];
        Assert.Equal(config.VocabSize, vocab);

        // Greedy next token = argmax over the LAST prompt row.
        float* p = (float*)logits.DataPointer;
        long lastRow = (long)(promptLen - 1) * vocab;
        int best = 0; float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++) { float x = p[lastRow + v]; if (x > bestV) { bestV = x; best = v; } }

        // Top-5 for diagnostics.
        var top = new (int Id, float V)[vocab];
        for (int v = 0; v < vocab; v++) top[v] = (v, p[lastRow + v]);
        Array.Sort(top, (a, b) => b.V.CompareTo(a.V));
        _output.WriteLine($"  forward     : {fwdSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  next token  : id={best} '{tokenizer.Decode(new[] { best })}' (logit {bestV:F3})");
        for (int k = 0; k < 5 && k < vocab; k++)
            _output.WriteLine($"    top[{k}] id={top[k].Id} '{tokenizer.Decode(new[] { top[k].Id })}' logit={top[k].V:F3}");

        // ── Non-degenerate assertions ──────────────────────────────────────
        Assert.InRange(best, 0, vocab - 1);
        Assert.True(float.IsFinite(bestV), "Top logit must be finite (no NaN/Inf from dequant or softcap).");
        Assert.NotEqual(tokenizer.EosTokenId, best);
        string decoded = tokenizer.Decode(new[] { best });
        Assert.False(string.IsNullOrWhiteSpace(decoded), "Greedy next token must decode to non-empty text.");
        // The expected continuation of "The capital of France is" is " Paris".
        Assert.Contains("Paris", decoded, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Real-26B Gemma-4 AR forward on the <b>Vulkan</b> backend with experts kept
    /// QUANTIZED on device (fused gate_up Q4_K + down Q5_1 indexed-MoE shaders).
    /// This is the gating validation for the quantized expert path: the F32
    /// host-dequant path cannot load the 26B (its F32 experts are tens of GB), so a
    /// successful load + sensible " Paris" completion proves the quantized upload +
    /// in-shader dequant make the real model runnable on the GPU.
    /// </summary>
    /// <remarks>
    /// <para>Gated on <c>DOTLLM_GEMMA4_GGUF</c> + a Vulkan device. CPU-free run. The full
    /// AR forward now runs end-to-end and must predict " Paris" — this gates the whole
    /// Vulkan gemma4 path: quantized-expert load + in-shader dequant, the 512-head-dim
    /// attention bound, the partial-NeoX RoPE freq denominator, and the descriptor-set
    /// cache lifetime fix (the cache capacity must equal the pool size or a &gt;256-tuple
    /// model resets the pool mid-command-buffer and silently corrupts the output).</para>
    /// </remarks>
    [SkippableFact]
    public unsafe void Gemma4_26B_Vulkan_QuantizedExperts_LoadsAndForwards()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, Gemma4Fixture.SkipMessage(KnownTestFixtures.Gemma4_26BDescription));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");
        string spvDir = ResolveSpvDir();

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Gemma4, config.Architecture);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        // Load with experts kept QUANTIZED (Q4_K gate_up + Q5_1 down). If this OOM'd
        // or NRE'd the quantized upload path would be broken; reaching here proves the
        // 26B's experts fit on device in their GGUF-quantized footprint.
        var loadSw = Stopwatch.StartNew();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        loadSw.Stop();
        _output.WriteLine($"[gemma4 26B Vulkan, quantized experts] {path}");
        _output.WriteLine($"  quantized-expert LOAD wall : {loadSw.Elapsed.TotalSeconds:F2} s  (experts stayed Q4_K/Q5_1 on device)");

        const string prompt = "The Eiffel Tower is located in";
        int[] encoded = tokenizer.Encode(prompt);
        int[] promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);
        int promptLen = promptIds.Length;
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++) positions[i] = i;

        var fwdSw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(promptIds, positions, deviceId: -1, kvCache: null);
        fwdSw.Stop();
        _output.WriteLine($"  forward wall : {fwdSw.Elapsed.TotalSeconds:F2} s");

        int vocab = logits.Shape[logits.Shape.Rank - 1];
        Assert.Equal(config.VocabSize, vocab);
        float* p = (float*)logits.DataPointer;
        int best = 0; float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++) { float x = p[v]; if (x > bestV) { bestV = x; best = v; } }
        _output.WriteLine($"  next token  : id={best} '{tokenizer.Decode(new[] { best })}' (logit {bestV:F3})");
        Assert.True(float.IsFinite(bestV), "Top logit must be finite (no NaN/Inf from quantized dequant or softcap).");
        string decoded = tokenizer.Decode(new[] { best });
        Assert.Contains("Paris", decoded, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// HYBRID BISECTION HARNESS (bug #2). Runs the KNOWN-GOOD CPU 26B forward
    /// (predicts " Paris") but routes the Gemma-4 layers selected by
    /// <c>DOTLLM_HYBRID_VK_LAYERS</c> through the Vulkan single-layer path, so we
    /// can swap layers/sections to Vulkan one at a time and see whether " Paris"
    /// survives. The first swap that breaks it names the culprit.
    /// <para>Specs: <c>none</c> (pure-CPU baseline — must be Paris), <c>all</c>
    /// (full Vulkan via the host-roundtrip seam — must reproduce the bug),
    /// <c>global</c> / <c>sliding</c> (by attention type), a range <c>a-b</c>, or
    /// a CSV <c>1,7,12</c>.</para>
    /// Gated on <c>DOTLLM_GEMMA4_GGUF</c> + a Vulkan device. Always asserts
    /// " Paris" so green = correct, red = broken for that swap set.
    /// </summary>
    [SkippableFact]
    public unsafe void Gemma4_26B_HybridLayerSwap_Bisect()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, Gemma4Fixture.SkipMessage(KnownTestFixtures.Gemma4_26BDescription));
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");
        string spec = (Environment.GetEnvironmentVariable("DOTLLM_HYBRID_VK_LAYERS") ?? "none").Trim();
        string spvDir = ResolveSpvDir();

        // CPU model is the driver (embedding + per-layer + final norm + lm_head);
        // the Vulkan model supplies single-layer compute for the selected layers.
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path!);
        using var _g = gguf;
        using var _m = model;
        var cpu = (TransformerModel)model;
        int numLayers = config.NumLayers, hiddenSize = config.HiddenSize;

        using var vkGguf = GgufFile.Open(path!);
        var vkConfig = GgufModelConfigExtractor.Extract(vkGguf.Metadata);
        using var vk = VulkanTransformerModel.LoadFromGguf(vkGguf, vkConfig, spvDir);

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        const string prompt = "The Eiffel Tower is located in";
        int[] enc = tokenizer.Encode(prompt);
        int[] promptIds = new int[enc.Length + 1];
        promptIds[0] = tokenizer.BosTokenId;
        Array.Copy(enc, 0, promptIds, 1, enc.Length);
        int seqLen = promptIds.Length;
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        // Resolve which layers run on Vulkan.
        var vkLayers = new HashSet<int>();
        switch (spec.ToLowerInvariant())
        {
            case "none": break;
            case "all": for (int l = 0; l < numLayers; l++) vkLayers.Add(l); break;
            case "global": for (int l = 0; l < numLayers; l++) if (config.IsFullAttentionLayer(l)) vkLayers.Add(l); break;
            case "sliding": for (int l = 0; l < numLayers; l++) if (!config.IsFullAttentionLayer(l)) vkLayers.Add(l); break;
            default:
                if (spec.Contains('-') && !spec.Contains(','))
                {
                    var parts = spec.Split('-');
                    int a = int.Parse(parts[0]), b = int.Parse(parts[1]);
                    for (int l = a; l <= b && l < numLayers; l++) vkLayers.Add(l);
                }
                else
                {
                    foreach (var s in spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                        vkLayers.Add(int.Parse(s));
                }
                break;
        }

        _output.WriteLine($"[hybrid bisect] spec='{spec}' → {vkLayers.Count} VK layers: [{string.Join(",", vkLayers.OrderBy(x => x))}]");

        // When DOTLLM_HYBRID_VK_HEAD=1, also run the FINAL head (norm + lm_head +
        // softcap) on Vulkan, fed the post-last-layer residual stream. Pin down
        // whether bug #2 lives in the embedding or the final head: spec=all gives
        // a correct residual stream (CPU head ⇒ Paris); if the VK head on that
        // same stream gives the wrong token, the head is the culprit.
        bool vkHead = Environment.GetEnvironmentVariable("DOTLLM_HYBRID_VK_HEAD") == "1";
        bool vkEmbed = Environment.GetEnvironmentVariable("DOTLLM_HYBRID_VK_EMBED") == "1";
        float[]? finalHidden = null;

        // Optionally replace the CPU embedding with the Vulkan embedding (inject
        // it after EmbeddingLookup). Combined with spec=all + VK_HEAD this gives a
        // fully-Vulkan pipeline EXCEPT the per-layer execution stays per-submit —
        // so if it breaks 'Paris' the embedding is bug #2; if not, the native
        // pipelined (single-command-buffer) execution is.
        if (vkEmbed)
        {
            float[] vkEmb = vk.RunGemma4EmbeddingOnHost(promptIds, positions);
            cpu.Gemma4PostEmbeddingHook = (hidden, _, sl) =>
                vkEmb.AsSpan(0, sl * hiddenSize).CopyTo(new Span<float>(hidden, sl * hiddenSize));
        }

        if (vkLayers.Count > 0)
        {
            cpu.Gemma4LayerOverrideSelector = layer => vkLayers.Contains(layer);
            cpu.Gemma4LayerOverride = (hidden, layer, sl) =>
            {
                vk.RunGemma4LayerOnHost(new Span<float>(hidden, sl * hiddenSize), layer, sl, positions);
                if (layer == numLayers - 1)
                    finalHidden = new Span<float>(hidden, sl * hiddenSize).ToArray();
            };
        }

        using ITensor logits = cpu.Forward(promptIds, positions, deviceId: -1, kvCache: null,
            adapter: null, AttentionMaskSpec.Causal);
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        float* p = (float*)logits.DataPointer;
        long lastRow = (long)(seqLen - 1) * vocab;          // CPU returns [seqLen, vocab]
        int best = 0; float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++) { float x = p[lastRow + v]; if (x > bestV) { bestV = x; best = v; } }
        string decoded = tokenizer.Decode(new[] { best });
        bool isParis = decoded.Contains("Paris", StringComparison.OrdinalIgnoreCase);
        _output.WriteLine($"  [CPU head] next token : id={best} '{decoded}' logit={bestV:F3}  PARIS={(isParis ? "YES" : "no")}");

        if (vkHead)
        {
            Skip.If(finalHidden is null, "VK-head test needs the last layer on VK — run with DOTLLM_HYBRID_VK_LAYERS=all (or include the last layer).");
            float[] vkLogits = vk.RunGemma4FinalHeadOnHost(finalHidden, seqLen);
            int vb = 0; float vv = float.NegativeInfinity;
            for (int v = 0; v < vkLogits.Length; v++) if (vkLogits[v] > vv) { vv = vkLogits[v]; vb = v; }
            string vkDecoded = tokenizer.Decode(new[] { vb });
            bool vkParis = vkDecoded.Contains("Paris", StringComparison.OrdinalIgnoreCase);
            _output.WriteLine($"  [VK head ] next token : id={vb} '{vkDecoded}' logit={vv:F3}  PARIS={(vkParis ? "YES" : "no")}");
            Assert.True(vkParis, $"VK head on a correct residual stream: expected 'Paris', got '{vkDecoded}' (id={vb}, logit={vv:F3}). ⇒ the final norm/lm_head/softcap is bug #2.");
            return;
        }

        Assert.True(isParis, $"spec='{spec}': expected 'Paris', got '{decoded}' (id={best}, logit={bestV:F3}).");
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException("SPIR-V blobs not found. Run native/vulkan/build.ps1.");
    }
}
