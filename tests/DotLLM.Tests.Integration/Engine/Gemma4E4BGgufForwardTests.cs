using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Real-model validation of the Gemma-4 <b>dense-PLE</b> (E4B) GGUF path on the CPU
/// backend — the issue-#136 acceptance gate. The released
/// <c>unsloth/gemma-4-E4B-it</c> GGUF declares <c>general.architecture = gemma4</c>
/// with NO experts and the full dense-PLE feature set this repo wired for #136:
/// Per-Layer Embeddings (<c>per_layer_token_embd/model_proj/proj_norm</c> +
/// per-layer <c>inp_gate/proj/post_norm</c>), 18 trailing shared-KV layers
/// (<c>attention.shared_kv_layers</c>), dual head dim 256/512, full-dim dual RoPE
/// with a <c>rope_freqs</c> proportional-factor tensor, per-layer
/// <c>layer_output_scale</c>, and the 30.0 final soft-cap. AltUp / Laurel /
/// activation-sparsity are Gemma-3n-era components the gemma4 arch dropped — the
/// GGUF carries no such tensors (verified against llama.cpp <c>gemma4.cpp</c>).
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_GEMMA4E4B_GGUF</c> (path to the .gguf); falls back to the
/// canonical HF-hub cache path when unset. Skipped when neither resolves.
/// Real-model runs are heavy — acquire the repo bench lock before running locally.
/// </remarks>
public sealed class Gemma4E4BGgufForwardTests
{
    private const string ModelPathEnvVar = "DOTLLM_GEMMA4E4B_GGUF";
    private const string DefaultHubPath =
        "C:/Users/james/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/" +
        "snapshots/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q4_K_M.gguf";

    private readonly ITestOutputHelper _output;

    public Gemma4E4BGgufForwardTests(ITestOutputHelper output) => _output = output;

    private static string? TryResolveModelPath()
    {
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(path) && File.Exists(path)) return path;
        return File.Exists(DefaultHubPath) ? DefaultHubPath : null;
    }

    [SkippableFact]
    public unsafe void Gemma4E4B_GreedyDecode_CompletesCapitalOfFranceWithParis()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, $"Set {ModelPathEnvVar} to the gemma-4-E4B-it GGUF to run this validation.");

        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path!);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        loadSw.Stop();
        using var _ = gguf;
        using var __ = model;

        // Architecture sanity: dense-PLE shape (no experts, PLE + shared KV present).
        Assert.Equal(Architecture.Gemma4, config.Architecture);
        Assert.Null(config.Moe);
        Assert.NotNull(config.PerLayerEmbedding);
        Assert.Equal(256, config.PerLayerEmbedding!.PerLayerDim);
        Assert.Equal(18, config.NumSharedKvLayers);
        Assert.Equal(256, config.HeadDim);
        Assert.Equal(512, config.GlobalHeadDim);
        Assert.Null(config.PartialRotaryFactor);

        _output.WriteLine($"[gemma4 E4B GGUF] {path}");
        _output.WriteLine($"  load wall  : {loadSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  layers     : {config.NumLayers} hidden {config.HiddenSize} vocab {config.VocabSize}");
        _output.WriteLine($"  ple dim    : {config.PerLayerEmbedding.PerLayerDim}  shared-kv layers: {config.NumSharedKvLayers}");

        // "The capital of France is" — llama.cpp llama-tokenize ids for this file:
        // [2(<bos>), 818(The), 5279( capital), 529( of), 7001( France), 563( is)].
        // The gemma4 tokenizer is merge-driven BPE; dotLLM currently encodes it via
        // the SentencePiece longest-match fallback — assert parity on this prompt
        // so silent tokenizer drift fails loudly.
        const string prompt = "The capital of France is";
        int[] encoded = tokenizer.Encode(prompt);
        int[] promptIds = new int[encoded.Length + 1];
        promptIds[0] = tokenizer.BosTokenId;
        Array.Copy(encoded, 0, promptIds, 1, encoded.Length);
        _output.WriteLine($"  prompt ids : [{string.Join(",", promptIds)}]");
        Assert.Equal(new[] { 2, 818, 5279, 529, 7001, 563 }, promptIds);

        // Greedy decode a few tokens through the per-layer-strided KV cache
        // (exercises the shared-KV donor reads on the cached decode path too).
        const int maxNew = 6;
        var generated = new List<int>();
        var sw = Stopwatch.StartNew();
        using (var kv = new SimpleKvCache(KvGeometry.FromConfig(config), maxSeqLen: promptIds.Length + maxNew + 2))
        {
            int[] pos = new int[promptIds.Length];
            for (int i = 0; i < pos.Length; i++) pos[i] = i;

            int next;
            using (ITensor logits = model.Forward(promptIds, pos, deviceId: -1, kvCache: kv))
                next = ArgMaxLastRow(logits, config.VocabSize);
            generated.Add(next);

            for (int stepIdx = 1; stepIdx < maxNew; stepIdx++)
            {
                int[] one = [generated[^1]];
                int[] onePos = [promptIds.Length + stepIdx - 1];
                using ITensor logits = model.Forward(one, onePos, deviceId: -1, kvCache: kv);
                next = ArgMaxLastRow(logits, config.VocabSize);
                generated.Add(next);
                if (next == tokenizer.EosTokenId) break;
            }
        }
        sw.Stop();

        string completion = tokenizer.Decode(generated.ToArray());
        _output.WriteLine($"  decode     : {sw.Elapsed.TotalSeconds:F2} s for {generated.Count} tokens");
        _output.WriteLine($"  generated  : [{string.Join(",", generated)}] '{completion}'");

        // " Paris" (id 9079) must lead the greedy completion — a broken PLE /
        // shared-KV / rope-factor wiring produces incoherent output, not the
        // sharp, specific, correct fact.
        Assert.Contains("Paris", completion, StringComparison.OrdinalIgnoreCase);
    }

    private static unsafe int ArgMaxLastRow(ITensor logits, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        long lastRow = (long)(logits.Shape[0] - 1) * vocab;
        int best = 0;
        float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float x = p[lastRow + v];
            if (x > bestV) { bestV = x; best = v; }
        }
        return best;
    }
}
