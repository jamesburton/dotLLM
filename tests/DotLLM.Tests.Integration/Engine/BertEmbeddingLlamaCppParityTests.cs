using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// External-anchor parity for the BERT-class encoder (issue #739): dotLLM's CPU encoder +
/// pooling + L2 normalisation against embeddings captured from llama.cpp's
/// <c>llama-server --embeddings -ngl 0</c> on the <b>same GGUF</b> with the checkpoint's own
/// pooling type. Capture with <c>tests/scripts/capture_llamacpp_bert_embeddings.py</c>; provenance
/// (build, flags) is stored in each fixture.
/// </summary>
/// <remarks>
/// The forward check feeds llama.cpp's <i>captured token ids</i> into the model so a tokenizer
/// difference cannot be mistaken for a forward-pass error; tokenization is asserted separately
/// against the same captured ids.
/// </remarks>
public sealed class BertEmbeddingLlamaCppParityTests(ITestOutputHelper output)
{
    /// <summary>Per-vector cosine gate. See the per-model measurements in the PR / docs.</summary>
    private const double CosineGate = 0.9999;

    /// <summary>
    /// Per-quantization gates, set from the measured variance BETWEEN independent providers rather than
    /// guessed (#739). Worst-case cosine over the 9 test inputs, same GGUF, same texts, token ids identical
    /// across every provider (llama.cpp /tokenize; ollama prompt_eval_count = 191 for all):
    /// <code>
    ///                          llama.cpp CPU vs   dotLLM vs        dotLLM vs     inter-provider
    ///                          llama.cpp Vulkan   llama.cpp CPU    unquantised   floor (gate basis)
    /// minilm-l6-v2 Q8_0        0.999564           0.999668         0.999132 (f16 ref; llama 0.999019)
    /// mxbai-large  Q8_0        0.999750           0.999880         0.999710 (f32 ref; llama 0.999718)
    /// nomic-v1.5   Q8_0        0.998983           0.999392         0.997502 (f32 ref; llama 0.997484)
    /// nomic-v1.5   Q4_K_M      0.982358*          0.982278         0.912007 (f32 ref; llama 0.909099)
    /// </code>
    /// * llama.cpp b8683 CPU vs b9016 CPU is 0.985683 and b9016 CPU vs Vulkan 0.982358 on Q4_K_M, so the
    /// independent providers themselves disagree by as much as dotLLM does. llama.cpp builds b8683 / b9016 /
    /// b9672 / b9747 (CPU, --device none), b9016 and b9747 (Vulkan, -ngl 99) and ollama 0.33.1 (CPU) were run;
    /// ollama and the b9016+ CPU builds are bit-identical (same ggml CPU kernels) so they count once. dotLLM's
    /// error against the UNQUANTISED reference equals llama.cpp's to 5 digits, i.e. it is the same quantization
    /// noise, not an extra defect. Gates sit just under each floor. The f32/f16 gate (0.9999) is the control
    /// that could have disagreed: it is tight and passes, so a real forward-pass bug cannot hide here.
    /// </summary>
    private static double GateFor(string key) => key switch
    {
        "minilm-q8" => 0.9995,
        "mxbai-q8" => 0.9997,
        "nomic-q8" => 0.998,
        "nomic-q4km" => 0.98,
        _ => CosineGate,
    };

    public sealed record Spec(string Fixture, string Env, string Org, string Repo, string File, string Description);

    public static TheoryData<string> Models => new()
    {
        "minilm-f16", "mxbai-f32", "nomic-f32", "mxbai-f16", "nomic-f16",
        "minilm-q8", "mxbai-q8", "nomic-q8", "nomic-q4km",
    };

    internal static Spec SpecFor(string key) => key switch
    {
        "minilm-f16" => new("llamacpp-minilm-l6-v2-f16.json", "DOTLLM_MINILM_GGUF", "second-state",
            "All-MiniLM-L6-v2-Embedding-GGUF", "all-MiniLM-L6-v2-ggml-model-f16.gguf", "all-MiniLM-L6-v2 f16 (bert, mean)"),
        "mxbai-f32" => new("llamacpp-mxbai-embed-large-v1-fp32.json", "DOTLLM_MXBAI_F32_GGUF", "ChristianAzinn",
            "mxbai-embed-large-v1-gguf", "mxbai-embed-large-v1_fp32.gguf", "mxbai-embed-large-v1 fp32 (bert, cls)"),
        "mxbai-f16" => new("llamacpp-mxbai-embed-large-v1-fp16.json", "DOTLLM_MXBAI_F16_GGUF", "ChristianAzinn",
            "mxbai-embed-large-v1-gguf", "mxbai-embed-large-v1_fp16.gguf", "mxbai-embed-large-v1 fp16 (bert, cls)"),
        "nomic-f32" => new("llamacpp-nomic-embed-text-v1.5-f32.json", "DOTLLM_NOMIC_F32_GGUF", "nomic-ai",
            "nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.f32.gguf", "nomic-embed-text-v1.5 f32 (nomic-bert, mean)"),
        "nomic-f16" => new("llamacpp-nomic-embed-text-v1.5-f16.json", "DOTLLM_NOMIC_F16_GGUF", "nomic-ai",
            "nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.f16.gguf", "nomic-embed-text-v1.5 f16 (nomic-bert, mean)"),
        "minilm-q8" => new("llamacpp-minilm-l6-v2-q8_0.json", "DOTLLM_MINILM_Q8_GGUF", "second-state",
            "All-MiniLM-L6-v2-Embedding-GGUF", "all-MiniLM-L6-v2-Q8_0.gguf", "all-MiniLM-L6-v2 Q8_0 (bert, mean)"),
        "mxbai-q8" => new("llamacpp-mxbai-embed-large-v1-q8_0.json", "DOTLLM_MXBAI_Q8_GGUF", "ChristianAzinn",
            "mxbai-embed-large-v1-gguf", "mxbai-embed-large-v1.Q8_0.gguf", "mxbai-embed-large-v1 Q8_0 (bert, cls)"),
        "nomic-q8" => new("llamacpp-nomic-embed-text-v1.5-q8_0.json", "DOTLLM_NOMIC_Q8_GGUF", "nomic-ai",
            "nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q8_0.gguf", "nomic-embed-text-v1.5 Q8_0 (nomic-bert, mean)"),
        "nomic-q4km" => new("llamacpp-nomic-embed-text-v1.5-q4_k_m.json", "DOTLLM_NOMIC_Q4KM_GGUF", "nomic-ai",
            "nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-embed-text-v1.5 Q4_K_M (nomic-bert, mean)"),
        _ => throw new ArgumentOutOfRangeException(nameof(key)),
    };

    private static JsonDocument LoadReference(Spec s)
        => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Embeddings", s.Fixture)));

    private static (string Text, int[] Tokens, float[] Embedding)[] ReadInputs(JsonDocument doc)
    {
        var inputs = doc.RootElement.GetProperty("inputs").EnumerateArray().ToArray();
        var embs = doc.RootElement.GetProperty("embeddings").EnumerateArray().ToArray();
        return inputs.Select((e, i) => (
            e.GetProperty("text").GetString()!,
            e.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            embs[i].EnumerateArray().Select(x => x.GetSingle()).ToArray())).ToArray();
    }

    internal static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static unsafe float[] Embed(
        IEmbeddingModel model, int[] tokens, int hidden, PoolingType pooling, int[]? positions = null)
    {
        positions ??= Enumerable.Range(0, tokens.Length).ToArray();
        using var h = model.ForwardHidden(tokens, positions, deviceId: 0);
        var span = new ReadOnlySpan<float>((void*)h.DataPointer, tokens.Length * hidden);
        var v = new float[hidden];
        EmbeddingPooler.Pool(span, tokens.Length, hidden, pooling, v);
        EmbeddingPooler.L2Normalize(v);
        return v;
    }

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void Tokenization_matches_llamacpp(string key)
    {
        var spec = SpecFor(key);
        var loc = TestFixtureResolver.ResolveFile(spec.Env, spec.Org, spec.Repo, spec.File);
        Skip.If(!loc.Found, loc.SkipMessage(spec.Description));
        using var doc = LoadReference(spec);
        using var gguf = GgufFile.Open(loc.Path!);
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);

        foreach (var (text, tokens, _) in ReadInputs(doc))
            Assert.True(tokens.SequenceEqual(tokenizer.Encode(text)), $"token ids differ from llama.cpp for '{text}'");
    }

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void Encoder_matches_llamacpp(string key)
    {
        var spec = SpecFor(key);
        var loc = TestFixtureResolver.ResolveFile(spec.Env, spec.Org, spec.Repo, spec.File);
        Skip.If(!loc.Found, loc.SkipMessage(spec.Description));

        using var doc = LoadReference(spec);
        using var gguf = GgufFile.Open(loc.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = BertEncoderModel.LoadFromGguf(gguf, config);
        var pooling = EmbeddingPooler.Resolve(null, model.DeclaredPoolingType);
        output.WriteLine($"{spec.Description}: pooling={pooling}, eps={config.NormEpsilon:E1}");

        double worst = 1;
        int i = 0;
        foreach (var (text, tokens, expected) in ReadInputs(doc))
        {
            var actual = Embed(model, tokens, config.HiddenSize, pooling);
            double cos = Cosine(actual, expected);
            worst = Math.Min(worst, cos);
            output.WriteLine($"  input[{i}] ({tokens.Length} tok): cos={cos:F9}");
            double gate = GateFor(key);
            Assert.True(cos >= gate, $"input[{i}] '{text[..Math.Min(30, text.Length)]}': cosine {cos:F9} < {gate}");
            i++;
        }
        output.WriteLine($"  worst cosine {worst:F9}");
    }

    /// <summary>
    /// The gate must be able to fail: each deliberately broken form has to land below
    /// <see cref="CosineGate"/> on at least one input (measured and printed).
    /// </summary>
    [SkippableTheory]
    [InlineData("minilm-f16")]
    [InlineData("nomic-f16")]
    public void Broken_forms_fall_below_the_gate(string key)
    {
        var spec = SpecFor(key);
        var loc = TestFixtureResolver.ResolveFile(spec.Env, spec.Org, spec.Repo, spec.File);
        Skip.If(!loc.Found, loc.SkipMessage(spec.Description));

        using var doc = LoadReference(spec);
        using var gguf = GgufFile.Open(loc.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = BertEncoderModel.LoadFromGguf(gguf, config);
        var declared = EmbeddingPooler.Resolve(null, model.DeclaredPoolingType);
        var inputs = ReadInputs(doc);

        double Min(Func<int[], float[]> embed)
            => inputs.Min(x => Cosine(embed(x.Tokens), x.Embedding));

        // Correct form clears the gate.
        double correct = Min(t => Embed(model, t, config.HiddenSize, declared));
        output.WriteLine($"correct: min cos {correct:F9}");
        Assert.True(correct >= CosineGate);

        // Wrong pooling mode.
        var other = declared == PoolingType.Mean ? PoolingType.Cls : PoolingType.Mean;
        double wrongPool = Min(t => Embed(model, t, config.HiddenSize, other));
        output.WriteLine($"wrong pooling ({other}): min cos {wrongPool:F9}");
        Assert.True(wrongPool < CosineGate);

        // Pooling that drops the last row ([SEP]) — an off-by-one in the pooled row set. Only
        // discriminates for mean pooling (CLS reads row 0).
        if (declared == PoolingType.Mean)
        {
            double dropSep = Min(t =>
            {
                using var hs = model.ForwardHidden(t, Enumerable.Range(0, t.Length).ToArray(), 0);
                return PoolDropLast(hs, t.Length, config.HiddenSize, declared);
            });
            output.WriteLine($"pooling without the last row: min cos {dropSep:F9}");
            Assert.True(dropSep < CosineGate);
        }

        // Shifted positions — a position-embedding off-by-one. Only absolute-position models are
        // sensitive: RoPE sees relative offsets, so a uniform shift is invisible to nomic-bert.
        if (config.Architecture == Architecture.Bert)
        {
            double shifted = Min(t => Embed(model, t, config.HiddenSize, declared,
                Enumerable.Range(1, t.Length).ToArray()));
            output.WriteLine($"positions shifted by 1: min cos {shifted:F9}");
            Assert.True(shifted < CosineGate);
        }
    }

    private static unsafe float[] PoolDropLast(Core.Tensors.ITensor h, int n, int hidden, PoolingType pooling)
    {
        var span = new ReadOnlySpan<float>((void*)h.DataPointer, n * hidden);
        var v = new float[hidden];
        EmbeddingPooler.Pool(span, Math.Max(1, n - 1), hidden, pooling, v);
        EmbeddingPooler.L2Normalize(v);
        return v;
    }
}
