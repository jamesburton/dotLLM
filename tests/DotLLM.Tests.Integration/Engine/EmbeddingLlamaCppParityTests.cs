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
/// External-anchor tests for the embeddings path (issue #451): dotLLM's
/// <see cref="IEmbeddingModel.ForwardHidden"/> + <see cref="EmbeddingPooler"/> against embeddings
/// captured from <b>llama.cpp's</b> <c>llama-server --embeddings</c> on the <b>same GGUF file</b>.
/// </summary>
/// <remarks>
/// <para><b>Why an external anchor.</b> Self-consistency — our pooling agreeing with our own
/// forward pass — cannot detect a systematically wrong hidden state: pooling the pre-norm
/// residual, stopping at the wrong layer, or pooling the wrong row would all still be
/// self-consistent. llama.cpp is the project's authoritative reference for GGUF op semantics, so
/// its vectors are the oracle. Reference data and full provenance (llama.cpp build, exact flags,
/// token ids) live in <c>Fixtures/Embeddings/llamacpp-smollm2-135m-instruct-q8_0.json</c>;
/// regenerate with <c>tests/scripts/capture-llamacpp-embeddings.ps1</c>.</para>
///
/// <para><b>The tolerance, and the measurements it was derived from.</b> The gate is
/// <see cref="CosineGate"/> = 0.9974 per vector (1 − cos ≤ 2.6e-3). It is placed at the
/// <i>log-midpoint</i> of the measured gap between correct and broken forms on this fixture
/// (SmolLM2-135M-Instruct Q8_0, 30 layers, hidden 576, two inputs of 10 and 11 tokens) — all of
/// which were measured <b>before</b> the number was chosen:</para>
/// <list type="table">
///   <listheader><term>form</term><description>cosine vs llama.cpp (per input)</description></listheader>
///   <item><term>correct — last</term><description>0.99985868 / 0.99975375</description></item>
///   <item><term>correct — mean</term><description>0.99946456 / 0.99993534</description></item>
///   <item><term>correct — cls</term><description>0.99998587 / 0.99998502</description></item>
///   <item><term>broken — mean over the first n−1 tokens</term><description>0.98717234 / 0.99604330</description></item>
///   <item><term>broken — last pooled at row n−2</term><description>0.69624046 / 0.62017976</description></item>
///   <item><term>broken — cls pooled at row 1</term><description>0.63925963 / 0.99954976</description></item>
///   <item><term>broken — pooled after N−1 layers</term><description>0.55710387 / 0.69294327</description></item>
///   <item><term>broken — pooled with the wrong mode</term><description>0.48373079 … 0.82652567</description></item>
/// </list>
/// <para>The worst correct value is 0.99946 (1 − cos = 5.35e-4). The worst-case-for-detection
/// broken value — the <i>minimum</i> a broken form scores across the inputs, since the test
/// asserts every input — is 0.98717 (1 − cos = 1.28e-2). √(5.35e-4 × 1.28e-2) = 2.6e-3, giving
/// ≈4.9× headroom on each side; a rounder 1e-3 would have left only 1.9× on the correct side, so
/// the extra margin is deliberate, not slack. Note the raw sizes: a single scalar threshold could
/// <i>not</i> separate "cls pooled at row 1" on input 1 (0.99955) from correct mean pooling on
/// input 0 (0.99946), which is precisely why the parity test asserts <b>every</b> input rather
/// than a mean or a best case — that broken form is caught on input 0 at 0.639.</para>
///
/// <para>Caveat on portability: all of these numbers were measured on one host (Zen 5 / AVX-512).
/// They are reproducible run-to-run there, and <see cref="Pooling_is_stable_under_single_threaded_execution"/>
/// measures the same six cosines with <c>ThreadingConfig.SingleThreaded</c> — they come out
/// <b>bit-identical</b> to the default-thread run, so thread count is ruled out as a source of
/// drift. A materially different ISA has not been measured.</para>
///
/// <para><b>Why the correct form is not exact.</b> Unproven but consistent with the data: dotLLM
/// quantises activations to Q8 for its GEMMs with its own block layout and rounding, so 30 layers
/// of Q8×Q8 accumulation would drift from llama.cpp's by a few parts in 1e3, and the residual does
/// grow with token position exactly as that would predict (cls, at position 0, is the closest at
/// 1.4e-5; last and mean, which see the whole sequence, are 10–40× worse). It is not thread-count
/// reduction order — that is measured and bit-identical. For scale, the repo's established cross-implementation
/// logit bound is 1 − cos ≤ 0.05 (<c>CrossBackendQuantGateTests.OneMinusCosineTolerance</c>);
/// this gate is 50× tighter than that.</para>
///
/// <para>The discrimination is asserted, not just documented, by
/// <see cref="Broken_forms_fall_below_the_gate"/> — the failure mode issues #417/#420/#421 were
/// about.</para>
/// </remarks>
public sealed class EmbeddingLlamaCppParityTests(ITestOutputHelper output)
{
    /// <summary>Per-vector cosine gate for the correct form. See the class remarks for the derivation.</summary>
    private const double CosineGate = 0.9974;

    private const string ModelDescription = "SmolLM2-135M-Instruct Q8_0 (embeddings anchor)";

    private static FixtureLocation ResolveModel() => TestFixtureResolver.ResolveFile(
        "DOTLLM_SMOLLM2_135M_INSTRUCT_Q8_GGUF",
        "bartowski", "SmolLM2-135M-Instruct-GGUF",
        "SmolLM2-135M-Instruct-Q8_0.gguf");

    private static JsonDocument LoadReference()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Embeddings",
            "llamacpp-smollm2-135m-instruct-q8_0.json");
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static float[] ReadVector(JsonElement array)
    {
        var v = new float[array.GetArrayLength()];
        int i = 0;
        foreach (var e in array.EnumerateArray())
            v[i++] = e.GetSingle();
        return v;
    }

    private static int[] ReadTokens(JsonElement array)
    {
        var v = new int[array.GetArrayLength()];
        int i = 0;
        foreach (var e in array.EnumerateArray())
            v[i++] = e.GetInt32();
        return v;
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
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

    /// <summary>
    /// The tokenizer must reproduce llama.cpp's token ids exactly. If it did not, every cosine
    /// below would be wrong for a reason that has nothing to do with the embeddings code — so this
    /// is asserted first and separately.
    /// </summary>
    [SkippableFact]
    public void Tokenization_matches_the_llamacpp_reference()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(ModelDescription));

        using var reference = LoadReference();
        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription, () => GgufFile.Open(loc.Path!));
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int index = 0;
        foreach (var input in reference.RootElement.GetProperty("inputs").EnumerateArray())
        {
            int[] expected = ReadTokens(input.GetProperty("tokens"));
            int[] actual = tokenizer.Encode(input.GetProperty("text").GetString()!);
            Assert.True(expected.SequenceEqual(actual),
                $"input[{index}] tokens differ from llama.cpp: expected [{string.Join(",", expected)}], got [{string.Join(",", actual)}].");
            index++;
        }
    }

    /// <summary>
    /// For each pooling mode, dotLLM's pooled + L2-normalised embedding must match llama.cpp's
    /// vector for the same text, on <b>every</b> input, to within <see cref="CosineGate"/>.
    /// </summary>
    [SkippableTheory]
    [InlineData("last", PoolingType.Last)]
    [InlineData("mean", PoolingType.Mean)]
    [InlineData("cls", PoolingType.Cls)]
    public void Pooling_matches_llamacpp_reference(string poolingName, PoolingType pooling)
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(ModelDescription));

        using var reference = LoadReference();
        var refPool = reference.RootElement.GetProperty("pooling").GetProperty(poolingName);
        var refEmbeddings = refPool.GetProperty("embeddings");

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription, () => TransformerModel.LoadFromGguf(gguf, config));

        int index = 0;
        int totalTokens = 0;
        foreach (var input in reference.RootElement.GetProperty("inputs").EnumerateArray())
        {
            int[] tokens = ReadTokens(input.GetProperty("tokens"));
            float[] expected = ReadVector(refEmbeddings[index]);
            Assert.Equal(config.HiddenSize, expected.Length);

            float[] actual = Embed(model, tokens, config.HiddenSize, pooling);
            double cosine = Cosine(actual, expected);
            output.WriteLine($"{poolingName} input[{index}]: cos={cosine:F9} (1-cos={1 - cosine:E3})");

            Assert.True(cosine >= CosineGate,
                $"{poolingName} pooling, input[{index}]: cosine vs llama.cpp = {cosine:F9}, gate {CosineGate}.");

            // llama.cpp's normalised vectors are unit-norm; ours must be too.
            double norm = Math.Sqrt(actual.Sum(x => (double)x * x));
            Assert.True(Math.Abs(norm - 1.0) < 1e-5, $"Normalised vector has norm {norm:F8}, expected 1.");

            totalTokens += tokens.Length;
            index++;
        }

        // usage.prompt_tokens is the sum of per-item token counts — llama.cpp reports the same.
        Assert.Equal(refPool.GetProperty("usage_prompt_tokens").GetInt32(), totalTokens);
    }

    /// <summary>
    /// The gate's margin on the correct side is set by float32 reduction order, which the CPU
    /// backend's thread count changes. Running the same forward single-threaded must still clear
    /// the gate — otherwise the tolerance is calibrated to this host's thread count rather than to
    /// the computation, and would be fragile on a machine with a different core count.
    /// </summary>
    [SkippableFact]
    public void Pooling_is_stable_under_single_threaded_execution()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(ModelDescription));

        using var reference = LoadReference();
        var root = reference.RootElement;

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription,
            () => TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));

        int index = 0;
        foreach (var input in root.GetProperty("inputs").EnumerateArray())
        {
            int[] tokens = ReadTokens(input.GetProperty("tokens"));
            foreach (var (name, pooling) in new (string, PoolingType)[]
                     { ("last", PoolingType.Last), ("mean", PoolingType.Mean), ("cls", PoolingType.Cls) })
            {
                float[] expected = ReadVector(root.GetProperty("pooling").GetProperty(name).GetProperty("embeddings")[index]);
                double cosine = Cosine(Embed(model, tokens, config.HiddenSize, pooling), expected);
                output.WriteLine($"single-threaded {name} input[{index}]: cos={cosine:F9}");
                Assert.True(cosine >= CosineGate,
                    $"single-threaded {name} pooling, input[{index}]: cosine {cosine:F9} below gate {CosineGate}.");
            }
            index++;
        }
    }

    /// <summary>
    /// Demonstrates that <see cref="CosineGate"/> actually discriminates. Each form below is a
    /// mistake this implementation could plausibly have made; each must score below the gate on
    /// at least one input, which is what makes <see cref="Pooling_matches_llamacpp_reference"/>
    /// (which asserts every input) fail for it.
    /// </summary>
    [SkippableFact]
    public void Broken_forms_fall_below_the_gate()
    {
        var loc = ResolveModel();
        Skip.If(!loc.Found, loc.SkipMessage(ModelDescription));

        using var reference = LoadReference();
        var root = reference.RootElement;

        using var gguf = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription, () => GgufFile.Open(loc.Path!));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CheckpointGuard.LoadOrSkip(loc.Path!, ModelDescription, () => TransformerModel.LoadFromGguf(gguf, config));

        int hidden = config.HiddenSize;
        int inputCount = root.GetProperty("inputs").GetArrayLength();

        // For each broken form, the MINIMUM cosine across inputs — that is the value that decides
        // whether the per-input parity assertion catches it.
        var worst = new Dictionary<string, double>(StringComparer.Ordinal);

        void Record(string form, double cosine)
        {
            worst[form] = worst.TryGetValue(form, out double prev) ? Math.Min(prev, cosine) : cosine;
        }

        for (int i = 0; i < inputCount; i++)
        {
            int[] tokens = ReadTokens(root.GetProperty("inputs")[i].GetProperty("tokens"));
            float[] refLast = ReadVector(root.GetProperty("pooling").GetProperty("last").GetProperty("embeddings")[i]);
            float[] refMean = ReadVector(root.GetProperty("pooling").GetProperty("mean").GetProperty("embeddings")[i]);
            float[] refCls = ReadVector(root.GetProperty("pooling").GetProperty("cls").GetProperty("embeddings")[i]);

            float[] ourLast = Embed(model, tokens, hidden, PoolingType.Last);
            float[] ourMean = Embed(model, tokens, hidden, PoolingType.Mean);
            float[] ourCls = Embed(model, tokens, hidden, PoolingType.Cls);

            // 1. Wrong pooling mode — what a mis-resolved default would produce.
            Record("wrong mode: last vs mean", Cosine(ourLast, refMean));
            Record("wrong mode: last vs cls", Cosine(ourLast, refCls));
            Record("wrong mode: mean vs last", Cosine(ourMean, refLast));
            Record("wrong mode: mean vs cls", Cosine(ourMean, refCls));
            Record("wrong mode: cls vs last", Cosine(ourCls, refLast));
            Record("wrong mode: cls vs mean", Cosine(ourCls, refMean));

            // 2. Off-by-one row selection — the subtlest realistic pooling bug.
            Record("off-by-one: last at row n-2", Cosine(PoolRow(model, tokens, hidden, tokens.Length - 2), refLast));
            Record("off-by-one: cls at row 1", Cosine(PoolRow(model, tokens, hidden, 1), refCls));

            // 3. Off-by-one token count in mean — drops the final token from the average.
            Record("off-by-one: mean over n-1 tokens", Cosine(PoolMeanPrefix(model, tokens, hidden, tokens.Length - 1), refMean));

            // 4. Wrong graph point — the hidden state taken before the last layer has run, standing
            //    in for "pooled something other than result_norm".
            Record("wrong graph point: N-1 layers", Cosine(EmbedTruncated(model, tokens, hidden, config.NumLayers - 1), refLast));
        }

        foreach (var (form, cosine) in worst.OrderBy(kv => kv.Value))
        {
            output.WriteLine($"broken form '{form}': worst-case cosine {cosine:F9}");
            Assert.True(cosine < CosineGate,
                $"Broken form '{form}' scored cosine {cosine:F9} on every input, which is NOT below the "
                + $"gate {CosineGate}. The parity test could therefore pass while this bug is present.");
        }

        // The un-normalised pooled vector must be far from unit norm, otherwise the parity test's
        // norm assertion could not detect a missing L2 normalisation on this fixture.
        int[] first = ReadTokens(root.GetProperty("inputs")[0].GetProperty("tokens"));
        double rawNorm = Math.Sqrt(EmbedRaw(model, first, hidden, PoolingType.Last).Sum(x => (double)x * x));
        output.WriteLine($"un-normalised pooled norm: {rawNorm:F4}");
        Assert.True(Math.Abs(rawNorm - 1.0) > 1.0,
            $"The un-normalised pooled vector has norm {rawNorm:F4}, too close to 1 for this fixture to "
            + "discriminate a missing L2 normalisation.");
    }

    private static float[] Embed(TransformerModel model, int[] tokens, int hiddenSize, PoolingType pooling)
    {
        float[] v = EmbedRaw(model, tokens, hiddenSize, pooling);
        EmbeddingPooler.L2Normalize(v);
        return v;
    }

    private static unsafe float[] EmbedRaw(TransformerModel model, int[] tokens, int hiddenSize, PoolingType pooling)
    {
        var vector = new float[hiddenSize];
        using var hidden = ForwardHidden(model, tokens);
        var span = new ReadOnlySpan<float>((void*)hidden.DataPointer, tokens.Length * hiddenSize);
        EmbeddingPooler.Pool(span, tokens.Length, hiddenSize, pooling, vector);
        return vector;
    }

    /// <summary>Deliberately-broken pooler: copies an arbitrary row instead of the pooled vector.</summary>
    private static unsafe float[] PoolRow(TransformerModel model, int[] tokens, int hiddenSize, int row)
    {
        var vector = new float[hiddenSize];
        using var hidden = ForwardHidden(model, tokens);
        new ReadOnlySpan<float>((void*)hidden.DataPointer, tokens.Length * hiddenSize)
            .Slice(row * hiddenSize, hiddenSize).CopyTo(vector);
        EmbeddingPooler.L2Normalize(vector);
        return vector;
    }

    /// <summary>Deliberately-broken pooler: means over only the first <paramref name="count"/> rows.</summary>
    private static unsafe float[] PoolMeanPrefix(TransformerModel model, int[] tokens, int hiddenSize, int count)
    {
        var vector = new float[hiddenSize];
        using var hidden = ForwardHidden(model, tokens);
        var span = new ReadOnlySpan<float>((void*)hidden.DataPointer, tokens.Length * hiddenSize);
        EmbeddingPooler.Pool(span, count, hiddenSize, PoolingType.Mean, vector);
        EmbeddingPooler.L2Normalize(vector);
        return vector;
    }

    /// <summary>Deliberately-broken forward: stops <c>NumLayers - layers</c> layers early.</summary>
    private static float[] EmbedTruncated(TransformerModel model, int[] tokens, int hiddenSize, int layers)
    {
        model.DebugMaxLayers = layers;
        try
        {
            return Embed(model, tokens, hiddenSize, PoolingType.Last);
        }
        finally
        {
            model.DebugMaxLayers = 0;
        }
    }

    private static Core.Tensors.ITensor ForwardHidden(TransformerModel model, int[] tokens)
    {
        var positions = new int[tokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        return ((IEmbeddingModel)model).ForwardHidden(tokens, positions, deviceId: 0);
    }
}
