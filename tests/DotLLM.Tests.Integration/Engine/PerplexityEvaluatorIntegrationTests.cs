using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration tests anchoring <see cref="PerplexityEvaluator"/> against a trusted reference
/// and validating it on a real GGUF model.
/// </summary>
/// <remarks>
/// <para><b>Anchor 1 (always-on, PyTorch reference).</b>
/// <see cref="MeanNll_FromPyTorchReferenceLogits_MatchesPrecomputed"/> reads the committed
/// <c>qwen2.5-0.5b-reference.json</c> — which is HuggingFace <c>AutoModelForCausalLM</c> bf16
/// output for "The capital of France is" — and feeds those reference logits straight into the
/// SAME <see cref="PerplexityEvaluator.ComputeWindowNll"/> that the CLI uses. It asserts the
/// harness reproduces the per-token NLL/perplexity computed independently in numpy
/// (meanNLL ≈ 3.3018, ppl ≈ 27.16 over 4 scored tokens). This proves the shipped NLL math is
/// correct against a PyTorch oracle, runs green in CI, and needs no model checkpoint.</para>
/// <para><b>Anchor 2 (plausibility on SmolLM-135M Q8_0).</b>
/// <see cref="Perplexity_SmolLM135M_Q80_OnCleanEnglish_IsPlausible"/> runs dotLLM's own forward
/// pass over a clean English passage and asserts the perplexity lands in a generous plausibility
/// band — wide enough never to flake, tight enough to catch structural breakage (leakage gives
/// ppl ≈ 1; a broken target/vocab gives thousands).</para>
/// </remarks>
[Collection("SmallModel")]
public sealed class PerplexityEvaluatorIntegrationTests
{
    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public PerplexityEvaluatorIntegrationTests(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    /// <summary>
    /// PyTorch anchor: NLL/perplexity computed from the committed HF bf16 reference logits must
    /// match the value computed independently in numpy. Tolerance is tight (the input is the same
    /// float data) but not bit-exact (f32 round-trip through JSON + double accumulation).
    /// </summary>
    [Fact]
    public void MeanNll_FromPyTorchReferenceLogits_MatchesPrecomputed()
    {
        string? path = ResolveReferenceJsonPath("qwen2.5-0.5b-reference.json");
        Assert.True(path is not null && File.Exists(path),
            "qwen2.5-0.5b-reference.json should be committed under "
            + "tests/DotLLM.Tests.Integration/Models/Loaders/references/.");

        var (inputIds, logits, vocab) = LoadReferenceLogits(path!);
        int rows = inputIds.Length;
        Assert.Equal(rows, logits.Length / vocab);

        var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(logits, rows, vocab, inputIds);
        double meanNll = sumNll / scored;
        double ppl = Math.Exp(meanNll);

        _output.WriteLine($"input_ids=[{string.Join(", ", inputIds)}] scored={scored}");
        _output.WriteLine($"sumNll={sumNll:F10} meanNll={meanNll:F10} ppl={ppl:F10}");

        // input_ids = [785, 6722, 315, 9625, 374] → 4 scored targets.
        Assert.Equal(4, scored);
        // Precomputed in double precision (numpy) over the same reference logits:
        //   sumNll = 13.2073407173, meanNll = 3.3018351793, ppl = 27.1624411594.
        Assert.Equal(3.3018351793, meanNll, precision: 4);
        Assert.Equal(27.1624411594, ppl, precision: 2);
    }

    /// <summary>
    /// <para>Directly-preferred anchor (gated on the Qwen2.5-0.5B safetensors checkpoint):
    /// run dotLLM's OWN forward pass over the reference token IDs, feed its logits into the same
    /// <see cref="PerplexityEvaluator.ComputeWindowNll"/>, and assert the resulting per-token NLL
    /// tracks the PyTorch reference value (meanNLL ≈ 3.3018, ppl ≈ 27.16) within a sane band.</para>
    /// <para>This closes the loop the always-on reference-logits test leaves transitive: it confirms
    /// dotLLM's forward + the harness together reproduce the oracle, NOT bit-exact (Q-vs-bf16 logit
    /// drift of ~1.0 perturbs NLL) but within tolerance. Skips cleanly when the checkpoint is absent,
    /// in which case the transitive argument (Qwen25_0_5B_LogitsMatchPyTorchReference proves logits
    /// match) plus the always-on math anchor still stand.</para>
    /// </summary>
    [Fact]
    public void Perplexity_Qwen25_0_5B_OwnForward_TracksPyTorchReference()
    {
        string? root = ResolveCheckpointRoot(
            "DOTLLM_QWEN25_CHECKPOINT_PATH",
            "C:/Users/james/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987");
        if (root is null)
        {
            _output.WriteLine(
                "[SKIP] Qwen2.5-0.5B checkpoint not found. Set DOTLLM_QWEN25_CHECKPOINT_PATH or place "
                + "the HF snapshot at the conventional path. The always-on reference-logits NLL anchor "
                + "and the transitive Qwen25_0_5B_LogitsMatchPyTorchReference test cover correctness "
                + "in the meantime.");
            return;
        }

        string? refPath = ResolveReferenceJsonPath("qwen2.5-0.5b-reference.json");
        Assert.True(refPath is not null && File.Exists(refPath));
        var (inputIds, _, vocab) = LoadReferenceLogits(refPath!);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(root);
        try
        {
            Assert.Equal(Architecture.Qwen, config.Architecture);
            Assert.Equal(vocab, config.VocabSize);

            int[] positions = new int[inputIds.Length];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;

            using ITensor logits = model.Forward(inputIds, positions, deviceId: -1);
            Assert.Equal(inputIds.Length, logits.Shape[0]);

            double meanNll, ppl;
            unsafe
            {
                var span = new ReadOnlySpan<float>(
                    (void*)logits.DataPointer, inputIds.Length * config.VocabSize);
                var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(
                    span, inputIds.Length, config.VocabSize, inputIds);
                meanNll = sumNll / scored;
                ppl = Math.Exp(meanNll);
                Assert.Equal(inputIds.Length - 1, scored);
            }

            _output.WriteLine(
                $"dotLLM Qwen2.5-0.5B own-forward: meanNll={meanNll:F6} ppl={ppl:F4} "
                + "(PyTorch reference: meanNll=3.3018, ppl=27.16)");

            // Q-vs-bf16 logit drift (~1.0 max-abs on this prompt) perturbs NLL; ±0.5 nats / ppl
            // within ~6 of 27.16 is a sane "tracks the reference" band — tight enough that a real
            // forward-pass regression (wrong shift, transposed weight) blows it out by an order
            // of magnitude.
            Assert.InRange(meanNll, 3.3018 - 0.5, 3.3018 + 0.5);
            Assert.InRange(ppl, 21.0, 33.5);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    /// <summary>
    /// Plausibility anchor: SmolLM-135M Q8_0 over a clean English paragraph should produce a
    /// low–mid-tens-ish perplexity. The band (2, 300) is deliberately generous: its job is to
    /// catch structural breakage (residual leakage ⇒ ≈1; wrong shift/target/vocab ⇒ thousands),
    /// not to pin a precise value.
    /// </summary>
    [Fact]
    public void Perplexity_SmolLM135M_Q80_OnCleanEnglish_IsPlausible()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        // A short, coherent English passage. Kept well under SmolLM's context window.
        const string passage =
            "The quick brown fox jumps over the lazy dog. "
            + "Paris is the capital of France, and London is the capital of England. "
            + "Water boils at one hundred degrees Celsius at sea level. "
            + "The sun rises in the east and sets in the west every single day.";

        int[] tokenIds = tokenizer.Encode(passage);
        Assert.True(tokenIds.Length >= 2, "Passage should tokenize to at least 2 tokens.");
        Assert.True(tokenIds.Length < config.MaxSequenceLength,
            $"Passage ({tokenIds.Length} tokens) should fit the {config.MaxSequenceLength}-token context.");

        var result = PerplexityEvaluator.Evaluate(model, tokenIds);

        _output.WriteLine(
            $"SmolLM-135M Q8_0: ppl={result.Perplexity:F4} meanNll={result.MeanNll:F6} "
            + $"scored={result.ScoredTokenCount} total={result.TotalTokenCount}");

        Assert.Equal(tokenIds.Length, result.TotalTokenCount);
        Assert.Equal(tokenIds.Length - 1, result.ScoredTokenCount);
        Assert.True(double.IsFinite(result.Perplexity), "Perplexity must be finite.");
        Assert.InRange(result.Perplexity, 2.0, 300.0);
    }

    /// <summary>
    /// Cross-check: the per-window guard fires when the forward pass returns the right shape,
    /// and Evaluate's single-pass result equals the manual ComputeWindowNll over the same logits
    /// — confirming Evaluate and the pure function agree for a short, single-window corpus.
    /// </summary>
    [Fact]
    public void Evaluate_SingleWindow_MatchesManualWindowNll()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] tokenIds = tokenizer.Encode("The capital of France is Paris.");
        Assert.True(tokenIds.Length is >= 2 and < 256);

        var result = PerplexityEvaluator.Evaluate(model, tokenIds);

        // Manual forward + ComputeWindowNll over the same single window.
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        using var logits = model.Forward(tokenIds, positions, deviceId: -1);
        unsafe
        {
            var span = new ReadOnlySpan<float>(
                (void*)logits.DataPointer, tokenIds.Length * config.VocabSize);
            var (sumNll, scored) = PerplexityEvaluator.ComputeWindowNll(
                span, tokenIds.Length, config.VocabSize, tokenIds);
            double meanNll = sumNll / scored;

            Assert.Equal(scored, result.ScoredTokenCount);
            Assert.Equal(meanNll, result.MeanNll, precision: 6);
            Assert.Equal(Math.Exp(meanNll), result.Perplexity, precision: 4);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static string? ResolveReferenceJsonPath(string fileName)
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            string candidate = Path.Combine(
                dir, "tests", "DotLLM.Tests.Integration", "Models", "Loaders", "references", fileName);
            if (File.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    private static string? ResolveCheckpointRoot(string envVar, string conventional)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(env) && ContainsSafetensorsCheckpoint(env))
            return env;
        if (ContainsSafetensorsCheckpoint(conventional))
            return conventional;
        return null;
    }

    private static bool ContainsSafetensorsCheckpoint(string path)
    {
        if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            return true;
        if (!Directory.Exists(path)) return false;
        if (File.Exists(Path.Combine(path, "model.safetensors"))) return true;
        if (File.Exists(Path.Combine(path, "model.safetensors.index.json"))) return true;
        return Directory.GetFiles(path, "model-*-of-*.safetensors").Length > 0;
    }

    private static (int[] InputIds, float[] Logits, int Vocab) LoadReferenceLogits(string path)
    {
        using FileStream fs = File.OpenRead(path);
        using var doc = JsonDocument.Parse(fs);
        var root = doc.RootElement;

        var idsEl = root.GetProperty("input_ids");
        int[] inputIds = new int[idsEl.GetArrayLength()];
        int k = 0;
        foreach (var e in idsEl.EnumerateArray()) inputIds[k++] = e.GetInt32();

        var shapeEl = root.GetProperty("logits_shape");
        int seqLen = shapeEl[0].GetInt32();
        int vocab = shapeEl[1].GetInt32();

        var logitsEl = root.GetProperty("logits");
        float[] flat = new float[(long)seqLen * vocab];
        int idx = 0;
        foreach (var rowEl in logitsEl.EnumerateArray())
            foreach (var cell in rowEl.EnumerateArray())
                flat[idx++] = (float)cell.GetDouble();

        return (inputIds, flat, vocab);
    }
}
