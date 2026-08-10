using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using System.Numerics.Tensors;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Quantitative, end-to-end (model-level) accuracy gate for the BitNet b1.58 (I2_S
/// ternary) forward pass. Complements the qualitative <see cref="BitNetForwardPassTests"/>
/// (predicts "Paris") and the kernel-level <c>CudaI2SGemvTest</c> GEMV parity:
/// <list type="number">
/// <item><b>CPU perplexity</b> on a fixed English passage — a single teacher-forced
/// prefill (CPU <c>Forward</c> returns all <c>[seqLen, vocab]</c> rows, so one pass
/// scores every next-token NLL). A sane perplexity for a capable 2B model on ordinary
/// prose is the load-bearing quantitative number; it is also the CPU golden reference
/// the upcoming Vulkan/iGPU I2_S kernel is validated against.</item>
/// <item><b>CPU↔CUDA last-token logits parity</b> — the two I2_S forward implementations
/// are independent, so end-to-end agreement (argmax match + cosine &gt; 0.999) is a strong
/// correctness signal beyond the per-GEMV parity already covered elsewhere.</item>
/// </list>
/// </summary>
/// <remarks>
/// Plain <c>[Fact]</c> with early-return no-op when <c>DOTLLM_BITNET_GGUF</c> is unset/missing
/// (and CUDA-unavailable for the parity fact) — no <c>SkippableFact</c> dependency. Set
/// <c>DOTLLM_BITNET_GGUF</c> to the i2_s GGUF; on a box where <c>~/.dotllm</c> is a junction
/// the SSH/test process can't traverse, point it at the direct path (e.g. T5500's
/// <c>E:\.dotllm\test-cache\microsoft\bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf</c>).
/// Kept in its own class to avoid cross-class GPU parallelism.
/// </remarks>
public sealed class BitNetAccuracyTests
{
    private readonly ITestOutputHelper _output;

    public BitNetAccuracyTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// BitNet I2_S fixture, resolved via <see cref="KnownTestFixtures.BitNetI2S"/>:
    /// <c>$DOTLLM_BITNET_GGUF</c>, then the dotLLM test cache, then the HF hub cache (#308).
    /// </summary>
    private static FixtureLocation BitNetFixture => KnownTestFixtures.BitNetI2S;

    private static string? ModelPath => BitNetFixture.Path;

    // ~120 tokens of ordinary English. Absolute perplexity here is the quantitative
    // accuracy signal for the CPU I2_S forward; the same prompt's last-token logits
    // are the CPU reference the CUDA (and later Vulkan) paths are compared against.
    private const string Passage =
        "The history of natural language processing began in the nineteen fifties, when researchers "
        + "first attempted to translate text between human languages using simple rule based systems. "
        + "Over the following decades, statistical methods gradually replaced handwritten rules, and the "
        + "introduction of neural networks transformed the field once more. Today, large language models "
        + "are trained on vast collections of text drawn from books, articles, and conversations, learning "
        + "to predict the next word from everything that came before it.";

    private const string ParityPrompt = "The capital of France is";

    [SkippableFact]
    public unsafe void Cpu_Perplexity_OnFixedPassage_IsSane()
    {
        Skip.If(!BitNetFixture.Found, BitNetFixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));

        using var gguf = GgufFile.Open(ModelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        int[] tokenIds = tokenizer.Encode(Passage);
        Assert.True(tokenIds.Length >= 32, $"passage too short ({tokenIds.Length} tokens).");
        int[] positions = Positions(tokenIds.Length);
        int vocab = config.VocabSize;

        // Single teacher-forced prefill: row t's logits = P(next | tokens[0..t]) under the
        // causal mask, so we score the NLL of tokens[t+1] from every row in one forward.
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(vocab, logits.Shape[1]);

        double sumNll = 0;
        int scored = 0;
        for (int t = 0; t < tokenIds.Length - 1; t++)
        {
            var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)t * vocab, vocab);
            sumNll += -StableLogProb(row, tokenIds[t + 1]);
            scored++;
        }
        double perplexity = Math.Exp(sumNll / scored);

        _output.WriteLine($"BitNet b1.58 2B4T (I2_S) CPU perplexity over {scored} scored tokens: {perplexity:F4}");
        _output.WriteLine($"  (mean NLL = {sumNll / scored:F4} nats)");

        Assert.True(double.IsFinite(perplexity), $"perplexity is not finite ({perplexity}).");
        // Sanity gate: a capable 2B model on ordinary English teacher-forced should sit in
        // the low single-to-low-double digits. A value above this bound (or below 1) means
        // the I2_S forward is broken (e.g. scale/packing/sub-norm/activation error), not just
        // quantization noise. The printed number is the reported quantitative result.
        Assert.InRange(perplexity, 1.0, 30.0);
    }

    [SkippableFact]
    public unsafe void CpuVsCuda_LastTokenLogits_Match()
    {
        Skip.If(!BitNetFixture.Found, BitNetFixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));
        Skip.If(!CudaDevice.IsAvailable(),
            "No CUDA device available.");

        using var gguf = GgufFile.Open(ModelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;

        int[] tokenIds = tokenizer.Encode(ParityPrompt);
        int[] positions = Positions(tokenIds.Length);

        // ── CPU reference (golden) ──
        float[] cpuVec = new float[vocab];
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, config))
        using (ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1))
        {
            long lastRow = (long)(tokenIds.Length - 1) * vocab;
            new ReadOnlySpan<float>((float*)cpuLogits.DataPointer + lastRow, vocab)
                .CopyTo(cpuVec);
        }

        // ── CUDA (system under test) ── returns last-token row only, shape [1, vocab].
        float[] gpuVec = new float[vocab];
        using (var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0))
        using (ITensor gpuLogits = gpuModel.Forward(tokenIds, positions, deviceId: 0))
        {
            new ReadOnlySpan<float>((float*)gpuLogits.DataPointer, vocab).CopyTo(gpuVec);
        }

        int cpuArgmax = ArgMax(cpuVec);
        int gpuArgmax = ArgMax(gpuVec);
        double cosine = CosineSimilarity(cpuVec, gpuVec);
        (float maxAbs, float meanAbs) = AbsDiff(cpuVec, gpuVec);

        _output.WriteLine($"CPU argmax={cpuArgmax} ('{tokenizer.DecodeToken(cpuArgmax).Trim()}')  "
            + $"CUDA argmax={gpuArgmax} ('{tokenizer.DecodeToken(gpuArgmax).Trim()}')");
        _output.WriteLine($"cosine(cpu, cuda)={cosine:F6}  max|Δ|={maxAbs:F4}  mean|Δ|={meanAbs:F4}");

        // The CPU (FP32) and CUDA (FP16-compute) I2_S forwards are independent implementations;
        // FP16 makes exact equality impossible, but the next-token decision and overall logit
        // direction must agree. Argmax match + cosine > 0.999 is the same gate the BitNet LoRA
        // parity test uses.
        Assert.True(cpuArgmax == gpuArgmax,
            $"BitNet CPU/CUDA argmax mismatch: CPU={cpuArgmax} CUDA={gpuArgmax} (cosine={cosine:F6}). "
            + "Indicates an I2_S forward divergence between the CPU and CUDA paths.");
        Assert.True(cosine > 0.999,
            $"BitNet CPU/CUDA cosine {cosine:F6} below 0.999 — significant end-to-end divergence.");
    }

    // ── Helpers ──

    private static int[] Positions(int count)
    {
        int[] p = new int[count];
        for (int i = 0; i < count; i++) p[i] = i;
        return p;
    }

    private static double StableLogProb(ReadOnlySpan<float> row, int target)
    {
        float max = row[0];
        for (int j = 1; j < row.Length; j++)
            if (row[j] > max) max = row[j];
        double sumExp = 0;
        for (int j = 0; j < row.Length; j++)
            sumExp += Math.Exp(row[j] - max);
        return (row[target] - max) - Math.Log(sumExp);
    }

    private static int ArgMax(float[] vec)
        => TensorPrimitives.IndexOfMax(new ReadOnlySpan<float>(vec));

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }

    private static (float maxAbs, float meanAbs) AbsDiff(float[] a, float[] b)
    {
        float maxAbs = 0;
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxAbs) maxAbs = d;
            sum += d;
        }
        return (maxAbs, (float)(sum / a.Length));
    }
}
