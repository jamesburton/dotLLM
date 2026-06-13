using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Same-process A/B parity test for the Q8_0 outer-product prefill path.
///
/// <para>Runs the <em>same model and the same multi-token prompt</em> through both Q8_0 prefill
/// reductions — the existing inner-product / cache-tiled path (flag off) and the R4 outer-product
/// microkernels (flag on, <see cref="TransformerModel.UseOuterProductQ8Prefill"/>) — and compares
/// the full output logits vector. Both paths consume the identical R4-repacked weights and are
/// integer-exact Q8_0 reductions that differ only in block-accumulation order, so the logits must
/// agree to near-FP-epsilon. Argmax/"Paris" checks are intentionally avoided as the gate because
/// argmax masks numerical drift; the full-vector comparison is the discriminating assertion.</para>
///
/// <para>Because both paths are integer-exact, a passing comparison alone cannot prove the
/// outer-product kernel actually ran (a silent fallback would also match). The test therefore
/// also asserts <see cref="MatMul.OuterProductGemmQ8_0InvocationCount"/> advanced during the
/// flag-on prefill — proof the new path executed.</para>
/// </summary>
[Collection("SmallModel")]
public class OuterProductPrefillParityTests
{
    private readonly SmallModelFixture _fixture;

    public OuterProductPrefillParityTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        // Parallel threading exercises the parallel outer-product worker (the production path),
        // not just the single-threaded fallback.
        var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer);
    }

    // Prompts chosen so the token count N lands on different remainder-tile boundaries of the
    // outer-product dispatcher (AVX2 steps by 3, AVX-512 by 6, then single-token tails):
    //   N small (2)   → no full 3-tile, all single-token tail path
    //   N=3-ish       → exact 3-tile, no tail
    //   N larger (>9) → multiple full tiles + a 1- or 2-token tail
    // This guards the "N not a multiple of 6/3" remainder handling in OuterProductGemmQ8_0.
    [Theory]
    [InlineData("Hello world")]
    [InlineData("The capital of France")]
    [InlineData("The capital of France is Paris and the weather today")]
    public void OuterProductPrefill_MatchesInnerProduct_FullLogits(string prompt)
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        // Multi-token prompt → prefill runs with n = tokenIds.Length > 1, the regime the
        // outer-product kernels target.
        int[] tokenIds = tokenizer.Encode(prompt);
        Assert.True(tokenIds.Length > 1,
            $"Need a multi-token prompt to exercise prefill (n>1); got {tokenIds.Length} tokens.");

        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        int vocabSize = model.Config.VocabSize;

        // --- Baseline: inner-product / cache-tiled path (flag off) ---
        // Forward returns logits for ALL prefill positions ([seqLen, vocabSize]); compare the
        // entire matrix, not just the last token, for the most discriminating coverage.
        model.UseOuterProductQ8Prefill = false;
        float[] baseline;
        int logitCount;
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1))
        {
            logitCount = (int)logits.ElementCount;
            Assert.Equal(tokenIds.Length * vocabSize, logitCount);
            baseline = new float[logitCount];
            unsafe
            {
                new ReadOnlySpan<float>((void*)logits.DataPointer, logitCount).CopyTo(baseline);
            }
        }

        // --- Outer-product path (flag on) ---
        long invocationsBefore = MatMul.OuterProductGemmQ8_0InvocationCount;
        model.UseOuterProductQ8Prefill = true;
        float[] outerProduct = new float[logitCount];
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1))
        {
            Assert.Equal(logitCount, (int)logits.ElementCount);
            unsafe
            {
                new ReadOnlySpan<float>((void*)logits.DataPointer, logitCount).CopyTo(outerProduct);
            }
        }
        long invocationsAfter = MatMul.OuterProductGemmQ8_0InvocationCount;

        // Proof the new path actually executed: SmolLM-135M has 30 layers × (Q,K,V,O,gate,up,down)
        // = 210 Q8_0 projections plus 1 LM head = 211 outer-product GEMM calls per forward.
        // (Embeddings are not repacked.) Assert a generous lower bound rather than the exact count.
        Assert.True(invocationsAfter - invocationsBefore >= 10,
            $"Outer-product GEMM was not invoked during flag-on prefill " +
            $"(delta={invocationsAfter - invocationsBefore}); the flag-on run silently fell back, " +
            "so the parity comparison would be vacuous.");

        // --- Compare full logits vectors ---
        // Both paths are integer-exact Q8_0 reductions; the only source of difference is FP
        // summation order across blocks (and the parallel partitioning of rows). A tight relative
        // tolerance with a small absolute floor (for logits near zero) captures that while still
        // failing on any real numerical bug.
        const float relTol = 1e-3f;
        const float absFloor = 1e-3f;

        float maxAbsDiff = 0f;
        float maxRelDiff = 0f;
        int worstIdx = -1;
        for (int i = 0; i < logitCount; i++)
        {
            float a = baseline[i];
            float b = outerProduct[i];
            Assert.True(float.IsFinite(a) && float.IsFinite(b),
                $"Non-finite logit at index {i}: baseline={a}, outer={b}");

            float absDiff = MathF.Abs(a - b);
            float denom = MathF.Max(MathF.Abs(a), MathF.Abs(b));
            float relDiff = denom > absFloor ? absDiff / denom : absDiff;

            if (relDiff > maxRelDiff)
            {
                maxRelDiff = relDiff;
                worstIdx = i;
            }
            maxAbsDiff = MathF.Max(maxAbsDiff, absDiff);
        }

        string worst = worstIdx >= 0
            ? $"at index {worstIdx} (baseline={baseline[worstIdx]}, outer={outerProduct[worstIdx]})"
            : "(vectors bit-identical)";

        Assert.True(maxRelDiff <= relTol,
            $"Outer-product prefill logits diverged from inner-product baseline. " +
            $"maxRelDiff={maxRelDiff:E3} (tol={relTol:E3}) {worst}; maxAbsDiff={maxAbsDiff:E3}.");

        // Surface the parity numbers in test output (the key deliverable metric).
        Console.WriteLine(
            $"[OuterProductPrefillParity] tokens={tokenIds.Length} logits={logitCount} " +
            $"outerProductGemmCalls={invocationsAfter - invocationsBefore} " +
            $"maxAbsDiff={maxAbsDiff:E3} maxRelDiff={maxRelDiff:E3}");
    }
}
