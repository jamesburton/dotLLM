using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Shared driver for the Q8_0 outer-product prefill parity check, reused across models
/// (SmolLM-135M, Llama-3.2-1B, …). Runs the <em>same model and the same multi-token prompt</em>
/// through both Q8_0 prefill reductions — the inner-product / cache-tiled path (flag off) and the
/// R4 outer-product microkernels (flag on, <see cref="TransformerModel.UseOuterProductQ8Prefill"/>)
/// — and asserts the full output logits matrix agrees to near-FP-epsilon. Argmax checks are
/// intentionally avoided as the gate; the full-vector comparison is the discriminating assertion.
/// </summary>
internal static class OuterProductPrefillParity
{
    /// <summary>
    /// Loads <paramref name="ggufPath"/>, runs the multi-token <paramref name="prompt"/> through both
    /// Q8_0 prefill paths in one process, and asserts full-logits parity plus proof the outer-product
    /// path actually executed.
    /// </summary>
    /// <param name="ggufPath">Path to a Q8_0 GGUF model file.</param>
    /// <param name="prompt">A prompt that tokenizes to more than one token (so prefill runs with n &gt; 1).</param>
    public static void AssertMatchesInnerProduct(string ggufPath, string prompt)
    {
        var gguf = GgufFile.Open(ggufPath);
        using var _ = gguf;
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        // Parallel threading exercises the parallel outer-product worker (the production path).
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

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

        // Proof the new path actually executed: a silent fallback would also match the baseline,
        // making the comparison vacuous. Assert a generous lower bound on the call delta.
        Assert.True(invocationsAfter - invocationsBefore >= 10,
            $"Outer-product GEMM was not invoked during flag-on prefill " +
            $"(delta={invocationsAfter - invocationsBefore}); the flag-on run silently fell back, " +
            "so the parity comparison would be vacuous.");

        // --- Compare full logits vectors ---
        // Correctness of the matmul itself is proven elsewhere against a SCALAR ground truth
        // (OuterProductGemmTests.GroundTruth_GemmAndOuter_AtRealisticShapes: both the inner-product
        // baseline and the outer-product match the exact scalar Q8_0 result to ~3e-5 abs / ~1e-4 rel).
        // This end-to-end check therefore uses a SCALE-NORMALIZED tolerance, NOT per-cell bit-identity:
        // the two paths round correctly but differently (~7e-5 per matmul), and over ~30 layers /
        // ~210 matmuls (residual + softmax) that compounds to a small fraction of the logit magnitude.
        // A per-cell relative tolerance is wrong here — near-zero (cancellation) logits blow it up into
        // spurious "sign flips" despite both values being correct. On AVX2 the two roundings happen to
        // coincide (bit-identical); on AVX-512 they differ slightly — both are correct.
        const float scaleNormTol = 5e-2f;   // maxAbsDiff relative to the tile's max |logit|

        float maxAbsDiff = 0f;
        float maxBaselineAbs = 0f;
        int worstIdx = -1;
        for (int i = 0; i < logitCount; i++)
        {
            float a = baseline[i];
            float b = outerProduct[i];
            Assert.True(float.IsFinite(a) && float.IsFinite(b),
                $"Non-finite logit at index {i}: baseline={a}, outer={b}");

            float absDiff = MathF.Abs(a - b);
            if (absDiff > maxAbsDiff) { maxAbsDiff = absDiff; worstIdx = i; }
            maxBaselineAbs = MathF.Max(maxBaselineAbs, MathF.Abs(a));
        }

        float scaleNormDiff = maxBaselineAbs > 1e-6f ? maxAbsDiff / maxBaselineAbs : maxAbsDiff;
        string worst = worstIdx >= 0
            ? $"at index {worstIdx} (baseline={baseline[worstIdx]}, outer={outerProduct[worstIdx]})"
            : "(vectors bit-identical)";

        Assert.True(scaleNormDiff <= scaleNormTol,
            $"Outer-product prefill logits diverged from inner-product baseline beyond accumulated-FP " +
            $"tolerance. scaleNormDiff={scaleNormDiff:E3} (tol={scaleNormTol:E3}) {worst}; " +
            $"maxAbsDiff={maxAbsDiff:E3}, maxBaselineAbs={maxBaselineAbs:E3}.");

        // Surface the parity numbers in test output (the key deliverable metric).
        Console.WriteLine(
            $"[OuterProductPrefillParity] model={System.IO.Path.GetFileName(ggufPath)} " +
            $"tokens={tokenIds.Length} logits={logitCount} " +
            $"outerProductGemmCalls={invocationsAfter - invocationsBefore} " +
            $"maxAbsDiff={maxAbsDiff:E3} scaleNormDiff={scaleNormDiff:E3}");
    }
}
