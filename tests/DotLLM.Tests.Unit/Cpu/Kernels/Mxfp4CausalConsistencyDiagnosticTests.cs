using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Diagnostic for issue #275: is the CPU MXFP4 per-row dot product itself batch-size dependent?
/// </summary>
/// <remarks>
/// <para>The #256 gate's "prefill leg" scores a single one-shot 256-token forward pass
/// (<c>TeacherForcedSinglePass</c>), while its "decode leg" re-runs SEPARATE, SHORTER (5-13 token)
/// one-shot forward passes at growing lengths (per issue #275's own correcting comment — NEITHER
/// leg exercises the seqLen==1 fused-decode/GEMV path; both are GEMM-shaped, differing only in how
/// many columns (N) are quantized/processed together in one <c>GemmMxfp4</c> call). If dotLLM's
/// CPU MXFP4 GEMM has any bug tied to the batch column count N (e.g. a scratch-buffer sizing/reuse
/// issue, an off-by-one in the per-token loop, a SIMD tail miscount for small N), it would produce
/// EXACTLY this signature: large-N calls (prefill) agree with GPU while small-N calls (decode) do
/// not, with no code difference in the row-dot kernel itself (already proven identical between
/// Gemm/Gemv by <c>Mxfp4Tests.GemmMxfp4_MatchesGemvPerRow</c>).</para>
/// <para>This test needs no GPU: it checks CPU against ITSELF. A correct causal transformer's
/// logits at position P depend only on tokens 0..P — never on how many additional tokens are also
/// being processed in the same batched call. So the logits for the same prefix, computed once as
/// part of a long single-pass call and once as part of a short single-pass call, must agree to
/// ordinary floating-point-order noise (not exactly, since accumulation order across a
/// differently-shaped SIMD loop can differ in the last few ULPs — but should be nowhere near the
/// 1.2e-2 cosine divergence the issue reports). If they do NOT agree, that is a genuine CPU-only,
/// GPU-independent proof of an N-dependent bug in the MXFP4 GEMM path — precisely the kind of
/// defect issue #275 asks for, demonstrated without needing CUDA/Vulkan hardware.</para>
/// </remarks>
public sealed class Mxfp4CausalConsistencyDiagnosticTests
{
    private readonly ITestOutputHelper _output;

    public Mxfp4CausalConsistencyDiagnosticTests(ITestOutputHelper output) => _output = output;

    private static string? ResolveFixture()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string path = Path.Combine(home, ".dotllm", "quant-ladder", "Llama-3.2-1B-pure", "Llama-3.2-1B-pure-MXFP4.gguf");
        return File.Exists(path) ? path : null;
    }

    [SkippableFact]
    public unsafe void Mxfp4Gemm_LogitsAtSharedPrefix_AreBatchSizeInvariant()
    {
        string? path = ResolveFixture();
        Skip.If(path is null,
            "Llama-3.2-1B-pure-MXFP4.gguf not found.");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.Auto);

        const string corpus =
            "The quick brown fox jumps over the lazy dog near the old stone bridge. " +
            "Scientists have long studied how small variations in early conditions can lead to " +
            "very different outcomes over time, a phenomenon popularly known as the butterfly " +
            "effect. In computing, this sensitivity shows up whenever a tiny rounding error " +
            "compounds across many sequential steps of a calculation.";

        int[] allTokens = tokenizer.Encode(corpus);
        Assert.True(allTokens.Length >= 64, $"corpus too short: {allTokens.Length} tokens");

        int vocab = model.Config.VocabSize;

        // Mirror the gate's actual shapes: a long (~64-token) single-pass call standing in for the
        // "prefill leg", and several short (5-13 token) single-pass calls standing in for the
        // "decode leg" — each scored at ITS OWN last position, which is a shared prefix of the long
        // call.
        int longLen = Math.Min(64, allTokens.Length);
        int[] longTokens = allTokens[..longLen];
        int[] longPositions = Enumerable.Range(0, longLen).ToArray();

        using var longLogitsTensor = model.Forward(longTokens, longPositions, deviceId: -1);

        // Copy every row of the long pass up front.
        var longAllRows = new float[longLen][];
        for (int r = 0; r < longLen; r++)
            longAllRows[r] = ReadRow(longLogitsTensor, longLen, vocab, r);

        int[] shortLens = { 5, 8, 10, 13, 20, 32 };
        double worstOneMinusCos = 0;
        int worstLen = -1;

        foreach (int shortLen in shortLens)
        {
            if (shortLen > longLen) continue;
            int[] shortTokens = allTokens[..shortLen];
            int[] shortPositions = Enumerable.Range(0, shortLen).ToArray();

            using var shortLogitsTensor = model.Forward(shortTokens, shortPositions, deviceId: -1);
            float[] shortLast = ReadRow(shortLogitsTensor, shortLen, vocab, shortLen - 1);
            float[] longAtSamePos = longAllRows[shortLen - 1];

            double cos = CosineSimilarity(shortLast, longAtSamePos);
            double oneMinusCos = 1.0 - cos;
            _output.WriteLine($"N={shortLen,3}: 1-cos vs N={longLen} call at same position = {oneMinusCos:E4}");

            if (oneMinusCos > worstOneMinusCos)
            {
                worstOneMinusCos = oneMinusCos;
                worstLen = shortLen;
            }
        }

        _output.WriteLine($"Worst: N={worstLen}, 1-cos={worstOneMinusCos:E4}");

        // Ordinary FP-order noise from a differently-shaped SIMD loop should be many orders of
        // magnitude below the issue's reported 1.2e-2. If MXFP4's GEMM has a genuine N-dependent
        // bug, this assertion fails and worstOneMinusCos pinpoints it — entirely CPU-side.
        Assert.True(worstOneMinusCos < 1e-6,
            $"CPU MXFP4 GEMM logits at a shared prefix position depend on the total batch size N " +
            $"(worst 1-cos={worstOneMinusCos:E4} at N={worstLen}) — this is a genuine, GPU-independent " +
            "batch-size-dependent bug in the CPU MXFP4 GEMM path.");
    }

    private static unsafe float[] ReadRow(DotLLM.Core.Tensors.ITensor tensor, int seqLen, int vocab, int row)
    {
        var span = new ReadOnlySpan<float>((void*)tensor.DataPointer, seqLen * vocab);
        int r = row < 0 ? seqLen - 1 : row;
        var result = new float[vocab];
        span.Slice(r * vocab, vocab).CopyTo(result);
        return result;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}
