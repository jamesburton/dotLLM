using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cpu;

/// <summary>
/// The packed Q3_K × Q8_K dot must add essentially no error beyond the unavoidable cost of
/// quantizing the activations to Q8_K.
/// </summary>
/// <remarks>
/// <para>
/// Decomposes the kernel's error on <b>real GGUF weights</b> into two parts:
/// </para>
/// <code>
/// exact  = Σ W_f32 · X_f32                      the true value
/// ideal  = Σ W_dequant · dequant(Q8K(X))        a perfect dot of the quantized inputs
/// actual = ComputeRowsQ3_K(W_q3k, Q8K(X))       what dotLLM computes
/// </code>
/// <para>
/// <c>ideal − exact</c> is the quantization floor and is not the kernel's fault. <c>actual −
/// ideal</c> <b>is</b>, and it should be at rounding level. Asserting the floor rather than an
/// absolute accuracy bound is what makes this test meaningful: a bound loose enough to accommodate
/// Q8_K's own ~6% relative error on a dot product would accommodate almost any kernel bug too.
/// </para>
/// <para>
/// Written for issue #519, which alleged dotLLM's Q3_K quantization cost ~2.9× llama.cpp's. This
/// decomposition showed the dot was already at the floor — there was no room for such a defect —
/// and the 2.9% turned out to be degraded-model amplification, not a kernel property. The test
/// stays so that a future change which *does* put error into the dot is caught as a kernel
/// regression rather than rediscovered through perplexity on a degraded model.
/// </para>
/// </remarks>
public sealed class Q3KDotIsAtQuantizationFloorTests
{
    private readonly ITestOutputHelper _output;

    public Q3KDotIsAtQuantizationFloorTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Error the dot may add beyond the quantization floor, as a fraction of ‖w_row‖·‖x‖.
    /// Observed worst is ~1.6e−8 on both tensors (AVX2), against a quantization floor of
    /// ~2e−4 — four orders of magnitude below it. The limit leaves ~600× headroom for a different
    /// SIMD tier while still sitting ~20× below the floor it has to discriminate against.
    /// </summary>
    private const double MaxDotErrorBeyondFloor = 1e-5;

    [SkippableTheory]
    [InlineData("blk.0.attn_q.weight")]
    [InlineData("blk.7.ffn_down.weight")]
    public unsafe void PackedQ3KDot_AddsNothingBeyondTheQ8KFloor(string tensorName)
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            ["DOTLLM_LLAMA32_1B_PURE_Q3_K_GGUF"], "quant-ladder", "Llama-3.2-1B-pure",
            ["Llama-3.2-1B-pure-Q3_K.gguf"],
            extraDirectories: [Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "quant-ladder", "Llama-3.2-1B-pure")]);
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.2-1B pure Q3_K GGUF"));

        using var gguf = GgufFile.Open(fixture.Path!);
        GgufTensorDescriptor tensor = gguf.Tensors.First(t =>
            t.QuantizationType == QuantizationType.Q3_K && t.Name == tensorName);

        int k = tensor.Shape[0];
        int m = Math.Min(tensor.Shape[1], 256);
        int superBlocks = k / 256;
        nint weights = gguf.DataBasePointer + (nint)tensor.DataOffset;

        var w = new float[(long)m * k];
        Dequantize.DequantizeQ3_K(weights, (long)m * k, w);

        // Gaussian activations with periodic outliers: a single per-256-block scale is at its
        // most lossy when the block has a few large values, so this is the regime where a sloppy
        // dot would be easiest to hide behind the floor.
        var rng = new Random(519);
        var x = new float[k];
        for (int i = 0; i < k; i++)
        {
            double g = Math.Sqrt(-2.0 * Math.Log(rng.NextDouble() + 1e-12))
                       * Math.Cos(2 * Math.PI * rng.NextDouble());
            x[i] = (float)g;
        }
        for (int i = 0; i < k; i += 97) x[i] *= 8f;

        var q8k = new byte[superBlocks * MatMul.Q8_K_BlockBytes];
        var actual = new float[m];
        var xq = new float[k];

        fixed (float* xp = x)
        fixed (byte* q8p = q8k)
        fixed (float* resp = actual)
        {
            MatMul.QuantizeF32ToQ8_K(xp, q8p, k);
            MatMul.ComputeRowsQ3_K((byte*)weights, q8p, resp, m, superBlocks);

            for (int b = 0; b < superBlocks; b++)
            {
                byte* blk = q8p + b * MatMul.Q8_K_BlockBytes;
                float d = *(float*)blk;
                sbyte* qs = (sbyte*)(blk + 4);
                for (int i = 0; i < 256; i++) xq[b * 256 + i] = d * qs[i];
            }
        }

        // Scale errors by ‖w_row‖·‖x‖ rather than by |exact|. A dot of ~2000 random signed terms
        // is near zero for some rows, and dividing by it turns a rounding difference into a huge
        // "relative error" — which is a property of the row, not of the kernel. ‖w‖·‖x‖ bounds
        // |exact| and is stable across rows and tensors.
        double xNorm = Math.Sqrt(x.Sum(v => (double)v * v));

        double sumFloor = 0, sumActual = 0, worstBeyondFloor = 0;
        for (int row = 0; row < m; row++)
        {
            long b = (long)row * k;
            double exact = 0, ideal = 0, wSq = 0;
            for (int i = 0; i < k; i++)
            {
                exact += (double)w[b + i] * x[i];
                ideal += (double)w[b + i] * xq[i];
                wSq += (double)w[b + i] * w[b + i];
            }

            double denom = Math.Max(Math.Sqrt(wSq) * xNorm, 1e-30);
            sumFloor += Math.Abs(ideal - exact) / denom;
            sumActual += Math.Abs(actual[row] - exact) / denom;
            worstBeyondFloor = Math.Max(worstBeyondFloor, Math.Abs(actual[row] - ideal) / denom);
        }

        _output.WriteLine($"{tensorName}: rows={m} k={k}");
        _output.WriteLine($"  Q8_K quantization floor : {sumFloor / m:E4}");
        _output.WriteLine($"  dotLLM packed dot       : {sumActual / m:E4}");
        _output.WriteLine($"  worst beyond the floor  : {worstBeyondFloor:E4}");

        Assert.True(worstBeyondFloor < MaxDotErrorBeyondFloor,
            $"packed Q3_K dot adds {worstBeyondFloor:E4} relative error beyond the Q8_K "
            + $"quantization floor (limit {MaxDotErrorBeyondFloor:E4}) — the dot itself is lossy, "
            + "not just the activation quantization.");
    }
}
