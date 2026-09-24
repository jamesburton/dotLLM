using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cpu;

/// <summary>
/// THROWAWAY diagnostic for issue #519 — decomposes the error of dotLLM's Q3_K × Q8_K dot into
/// (a) the unavoidable cost of quantizing activations to Q8_K and (b) anything the dot itself adds.
/// Delete once #519 is settled.
/// </summary>
/// <remarks>
/// <para>
/// exact  = sum(W_f32 · X_f32)                        — the true value
/// ideal  = sum(W_dequant · dequant(Q8K(X)))          — a perfect dot of the quantized inputs
/// actual = VecDotQ3_K_Q8_K(W_q3k, Q8K(X))            — what dotLLM computes
/// </para>
/// <para>
/// If <c>actual == ideal</c> then dotLLM's dot is exact given its inputs and all of its loss is
/// activation quantization — in which case llama.cpp cannot be 3× better unless its *quantizer*
/// differs. If <c>actual != ideal</c>, the dot is adding error of its own and that is the lead.
/// </para>
/// </remarks>
public sealed class Q3KErrorDecompositionProbe
{
    private readonly ITestOutputHelper _output;

    public Q3KErrorDecompositionProbe(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void DecomposeQ3KDotError()
    {
        string model = Environment.GetEnvironmentVariable("DOTLLM_Q3K_PROBE_MODEL") ?? "";
        string outDir = Environment.GetEnvironmentVariable("DOTLLM_Q3K_PROBE_OUT") ?? "";
        Skip.If(model.Length == 0 || !File.Exists(model), "DOTLLM_Q3K_PROBE_MODEL not set");

        using var gguf = GgufFile.Open(model);

        GgufTensorDescriptor tensor = gguf.Tensors.First(t =>
            t.QuantizationType == QuantizationType.Q3_K && t.Name == "blk.0.attn_q.weight");

        int k = tensor.Shape[0];
        int m = Math.Min(tensor.Shape[1], 512);
        int superBlocks = k / 256;

        // Dequantized weights (bit-exact vs gguf-py, verified in #515).
        var w = new float[(long)m * k];
        Dequantize.DequantizeQ3_K(gguf.DataBasePointer + (nint)tensor.DataOffset, (long)m * k, w);

        // Activations with a realistic dynamic range: mostly small, a few outliers, which is what
        // makes Q8_K's single per-256-block scale lossy in the first place.
        var rng = new Random(519);
        var x = new float[k];
        for (int i = 0; i < k; i++)
        {
            double g = Math.Sqrt(-2.0 * Math.Log(rng.NextDouble() + 1e-12)) * Math.Cos(2 * Math.PI * rng.NextDouble());
            x[i] = (float)g;
        }
        for (int i = 0; i < k; i += 97) x[i] *= 8f;   // outliers

        int q8kBytes = superBlocks * MatMul.Q8_K_BlockBytes;
        var q8k = new byte[q8kBytes];
        var actual = new float[m];

        fixed (float* xp = x)
        fixed (byte* q8p = q8k)
        fixed (float* resp = actual)
        {
            MatMul.QuantizeF32ToQ8_K(xp, q8p, k);
            MatMul.ComputeRowsQ3_K((byte*)(gguf.DataBasePointer + (nint)tensor.DataOffset), q8p, resp, m, superBlocks);
        }

        // Reconstruct what the dot actually saw: dequant(Q8K(x)).
        var xq = new float[k];
        fixed (byte* q8p = q8k)
        {
            for (int b = 0; b < superBlocks; b++)
            {
                byte* blk = q8p + b * MatMul.Q8_K_BlockBytes;
                float d = *(float*)blk;
                sbyte* qs = (sbyte*)(blk + 4);
                for (int i = 0; i < 256; i++) xq[b * 256 + i] = d * qs[i];
            }
        }

        double sumRelIdeal = 0, sumRelActual = 0, worstDotDelta = 0;
        for (int row = 0; row < m; row++)
        {
            long b = (long)row * k;
            double exact = 0, ideal = 0;
            for (int i = 0; i < k; i++)
            {
                exact += (double)w[b + i] * x[i];
                ideal += (double)w[b + i] * xq[i];
            }
            double scale = Math.Abs(exact) > 1e-9 ? Math.Abs(exact) : 1.0;
            sumRelIdeal += Math.Abs(ideal - exact) / scale;
            sumRelActual += Math.Abs(actual[row] - exact) / scale;
            worstDotDelta = Math.Max(worstDotDelta, Math.Abs(actual[row] - ideal) / scale);
        }

        _output.WriteLine($"rows={m} k={k}");
        _output.WriteLine($"mean |ideal-exact|/|exact|  (Q8_K quantization floor) : {sumRelIdeal / m:E4}");
        _output.WriteLine($"mean |actual-exact|/|exact| (dotLLM packed dot)       : {sumRelActual / m:E4}");
        _output.WriteLine($"worst |actual-ideal|/|exact| (error the DOT adds)     : {worstDotDelta:E4}");

        if (outDir.Length > 0)
        {
            File.WriteAllText(Path.Combine(outDir, "probe_x.txt"),
                string.Join('\n', x.Select(v => v.ToString("R"))));
            File.WriteAllBytes(Path.Combine(outDir, "probe_q8k.bin"), q8k);
            _output.WriteLine($"dumped activations + Q8_K bytes to {outDir}");
        }

        Assert.True(m > 0);
    }
}
