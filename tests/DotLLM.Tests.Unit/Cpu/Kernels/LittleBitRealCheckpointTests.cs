using System.Text.Json;
using DotLLM.Cpu.Kernels.Experimental;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Quantization;
using DotLLM.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #864 step 1: real packed LittleBit tensors from the Qwen3-0.6B / 0.55 bpw export, run through the spike's
/// reference and AVX2 kernels, against a PyTorch reference generated offline (<c>LittleBitData/gen_real_fixture.py</c>).
/// Reads the checkpoint from <c>DOTLLM_LITTLEBIT_CKPT</c> (directory) or
/// <c>~/.dotllm/test-cache/littlebit-qwen3-0.6b-055</c>; returns early (clean skip) when absent.
/// Precision decision: checkpoint scales are bf16; they are widened exactly to F32 and l = bf16(v1*u2) is rounded once
/// to bf16 like the reference forward. The kernel accumulates in F32 (the reference rounds every stage to bf16).
/// </summary>
public sealed class LittleBitRealCheckpointTests(ITestOutputHelper output)
{
    private static string? CheckpointFile()
    {
        string? dir = Environment.GetEnvironmentVariable("DOTLLM_LITTLEBIT_CKPT");
        if (string.IsNullOrEmpty(dir))
            dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "littlebit-qwen3-0.6b-055");
        string f = Path.Combine(dir, "model.safetensors");
        return File.Exists(f) ? f : null;
    }

    private static float[] F32(string b64) { var b = Convert.FromBase64String(b64); var r = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, r, 0, b.Length); return r; }
    private static float[] Bf16(string b64)
    {
        var b = Convert.FromBase64String(b64); var r = new float[b.Length / 2];
        for (int i = 0; i < r.Length; i++) r[i] = BitConverter.UInt32BitsToSingle((uint)(b[2 * i] | b[2 * i + 1] << 8) << 16);
        return r;
    }

    private static double MaxRel(IReadOnlyList<double> expected, IReadOnlyList<float> got)
    {
        double scale = expected.Max(Math.Abs), worst = 0;
        for (int i = 0; i < expected.Count; i++) worst = Math.Max(worst, Math.Abs(expected[i] - got[i]) / scale);
        return worst;
    }

    [Fact]
    public unsafe void RealLayers_MatchPyTorchReference()
    {
        string? file = CheckpointFile();
        if (file is null) { output.WriteLine("SKIP: LittleBit checkpoint absent (set DOTLLM_LITTLEBIT_CKPT)."); return; }

        using var doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Cpu", "Kernels", "LittleBitData", "littlebit_real_fixture.json")));
        using var st = SafetensorsFile.Open(file);
        using var pool = new ComputeThreadPool(4);

        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string prefix = c.GetProperty("prefix").GetString()!;
            float[] x = Bf16(c.GetProperty("x_bf16").GetString()!);
            float[] y64 = F32(c.GetProperty("y_f64_as_f32").GetString()!);
            float[] y16 = Bf16(c.GetProperty("y_bf16_as_bf16").GetString()!);

            using var layer = LittleBitLoader.LoadLayer(st, prefix);
            Assert.Equal(c.GetProperty("d_out").GetInt32(), layer.DOut);
            Assert.Equal(c.GetProperty("d_in").GetInt32(), layer.DIn);
            Assert.Equal(c.GetProperty("r").GetInt32(), layer.Paths[0].R);
            Assert.Equal(c.GetProperty("paths").GetInt32(), layer.Paths.Length);

            double[] refD = LittleBitReference.Gemv(layer, x);
            var (dIn, r) = layer.ScratchDims;
            using var sc = new LittleBitScratch(dIn, r);
            var avx = new float[layer.DOut];
            var avxPooled = new float[layer.DOut];
            fixed (float* xp = x, yp = avx, yq = avxPooled)
            {
                layer.Gemv(xp, yp, sc, null);
                layer.Gemv(xp, yq, sc, pool);
            }

            double refVs64 = MaxRel(refD.Select(v => (double)(float)v).ToArray(), y64);
            double avxVs64 = MaxRel(refD, avx);
            double avxVsPyF32 = MaxRel(y64.Select(v => (double)v).ToArray(), avx);
            double avxVsBf16 = MaxRel(y16.Select(v => (double)v).ToArray(), avx);
            output.WriteLine($"{prefix} {layer.DOut}x{layer.DIn} r={layer.Paths[0].R} paths={layer.Paths.Length}: " +
                             $"scalarRef-vs-torchF64 {refVs64:E2}  avx2-vs-scalarRef {avxVs64:E2}  avx2-vs-torchF64 {avxVsPyF32:E2}  " +
                             $"avx2-vs-torchBf16 {avxVsBf16:E2}  (max|err|/max|y|)");

            Assert.True(refVs64 < 1e-6, $"{prefix}: scalar reference vs torch float64 {refVs64:E2}");
            Assert.True(avxVs64 < 1e-5, $"{prefix}: avx2 vs scalar reference {avxVs64:E2}");
            Assert.True(avxVsPyF32 < 1e-5, $"{prefix}: avx2 vs torch float64 {avxVsPyF32:E2}");
            Assert.Equal(avx, avxPooled);
            // The torch bf16 forward rounds every stage to bf16 (8-bit mantissa); this bounds, not equals, our F32 result.
            Assert.True(avxVsBf16 < 0.05, $"{prefix}: avx2 vs torch bf16 forward {avxVsBf16:E2}");
        }
    }
}
