using System.Text.Json;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// TERN_B64_F16 kernels against the test vectors exported by the OracleWrittenTernary session (real OLMoE L7 expert 54 and SmolLM2 L15 tensors; reference
/// <c>y = W_hat x</c> with FP16-rounded scales and float64 accumulation, produced by THEIR packer, so a layout error in our reader or writer cannot cancel out).
/// Skips cleanly when the vector directory is absent (<c>DOTLLM_TERN_VECTORS_DIR</c> overrides the default location).
/// </summary>
[Collection("TernKernelSerial")]
public sealed unsafe class TernB64F16OracleVectorTests
{
    private const string DefaultDir = @"C:\Development\OracleWrittenTernary\artifacts\dotllm-vectors";
    private readonly ITestOutputHelper _out;
    public TernB64F16OracleVectorTests(ITestOutputHelper output) => _out = output;

    private static string? VectorDir()
    {
        string dir = Environment.GetEnvironmentVariable("DOTLLM_TERN_VECTORS_DIR") ?? DefaultDir;
        return File.Exists(Path.Combine(dir, "manifest.json")) ? dir : null;
    }

    private static float[] ReadF32(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    public static IEnumerable<object[]> Cases()
    {
        string? dir = VectorDir();
        if (dir is null) { yield return ["<absent>"]; yield break; }
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")));
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
            yield return [c.GetProperty("name").GetString()!];
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Kernels_MatchOracleVectors(string name)
    {
        string? dir = VectorDir();
        Skip.If(dir is null, "OracleWrittenTernary dotllm-vectors not present");
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir!, "manifest.json")));
        var c = doc.RootElement.GetProperty("cases").EnumerateArray().First(e => e.GetProperty("name").GetString() == name);
        int m = c.GetProperty("out").GetInt32(), k = c.GetProperty("in").GetInt32(), n = c.GetProperty("n_inputs").GetInt32();

        byte[] w = File.ReadAllBytes(Path.Combine(dir!, name + ".tern"));
        float[] x = ReadF32(Path.Combine(dir!, name + ".x.f32"));
        float[] y = ReadF32(Path.Combine(dir!, name + ".y.f32"));
        Assert.Equal((long)m * k / 64 * 18, w.Length);
        Assert.Equal(n * k, x.Length);
        Assert.Equal(n * m, y.Length);
        fixed (byte* pw = w) TernB64F16.Validate(name, pw, m, k);   // their packer never emits 0b11

        var f32 = new float[n * m]; var q8 = new float[n * m];
        fixed (byte* pw = w) fixed (float* px = x) fixed (float* pf = f32) fixed (float* pq = q8)
        {
            TernB64F16.GemmF32Act(pw, px, pf, m, k, n, null);
            bool prev = TernB64F16.UseF32ActivationReference;
            TernB64F16.UseF32ActivationReference = false;
            try { TernB64F16.GemmQ8(pw, px, pq, m, k, n, null); }
            finally { TernB64F16.UseF32ActivationReference = prev; }
        }

        double rms = Math.Sqrt(y.Sum(v => (double)v * v) / y.Length);
        double f32Max = 0, q8Max = 0, q8Sq = 0;
        for (int i = 0; i < y.Length; i++)
        {
            f32Max = Math.Max(f32Max, Math.Abs(f32[i] - y[i]) / rms);
            double e = Math.Abs(q8[i] - y[i]) / rms;
            q8Max = Math.Max(q8Max, e); q8Sq += e * e;
        }
        double q8Rms = Math.Sqrt(q8Sq / y.Length);
        _out.WriteLine($"{name}: y-rms {rms:G4}; F32-act max err/rms {f32Max:E2}; int8-act max err/rms {q8Max:E2}, rms err/rms {q8Rms:E2}");
        Assert.True(f32Max < 1e-5, $"{name}: F32-activation kernel max |err|/rms(y) = {f32Max}");
        Assert.True(q8Max < 0.06 && q8Rms < 0.01, $"{name}: int8-activation kernel max {q8Max}, rms {q8Rms} (relative to rms(y))");
    }

    [SkippableFact]
    public void TinyHandCheckedCase_BitOrder()
    {
        string? dir = VectorDir();
        Skip.If(dir is null, "OracleWrittenTernary dotllm-vectors not present");
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir!, "manifest.json")));
        var t = doc.RootElement.GetProperty("tiny");
        int m = t.GetProperty("out").GetInt32(), k = t.GetProperty("in").GetInt32();
        byte[] packed = Convert.FromHexString(t.GetProperty("packed_hex").GetString()!);
        float[][] xs = t.GetProperty("x").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        float[][] ys = t.GetProperty("y").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        Assert.Equal(m * k / 64 * 18, packed.Length);
        for (int i = 0; i < xs.Length; i++)
        {
            var got = new float[m];
            fixed (byte* pw = packed) fixed (float* px = xs[i]) fixed (float* pc = got)
                TernB64F16.GemmF32Act(pw, px, pc, m, k, 1, null);
            for (int r = 0; r < m; r++) Assert.Equal(ys[i][r], got[r], 4);
        }
    }
}
