using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using DotLLM.Cpu.Kernels.Experimental;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;
using DotLLM.Models.Quantization;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #866: single-path NanoQuant operator. The crop fixture holds REAL bits and scales cropped from
/// <c>arelath/Qwen3-0.6B-nanoquant-GGUF</c> blk.0.ffn_gate with a float64 numpy reference
/// (<c>NanoQuantData/gen_fixture.py</c>); the real-file tests read the GGUF from <c>DOTLLM_NANOQUANT_GGUF</c> (or the HF
/// hub cache) and return early (clean skip) when it is absent.
/// </summary>
public sealed unsafe class NanoQuantTests(ITestOutputHelper output)
{
    private static JsonDocument LoadJson(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Cpu", "Kernels", "NanoQuantData", name)));

    private static float[] F(JsonElement e) => e.EnumerateArray().Select(x => x.GetSingle()).ToArray();
    private static int[] I(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();
    private static int[] Flat(JsonElement e) => e.EnumerateArray().SelectMany(r => r.EnumerateArray().Select(x => x.GetInt32())).ToArray();

    internal static string? RealGguf()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_NANOQUANT_GGUF");
        if (!string.IsNullOrEmpty(env)) return File.Exists(env) ? env : null;
        string snaps = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub",
            "models--arelath--Qwen3-0.6B-nanoquant-GGUF", "snapshots");
        return Directory.Exists(snaps)
            ? Directory.EnumerateFiles(snaps, "qwen3-0-6b-nanoquant.gguf", SearchOption.AllDirectories).FirstOrDefault()
            : null;
    }

    private sealed record Fx(int DOut, int DIn, int R, int[] U, int[] V, float[] Pre, float[] Mid, float[] Post, int[] Idx, float[] Sal, float[] X, double[] Y);

    private static Fx LoadCrop()
    {
        using var doc = LoadJson("nanoquant_crop_fixture.json");
        var r = doc.RootElement;
        return new Fx(r.GetProperty("d_out").GetInt32(), r.GetProperty("d_in").GetInt32(), r.GetProperty("r").GetInt32(),
            Flat(r.GetProperty("U")), Flat(r.GetProperty("V")), F(r.GetProperty("scale_pre")), F(r.GetProperty("scale_mid")),
            F(r.GetProperty("scale_post")), I(r.GetProperty("salient_idx")), r.GetProperty("salient_weight").EnumerateArray()
                .SelectMany(row => row.EnumerateArray().Select(x => x.GetSingle())).ToArray(),
            F(r.GetProperty("x")), r.GetProperty("y").EnumerateArray().Select(e => e.GetDouble()).ToArray());
    }

    private static NanoQuantLayer Build(Fx f) => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid, f.Post, f.Idx, f.Sal);

    [Fact]
    public void ScalarReference_MatchesNumpyFloat64()
    {
        var f = LoadCrop();
        var y = NanoQuantReference.Gemv(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid, f.Post, f.Idx, f.Sal, f.X);
        Assert.Equal(f.DOut, y.Length);
        for (int i = 0; i < y.Length; i++) Assert.Equal(f.Y[i], y[i], 1e-9 * (1 + Math.Abs(f.Y[i])));
    }

    [Fact]
    public void Avx2Kernel_MatchesNumpyFloat64_OnRealCrop()
    {
        if (!Avx2.IsSupported) return;
        var f = LoadCrop();
        using var layer = Build(f);
        Assert.Equal(2, layer.SalientCount);
        using var sc = layer.CreateScratch();
        var got = new float[f.DOut];
        fixed (float* x = f.X, y = got) layer.Gemv(x, y, sc, null);
        double scale = f.Y.Max(Math.Abs);
        for (int i = 0; i < got.Length; i++) Assert.True(Math.Abs(f.Y[i] - got[i]) / scale < 1e-5, $"row {i}: {f.Y[i]} vs {got[i]}");
    }

    [Fact]
    public void SalientPath_IsActive_AndAddsRawX()
    {
        if (!Avx2.IsSupported) return;
        var f = LoadCrop();
        // Same layer without the salient path must differ by exactly sum_s w[o,s] * x[idx[s]].
        using var with = Build(f);
        using var without = NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid, f.Post, [], []);
        using var sc = with.CreateScratch();
        var a = new float[f.DOut]; var b = new float[f.DOut];
        fixed (float* x = f.X, ya = a, yb = b) { with.Gemv(x, ya, sc, null); without.Gemv(x, yb, sc, null); }
        int k = f.Idx.Length;
        for (int o = 0; o < f.DOut; o++)
        {
            double add = 0;
            for (int s = 0; s < k; s++) add += f.Sal[o * k + s] * (double)f.X[f.Idx[s]];
            Assert.Equal(add, a[o] - b[o], 1e-4 * (1 + Math.Abs(add)));
        }
        Assert.Contains(Enumerable.Range(0, f.DOut), o => Math.Abs(a[o] - b[o]) > 1e-6);
    }

    [Fact]
    public void VnniKernel_TracksReference_WithinInt8Tolerance()
    {
        if (!LittleBitLayer.VnniSupported) return;
        var f = LoadCrop();
        using var layer = Build(f);
        using var sc = layer.CreateScratch();
        var got = new float[f.DOut];
        fixed (float* x = f.X, y = got) layer.Gemv(x, y, sc, null, LittleBitKernel.VnniInt8);
        double scale = f.Y.Max(Math.Abs);
        for (int i = 0; i < got.Length; i++) Assert.True(Math.Abs(f.Y[i] - got[i]) / scale < 0.05, $"row {i}: {f.Y[i]} vs {got[i]}");
    }

    [Theory]
    [InlineData(1, 1, 1, 0)]
    [InlineData(37, 70, 33, 3)]
    [InlineData(64, 128, 96, 4)]
    [InlineData(5, 31, 32, 1)]
    public void RandomLayers_KernelMatchesReference_NonMultipleOf32(int dOut, int dIn, int r, int k)
    {
        if (!Avx2.IsSupported) return;
        var rng = new Random(1234 + dOut);
        int uw = (r + 31) / 32, vw = (dIn + 31) / 32;
        var u = new int[dOut * uw]; var v = new int[r * vw];
        for (int i = 0; i < u.Length; i++) u[i] = rng.Next(int.MinValue, int.MaxValue);
        for (int i = 0; i < v.Length; i++) v[i] = rng.Next(int.MinValue, int.MaxValue);
        // Tail bits are +1 (clear) by the format; the reference reads only valid bits, the kernel must ignore the rest.
        if (r % 32 != 0) for (int o = 0; o < dOut; o++) u[o * uw + uw - 1] &= (1 << (r % 32)) - 1;
        if (dIn % 32 != 0) for (int j = 0; j < r; j++) v[j * vw + vw - 1] &= (1 << (dIn % 32)) - 1;
        // Scales on a bf16 grid inside fp16 range (as in real files).
        float Sc() => BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(0.1f + 0.9f * rng.NextSingle()) & 0xFFFF0000u);
        var pre = Enumerable.Range(0, dIn).Select(_ => Sc()).ToArray();
        var mid = Enumerable.Range(0, r).Select(_ => Sc()).ToArray();
        var post = Enumerable.Range(0, dOut).Select(_ => Sc()).ToArray();
        var idx = Enumerable.Range(0, k).Select(s => s * (dIn / Math.Max(k, 1))).ToArray();
        foreach (int c in idx) pre[c] = 0f;
        var sal = Enumerable.Range(0, k * dOut).Select(_ => rng.NextSingle() - 0.5f).ToArray();
        var x = Enumerable.Range(0, dIn).Select(_ => rng.NextSingle() * 2 - 1).ToArray();
        var want = NanoQuantReference.Gemv(dOut, dIn, r, u, v, pre, mid, post, idx, sal, x);
        using var layer = NanoQuantLayer.FromPacked(dOut, dIn, r, u, v, pre, mid, post, idx, sal);
        using var sc = layer.CreateScratch();
        using var pool = new ComputeThreadPool(2);
        foreach (var p in new ComputeThreadPool?[] { null, pool })
        {
            var got = new float[dOut];
            fixed (float* xp = x, yp = got) layer.Gemv(xp, yp, sc, p);
            double scale = Math.Max(1e-6, want.Max(Math.Abs));
            for (int i = 0; i < dOut; i++) Assert.True(Math.Abs(want[i] - got[i]) / scale < 1e-5, $"row {i}: {want[i]} vs {got[i]}");
        }
    }

    [Fact]
    public void FromPacked_RejectsMalformedInput()
    {
        var f = LoadCrop();
        Assert.Throws<ArgumentException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U[1..], f.V, f.Pre, f.Mid, f.Post, f.Idx, f.Sal));
        Assert.Throws<ArgumentException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V[1..], f.Pre, f.Mid, f.Post, f.Idx, f.Sal));
        Assert.Throws<ArgumentException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid[1..], f.Post, f.Idx, f.Sal));
        Assert.Throws<ArgumentException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid, f.Post, f.Idx, f.Sal[1..]));
        Assert.Throws<InvalidDataException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid, f.Post, [f.DIn], f.Sal[..f.DOut]));
        Assert.Throws<InvalidDataException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, f.Mid, f.Post, [f.Idx[1], f.Idx[0]], f.Sal));
        var pre = (float[])f.Pre.Clone(); pre[f.Idx[0]] = 0.5f;   // llama.cpp contract: scale_pre is exactly 0 at salient indices
        Assert.Throws<InvalidDataException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, pre, f.Mid, f.Post, f.Idx, f.Sal));
        var mid = (float[])f.Mid.Clone(); mid[0] = 1e-6f;          // not exact in fp16 (subnormal): refuse to round silently
        Assert.Throws<InvalidDataException>(() => NanoQuantLayer.FromPacked(f.DOut, f.DIn, f.R, f.U, f.V, f.Pre, mid, f.Post, f.Idx, f.Sal));
    }

    // ------------------------------------------------------------------ real GGUF (skips when absent)

    [Fact]
    public void RealGguf_Inventory_MatchesDocumentedHeader()
    {
        string? path = RealGguf();
        if (path is null) { output.WriteLine("SKIP: NanoQuant GGUF absent (set DOTLLM_NANOQUANT_GGUF)."); return; }
        using var file = GgufFile.Open(path);
        Assert.True(NanoQuantLoader.IsNanoQuant(file));
        NanoQuantLoader.ValidateMetadata(file.Metadata);
        Assert.Equal(1095, file.Tensors.Count);
        var bases = NanoQuantLoader.FindBases(file);
        Assert.Equal(28 * 5, bases.Count);   // attn_qkv, attn_output, ffn_gate/up/down per block
        Assert.Equal(140 * 7, file.Tensors.Count(t => t.Name.Contains(".nq_")));
    }

    [Fact]
    public void RealGguf_FullLayers_MatchNumpyFloat64()
    {
        string? path = RealGguf();
        if (path is null) { output.WriteLine("SKIP: NanoQuant GGUF absent (set DOTLLM_NANOQUANT_GGUF)."); return; }
        if (!Avx2.IsSupported) return;
        using var doc = LoadJson("nanoquant_full_expect.json");
        using var file = GgufFile.Open(path);
        using var pool = new ComputeThreadPool(4);
        foreach (var c in doc.RootElement.EnumerateObject())
        {
            using var layer = NanoQuantLoader.Load(file, c.Name);
            Assert.Equal(c.Value.GetProperty("d_out").GetInt32(), layer.DOut);
            Assert.Equal(c.Value.GetProperty("d_in").GetInt32(), layer.DIn);
            Assert.Equal(c.Value.GetProperty("r").GetInt32(), layer.R);
            Assert.Equal(c.Value.GetProperty("k").GetInt32(), layer.SalientCount);
            var x = Enumerable.Range(0, layer.DIn).Select(i => (float)(1.5 * Math.Sin(0.37 * i + 0.1))).ToArray();
            using var sc = layer.CreateScratch();
            var y = new float[layer.DOut];
            fixed (float* xp = x, yp = y) layer.Gemv(xp, yp, sc, pool);
            var first = c.Value.GetProperty("y_first").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            double sumsq = c.Value.GetProperty("y_sumsq").GetDouble();
            double rms = Math.Sqrt(sumsq / layer.DOut);
            double worst = 0;
            for (int i = 0; i < first.Length; i++) worst = Math.Max(worst, Math.Abs(first[i] - y[i]) / rms);
            double gotSq = y.Sum(v => (double)v * v);
            double rel = Math.Abs(gotSq - sumsq) / sumsq;
            output.WriteLine($"{c.Name}: worst first-32 |dy|/rms = {worst:E2}, sumsq rel err = {rel:E2}");
            Assert.True(worst < 1e-4, $"{c.Name} first-32 rows");
            Assert.True(rel < 1e-5, $"{c.Name} sum of squares");
            Assert.Equal(c.Value.GetProperty("y_sum").GetDouble(), y.Sum(v => (double)v), Math.Sqrt(sumsq) * 1e-4);
        }
    }
}
