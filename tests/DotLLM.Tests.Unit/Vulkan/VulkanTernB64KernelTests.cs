using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// <c>matmul_tern_b64_mmvq</c> (TERN_B64_F16 overlay kernel, #1706 S3), dense and MoE-indexed, against three oracles:
/// (1) a SAME-TIER scalar double oracle that decodes the packed weights independently and consumes the very int8 activations / scales the GPU quantizer
/// produced (so the only freedom left is float summation order: the bound scales with sqrt(groups) x sum|terms|, and every shader mutant is far outside it);
/// (2) the CPU production kernel <see cref="TernB64F16.GemmQ8"/> (different activation quantizer rounding: KL-level, relative to rms(y));
/// (3) the F32-activation CPU reference <see cref="TernB64F16.GemmF32Act"/> and the OracleWrittenTernary test vectors (skipped when absent).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanTernB64KernelTests
{
    private const string DefaultVectorDir = @"C:\Development\OracleWrittenTernary\artifacts\dotllm-vectors";
    private readonly ITestOutputHelper _out;
    public VulkanTernB64KernelTests(ITestOutputHelper output) => _out = output;

    // ---------------------------------------------------------------- fixtures

    /// <summary>Random trits (about 42 % zeros, like real converted experts) with structured corners: a zero-scale block, an all-zero row, an all +1 block, an all -1 block.</summary>
    private static byte[] MakeTensor(Random rng, int m, int k, bool structured = true)
    {
        int blocks = m * (k / 64);
        var trits = new sbyte[m * k];
        var scales = new float[blocks];
        for (int i = 0; i < trits.Length; i++)
        {
            double u = rng.NextDouble();
            trits[i] = u < 0.42 ? (sbyte)0 : u < 0.71 ? (sbyte)1 : (sbyte)-1;
        }
        for (int b = 0; b < blocks; b++) scales[b] = (float)(0.002 + rng.NextDouble() * 0.2);
        if (structured && blocks >= 4)
        {
            scales[1] = 0f;                                              // zero-scale block with live trits
            for (int j = 0; j < 64; j++) trits[2 * 64 + j] = 1;          // all +1
            for (int j = 0; j < 64; j++) trits[3 * 64 + j] = -1;         // all -1
            if (m >= 2) { int bpr = k / 64; for (int j = 0; j < k; j++) trits[k + j] = 0; scales[bpr] = 0.1f; }   // row 1: all zero trits
        }
        return TernB64F16.Pack(trits, scales);
    }

    private static float[] Acts(Random rng, int count)
    {
        var x = new float[count];
        for (int i = 0; i < count; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * (1 + 3 * rng.NextDouble()));
        return x;
    }

    private sealed record Ctx(VulkanDevice Device, string SpvDir, QuantizeQ8_1RowsKernel Quant, MatMulTernB64Kernel Kernel) : IDisposable
    {
        public void Dispose() { Kernel.Dispose(); Quant.Dispose(); Device.Dispose(); }
    }

    private static Ctx Create()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "device has no integer dot product");
        var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing");
        var kernel = MatMulTernB64Kernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_tern_b64_mmvq.spv missing (rebuild DotLLM.Vulkan)");
        return new Ctx(device, spvDir, quant, kernel);
    }

    private static VulkanDevice.Buffer UploadPadded(VulkanDevice d, byte[] data)
    {
        var buf = d.Allocate(((long)data.Length + 3) & ~3L);
        d.Upload(new ReadOnlySpan<byte>(data), buf);
        return buf;
    }

    private static float[] RunDense(Ctx c, byte[] packed, float[] x, int m, int k, int n, out sbyte[] xq, out float[] xds)
    {
        var d = c.Device;
        using var w = UploadPadded(d, packed);
        using var xb = d.Allocate((long)x.Length * 4);
        using var xqb = d.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdb = d.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var yb = d.Allocate((long)n * m * 4);
        d.Upload(x, xb);
        using (var ctx = d.CreateSubmitContext())
        {
            ctx.Begin();
            c.Quant.Record(ctx.CommandBuffer, xb, xqb, xdb, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            c.Kernel.Record(ctx.CommandBuffer, w, xqb, xdb, yb, m, k, n);
            ctx.SubmitAndWait();
        }
        var y = new float[n * m];
        d.Download(yb, y);
        xq = MoeMmvqParitySupport.DownloadActivationBytes(d, xqb, n, k);
        xds = MoeMmvqParitySupport.DownloadActivationScales(d, xdb, n, k);
        return y;
    }

    /// <summary>Independent scalar decode of the packed format (not shared with the shader or the CPU kernel): returns y and the per-output sum|terms|.</summary>
    private static (double[] Y, double[] AbsSum) Oracle(byte[] packed, long tensorByteOffset, int row, int k, sbyte[] xq, float[] xds, int xRow)
    {
        int blocks = k / 64;
        double y = 0, abs = 0;
        long rowBase = tensorByteOffset + (long)row * blocks * 18;
        for (int b = 0; b < blocks; b++)
        {
            long p = rowBase + b * 18;
            double scale = (double)BitConverter.UInt16BitsToHalf((ushort)(packed[p] | (packed[p + 1] << 8)));
            for (int half = 0; half < 2; half++)
            {
                int dot = 0;
                for (int j = 0; j < 32; j++)
                {
                    int col = half * 32 + j;
                    int code = (packed[p + 2 + (col >> 2)] >> (2 * (col & 3))) & 3;
                    int t = code == 1 ? 1 : code == 2 ? -1 : 0;
                    dot += t * xq[(long)xRow * k + b * 64 + col];
                }
                double dx = xds[((long)xRow * (k / 32) + b * 2 + half) * 2];
                double term = scale * dx * dot;
                y += term;
                abs += Math.Abs(scale * dx) * Math.Abs(dot);
            }
        }
        return ([y], [abs]);
    }

    private static void AssertSameTier(double expected, double absSum, float actual, int groups, string what)
    {
        // float accumulation of `groups` partial products in a different order than the double oracle: |err| <= c * sqrt(groups) * eps32 * sum|terms|.
        double tol = 4.0 * Math.Sqrt(groups) * Math.Pow(2, -24) * absSum + 1e-12;
        Assert.True(Math.Abs(expected - actual) <= tol, $"{what}: expected {expected:G9}, got {actual:G9}, |err| {Math.Abs(expected - actual):E2} > tol {tol:E2} (sum|terms| {absSum:E2})");
    }

    // ---------------------------------------------------------------- dense

    [SkippableTheory]
    [InlineData(1, 64, 1)]
    [InlineData(7, 128, 3)]
    [InlineData(33, 576, 1)]
    [InlineData(33, 576, 9)]
    [InlineData(64, 1024, 8)]
    [InlineData(17, 2048, 33)]
    [InlineData(1536, 576, 1)]
    [InlineData(576, 1536, 4)]
    public void Dense_MatchesSameTierOracle_AndCpuKernels(int m, int k, int n)
    {
        using var c = Create();
        var rng = new Random(0x7E12 + m * 31 + k * 7 + n);
        byte[] w = MakeTensor(rng, m, k);
        float[] x = Acts(rng, n * k);
        float[] y = RunDense(c, w, x, m, k, n, out sbyte[] xq, out float[] xds);

        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                var (ey, ea) = Oracle(w, 0, r, k, xq, xds, t);
                AssertSameTier(ey[0], ea[0], y[t * m + r], k / 16, $"dense m={m} k={k} n={n} row {r} token {t}");
            }

        // KL-level vs the CPU production kernel (own Q8_0 quantizer, fp16 activation scales) and the F32-activation reference.
        var q8 = new float[n * m]; var f32 = new float[n * m];
        fixed (byte* pw = w) fixed (float* px = x) fixed (float* pq = q8) fixed (float* pf = f32)
        {
            bool prev = TernB64F16.UseF32ActivationReference; TernB64F16.UseF32ActivationReference = false;
            try { TernB64F16.GemmQ8(pw, px, pq, m, k, n, null); } finally { TernB64F16.UseF32ActivationReference = prev; }
            TernB64F16.GemmF32Act(pw, px, pf, m, k, n, null);
        }
        double rms = Math.Sqrt(f32.Sum(v => (double)v * v) / f32.Length);
        double eCpu = 0, eF32 = 0, eF32Max = 0;
        for (int i = 0; i < y.Length; i++)
        {
            eCpu += Math.Pow((y[i] - q8[i]) / rms, 2);
            double e = (y[i] - f32[i]) / rms; eF32 += e * e; eF32Max = Math.Max(eF32Max, Math.Abs(e));
        }
        eCpu = Math.Sqrt(eCpu / y.Length); eF32 = Math.Sqrt(eF32 / y.Length);
        _out.WriteLine($"dense m={m} k={k} n={n}: rms err/rms(y) vs CPU int8 kernel {eCpu:E2}; vs F32-act reference {eF32:E2} (max {eF32Max:E2})");
        Assert.True(eCpu < 0.01, $"GPU vs CPU int8 kernel rms err/rms = {eCpu}");
        Assert.True(eF32 < 0.01 && eF32Max < 0.06, $"GPU vs F32-activation reference rms {eF32}, max {eF32Max} (relative to rms(y))");
    }

    [SkippableFact]
    public void Dense_ZeroActivationRowAndZeroScaleBlocks_GiveExactZero()
    {
        using var c = Create();
        const int m = 8, k = 192, n = 2;
        var rng = new Random(11);
        byte[] w = MakeTensor(rng, m, k);
        float[] x = Acts(rng, n * k);
        Array.Clear(x, 0, k);   // token 0: all-zero activation row (Q8_1 scale 0)
        float[] y = RunDense(c, w, x, m, k, n, out _, out _);
        for (int r = 0; r < m; r++) Assert.Equal(0f, y[r]);
        Assert.Contains(y.Skip(m), v => v != 0f);   // control: the other token is not all zero
    }

    // ---------------------------------------------------------------- oracle vectors (their packer)

    public static IEnumerable<object[]> VectorCases()
    {
        string dir = Environment.GetEnvironmentVariable("DOTLLM_TERN_VECTORS_DIR") ?? DefaultVectorDir;
        if (!File.Exists(Path.Combine(dir, "manifest.json"))) { yield return ["<absent>"]; yield break; }
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")));
        foreach (var cs in doc.RootElement.GetProperty("cases").EnumerateArray()) yield return [cs.GetProperty("name").GetString()!];
    }

    [SkippableTheory]
    [MemberData(nameof(VectorCases))]
    public void Dense_MatchesOracleWrittenTernaryVectors(string name)
    {
        string dir = Environment.GetEnvironmentVariable("DOTLLM_TERN_VECTORS_DIR") ?? DefaultVectorDir;
        Skip.IfNot(File.Exists(Path.Combine(dir, "manifest.json")), "OracleWrittenTernary dotllm-vectors not present");
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")));
        var cs = doc.RootElement.GetProperty("cases").EnumerateArray().First(e => e.GetProperty("name").GetString() == name);
        int m = cs.GetProperty("out").GetInt32(), k = cs.GetProperty("in").GetInt32(), n = cs.GetProperty("n_inputs").GetInt32();
        byte[] w = File.ReadAllBytes(Path.Combine(dir, name + ".tern"));
        static float[] ReadF32(string p) { byte[] b = File.ReadAllBytes(p); var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
        float[] x = ReadF32(Path.Combine(dir, name + ".x.f32")), yRef = ReadF32(Path.Combine(dir, name + ".y.f32"));

        using var c = Create();
        float[] y = RunDense(c, w, x, m, k, n, out _, out _);
        double rms = Math.Sqrt(yRef.Sum(v => (double)v * v) / yRef.Length), max = 0, sq = 0;
        for (int i = 0; i < y.Length; i++) { double e = Math.Abs(y[i] - yRef[i]) / rms; max = Math.Max(max, e); sq += e * e; }
        double rmsErr = Math.Sqrt(sq / y.Length);
        _out.WriteLine($"{name}: GPU int8-act vs their float64 y: max err/rms {max:E2}, rms err/rms {rmsErr:E2}");
        Assert.True(max < 0.06 && rmsErr < 0.01, $"{name}: max {max}, rms {rmsErr}");
    }

    // ---------------------------------------------------------------- indexed (per-expert mixed format)

    [SkippableTheory]
    [InlineData(16, 64, 128, 24, 4)]     // xDiv = topK: one activation row per token, broadcast to its 4 expert slots
    [InlineData(16, 64, 128, 24, 1)]     // xDiv = 1: per-row activations (the down projection)
    [InlineData(64, 96, 576, 40, 8)]     // OLMoE-like expert count / top-8, odd block count
    [InlineData(8, 32, 1024, 16, 1)]
    public void Indexed_OnlyTernaryExpertsWritten_EachWithItsOwnTensor(int numExperts, int m, int k, int n, int xDiv)
    {
        using var c = Create();
        var d = c.Device;
        var rng = new Random(0x1DE5 + numExperts * 13 + m + k + n + xDiv);
        var ternary = new List<int>();
        for (int e = 0; e < numExperts; e += 3) ternary.Add((e * 5 + 1) % numExperts);   // a spread subset, never all
        ternary = ternary.Distinct().OrderBy(v => v).ToList();
        Assert.InRange(ternary.Count, 2, numExperts - 1);
        int[] slotOf = Enumerable.Repeat(-1, numExperts).ToArray();
        long tensorBytes = (long)m * (k / 64) * 18;
        var bank = new byte[tensorBytes * ternary.Count];
        for (int s = 0; s < ternary.Count; s++)
        {
            slotOf[ternary[s]] = s;
            MakeTensor(rng, m, k, structured: false).CopyTo(bank, s * tensorBytes);   // DISTINCT tensor per expert
        }
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = (i * 7 + i / 3) % numExperts;   // hits ternary and original experts, repeats
        Assert.Contains(idx, e => slotOf[e] >= 0); Assert.Contains(idx, e => slotOf[e] < 0);
        int xRows = (n + xDiv - 1) / xDiv;
        float[] x = Acts(rng, xRows * k);
        const float Sentinel = 12345.0f;

        using var bankB = UploadPadded(d, bank);
        using var xb = d.Allocate((long)x.Length * 4);
        using var xqb = d.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(xRows, k));
        using var xdb = d.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(xRows, k));
        using var idxB = d.Allocate((long)n * 4);
        using var slotB = d.Allocate((long)numExperts * 4);
        using var yb = d.Allocate((long)n * m * 4);
        d.Upload(x, xb);
        d.Upload(MemoryMarshal.AsBytes<int>(idx), idxB);
        d.Upload(MemoryMarshal.AsBytes<int>(slotOf), slotB);
        d.Upload(Enumerable.Repeat(Sentinel, n * m).ToArray(), yb);
        using (var ctx = d.CreateSubmitContext())
        {
            ctx.Begin();
            c.Quant.Record(ctx.CommandBuffer, xb, xqb, xdb, xRows, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            c.Kernel.RecordIndexed(ctx.CommandBuffer, bankB, xqb, xdb, idxB, slotB, yb, m, k, n, numExperts, xDiv);
            ctx.SubmitAndWait();
        }
        var y = new float[n * m]; d.Download(yb, y);
        sbyte[] xq = MoeMmvqParitySupport.DownloadActivationBytes(d, xqb, xRows, k);
        float[] xds = MoeMmvqParitySupport.DownloadActivationScales(d, xdb, xRows, k);

        int written = 0, untouched = 0;
        for (int r = 0; r < n; r++)
            for (int row = 0; row < m; row++)
            {
                float got = y[r * m + row];
                int slot = slotOf[idx[r]];
                if (slot < 0) { Assert.Equal(Sentinel, got); untouched++; continue; }
                var (ey, ea) = Oracle(bank, slot * tensorBytes, row, k, xq, xds, r / xDiv);
                AssertSameTier(ey[0], ea[0], got, k / 16, $"indexed row {r} (expert {idx[r]}, slot {slot}) out {row}");
                written++;
            }
        Assert.True(written > 0 && untouched > 0);
        _out.WriteLine($"indexed E={numExperts} m={m} k={k} n={n} xDiv={xDiv}: {written} ternary cells match the oracle, {untouched} original-format cells untouched");
    }
}
