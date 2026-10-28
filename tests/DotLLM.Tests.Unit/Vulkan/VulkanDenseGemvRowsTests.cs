using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1220: the Q4_K GEMV with NR output rows per workgroup must give, for every column count, the same bits per column as its own
/// one-column (decode) dispatch, also when the two use a different NR (the model decodes at NR = 2 and verifies at NR = 4). That equality is what keeps a dense qwen35 MTP verify identical to plain greedy decode, since the dense
/// model decodes on the same kernel. Against the separate coalesced 1-row GEMV it agrees to FMA-contraction ULPs only.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanDenseGemvRowsTests
{
    private readonly ITestOutputHelper _output;

    public VulkanDenseGemvRowsTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(37, 5120, 1, 2)]   // NR = 1: odd M
    [InlineData(37, 5120, 2, 3)]
    [InlineData(37, 5120, 4, 2)]   // M % NR != 0: the last workgroup's spare rows must not be stored
    [InlineData(64, 17408, 2, 4)]  // ffn_down K, MTP K = 3 verify
    [InlineData(65, 5120, 8, 8)]   // NR = 8: 64 partials, M % NR != 0
    [InlineData(30, 768, 3, 3)]    // K = 3 super-blocks: window tail
    [InlineData(1, 512, 4, 3)]     // M below NR (shapes from #1193's R4 cases)
    [InlineData(3, 4096, 4, 7)]
    [InlineData(41, 1280, 4, 4)]
    public void Q4KRows_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int rowsPerWorkgroup, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x1220 + m * 7 + k * 11 + n * 13 + rowsPerWorkgroup);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = Quantize.FromFloat32(w, (long)m * k, QuantizationType.Q4_K);

        using var rows = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q4_K, 2, rowsPerWorkgroup)
                         ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_gemv_f32_multi.spv unavailable");
        AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q4_K,
            (cmd, bw, bx, by, nn) => rows.Record(cmd, bw, bx, by, m, k, nn));
    }

    [SkippableTheory]
    [InlineData(37, 5120, 2, 2)]
    [InlineData(37, 5120, 4, 3)]   // M % NR != 0
    [InlineData(48, 5120, 2, 4)]   // ssm_alpha / beta
    [InlineData(40, 96, 3, 8)]     // K = 3 blocks: window tail
    public void Q8Rows_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int rowsPerWorkgroup, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x1220 + m * 7 + k * 11 + n * 13 + rowsPerWorkgroup);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = Quantize.FromFloat32(w, (long)m * k, QuantizationType.Q8_0);
        using var rows = MatMulQ8_0GemvMultiKernel.TryCreate(device, spvDir, 2, rowsPerWorkgroup)
                         ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_gemv_multi.spv unavailable");
        AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q8_0,
            (cmd, bw, bx, by, nn) => rows.Record(cmd, bw, bx, by, m, k, nn));
    }

    [SkippableTheory]
    [InlineData(37, 5120, 2, 2)]
    [InlineData(37, 6144, 4, 3)]   // attn_output K, M % NR != 0
    [InlineData(64, 5120, 1, 4)]
    [InlineData(30, 768, 3, 8)]    // K = 3 super-blocks: window tail
    public void Q6KRows_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int rowsPerWorkgroup, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x6220 + m * 7 + k * 11 + n * 13 + rowsPerWorkgroup);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = new byte[m * (k / 256) * 210];   // random Q6_K blocks (no CPU quantiser), small fp16 d at byte 208
        rng.NextBytes(wq);
        for (int off = 0; off < wq.Length; off += 210)
            BitConverter.TryWriteBytes(wq.AsSpan(off + 208, 2), (Half)(0.0005f + 0.001f * (float)rng.NextDouble()));
        using var rows = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K_Rows, 2, rowsPerWorkgroup)
                         ?? throw new Xunit.Sdk.XunitException("matmul_q6_k_gemv_f32_rows.spv unavailable");
        AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q6_K,
            (cmd, bw, bx, by, nn) => rows.Record(cmd, bw, bx, by, m, k, nn));
    }

    /// <summary>
    /// #1365: the 9..16-column pipelines (NR = 1..4 rows per workgroup) keep the same bits per column as the one-column dispatch, so a 9..16-row
    /// dense qwen35 verify / DFlash block is still identical to plain greedy decode. Discriminating shapes: M not a multiple of NR (spare rows
    /// of the last workgroup must not be stored), K of 3 super-blocks (window tail), every column a distinct activation row.
    /// </summary>
    [SkippableTheory]
    [InlineData(37, 5120, 4, 9)]
    [InlineData(37, 5120, 4, 12)]
    [InlineData(37, 5120, 3, 13)]   // M % NR != 0 at NR = 3
    [InlineData(65, 5120, 4, 16)]   // M % NR != 0, the widest pipeline
    [InlineData(64, 17408, 4, 16)]  // ffn_down K
    [InlineData(30, 768, 2, 11)]    // K = 3 super-blocks: window tail
    [InlineData(1, 512, 4, 16)]     // M below NR
    [InlineData(41, 1280, 1, 10)]
    public void Q4KWide_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int wideRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x1365 + m * 7 + k * 11 + n * 13 + wideRows);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = Quantize.FromFloat32(w, (long)m * k, QuantizationType.Q4_K);
        using var rows = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q4_K, 2, 4, wideRows)
                         ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_gemv_f32_multi.spv unavailable");
        Assert.Equal(16, rows.MaxColumnsSupported);
        AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q4_K,
            (cmd, bw, bx, by, nn) => rows.Record(cmd, bw, bx, by, m, k, nn));
    }

    [SkippableTheory]
    [InlineData(37, 5120, 4, 9)]
    [InlineData(48, 5120, 4, 16)]   // ssm_alpha / beta
    [InlineData(37, 6144, 3, 13)]   // M % NR != 0
    [InlineData(40, 96, 2, 12)]     // K = 3 blocks: window tail
    public void Q8Wide_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int wideRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x1365 + m * 7 + k * 11 + n * 13 + wideRows);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = Quantize.FromFloat32(w, (long)m * k, QuantizationType.Q8_0);
        using var rows = MatMulQ8_0GemvMultiKernel.TryCreate(device, spvDir, 2, 4, wideRows)
                         ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_gemv_multi.spv unavailable");
        AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q8_0,
            (cmd, bw, bx, by, nn) => rows.Record(cmd, bw, bx, by, m, k, nn));
    }

    [SkippableTheory]
    [InlineData(37, 6144, 4, 9)]    // attn_output K, M % NR != 0
    [InlineData(64, 5120, 4, 16)]
    [InlineData(30, 768, 3, 12)]    // K = 3 super-blocks: window tail
    [InlineData(65, 5120, 1, 14)]
    public void Q6KWide_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int wideRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x6365 + m * 7 + k * 11 + n * 13 + wideRows);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = new byte[m * (k / 256) * 210];
        rng.NextBytes(wq);
        for (int off = 0; off < wq.Length; off += 210)
            BitConverter.TryWriteBytes(wq.AsSpan(off + 208, 2), (Half)(0.0005f + 0.001f * (float)rng.NextDouble()));
        using var rows = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K_Rows, 2, 4, wideRows)
                         ?? throw new Xunit.Sdk.XunitException("matmul_q6_k_gemv_f32_rows.spv unavailable");
        AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q6_K,
            (cmd, bw, bx, by, nn) => rows.Record(cmd, bw, bx, by, m, k, nn));
    }

    /// <summary>The 8-column formats refuse 9..16 rows instead of reading a pipeline that does not exist.</summary>
    [SkippableFact]
    public void FormatsWithoutWidePipelinesStopAtEight()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic, "No subgroup arithmetic.");
        using var q5 = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q5_K);
        Skip.If(q5 is null, "Q5_K multi unavailable");
        Assert.Equal(8, q5!.MaxColumnsSupported);
        Assert.True(q5.AcceptsColumns(8, 256));
        Assert.False(q5.AcceptsColumns(9, 256));
    }

    internal static void AssertColumnsMatchOneColumnDispatch(VulkanDevice device, byte[] wq, float[] x, int m, int k, int n, QuantizationType qt,
                                                             Action<nint, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, int> record,
                                                             float scale = 1f)
    {
        using var bw = device.Allocate(((long)wq.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var by = device.Allocate((long)n * m * 4 + 4);
        using var bx1 = device.Allocate((long)k * 4);
        using var by1 = device.Allocate((long)m * 4);
        device.Upload(new ReadOnlySpan<byte>(wq), bw);
        device.Upload(x, bx);
        float[] sentinel = new float[n * m + 1];
        sentinel[^1] = 1234.5f;
        device.Upload(sentinel, by);
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); record(ctx.CommandBuffer, bw, bx, by, n); ctx.SubmitAndWait(); }
        float[] actual = new float[n * m + 1];
        device.Download(by, actual);
        Assert.Equal(1234.5f, actual[^1]);
        float[] wf = DequantizeAny(wq, m, k, qt);
        for (int c = 0; c < n; c++)
        {
            device.Upload(x.AsSpan(c * k, k).ToArray(), bx1);
            using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); record(ctx.CommandBuffer, bw, bx1, by1, 1); ctx.SubmitAndWait(); }
            float[] exp = new float[m];
            device.Download(by1, exp);
            Assert.Equal(exp, actual.AsSpan(c * m, m).ToArray());

            // And numerically the GEMV against the dequantised weights (catches a wrong row / column mapping shared by both dispatches).
            for (int r = 0; r < m; r++)
            {
                double dot = 0;
                for (int i = 0; i < k; i++) dot += (double)wf[(long)r * k + i] * x[c * k + i];
                dot *= scale;
                Assert.True(Math.Abs(dot - exp[r]) <= 1e-3 + 1e-4 * Math.Abs(dot), $"row {r} col {c}: {exp[r]} vs {dot}");
            }
        }
    }

    /// <summary>
    /// Opt-in timing (<c>DOTLLM_GEMV_ROWS_BENCH=1</c>) at Qwen3.8-27B's Q4_K FFN and Q8_0 GDN shapes: the coalesced 1-row decode GEMV against the
    /// rows-per-workgroup kernels for NR = 2, 4, 6, 8 and 1..4 columns. Minimum over rounds of 20 back-to-back dispatches; GB/s counts each weight byte once.
    /// </summary>
    [SkippableFact]
    public void Bench_RowsPerWorkgroup()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMV_ROWS_BENCH") == "1", "DOTLLM_GEMV_ROWS_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic, "No subgroup arithmetic.");
        using var q4Single = MatMulQ4KGemvF32Kernel.Create(device, spvDir);
        using var q8Single = MatMulQ8_0Kernel.Create(device, spvDir, coalesced: true);
        using var q6Single = MatMulQ6KGemvF32Kernel.Create(device, spvDir);
        var q6 = new MatMulQuantGemvMultiKernel[4];
        var q4 = new MatMulQuantGemvMultiKernel[4];
        var q8 = new MatMulQ8_0GemvMultiKernel[4];
        int[] nrs = [2, 4, 6, 8];
        for (int r = 0; r < 4; r++)
        {
            q4[r] = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q4_K, nrs[r])!;
            q8[r] = MatMulQ8_0GemvMultiKernel.TryCreate(device, spvDir, nrs[r])!;
            q6[r] = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K_Rows, nrs[r])!;
        }
        try
        {
            foreach (var (qt, m, k) in new[] { (QuantizationType.Q4_K, 17408, 5120), (QuantizationType.Q4_K, 5120, 17408),
                                               (QuantizationType.Q8_0, 10240, 5120), (QuantizationType.Q8_0, 5120, 6144),
                                               (QuantizationType.Q6_K, 248320, 5120), (QuantizationType.Q6_K, 5120, 6144) })
            {
                var rng = new Random(m + k);
                bool isQ4 = qt == QuantizationType.Q4_K, isQ6 = qt == QuantizationType.Q6_K;
                int blockBytes = isQ4 ? 144 : isQ6 ? 210 : 34, blockElems = isQ4 || isQ6 ? 256 : 32;
                var bytes = new byte[(long)m * (k / blockElems) * blockBytes];
                rng.NextBytes(bytes);
                for (long off = 0; off < bytes.Length; off += blockBytes)
                {
                    BitConverter.TryWriteBytes(bytes.AsSpan((int)(isQ6 ? off + 208 : off), 2), (Half)0.01f);
                    if (isQ4) BitConverter.TryWriteBytes(bytes.AsSpan((int)off + 2, 2), (Half)0.002f);
                }
                using var bw = Upload(device, bytes);
                float[] x = F16Bf16Fixture.RandomFloats(rng, 4 * k, 1.0f);
                using var bx = Upload(device, System.Runtime.InteropServices.MemoryMarshal.AsBytes(x.AsSpan()).ToArray());
                using var by = device.AllocateDeviceLocal(4L * m * 4);
                var arms = new List<(string Name, Action<nint> Rec)>
                {
                    ("coalesced n=1", isQ4 ? cmd => q4Single.Record(cmd, bw, bx, by, m, k)
                        : isQ6 ? cmd => q6Single.Record(cmd, bw, bx, by, m, k) : cmd => q8Single.Record(cmd, bw, bx, by, m, k)),
                };
                for (int n = 1; n <= 4; n++)
                    for (int r = 0; r < 4; r++)
                    {
                        int nn = n, rr = r;
                        arms.Add(($"NR={nrs[rr]} n={nn}", isQ4 ? cmd => q4[rr].Record(cmd, bw, bx, by, m, k, nn)
                            : isQ6 ? cmd => q6[rr].Record(cmd, bw, bx, by, m, k, nn)
                            : cmd => q8[rr].Record(cmd, bw, bx, by, m, k, nn)));
                    }
                double[] best = Time(device, arms.Select(a => a.Rec).ToArray());
                for (int i = 0; i < arms.Count; i++)
                    _output.WriteLine($"gemv-bench {qt} m={m} k={k} {arms[i].Name,-14} {best[i] * 1000:F0} us  {bytes.Length / (best[i] / 1000) / 1e9:F1} GB/s");
            }
        }
        finally
        {
            foreach (var kk in q4) kk?.Dispose();
            foreach (var kk in q8) kk?.Dispose();
            foreach (var kk in q6) kk?.Dispose();
        }
    }

    /// <summary>Minimum per-dispatch milliseconds of each arm over 5 rounds of 20 back-to-back dispatches.</summary>
    internal static double[] Time(VulkanDevice device, Action<nint>[] arms, int reps = 20, int rounds = 5)
    {
        var best = Enumerable.Repeat(double.MaxValue, arms.Length).ToArray();
        for (int round = 0; round < rounds; round++)
            for (int i = 0; i < arms.Length; i++)
            {
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int rep = 0; rep < reps; rep++)
                {
                    arms[i](ctx.CommandBuffer);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                }
                var sw = System.Diagnostics.Stopwatch.StartNew();
                ctx.SubmitAndWait();
                best[i] = Math.Min(best[i], sw.Elapsed.TotalMilliseconds / reps);
            }
        return best;
    }

    private static float[] DequantizeAny(byte[] wq, int m, int k, QuantizationType qt)
    {
        var w = new float[(long)m * k];
        unsafe
        {
            fixed (byte* p = wq)
                Dequantize.ToFloat32((nint)p, (long)m * k, qt, w);
        }
        return w;
    }

    private static VulkanDevice.Buffer Upload(VulkanDevice device, byte[] bytes)
    {
        using var staging = device.Allocate((bytes.Length + 3L) & ~3L);
        device.Upload(bytes, staging);
        var dst = device.AllocateDeviceLocal((bytes.Length + 3L) & ~3L);
        device.CopyBufferSynchronous(staging, dst, (ulong)bytes.Length);
        return dst;
    }
}
