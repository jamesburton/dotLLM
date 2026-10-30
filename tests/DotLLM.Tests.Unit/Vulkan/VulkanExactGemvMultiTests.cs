using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1244: the deterministic multi-column NVFP4 / IQ2_XXS / Q8_0 GEMVs decode each weight block once for every activation column, and each
/// column must still give the same bits as its own one-column (decode) dispatch, also when decode and verify use a different number of rows
/// per workgroup. That equality keeps a Nemotron-H speculative verify identical to plain greedy decoding. Distinct random activations per
/// column, odd M (partial last workgroup) and window tails make a column / row mix-up or an NC-dependent accumulation order visible.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanExactGemvMultiTests
{
    private readonly ITestOutputHelper _output;

    public VulkanExactGemvMultiTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(37, 4096, 1, 4, 8)]    // Bolt ssm_in K, LiLiCorr K = 7 verify
    [InlineData(37, 8192, 2, 4, 3)]    // ssm_out K, M % NR != 0
    [InlineData(65, 192, 1, 8, 8)]     // K = 3 blocks: window tail; NR * NC = 64
    [InlineData(3, 4096, 1, 4, 16)]    // a DFlash block of 16 (NR clamps to 4), M below NR
    [InlineData(29, 1024, 2, 2, 2)]
    public void Nvfp4_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int decodeRows, int multiRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x1244 + m * 7 + k * 11 + n * 13 + multiRows);
        byte[] wq = new byte[m * (k / 64) * 36];
        rng.NextBytes(wq);
        byte[] edge = [0x00, 0x7F, 0xFF, 0x01, 0x80, 0x7E];
        for (int b = 0; b < wq.Length; b += 36)
            for (int s = 0; s < 4; s++)
                wq[b + s] = rng.Next(8) == 0 ? edge[rng.Next(edge.Length)] : (byte)rng.Next(0x20, 0x50);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        const float scale = 1.3e-3f;
        using var kernel = Create(device, spvDir, ExactGemvFormat.Nvfp4, decodeRows, multiRows);
        VulkanDenseGemvRowsTests.AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.NVFP4,
            (cmd, bw, bx, by, nn) => kernel.Record(cmd, bw, bx, by, m, k, nn, scale), scale);
    }

    [SkippableTheory]
    [InlineData(37, 4096, 1, 4, 8)]    // Bolt ffn_up_shexp K
    [InlineData(37, 5376, 2, 4, 3)]    // ffn_down_shexp K = 21 super-blocks: uneven lane strides
    [InlineData(9, 256, 1, 8, 8)]      // one super-block: lanes past the row's 32 groups
    [InlineData(5, 1024, 1, 4, 16)]
    [InlineData(37, 4096, 1, 2, 14)]   // #1366: 14..16 columns run NR = 4 even when multiRows = 2 (the NR = 2 cliff)
    [InlineData(37, 5376, 1, 2, 15)]
    [InlineData(37, 4096, 1, 2, 16)]
    [InlineData(37, 4096, 1, 2, 13)]   // still NR = 2
    public void Iq2Xxs_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int decodeRows, int multiRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var codebooks = Iq2Codebooks.Create(device);
        var rng = new Random(0x2244 + m * 7 + k * 11 + n * 13 + multiRows);
        byte[] wq = new byte[m * (k / 256) * 66];   // random IQ2_XXS blocks (every grid / sign index is valid), small fp16 d
        rng.NextBytes(wq);
        for (int off = 0; off < wq.Length; off += 66)
            BitConverter.TryWriteBytes(wq.AsSpan(off, 2), (Half)(0.0005f + 0.001f * (float)rng.NextDouble()));
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        using var kernel = Create(device, spvDir, ExactGemvFormat.Iq2Xxs, decodeRows, multiRows);
        VulkanDenseGemvRowsTests.AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.IQ2_XXS,
            (cmd, bw, bx, by, nn) => kernel.Record(cmd, bw, bx, by, m, k, nn, grid: codebooks.Iq2XxsGrid, ksigns: codebooks.Ksigns));
    }

    [SkippableTheory]
    [InlineData(37, 4096, 1, 4, 8)]    // Bolt latent / attention K
    [InlineData(41, 1024, 2, 4, 16)]   // latent_up K, a block of 16
    [InlineData(40, 96, 1, 8, 3)]      // K = 3 blocks: window tail
    public void Q8_0_EachColumnBitIdenticalToOneColumnDispatch(int m, int k, int decodeRows, int multiRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x3244 + m * 7 + k * 11 + n * 13 + multiRows);
        float[] w = F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        byte[] wq = Quantize.FromFloat32(w, (long)m * k, QuantizationType.Q8_0);
        using var kernel = Create(device, spvDir, ExactGemvFormat.Q8_0, decodeRows, multiRows);
        VulkanDenseGemvRowsTests.AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q8_0,
            (cmd, bw, bx, by, nn) => kernel.Record(cmd, bw, bx, by, m, k, nn));
    }

    /// <summary>
    /// #1420: Q5_0 / Q4_0 (Nemotron-H Nano Q4_K_M keeps ssm_in / ffn_up in Q5_0). Random blocks (every nibble and 5th-bit pattern, so a transposed
    /// qh bit or nibble half is visible in the dequantised-reference check), small fp16 d at byte 0; 1..16 columns, M % NR != 0, K of 3 blocks.
    /// </summary>
    [SkippableTheory]
    [InlineData(ExactGemvFormat.Q5_0, 37, 3136, 1, 4, 8)]    // Nano ssm_in K
    [InlineData(ExactGemvFormat.Q5_0, 41, 3136, 2, 4, 16)]
    [InlineData(ExactGemvFormat.Q5_0, 40, 96, 1, 8, 3)]      // K = 3 blocks: window tail
    [InlineData(ExactGemvFormat.Q5_0, 29, 12544, 1, 2, 13)]  // ffn_down-like K
    [InlineData(ExactGemvFormat.Q4_0, 37, 3136, 1, 4, 8)]
    [InlineData(ExactGemvFormat.Q4_0, 41, 3136, 2, 4, 16)]
    [InlineData(ExactGemvFormat.Q4_0, 40, 96, 1, 8, 3)]
    [InlineData(ExactGemvFormat.Q4_0, 29, 12544, 1, 2, 13)]
    public void LegacyQ_EachColumnBitIdenticalToOneColumnDispatch(ExactGemvFormat format, int m, int k, int decodeRows, int multiRows, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var rng = new Random(0x1420 + m * 7 + k * 11 + n * 13 + multiRows + (int)format);
        int blockBytes = format == ExactGemvFormat.Q5_0 ? 22 : 18;
        byte[] wq = new byte[m * (k / 32) * blockBytes];
        rng.NextBytes(wq);
        for (int off = 0; off < wq.Length; off += blockBytes)
            BitConverter.TryWriteBytes(wq.AsSpan(off, 2), (Half)(0.002f + 0.004f * (float)rng.NextDouble()));
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        using var kernel = Create(device, spvDir, format, decodeRows, multiRows);
        VulkanDenseGemvRowsTests.AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n,
            format == ExactGemvFormat.Q5_0 ? QuantizationType.Q5_0 : QuantizationType.Q4_0,
            (cmd, bw, bx, by, nn) => kernel.Record(cmd, bw, bx, by, m, k, nn));
    }

    /// <summary>
    /// A dispatch wider than 32,768 workgroups folds into y (<see cref="GemvColumns"/>); every row past the fold must still be computed
    /// (Bolt's 131,072-row LM head at one row per workgroup).
    /// </summary>
    [SkippableFact]
    public void Q8_0_FoldedDispatch_ComputesEveryRow()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        const int m = 40000, k = 64, n = 2;
        var rng = new Random(0x4244);
        byte[] wq = Quantize.FromFloat32(F16Bf16Fixture.RandomFloats(rng, m * k, 0.1f), (long)m * k, QuantizationType.Q8_0);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        using var kernel = Create(device, spvDir, ExactGemvFormat.Q8_0, 1, 1);
        VulkanDenseGemvRowsTests.AssertColumnsMatchOneColumnDispatch(device, wq, x, m, k, n, QuantizationType.Q8_0,
            (cmd, bw, bx, by, nn) => kernel.Record(cmd, bw, bx, by, m, k, nn));
    }

    /// <summary>
    /// Opt-in timing (<c>DOTLLM_GEMV_ROWS_BENCH=1</c>) at Bolt's shapes: the #1237 per-column GEMVs against the multi-column kernels for
    /// 1, 3 and 8 columns and NR = 1, 2, 4. Minimum over rounds of 20 back-to-back dispatches.
    /// </summary>
    [SkippableFact]
    public void Bench_BoltShapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMV_ROWS_BENCH") == "1", "DOTLLM_GEMV_ROWS_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var codebooks = Iq2Codebooks.Create(device);
        using var nvfp4Cols = MatMulNvfp4F32Kernel.Create(device, spvDir);
        using var q8Cols = MatMulQ8_0Kernel.Create(device, spvDir);
        using var iq2Cols = MatMulIq2XxsGemvF32Kernel.CreateWithCodebooks(device, spvDir, codebooks);
        int[] nrs = [1, 2, 4];
        var multi = new Dictionary<(ExactGemvFormat, int), MatMulExactGemvMultiKernel>();
        foreach (var f in new[] { ExactGemvFormat.Nvfp4, ExactGemvFormat.Iq2Xxs, ExactGemvFormat.Q8_0 })
            foreach (int nr in nrs)
                multi[(f, nr)] = Create(device, spvDir, f, nr, nr);
        try
        {
            foreach (var (f, m, k) in new[] { (ExactGemvFormat.Nvfp4, 18560, 4096), (ExactGemvFormat.Nvfp4, 4096, 8192),
                                              (ExactGemvFormat.Iq2Xxs, 5376, 4096), (ExactGemvFormat.Iq2Xxs, 4096, 5376),
                                              (ExactGemvFormat.Q8_0, 1024, 4096), (ExactGemvFormat.Q8_0, 131072, 4096) })
            {
                var rng = new Random(m + k);
                var (blockElems, blockBytes) = f switch { ExactGemvFormat.Nvfp4 => (64, 36), ExactGemvFormat.Iq2Xxs => (256, 66), _ => (32, 34) };
                var bytes = new byte[(long)m * (k / blockElems) * blockBytes];
                rng.NextBytes(bytes);
                using var bw = device.AllocateDeviceLocal((bytes.Length + 3L) & ~3L);
                using (var staging = device.Allocate((bytes.Length + 3L) & ~3L))
                {
                    device.Upload(new ReadOnlySpan<byte>(bytes), staging);
                    device.CopyBufferSynchronous(staging, bw, (ulong)bytes.Length);
                }
                using var bx = device.Allocate(8L * k * 4);
                device.Upload(F16Bf16Fixture.RandomFloats(rng, 8 * k, 1.0f), bx);
                using var by = device.AllocateDeviceLocal(8L * m * 4);
                var arms = new List<(string Name, Action<nint> Rec)>();
                foreach (int n in new[] { 1, 3, 8 })
                {
                    int nn = n;
                    arms.Add(($"cols n={nn}", f switch
                    {
                        ExactGemvFormat.Nvfp4 => cmd => nvfp4Cols.RecordGemvColumns(cmd, bw, bx, by, m, k, nn, 1f),
                        ExactGemvFormat.Iq2Xxs => cmd => iq2Cols.Record(cmd, bw, bx, by, m, k, nn),
                        _ => cmd => q8Cols.Record(cmd, bw, bx, by, m, k, nn),
                    }));
                    foreach (int nr in nrs)
                    {
                        var kk = multi[(f, nr)];
                        arms.Add(($"multi NR={nr} n={nn}", cmd => kk.Record(cmd, bw, bx, by, m, k, nn, grid: f == ExactGemvFormat.Iq2Xxs ? codebooks.Iq2XxsGrid : null,
                            ksigns: f == ExactGemvFormat.Iq2Xxs ? codebooks.Ksigns : null)));
                    }
                }
                double[] best = VulkanDenseGemvRowsTests.Time(device, arms.Select(a => a.Rec).ToArray());
                for (int i = 0; i < arms.Count; i++)
                    _output.WriteLine($"exact-gemv-bench {f} m={m} k={k} {arms[i].Name,-16} {best[i] * 1000:F0} us  {bytes.Length / (best[i] / 1000) / 1e9:F1} GB/s");
            }
        }
        finally
        {
            foreach (var kk in multi.Values) kk.Dispose();
        }
    }

    private static MatMulExactGemvMultiKernel Create(VulkanDevice device, string spvDir, ExactGemvFormat format, int decodeRows, int multiRows)
        => MatMulExactGemvMultiKernel.TryCreate(device, spvDir, format, decodeRows, multiRows)
           ?? throw new Xunit.Sdk.XunitException($"{format} exact GEMV shader unavailable");
}
