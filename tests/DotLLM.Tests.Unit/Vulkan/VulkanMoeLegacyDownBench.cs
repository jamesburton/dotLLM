using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #849 microbench (opt-in: <c>DOTLLM_RUN_BENCH=1</c>): the qwen4exp down projection (512 experts, top-10, K = 640, M = 1024) through the arms a
/// legacy-quant bank can take. "F32 widened" is what the loader did BEFORE #849 (dequantise the bank to F32 and run the scalar F32 indexed kernel);
/// the rest are the now-resident arms. All arms run inside ONE process, interleaved and order-reversed (A B C C B A, three rounds), because this
/// box's two process-scoped confounds (UMA bandwidth contention, cold-vs-warm clock ramp) move absolute numbers by tens of percent; read the RATIOS.
/// Run under the GPU lock.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeLegacyDownBench
{
    private const int E = 512, TopK = 10, K = 640, M = 1024;
    private readonly ITestOutputHelper _out;
    public VulkanMoeLegacyDownBench(ITestOutputHelper output) => _out = output;

    private static byte[] RandomBank(Random rng, int blockBytes, bool q51)
    {
        long blocks = (long)E * M * (K / 32);
        byte[] bank = new byte[blocks * blockBytes];
        rng.NextBytes(bank);
        for (long b = 0; b < blocks; b++)
        {
            long off = b * blockBytes;
            ushort d = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 0.04 + 0.002));
            bank[off] = (byte)d; bank[off + 1] = (byte)(d >> 8);
            if (q51)
            {
                ushort m = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 0.2 - 0.1));
                bank[off + 2] = (byte)m; bank[off + 3] = (byte)(m >> 8);
            }
        }
        return bank;
    }

    /// <summary>Runs <paramref name="record"/> <paramref name="reps"/> times in one submission and returns the per-rep wall time in microseconds.</summary>
    private static double Time(VulkanDevice device, int reps, Action<nint> record)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        for (int i = 0; i < reps; i++)
        {
            record(ctx.CommandBuffer);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
        }
        var sw = Stopwatch.StartNew();
        ctx.SubmitAndWait();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / reps;
    }

    private static void Report(ITestOutputHelper o, string title, Dictionary<string, List<double>> t)
    {
        o.WriteLine(title);
        double? baseline = null;
        foreach (var (name, v) in t)
        {
            v.Sort();
            double med = v[v.Count / 2];
            baseline ??= med;
            o.WriteLine($"  {name,-34} median {med,9:F1} us   ({baseline / med:F2}x vs first arm)");
        }
    }

    [SkippableFact]
    public void DownProjection_DecodeAndPrefill_ArmsAB()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_RUN_BENCH") == "1", "opt-in microbench (DOTLLM_RUN_BENCH=1)");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "needs integer dot product");

        var rng = new Random(849);
        byte[] q51 = RandomBank(rng, 24, true), q80 = RandomBank(rng, 34, false);
        // The pre-#849 baseline: the same Q5_1 bank dequantised to F32 (by the CPU reference) - the scalar F32 indexed kernel reads 4 bytes/weight.
        float[] f32 = new float[(long)E * M * K];
        unsafe
        {
            fixed (byte* p = q51)
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)p, f32.LongLength, DotLLM.Core.Configuration.QuantizationType.Q5_1, f32);
        }

        using var bF32 = device.AllocateDeviceLocal((long)f32.Length * 4);
        device.Upload(f32, bF32);
        f32 = [];
        using var b51 = device.AllocateDeviceLocal(q51.Length);
        device.Upload(q51, b51);
        using var b80 = device.AllocateDeviceLocal(q80.Length);
        device.Upload(q80, b80);
        using var ones = device.AllocateDeviceLocal(E * 4);
        device.Upload(Enumerable.Repeat(1f, E).ToArray(), ones);

        using var kF32 = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        using var k51 = MoeIndexedMatmulQ5_1F32Kernel.Create(device, spvDir);
        using var k80 = MoeIndexedMatmulQ8_0F32Kernel.Create(device, spvDir);
        using var m51 = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spvDir) ?? throw new InvalidOperationException("q5_1 mmvq");
        using var m80 = MoeIndexedMatmulQ8_0MmvqKernel.TryCreate(device, spvDir) ?? throw new InvalidOperationException("q8_0 mmvq");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new InvalidOperationException("quantize");

        // ---- Decode: one token, top-10 experts. ----
        {
            int n = TopK;
            int[] idx = Enumerable.Range(0, n).Select(_ => rng.Next(E)).ToArray();
            float[] x = Enumerable.Range(0, n * K).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            using var xb = device.Allocate((long)x.Length * 4); device.Upload(x, xb);
            using var ib = device.Allocate(n * 4); device.Upload(MemoryMarshal.AsBytes<int>(idx), ib);
            using var yb = device.Allocate((long)n * M * 4);
            using var xq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, K));
            using var xds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, K));
            var arms = new (string, Action<nint>)[]
            {
                ("Q5_1 as F32 widened (pre-849)", c => kF32.Record(c, bF32, xb, ib, yb, m: M, k: K, n: n, numExperts: E)),
                ("Q5_1 scalar F32-in", c => k51.Record(c, b51, xb, ib, yb, ones, m: M, k: K, n: n, numExperts: E)),
                ("Q5_1 quantize + MMVQ", c => { quant.Record(c, xb, xq, xds, n, K); KernelSupport.ComputeToComputeBarrier(c); m51.Record(c, b51, xq, xds, ib, yb, ones, M, K, n, E); }),
                ("Q8_0 scalar F32-in", c => k80.Record(c, b80, xb, ib, yb, m: M, k: K, n: n, numExperts: E)),
                ("Q8_0 quantize + MMVQ", c => { quant.Record(c, xb, xq, xds, n, K); KernelSupport.ComputeToComputeBarrier(c); m80.Record(c, b80, xq, xds, ib, yb, M, K, n, E); }),
            };
            var t = arms.ToDictionary(a => a.Item1, _ => new List<double>());
            foreach (var a in arms) Time(device, 8, a.Item2);   // warm-up
            for (int round = 0; round < 3; round++)
            {
                foreach (var a in arms) t[a.Item1].Add(Time(device, 50, a.Item2));
                foreach (var a in arms.Reverse()) t[a.Item1].Add(Time(device, 50, a.Item2));
            }
            Report(_out, $"DECODE down projection (n={n} rows, E={E}, M={M}, K={K}):", t);
        }

        // ---- Prefill: 512 tokens x top-10, rows grouped by expert. ----
        {
            const int T = 512;
            int rows = T * TopK;
            int[] counts = new int[E];
            for (int i = 0; i < rows; i++) counts[rng.Next(E)]++;
            uint[] offsets = new uint[E + 1];
            int[] expertOfRow = new int[rows];
            for (int e = 0, r = 0; e < E; e++) { offsets[e + 1] = offsets[e] + (uint)counts[e]; for (int j = 0; j < counts[e]; j++) expertOfRow[r++] = e; }
            float[] x = Enumerable.Range(0, rows * K).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            using var xb = device.Allocate((long)x.Length * 4); device.Upload(x, xb);
            using var ib = device.Allocate(rows * 4); device.Upload(MemoryMarshal.AsBytes<int>(expertOfRow), ib);
            using var ob = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * 4); device.Upload(MemoryMarshal.AsBytes<uint>(offsets), ob);
            using var ab = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
            using var yb = device.Allocate((long)rows * M * 4);
            using var g51 = MoeGroupedMatmulLegacyQuantCoopmatKernel.Create(device, spvDir, MoeGroupedLegacyQuant.Q5_1);
            using var g80 = MoeGroupedMatmulLegacyQuantCoopmatKernel.Create(device, spvDir, MoeGroupedLegacyQuant.Q8_0);
            using var build = MoeBuildTileListKernel.Create(device, spvDir);
            var arms = new (string, Action<nint>)[]
            {
                ("Q5_1 as F32 widened (pre-849)", c => kF32.Record(c, bF32, xb, ib, yb, m: M, k: K, n: rows, numExperts: E)),
                ("Q5_1 scalar F32-in", c => k51.Record(c, b51, xb, ib, yb, ones, m: M, k: K, n: rows, numExperts: E)),
                ("Q5_1 grouped coopmat (+tile list)", c =>
                {
                    build.Record(c, ob, ab, E, g51.MTiles(M), g51.MTiles(M), g51.RowTile);
                    KernelSupport.ComputeToIndirectAndComputeBarrier(c);
                    g51.RecordIndirect(c, b51, xb, ob, yb, ones, false, ab, 0, M, K, rows, E);
                }),
                ("Q8_0 scalar F32-in", c => k80.Record(c, b80, xb, ib, yb, m: M, k: K, n: rows, numExperts: E)),
                ("Q8_0 grouped coopmat (+tile list)", c =>
                {
                    build.Record(c, ob, ab, E, g80.MTiles(M), g80.MTiles(M), g80.RowTile);
                    KernelSupport.ComputeToIndirectAndComputeBarrier(c);
                    g80.RecordIndirect(c, b80, xb, ob, yb, ones, false, ab, 0, M, K, rows, E);
                }),
            };
            var t = arms.ToDictionary(a => a.Item1, _ => new List<double>());
            foreach (var a in arms) Time(device, 1, a.Item2);
            for (int round = 0; round < 3; round++)
            {
                foreach (var a in arms) t[a.Item1].Add(Time(device, 3, a.Item2));
                foreach (var a in arms.Reverse()) t[a.Item1].Add(Time(device, 3, a.Item2));
            }
            Report(_out, $"PREFILL down projection (T={T} -> {rows} routed rows, E={E}, M={M}, K={K}):", t);
        }
    }
}
