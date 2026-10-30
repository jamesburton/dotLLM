using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1161: the grouped Q4_1 v2 kernels against the v1 grouped kernel they replace. v2 gathers each packed row's token row through the
/// permutation (topK = 2 here, rows in a random order), fuses SwiGLU into gate/up, and writes down outputs straight to the (token, slot) rows
/// with the row scale applied. The oracle is v1 on the explicitly gathered rows (silu applied on the CPU, scatter + scale on the CPU). The
/// matmuls share operand precision and K order, so down must be bit-identical and the activation agrees to the exp() rounding. Experts hold
/// 0-3, 15-17, 33, 63-65 and 131 rows: v2's 64-row work items, their partial last fragment and multi-item experts all run.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeGroupedQ4_1V2Tests
{
    private const int TopK = 2;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the tests with the xUnit output sink (bench table).</summary>
    /// <param name="output">Output sink.</param>
    public VulkanMoeGroupedQ4_1V2Tests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(100, 64, "16,15,17,2")]                                      // one staging round, m not a multiple of 64
    [InlineData(512, 2048, "0,3,0,17,40,16,15,1,33,64,2,0,65,63,131")]      // Flint gate/up: m = 512, K = hidden 2048
    public void GateUp_MatchesV1PlusSwiGlu(int m, int k, string countsCsv)
    {
        using var h = Harness.Create(countsCsv, out int[] counts);
        if (h is null) return;
        var rng = new Random(0x1161 + m + k);
        int rows = h.Rows, tokens = rows / TopK;
        byte[] gate = RandomQ4_1Bank(rng, counts.Length, m, k), up = RandomQ4_1Bank(rng, counts.Length, m, k);
        float[] x = RandomFloats(rng, tokens * k);
        float[] packed = new float[(long)rows * k];
        for (int p = 0; p < rows; p++) Array.Copy(x, (long)(h.Perm[p] / TopK) * k, packed, (long)p * k, k);

        float[] g = h.RunV1(gate, packed, m, k), u = h.RunV1(up, packed, m, k);
        float[] act = h.RunGateUp(gate, up, x, m, k);

        float maxAbs = 0, worst = 0;
        for (int i = 0; i < act.Length; i++)
        {
            float want = g[i] / (1.0f + MathF.Exp(-g[i])) * u[i];
            maxAbs = MathF.Max(maxAbs, MathF.Abs(want));
            float d = MathF.Abs(act[i] - want);
            worst = MathF.Max(worst, d / MathF.Max(1e-3f, MathF.Abs(want)));
            Assert.True(d <= 1e-5f * MathF.Abs(want) + 1e-6f, $"idx {i} (row {i / m}, col {i % m}): v2 {act[i]:G9} vs v1+silu {want:G9}");
        }
        Assert.True(maxAbs > 1e-2f, "degenerate activations: the comparison would be vacuous.");
    }

    [SkippableTheory]
    [InlineData(2048, 512, "0,3,0,17,40,16,15,1,33,64,2,0,65,63,131", true)]   // Flint down: m = hidden 2048, K = 512, row scale on
    [InlineData(200, 64, "16,15,17,2", false)]                                 // m not a multiple of the 128-row tile, no scale
    public void Down_MatchesV1ScatterAndScale(int m, int k, string countsCsv, bool scale)
    {
        using var h = Harness.Create(countsCsv, out int[] counts);
        if (h is null) return;
        var rng = new Random(0x2161 + m + k);
        int rows = h.Rows;
        byte[] down = RandomQ4_1Bank(rng, counts.Length, m, k);
        float[] act = RandomFloats(rng, rows * k);
        float[] rowScale = new float[rows];
        for (int p = 0; p < rows; p++) rowScale[p] = MathF.ScaleB(1.0f, rng.Next(0, 6));

        float[] yp = h.RunV1(down, act, m, k);
        float[] y = h.RunDown(down, act, scale ? rowScale : null, m, k);

        for (int p = 0; p < rows; p++)
        {
            int dst = h.Perm[p];
            for (int c = 0; c < m; c++)
            {
                float want = scale ? yp[(long)p * m + c] * rowScale[p] : yp[(long)p * m + c];
                Assert.True(y[(long)dst * m + c] == want, $"packed row {p} -> row {dst}, col {c}: v2 {y[(long)dst * m + c]:G9} vs v1 {want:G9}");
            }
        }
    }

    /// <summary>
    /// Opt-in A/B (<c>DOTLLM_MOE_Q4_1_BENCH=1</c>) at Flint's routed shapes: 512 tokens x top-6 of 256 experts (random routing), hidden
    /// 2048, expert width 512. Times v1 (gate GEMM + up GEMM on the 16-row tile list over pre-gathered rows) against v2 gate/up, and v1 down
    /// against v2 down, as the minimum over rounds of 5 back-to-back dispatches; GB/s counts each weight byte once.
    /// </summary>
    [SkippableFact]
    public void Bench_FlintRoutedShapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_MOE_Q4_1_BENCH") == "1", "DOTLLM_MOE_Q4_1_BENCH=1 to enable.");
        const int E = 256, Hidden = 2048, Inter = 512, Tokens = 512, K6 = 6, Reps = 5, Rounds = 7;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedQ4_1V2Kernel.IsSupportedOn(device, spvDir), "v2 kernels unavailable.");
        var rng = new Random(1161);

        // Routing: each token picks 6 distinct experts; packed rows sorted by expert, perm = packed -> token * 6 + slot.
        var picks = new int[Tokens * K6];
        for (int t = 0; t < Tokens; t++)
        {
            var set = new HashSet<int>();
            while (set.Count < K6) set.Add(rng.Next(E));
            set.CopyTo(picks, t * K6);
        }
        int rows = picks.Length;
        int[] order = Enumerable.Range(0, rows).OrderBy(i => picks[i]).ToArray();
        uint[] offsets = new uint[E + 1];
        foreach (int e in picks) offsets[e + 1]++;
        for (int e = 0; e < E; e++) offsets[e + 1] += offsets[e];

        using var gate = UploadBytes(device, RandomQ4_1Bank(rng, E, Inter, Hidden));
        using var up = UploadBytes(device, RandomQ4_1Bank(rng, E, Inter, Hidden));
        using var down = UploadBytes(device, RandomQ4_1Bank(rng, E, Hidden, Inter));
        float[] x = RandomFloats(rng, Tokens * Hidden);
        float[] packed = new float[(long)rows * Hidden];
        for (int p = 0; p < rows; p++) Array.Copy(x, (long)(order[p] / K6) * Hidden, packed, (long)p * Hidden, Hidden);
        using var bx = UploadBytes(device, MemoryMarshal.AsBytes(x.AsSpan()).ToArray());
        using var bpacked = UploadBytes(device, MemoryMarshal.AsBytes(packed.AsSpan()).ToArray());
        using var bperm = UploadBytes(device, MemoryMarshal.AsBytes(order.AsSpan()).ToArray());
        using var bact = UploadBytes(device, MemoryMarshal.AsBytes(RandomFloats(rng, rows * Inter).AsSpan()).ToArray());
        using var off16 = device.AllocateDeviceLocal(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using var off64 = device.AllocateDeviceLocal(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using (var st = UploadBytes(device, MemoryMarshal.AsBytes(offsets.AsSpan()).ToArray()))
        {
            device.CopyBufferSynchronous(st, off16, (ulong)((E + 1) * sizeof(uint)));
            device.CopyBufferSynchronous(st, off64, (ulong)((E + 1) * sizeof(uint)));
        }
        using var args16 = device.AllocateDeviceLocal(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var args64 = device.AllocateDeviceLocal(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var yInter = device.AllocateDeviceLocal((long)rows * Inter * sizeof(float));
        using var yInter2 = device.AllocateDeviceLocal((long)rows * Inter * sizeof(float));
        using var yHidden = device.AllocateDeviceLocal((long)rows * Hidden * sizeof(float));
        using var v1 = MoeGroupedMatmulIqCoopmatKernel.Create(device, spvDir, MoeIqQuant.Q4_1);
        using var v2 = MoeGroupedQ4_1V2Kernel.Create(device, spvDir);
        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, off16, args16, E, v1.MTiles(Inter), v1.MTiles(Hidden), MoeBuildTileListKernel.TileRows);
            build.Record(ctx.CommandBuffer, off64, args64, E, MoeGroupedQ4_1V2Kernel.GateUpMTiles(Inter), MoeGroupedQ4_1V2Kernel.DownMTiles(Hidden),
                MoeGroupedQ4_1V2Kernel.RowTile);
            ctx.SubmitAndWait();
        }

        double gateUpBytes = 2.0 * E * Inter * (Hidden / 32) * 20, downBytes = (double)E * Hidden * (Inter / 32) * 20;
        var arms = new (string Name, double Bytes, Action<nint> Rec)[]
        {
            ("v1 gate+up", gateUpBytes, cmd =>
            {
                v1.RecordIndirect(cmd, gate, bpacked, off16, yInter, args16, 0, Inter, Hidden, rows, E);
                v1.RecordIndirect(cmd, up, bpacked, off16, yInter2, args16, 0, Inter, Hidden, rows, E);
            }),
            ("v2 gateup", gateUpBytes, cmd => v2.RecordGateUpIndirect(cmd, gate, up, bx, off64, bperm, yInter, args64, 0, Inter, Hidden, rows, E, K6)),
            ("v1 down", downBytes, cmd => v1.RecordIndirect(cmd, down, bact, off16, yHidden, args16, MoeBuildTileListKernel.ArgsStrideBytes, Hidden, Inter, rows, E)),
            ("v2 down", downBytes, cmd => v2.RecordDownIndirect(cmd, down, bact, off64, bperm, yHidden, null, args64, MoeBuildTileListKernel.ArgsStrideBytes,
                Hidden, Inter, rows, E)),
        };
        var best = Enumerable.Repeat(double.MaxValue, arms.Length).ToArray();
        for (int round = 0; round < Rounds; round++)
        {
            for (int i = 0; i < arms.Length; i++)
            {
                int a = (round & 1) == 0 ? i : arms.Length - 1 - i;
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int r = 0; r < Reps; r++)
                {
                    arms[a].Rec(ctx.CommandBuffer);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                }
                var sw = System.Diagnostics.Stopwatch.StartNew();
                ctx.SubmitAndWait();
                best[a] = Math.Min(best[a], sw.Elapsed.TotalMilliseconds / Reps);
            }
        }
        for (int a = 0; a < arms.Length; a++)
            _output.WriteLine($"moe-bench {arms[a].Name,-12} {best[a]:F3} ms  {arms[a].Bytes / (best[a] / 1000) / 1e9:F1} GB/s");
    }

    private static VulkanDevice.Buffer UploadBytes(VulkanDevice device, byte[] bytes)
    {
        using var staging = device.Allocate((bytes.Length + 3L) & ~3L);
        device.Upload(bytes, staging);
        var dst = device.AllocateDeviceLocal((bytes.Length + 3L) & ~3L);
        device.CopyBufferSynchronous(staging, dst, (ulong)bytes.Length);
        return dst;
    }

    private static byte[] RandomQ4_1Bank(Random rng, int experts, int m, int k)
    {
        int rowBytes = k / 32 * 20;
        var bytes = new byte[(long)experts * m * rowBytes];
        rng.NextBytes(bytes);
        for (long off = 0; off < bytes.Length; off += 20)
            IqBlockScale.Set(bytes, off, 20, (Half)(0.002f + (float)rng.NextDouble() * 0.02f));
        return bytes;
    }

    private static float[] RandomFloats(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>Device, kernels, expert offsets and a random packed-row permutation shared by the cases.</summary>
    private sealed class Harness : IDisposable
    {
        private readonly VulkanDevice _device;
        private readonly MoeGroupedMatmulIqCoopmatKernel _v1;
        private readonly MoeGroupedQ4_1V2Kernel _v2;
        private readonly MoeBuildTileListKernel _build;
        private readonly int[] _counts;
        private readonly uint[] _offsets;

        public int Rows { get; }

        public int[] Perm { get; }

        private Harness(VulkanDevice device, string spvDir, int[] counts)
        {
            _device = device;
            _counts = counts;
            _v1 = MoeGroupedMatmulIqCoopmatKernel.Create(device, spvDir, MoeIqQuant.Q4_1);
            _v2 = MoeGroupedQ4_1V2Kernel.Create(device, spvDir);
            _build = MoeBuildTileListKernel.Create(device, spvDir);
            Rows = counts.Sum();
            _offsets = new uint[counts.Length + 1];
            for (int e = 0; e < counts.Length; e++) _offsets[e + 1] = _offsets[e] + (uint)counts[e];
            Perm = Enumerable.Range(0, Rows).OrderBy(_ => Random.Shared.Next()).ToArray();
        }

        public static Harness? Create(string countsCsv, out int[] counts)
        {
            counts = Array.ConvertAll(countsCsv.Split(','), int.Parse);
            VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
            var device = VulkanDevice.Create();
            if (!MoeGroupedQ4_1V2Kernel.IsSupportedOn(device, spvDir) || !MoeGroupedMatmulIqCoopmatKernel.IsSupportedOn(device, spvDir, MoeIqQuant.Q4_1)
                || !MoeBuildTileListKernel.IsSupportedOn(spvDir))
            {
                device.Dispose();
                Skip.If(true, "coopmat / native wave64 / SPIR-V unavailable.");
            }
            Assert.True(counts.Sum() % TopK == 0, "rows must be a multiple of topK");
            return new Harness(device, spvDir, counts);
        }

        public float[] RunV1(byte[] bank, float[] packed, int m, int k)
        {
            using var w = Upload(bank);
            using var x = Upload(MemoryMarshal.AsBytes(packed.AsSpan()).ToArray());
            using var off = Upload(MemoryMarshal.AsBytes(_offsets.AsSpan()).ToArray());
            using var y = _device.Allocate((long)Rows * m * sizeof(float));
            _v1.Launch(w, x, off, y, m, k, Rows, _counts.Length, maxRowsPerExpert: _counts.Max());
            var r = new float[(long)Rows * m];
            _device.Download(y, r);
            return r;
        }

        public float[] RunGateUp(byte[] gate, byte[] up, float[] x, int m, int k)
            => Run(m, (cmd, off, perm, args, y) =>
            {
                var g = Upload(gate);
                var u = Upload(up);
                var xb = Upload(MemoryMarshal.AsBytes(x.AsSpan()).ToArray());
                _build.Record(cmd, off, args, _counts.Length, MoeGroupedQ4_1V2Kernel.GateUpMTiles(m), 1, MoeGroupedQ4_1V2Kernel.RowTile);
                KernelSupport.ComputeToIndirectAndComputeBarrier(cmd);
                _v2.RecordGateUpIndirect(cmd, g, u, xb, off, perm, y, args, 0, m, k, Rows, _counts.Length, TopK);
                return [g, u, xb];
            });

        public float[] RunDown(byte[] down, float[] act, float[]? scale, int m, int k)
            => Run(m, (cmd, off, perm, args, y) =>
            {
                var d = Upload(down);
                var a = Upload(MemoryMarshal.AsBytes(act.AsSpan()).ToArray());
                var s = scale is null ? null : Upload(MemoryMarshal.AsBytes(scale.AsSpan()).ToArray());
                _build.Record(cmd, off, args, _counts.Length, 1, MoeGroupedQ4_1V2Kernel.DownMTiles(m), MoeGroupedQ4_1V2Kernel.RowTile);
                KernelSupport.ComputeToIndirectAndComputeBarrier(cmd);
                _v2.RecordDownIndirect(cmd, d, a, off, perm, y, s, args, MoeBuildTileListKernel.ArgsStrideBytes, m, k, Rows, _counts.Length);
                return s is null ? [d, a] : [d, a, s];
            });

        // `record` returns the input buffers it uploaded; they are released after the submit completes.
        private float[] Run(int m, Func<nint, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer[]> record)
        {
            using var off = _device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(_counts.Length, Rows) * sizeof(uint));
            _device.Upload(MemoryMarshal.AsBytes(_offsets.AsSpan()), off);
            using var perm = Upload(MemoryMarshal.AsBytes(Perm.AsSpan()).ToArray());
            using var args = _device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
            using var y = _device.Allocate((long)Rows * m * sizeof(float));
            VulkanDevice.Buffer[] inputs;
            using (var ctx = _device.CreateSubmitContext())
            {
                ctx.Begin();
                inputs = record(ctx.CommandBuffer, off, perm, args, y);
                ctx.SubmitAndWait();
            }
            foreach (var b in inputs) b.Dispose();
            var r = new float[(long)Rows * m];
            _device.Download(y, r);
            return r;
        }

        private VulkanDevice.Buffer Upload(byte[] bytes)
        {
            var b = _device.Allocate((bytes.Length + 3L) & ~3L);
            _device.Upload(bytes, b);
            return b;
        }

        public void Dispose()
        {
            _v1.Dispose();
            _v2.Dispose();
            _build.Dispose();
            _device.Dispose();
        }
    }
}
