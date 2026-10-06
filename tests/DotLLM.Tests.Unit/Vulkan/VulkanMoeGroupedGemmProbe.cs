using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in probe (DOTLLM_MOE_GROUPED_PROBE=1): time of one grouped Q4_K MoE GEMM at the Qwen3.6-35B-A3B gate/up shape
/// (256 experts, m=512, k=2048, 4096 routed rows) for a few row-per-expert distributions, via the legacy 3-D grid and the
/// indirect tile list. Weights are random bytes (timing only).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeGroupedGemmProbe
{
    private readonly ITestOutputHelper _out;
    public VulkanMoeGroupedGemmProbe(ITestOutputHelper output) => _out = output;

    private static uint[] Offsets(int[] counts)
    {
        var o = new uint[counts.Length + 1];
        for (int i = 0; i < counts.Length; i++) o[i + 1] = o[i] + (uint)counts[i];
        return o;
    }

    [SkippableFact]
    public void Probe_GateUpShapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_MOE_GROUPED_PROBE") == "1", "opt-in probe");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulKQuantCoopmatKernel.IsSupportedOn(device, spvDir, MoeGroupedKQuant.Q4_K) && MoeBuildTileListKernel.IsSupportedOn(spvDir), "kernels unavailable");

        const int E = 256, M = 512, K = 2048;
        long bankBytes = (long)E * M * (K / 256) * 144;
        using var bank = device.AllocateDeviceLocal(bankBytes);
        var rng = new Random(1);

        var scenarios = new List<(string Name, int[] Counts)>();
        scenarios.Add(("uniform16", Enumerable.Repeat(16, E).ToArray()));
        // Skewed: Poisson-ish around 16 via sum of 8 uniform draws scaled, renormalised to 4096 rows.
        int[] skew = new int[E];
        double[] w = new double[E];
        for (int i = 0; i < E; i++) w[i] = Math.Pow(rng.NextDouble() + 0.05, 2.2);
        double ws = w.Sum();
        for (int i = 0; i < E; i++) skew[i] = (int)Math.Round(w[i] / ws * 4096);
        scenarios.Add(("skewed", skew));
        scenarios.Add(("sparse64", Enumerable.Range(0, E).Select(i => i < 64 ? 64 : 0).ToArray()));
        scenarios.Add(("uniform64", Enumerable.Repeat(64, E).ToArray()));   // pp2048-like: 4 row tiles per expert
        scenarios.Add(("uniform24", Enumerable.Repeat(24, E).ToArray()));

        foreach (var (name, counts) in scenarios)
        {
            int rows = counts.Sum();
            using var input = device.AllocateDeviceLocal((long)rows * K * 4);
            using var output = device.AllocateDeviceLocal((long)rows * M * 4);
            using var offs = device.AllocateDeviceLocal(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * 4);
            using var args = device.AllocateDeviceLocal(2 * MoeBuildTileListKernel.ArgsStrideBytes);
            device.Upload(new ReadOnlySpan<byte>(System.Runtime.InteropServices.MemoryMarshal.AsBytes(Offsets(counts).AsSpan()).ToArray()), offs);
            int tiles = counts.Sum(c => (c + 15) / 16);
            int tiles32 = counts.Sum(c => (c + 31) / 32);

            using var build = MoeBuildTileListKernel.Create(device, spvDir);

            string line = $"{name,-10} rows={rows,5} tiles16={tiles,4} tiles32={tiles32,4}: ";
            foreach (var (mode, tileM, pair) in new[] { ("indirect", 64, false), ("indirect", 64, true), ("indirect", 64, false), ("indirect", 64, true) })
            {
                using var kern = MoeGroupedMatmulKQuantCoopmatKernel.Create(device, spvDir, MoeGroupedKQuant.Q4_K, tileM, pair);
                double best = double.MaxValue;
                for (int trial = 0; trial < 5; trial++)
                {
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    if (mode == "indirect")
                    {
                        build.Record(ctx.CommandBuffer, offs, args, E, kern.MTiles(M), kern.MTiles(M), kern.RowTile);
                        KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
                    }
                    for (int r = 0; r < 20; r++)
                    {
                        if (mode == "indirect") kern.RecordIndirect(ctx.CommandBuffer, bank, input, offs, output, args, 0, M, K, rows, E);
                        else kern.Record(ctx.CommandBuffer, bank, input, offs, output, M, K, rows, E, maxRowsPerExpert: counts.Max());
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    }
                    var sw = Stopwatch.StartNew();
                    ctx.SubmitAndWait();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / 20);
                }
                double gbps = bankBytes / (best * 1e-3) / 1e9;
                line += $" m{tileM}{(pair ? "r2" : "")}={best:F3}ms({gbps:F0}GB/s)";
            }
            _out.WriteLine(line);
        }
    }
}
