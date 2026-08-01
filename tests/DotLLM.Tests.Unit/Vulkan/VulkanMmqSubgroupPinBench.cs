using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #241 step 3 — same-session, ORDER-REVERSED A/B of a
/// <c>requiredSubgroupSize</c> pin on the Q8_0 MMQ prefill pipeline.
/// </summary>
/// <remarks>
/// <para>
/// Step 1 (<see cref="VulkanMmqSubgroupSizeBench"/>) established that on the AMD
/// proprietary driver every static instrument available to us is BLIND to the
/// pin: <c>VkPipelineExecutablePropertiesKHR.subgroupSize</c> reports the
/// workgroup size (256 for this 16x16 kernel — not a legal wave width), and both
/// <c>VK_AMD_shader_info</c> statistics and its ISA disassembly are byte-identical
/// whether the pipeline is created unpinned, pinned to 32, or pinned to 64. That
/// leaves timing as the only remaining instrument, which is what this bench is.
/// </para>
/// <para>
/// The kernel contains no subgroup operations — it is a barrier + LDS register-
/// tiled GEMM over a 256-thread workgroup, which is a whole number of waves at
/// both 32 and 64 — so the pin is numerically a no-op and no workgroup-size
/// change is needed to pair with it. (That pairing requirement in #238/#239
/// applies to the 32-thread coopmat GEMM, not here.)
/// </para>
/// <para>
/// Methodology matches <see cref="VulkanMmqDispatchOverheadBench"/>: GPU
/// timestamps around barrier-serialized dispatches (the production forward pass
/// puts a compute-to-compute barrier before every matmul), not host stopwatches.
/// Variants are interleaved within one process and one device — and the pass
/// ORDER is reversed on alternate rounds — because cold-vs-warm launch A/B on
/// this box yields 2-3x phantom deltas from GPU clock ramp.
/// </para>
/// <para>
/// lm_head-scale shapes are reported SEPARATELY: a projection whose weights fit
/// in gfx1151's 32 MB MALL stays resident across back-to-back re-dispatches and
/// flatters any kernel, so only the lm_head rows are bandwidth-honest.
/// </para>
/// <para>Enable with <c>DOTLLM_MMQ_SUBGROUP_PIN_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMmqSubgroupPinBench
{
    private const int WarmupPasses = 2;
    private const int Rounds = 9;   // paired rounds; order reversed on odd rounds
    private const int Reps = 16;    // dispatches per timed pass

    private readonly ITestOutputHelper _output;
    public VulkanMmqSubgroupPinBench(ITestOutputHelper output) => _output = output;

    // Per-layer projections at both scales the #384-#391 investigation compared.
    private static readonly (string Tag, int M, int K, int N)[] Projections =
    [
        ("SmolLM Q/K/V   (M=576  K=576  N=512)",   576,  576, 512),
        ("SmolLM gate/up (M=1536 K=576  N=512)",  1536,  576, 512),
        ("3B     Q/K/V   (M=3072 K=3072 N=512)",  3072, 3072, 512),
        ("3B     gate/up (M=8192 K=3072 N=512)",  8192, 3072, 512),
    ];

    // lm_head scale — weights far exceed the 32 MB MALL, so these do not get
    // the residency flattery the projections above do.
    private static readonly (string Tag, int M, int K, int N)[] LmHeads =
    [
        ("SmolLM lm_head (M=49152  K=576  N=64)  ~30 MB",   49152,  576, 64),
        ("3B     lm_head (M=128256 K=3072 N=64)  ~419 MB", 128256, 3072, 64),
    ];

    [SkippableFact]
    public unsafe void Bench_MmqWave32Pin()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_MMQ_SUBGROUP_PIN_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_MMQ_SUBGROUP_PIN_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device lacks integer-dot-product support — MMQ unavailable.");
        Skip.IfNot(device.SupportsRequiredSubgroupSize(32, VkShaderStageFlags.Compute),
            "Device cannot pin a compute subgroup size of 32.");

        float tsPeriodNs = device.TimestampPeriodNs;
        Skip.IfNot(tsPeriodNs > 0f, "Device does not report a usable timestamp period.");

        using var baseline = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported.");
        using var pinned32 = MatMulQ8_0MmqKernel.TryCreate(device, spvDir, 32)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported.");

        nint queryPool = CreateQueryPool(device, Reps + 1);
        try
        {
            _output.WriteLine($"Device: {device.DeviceName}  timestampPeriod={tsPeriodNs} ns/tick");
            _output.WriteLine($"device default subgroupSize={device.SubgroupSize}; workgroup is 16x16=256 threads (8 waves @32, 4 waves @64)");
            _output.WriteLine($"{Rounds} order-reversed paired rounds x {Reps} barrier-serialized dispatches, median of rounds");
            _output.WriteLine("");

            RunTable("### Per-layer projections (weights <= 32 MB — MALL-resident, ratios flattered)", Projections);
            _output.WriteLine("");
            RunTable("### lm_head scale (weights exceed the 32 MB MALL — the honest rows)", LmHeads);

            void RunTable(string heading, (string Tag, int M, int K, int N)[] shapes)
            {
                _output.WriteLine(heading);
                _output.WriteLine("| shape | unpinned µs/dispatch | wave32-pinned µs/dispatch | pinned/unpinned |");
                _output.WriteLine("|---|---:|---:|---:|");
                foreach (var (tag, m, k, n) in shapes)
                {
                    using var shape = AllocShape(device, m, k, n);
                    var (baseUs, pinUs) = PairedAb(device, baseline, pinned32, shape, m, k, n, queryPool, tsPeriodNs);
                    _output.WriteLine($"| {tag} | {baseUs,20:F2} | {pinUs,25:F2} | {baseUs / pinUs,15:F3}x |");
                }
            }
        }
        finally
        {
            VulkanApi.vkDestroyQueryPool(device.Handle, queryPool, 0);
        }
    }

    // Interleaved, order-reversed A/B: round r runs (A,B) for even r and (B,A)
    // for odd r, so any monotonic drift (clock ramp, thermal) lands on both
    // variants equally instead of on whichever ran first.
    private static (double A, double B) PairedAb(
        VulkanDevice device, MatMulQ8_0MmqKernel a, MatMulQ8_0MmqKernel b,
        Shape shape, int m, int k, int n, nint queryPool, float tsPeriodNs)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, a, shape, m, k, n, queryPool, tsPeriodNs);
            RunPass(device, b, shape, m, k, n, queryPool, tsPeriodNs);
        }

        var aUs = new double[Rounds];
        var bUs = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            if ((r & 1) == 0)
            {
                aUs[r] = RunPass(device, a, shape, m, k, n, queryPool, tsPeriodNs);
                bUs[r] = RunPass(device, b, shape, m, k, n, queryPool, tsPeriodNs);
            }
            else
            {
                bUs[r] = RunPass(device, b, shape, m, k, n, queryPool, tsPeriodNs);
                aUs[r] = RunPass(device, a, shape, m, k, n, queryPool, tsPeriodNs);
            }
        }
        Array.Sort(aUs);
        Array.Sort(bUs);
        return (aUs[Rounds / 2], bUs[Rounds / 2]);
    }

    private static unsafe double RunPass(
        VulkanDevice device, MatMulQ8_0MmqKernel mmq, Shape shape, int m, int k, int n,
        nint queryPool, float tsPeriodNs)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        VulkanApi.vkCmdResetQueryPool(ctx.CommandBuffer, queryPool, 0, (uint)(Reps + 1));
        VulkanApi.vkCmdWriteTimestamp(ctx.CommandBuffer, VkPipelineStageFlags.BottomOfPipe, queryPool, 0);
        for (int i = 0; i < Reps; i++)
        {
            mmq.Record(ctx.CommandBuffer, shape.W, shape.Xq, shape.Xds, shape.C, m, k, n);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            VulkanApi.vkCmdWriteTimestamp(ctx.CommandBuffer, VkPipelineStageFlags.BottomOfPipe, queryPool, (uint)(i + 1));
        }
        ctx.SubmitAndWait();

        Span<ulong> ts = stackalloc ulong[Reps + 1];
        fixed (ulong* p = ts)
        {
            int rc = VulkanApi.vkGetQueryPoolResults(
                device.Handle, queryPool, 0, (uint)(Reps + 1),
                (nuint)((Reps + 1) * sizeof(ulong)), (nint)p, sizeof(ulong), flags: 0x1 | 0x2);
            if (rc < 0) throw new Xunit.Sdk.XunitException($"vkGetQueryPoolResults failed: {rc}");
        }
        return (ts[Reps] - ts[0]) * (tsPeriodNs / 1000.0) / Reps;
    }

    private readonly record struct Shape(
        VulkanDevice.Buffer W, VulkanDevice.Buffer Xq, VulkanDevice.Buffer Xds, VulkanDevice.Buffer C) : IDisposable
    {
        public void Dispose() { W.Dispose(); Xq.Dispose(); Xds.Dispose(); C.Dispose(); }
    }

    private static Shape AllocShape(VulkanDevice device, int m, int k, int n)
    {
        int blocksPerRow = k / MatMulQ8_0MmqKernel.Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * MatMulQ8_0MmqKernel.Q8_0BlockBytes;
        long wBytes = (((long)m * rowBytes) + 3) & ~3L;
        var w = device.Allocate(wBytes);
        var xq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        var xds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        var c = device.Allocate((long)n * m * sizeof(float));

        var rng = new Random(unchecked(m * 131 + k * 7 + n));
        byte[] wBuf = new byte[wBytes];
        rng.NextBytes(wBuf);
        device.Upload(new ReadOnlySpan<byte>(wBuf), w);
        byte[] xqBuf = new byte[QuantizeQ8_1RowsKernel.PackedBytes(n, k)];
        rng.NextBytes(xqBuf);
        device.Upload(new ReadOnlySpan<byte>(xqBuf), xq);
        var xds2 = new float[(long)n * blocksPerRow * 2];
        for (int i = 0; i < xds2.Length; i += 2) { xds2[i] = 1.0f; xds2[i + 1] = 0.0f; }
        device.Upload(xds2, xds);

        return new Shape(w, xq, xds, c);
    }

    private static nint CreateQueryPool(VulkanDevice device, int count)
    {
        var qci = new VkQueryPoolCreateInfo
        {
            sType = 11,    // VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO
            queryType = 2, // VK_QUERY_TYPE_TIMESTAMP
            queryCount = (uint)count,
        };
        int rc = VulkanApi.vkCreateQueryPool(device.Handle, qci, 0, out nint pool);
        if (rc < 0 || pool == 0)
            throw new Xunit.Sdk.XunitException($"vkCreateQueryPool failed: {rc}");
        return pool;
    }
}
