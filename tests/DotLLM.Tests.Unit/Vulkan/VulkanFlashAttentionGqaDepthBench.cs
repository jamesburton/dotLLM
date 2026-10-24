using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in A/B for issue #441 at the shape that matters for long sessions: a prefill tile of <c>rows</c> query rows appended at KV
/// depth <c>d</c> (position offset d, seqKv d + rows), Bonsai 2's 24 / 4 heads at head_dim 256, F16 KV.
/// </summary>
/// <remarks>
/// <para>Arms: the pre-#441 coopmat hd256 kernel (what Strix ran), the pre-#441 scalar hd256 br4 kernel (what the Arc ran), and both
/// implementations of <see cref="VulkanFlashAttentionGqaKernel"/>. Each arm is measured in one process, interleaved, order reversed on
/// alternate rounds; the metric is the minimum wall time of a submit-and-wait. TFLOP/s counts the causal QK^T + P.V work
/// (<see cref="VulkanPrefillSubmitBudget.AttentionGflops"/>), not the hi/lo split's extra MMAs.</para>
/// <para>A dispatch must stay inside the Windows 2 s TDR: an arm whose previous depth predicts more than ~0.8 s runs fewer rows at
/// the next depth (TFLOP/s is per actual work). Before timing, every arm is checked against the scalar flash kernel at the first
/// depth.</para>
/// <para>Enable with <c>DOTLLM_FLASH_GQA_DEPTH=1</c>; <c>DOTLLM_FLASH_GQA_DEPTHS</c> (default 8192,32768,65536,131072),
/// <c>DOTLLM_FLASH_GQA_ROWS</c> (default 1024), <c>DOTLLM_FLASH_GQA_ROUNDS</c> (default 3), <c>DOTLLM_FLASH_GQA_ARMS</c>
/// (comma list of arm names; default all available).</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionGqaDepthBench
{
    private const int NumHeads = 24, NumKvHeads = 4, HeadDim = 256;
    private const double TargetSeconds = 0.8;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the bench with the xUnit output sink.</summary>
    /// <param name="output">Sink for the results table.</param>
    public VulkanFlashAttentionGqaDepthBench(ITestOutputHelper output) => _output = output;

    /// <summary>Runs the depth sweep.</summary>
    [SkippableFact]
    public void AB_Depth_Bonsai2Shape()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_FLASH_GQA_DEPTH") == "1", "DOTLLM_FLASH_GQA_DEPTH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int[] depths = (Environment.GetEnvironmentVariable("DOTLLM_FLASH_GQA_DEPTHS") ?? "8192,32768,65536,131072")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int rows = EnvInt("DOTLLM_FLASH_GQA_ROWS", 1024);
        int rounds = EnvInt("DOTLLM_FLASH_GQA_ROUNDS", 3);
        string? armFilter = Environment.GetEnvironmentVariable("DOTLLM_FLASH_GQA_ARMS");

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device: {device.DeviceName} subgroup={device.SubgroupSize} rows={rows} rounds={rounds}");

        var arms = new List<(string Name, IDisposable Kernel, Action<VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, int, int, int> Launch)>();
        void Add(string name, IDisposable? kernel, Action<VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, int, int, int> launch)
        {
            if (kernel is null) return;
            if (armFilter is not null && !armFilter.Split(',').Contains(name, StringComparer.OrdinalIgnoreCase))
            {
                kernel.Dispose();
                return;
            }

            arms.Add((name, kernel, launch));
        }

        var oldCm = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, VulkanKvStorage.F16);
        if (oldCm is not null && oldCm.SupportedMaxHeadDim < HeadDim)
        {
            oldCm.Dispose();
            oldCm = null;
        }

        Add("old-cm-hd256", oldCm, (q, k, v, o, sq, skv, off) => oldCm!.Launch(q, k, v, o, sq, skv, NumHeads, NumKvHeads, HeadDim, off));
        var oldBr4 = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir, FlashAttentionWideVariant.Hd256Br4, includeHd512: false, kvStorage: VulkanKvStorage.F16);
        Add("old-scalar-br4", oldBr4, (q, k, v, o, sq, skv, off) => oldBr4!.Launch(q, k, v, o, sq, skv, NumHeads, NumKvHeads, HeadDim, off));
        var gqaCm = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16);
        Add("gqa-cm", gqaCm, (q, k, v, o, sq, skv, off) => gqaCm!.Launch(q, k, v, o, sq, skv, NumHeads, NumKvHeads, HeadDim, off));
        var gqaScalar = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.F16);
        Add("gqa-scalar", gqaScalar, (q, k, v, o, sq, skv, off) => gqaScalar!.Launch(q, k, v, o, sq, skv, NumHeads, NumKvHeads, HeadDim, off));
        Skip.If(arms.Count == 0, "no arm available");

        int maxKv = depths.Max() + rows;
        long kvElems = (long)maxKv * NumKvHeads * HeadDim;
        var rng = new Random(441);
        float[] q = new float[rows * NumHeads * HeadDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        ushort[] block = new ushort[1 << 20];
        for (int i = 0; i < block.Length; i++) block[i] = BitConverter.HalfToUInt16Bits((Half)(float)((rng.NextDouble() * 2 - 1) * 3));

        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufK = device.Allocate(kvElems * sizeof(ushort));
        using var bufV = device.Allocate(kvElems * sizeof(ushort));
        using var bufOut = device.Allocate((long)q.Length * sizeof(float));
        device.Upload(q.AsSpan(), bufQ);
        FillTiled(device, bufK, kvElems, block);
        FillTiled(device, bufV, kvElems, block);

        // Correctness gate at the first depth (short row count so the slow arms stay quick): every arm vs the scalar flash kernel.
        {
            int gateRows = Math.Min(rows, 128), d = depths[0];
            float[]? reference = null;
            foreach (var arm in arms.OrderBy(a => a.Name == "old-scalar-br4" ? 0 : 1))
            {
                arm.Launch(bufQ, bufK, bufV, bufOut, gateRows, d + gateRows, d);
                float[] got = new float[gateRows * NumHeads * HeadDim];
                device.Download(bufOut, got);
                if (reference is null)
                {
                    reference = got;
                    continue;
                }

                double maxAbs = 0;
                for (int i = 0; i < got.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - reference[i]));
                _output.WriteLine($"gate {arm.Name} vs first arm: maxAbs={maxAbs:E3}");
                Assert.True(maxAbs < 2e-2, $"{arm.Name} diverges (maxAbs={maxAbs:E3})");
            }
        }

        var lastSecondsPerGflop = arms.ToDictionary(a => a.Name, _ => 0.0);
        _output.WriteLine("| arm | depth | rows | min ms | attn TFLOP/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|");
        foreach (int d in depths)
        {
            var best = arms.ToDictionary(a => a.Name, _ => double.MaxValue);
            var armRows = new Dictionary<string, int>();
            foreach (var arm in arms)
            {
                int r = rows;
                double spg = lastSecondsPerGflop[arm.Name];
                while (spg > 0 && r > 16 && spg * VulkanPrefillSubmitBudget.AttentionGflops(r, d + r, NumHeads, HeadDim) > TargetSeconds)
                    r /= 2;
                armRows[arm.Name] = r;
                arm.Launch(bufQ, bufK, bufV, bufOut, r, d + r, d);   // warm-up
            }

            for (int round = 0; round < rounds; round++)
            {
                var order = round % 2 == 0 ? arms : Enumerable.Reverse(arms).ToList();
                foreach (var arm in order)
                {
                    int r = armRows[arm.Name];
                    var sw = Stopwatch.StartNew();
                    arm.Launch(bufQ, bufK, bufV, bufOut, r, d + r, d);
                    double s = sw.Elapsed.TotalSeconds;
                    if (s < best[arm.Name]) best[arm.Name] = s;
                }
            }

            foreach (var arm in arms)
            {
                int r = armRows[arm.Name];
                double gflop = VulkanPrefillSubmitBudget.AttentionGflops(r, d + r, NumHeads, HeadDim);
                lastSecondsPerGflop[arm.Name] = best[arm.Name] / gflop;
                _output.WriteLine($"| {arm.Name} | {d} | {r} | {best[arm.Name] * 1000:F1} | {gflop / best[arm.Name] / 1000:F2} |");
            }
        }

        foreach (var arm in arms) arm.Kernel.Dispose();
    }

    private static void FillTiled(VulkanDevice device, VulkanDevice.Buffer buf, long elems, ushort[] block)
    {
        // Upload the block repeatedly through one host-visible staging buffer of up to 64 MiB.
        int chunkElems = (int)Math.Min(32 << 20, elems);
        ushort[] chunk = new ushort[chunkElems];
        for (int i = 0; i < chunk.Length; i++) chunk[i] = block[i % block.Length];
        using var staging = device.Allocate((long)chunkElems * sizeof(ushort));
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(chunk.AsSpan()), staging);
        for (long off = 0; off < elems; off += chunkElems)
        {
            long n = Math.Min(chunkElems, elems - off);
            device.CopyBufferRangeSynchronous(staging, buf, srcOffset: 0, dstOffset: (ulong)(off * sizeof(ushort)), size: (ulong)(n * sizeof(ushort)));
        }
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;
}
