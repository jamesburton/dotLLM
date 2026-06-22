using System.Diagnostics;
using System.Text;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark for the MMVQ shared-activation-quant optimisation
/// (<see cref="VulkanTransformerModel.MmvqNoShareEnvVar"/>). Times the two
/// same-input decode projection groups — Q/K/V (post-attn-norm input) and
/// gate/up (post-ffn-norm input) — recorded the production way:
/// <c>quantize_q8_1 → barrier → GEMV per projection</c>, with a
/// compute→compute barrier between dispatches. Buffers are host-visible
/// (Arc/Xe-LPG is unified memory, so this is GPU-resident — same allocation the
/// established VulkanGemvWorkgroupSweep / VulkanSubgroupMicroBench harnesses use).
/// <para>
/// SHARE records ONE quantize for the whole group; NO_SHARE records one
/// quantize per projection (the original per-call <c>RecordMatmul</c> form).
/// The delta is the redundant quantize dispatches + their barriers the
/// optimisation removes. The bench also sums the full decode-step Q8_0 GEMV
/// cost so the saving can be reported as a fraction of a realistic decode step
/// (the denominator that decides "win vs marginal").
/// </para>
/// <para>
/// Enable with <c>DOTLLM_VULKAN_MMVQ_SHARE_BENCH=1</c>. Arc only:
/// <c>DOTLLM_VULKAN_DEVICE_VENDOR=0x8086</c>. Emits a Markdown table to stdout.
/// </para>
/// </summary>
/// <remarks>
/// Not a BenchmarkDotNet harness — single-process walltime, batched-fence +
/// paired-median, matching the project's existing Vulkan micro-benches. iGPU
/// clocks vary run-to-run; the median over <see cref="Passes"/> passes is the
/// load-bearing number, single samples are not.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMmvqSharedQuantBench
{
    // Kept modest so a pass stays well under the GPU watchdog (TDR) window on
    // the Arc iGPU — the lm_head-class shape is heavy and back-to-back launches
    // accumulate. 16 * 4 = 64 timed groups per pass is plenty for a stable
    // median of the dispatch+barrier delta.
    private const int BatchSize = 16;       // launches per submit (amortise fence)
    private const int BatchesPerPass = 4;   // 16 * 4 = 64 timed groups per pass
    private const int WarmupPasses = 2;
    private const int Passes = 7;

    private readonly ITestOutputHelper _output;

    public VulkanMmvqSharedQuantBench(ITestOutputHelper output) => _output = output;

    // Decode (seqLen==1) projection groups at SmolLM/Llama-3.2-1B-class shapes
    // (hidden=2048, GQA). Each group shares one activation row of length K.
    // CountInStep weights the group's per-decode-step contribution.
    private static readonly (string Tag, int K, int[] OutDims)[] Groups =
    [
        // Q/K/V over post-attn-norm hidden (K = hidden). GQA: Q=2048, K=V=512.
        ("qkv (Q=2048,K=V=512)", 2048, [2048, 512, 512]),
        // gate/up over post-ffn-norm hidden (K = hidden). Both = intermediate.
        ("gate_up (8192,8192)", 2048, [8192, 8192]),
    ];

    // Singleton projections in a decode step that do NOT share input (for the
    // full-step GEMV denominator): o_proj, down_proj, lm_head.
    private static readonly (string Tag, int M, int K)[] Singletons =
    [
        ("o_proj",  2048, 2048),
        ("down",    2048, 8192),
        ("lm_head", 32000, 2048),
    ];

    // Heavy singletons (lm_head-class) get a small launch batch so a single
    // submit stays well inside the GPU watchdog window.
    private const int HeavyBatchSize = 4;

    [SkippableFact]
    public void Bench_SharedVsPerProjectionQuant()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MMVQ_SHARE_BENCH") == "1",
            "DOTLLM_VULKAN_MMVQ_SHARE_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device lacks integer-dot-product; MMVQ path unavailable.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)!;
        using var gemv = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)!;
        Skip.If(quant is null || gemv is null, "MMVQ SPVs missing.");

        var sb = new StringBuilder();
        sb.AppendLine("| group | K | per-proj µs | shared µs | saved µs/group | speedup |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|");

        double totalSavedPerStep = 0;
        foreach (var (tag, k, outDims) in Groups)
        {
            double perProj = TimeGroup(device, quant, gemv, k, outDims, share: false);
            double shared = TimeGroup(device, quant, gemv, k, outDims, share: true);
            double saved = perProj - shared;
            totalSavedPerStep += saved; // one such group per layer
            sb.AppendLine(
                $"| {tag} | {k} | {perProj:F2} | {shared:F2} | {saved:F2} | {perProj / shared:F2}x |");
        }

        // Denominator: full decode-step Q8_0 GEMV cost = shared groups + singletons.
        double stepGemv = 0;
        foreach (var (_, k, outDims) in Groups)
            stepGemv += TimeGroup(device, quant, gemv, k, outDims, share: true);
        foreach (var (_, m, k) in Singletons)
            stepGemv += TimeSingletonGemv(device, quant, gemv, m, k);

        _output.WriteLine("Device: " + device.DeviceName);
        _output.WriteLine($"IntegerDotProduct: {device.HasIntegerDotProduct}, SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine($"Schedule: {WarmupPasses} warmup + {Passes} timed passes,"
            + $" {BatchesPerPass} batches × {BatchSize} groups per pass (median pass)");
        _output.WriteLine(string.Empty);
        _output.WriteLine(sb.ToString());
        _output.WriteLine(string.Empty);
        _output.WriteLine(
            $"Saved per layer (Q/K/V + gate/up redundant quant): {totalSavedPerStep:F2} µs");
        _output.WriteLine(
            $"Full decode-step Q8_0 GEMV cost (1 layer's worth, shapes above): {stepGemv:F2} µs");
        if (stepGemv > 0)
            _output.WriteLine(
                $"Saving as fraction of one layer's GEMV cost: {100.0 * totalSavedPerStep / stepGemv:F2}%");
    }

    // Records `share ? 1 : outDims.Length` quantize dispatches + one GEMV per
    // outDim, with a compute→compute barrier after the/each quantize and between
    // GEMVs — mirrors the production RecordSharedInputMmvqGroup vs per-call form.
    private double TimeGroup(
        VulkanDevice device, QuantizeQ8_1Kernel quant, MatMulQ8_0MmvqKernel gemv,
        int k, int[] outDims, bool share)
    {
        int maxM = 0;
        foreach (int m in outDims) maxM = Math.Max(maxM, m);

        var rng = new Random(7);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));

        // One weight blob + output per projection (distinct buffers, like prod).
        var weights = new VulkanDevice.Buffer[outDims.Length];
        var outputs = new VulkanDevice.Buffer[outDims.Length];
        try
        {
            UploadRandom(device, bufX, k, rng);
            for (int p = 0; p < outDims.Length; p++)
            {
                long rowBytes = (long)(k / 32) * 34;
                weights[p] = device.Allocate((long)outDims[p] * rowBytes);
                outputs[p] = device.Allocate((long)outDims[p] * sizeof(float));
                UploadRandomBytes(device, weights[p], (long)outDims[p] * rowBytes, rng);
            }

            void Record(nint cmd)
            {
                if (share)
                {
                    quant.Record(cmd, bufX, bufXq, bufXds, k);
                    KernelSupport.ComputeToComputeBarrier(cmd);
                    for (int p = 0; p < outDims.Length; p++)
                        gemv.Record(cmd, weights[p], bufXq, bufXds, outputs[p], outDims[p], k);
                }
                else
                {
                    for (int p = 0; p < outDims.Length; p++)
                    {
                        if (p > 0) KernelSupport.ComputeToComputeBarrier(cmd);
                        quant.Record(cmd, bufX, bufXq, bufXds, k);
                        KernelSupport.ComputeToComputeBarrier(cmd);
                        gemv.Record(cmd, weights[p], bufXq, bufXds, outputs[p], outDims[p], k);
                    }
                }
            }

            return MeasureMedian(device, Record);
        }
        finally
        {
            foreach (var b in weights) b?.Dispose();
            foreach (var b in outputs) b?.Dispose();
        }
    }

    private double TimeSingletonGemv(
        VulkanDevice device, QuantizeQ8_1Kernel quant, MatMulQ8_0MmvqKernel gemv,
        int m, int k)
    {
        var rng = new Random(11);
        long rowBytes = (long)(k / 32) * 34;
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufW = device.Allocate((long)m * rowBytes);
        using var bufY = device.Allocate((long)m * sizeof(float));
        UploadRandom(device, bufX, k, rng);
        UploadRandomBytes(device, bufW, (long)m * rowBytes, rng);

        void Record(nint cmd)
        {
            quant.Record(cmd, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(cmd);
            gemv.Record(cmd, bufW, bufXq, bufXds, bufY, m, k);
        }

        // Heavy shape (large M) → smaller launch batch to dodge the watchdog.
        int batch = m >= 16384 ? HeavyBatchSize : BatchSize;
        return MeasureMedian(device, Record, batch);
    }

    private static double MeasureMedian(VulkanDevice device, Action<nint> record, int batchSize = BatchSize)
    {
        for (int w = 0; w < WarmupPasses; w++) RunPass(device, record, batchSize);
        var passUs = new double[Passes];
        for (int p = 0; p < Passes; p++) passUs[p] = RunPass(device, record, batchSize);
        Array.Sort(passUs);
        return passUs[Passes / 2];
    }

    // One pass = BatchesPerPass submits, each recording batchSize groups behind a
    // single fence. Returns mean µs per group over the pass.
    private static double RunPass(VulkanDevice device, Action<nint> record, int batchSize)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        for (int b = 0; b < BatchesPerPass; b++)
        {
            ctx.Begin();
            for (int i = 0; i < batchSize; i++)
                record(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }
        sw.Stop();
        long groups = (long)BatchesPerPass * batchSize;
        return sw.Elapsed.TotalMicroseconds / groups;
    }

    private static void UploadRandom(VulkanDevice device, VulkanDevice.Buffer buf, int k, Random rng)
    {
        var x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        device.Upload(x, buf);
    }

    private static void UploadRandomBytes(VulkanDevice device, VulkanDevice.Buffer buf, long bytes, Random rng)
    {
        var data = new byte[bytes];
        rng.NextBytes(data);
        device.Upload(data, buf);
    }
}
