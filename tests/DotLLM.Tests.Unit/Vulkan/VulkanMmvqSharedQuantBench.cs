using System.Diagnostics;
using System.Globalization;
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
/// <para>
/// The group speedup is measured with an <b>interleaved paired A/B</b>: per pass
/// the no-share and share record forms are timed back-to-back (order alternated
/// across passes) and the per-pass <i>ratio</i> is medianed. This is deliberate —
/// the Arc iGPU throttles under sustained load, so the older measure-all-A-then-all-B
/// form drifted (the second half ran hotter) and could invert the comparison; the
/// paired ratio cancels clock/thermal drift and isolates the quant-dispatch delta.
/// </para>
/// <para>
/// Watchdog (TDR) note: the Arc here drives the display, so Windows' default 2s
/// GPU watchdog resets the device under heavy sustained compute (intermittent
/// <c>VK_ERROR_DEVICE_LOST</c>). Submit sizes are env-tunable to stay inside it:
/// <c>DOTLLM_MMVQ_BENCH_BATCH</c> / <c>_BATCHES</c> / <c>_PASSES</c> / <c>_WARMUP</c>
/// / <c>_HEAVYBATCH</c> (all default to the consts below). <c>_GROUPS_ONLY=1</c>
/// skips the heavy lm_head/down denominator singletons (fewest submits → best
/// completion odds); <c>_DIAG=&lt;path&gt;</c> logs per-shape progress. On a box
/// where the compute GPU is not the display GPU (or with a raised TdrDelay) the
/// defaults run clean.
/// </para>
/// </summary>
/// <remarks>
/// Not a BenchmarkDotNet harness — single-process walltime, batched-fence +
/// interleaved paired-median, matching the project's existing Vulkan micro-benches.
/// iGPU clocks vary run-to-run; the median paired ratio over <see cref="Passes"/>
/// passes is the load-bearing number, single samples are not.
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

    // Env overrides (diagnostic / TDR-tuning on the Arc iGPU). All default to the
    // consts above. Set DOTLLM_MMVQ_BENCH_BATCH=1 / _BATCHES=1 / _PASSES=1 /
    // _WARMUP=0 for a single-dispatch isolation run; DOTLLM_MMVQ_BENCH_DIAG=1 to
    // log progress before each timed shape.
    private static int EnvInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;
    private static int EnvInt0(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= 0 ? v : fallback;
    private static readonly int BatchSizeEff = EnvInt("DOTLLM_MMVQ_BENCH_BATCH", BatchSize);
    private static readonly int BatchesPerPassEff = EnvInt("DOTLLM_MMVQ_BENCH_BATCHES", BatchesPerPass);
    private static readonly int PassesEff = EnvInt("DOTLLM_MMVQ_BENCH_PASSES", Passes);
    private static readonly int WarmupEff = EnvInt0("DOTLLM_MMVQ_BENCH_WARMUP", WarmupPasses);
    private static readonly int HeavyBatchEff = EnvInt("DOTLLM_MMVQ_BENCH_HEAVYBATCH", HeavyBatchSize);

    private static void Diag(string msg)
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_MMVQ_BENCH_DIAG");
        if (!string.IsNullOrEmpty(path) && !string.Equals(path, "1", StringComparison.Ordinal))
        {
            try { System.IO.File.AppendAllText(path, "[bench] " + msg + Environment.NewLine); }
            catch { /* best-effort diagnostic */ }
        }
    }

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
            string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MMVQ_SHARE_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_VULKAN_MMVQ_SHARE_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        VulkanDevice? device = null;
        QuantizeQ8_1Kernel? quant = null;
        MatMulQ8_0MmvqKernel? gemv = null;
        try
        {
            device = VulkanDevice.Create();
            Skip.IfNot(device.HasIntegerDotProduct,
                "Device lacks integer-dot-product; MMVQ path unavailable.");

            quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)!;
            gemv = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)!;
            Skip.If(quant is null || gemv is null, "MMVQ SPVs missing.");

            var sb = new StringBuilder();
            sb.AppendLine("| group | K | per-proj µs | shared µs | saved µs/group | speedup |");
            sb.AppendLine("|---|---:|---:|---:|---:|---:|");

            double totalSavedPerStep = 0;
            foreach (var (tag, k, outDims) in Groups)
            {
                // Interleaved paired A/B: per pass, time no-share then share back-to-back
                // so both samples see the same iGPU clock/thermal state, then median the
                // per-pass RATIO. The Arc throttles under sustained load, so the old
                // measure-all-no-share-then-all-share form drifted (second half ran
                // hotter → inverted the comparison); the paired ratio cancels that.
                var (perProj, shared, ratio) = TimeGroupInterleaved(device, quant, gemv, k, outDims);
                double saved = perProj - shared;
                totalSavedPerStep += saved; // one such group per layer
                sb.AppendLine(
                    CultureInfo.InvariantCulture,
                    $"| {tag} | {k} | {perProj:F2} | {shared:F2} | {saved:F2} | {ratio:F2}x |");
            }

            // Denominator: full decode-step Q8_0 GEMV cost = shared groups + singletons.
            // GROUPS_ONLY skips this (the heavy lm_head/down singletons dominate total
            // GPU submits and so the intermittent device-lost risk on the display Arc) —
            // the headline speedup table above is the load-bearing result; the % line is
            // a secondary contextualiser.
            bool groupsOnly = string.Equals(Environment.GetEnvironmentVariable("DOTLLM_MMVQ_BENCH_GROUPS_ONLY"), "1", StringComparison.Ordinal);
            double stepGemv = 0;
            if (!groupsOnly)
            {
                foreach (var (_, k, outDims) in Groups)
                    stepGemv += TimeGroup(device, quant, gemv, k, outDims, share: true);
                foreach (var (_, m, k) in Singletons)
                    stepGemv += TimeSingletonGemv(device, quant, gemv, m, k);
            }

            _output.WriteLine("Device: " + device.DeviceName);
            _output.WriteLine($"IntegerDotProduct: {device.HasIntegerDotProduct}, SubgroupSize: {device.SubgroupSize}");
            _output.WriteLine($"Schedule: {WarmupEff} warmup + {PassesEff} interleaved paired passes,"
                + $" {BatchesPerPassEff} batches × {BatchSizeEff} groups per pass (median paired ratio)");
            _output.WriteLine(string.Empty);
            _output.WriteLine(sb.ToString());
            _output.WriteLine(string.Empty);
            _output.WriteLine(
                $"Saved per layer (Q/K/V + gate/up redundant quant): {totalSavedPerStep:F2} µs");
            if (!groupsOnly)
            {
                _output.WriteLine(
                    $"Full decode-step Q8_0 GEMV cost (1 layer's worth, shapes above): {stepGemv:F2} µs");
                if (stepGemv > 0)
                    _output.WriteLine(
                        $"Saving as fraction of one layer's GEMV cost: {100.0 * totalSavedPerStep / stepGemv:F2}%");
            }
        }
        finally
        {
            // The Arc here is the display GPU, so Windows' 2s watchdog can reset it
            // (device-lost) during teardown under sustained load. A teardown reset must
            // NOT fail a bench whose measurement already completed — measurement-time
            // device-lost still propagates (it throws from RunPass before we get here
            // with a result). Dispose device LAST (kernels reference it).
            SafeDispose(gemv);
            SafeDispose(quant);
            SafeDispose(device);
        }
    }

    // IDISP007 false positive: the analyzer treats any parameter as "injected",
    // but every caller here passes objects this method itself created locally
    // (device/quant/gemv are all constructed in Bench_SharedVsPerProjectionQuant
    // and torn down via this helper) — disposal ownership is local, not external.
    private static void SafeDispose(IDisposable? d)
    {
        try { d?.Dispose(); }
        catch (DotLLM.Vulkan.Interop.VulkanException) { /* teardown GPU-reset on the display iGPU — measurement already done */ }
    }

    // Interleaved paired timing of the same group's no-share vs share record forms.
    // Returns (median no-share µs/group, median share µs/group, median paired ratio).
    // The two forms share ONE set of buffers (same input/weights/outputs); per pass
    // they are timed adjacently so the paired ratio is robust to iGPU clock drift.
    private (double perProj, double shared, double ratio) TimeGroupInterleaved(
        VulkanDevice device, QuantizeQ8_1Kernel quant, MatMulQ8_0MmvqKernel gemv,
        int k, int[] outDims)
    {
        var rng = new Random(7);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
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

            void RecNoShare(nint cmd)
            {
                for (int p = 0; p < outDims.Length; p++)
                {
                    if (p > 0) KernelSupport.ComputeToComputeBarrier(cmd);
                    quant.Record(cmd, bufX, bufXq, bufXds, k);
                    KernelSupport.ComputeToComputeBarrier(cmd);
                    gemv.Record(cmd, weights[p], bufXq, bufXds, outputs[p], outDims[p], k);
                }
            }
            void RecShare(nint cmd)
            {
                quant.Record(cmd, bufX, bufXq, bufXds, k);
                KernelSupport.ComputeToComputeBarrier(cmd);
                for (int p = 0; p < outDims.Length; p++)
                    gemv.Record(cmd, weights[p], bufXq, bufXds, outputs[p], outDims[p], k);
            }

            Diag($"interleaved k={k} outDims=[{string.Join(",", outDims)}] batch={BatchSizeEff} passes={PassesEff}");
            for (int w = 0; w < WarmupEff; w++)
            {
                RunPass(device, RecNoShare, BatchSizeEff);
                RunPass(device, RecShare, BatchSizeEff);
            }
            var noUs = new double[PassesEff];
            var shUs = new double[PassesEff];
            var ratios = new double[PassesEff];
            for (int p = 0; p < PassesEff; p++)
            {
                // Alternate the within-pair order so any residual first-vs-second
                // submit bias (cache/clock warm-up of the second submit) cancels
                // across passes — the paired ratio then isolates the quant-dispatch
                // delta, not measurement order.
                double tNo, tSh;
                if ((p & 1) == 0)
                {
                    tNo = RunPass(device, RecNoShare, BatchSizeEff);
                    tSh = RunPass(device, RecShare, BatchSizeEff);
                }
                else
                {
                    tSh = RunPass(device, RecShare, BatchSizeEff);
                    tNo = RunPass(device, RecNoShare, BatchSizeEff);
                }
                noUs[p] = tNo;
                shUs[p] = tSh;
                ratios[p] = tSh > 0 ? tNo / tSh : 0;
            }
            Array.Sort(noUs);
            Array.Sort(shUs);
            Array.Sort(ratios);
            var result = (noUs[PassesEff / 2], shUs[PassesEff / 2], ratios[PassesEff / 2]);
            Diag($"interleaved k={k} DONE perProj={result.Item1:F2} shared={result.Item2:F2} ratio={result.Item3:F2}");
            return result;
        }
        finally
        {
            foreach (var b in weights) b?.Dispose();
            foreach (var b in outputs) b?.Dispose();
        }
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

            Diag($"group k={k} outDims=[{string.Join(",", outDims)}] share={share} batch={BatchSizeEff}");
            double r = MeasureMedian(device, Record);
            Diag($"group k={k} share={share} DONE {r:F2} us/group");
            return r;
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

        // Singletons are denominator-only (the % of decode-step line), so they
        // always use the small launch batch — keeps each submit well under the
        // Arc's ~2s GPU watchdog (TDR) window even for the lm_head-class shape,
        // without diluting the load-bearing group speedup measurement above
        // (which keeps the full BatchSizeEff for a stable per-group median).
        int batch = HeavyBatchEff;
        Diag($"singleton m={m} k={k} batch={batch} warmup={WarmupEff} passes={PassesEff} batchesPerPass={BatchesPerPassEff}");
        double r = MeasureMedian(device, Record, batch);
        Diag($"singleton m={m} k={k} DONE {r:F2} us/group");
        return r;
    }

    private static double MeasureMedian(VulkanDevice device, Action<nint> record, int batchSize = -1)
    {
        if (batchSize < 0) batchSize = BatchSizeEff;
        for (int w = 0; w < WarmupEff; w++) RunPass(device, record, batchSize);
        var passUs = new double[PassesEff];
        for (int p = 0; p < PassesEff; p++) passUs[p] = RunPass(device, record, batchSize);
        Array.Sort(passUs);
        return passUs[PassesEff / 2];
    }

    // One pass = BatchesPerPass submits, each recording batchSize groups behind a
    // single fence. Returns mean µs per group over the pass.
    private static double RunPass(VulkanDevice device, Action<nint> record, int batchSize)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        for (int b = 0; b < BatchesPerPassEff; b++)
        {
            ctx.Begin();
            for (int i = 0; i < batchSize; i++)
                record(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }
        sw.Stop();
        long groups = (long)BatchesPerPassEff * batchSize;
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
