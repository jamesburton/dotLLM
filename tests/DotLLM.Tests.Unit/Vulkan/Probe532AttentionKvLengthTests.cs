using System.Text;
using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #532 — does a GPU attention kernel's output for a given query
/// position depend on the KV-cache length beyond that query's causally visible
/// prefix? This is the GPU half of the CPU defect established in #525.
/// </summary>
/// <remarks>
/// These probes PRINT; assertions are minimal (harness guards only) so a
/// difference is reported rather than swallowed. Comparison is BITWISE
/// (<c>SingleToInt32Bits</c>), not tolerance-based — the CPU defect is 1.19E-07,
/// far inside every parity tolerance in this suite.
/// </remarks>
public class Probe532CpuControlTests
{
    private readonly ITestOutputHelper _out;
    public Probe532CpuControlTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// SENSITIVITY CONTROL. The same padded-vs-unpadded comparison applied to the
    /// CPU <c>Attention.Execute</c> fast path, which #525 established IS affected.
    /// If this prints 0 differences the harness is blind and no GPU zero below
    /// means anything.
    /// </summary>
    [Fact]
    public void Control_CpuAttention_IsKvLengthDependent()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== CONTROL: CPU Attention.Execute, visible prefix V, KV padded to V+pad ===");
        bool anyDiff = false;
        foreach (int v in new[] { 3, 5, 17 })
        {
            for (int pad = 1; pad <= 4; pad++)
            {
                var r = Probe532Harness.CpuArmPair(v, v + pad, numHeads: 9, numKvHeads: 3, headDim: 64);
                if (r.Differing > 0) anyDiff = true;
                sb.AppendLine($"  V={v,3} seqKv={v + pad,3}: differing={r.Differing,6}/{r.Total,-6} maxAbs={r.MaxAbs:E3}");
            }
        }
        _out.WriteLine(sb.ToString());
        Assert.True(anyDiff,
            "CONTROL FAILED: the CPU kernel showed no KV-length dependence, so this harness " +
            "cannot discriminate and every GPU result below is meaningless.");
    }
}

[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class Probe532VulkanAttentionKvLengthTests
{
    private readonly ITestOutputHelper _out;
    public Probe532VulkanAttentionKvLengthTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Vulkan DENSE per-token kernel (<c>attention_f32*.comp</c>) — the production
    /// path for decode with <c>seqKv &lt;= 16</c> and the prefill fallback when no
    /// flash kernel is available.
    /// Arm A: seqQ = V fixed, seqKv = V vs V+pad (isolates KV length alone).
    /// </summary>
    [SkippableFact]
    public void Vulkan_Dense_KvPadding()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = AttentionF32Kernel.Create(device, spvDir);

        var sb = new StringBuilder();
        sb.AppendLine($"=== Vulkan DENSE attention — DispatchMode={kernel.Mode} ===");
        sb.AppendLine("Arm A (prefill shape): seqQ=V, seqKv=V vs V+pad, posOff=0, compare all V rows");

        long worst = 0;
        foreach (var (v, kvPad) in new[]
                 {
                     (3, 4), (3, 5), (3, 6), (3, 7),
                     (5, 6), (5, 7), (5, 8), (5, 9),
                     (17, 18), (17, 19), (17, 20), (17, 21),
                     (250, 251), (250, 256), (250, 260),   // last visible tile grows
                     (3, 300),                             // adds a FULLY MASKED second TILE_KV tile
                     (250, 600),                           // two extra fully-masked tiles
                 })
        {
            var r = Probe532Harness.VulkanDenseArmPair(device, kernel, v, kvPad, numHeads: 9, numKvHeads: 3, headDim: 64, seqQ: v);
            worst = Math.Max(worst, r.Differing);
            sb.AppendLine($"  V={v,4} seqKv={kvPad,4}: differing={r.Differing,6}/{r.Total,-7} maxAbs={r.MaxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine("Arm B (decode shape): seqQ=1, posOff=V-1, seqKv=V vs V+pad");
        foreach (var (v, kvPad) in new[] { (3, 5), (3, 8), (5, 9), (9, 16), (16, 260) })
        {
            var r = Probe532Harness.VulkanDenseArmPair(device, kernel, v, kvPad, numHeads: 9, numKvHeads: 3, headDim: 64, seqQ: 1);
            worst = Math.Max(worst, r.Differing);
            sb.AppendLine($"  posQ={v - 1,4} seqKv={kvPad,4}: differing={r.Differing,6}/{r.Total,-7} maxAbs={r.MaxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: Vulkan dense attention is BITWISE INVARIANT to KV padding."
            : $"VERDICT: Vulkan dense attention IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Vulkan FLASH kernel — the production PREFILL path (<c>seqQ &gt; 1</c>), i.e.
    /// the exact shape in which the CPU defect surfaced (chunked prefill).
    /// </summary>
    [SkippableFact]
    public void Vulkan_Flash_KvPadding()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var flash = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir);
        Skip.If(flash is null, "Flash attention kernel unavailable on this device.");

        var sb = new StringBuilder();
        sb.AppendLine("=== Vulkan FLASH attention (prefill path) ===");
        sb.AppendLine("seqQ=V, seqKv=V vs V+pad, posOff=0, compare all V rows");

        long worst = 0;
        foreach (var (v, kvPad) in new[]
                 {
                     (3, 4), (3, 5), (3, 6), (3, 7),
                     (5, 6), (5, 7), (5, 8), (5, 9),
                     (17, 18), (17, 19), (17, 20), (17, 21),
                     (250, 260), (3, 300), (250, 600),
                 })
        {
            var r = Probe532Harness.VulkanFlashArmPair(device, flash!, v, kvPad, numHeads: 9, numKvHeads: 3, headDim: 64);
            worst = Math.Max(worst, r.Differing);
            sb.AppendLine($"  V={v,4} seqKv={kvPad,4}: differing={r.Differing,6}/{r.Total,-7} maxAbs={r.MaxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: Vulkan flash attention is BITWISE INVARIANT to KV padding."
            : $"VERDICT: Vulkan flash attention IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Vulkan FLASH COOPMAT kernel — the FIRST-CHOICE production PREFILL path
    /// (<c>RecordAttention</c> prefers it over the scalar FA shader for
    /// <c>seqQ &gt; 1</c> whenever the device supports it). This is the closest
    /// GPU analogue of the CPU shape in which #525 surfaced.
    /// </summary>
    [SkippableFact]
    public void Vulkan_FlashCoopmat_KvPadding()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var flash = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir);
        Skip.If(flash is null, "Cooperative-matrix flash attention unavailable on this device.");

        var sb = new StringBuilder();
        sb.AppendLine("=== Vulkan FLASH COOPMAT attention (preferred prefill path) ===");
        long worst = 0;
        foreach (var (v, kvPad) in new[]
                 {
                     (3, 4), (3, 5), (3, 6), (3, 7),
                     (5, 6), (5, 7), (5, 8), (5, 9),
                     (17, 18), (17, 19), (17, 20), (17, 21),
                     (250, 251), (250, 260), (3, 300), (250, 600),
                 })
        {
            var r = Probe532Harness.VulkanFlashCoopmatArmPair(device, flash!, v, kvPad, numHeads: 9, numKvHeads: 3, headDim: 64);
            worst = Math.Max(worst, r.Differing);
            sb.AppendLine($"  V={v,4} seqKv={kvPad,4}: differing={r.Differing,6}/{r.Total,-7} maxAbs={r.MaxAbs:E3}");
        }
        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: Vulkan flash-coopmat attention is BITWISE INVARIANT to KV padding."
            : $"VERDICT: Vulkan flash-coopmat attention IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Vulkan SPLIT-KV. Structurally exposed: <c>S = clamp(256/numHeads, 1, ceil(seqKv/16))</c>
    /// and <c>splitLen = ceil(seqKv / S)</c>, so growing <c>seqKv</c> moves the reduction
    /// boundaries through the VISIBLE region. The pairs below are guarded to ensure the
    /// boundaries actually move (else the probe measures nothing).
    /// </summary>
    [SkippableFact]
    public void Vulkan_SplitKv_KvPaddingAndBoundaryMove()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        var sb = new StringBuilder();
        sb.AppendLine("=== Vulkan SPLIT-KV (flash-decoding) attention ===");
        sb.AppendLine("Decode, posQ fixed, seqKv = posQ+1 (baseline) vs padded; nh=9 (SmolLM) so the");
        sb.AppendLine("occupancy cap S=28 binds and splitLen = ceil(seqKv/28) MOVES with seqKv.");

        long worst = 0;
        foreach (var (baseKv, padKv) in new[]
                 {
                     (1024, 1025), (1024, 1050), (1024, 1052), (1024, 1064),
                     (600, 601), (600, 628), (777, 805),
                     (256, 257), (256, 284),
                 })
        {
            int s0 = VulkanSplitKvAttentionKernel.ComputeSplits(baseKv, 9);
            int s1 = VulkanSplitKvAttentionKernel.ComputeSplits(padKv, 9);
            int len0 = (baseKv + s0 - 1) / s0;
            int len1 = (padKv + s1 - 1) / s1;
            bool moves = s0 != s1 || len0 != len1;

            var r = Probe532Harness.VulkanSplitKvArmPair(device, kernel, baseKv, padKv, numHeads: 9, numKvHeads: 3, headDim: 64);
            worst = Math.Max(worst, r.Differing);
            sb.AppendLine($"  posQ={baseKv - 1,5} seqKv {baseKv,5}->{padKv,5} " +
                          $"(S {s0}->{s1}, splitLen {len0}->{len1}, boundariesMove={moves,-5}): " +
                          $"differing={r.Differing,5}/{r.Total,-5} maxAbs={r.MaxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: Vulkan split-KV is BITWISE INVARIANT to KV padding."
            : $"VERDICT: Vulkan split-KV IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Cross-check: does split-KV agree BITWISE with the dense kernel for the same
    /// decode step? Answers "is the split-KV grouping itself a divergence source"
    /// independently of padding — relevant because split-KV is default-ON from ctx 17.
    /// </summary>
    [SkippableFact]
    public void Vulkan_SplitKv_VsDense_SameStep()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var dense = AttentionF32Kernel.Create(device, spvDir);
        using var split = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        var sb = new StringBuilder();
        sb.AppendLine($"=== Vulkan split-KV vs dense, same decode step (dense mode={dense.Mode}) ===");
        foreach (int kv in new[] { 17, 64, 256, 600, 1024 })
        {
            var r = Probe532Harness.VulkanSplitVsDense(device, dense, split, kv, numHeads: 9, numKvHeads: 3, headDim: 64);
            sb.AppendLine($"  seqKv={kv,5} S={VulkanSplitKvAttentionKernel.ComputeSplits(kv, 9),3}: " +
                          $"differing={r.Differing,5}/{r.Total,-5} maxAbs={r.MaxAbs:E3}");
        }
        _out.WriteLine(sb.ToString());
    }
}

internal static class Probe532Harness
{
    internal readonly record struct Diff(long Differing, long Total, float MaxAbs);

    private const int Seed = 0x532;

    internal static float[] Rand(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    /// <summary>Bitwise compare the first <paramref name="count"/> elements.</summary>
    internal static Diff Compare(float[] a, float[] b, int count)
    {
        long diff = 0;
        float maxAbs = 0;
        for (int i = 0; i < count; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                diff++;
                maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i]));
            }
        }
        return new Diff(diff, count, maxAbs);
    }

    // ── CPU control ─────────────────────────────────────────────────────────

    internal static Diff CpuArmPair(int visible, int paddedKv, int numHeads, int numKvHeads, int headDim)
    {
        var rng = new Random(Seed + visible * 31 + paddedKv);
        float[] q = Rand(rng, visible * numHeads * headDim);
        float[] k = Rand(rng, paddedKv * numKvHeads * headDim);
        float[] v = Rand(rng, paddedKv * numKvHeads * headDim);

        float[] outShort = new float[visible * numHeads * headDim];
        float[] outLong = new float[visible * numHeads * headDim];

        Attention.Execute(q, k, v, outShort, visible, visible, numHeads, numKvHeads, headDim, 0);
        Attention.Execute(q, k, v, outLong, visible, paddedKv, numHeads, numKvHeads, headDim, 0);
        return Compare(outShort, outLong, outShort.Length);
    }

    // ── Vulkan dense ────────────────────────────────────────────────────────

    /// <summary>
    /// seqQ rows at positionOffset 0 (prefill) or the single row at positionOffset
    /// <paramref name="visible"/>-1 (decode). K/V buffer is the SAME allocation in both
    /// arms; only the declared seqKv changes.
    /// </summary>
    internal static Diff VulkanDenseArmPair(
        VulkanDevice device, AttentionF32Kernel kernel,
        int visible, int paddedKv, int numHeads, int numKvHeads, int headDim, int seqQ)
    {
        int posOff = seqQ == 1 ? visible - 1 : 0;
        var rng = new Random(Seed + visible * 31 + paddedKv * 7 + seqQ);
        float[] qh = Rand(rng, seqQ * numHeads * headDim);
        float[] kh = Rand(rng, paddedKv * numKvHeads * headDim);
        float[] vh = Rand(rng, paddedKv * numKvHeads * headDim);
        int outLen = seqQ * numHeads * headDim;

        using var bq = device.Allocate((long)qh.Length * sizeof(float));
        using var bk = device.Allocate((long)kh.Length * sizeof(float));
        using var bv = device.Allocate((long)vh.Length * sizeof(float));
        using var bo1 = device.Allocate((long)outLen * sizeof(float));
        using var bo2 = device.Allocate((long)outLen * sizeof(float));
        device.Upload(qh.AsSpan(), bq);
        device.Upload(kh.AsSpan(), bk);
        device.Upload(vh.AsSpan(), bv);

        kernel.Launch(bq, bk, bv, bo1, seqQ, visible, numHeads, numKvHeads, headDim, positionOffset: posOff);
        kernel.Launch(bq, bk, bv, bo2, seqQ, paddedKv, numHeads, numKvHeads, headDim, positionOffset: posOff);

        float[] a = new float[outLen], b = new float[outLen];
        device.Download(bo1, a);
        device.Download(bo2, b);
        return Compare(a, b, outLen);
    }

    // ── Vulkan flash ────────────────────────────────────────────────────────

    internal static Diff VulkanFlashArmPair(
        VulkanDevice device, VulkanFlashAttentionF32Kernel kernel,
        int visible, int paddedKv, int numHeads, int numKvHeads, int headDim)
    {
        var rng = new Random(Seed + visible * 31 + paddedKv * 7 + 1000);
        float[] qh = Rand(rng, visible * numHeads * headDim);
        float[] kh = Rand(rng, paddedKv * numKvHeads * headDim);
        float[] vh = Rand(rng, paddedKv * numKvHeads * headDim);
        int outLen = visible * numHeads * headDim;

        using var bq = device.Allocate((long)qh.Length * sizeof(float));
        using var bk = device.Allocate((long)kh.Length * sizeof(float));
        using var bv = device.Allocate((long)vh.Length * sizeof(float));
        using var bo1 = device.Allocate((long)outLen * sizeof(float));
        using var bo2 = device.Allocate((long)outLen * sizeof(float));
        device.Upload(qh.AsSpan(), bq);
        device.Upload(kh.AsSpan(), bk);
        device.Upload(vh.AsSpan(), bv);

        kernel.Launch(bq, bk, bv, bo1, visible, visible, numHeads, numKvHeads, headDim, positionOffset: 0);
        kernel.Launch(bq, bk, bv, bo2, visible, paddedKv, numHeads, numKvHeads, headDim, positionOffset: 0);

        float[] a = new float[outLen], b = new float[outLen];
        device.Download(bo1, a);
        device.Download(bo2, b);
        return Compare(a, b, outLen);
    }

    internal static Diff VulkanFlashCoopmatArmPair(
        VulkanDevice device, VulkanFlashAttentionCoopmatKernel kernel,
        int visible, int paddedKv, int numHeads, int numKvHeads, int headDim)
    {
        var rng = new Random(Seed + visible * 31 + paddedKv * 7 + 1500);
        float[] qh = Rand(rng, visible * numHeads * headDim);
        float[] kh = Rand(rng, paddedKv * numKvHeads * headDim);
        float[] vh = Rand(rng, paddedKv * numKvHeads * headDim);
        int outLen = visible * numHeads * headDim;

        using var bq = device.Allocate((long)qh.Length * sizeof(float));
        using var bk = device.Allocate((long)kh.Length * sizeof(float));
        using var bv = device.Allocate((long)vh.Length * sizeof(float));
        using var bo1 = device.Allocate((long)outLen * sizeof(float));
        using var bo2 = device.Allocate((long)outLen * sizeof(float));
        device.Upload(qh.AsSpan(), bq);
        device.Upload(kh.AsSpan(), bk);
        device.Upload(vh.AsSpan(), bv);

        kernel.Launch(bq, bk, bv, bo1, visible, visible, numHeads, numKvHeads, headDim, positionOffset: 0);
        kernel.Launch(bq, bk, bv, bo2, visible, paddedKv, numHeads, numKvHeads, headDim, positionOffset: 0);

        float[] a = new float[outLen], b = new float[outLen];
        device.Download(bo1, a);
        device.Download(bo2, b);
        return Compare(a, b, outLen);
    }

    // ── Vulkan split-KV ─────────────────────────────────────────────────────

    internal static Diff VulkanSplitKvArmPair(
        VulkanDevice device, VulkanSplitKvAttentionKernel kernel,
        int baseKv, int paddedKv, int numHeads, int numKvHeads, int headDim)
    {
        int posOff = baseKv - 1;
        var rng = new Random(Seed + baseKv * 31 + paddedKv * 7 + 2000);
        float[] qh = Rand(rng, numHeads * headDim);
        float[] kh = Rand(rng, paddedKv * numKvHeads * headDim);
        float[] vh = Rand(rng, paddedKv * numKvHeads * headDim);
        int outLen = numHeads * headDim;

        using var bq = device.Allocate((long)qh.Length * sizeof(float));
        using var bk = device.Allocate((long)kh.Length * sizeof(float));
        using var bv = device.Allocate((long)vh.Length * sizeof(float));
        using var bo1 = device.Allocate((long)outLen * sizeof(float));
        using var bo2 = device.Allocate((long)outLen * sizeof(float));
        device.Upload(qh.AsSpan(), bq);
        device.Upload(kh.AsSpan(), bk);
        device.Upload(vh.AsSpan(), bv);

        kernel.Launch(bq, bk, bv, bo1, 1, baseKv, numHeads, numKvHeads, headDim, positionOffset: posOff);
        kernel.Launch(bq, bk, bv, bo2, 1, paddedKv, numHeads, numKvHeads, headDim, positionOffset: posOff);

        float[] a = new float[outLen], b = new float[outLen];
        device.Download(bo1, a);
        device.Download(bo2, b);
        return Compare(a, b, outLen);
    }

    internal static Diff VulkanSplitVsDense(
        VulkanDevice device, AttentionF32Kernel dense, VulkanSplitKvAttentionKernel split,
        int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        int posOff = seqKv - 1;
        var rng = new Random(Seed + seqKv * 31 + 3000);
        float[] qh = Rand(rng, numHeads * headDim);
        float[] kh = Rand(rng, seqKv * numKvHeads * headDim);
        float[] vh = Rand(rng, seqKv * numKvHeads * headDim);
        int outLen = numHeads * headDim;

        using var bq = device.Allocate((long)qh.Length * sizeof(float));
        using var bk = device.Allocate((long)kh.Length * sizeof(float));
        using var bv = device.Allocate((long)vh.Length * sizeof(float));
        using var bo1 = device.Allocate((long)outLen * sizeof(float));
        using var bo2 = device.Allocate((long)outLen * sizeof(float));
        device.Upload(qh.AsSpan(), bq);
        device.Upload(kh.AsSpan(), bk);
        device.Upload(vh.AsSpan(), bv);

        dense.Launch(bq, bk, bv, bo1, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOff);
        split.Launch(bq, bk, bv, bo2, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOff);

        float[] a = new float[outLen], b = new float[outLen];
        device.Download(bo1, a);
        device.Download(bo2, b);
        return Compare(a, b, outLen);
    }
}
