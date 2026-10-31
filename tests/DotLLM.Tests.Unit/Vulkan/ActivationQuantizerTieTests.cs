using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1340: rounding contract of the Q8_1 activation quantizers. A one-ULP difference in <c>d</c>, <c>1/d</c> or the tie rule is
/// digitised into a whole-integer change of one activation, which then moves every output row of the next projection.
/// The contract (CPU reference, == llama.cpp's Q8_0/Q8_1 reference quantizer): <c>d = amax / 127</c> (IEEE), <c>id = 1 / d</c>
/// (IEEE), <c>q = roundHalfToEven(x * id)</c> clamped to [-127, 127]. These tests drive the standalone single-row and
/// row-wise Vulkan quantizers over a corpus of EXACT ties (power-of-two <c>d</c>, so <c>x * id == k + 0.5</c> exactly) plus
/// near-tie values at generic scales, and compare the int8 lanes and the scale bit-for-bit with the CPU scalar reference.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class ActivationQuantizerTieTests
{
    private const int Group = 32;
    private readonly ITestOutputHelper _out;
    public ActivationQuantizerTieTests(ITestOutputHelper output) => _out = output;

    /// <summary>CPU reference: returns q lanes (sbyte per element) and the float scale amax/127.</summary>
    private static (sbyte[] Q, float[] D) CpuReference(float[] x)
    {
        int blocks = x.Length / Group;
        var q = new sbyte[x.Length];
        var d = new float[blocks];
        var dest = new byte[blocks * 36];
        fixed (float* src = x)
        fixed (byte* dst = dest)
            MatMul.QuantizeF32ToQ8_1Scalar(src, dst, x.Length);
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < Group; i++) amax = MathF.Max(amax, MathF.Abs(x[b * Group + i]));
            d[b] = amax / 127f;
            for (int i = 0; i < Group; i++) q[b * Group + i] = (sbyte)dest[b * 36 + 4 + i];
        }
        return (q, d);
    }

    private static (sbyte[] Q, float[] D) Unpack(uint[] packed, float[] ds)
    {
        var q = new sbyte[packed.Length * 4];
        MemoryMarshal.AsBytes(packed.AsSpan()).CopyTo(MemoryMarshal.AsBytes(q.AsSpan()));
        var d = new float[ds.Length / 2];
        for (int i = 0; i < d.Length; i++) d[i] = ds[2 * i];
        return (q, d);
    }

    private void Compare(string kernel, float[] x, string[] labels, (sbyte[] Q, float[] D) cpu, (sbyte[] Q, float[] D) gpu)
    {
        int badQ = 0, badD = 0, firstBadBlock = -1;
        var byKind = new SortedDictionary<string, (int Blocks, int DBad, int QBad)>();
        for (int b = 0; b < labels.Length; b++)
        {
            // Skip degenerate sub-normal handling here; only compare when the CPU scale is finite and the reciprocal is finite.
            bool dBad = BitConverter.SingleToInt32Bits(cpu.D[b]) != BitConverter.SingleToInt32Bits(gpu.D[b]);
            bool qBad = false;
            for (int i = 0; i < Group; i++)
                if (cpu.Q[b * Group + i] != gpu.Q[b * Group + i]) { qBad = true; badQ++; }
            if (dBad) badD++;
            string kind = labels[b].StartsWith("tie", StringComparison.Ordinal) ? "tie" : labels[b].StartsWith("generic", StringComparison.Ordinal) ? "generic" : labels[b];
            byKind.TryGetValue(kind, out var kc);
            int laneBad = 0;
            for (int i = 0; i < Group; i++) if (cpu.Q[b * Group + i] != gpu.Q[b * Group + i]) laneBad++;
            byKind[kind] = (kc.Blocks + 1, kc.DBad + (dBad ? 1 : 0), kc.QBad + laneBad);
            if ((dBad || qBad) && firstBadBlock < 0) firstBadBlock = b;
        }
        if (firstBadBlock >= 0)
        {
            int b = firstBadBlock;
            _out.WriteLine($"[{kernel}] first mismatching block {b} '{labels[b]}': d cpu={cpu.D[b]:R} gpu={gpu.D[b]:R}");
            for (int i = 0; i < Group; i++)
                if (cpu.Q[b * Group + i] != gpu.Q[b * Group + i])
                    _out.WriteLine($"   lane {i}: x={x[b * Group + i]:R} cpu q={cpu.Q[b * Group + i]} gpu q={gpu.Q[b * Group + i]}");
        }
        foreach (var (kind, c) in byKind) _out.WriteLine($"[{kernel}]   {kind}: blocks={c.Blocks} scalesBad={c.DBad} lanesBad={c.QBad}");
        _out.WriteLine($"[{kernel}] blocks={labels.Length} mismatching lanes={badQ} mismatching scales={badD}");
        Assert.True(badQ == 0 && badD == 0, $"[{kernel}] {badQ} lanes / {badD} scales differ from the CPU reference (first block {firstBadBlock}).");
    }

    [SkippableFact]
    public void SingleRowQuantizer_MatchesCpuReference_OnTieCorpus()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        float[] x = ActivationQuantizerCorpus.BuildCorpus(out string[] labels);
        var cpu = CpuReference(x);
        using var device = VulkanDevice.Create();
        using var kernel = QuantizeQ8_1Kernel.TryCreate(device, spvDir);
        Skip.If(kernel is null, "quantize_q8_1.spv missing");
        int k = x.Length;
        using var bx = device.Allocate((long)k * 4);
        using var bq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bd = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        device.Upload(x.AsSpan(), bx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel!.Record(ctx.CommandBuffer, bx, bq, bd, k);
            ctx.SubmitAndWait();
        }
        var packedF = new float[k / 4];
        var ds = new float[k / Group * 2];
        device.Download(bq, packedF);
        device.Download(bd, ds);
        var packed = MemoryMarshal.Cast<float, uint>(packedF).ToArray();
        Compare("quantize_q8_1", x, labels, cpu, Unpack(packed, ds));
    }

    [SkippableFact]
    public void RowsQuantizer_MatchesCpuReference_OnTieCorpus()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        float[] x = ActivationQuantizerCorpus.BuildCorpus(out string[] labels);
        // Pad to whole 64-element rows so the rows kernel gets n >= 2 rows.
        int blocks = labels.Length;
        var cpu = CpuReference(x);
        using var device = VulkanDevice.Create();
        using var kernel = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir);
        Skip.If(kernel is null, "quantize_q8_1_rows.spv missing");
        int rowK = Group;                       // one block per row: every block is its own row
        int n = blocks;
        using var bx = device.Allocate((long)x.Length * 4);
        using var bq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, rowK));
        using var bd = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, rowK));
        device.Upload(x.AsSpan(), bx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel!.Record(ctx.CommandBuffer, bx, bq, bd, n, rowK);
            ctx.SubmitAndWait();
        }
        var packedF = new float[n * rowK / 4];
        var ds = new float[n * (rowK / Group) * 2];
        device.Download(bq, packedF);
        device.Download(bd, ds);
        var packed = MemoryMarshal.Cast<float, uint>(packedF).ToArray();
        Compare("quantize_q8_1_rows", x, labels, cpu, Unpack(packed, ds));
    }
}
