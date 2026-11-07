using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1500: rounding contract of the Vulkan KV-cache quantizers (<c>kv_quantize_q8_0_f32</c>, <c>kv_quantize_q4_0_f32</c>,
/// <c>convert_f32_to_q8_0_kv_range</c>). Same contract as the #1340 activation quantizers: <c>d = amax / qmax</c> (IEEE),
/// <c>id = 1 / d</c> (IEEE), <c>q = roundHalfToEven(x * id)</c>. Drives each shader over the exact-tie / near-tie corpus
/// and compares the quant lanes and the stored float scale bit-for-bit with the CPU scalar reference
/// (<see cref="MatMul.QuantizeF32ToQ8_0Scalar"/> / <see cref="KvQuantize.F32ToQ4_0Scalar"/>). A KV rounding difference
/// feeds back into every later token, so one flipped lane is a defect, not noise.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class KvQuantizerTieTests
{
    private const int Group = 32;
    private readonly ITestOutputHelper _out;
    public KvQuantizerTieTests(ITestOutputHelper output) => _out = output;

    private static (int[] Q, float[] D) CpuQ8_0(float[] x)
    {
        int blocks = x.Length / Group;
        var dest = new byte[blocks * 34];
        fixed (float* src = x) fixed (byte* dst = dest) MatMul.QuantizeF32ToQ8_0Scalar(src, dst, x.Length);
        var q = new int[x.Length];
        var d = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < Group; i++) amax = MathF.Max(amax, MathF.Abs(x[b * Group + i]));
            d[b] = amax / 127f;                       // the split layout stores the float scale, not the Half
            for (int i = 0; i < Group; i++) q[b * Group + i] = (sbyte)dest[b * 34 + 2 + i];
        }
        return (q, d);
    }

    private static (int[] Q, float[] D) CpuQ4_0(float[] x)
    {
        int blocks = x.Length / Group;
        var dest = new byte[blocks * 18];
        fixed (float* src = x) fixed (byte* dst = dest) KvQuantize.F32ToQ4_0Scalar(src, dst, x.Length);
        var q = new int[x.Length];
        var d = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < Group; i++) amax = MathF.Max(amax, MathF.Abs(x[b * Group + i]));
            d[b] = amax / 7f;
            for (int j = 0; j < 16; j++)
            {
                byte p = dest[b * 18 + 2 + j];
                q[b * Group + 2 * j] = p & 0xF;          // unsigned nibble, as the shader stores it
                q[b * Group + 2 * j + 1] = p >> 4;
            }
        }
        return (q, d);
    }

    private void Compare(string kernel, float[] x, string[] labels, (int[] Q, float[] D) cpu, (int[] Q, float[] D) gpu)
    {
        int badQ = 0, badD = 0, firstBad = -1;
        var byKind = new SortedDictionary<string, (int Blocks, int DBad, int QBad)>();
        for (int b = 0; b < labels.Length; b++)
        {
            bool dBad = BitConverter.SingleToInt32Bits(cpu.D[b]) != BitConverter.SingleToInt32Bits(gpu.D[b]);
            int laneBad = 0;
            for (int i = 0; i < Group; i++) if (cpu.Q[b * Group + i] != gpu.Q[b * Group + i]) laneBad++;
            badQ += laneBad;
            if (dBad) badD++;
            string kind = labels[b].StartsWith("tie", StringComparison.Ordinal) ? "tie" : labels[b].StartsWith("generic", StringComparison.Ordinal) ? "generic" : labels[b];
            byKind.TryGetValue(kind, out var kc);
            byKind[kind] = (kc.Blocks + 1, kc.DBad + (dBad ? 1 : 0), kc.QBad + laneBad);
            if ((dBad || laneBad > 0) && firstBad < 0) firstBad = b;
        }
        if (firstBad >= 0)
        {
            int b = firstBad;
            _out.WriteLine($"[{kernel}] first mismatching block {b} '{labels[b]}': d cpu={cpu.D[b]:R} gpu={gpu.D[b]:R}");
            for (int i = 0; i < Group; i++)
                if (cpu.Q[b * Group + i] != gpu.Q[b * Group + i])
                    _out.WriteLine($"   lane {i}: x={x[b * Group + i]:R} cpu q={cpu.Q[b * Group + i]} gpu q={gpu.Q[b * Group + i]}");
        }
        foreach (var (kind, c) in byKind) _out.WriteLine($"[{kernel}]   {kind}: blocks={c.Blocks} scalesBad={c.DBad} lanesBad={c.QBad}");
        _out.WriteLine($"[{kernel}] blocks={labels.Length} mismatching lanes={badQ} mismatching scales={badD}");
        Assert.True(badQ == 0 && badD == 0, $"[{kernel}] {badQ} lanes / {badD} scales differ from the CPU reference (first block {firstBad}).");
    }

    private static uint[] DownloadUints(VulkanDevice device, VulkanDevice.Buffer buf, int count)
    {
        var f = new float[count];
        device.Download(buf, f);
        return MemoryMarshal.Cast<float, uint>(f).ToArray();
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void SplitKvQuantizer_MatchesCpuReference_OnTieCorpus(KvCacheDType dtype)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int qmax = dtype == KvCacheDType.Q8_0 ? 127 : 7;
        float[] x = ActivationQuantizerCorpus.BuildCorpus(qmax, out string[] labels);
        var cpu = dtype == KvCacheDType.Q8_0 ? CpuQ8_0(x) : CpuQ4_0(x);
        int blocks = labels.Length;

        using var device = VulkanDevice.Create();
        using var kernel = KvQuantizeKernel.Create(device, spvDir, dtype);
        int uintsPerBlock = kernel.QuantUintsPerBlock;
        using var bx = device.Allocate((long)x.Length * 4);
        using var bScales = device.Allocate((long)blocks * 4);
        using var bQuants = device.Allocate((long)blocks * uintsPerBlock * 4);
        device.Upload(x.AsSpan(), bx);
        kernel.Launch(bx, bScales, bQuants, blocks, srcOffset: 0, dstOffset: 0);

        var d = new float[blocks];
        device.Download(bScales, d);
        uint[] packed = DownloadUints(device, bQuants, blocks * uintsPerBlock);
        var q = new int[x.Length];
        if (dtype == KvCacheDType.Q8_0)
        {
            var bytes = MemoryMarshal.AsBytes(packed.AsSpan());
            for (int i = 0; i < q.Length; i++) q[i] = (sbyte)bytes[i];
        }
        else
        {
            for (int b = 0; b < blocks; b++)
                for (int g = 0; g < 4; g++)
                {
                    uint w = packed[b * 4 + g];
                    for (int j = 0; j < 8; j++) q[b * Group + g * 8 + j] = (int)((w >> (4 * j)) & 0xF);
                }
        }
        Compare($"kv_quantize_{dtype.ToString().ToLowerInvariant()}_f32", x, labels, cpu, (q, d));
    }

    [SkippableFact]
    public void Q8RangeWriter_MatchesCpuReference_OnTieCorpus()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        float[] x = ActivationQuantizerCorpus.BuildCorpus(127, out string[] labels);
        var cpu = CpuQ8_0(x);
        int blocks = labels.Length;

        using var device = VulkanDevice.Create();
        using var kernel = VulkanKvQ8WriteKernel.Create(device, spvDir);
        using var bx = device.Allocate((long)x.Length * 4);
        using var bDst = device.Allocate((long)blocks * 36);
        device.Upload(x.AsSpan(), bx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.RecordF32ToQ8(ctx.CommandBuffer, bx, bDst, dstElemOffset: 0, count: x.Length);
            ctx.SubmitAndWait();
        }
        uint[] words = DownloadUints(device, bDst, blocks * 9);
        var d = new float[blocks];
        var q = new int[x.Length];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = BitConverter.Int32BitsToSingle((int)words[b * 9]);
            for (int w = 0; w < 8; w++)
                for (int j = 0; j < 4; j++)
                    q[b * Group + w * 4 + j] = (sbyte)((words[b * 9 + 1 + w] >> (8 * j)) & 0xFF);
        }
        Compare("convert_f32_to_q8_0_kv_range", x, labels, cpu, (q, d));
    }
}
