using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1109: the Vulkan precision-study seams (<see cref="VulkanPrecisionExperiment"/>). Each variant must actually engage (its output
/// differs from the shipped kernel's) and stay inside the error the study measured for it on Strix and the Arc; the score-statistics
/// variant must reproduce the shipped arithmetic exactly; the F16 rounding of F32 KV stores must equal <see cref="Half"/>'s conversion.
/// </summary>
/// <remarks>
/// The bounds are on the experiment arms, not on any shipped kernel (whose tolerances are untouched): kernel-level relRms measured on
/// Bonsai-shaped data (24 / 4 heads, head_dim 256) was qk16 2-9e-4, pv16t 1-3e-4, Q-lo off 2e-4, whole-range F16 O 6e-3 to 6e-2 (it
/// grows with depth), decode F16 statistics 2-10e-5 and F16 P.V 4e-4 to 2e-3. The lower bound (&gt; 0) is what discriminates: a variant
/// that silently fell back to the shipped pipeline produces exactly the default output and fails.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanPrecisionExperimentTests : IDisposable
{
    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;

    /// <summary>Resets every switch after each test (they are process-wide statics).</summary>
    public void Dispose()
    {
        VulkanPrecisionExperiment.Flash = FlashAccVariant.Default;
        VulkanPrecisionExperiment.Decode = DecodeAccVariant.Default;
        VulkanPrecisionExperiment.Gemm = GemmAccVariant.Default;
        VulkanPrecisionExperiment.KvRoundF16 = false;
        VulkanPrecisionExperiment.StatsBuffer = null;
    }

    /// <summary>The environment spellings parse to the intended variants and reject unknown values.</summary>
    [Fact]
    public void Parse_KnownAndUnknownValues()
    {
        Assert.Equal(FlashAccVariant.Default, VulkanPrecisionExperiment.ParseFlash(null));
        Assert.Equal(FlashAccVariant.Pv16Tile, VulkanPrecisionExperiment.ParseFlash("pv16t"));
        Assert.Equal(FlashAccVariant.NoQlo, VulkanPrecisionExperiment.ParseFlash("NOQLO"));
        Assert.Equal(DecodeAccVariant.Stats16Pv16, VulkanPrecisionExperiment.ParseDecode("st16pv16"));
        Assert.Equal(GemmAccVariant.F16Promote256, VulkanPrecisionExperiment.ParseGemm("f16p256"));
        Assert.Throws<ArgumentException>(() => VulkanPrecisionExperiment.ParseFlash("bf16"));
    }

    /// <summary>Every GQA flash-prefill variant engages and stays inside its measured error; the stats variant is exact.</summary>
    [SkippableFact]
    public void FlashVariants_EngageWithinMeasuredError()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F16);
        Skip.If(kernel is null, "GQA flash kernel unavailable");
        const int seqQ = 40, depth = 600;
        var (bufQ, bufK, bufV, bufOut) = Fill(device, seqQ, depth + seqQ, qScale: 1f);
        using (bufQ)
        using (bufK)
        using (bufV)
        using (bufOut)
        {
            float[] Run(FlashAccVariant v)
            {
                VulkanPrecisionExperiment.Flash = v;
                kernel!.Launch(bufQ, bufK, bufV, bufOut, seqQ, depth + seqQ, NumHeads, NumKvHeads, HeadDim, depth);
                var o = new float[seqQ * NumHeads * HeadDim];
                device.Download(bufOut, o);
                return o;
            }

            float[] reference = Run(FlashAccVariant.Default);
            var bounds = new List<(FlashAccVariant V, double Max)>
            {
                (FlashAccVariant.Qk16, 3e-3), (FlashAccVariant.Pv16Tile, 3e-3), (FlashAccVariant.Pv16, 5e-2),
            };
            if (kernel!.IsCoopmat) bounds.Add((FlashAccVariant.NoQlo, 3e-3));
            foreach (var (v, max) in bounds)
            {
                double rel = RelRms(reference, Run(v));
                Assert.True(rel > 0 && rel < max, $"{v}: relRms {rel:E3} outside (0, {max:E1})");
            }

            if (kernel.IsCoopmat)
            {
                using var stats = device.AllocateHostReadback(VulkanPrecisionExperiment.StatsWords * sizeof(uint));
                device.Upload(new byte[VulkanPrecisionExperiment.StatsWords * sizeof(uint)], stats);
                VulkanPrecisionExperiment.StatsBuffer = stats;
                float[] withStats = Run(FlashAccVariant.Stats);
                Assert.Equal(reference, withStats);
                var w = new float[VulkanPrecisionExperiment.StatsWords];
                device.Download(stats, w);
                uint rows = BitConverter.SingleToUInt32Bits(w[322]);
                Assert.Equal((uint)(seqQ * NumHeads), rows);
                Assert.True(w[0] > 0f, "max |s| not recorded");
            }
        }
    }

    /// <summary>Every split-KV decode variant engages and stays inside its measured error.</summary>
    [SkippableFact]
    public void DecodeVariants_EngageWithinMeasuredError()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.F16);
        kernel.EnableGqaByDefault();
        Skip.IfNot(kernel.UsesGqa(NumHeads, NumKvHeads, HeadDim), "GQA decode pass unavailable");
        const int seqKv = 3000;
        var (bufQ, bufK, bufV, bufOut) = Fill(device, 1, seqKv, qScale: 1f);
        using (bufQ)
        using (bufK)
        using (bufV)
        using (bufOut)
        {
            float[] Run(DecodeAccVariant v)
            {
                VulkanPrecisionExperiment.Decode = v;
                kernel.Launch(bufQ, bufK, bufV, bufOut, 1, seqKv, NumHeads, NumKvHeads, HeadDim, seqKv - 1);
                var o = new float[NumHeads * HeadDim];
                device.Download(bufOut, o);
                return o;
            }

            float[] reference = Run(DecodeAccVariant.Default);
            foreach (var v in new[] { DecodeAccVariant.Stats16, DecodeAccVariant.Pv16, DecodeAccVariant.Stats16Pv16 })
            {
                double rel = RelRms(reference, Run(v));
                Assert.True(rel > 0 && rel < 5e-3, $"{v}: relRms {rel:E3} outside (0, 5e-3)");
            }
        }
    }

    /// <summary>The in-place F16 rounding of an F32 KV range equals <see cref="Half"/>'s conversion, and leaves the rest alone.</summary>
    [SkippableFact]
    public void KvRoundF16_MatchesHalfConversion()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        var device = VulkanModelLoader.SharedDevice;
        var rng = new Random(1109);
        float[] x = new float[5000];
        for (int i = 0; i < x.Length; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(10, rng.Next(-3, 4)));
        x[7] = 1e6f;   // saturates to the F16 range, as the F16 writer does
        using var buf = device.Allocate((long)x.Length * sizeof(float));
        device.Upload(x.AsSpan(), buf);
        VulkanPrecisionExperiment.KvRoundF16 = true;
        VulkanKvRoundF16Kernel.InvalidateAll();
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            VulkanKvRoundF16Kernel.RecordIfEnabled(device, ctx.CommandBuffer, buf, 100, 4000);
            ctx.SubmitAndWait();
        }

        float[] got = new float[x.Length];
        device.Download(buf, got);
        VulkanKvRoundF16Kernel.InvalidateAll();
        for (int i = 0; i < x.Length; i++)
        {
            float expected = i is >= 100 and < 4100 ? (float)(Half)Math.Clamp(x[i], -65504f, 65504f) : x[i];

            // Normal halves are exact. Below the smallest normal half (6.1e-5) the writer's integer-domain mantissa rounding and the
            // driver's pack disagree with Half by at most one subnormal half step (2^-24); the shipped F16 KV writer
            // (convert_f32_to_f16_range) does the same, which is what this arm must reproduce.
            if (Math.Abs(x[i]) < 6.1035e-5f && i is >= 100 and < 4100)
                Assert.True(Math.Abs(expected - got[i]) <= 5.97e-8f, $"element {i}: {x[i]} -> {got[i]}, expected {expected}");
            else
                Assert.True(expected == got[i], $"element {i}: {x[i]} -> {got[i]}, expected {expected}");
        }
    }

    private static (VulkanDevice.Buffer Q, VulkanDevice.Buffer K, VulkanDevice.Buffer V, VulkanDevice.Buffer Out) Fill(
        VulkanDevice device, int seqQ, int seqKv, float qScale)
    {
        var rng = new Random(1109 + seqQ + seqKv);
        float[] q = new float[seqQ * NumHeads * HeadDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1) * qScale;
        int kvElems = seqKv * NumKvHeads * HeadDim;
        ushort[] k = new ushort[kvElems], v = new ushort[kvElems];
        for (int i = 0; i < kvElems; i++)
        {
            k[i] = BitConverter.HalfToUInt16Bits((Half)(float)((rng.NextDouble() * 2 - 1) * 3));
            v[i] = BitConverter.HalfToUInt16Bits((Half)(float)((rng.NextDouble() * 2 - 1) * 3));
        }

        var bufQ = device.Allocate((long)q.Length * sizeof(float));
        var bufK = device.Allocate((long)kvElems * sizeof(ushort));
        var bufV = device.Allocate((long)kvElems * sizeof(ushort));
        var bufOut = device.Allocate((long)q.Length * sizeof(float));
        device.Upload(q.AsSpan(), bufQ);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(k.AsSpan()), bufK);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(v.AsSpan()), bufV);
        return (bufQ, bufK, bufV, bufOut);
    }

    private static double RelRms(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double e = b[i] - a[i];
            num += e * e;
            den += (double)a[i] * a[i];
        }

        return Math.Sqrt(num / den);
    }
}
