using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1102: the <c>_kvf16</c> twins of the attention kernels read an F16 K / V cache in place. Each case runs the same kernel
/// family twice on the same inputs: the F16 instance on the packed halves, and the F32 instance on the same rows rounded to F16 and
/// stored as F32 (what a dequant-to-scratch would hand it).
/// </summary>
/// <remarks>
/// <para>Two assertions per case:</para>
/// <list type="number">
/// <item>The F16 output is <b>bit-identical</b> to the F32 instance's on the rounded rows. <c>unpackHalf2x16</c> is exact and both
/// pipelines compile from one body (<c>kv_load.glsl</c>), so any difference is a load bug (stride, half select, head offset).</item>
/// <item>The F16 output matches the scalar CPU reference <see cref="Attention.ExecuteScalar"/> on the rounded rows within the kernel
/// family's existing tolerance (split-KV / flash F32 / naive: abs 1e-4, rel 1e-3; coopmat: abs 1e-3, rel 1e-2).</item>
/// </list>
/// <para>Shapes are non-degenerate per the cross-backend rule: GQA with more than one KV head (Bonsai 2's 24 / 4 and 8 / 4, so
/// <c>hq / group</c> and <c>hq % numKvHeads</c> disagree), a query not at the last position, KV lengths that are not multiples of the
/// kernels' tiles or the split length, and head_dim 256 (Bonsai 2) as well as smaller heads.</para>
/// <para>Mutants (run by hand, issue #1102): reading word <c>i</c> instead of <c>i &gt;&gt; 1</c> in <c>kv_load.glsl</c>, selecting the
/// wrong half, and a head broadcast of <c>hq % numKvHeads</c> in the F16 split-KV body each fail every case of the affected family.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanF16KvAttentionKernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;
    private const float CoopmatAbsTol = 1e-3f;
    private const float CoopmatRelTol = 1e-2f;

    private delegate void LaunchFn(VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output);

    /// <summary>Split-KV decode (seqQ 1) over an F16 cache.</summary>
    /// <param name="seqKv">Cached positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <param name="positionOffset">The query's absolute position.</param>
    [SkippableTheory]
    [InlineData(777, 24, 4, 256, 776)]   // Bonsai 2 shape, ragged last split
    [InlineData(1000, 8, 4, 256, 600)]   // query mid-cache: the masked tail spans later splits
    [InlineData(4100, 24, 4, 256, 4099)] // splits wider than the 256-row tile
    [InlineData(301, 6, 2, 64, 300)]     // a small head: lanes cover several elements per word pair
    public void SplitKv_Decode_ReadsF16InPlace(int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset)
    {
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads));
        RunParity(1, seqKv, numHeads, numKvHeads, headDim, positionOffset, AbsTol, RelTol,
            (device, spvDir, storage) =>
            {
                var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage);
                return (kernel, (q, k, v, o) => kernel.Launch(q, k, v, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset));
            });
    }

    /// <summary>The scalar flash prefill (the hd256 wide variants on a device without cooperative matrices, e.g. the Arc).</summary>
    /// <param name="variant">The wide variant (query rows per workgroup).</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">Cached positions.</param>
    /// <param name="positionOffset">The first query row's absolute position.</param>
    [SkippableTheory]
    [InlineData(FlashAttentionWideVariant.Hd256Br4, 37, 301, 264)]   // chunked prefill: offset > 0, partial Q and KV tiles
    [InlineData(FlashAttentionWideVariant.Hd256Br8, 37, 301, 264)]
    [InlineData(FlashAttentionWideVariant.Hd256Br16, 37, 301, 264)]
    [InlineData(FlashAttentionWideVariant.Hd256Br4, 70, 70, 0)]      // fresh prefill
    public void FlashF32_Hd256_Prefill_ReadsF16InPlace(FlashAttentionWideVariant variant, int seqQ, int seqKv, int positionOffset)
    {
        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        RunParity(seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, AbsTol, RelTol,
            (device, spvDir, storage) =>
            {
                var kernel = VulkanFlashAttentionF32Kernel.Create(device, spvDir, variant, kvStorage: storage);
                Assert.Equal(VulkanFlashAttentionF32Kernel.WideMaxHeadDim, kernel.SupportedMaxHeadDim);
                return (kernel, (q, k, v, o) => kernel.Launch(q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset));
            });
    }

    /// <summary>The scalar flash prefill's 128-wide base pipeline over an F16 cache.</summary>
    [SkippableFact]
    public void FlashF32_Hd128_Prefill_ReadsF16InPlace()
    {
        const int seqQ = 45, seqKv = 173, numHeads = 8, numKvHeads = 4, headDim = 128, positionOffset = 128;
        RunParity(seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, AbsTol, RelTol,
            (device, spvDir, storage) =>
            {
                var kernel = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.None, kvStorage: storage);
                return (kernel, (q, k, v, o) => kernel.Launch(q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset));
            });
    }

    /// <summary>The cooperative-matrix flash prefill (Strix's hd256 path, and the hd128 base pipeline).</summary>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">Cached positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <param name="positionOffset">The first query row's absolute position.</param>
    [SkippableTheory]
    [InlineData(100, 612, 24, 4, 256, 512)]  // chunked prefill at an offset
    [InlineData(301, 333, 8, 4, 256, 0)]     // partial tiles both ways
    [InlineData(77, 205, 8, 4, 128, 128)]    // the 128-wide base pipeline
    public void Coopmat_Prefill_ReadsF16InPlace(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset)
    {
        RunParity(seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, CoopmatAbsTol, CoopmatRelTol,
            (device, spvDir, storage) =>
            {
                Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "no 16x16x16 f16 cooperative-matrix tile on this device");
                var kernel = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, storage);
                Assert.NotNull(kernel);
                Assert.True(headDim <= kernel!.SupportedMaxHeadDim, $"the {storage} coopmat kernel must dispatch head_dim {headDim}");
                return (kernel, (q, k, v, o) => kernel.Launch(q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset));
            });
    }

    /// <summary>The per-token kernel: short decode below the split threshold, and a short prefill when no flash kernel applies.</summary>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">Cached positions.</param>
    /// <param name="positionOffset">The first query row's absolute position.</param>
    [SkippableTheory]
    [InlineData(1, 13, 12)]   // decode at seqKv <= 16: the split-KV kernel does not engage
    [InlineData(5, 29, 24)]   // a short prefill tile at an offset
    public void Naive_ReadsF16InPlace(int seqQ, int seqKv, int positionOffset)
    {
        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        RunParity(seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, AbsTol, RelTol,
            (device, spvDir, storage) =>
            {
                var kernel = AttentionF32Kernel.Create(device, spvDir, storage);
                Assert.NotEqual(AttentionF32Kernel.DispatchMode.Coopmat, kernel.Mode);
                return (kernel, (q, k, v, o) => kernel.Launch(q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset));
            });
    }

    // ─────────────────────────────────────────────────────────────

    private static void RunParity(
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset, float absTol, float relTol,
        Func<VulkanDevice, string, VulkanKvStorage, (IDisposable Kernel, LaunchFn Launch)> make)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x1102 + seqQ * 31 + seqKv * 17 + numHeads * 7 + headDim);
        float[] q = RandomFloats(rng, seqQ * numHeads * headDim, 1f);

        // K / V with a spread of magnitudes so the F16 rounding is not a no-op, then rounded once: the F16 cache holds exactly these.
        int kvElems = seqKv * numKvHeads * headDim;
        float[] k = RandomFloats(rng, kvElems, 3f);
        float[] v = RandomFloats(rng, kvElems, 3f);
        ushort[] kHalf = new ushort[kvElems], vHalf = new ushort[kvElems];
        for (int i = 0; i < kvElems; i++)
        {
            Half hk = (Half)k[i], hv = (Half)v[i];
            kHalf[i] = BitConverter.HalfToUInt16Bits(hk);
            vHalf[i] = BitConverter.HalfToUInt16Bits(hv);
            k[i] = (float)hk;
            v[i] = (float)hv;
        }

        float[] expected = new float[seqQ * numHeads * headDim];
        Attention.ExecuteScalar(q, k, v, expected, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            1.0f / MathF.Sqrt(headDim), default, null, 0f, AttentionMaskMode.Causal, 0);

        using var device = VulkanDevice.Create();
        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufK32 = device.Allocate((long)kvElems * sizeof(float));
        using var bufV32 = device.Allocate((long)kvElems * sizeof(float));
        using var bufK16 = device.Allocate((long)kvElems * sizeof(ushort));
        using var bufV16 = device.Allocate((long)kvElems * sizeof(ushort));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));
        device.Upload(q.AsSpan(), bufQ);
        device.Upload(k.AsSpan(), bufK32);
        device.Upload(v.AsSpan(), bufV32);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(kHalf.AsSpan()), bufK16);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(vHalf.AsSpan()), bufV16);

        float[] f32 = new float[expected.Length];
        var (k32, launch32) = make(device, spvDir, VulkanKvStorage.F32);
        using (k32)
        {
            launch32(bufQ, bufK32, bufV32, bufOut);
            device.Download(bufOut, f32);
        }

        float[] f16 = new float[expected.Length];
        var (k16, launch16) = make(device, spvDir, VulkanKvStorage.F16);
        using (k16)
        {
            launch16(bufQ, bufK16, bufV16, bufOut);
            device.Download(bufOut, f16);
        }

        AssertClose(expected, f16, absTol, relTol, "F16 kernel vs CPU reference on the F16-rounded rows");
        int firstDiff = -1;
        for (int i = 0; i < f16.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(f16[i]) != BitConverter.SingleToInt32Bits(f32[i]))
            {
                firstDiff = i;
                break;
            }
        }

        Assert.True(firstDiff < 0,
            firstDiff < 0 ? string.Empty
                : $"F16 kernel differs from the F32 kernel on the rounded rows at [{firstDiff}]: {f16[firstDiff]:R} vs {f32[firstDiff]:R}");
    }

    private static float[] RandomFloats(Random rng, int count, float scale)
    {
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return a;
    }

    private static void AssertClose(float[] expected, float[] actual, float absTol, float relTol, string what)
    {
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float rel = diff / MathF.Max(MathF.Abs(expected[i]), 1e-7f);
            maxAbs = MathF.Max(maxAbs, diff);
            maxRel = MathF.Max(maxRel, rel);
            if (!(diff <= absTol || rel <= relTol)) errors++;
        }

        Assert.True(errors == 0, $"{what}: {errors}/{expected.Length} outside abs {absTol} / rel {relTol} (max abs {maxAbs:G6}, max rel {maxRel:G6})");
    }
}
