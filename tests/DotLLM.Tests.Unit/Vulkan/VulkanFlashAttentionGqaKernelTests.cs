using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #441: the GQA-grouped tiled prefill attention (<see cref="VulkanFlashAttentionGqaKernel"/>), both implementations and both
/// KV formats, against the scalar CPU reference <see cref="Attention.ExecuteScalar"/>.
/// </summary>
/// <remarks>
/// <para>Tolerances are the attention family's existing ones, not widened: the scalar (all-F32) implementation abs 1e-4 / rel 1e-3 like
/// the scalar flash kernels; the cooperative-matrix one abs 1e-3 / rel 1e-2 like <see cref="VulkanFlashAttentionCoopmatKernel"/>.
/// The F16 case feeds the reference the same F16-rounded rows the cache holds.</para>
/// <para>Shapes are non-degenerate per the cross-backend rule: GQA groups of 6 (Bonsai 2's 24 / 4) and 8 with more than one KV head,
/// so a wrong row-to-head mapping (<c>hq % numKvHeads</c>, or the token / head split of the grouped row index) moves rows to other heads;
/// positions offset past zero; query and KV lengths that are not multiples of either implementation's tiles; head_dim 256 and 128.</para>
/// <para>Mutants (run by hand, issue #441), each failing the cases noted: a row's head taken as <c>m % numHeads</c> instead of
/// <c>hkv * G + m % G</c>; a skipped O rescale; and a causal mask off by one. See the PR for the run.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionGqaKernelTests
{
    private const float ScalarAbsTol = 1e-4f;
    private const float ScalarRelTol = 1e-3f;
    private const float CoopmatAbsTol = 1e-3f;
    private const float CoopmatRelTol = 1e-2f;

    /// <summary>Selects an implementation for a case.</summary>
    public enum Impl
    {
        /// <summary>The scalar implementation (non-coopmat devices such as the Arc).</summary>
        Scalar,

        /// <summary>The cooperative-matrix implementation (Strix, NVIDIA).</summary>
        Coopmat,
    }

    /// <summary>Prefill parity over both KV formats.</summary>
    /// <param name="impl">The implementation.</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">KV positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="positionOffset">Absolute position of the first query row.</param>
    [SkippableTheory]
    [InlineData(Impl.Scalar, 100, 612, 24, 4, 256, 512)]    // Bonsai 2 heads, chunked prefill at an offset, ragged tiles
    [InlineData(Impl.Scalar, 70, 70, 24, 4, 256, 0)]        // fresh prefill
    [InlineData(Impl.Scalar, 45, 1000, 16, 2, 256, 955)]    // group of 8
    [InlineData(Impl.Scalar, 77, 205, 8, 4, 128, 128)]      // head_dim 128
    [InlineData(Impl.Scalar, 33, 97, 4, 4, 256, 64)]        // MHA (group of 1)
    [InlineData(Impl.Scalar, 130, 2100, 24, 4, 256, 1970)]  // many KV tiles
    [InlineData(Impl.Coopmat, 100, 612, 24, 4, 256, 512)]
    [InlineData(Impl.Coopmat, 70, 70, 24, 4, 256, 0)]
    [InlineData(Impl.Coopmat, 45, 1000, 16, 2, 256, 955)]
    [InlineData(Impl.Coopmat, 77, 205, 8, 4, 128, 128)]
    [InlineData(Impl.Coopmat, 33, 97, 4, 4, 256, 64)]
    [InlineData(Impl.Coopmat, 130, 2100, 24, 4, 256, 1970)]
    public void Prefill_MatchesCpuReference(Impl impl, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset)
        => RunParity(impl, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, qScale: 1f);

    /// <summary>
    /// The default configuration (#1148: Q lo-half term off) at Bonsai-like magnitudes, Q ±1 and K ±3, where the #1109 study measured
    /// the split's term at 0.5–7e-3 and KL at the noise floor. The case fails with Q ±12 (see the split-on case below): that range is
    /// 2–3x beyond any measured hd256 hybrid.
    /// </summary>
    [SkippableFact]
    public void Prefill_DefaultNoSplit_MatchesCpuReference()
        => RunParity(Impl.Coopmat, 61, 333, 24, 4, 256, 272, qScale: 1f, splitQ: null);

    /// <summary>
    /// Large scores beyond the measured range (Q ±12): a plain F16 Q.K^T misses the cooperative-matrix tolerance (max abs 1.3e-2),
    /// the hi/lo Q split (<c>DOTLLM_VULKAN_FA_GQA_QLO=1</c>) must hold it. Guards the opt-in path.
    /// </summary>
    /// <param name="impl">The implementation.</param>
    [SkippableTheory]
    [InlineData(Impl.Scalar)]
    [InlineData(Impl.Coopmat)]
    public void Prefill_LargeScores_MatchesCpuReference(Impl impl)
        => RunParity(impl, 61, 333, 24, 4, 256, 272, qScale: 12f, splitQ: true);

    /// <summary>
    /// Issue #1200: the attention-sink twins against the scalar CPU reference with sinks, on Flint's shape (32 query / 4 KV heads,
    /// head_dim 128) for a sliding layer (window 513 &lt; the KV span) and a full (NoPE) layer at depth, gpt-oss's (64 / 8, head_dim 64,
    /// window 128), and a fresh windowed prefill. Sinks are drawn in [-8, 8] so they move the output even against 1.5K keys (checked).
    /// </summary>
    /// <param name="impl">The implementation.</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">KV positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="positionOffset">Absolute position of the first query row.</param>
    /// <param name="window">Sliding window (0 = full layer).</param>
    [SkippableTheory]
    [InlineData(Impl.Scalar, 100, 1124, 32, 4, 128, 1024, 513)]   // Flint sliding layer, chunk at depth (window-skipped tiles)
    [InlineData(Impl.Scalar, 77, 1500, 32, 4, 128, 1423, 0)]      // Flint full (NoPE) layer
    [InlineData(Impl.Scalar, 45, 300, 64, 8, 64, 255, 128)]       // gpt-oss shape
    [InlineData(Impl.Scalar, 70, 70, 32, 4, 128, 0, 33)]          // fresh prefill, window < seq
    [InlineData(Impl.Coopmat, 100, 1124, 32, 4, 128, 1024, 513)]
    [InlineData(Impl.Coopmat, 77, 1500, 32, 4, 128, 1423, 0)]
    [InlineData(Impl.Coopmat, 45, 300, 64, 8, 64, 255, 128)]
    [InlineData(Impl.Coopmat, 70, 70, 32, 4, 128, 0, 33)]
    public void PrefillSinks_MatchesCpuReference(
        Impl impl, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset, int window)
        => RunParity(impl, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, qScale: 1f, window: window, sinks: true);

    /// <summary>
    /// Issue #1200 / #1176: the compact sliding-window ring hands the kernel a KV view whose first row is a multiple of 256 positions into
    /// the sequence, with the position offset rebased by the same amount. The sink twins must give bit-identical rows on that view and on
    /// the full-length cache (tile starts stay aligned; skipped tiles are exact no-ops).
    /// </summary>
    /// <param name="impl">The implementation.</param>
    /// <param name="storage">The KV format.</param>
    [SkippableTheory]
    [InlineData(Impl.Scalar, VulkanKvStorage.F16)]
    [InlineData(Impl.Coopmat, VulkanKvStorage.F16)]
    [InlineData(Impl.Coopmat, VulkanKvStorage.F32)]
    public void PrefillSinks_RingView_BitIdenticalToFullCache(Impl impl, VulkanKvStorage storage)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int seqQ = 96, seqKv = 1600, numHeads = 32, numKvHeads = 4, headDim = 128, window = 513, ringBase = 768;
        int offset = seqKv - seqQ;
        var rng = new Random(1200);
        float[] q = RandomFloats(rng, seqQ * numHeads * headDim, 1f);
        float[] k = RandomFloats(rng, seqKv * numKvHeads * headDim, 3f);
        float[] v = RandomFloats(rng, seqKv * numKvHeads * headDim, 3f);
        float[] sinkLogits = RandomFloats(rng, numHeads, 3f);
        int rowFloats = numKvHeads * headDim;

        using var device = VulkanDevice.Create();
        using var kernel = impl == Impl.Coopmat
            ? VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, storage, sinks: true)
            : VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, storage, sinks: true);
        Skip.If(kernel is null, $"the {impl} GQA flash sink kernel is unavailable on {device.DeviceName}");
        using var bufQ = Upload(device, q);
        using var bufS = Upload(device, sinkLogits);
        using var kFull = UploadKv(device, k, storage);
        using var vFull = UploadKv(device, v, storage);
        using var kRing = UploadKv(device, k[(ringBase * rowFloats)..], storage);
        using var vRing = UploadKv(device, v[(ringBase * rowFloats)..], storage);
        using var bufOut = device.Allocate((long)q.Length * sizeof(float));

        float[] full = new float[q.Length], ring = new float[q.Length];
        kernel!.Launch(bufQ, kFull, vFull, bufOut, seqQ, seqKv, numHeads, numKvHeads, headDim, offset, window, sinks: bufS);
        device.Download(bufOut, full);
        kernel.Launch(bufQ, kRing, vRing, bufOut, seqQ, seqKv - ringBase, numHeads, numKvHeads, headDim, offset - ringBase, window,
            sinks: bufS);
        device.Download(bufOut, ring);
        for (int i = 0; i < full.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(full[i]) == BitConverter.SingleToInt32Bits(ring[i]), $"[{i}] {full[i]:R} vs {ring[i]:R}");
    }

    private static VulkanDevice.Buffer Upload(VulkanDevice device, float[] data)
    {
        var buf = device.Allocate((long)data.Length * sizeof(float));
        device.Upload(data.AsSpan(), buf);
        return buf;
    }

    private static VulkanDevice.Buffer UploadKv(VulkanDevice device, float[] data, VulkanKvStorage storage)
    {
        if (storage == VulkanKvStorage.F32) return Upload(device, data);
        ushort[] halves = new ushort[data.Length];
        for (int i = 0; i < data.Length; i++) halves[i] = BitConverter.HalfToUInt16Bits((Half)data[i]);
        var buf = device.Allocate((long)halves.Length * sizeof(ushort));
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(halves.AsSpan()), buf);
        return buf;
    }

    private static void RunParity(
        Impl impl, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset, float qScale, bool? splitQ = null,
        int window = 0, bool sinks = false)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(441 + seqQ * 31 + seqKv * 17 + numHeads * 7 + headDim);
        float[] q = RandomFloats(rng, seqQ * numHeads * headDim, qScale);
        float[] sinkLogits = sinks ? RandomFloats(rng, numHeads, 8f) : [];
        int? windowOrNull = window > 0 ? window : null;
        int kvElems = seqKv * numKvHeads * headDim;
        float[] k = RandomFloats(rng, kvElems, 3f);
        float[] v = RandomFloats(rng, kvElems, 3f);
        float[] k16 = new float[kvElems], v16 = new float[kvElems];
        ushort[] kHalf = new ushort[kvElems], vHalf = new ushort[kvElems];
        for (int i = 0; i < kvElems; i++)
        {
            Half hk = (Half)k[i], hv = (Half)v[i];
            kHalf[i] = BitConverter.HalfToUInt16Bits(hk);
            vHalf[i] = BitConverter.HalfToUInt16Bits(hv);
            k16[i] = (float)hk;
            v16[i] = (float)hv;
        }

        float scale = 1.0f / MathF.Sqrt(headDim);
        float[] expected32 = new float[seqQ * numHeads * headDim];
        Attention.ExecuteScalar(q, k, v, expected32, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            scale, default, windowOrNull, 0f, AttentionMaskMode.Causal, 0, sinkLogits);
        float[] expected16 = new float[expected32.Length];
        Attention.ExecuteScalar(q, k16, v16, expected16, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            scale, default, windowOrNull, 0f, AttentionMaskMode.Causal, 0, sinkLogits);
        if (sinks)
        {
            // Discrimination: a sink-ignoring kernel must fail this fixture.
            float[] noSink = new float[expected32.Length];
            Attention.ExecuteScalar(q, k, v, noSink, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                scale, default, windowOrNull, 0f, AttentionMaskMode.Causal, 0);
            float gap = 0;
            for (int i = 0; i < noSink.Length; i++) gap = MathF.Max(gap, MathF.Abs(noSink[i] - expected32[i]));
            Assert.True(gap > 1e-2f, $"fixture too weak: the sinks move the output by only {gap}");
        }

        using var device = VulkanDevice.Create();
        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufK32 = device.Allocate((long)kvElems * sizeof(float));
        using var bufV32 = device.Allocate((long)kvElems * sizeof(float));
        using var bufK16 = device.Allocate((long)kvElems * sizeof(ushort));
        using var bufV16 = device.Allocate((long)kvElems * sizeof(ushort));
        using var bufOut = device.Allocate((long)expected32.Length * sizeof(float));
        using var bufSinks = device.Allocate(Math.Max(1, sinkLogits.Length) * (long)sizeof(float));
        if (sinks) device.Upload(sinkLogits.AsSpan(), bufSinks);
        device.Upload(q.AsSpan(), bufQ);
        device.Upload(k.AsSpan(), bufK32);
        device.Upload(v.AsSpan(), bufV32);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(kHalf.AsSpan()), bufK16);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(vHalf.AsSpan()), bufV16);

        float absTol = impl == Impl.Coopmat ? CoopmatAbsTol : ScalarAbsTol;
        float relTol = impl == Impl.Coopmat ? CoopmatRelTol : ScalarRelTol;
        var failures = new List<string>();
        foreach (VulkanKvStorage storage in new[] { VulkanKvStorage.F32, VulkanKvStorage.F16 })
        {
            using var kernel = impl == Impl.Coopmat
                ? VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, storage, splitQ, sinks)
                : VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, storage, sinks);
            Skip.If(kernel is null, $"the {impl} GQA flash kernel is unavailable on {device.DeviceName}");
            Assert.Equal(impl == Impl.Coopmat, kernel!.IsCoopmat);
            Assert.True(kernel.CanDispatch(headDim, numHeads, numKvHeads));

            float[] actual = new float[expected32.Length];
            device.Upload(new float[actual.Length].AsSpan(), bufOut);
            var (bk, bv) = storage == VulkanKvStorage.F16 ? (bufK16, bufV16) : (bufK32, bufV32);
            kernel.Launch(bufQ, bk, bv, bufOut, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, window,
                sinks: sinks ? bufSinks : null);
            device.Download(bufOut, actual);
            string? failure = Compare(storage == VulkanKvStorage.F16 ? expected16 : expected32, actual, absTol, relTol,
                $"{impl} {storage} on {device.DeviceName}");
            if (failure is not null) failures.Add(failure);
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static float[] RandomFloats(Random rng, int count, float scale)
    {
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return a;
    }

    private static string? Compare(float[] expected, float[] actual, float absTol, float relTol, string what)
    {
        int errors = 0, first = -1;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float rel = diff / MathF.Max(MathF.Abs(expected[i]), 1e-7f);
            if (!float.IsFinite(actual[i])) diff = rel = float.PositiveInfinity;
            maxAbs = MathF.Max(maxAbs, diff);
            maxRel = MathF.Max(maxRel, rel);
            if (!(diff <= absTol || rel <= relTol))
            {
                errors++;
                if (first < 0) first = i;
            }
        }

        return errors == 0 ? null :
            ($"{what}: {errors}/{expected.Length} outside abs {absTol} / rel {relTol} (max abs {maxAbs:G6}, max rel {maxRel:G6}, " +
            $"first [{first}] {(first >= 0 ? expected[first] : 0):G6} vs {(first >= 0 ? actual[first] : 0):G6})");
    }
}
