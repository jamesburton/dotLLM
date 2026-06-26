using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Core.PositionEncoding;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan split-KV (Flash-Decoding) decode
/// attention kernel <see cref="VulkanSplitKvAttentionKernel"/>.
/// </summary>
/// <remarks>
/// <para>
/// Compared against <see cref="Attention.ExecuteScalar"/> — the same scalar CPU
/// oracle the per-token kernel is validated against — at rel 1e-3 / abs 1e-4.
/// </para>
/// <para>
/// Every shape here is chosen so <see cref="VulkanSplitKvAttentionKernel.WouldSplit"/>
/// is true (<c>S &gt;= 2</c>): the point is to exercise the split + cross-split
/// merge, not a degenerate single split. Shapes are deliberately
/// <i>discriminating</i> per the cross-backend test mandate — GQA
/// (<c>numHeads &gt; 1, numKvHeads &gt; 1</c>), <c>seqKv</c> not a multiple of the
/// split count (uneven last split), splits wider than the internal
/// <c>TILE_KV=256</c> tile, and masked tails — so a broadcast / boundary / merge
/// bug cannot hide behind a symmetric shape.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSplitKvAttentionKernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void Decode_Gqa_TwoSplits_Even()
    {
        // seqKv=400 -> S=2 (ceil(400/256)), splitLen=200 (even). GQA 4/2.
        RunOne(seqKv: 400, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 399);
    }

    [SkippableFact]
    public void Decode_Gqa_UnevenLastSplit()
    {
        // seqKv=777 -> S=4, splitLen=195 -> splits [0,195)[195,390)[390,585)[585,777):
        // last split is 192 wide (uneven). Discriminates a split-boundary bug.
        RunOne(seqKv: 777, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 776);
    }

    [SkippableFact]
    public void Decode_SmolLm_Gqa8_LongContext()
    {
        // SmolLM head config (9/3) at 1024 context -> S=4.
        RunOne(seqKv: 1024, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 1023);
    }

    [SkippableFact]
    public void Decode_HeadDim128()
    {
        // Llama-style head_dim 128 (one lane handles 2 dims at sgSize=64).
        RunOne(seqKv: 600, numHeads: 8, numKvHeads: 8, headDim: 128, positionOffset: 599);
    }

    [SkippableFact]
    public void Decode_SplitWiderThanTile()
    {
        // numHeads=32 -> byOccupancy=8; seqKv=4096 -> byKv=16 -> S=8, splitLen=512.
        // Each split processes 512 KV across TWO internal TILE_KV=256 tiles, so the
        // within-split online-softmax tile loop is exercised under splitting.
        RunOne(seqKv: 4096, numHeads: 32, numKvHeads: 8, headDim: 64, positionOffset: 4095);
    }

    [SkippableFact]
    public void Decode_Causal_MaskedTail_AcrossSplits()
    {
        // Causal with the query NOT at the last position: posQ=300, seqKv=600.
        // Positions 301..599 are masked (future), spanning later splits — the
        // masked region must contribute nothing through the merge.
        RunOne(seqKv: 600, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 300);
    }

    [SkippableFact]
    public void Decode_SlidingWindow_AcrossSplits()
    {
        // Sliding window 128 with posQ=699: only the most recent 128 keys are
        // visible, so early splits are fully masked (m=-inf, l=0 partials).
        RunOne(seqKv: 700, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 699,
            slidingWindow: 128);
    }

    [SkippableFact]
    public void Decode_Alibi_AcrossSplits()
    {
        RunOne(seqKv: 600, numHeads: 6, numKvHeads: 2, headDim: 64, positionOffset: 599,
            useAlibi: true);
    }

    [SkippableFact]
    public void Decode_SoftCap_AcrossSplits()
    {
        // Gemma-2/3 style attention soft-cap applied per raw score before softmax.
        RunOne(seqKv: 600, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 599,
            softCap: 30.0f);
    }

    [SkippableFact]
    public void Decode_Bidirectional_AcrossSplits()
    {
        // positionOffset=0, bidirectional: the single query attends to ALL keys
        // (a causal mask would leave only key 0), spanning every split.
        RunOne(seqKv: 600, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0,
            maskMode: AttentionMaskMode.Bidirectional);
    }

    // ─────────────────────────────────────────────────────────────

    private static void RunOne(int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset,
        bool useAlibi = false, int slidingWindow = 0, float softCap = 0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqQ = 1; // decode-only
        // Guard: the test must actually split, or it is not testing this kernel.
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads),
            $"Shape (seqKv={seqKv}, numHeads={numHeads}) does not split (S={VulkanSplitKvAttentionKernel.ComputeSplits(seqKv, numHeads)}).");

        var rng = new Random(0x5917 + seqKv * 17 + numHeads * 7 + headDim);
        float[] qh = RandomFloats(rng, seqQ * numHeads * headDim);
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] expected = new float[seqQ * numHeads * headDim];

        int? sw = slidingWindow > 0 ? slidingWindow : null;
        float scale = 1.0f / MathF.Sqrt(headDim);
        ReadOnlySpan<float> slopes = useAlibi
            ? AlibiPositionEncoding.CreateSlopes(numHeads)
            : default;
        Attention.ExecuteScalar(qh, kh, vh, expected,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            scale, slopes, sw, softCap, maskMode, prefixLen);

        // GPU path.
        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
        using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
        using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(qh.AsSpan(), bufQ);
        device.Upload(kh.AsSpan(), bufK);
        device.Upload(vh.AsSpan(), bufV);

        kernel.Launch(bufQ, bufK, bufV, bufOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow: slidingWindow, useAlibi: useAlibi,
            softCap: softCap, maskMode: maskMode, prefixLen: prefixLen);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, seqQ, seqKv, numHeads, numKvHeads, headDim);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1]
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Split-KV attention drift exceeded tolerance " +
            $"(seqKv={seqKv},nh={numHeads},nkv={numKvHeads},hd={headDim}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
