using System;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// CPU↔Vulkan parity for the PrismML Hadamard activation transform.
/// </summary>
/// <remarks>
/// <see cref="Hadamard"/> is the oracle, and it is itself validated against the dense rotation
/// matrix the PrismML llama.cpp fork materializes (see <c>HadamardTests</c>). The widths here are
/// the three that Bonsai 2 actually uses — 5120, 6144 and 17408 — so a bug confined to one of the
/// real shapes cannot hide behind a convenient test size.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanHadamardFwhtF32KernelTests
{
    private const int BlockSize = 1024;

    [SkippableTheory]
    [InlineData(1, 1024)]    // single block
    [InlineData(1, 5120)]    // Bonsai 2 hidden — attn/ffn inputs, lm_head
    [InlineData(4, 5120)]    // multi-row prefill
    [InlineData(1, 6144)]    // ssm_out / attn_output input
    [InlineData(1, 17408)]   // ffn_down input
    [InlineData(3, 2048)]
    public void Forward_WithoutSigns_MatchesCpuOracle(int rows, int width)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var src = RandomFloats(new Random(0x4AD0 + rows * 31 + width), rows * width);

        var expected = new float[src.Length];
        RunCpuForward(src, signs: null, expected, rows, width);

        var actual = RunGpu(spvDir, src, signs: null, rows, width, inverse: false, permute: null);

        AssertClose(expected, actual, rows, width);
    }

    [SkippableTheory]
    [InlineData(1, 5120)]
    [InlineData(2, 6144)]
    [InlineData(1, 17408)]
    public void Forward_WithSigns_MatchesCpuOracle(int rows, int width)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var src = RandomFloats(new Random(0x51A + width), rows * width);
        var signs = RandomSigns(new Random(0x51B + width), width);

        var expected = new float[src.Length];
        RunCpuForward(src, signs, expected, rows, width);

        var actual = RunGpu(spvDir, src, signs, rows, width, inverse: false, permute: null);

        AssertClose(expected, actual, rows, width);
    }

    /// <summary>
    /// The inverse applies signs AFTER the rotation. A shader that reuses the forward order passes
    /// every sign-free test above, so this is the case that catches it.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 5120)]
    [InlineData(4, 5120)]
    public void Inverse_AppliesSignsAfterRotation_MatchesCpuOracle(int rows, int width)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var src = RandomFloats(new Random(0x11B + width), rows * width);
        var signs = RandomSigns(new Random(0x11C + width), width);

        var signBytes = ToSbyte(signs);
        var expected = new float[src.Length];
        for (int t = 0; t < rows; t++)
        {
            Hadamard.InverseRow(
                src.AsSpan(t * width, width), signBytes, expected.AsSpan(t * width, width), BlockSize);
        }

        var actual = RunGpu(spvDir, src, signs, rows, width, inverse: true, permute: null);
        AssertClose(expected, actual, rows, width);

        // Discriminator: the forward order must give a different answer, or this test proves nothing.
        var forward = new float[src.Length];
        RunCpuForward(src, signs, forward, rows, width);
        Assert.True(MaxAbsDiff(expected, forward) > 1e-4f,
            "forward and inverse sign order must differ, or the test cannot detect a swapped shader");
    }

    /// <summary>
    /// The <c>ssm_out</c> path: tiled→grouped value-head remap folded into the load, with signs
    /// indexed in the permuted order. Bonsai 2 geometry is dState 128, nKHead 16, rep 3 → 6144.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]
    public void Forward_WithGdnPermute_MatchesCpuOracle(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int dState = 128, nKHead = 16, rep = 3;
        const int width = dState * nKHead * rep; // 6144

        var src = RandomFloats(new Random(0x6144 + rows), rows * width);
        var signs = RandomSigns(new Random(0x6145), width);
        var signBytes = ToSbyte(signs);

        // CPU oracle: permute the row, then forward-rotate the permuted row.
        var expected = new float[src.Length];
        var permuted = new float[width];
        for (int t = 0; t < rows; t++)
        {
            Hadamard.PermuteTiledToGrouped(src.AsSpan(t * width, width), permuted, dState, nKHead, rep);
            Hadamard.ForwardRow(permuted, signBytes, expected.AsSpan(t * width, width), BlockSize);
        }

        var actual = RunGpu(spvDir, src, signs, rows, width, inverse: false,
                            permute: new HadamardFwhtF32Kernel.GdnPermute(dState, nKHead, rep));

        AssertClose(expected, actual, rows, width);

        // Discriminator: without the permute the answer must differ, otherwise a shader that
        // ignores the permute flag would pass.
        var unpermuted = RunGpu(spvDir, src, signs, rows, width, inverse: false, permute: null);
        Assert.True(MaxAbsDiff(expected, unpermuted) > 1e-4f,
            "permuted and unpermuted results must differ, or the test cannot detect an ignored permute");
    }

    private static void RunCpuForward(float[] src, float[]? signs, float[] dst, int rows, int width)
    {
        var signSpan = signs is null ? ReadOnlySpan<sbyte>.Empty : ToSbyte(signs);
        for (int t = 0; t < rows; t++)
        {
            Hadamard.ForwardRow(
                src.AsSpan(t * width, width), signSpan, dst.AsSpan(t * width, width), BlockSize);
        }
    }

    private static float[] RunGpu(
        string spvDir, float[] src, float[]? signs, int rows, int width,
        bool inverse, HadamardFwhtF32Kernel.GdnPermute? permute)
    {
        using var device = VulkanDevice.Create();
        using var kernel = HadamardFwhtF32Kernel.Create(device, spvDir);

        // Signs buffer is always bound (the descriptor layout requires it) even when unused.
        float[] signData = signs ?? new float[width];

        using var bufSrc = device.Allocate((long)src.Length * sizeof(float));
        using var bufDst = device.Allocate((long)src.Length * sizeof(float));
        using var bufSigns = device.Allocate((long)signData.Length * sizeof(float));
        device.Upload(src, bufSrc);
        device.Upload(signData, bufSigns);

        kernel.Launch(bufSrc, bufDst, bufSigns, rows, width, BlockSize,
                      applySigns: signs is not null, inverse: inverse, permute: permute);

        var actual = new float[src.Length];
        device.Download(bufDst, actual);
        return actual;
    }

    /// <summary>
    /// Relative tolerance against the CPU oracle. Both sides run the same butterfly in the same
    /// order, but the GPU stages through shared memory and may contract differently, so a small
    /// relative bound is the honest instrument — a wrong transform is off by order 1, not 1e-5.
    /// </summary>
    private static void AssertClose(float[] expected, float[] actual, int rows, int width)
    {
        float scale = 0;
        for (int i = 0; i < expected.Length; i++)
            scale = MathF.Max(scale, MathF.Abs(expected[i]));
        scale = MathF.Max(scale, 1e-9f);

        float worst = 0;
        int worstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float rel = MathF.Abs(expected[i] - actual[i]) / scale;
            if (rel > worst) { worst = rel; worstIdx = i; }
        }

        if (worst > 2e-5f)
        {
            Assert.Fail(
                $"max relative error {worst:E3} at index {worstIdx} (row {worstIdx / width}, col {worstIdx % width} " +
                $"of {rows}x{width}); expected {expected[worstIdx]}, actual {actual[worstIdx]}");
        }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    private static sbyte[] ToSbyte(float[] signs)
    {
        var s = new sbyte[signs.Length];
        for (int i = 0; i < signs.Length; i++) s[i] = (sbyte)(signs[i] < 0 ? -1 : 1);
        return s;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    private static float[] RandomSigns(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = rng.Next(2) == 0 ? -1f : 1f;
        return arr;
    }
}
