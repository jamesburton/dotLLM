using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #695: <see cref="GdnConvSiluF32Kernel"/> reads the conv state and the qkv rows directly and applies SiLU in one pass. It must equal
/// the previous chain (concatenate [state | qkv], <see cref="Conv1dCausalF32Kernel"/>, <see cref="SiluInplaceF32Kernel"/>) EXACTLY.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGdnConvSiluTests
{
    [SkippableTheory]
    [InlineData(4, 300, 8)]       // channels not a multiple of 256, one time tile
    [InlineData(4, 8192, 17)]     // two time tiles (TT = 16), ragged
    [InlineData(4, 512, 100)]
    [InlineData(3, 257, 33)]
    [InlineData(6, 1000, 40)]
    [InlineData(2, 64, 20)]
    public void Fused_EqualsConvThenSilu(int dConv, int channels, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GdnConvSiluF32Kernel.IsSupportedOn(spvDir), "SPIR-V missing.");
        using var device = VulkanDevice.Create();
        using var fused = GdnConvSiluF32Kernel.Create(device, spvDir);
        using var conv = Conv1dCausalF32Kernel.Create(device, spvDir);
        using var silu = SiluInplaceF32Kernel.Create(device, spvDir);

        var rng = new Random(695 + dConv * 7 + channels + seqLen);
        float[] Rand(int n, float s) => Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * s)).ToArray();
        float[] state = Rand((dConv - 1) * channels, 2f), qkv = Rand(seqLen * channels, 2f);
        float[] w = Rand(dConv * channels, 1f), bias = Rand(channels, 0.5f);

        float[] concat = new float[(dConv - 1 + seqLen) * channels];
        state.CopyTo(concat, 0);
        qkv.CopyTo(concat, state.Length);

        using var bState = device.Allocate((long)state.Length * 4);
        using var bQkv = device.Allocate((long)qkv.Length * 4);
        using var bConcat = device.Allocate((long)concat.Length * 4);
        using var bW = device.Allocate((long)w.Length * 4);
        using var bBias = device.Allocate((long)bias.Length * 4);
        using var bRef = device.Allocate((long)qkv.Length * 4);
        using var bOut = device.Allocate((long)qkv.Length * 4);
        device.Upload(state, bState); device.Upload(qkv, bQkv); device.Upload(concat, bConcat);
        device.Upload(w, bW); device.Upload(bias, bBias);

        conv.Launch(bConcat, bW, bBias, bRef, dConv, channels, seqLen);
        silu.Launch(bRef, seqLen * channels);
        fused.Launch(bState, bQkv, bW, bBias, bOut, dConv, channels, seqLen);

        float[] a = new float[qkv.Length], b = new float[qkv.Length];
        device.Download(bRef, a);
        device.Download(bOut, b);
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]),
                $"d{dConv} c{channels} t{seqLen} idx {i} (t {i / channels}, c {i % channels}): chain {a[i]:G9} != fused {b[i]:G9}");
    }
}
