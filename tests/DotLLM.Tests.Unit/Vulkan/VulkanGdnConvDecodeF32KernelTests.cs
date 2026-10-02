using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity for the fused single-token GDN conv (conv1d + SiLU + conv-state shift). Reference = the multi-token path it replaces:
/// <c>Conv1dCausal.Execute</c> over <c>[state | x]</c>, host SiLU, and the trailing <c>dConv-1</c> rows as the new state.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanGdnConvDecodeF32KernelTests
{
    [SkippableTheory]
    [InlineData(4, 8)]
    [InlineData(4, 8192)]   // real GDN conv width order (35B-A3B convDim = 8192)
    [InlineData(2, 33)]     // dConv=2: single state row, no inner-loop iterations
    [InlineData(3, 17)]
    [InlineData(5, 300)]    // channels spans several 256-thread workgroups
    public void Launch_MatchesMultiTokenPath_AndShiftsState(int dConv, int channels)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x6D0C + dConv * 131 + channels);
        float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }

        float[] state = R((dConv - 1) * channels), x = R(channels), w = R(dConv * channels), bias = R(channels);

        // Reference: ConvInput = [state | x], conv, SiLU; new state = rows 1..dConv-1 of ConvInput.
        float[] convInput = new float[dConv * channels];
        state.CopyTo(convInput, 0);
        x.CopyTo(convInput, (dConv - 1) * channels);
        float[] expected = new float[channels];
        Conv1dCausal.Execute(convInput, w, bias, expected, dConv, channels, 1);
        for (int i = 0; i < channels; i++) expected[i] = expected[i] / (1f + MathF.Exp(-expected[i]));
        float[] expectedState = new float[(dConv - 1) * channels];
        Array.Copy(convInput, channels, expectedState, 0, expectedState.Length);

        using var device = VulkanDevice.Create();
        using var kernel = GdnConvDecodeF32Kernel.Create(device, spvDir);
        using var bState = device.Allocate((long)state.Length * 4);
        using var bW = device.Allocate((long)w.Length * 4);
        using var bBias = device.Allocate((long)bias.Length * 4);
        using var bX = device.Allocate((long)x.Length * 4);
        device.Upload(state, bState); device.Upload(w, bW); device.Upload(bias, bBias); device.Upload(x, bX);

        kernel.Launch(bState, bW, bBias, bX, dConv, channels);

        float[] actual = new float[channels], actualState = new float[state.Length];
        device.Download(bX, actual);
        device.Download(bState, actualState);

        for (int i = 0; i < channels; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f + 1e-3f * MathF.Abs(expected[i]), $"y[{i}] cpu={expected[i]} vk={actual[i]}");
        for (int i = 0; i < actualState.Length; i++)
            Assert.True(expectedState[i] == actualState[i], $"state[{i}] expected {expectedState[i]} got {actualState[i]} (the shift must be exact)");
    }
}
