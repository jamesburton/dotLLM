using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1015: <see cref="Lfm2ShortConvF32Kernel"/> (LFM2 gated short convolution) against a scalar reference, including the in-place state roll
/// and a chained second call that consumes the rolled state (the decode / chunked-prefill path).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanLfm2ShortConvTests
{
    private static (float[] y, float[] newState) Reference(float[] state, float[] proj, float[] w, int L, int H, int T)
    {
        var xs = new float[(L - 1 + T) * H];
        Array.Copy(state, xs, (L - 1) * H);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < H; c++)
                xs[(L - 1 + t) * H + c] = proj[t * 3 * H + c] * proj[t * 3 * H + 2 * H + c];   // B * X
        var y = new float[T * H];
        for (int t = 0; t < T; t++)
            for (int c = 0; c < H; c++)
            {
                float acc = 0;
                for (int k = 0; k < L; k++) acc += w[c * L + k] * xs[(t + k) * H + c];
                y[t * H + c] = acc * proj[t * 3 * H + H + c];                                 // * C
            }
        var ns = new float[(L - 1) * H];
        Array.Copy(xs, T * H, ns, 0, (L - 1) * H);
        return (y, ns);
    }

    [SkippableTheory]
    [InlineData(3, 300, 1)]       // decode: T < L-1, state rows survive the roll
    [InlineData(3, 2048, 2)]      // T == L-1
    [InlineData(3, 2048, 37)]     // ragged time tiles (TT = 16)
    [InlineData(4, 513, 100)]
    [InlineData(2, 64, 20)]
    public void ConvAndStateRoll_MatchScalarReference_AcrossTwoChainedCalls(int L, int H, int T)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(Lfm2ShortConvF32Kernel.IsSupportedOn(spvDir), "SPIR-V missing.");
        using var device = VulkanDevice.Create();
        using var kernel = Lfm2ShortConvF32Kernel.Create(device, spvDir);

        var rng = new Random(1015 + L * 13 + H + T);
        float[] Rand(int n, float s) => Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * s)).ToArray();
        float[] state = Rand((L - 1) * H, 1.5f), w = Rand(L * H, 1f);
        using var bState = device.Allocate((long)state.Length * 4);
        using var bW = device.Allocate((long)w.Length * 4);
        device.Upload(state, bState); device.Upload(w, bW);

        float[] refState = (float[])state.Clone();
        for (int call = 0; call < 2; call++)
        {
            int t = call == 0 ? T : 1 + T / 3;
            float[] proj = Rand(t * 3 * H, 1.5f);
            using var bProj = device.Allocate((long)proj.Length * 4);
            using var bOut = device.Allocate((long)t * H * 4);
            device.Upload(proj, bProj);
            kernel.Launch(bState, bProj, bW, bOut, L, H, t);

            var (y, ns) = Reference(refState, proj, w, L, H, t);
            refState = ns;
            float[] gotY = new float[t * H], gotState = new float[(L - 1) * H];
            device.Download(bOut, gotY);
            device.Download(bState, gotState);
            for (int i = 0; i < y.Length; i++)
                Assert.True(MathF.Abs(y[i] - gotY[i]) <= 1e-5f * (1 + MathF.Abs(y[i])), $"call {call} y[{i}] (t {i / H}, c {i % H}): ref {y[i]:G9} != gpu {gotY[i]:G9}");
            for (int i = 0; i < ns.Length; i++)
                Assert.True(ns[i] == gotState[i], $"call {call} state[{i}]: ref {ns[i]:G9} != gpu {gotState[i]:G9}");
        }
    }
}
