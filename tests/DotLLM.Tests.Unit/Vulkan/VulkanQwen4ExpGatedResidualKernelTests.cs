using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Op-level parity of the Qwen4-Exp (#818) gated-residual shader and the sigmoid GDN output gate against the CPU oracle's own functions
/// (<see cref="Qwen4ExpGatedResidual"/>, <c>RmsNorm</c>). The model-level tests localise a failure to a layer; these localise it to an op
/// without needing a model, and cover shapes the tiny model never reaches (non-multiple-of-256 element counts, S != 4).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpGatedResidualKernelTests
{
    private static float[] Random(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return a;
    }

    private static void AssertClose(float[] expected, float[] actual, string what, float rel = 2e-5f, float abs = 2e-6f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= abs + rel * MathF.Abs(expected[i]),
                $"{what}[{i}]: cpu={expected[i]:G9} gpu={actual[i]:G9}");
    }

    [SkippableTheory]
    [InlineData(1, 4, 64)]
    [InlineData(5, 4, 64)]
    [InlineData(3, 4, 100)]      // H not a multiple of the 256-thread workgroup
    [InlineData(7, 3, 37)]       // S != 4 and an odd H: no hard-coded stream count or alignment
    public void GatedResidualOps_MatchCpu(int T, int S, int H)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var gr = Qwen4ExpGatedResidualKernel.Create(device, spvDir);
        int row = S * H, lr = 11;

        // BROADCAST
        float[] emb = Random(T * H, 1);
        float[] expR = new float[T * row];
        Qwen4ExpGatedResidual.Broadcast(emb, S, H, expR, T);
        using var bEmb = device.AllocateDeviceLocal(emb.Length * 4L);
        using var bR = device.AllocateDeviceLocal(expR.Length * 4L);
        device.Upload(emb, bEmb);
        Run(device, c => gr.RecordBroadcast(c, bR, bEmb, T, S, H));
        AssertClose(expR, Download(device, bR, expR.Length), "broadcast", rel: 0, abs: 0);

        // ACTIVATE (silu(v / S))
        float[] low = Random(T * lr, 2, 4f);
        float[] expLow = (float[])low.Clone();
        Qwen4ExpGatedResidual.ActivateLowRank(expLow, S);
        using var bLow = device.AllocateDeviceLocal(low.Length * 4L);
        device.Upload(low, bLow);
        Run(device, c => gr.RecordActivateLowRank(c, bLow, T * lr, S));
        AssertClose(expLow, Download(device, bLow, low.Length), "activate");

        // MIXMEAN
        float[] mix = Random(T * row, 3, 3f), xn = Random(T * row, 4);
        float[] expH = new float[T * H];
        Qwen4ExpGatedResidual.MixAndMean((float[])mix.Clone(), xn, S, H, expH, T);
        using var bMix = device.AllocateDeviceLocal(mix.Length * 4L);
        using var bXn = device.AllocateDeviceLocal(xn.Length * 4L);
        using var bH = device.AllocateDeviceLocal(expH.Length * 4L);
        device.Upload(mix, bMix); device.Upload(xn, bXn);
        Run(device, c => gr.RecordMixMean(c, bH, bMix, bXn, T, S, H));
        AssertClose(expH, Download(device, bH, expH.Length), "mixmean");

        // INJECT GAINS
        float[] inj = Random(T * S, 5, 6f);
        float[] expG = (float[])inj.Clone();
        Qwen4ExpGatedResidual.InjectGains(expG, S);
        using var bG = device.AllocateDeviceLocal(inj.Length * 4L);
        device.Upload(inj, bG);
        Run(device, c => gr.RecordInjectGains(c, bG, T, S));
        AssertClose(expG, Download(device, bG, inj.Length), "inject");

        // WRITE (against a non-trivial residual, with distinct per-stream gains so a stream mix-up shows)
        float[] res = Random(T * row, 6);
        float[] y = Random(T * H, 7);
        float[] expW = (float[])res.Clone();
        Qwen4ExpGatedResidual.Write(expW, y, expG, S, H, T);
        using var bRes = device.AllocateDeviceLocal(res.Length * 4L);
        using var bY = device.AllocateDeviceLocal(y.Length * 4L);
        device.Upload(res, bRes); device.Upload(y, bY);
        Run(device, c => gr.RecordWrite(c, bRes, bY, bG, T, S, H));
        AssertClose(expW, Download(device, bRes, res.Length), "write");
    }

    [SkippableTheory]
    [InlineData(1, 4, 16)]
    [InlineData(6, 4, 128)]
    public void SigmoidPostScanGate_MatchesCpu_AndDiffersFromSilu(int T, int nVHead, int dState)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var sig = GdnPostScanGateF32Kernel.Create(device, spvDir, sigmoidGate: true);
        using var silu = GdnPostScanGateF32Kernel.Create(device, spvDir, sigmoidGate: false);
        const float eps = 1e-6f;

        float[] core = Random(T * nVHead * dState, 11), z = Random(T * nVHead * dState, 12, 3f), norm = Random(dState, 13);
        for (int i = 0; i < norm.Length; i++) norm[i] += 1.2f;
        float[] expSig = (float[])core.Clone(), expSilu = (float[])core.Clone();
        for (int h = 0; h < T * nVHead; h++)
        {
            int off = h * dState;
            RmsNorm.Execute(expSig.AsSpan(off, dState), norm, eps, expSig.AsSpan(off, dState));
            RmsNorm.Execute(expSilu.AsSpan(off, dState), norm, eps, expSilu.AsSpan(off, dState));
            for (int i = 0; i < dState; i++)
            {
                float zi = z[off + i];
                expSig[off + i] *= 1f / (1f + MathF.Exp(-zi));
                expSilu[off + i] *= zi / (1f + MathF.Exp(-zi));
            }
        }

        using var bOut = device.AllocateDeviceLocal(core.Length * 4L);
        using var bZ = device.AllocateDeviceLocal(z.Length * 4L);
        using var bNorm = device.AllocateDeviceLocal(norm.Length * 4L);
        device.Upload(z, bZ); device.Upload(norm, bNorm);

        device.Upload(core, bOut);
        Run(device, c => sig.Record(c, bOut, bZ, bNorm, T, nVHead, dState, eps));
        float[] gotSig = Download(device, bOut, core.Length);
        AssertClose(expSig, gotSig, "sigmoid-gated");

        device.Upload(core, bOut);
        Run(device, c => silu.Record(c, bOut, bZ, bNorm, T, nVHead, dState, eps));
        float[] gotSilu = Download(device, bOut, core.Length);
        AssertClose(expSilu, gotSilu, "silu-gated");

        // Sensitivity control: the two gates must actually disagree on this data, otherwise the test could not tell them apart.
        double diff = 0;
        for (int i = 0; i < gotSig.Length; i++) diff = Math.Max(diff, Math.Abs(gotSig[i] - gotSilu[i]));
        Assert.True(diff > 1e-2, $"sigmoid and silu gates are indistinguishable on this data (max diff {diff:E2})");
    }

    private static void Run(VulkanDevice device, Action<nint> record)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        record(ctx.CommandBuffer);
        ctx.SubmitAndWait();
    }

    private static float[] Download(VulkanDevice device, VulkanDevice.Buffer buf, int n)
    {
        var a = new float[n];
        device.Download(buf, a);
        return a;
    }
}
