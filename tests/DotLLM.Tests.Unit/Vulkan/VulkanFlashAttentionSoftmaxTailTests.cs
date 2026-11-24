using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1953 on Vulkan: the cooperative-matrix flash kernels feed P = exp(s - max) to the P.V coopmat as F16. In (0, 1] a key ~17.3+
/// nats below the row max rounds to 0, so behind an attention sink at 32K keys the row's tail mass vanished from numerator AND
/// denominator (the running sum adds the rounded P): the output renormalised over the sink alone. The fix scales P by 2^15
/// (<c>P_SHIFT_LN</c>), which cancels in O / sum.
/// <para>Fixture (as the CUDA <c>SinkDominatedLongRow</c> tests): one sink key (V = 0) 15.6-18.4 nats above every other key (V 48..80,
/// one sign), 32,771 keys, 16 query rows at the end; every value pre-rounded to F16 so the kernels' F16 staging is exact and the only
/// difference to the scalar CPU reference (F32 P) is the P path. Before the fix every output here is 0.</para>
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionSoftmaxTailTests
{
    private const int SeqQ = 16, SeqKv = 32771, NumHeads = 8, NumKvHeads = 2;
    private readonly ITestOutputHelper _out;
    public VulkanFlashAttentionSoftmaxTailTests(ITestOutputHelper output) => _out = output;

    private static float H(float x) => (float)(Half)x;

    private static (float[] Q, float[] K, float[] V, float[] Expected) Fixture(int headDim)
    {
        int stride = NumKvHeads * headDim;
        var rng = new Random(1953 + headDim + SeqKv);
        var k = new float[(long)SeqKv * stride];
        var v = new float[(long)SeqKv * stride];
        for (int j = 0; j < SeqKv; j++)
            for (int h = 0; h < NumKvHeads; h++)
            {
                long b = (long)j * stride + h * headDim;
                k[b] = H(j == 0 ? 20f : -2f - 4f * (float)rng.NextDouble());
                for (int d = 1; d < headDim; d++) k[b + d] = H((float)(rng.NextDouble() * 0.2 - 0.1));
                for (int d = 0; d < headDim; d++) v[b + d] = H(j == 0 ? 0f : 48f + 32f * (float)rng.NextDouble());
            }
        var q = new float[SeqQ * NumHeads * headDim];
        float q0 = 8f * MathF.Sqrt(headDim / 128f);
        for (int r = 0; r < SeqQ; r++)
            for (int h = 0; h < NumHeads; h++)
                q[(r * NumHeads + h) * headDim] = H(q0 * (1f + 0.02f * h));
        var expected = new float[q.Length];
        Attention.ExecuteScalar(q, k, v, expected, SeqQ, SeqKv, NumHeads, NumKvHeads, headDim, SeqKv - SeqQ,
            1f / MathF.Sqrt(headDim), default, null, 0f, AttentionMaskMode.Causal, 0, []);
        return (q, k, v, expected);
    }

    private void AssertTail(string what, float[] expected, float[] actual)
    {
        double worst = 0;
        int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(actual[i] - expected[i]) / Math.Max(Math.Abs(expected[i]), 1e-30);
            if (!(err <= worst)) { worst = err; at = i; }
        }
        _out.WriteLine($"{what}: worst relative error {worst:E3} at {at} ({actual[at]:E4} vs CPU {expected[at]:E4})");
        Assert.True(worst <= 0.01, $"{what} loses the softmax tail: {actual[at]:E4} vs CPU {expected[at]:E4} at {at} (relative {worst:E3})");
    }

    private static VulkanDevice.Buffer Upload(VulkanDevice device, float[] data)
    {
        var buf = device.Allocate((long)data.Length * sizeof(float));
        device.Upload(data.AsSpan(), buf);
        return buf;
    }

    private static VulkanDevice.Buffer UploadHalf(VulkanDevice device, float[] data)
    {
        var h = new ushort[data.Length];
        for (int i = 0; i < h.Length; i++) h[i] = BitConverter.HalfToUInt16Bits((Half)data[i]);
        var buf = device.Allocate((long)h.Length * sizeof(ushort));
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(h.AsSpan()), buf);
        return buf;
    }

    /// <summary><c>attention_flash_f32_coopmat</c> (hd64 / hd128 / hd256 SPVs), F32 KV.</summary>
    [SkippableTheory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void FlashCoopmat_SinkDominatedLongRow_KeepsTheTailMass(int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "no coopmat tile");
        var (q, k, v, expected) = Fixture(headDim);
        using var kernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir);
        using var bq = Upload(device, q);
        using var bk = Upload(device, k);
        using var bv = Upload(device, v);
        using var bo = device.Allocate((long)q.Length * sizeof(float));
        kernel.Launch(bq, bk, bv, bo, SeqQ, SeqKv, NumHeads, NumKvHeads, headDim, positionOffset: SeqKv - SeqQ);
        float[] actual = new float[q.Length];
        device.Download(bo, actual);
        AssertTail($"flash coopmat hd{headDim} 32K on {device.DeviceName}", expected, actual);
    }

    /// <summary><c>attention_flash_gqa_cm</c>, F16 KV (the dense F16 cache), hd128 / hd256.</summary>
    [SkippableTheory]
    [InlineData(128)]
    [InlineData(256)]
    public void FlashGqaCoopmat_SinkDominatedLongRow_KeepsTheTailMass(int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (q, k, v, expected) = Fixture(headDim);
        using var kernel = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16, null, false);
        Skip.If(kernel is null, $"the coopmat GQA flash kernel is unavailable on {device.DeviceName}");
        Skip.IfNot(kernel!.CanDispatch(headDim, NumHeads, NumKvHeads), "shape not dispatchable");
        using var bq = Upload(device, q);
        using var bk = UploadHalf(device, k);
        using var bv = UploadHalf(device, v);
        using var bo = device.Allocate((long)q.Length * sizeof(float));
        kernel.Launch(bq, bk, bv, bo, SeqQ, SeqKv, NumHeads, NumKvHeads, headDim, SeqKv - SeqQ, 0, sinks: null);
        float[] actual = new float[q.Length];
        device.Download(bo, actual);
        AssertTail($"GQA flash coopmat F16 KV hd{headDim} 32K on {device.DeviceName}", expected, actual);
    }
}
