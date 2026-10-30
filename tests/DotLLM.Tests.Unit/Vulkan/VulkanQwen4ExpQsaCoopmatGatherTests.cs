using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1356: the cooperative-matrix QSA gather attention (<c>qsa_attention_cm_f16kv</c>) against the scalar F16-KV gather kernel on identical
/// inputs. The scalar kernel is the reference (f32 P.V); the matrix kernel rounds P to f16 and splits Q hi/lo, so the two agree to f16-class error,
/// not bitwise. Different data per head, mixed dense/sparse queries, ragged tails (partial 32-key tiles), several splits and a x16-amplitude arm
/// (large scores are where a plain f16 QK^T breaks, see the hd256 flash kernel).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpQsaCoopmatGatherTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpQsaCoopmatGatherTests(ITestOutputHelper output) => _out = output;

    private static float[] Rand(Random rng, long n, float scale)
    {
        var a = new float[n];
        for (long i = 0; i < n; i++) a[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static uint[] PackHalves(float[] x)
    {
        var w = new uint[x.Length / 2];
        for (int i = 0; i < w.Length; i++)
            w[i] = BitConverter.HalfToUInt16Bits((Half)x[2 * i]) | ((uint)BitConverter.HalfToUInt16Bits((Half)x[2 * i + 1]) << 16);
        return w;
    }

    private static void Run(VulkanDevice d, Action<nint> record)
    {
        using var ctx = d.CreateSubmitContext();
        ctx.Begin();
        KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
        record(ctx.CommandBuffer);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
    }

    [SkippableTheory]
    [InlineData(12, 2, 64, 8, 700, 1, 1f, 1f)]       // released geometry, one split, sparse (budget 8 blocks of 64)
    [InlineData(12, 2, 64, 8, 700, 3, 1f, 1f)]       // several splits (a split's chunk ends mid-tile)
    [InlineData(12, 2, 64, 40, 900, 2, 1f, 1f)]      // budget above the block count of early queries: dense queries mixed in
    [InlineData(16, 2, 32, 8, 333, 2, 1f, 1f)]       // full 16-row tile, ragged tail
    [InlineData(8, 2, 64, 8, 700, 1, 1f, 1f)]        // 4 heads per KV head: padding rows must stay inert
    [InlineData(12, 2, 64, 8, 700, 2, 16f, 4f)]      // large-amplitude scores
    public void CoopmatGather_TracksTheScalarGather(int heads, int kvHeads, int blockSize, int budget, int tokens, int splits, float qScale, float kvScale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spvDir);
        Skip.IfNot(k.F16Supported, "F16 SPIR-V twins not built");
        Skip.IfNot(k.CoopmatSupported, "cooperative-matrix gather not available on this device");
        const int headDim = 256;
        var rng = new Random(1356 + heads * 7 + splits);
        int kvStride = kvHeads * headDim, Q = 40, first = tokens - Q;
        var q = Rand(rng, (long)Q * heads * headDim, qScale);
        var kk = Rand(rng, (long)tokens * kvStride, kvScale);
        var vv = Rand(rng, (long)tokens * kvStride, 2f);
        var sel = new int[Q * budget];
        for (int tq = 0; tq < Q; tq++)
        {
            int nb = (first + tq + 1) / blockSize;
            var ids = Enumerable.Range(0, nb).OrderBy(_ => rng.Next()).Take(budget).OrderBy(x => x).ToArray();
            for (int i = 0; i < budget; i++) sel[tq * budget + i] = i < ids.Length ? ids[i] : 0;
        }
        long partRows = (long)Q * heads * splits;
        using var bQ = device.AllocateDeviceLocal(q.Length * 4L);
        using var bK = device.AllocateDeviceLocal(kk.Length * 2L);
        using var bV = device.AllocateDeviceLocal(vv.Length * 2L);
        using var bSel = device.AllocateDeviceLocal(sel.Length * 4L);
        using var bPart = device.AllocateDeviceLocal(partRows * headDim * 4);
        using var bMs = device.AllocateDeviceLocal(partRows * 2 * 4);
        using var bRef = device.AllocateDeviceLocal((long)Q * heads * headDim * 4);
        using var bCm = device.AllocateDeviceLocal((long)Q * heads * headDim * 4);
        device.Upload(q, bQ);
        device.Upload(MemoryMarshal.AsBytes<uint>(PackHalves(kk)), bK);
        device.Upload(MemoryMarshal.AsBytes<uint>(PackHalves(vv)), bV);
        device.Upload(MemoryMarshal.AsBytes<int>(sel), bSel);
        void Attn(VulkanDevice.Buffer o, bool cm)
        {
            bool saved = Qwen4ExpQsaKernels.CoopmatGather;
            Qwen4ExpQsaKernels.CoopmatGather = cm;
            try { Run(device, cmd => k.RecordAttention(cmd, bQ, bK, bV, bSel, bPart, bMs, o, 0, Q, first, heads, kvHeads, headDim, blockSize, budget, splits, true)); }
            finally { Qwen4ExpQsaKernels.CoopmatGather = saved; }
        }
        long before = k.CoopmatAttentionDispatches;
        Attn(bRef, false);
        Assert.Equal(before, k.CoopmatAttentionDispatches);
        Attn(bCm, true);
        Assert.Equal(before + 1, k.CoopmatAttentionDispatches);
        var a = new float[Q * heads * headDim]; var b = new float[a.Length];
        device.Download(bRef, a); device.Download(bCm, b);
        // per-head error: a wrong head -> row mapping shows up as one head diverging, which the aggregate could hide
        double worst = 0;
        for (int tq = 0; tq < Q; tq++)
            for (int h = 0; h < heads; h++)
            {
                double n = 0, den = 0;
                for (int d = 0; d < headDim; d++)
                {
                    int i = (tq * heads + h) * headDim + d;
                    double e = a[i] - b[i]; n += e * e; den += (double)a[i] * a[i];
                }
                worst = Math.Max(worst, Math.Sqrt(n / Math.Max(den, 1e-30)));
            }
        _out.WriteLine($"heads {heads}/{kvHeads} block {blockSize} budget {budget} tokens {tokens} splits {splits} q x{qScale} kv x{kvScale}: worst per-(query, head) relL2 {worst:E2}");
        Assert.True(worst < 3e-3, $"cooperative-matrix gather diverges from the scalar gather: {worst:E2}");
        Assert.True(a.Any(v => Math.Abs(v) > 1e-3), "control: output is all ~0, the comparison proves nothing");
    }
}
