using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1220: the snapshot scan used by a speculative verify must leave, after every row, exactly the state the model's DECODE scan arm leaves
/// when it is fed the same tokens one at a time — otherwise a rolled-back verify state is only ULP-close to plain decode and greedy MTP can
/// drift from plain greedy. Checked for the row-split arm (the default) and the LDS-fused arm, at <c>NVHead / NKHead = 2</c> with distinct
/// per-k-head inputs (a head-broadcast mix-up changes the result).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanGdnScanSnapshotRowSplitTests
{
    [SkippableTheory]
    [InlineData(GdnScanMultiTokenF32Kernel.Variant.RowSplit, 4, 3)]
    [InlineData(GdnScanMultiTokenF32Kernel.Variant.RowSplit, 5, 4)]
    [InlineData(GdnScanMultiTokenF32Kernel.Variant.LdsFused, 4, 3)]
    public void SnapshotsAndOutputs_BitIdenticalToTokenByTokenDecodeScan(GdnScanMultiTokenF32Kernel.Variant variant, int seqLen, int snapRows)
    {
        const int NVHead = 4, NKHead = 2, DState = 128;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var decode = GdnScanMultiTokenF32Kernel.Create(device, spvDir, variant);
        using var snap = GdnScanMultiTokenSnapshotF32Kernel.Create(device, spvDir, variant, DState);
        Assert.Equal(variant == GdnScanMultiTokenF32Kernel.Variant.RowSplit, snap.IsRowSplit);

        var rng = new Random(1220 + seqLen * 17 + (int)variant);
        int stateElems = NVHead * DState * DState;
        float[] state0 = Rand(rng, stateElems, 0.1f);
        float[] q = Rand(rng, seqLen * NKHead * DState, 1f), k = Rand(rng, seqLen * NKHead * DState, 1f);
        float[] v = Rand(rng, seqLen * NVHead * DState, 1f);
        float[] g = new float[seqLen * NVHead], beta = new float[seqLen * NVHead];
        for (int i = 0; i < g.Length; i++) { g[i] = 0.5f + 0.5f * (float)rng.NextDouble(); beta[i] = (float)rng.NextDouble(); }

        using var bState = Up(device, state0);
        using var bq = Up(device, q); using var bk = Up(device, k); using var bv = Up(device, v);
        using var bg = Up(device, g); using var bb = Up(device, beta);
        using var bOut = device.Allocate((long)seqLen * NVHead * DState * 4);
        using var bSnap = device.Allocate((long)snapRows * stateElems * 4);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            snap.Record(ctx.CommandBuffer, bState, bq, bk, bv, bg, bb, bOut, bSnap, snapRows, seqLen, NVHead, NKHead, DState);
            ctx.SubmitAndWait();
        }
        float[] finalState = Down(device, bState, stateElems), outs = Down(device, bOut, seqLen * NVHead * DState);
        float[] snaps = Down(device, bSnap, snapRows * stateElems);

        // Reference: the decode arm, one token per dispatch.
        using var rState = Up(device, state0);
        using var rq = device.Allocate(NKHead * DState * 4); using var rk = device.Allocate(NKHead * DState * 4);
        using var rv = device.Allocate(NVHead * DState * 4);
        using var rg = device.Allocate(NVHead * 4); using var rb = device.Allocate(NVHead * 4);
        using var rOut = device.Allocate(NVHead * DState * 4);
        for (int t = 0; t < seqLen; t++)
        {
            device.Upload(q.AsSpan(t * NKHead * DState, NKHead * DState).ToArray(), rq);
            device.Upload(k.AsSpan(t * NKHead * DState, NKHead * DState).ToArray(), rk);
            device.Upload(v.AsSpan(t * NVHead * DState, NVHead * DState).ToArray(), rv);
            device.Upload(g.AsSpan(t * NVHead, NVHead).ToArray(), rg);
            device.Upload(beta.AsSpan(t * NVHead, NVHead).ToArray(), rb);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                decode.Record(ctx.CommandBuffer, rState, rq, rk, rv, rg, rb, rOut, 1, NVHead, NKHead, DState);
                ctx.SubmitAndWait();
            }
            float[] st = Down(device, rState, stateElems);
            Assert.Equal(Down(device, rOut, NVHead * DState), outs.AsSpan(t * NVHead * DState, NVHead * DState).ToArray());
            if (t < snapRows)
                Assert.Equal(st, snaps.AsSpan(t * stateElems, stateElems).ToArray());
            if (t == seqLen - 1)
                Assert.Equal(st, finalState);
        }
    }

    private static float[] Rand(Random rng, int n, float range)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * range;
        return a;
    }

    private static VulkanDevice.Buffer Up(VulkanDevice device, float[] data)
    {
        var b = device.Allocate((long)data.Length * 4);
        device.Upload(data, b);
        return b;
    }

    private static float[] Down(VulkanDevice device, VulkanDevice.Buffer b, int n)
    {
        var a = new float[n];
        device.Download(b, a);
        return a;
    }
}
