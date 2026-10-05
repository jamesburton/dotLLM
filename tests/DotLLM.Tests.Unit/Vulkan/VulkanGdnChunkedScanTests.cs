using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>#703: the chunked (WY) GDN scan against the sequential scan, on L2-normalised q/k like the model produces.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGdnChunkedScanTests
{
    private readonly ITestOutputHelper _out;
    public VulkanGdnChunkedScanTests(ITestOutputHelper output) => _out = output;

    private static float[] Normalised(Random rng, int rows, int dim)
    {
        var a = new float[rows * dim];
        for (int r = 0; r < rows; r++)
        {
            double n = 0;
            for (int d = 0; d < dim; d++) { float x = (float)(rng.NextDouble() * 2 - 1); a[r * dim + d] = x; n += (double)x * x; }
            float inv = (float)(1.0 / Math.Sqrt(n));
            for (int d = 0; d < dim; d++) a[r * dim + d] *= inv;
        }
        return a;
    }

    private static double RelMax(float[] e, float[] a)
    {
        double scale = 0, err = 0;
        for (int i = 0; i < e.Length; i++) scale = Math.Max(scale, Math.Abs(e[i]));
        for (int i = 0; i < e.Length; i++) err = Math.Max(err, Math.Abs((double)e[i] - a[i]));
        return err / Math.Max(scale, 1e-30);
    }

    [SkippableTheory]
    [InlineData(64, 0.9f)]
    [InlineData(200, 0.9f)]       // ragged last chunk (200 = 3 x 64 + 8)
    [InlineData(256, 0.5f)]       // strong decay
    [InlineData(130, 0.98f)]
    [InlineData(1, 0.9f)]         // a single token
    public void Chunked_MatchesSequentialScan(int seqLen, float gMin)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GdnChunkedScanF32Kernel.IsSupportedOn(spvDir), "chunked SPIR-V missing.");
        const int nVHead = 8, nKHead = 4, dState = 128;
        var rng = new Random(703 + seqLen);
        float[] state0 = new float[nVHead * dState * dState];
        for (int i = 0; i < state0.Length; i++) state0[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
        float[] q = Normalised(rng, seqLen * nKHead, dState), k = Normalised(rng, seqLen * nKHead, dState);
        float[] v = new float[seqLen * nVHead * dState];
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] g = new float[seqLen * nVHead], beta = new float[seqLen * nVHead];
        for (int i = 0; i < g.Length; i++) { g[i] = gMin + (1f - gMin) * (float)rng.NextDouble(); beta[i] = (float)rng.NextDouble(); }

        using var device = VulkanDevice.Create();
        float[][] outs = new float[2][], states = new float[2][];
        for (int arm = 0; arm < 2; arm++)
        {
            using var bs = device.Allocate((long)state0.Length * 4);
            using var bq = device.Allocate((long)q.Length * 4); using var bk = device.Allocate((long)k.Length * 4);
            using var bv = device.Allocate((long)v.Length * 4); using var bg = device.Allocate((long)g.Length * 4);
            using var bb = device.Allocate((long)beta.Length * 4); using var bo = device.Allocate((long)v.Length * 4);
            device.Upload(state0, bs); device.Upload(q, bq); device.Upload(k, bk); device.Upload(v, bv); device.Upload(g, bg); device.Upload(beta, bb);
            if (arm == 0)
            {
                using var seq = GdnScanMultiTokenF32Kernel.Create(device, spvDir, GdnScanMultiTokenF32Kernel.Variant.LdsFused);
                seq.Launch(bs, bq, bk, bv, bg, bb, bo, seqLen, nVHead, nKHead, dState);
            }
            else
            {
                using var chunked = GdnChunkedScanF32Kernel.Create(device, spvDir);
                chunked.Launch(bs, bq, bk, bv, bg, bb, bo, seqLen, nVHead, nKHead);
            }
            outs[arm] = new float[v.Length]; states[arm] = new float[state0.Length];
            device.Download(bo, outs[arm]); device.Download(bs, states[arm]);
        }
        double eo = RelMax(outs[0], outs[1]), es = RelMax(states[0], states[1]);
        _out.WriteLine($"seqLen {seqLen} gMin {gMin}: output rel max err {eo:E3}, state rel max err {es:E3}");
        Assert.True(eo < 2e-4, $"output rel max err {eo:E3}");
        Assert.True(es < 2e-4, $"state rel max err {es:E3}");
    }

    /// <summary>Opt-in (DOTLLM_GDN_CHUNK_PROBE=1): kernel-only time of the shipping row-split scan vs the chunked scan at the 35B head shape.</summary>
    [SkippableTheory]
    [InlineData(512)]
    [InlineData(2048)]
    public void Probe_KernelTime(int seqLen)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GDN_CHUNK_PROBE") == "1", "opt-in probe");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int nVHead = 48, nKHead = 16, dState = 128;
        var rng = new Random(7);
        float[] state0 = new float[nVHead * dState * dState];
        float[] q = Normalised(rng, seqLen * nKHead, dState), k = Normalised(rng, seqLen * nKHead, dState);
        float[] v = new float[seqLen * nVHead * dState];
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] g = new float[seqLen * nVHead], beta = new float[seqLen * nVHead];
        for (int i = 0; i < g.Length; i++) { g[i] = 0.9f + 0.1f * (float)rng.NextDouble(); beta[i] = (float)rng.NextDouble(); }
        using var device = VulkanDevice.Create();
        using var bs = device.Allocate((long)state0.Length * 4);
        using var bq = device.Allocate((long)q.Length * 4); using var bk = device.Allocate((long)k.Length * 4);
        using var bv = device.Allocate((long)v.Length * 4); using var bg = device.Allocate((long)g.Length * 4);
        using var bb = device.Allocate((long)beta.Length * 4); using var bo = device.Allocate((long)v.Length * 4);
        device.Upload(state0, bs); device.Upload(q, bq); device.Upload(k, bk); device.Upload(v, bv); device.Upload(g, bg); device.Upload(beta, bb);
        using var rs = GdnScanMultiTokenF32Kernel.Create(device, spvDir, GdnScanMultiTokenF32Kernel.Variant.RowSplit);
        using var ch = GdnChunkedScanF32Kernel.Create(device, spvDir);
        ch.Reserve(seqLen, nVHead);
        double Time(Action<nint> rec)
        {
            double best = double.MaxValue;
            for (int trial = 0; trial < 6; trial++)
            {
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int r = 0; r < 10; r++) { rec(ctx.CommandBuffer); KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer); }
                var sw = Stopwatch.StartNew();
                ctx.SubmitAndWait();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds / 10);
            }
            return best;
        }
        for (int rep = 0; rep < 2; rep++)
        {
            double a = Time(c => rs.Record(c, bs, bq, bk, bv, bg, bb, bo, seqLen, nVHead, nKHead, dState));
            double b = Time(c => ch.Record(c, bs, bq, bk, bv, bg, bb, bo, seqLen, nVHead, nKHead));
            ch.StageMask = 1;
            double b1 = Time(c => ch.Record(c, bs, bq, bk, bv, bg, bb, bo, seqLen, nVHead, nKHead));
            ch.StageMask = 2;
            double b2 = Time(c => ch.Record(c, bs, bq, bk, bv, bg, bb, bo, seqLen, nVHead, nKHead));
            ch.StageMask = 3;
            _out.WriteLine($"seqLen {seqLen}: rowsplit {a:F3} ms, chunked {b:F3} ms ({a / b:F2}x); prep-only {b1:F3} ms, scan-only {b2:F3} ms");
        }
    }
}
