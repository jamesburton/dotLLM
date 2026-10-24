using DotLLM.Models.Architectures.DFlash;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>DFlash on the Vulkan <c>qwen35</c> trunk (issues #1175, #1193): GPU verify and GPU drafter, token-identical to greedy.</summary>
[Collection("VulkanKernels")]
public sealed class VulkanDFlashTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-vk-dflash-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;
    private readonly List<VulkanDevice> _devices = [];

    public VulkanDFlashTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        foreach (var d in _devices) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    private sealed unsafe class DeviceMemory(VulkanDFlashBackend backend) : DFlashDrafterTests.ITargetMemory
    {
        public void Up(float* src, nint dst, int count) => backend.UploadForTest(src, dst, count);

        public void Down(nint src, float* dst, int count) => backend.DownloadForTest(src, dst, count);
    }

    /// <summary>
    /// The Vulkan drafter (device weights, F16 ring, ring attention with window and rollback, convolutions, offset views) against
    /// the stateless reference; the head reads drafted rows at a non-zero offset into the block scratch.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Draft_OnDevice_MatchesStatelessReference(bool dflash2)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string trunk = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_dir, "trunk.gguf"), withMtp: false);
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var device = VulkanDevice.Create();
        _devices.Add(device);
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        VulkanDFlashBackend? backend = null;
        // F32 fixture weights; the tolerance covers the GPU reduction order.
        DFlashDrafterTests.AssertMatchesReference(_dir, dflash2,
            () => backend = (VulkanDFlashBackend)model.CreateDFlashBackend(),
            t => new DFlashDrafterTests.StagedTarget(t, new DeviceMemory(backend!)), 1e-3f, _out);
    }

    /// <summary>
    /// The ring attention over a context longer than one shared-memory tile (512 keys), with the window masking the oldest keys
    /// of the earliest queries, a ring that wraps, a first tile masked entirely (the online softmax starts at -inf), GQA, a head dim that leaves part of the second dim-per-thread idle, and the
    /// layer's rings at a non-zero offset: against a double-precision reference.
    /// </summary>
    [SkippableFact]
    public void Attention_LongContextAcrossTiles_MatchesReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int B = 5, Heads = 4, KvHeads = 2, Hd = 192, Cap = 1100, Lo = 300, Start = 1350, Window = 500;   // the first 512-key tile is fully masked for every query
        int kvDim = KvHeads * Hd, qDim = Heads * Hd;
        long layerHalves = (long)Cap * kvDim;    // ring K of layer 0 precedes the ones under test
        var rng = new Random(5);
        float R() => rng.NextSingle() * 2 - 1;
        var q = new float[B * qDim];
        var kb = new float[B * kvDim];
        var vb = new float[B * kvDim];
        for (int i = 0; i < q.Length; i++) q[i] = R();
        for (int i = 0; i < kb.Length; i++) { kb[i] = R(); vb[i] = R(); }
        var ringK = new Half[2 * layerHalves];
        var ringV = new Half[2 * layerHalves];
        for (int i = 0; i < ringK.Length; i++) { ringK[i] = (Half)R(); ringV[i] = (Half)R(); }

        var device = VulkanDevice.Create();
        _devices.Add(device);
        using var kernels = global::DotLLM.Vulkan.Kernels.VulkanDFlashKernels.Create(device, spvDir);
        using var qB = device.Allocate(q.Length * 4L);
        using var kB = device.Allocate(kb.Length * 4L);
        using var vB = device.Allocate(vb.Length * 4L);
        using var rkB = device.Allocate(ringK.Length * 2L);
        using var rvB = device.Allocate(ringV.Length * 2L);
        using var oB = device.AllocateHostReadback(q.Length * 4L);
        device.Upload(q, qB);
        device.Upload(kb, kB);
        device.Upload(vb, vB);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(ringK.AsSpan()), rkB);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(ringV.AsSpan()), rvB);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernels.RecordAttention(ctx.CommandBuffer, qB, kB, vB, rkB, rvB, oB, B, Heads, KvHeads, Hd, Cap, Lo, Start, Window,
                1f / MathF.Sqrt(Hd), layerHalves * 2, layerHalves * 2);
            ctx.SubmitAndWait();
        }
        var got = new float[q.Length];
        device.Download(oB, got);

        int masked = 0;
        for (int row = 0; row < B; row++)
            for (int h = 0; h < Heads; h++)
            {
                int kvh = h / (Heads / KvHeads);
                var scores = new List<(double S, Func<int, double> V)>();
                for (int pos = Lo; pos < Start; pos++)
                {
                    if (Start + row - pos >= Window) { masked++; continue; }
                    long o = layerHalves + (long)(pos % Cap) * kvDim + kvh * Hd;
                    double s = 0;
                    for (int d = 0; d < Hd; d++) s += q[row * qDim + h * Hd + d] * (double)ringK[o + d];
                    scores.Add((s / Math.Sqrt(Hd), d => (double)ringV[o + d]));
                }
                for (int j = 0; j < B; j++)
                {
                    double s = 0;
                    int o = j * kvDim + kvh * Hd;
                    for (int d = 0; d < Hd; d++) s += q[row * qDim + h * Hd + d] * (double)kb[o + d];
                    scores.Add((s / Math.Sqrt(Hd), d => vb[o + d]));
                }
                double max = scores.Max(x => x.S), sum = scores.Sum(x => Math.Exp(x.S - max));
                for (int d = 0; d < Hd; d++)
                {
                    double want = scores.Sum(x => Math.Exp(x.S - max) * x.V(d)) / sum;
                    float g = got[row * qDim + h * Hd + d];
                    Assert.True(Math.Abs(g - want) <= 1e-4 + 1e-3 * Math.Abs(want), $"row {row} head {h} dim {d}: {g} vs {want}");
                }
            }
        Assert.True(masked > 0, "the window must mask some context keys");
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecode_WithDFlash_IsTokenIdenticalToGreedy(bool snapshots)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        DFlashDrafterTests.AssertTokenIdentical(_dir, (gguf, config) =>
        {
            var device = VulkanDevice.Create();
            _devices.Add(device);
            var m = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            return (m, m.CreateKvCache(config.MaxSequenceLength));
        }, snapshots, _out);
    }
}
