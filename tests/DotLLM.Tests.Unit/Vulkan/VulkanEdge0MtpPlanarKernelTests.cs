using System.Runtime.InteropServices;
using DotLLM.Models.Edge0;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0 MTP head (G6, #1990): the planar Q2_0 indexed MMVQ reads edge0-strata's expert blobs in place. Oracle: the CONTIGUOUS Q2_0 kernel (same dp4a math, ggml block
/// addressing) run on banks relaid out from the same blobs by <see cref="Edge0MtpPack.RelayoutToGgmlBanks"/> (itself checked against an independent planar decoder in
/// <c>Edge0MtpPackTests</c>), on the REAL expert bytes: results must be bit-identical for gate, up and down, with expert indices that mix and repeat.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanEdge0MtpPlanarKernelTests
{
    [SkippableTheory]
    [InlineData(PlanarQ2Projection.Gate, 10)]
    [InlineData(PlanarQ2Projection.Up, 10)]
    [InlineData(PlanarQ2Projection.Down, 10)]
    [InlineData(PlanarQ2Projection.Up, 1)]
    public unsafe void Planar_IsBitIdenticalToTheContiguousKernelOnRelaidOutBanks(PlanarQ2Projection proj, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var fx = TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf", "native_experts.txt", "experts.bin", "ple-fp8.gguf", "mtp/experts.bin");
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k (mtp/)"));
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var q8 = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var planar = MoeIndexedPlanarQ2MmvqKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("planar spv missing.");
        using var plain = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, MoeIqQuant.Q2_0) ?? throw new Xunit.Sdk.XunitException("Q2_0 spv missing.");
        using var pack = Edge0MtpPack.Open(Edge0MtpPack.Find(Path.GetDirectoryName(fx.Path!)!)!, 248320);

        const int E = 16;
        bool down = proj == PlanarQ2Projection.Down;
        int m = down ? 2560 : 640, k = down ? 640 : 2560;
        var (gate, up, dn) = pack.RelayoutToGgmlBanks();
        nint src = proj switch { PlanarQ2Projection.Gate => gate, PlanarQ2Projection.Up => up, _ => dn };
        long bankBytes = (long)E * m * (k / 64) * 18;
        var blobs = new byte[E * Edge0MtpPack.ExpertBlobBytes];
        new ReadOnlySpan<byte>((void*)pack.ExpertBlob(0), blobs.Length).CopyTo(blobs);
        var split = new byte[bankBytes];
        new ReadOnlySpan<byte>((void*)src, split.Length).CopyTo(split);

        var rng = new Random(0x6A + (int)proj + n);
        float[] x = new float[n * k]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] idx = MoeMmvqParitySupport.RandomIndices(rng, n, E, Math.Min(n, 6));
        using var bBlobs = device.Allocate((blobs.Length + 3L) & ~3L); using var bSplit = device.Allocate((split.Length + 3L) & ~3L);
        using var bX = device.Allocate(x.Length * 4L); using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bIdx = device.Allocate(n * 4L); using var yA = device.Allocate((long)n * m * 4); using var yB = device.Allocate((long)n * m * 4);
        device.Upload(blobs, bBlobs); device.Upload(split, bSplit); device.Upload(x, bX); device.Upload(MemoryMarshal.AsBytes<int>(idx), bIdx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q8.Record(ctx.CommandBuffer, bX, bXq, bXds, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            planar.Record(ctx.CommandBuffer, bBlobs, bXq, bXds, bIdx, yA, proj, m, k, n, E);
            plain.Record(ctx.CommandBuffer, bSplit, bXq, bXds, bIdx, yB, m, k, n, E);
            ctx.SubmitAndWait();
        }
        float[] a = new float[n * m], b = new float[n * m]; device.Download(yA, a); device.Download(yB, b);
        Assert.True(b.Max(MathF.Abs) > 1e-3f, "reference degenerate");
        for (int i = 0; i < a.Length; i++) Assert.True(a[i].Equals(b[i]), $"{proj} idx {i} (row {i / m}, col {i % m}, expert {idx[i / m]}): planar {a[i]:G9} vs contiguous {b[i]:G9}");
    }
}
