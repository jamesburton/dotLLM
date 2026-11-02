using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase G Task 4, CPU half: the backend-aware <see cref="SelectiveRecomputer.Recompute(IModel, IChunkBackend, IKvCache, IReadOnlyList{ChunkKvEntry}, ReadOnlySpan{int}, RecomputeMode, int)"/>
/// keeps the per-run loop on the CPU (bit-identical to <see cref="SelectiveRecomputer.Run"/>) and refuses the single-pass mode there, because
/// the CPU attention masks a query at <c>positions[0] + i</c>.
/// </summary>
public sealed class VulkanRecomputeTests : IDisposable
{
    internal const string Fingerprint = BackendSeamTests.Fingerprint;

    // Four chunks (L = 22) and a suffix, as SelectiveRecomputeTests; context + suffix fit SyntheticDenseGguf.Context = 64.
    internal static readonly int[] ChunkA = [3, 17, 5, 29, 11, 40];
    internal static readonly int[] ChunkB = [8, 22, 1, 35];
    internal static readonly int[] ChunkC = [12, 7, 44, 2, 19];
    internal static readonly int[] ChunkD = [31, 26, 0, 45, 16, 23, 10];
    internal static readonly int[] Suffix = [9, 14, 38];
    internal static readonly int[][] Chunks = [ChunkA, ChunkB, ChunkC, ChunkD];
    internal static readonly int[] Context = [.. ChunkA, .. ChunkB, .. ChunkC, .. ChunkD];

    /// <summary>A scattered selection with runs of 1, 2 and 3 positions, in three different chunks (4 runs).</summary>
    internal static readonly int[] Scattered = [2, 7, 8, 13, 14, 15, 19];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkrecompute-" + Guid.NewGuid().ToString("N"));

    public VulkanRecomputeTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public void CpuBackend_AutoAndPerRun_EqualLegacyRunBitExact()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        List<ChunkKvEntry> entries = Encode(tiny.Model, cpu);
        using IKvCache legacy = Assembled(tiny.Config, cpu, entries);
        SelectiveRecomputeResult expected = SelectiveRecomputer.Run(tiny.Model, legacy, entries, Scattered);
        Assert.Equal(4, expected.ForwardCalls);

        foreach (RecomputeMode mode in new[] { RecomputeMode.Auto, RecomputeMode.PerRun })
        {
            using IKvCache cache = Assembled(tiny.Config, cpu, entries);
            SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(tiny.Model, cpu, cache, entries, Scattered, mode);
            Assert.Equal(expected.ForwardCalls, r.ForwardCalls);
            Assert.Equal(Scattered.Length, r.TokensRecomputed);
            Assert.Equal(0, r.QueryPositionAttentionLayers);
            for (int l = 0; l < tiny.Config.NumLayers; l++)
            {
                (float[] ek, float[] ev) = Read(cpu, legacy, l, Context.Length, tiny.KvStride);
                (float[] ak, float[] av) = Read(cpu, cache, l, Context.Length, tiny.KvStride);
                Assert.Equal(ek, ak);
                Assert.Equal(ev, av);
            }
        }
    }

    [Fact]
    public void CpuBackend_SinglePass_IsRefusedBeforeWriting()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        List<ChunkKvEntry> entries = Encode(tiny.Model, cpu);
        using IKvCache cache = Assembled(tiny.Config, cpu, entries);
        (float[] k0, float[] v0) = Read(cpu, cache, 1, Context.Length, tiny.KvStride);
        var ex = Assert.Throws<NotSupportedException>(() =>
            SelectiveRecomputer.Recompute(tiny.Model, cpu, cache, entries, Scattered, RecomputeMode.SinglePass));
        Assert.Contains("single-pass", ex.Message, StringComparison.OrdinalIgnoreCase);
        (float[] k1, float[] v1) = Read(cpu, cache, 1, Context.Length, tiny.KvStride);
        Assert.Equal(k0, k1);
        Assert.Equal(v0, v1);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void Recompute_NonPositiveRowCap_IsRefused(int cap)
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        List<ChunkKvEntry> entries = Encode(tiny.Model, cpu);
        using IKvCache cache = Assembled(tiny.Config, cpu, entries);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SelectiveRecomputer.Recompute(tiny.Model, cpu, cache, entries, Scattered, RecomputeMode.Auto, cap));
    }

    [Fact]
    public void Recompute_TokenOverload_EqualsChunkOverload_AndValidatesLikeRun()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        List<ChunkKvEntry> entries = Encode(tiny.Model, cpu);
        using IKvCache a = Assembled(tiny.Config, cpu, entries);
        using IKvCache b = Assembled(tiny.Config, cpu, entries);
        SelectiveRecomputer.Recompute(tiny.Model, cpu, a, entries, Scattered);
        SelectiveRecomputer.Recompute(tiny.Model, cpu, b, Context, Scattered);
        for (int l = 0; l < tiny.Config.NumLayers; l++)
            Assert.Equal(Read(cpu, a, l, Context.Length, tiny.KvStride).K, Read(cpu, b, l, Context.Length, tiny.KvStride).K);

        Assert.Throws<ArgumentException>(() => SelectiveRecomputer.Recompute(tiny.Model, cpu, b, Context, [3, 3]));
        Assert.Throws<ArgumentException>(() => SelectiveRecomputer.Recompute(tiny.Model, cpu, b, Context, [Context.Length]));
        Assert.Throws<ArgumentException>(() => SelectiveRecomputer.Recompute(tiny.Model, cpu, b, Context.Append(1).ToArray(), [0]));
    }

    internal static List<ChunkKvEntry> Encode(IModel model, IChunkBackend backend)
    {
        var encoder = new ChunkEncoder(Fingerprint);
        return Chunks.Select(c => encoder.Encode(model, backend, c)).ToList();
    }

    internal static IKvCache Assembled(ModelConfig config, IChunkBackend backend, IReadOnlyList<ChunkKvEntry> entries)
    {
        IKvCache cache = backend.CreateCache(SyntheticDenseGguf.Context);
        new ChunkAssembler(config, backend).Assemble(cache, entries, Fingerprint);
        return cache;
    }

    internal static (float[] K, float[] V) Read(IChunkBackend backend, IKvCache cache, int layer, int count, int stride, int start = 0)
    {
        float[] k = new float[count * stride], v = new float[count * stride];
        backend.ReadRows(cache, layer, start, count, k, v);
        return (k, v);
    }
}

/// <summary>
/// GPU half: single-pass selective recompute on Vulkan (one <c>Forward(subsetIds, subsetPositions, cache)</c> through the per-query-position
/// attention, dotLLM #1534) on the synthetic F32 model. Tolerances: within Vulkan, the same forward shape is bit-exact (G0 pair (c)); forwards
/// on the same attention kernel but of different shapes differ by F32 rounding only (no Q4_K dispatch regimes exist here; G0 Scope), so 1e-4
/// absolute on values of magnitude ~5. The ordinary Vulkan prefill attention and the per-query-position kernel differ by ~3e-3 past layer 0
/// (measured here; the per-query-position kernel matches the CPU to ~5e-6), so comparisons across the two kernels, and CPU vs Vulkan, use
/// Task 3's 1e-2 past layer 0 and 1e-5 at layer 0.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanRecomputeTestsGpu : IDisposable
{
    private const float SameBackendTol = 1e-4f;
    private const float Layer0Tol = 1e-5f;
    private const float CrossBackendDeepTol = 1e-2f;
    private static readonly int[] Context = VulkanRecomputeTests.Context;
    private static readonly int L = Context.Length;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkrecompute-gpu-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public VulkanRecomputeTestsGpu(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    /// <summary>The synthetic model on the CPU and on Vulkan, the Vulkan chunk backend on the model's device, and Vulkan-encoded chunks.</summary>
    private sealed class Rig : IDisposable
    {
        public Rig(string scratch, string spvDir)
        {
            Tiny = BackendSeamTests.TinyModel.Load(scratch, "qwen3");
            Device = VulkanDevice.Create();
            Model = VulkanTransformerModel.LoadFromGguf(Device, Tiny.Gguf, Tiny.Config, spvDir);
            Backend = new VulkanChunkBackend(Device, spvDir, Tiny.Config, ChunkModelFeatures.FromGguf(Tiny.Gguf));
            Cpu = new CpuChunkBackend(Tiny.Config);
            Entries = VulkanRecomputeTests.Encode(Model, Backend);
        }

        public BackendSeamTests.TinyModel Tiny { get; }
        public VulkanDevice Device { get; }
        public VulkanTransformerModel Model { get; }
        public VulkanChunkBackend Backend { get; }
        public CpuChunkBackend Cpu { get; }
        public List<ChunkKvEntry> Entries { get; }
        public int Layers => Tiny.Config.NumLayers;
        public int Stride => Tiny.KvStride;

        /// <summary>A naive-concatenation Vulkan cache of the four chunks (length L).</summary>
        public IKvCache Naive() => VulkanRecomputeTests.Assembled(Tiny.Config, Backend, Entries);

        /// <summary>Plain Vulkan prefill of <paramref name="ids"/> at positions 0.. in one forward; returns the last row's logits.</summary>
        public unsafe IKvCache Prefill(int[] ids, out float[] lastLogits)
        {
            IKvCache cache = Backend.CreateCache(SyntheticDenseGguf.Context);
            using ITensor t = ((IModel)Model).Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), 0, cache, lastTokenLogitsOnly: true);
            int vocab = t.Shape[1];
            lastLogits = new ReadOnlySpan<float>((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab).ToArray();
            return cache;
        }

        public (float[] K, float[] V) Read(IKvCache cache, int layer, int count = -1, int start = 0)
            => VulkanRecomputeTests.Read(Backend, cache, layer, count < 0 ? L : count, Stride, start);

        public void Dispose()
        {
            Backend.Dispose();
            Model.Dispose();
            Device.Dispose();
            Tiny.Dispose();
        }
    }

    [SkippableFact]
    public void Recompute_AllPositions_EqualsFullPrefillWithinFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        using IKvCache full = rig.Prefill(Context, out float[] fullLast);
        int[] all = Enumerable.Range(0, L).ToArray();

        // (1) Cache of exactly L rows: all positions are one contiguous run ending at the cache length, i.e. an ordinary prefill
        // forward of the same shape as the reference (flash path, no per-query-position kernel): bit-exact (pair (c)).
        using (IKvCache naive = rig.Naive())
        {
            long before = rig.Model.AttentionQueryPositionsCount;
            SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, naive, rig.Entries, all);
            Assert.Equal(0, rig.Model.AttentionQueryPositionsCount - before);
            Assert.Equal(1, r.ForwardCalls);
            Assert.Equal(L, r.TokensRecomputed);
            Assert.Equal(fullLast, r.LastPositionLogits);
            for (int l = 0; l < rig.Layers; l++)
            {
                Assert.Equal(rig.Read(full, l).K, rig.Read(naive, l).K);
                Assert.Equal(rig.Read(full, l).V, rig.Read(naive, l).V);
            }
        }

        // (2) The same, with suffix rows after the context (the cache is longer than the recomputed span): the per-query-position
        // kernel runs on every layer, and the result equals the full prefill to F32 rounding. Rows >= L are left untouched.
        using (IKvCache naive = rig.Naive())
        {
            AppendSuffix(rig, naive);
            (float[] sk, float[] sv) = rig.Read(naive, rig.Layers - 1, VulkanRecomputeTests.Suffix.Length, L);
            long before = rig.Model.AttentionQueryPositionsCount;
            SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, naive, rig.Entries, all);
            Assert.Equal(rig.Layers, rig.Model.AttentionQueryPositionsCount - before);
            Assert.Equal(1, r.ForwardCalls);
            Assert.Equal(rig.Layers, r.QueryPositionAttentionLayers);
            Assert.Equal(L + VulkanRecomputeTests.Suffix.Length, naive.CurrentLength);
            // Measured on this model: the per-query-position kernel (plain F32) matches the CPU full prefill to ~5e-6, while the ordinary
            // Vulkan prefill attention differs from both by ~3e-3 past layer 0 (it is the source of the synthetic model's CPU-vs-Vulkan gap).
            // So the tight oracle is the CPU full prefill; against the Vulkan full prefill the bound is the cross-kernel 1e-2.
            float vsVulkan = 0, vsCpu = 0;
            using IKvCache cpuFull = rig.Cpu.CreateCache(SyntheticDenseGguf.Context);
            float[] cpuLast;
            unsafe
            {
                using ITensor t = rig.Tiny.Model.Forward(Context, all, -1, cpuFull, lastTokenLogitsOnly: true);
                int vocab = t.Shape[1];
                cpuLast = new ReadOnlySpan<float>((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab).ToArray();
            }
            for (int l = 0; l < rig.Layers; l++)
            {
                (float[] fk, float[] fv) = rig.Read(full, l);
                (float[] qk, float[] qv) = rig.Read(naive, l);
                (float[] ck, float[] cv) = VulkanRecomputeTests.Read(rig.Cpu, cpuFull, l, L, rig.Stride);
                float dv = MathF.Max(BackendSeamTests.MaxAbsDiff(fk, qk), BackendSeamTests.MaxAbsDiff(fv, qv));
                float dc = MathF.Max(BackendSeamTests.MaxAbsDiff(ck, qk), BackendSeamTests.MaxAbsDiff(cv, qv));
                vsVulkan = MathF.Max(vsVulkan, dv);
                vsCpu = MathF.Max(vsCpu, dc);
                _output.WriteLine($"layer {l}: qpos vs Vulkan full {dv:E3}; qpos vs CPU full {dc:E3}; Vulkan full vs CPU full "
                    + $"{MathF.Max(BackendSeamTests.MaxAbsDiff(ck, fk), BackendSeamTests.MaxAbsDiff(cv, fv)):E3}");
            }
            float logitsVk = BackendSeamTests.MaxAbsDiff(fullLast, r.LastPositionLogits!);
            float logitsCpu = BackendSeamTests.MaxAbsDiff(cpuLast, r.LastPositionLogits!);
            _output.WriteLine($"all positions via ATTN_QPOS: K/V max |d| vs Vulkan full {vsVulkan:E3}, vs CPU full {vsCpu:E3}; "
                + $"last logits {logitsVk:E3} / {logitsCpu:E3}");
            Assert.True(vsCpu <= SameBackendTol, $"K/V vs CPU full prefill {vsCpu}");
            Assert.True(logitsCpu <= SameBackendTol * 10, $"logits vs CPU full prefill {logitsCpu}");
            Assert.True(vsVulkan <= CrossBackendDeepTol, $"K/V vs Vulkan full prefill {vsVulkan}");
            Assert.True(logitsVk <= CrossBackendDeepTol, $"logits vs Vulkan full prefill {logitsVk}");
            (float[] sk2, float[] sv2) = rig.Read(naive, rig.Layers - 1, VulkanRecomputeTests.Suffix.Length, L);
            Assert.Equal(sk, sk2);
            Assert.Equal(sv, sv2);
        }
    }

    [SkippableFact]
    public void Recompute_SinglePosition_EqualsPerRunPathWithinFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        int[] one = [13];
        using IKvCache single = rig.Naive(), perRun = rig.Naive(), naive = rig.Naive();
        long before = rig.Model.AttentionQueryPositionsCount;
        SelectiveRecomputeResult s = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, single, rig.Entries, one, RecomputeMode.SinglePass);
        Assert.Equal(rig.Layers, rig.Model.AttentionQueryPositionsCount - before); // a row inside a filled cache: per-query-position kernel
        SelectiveRecomputeResult p = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, perRun, rig.Entries, one, RecomputeMode.PerRun);
        Assert.Equal(1, s.ForwardCalls);
        Assert.Equal(1, p.ForwardCalls);
        Assert.Null(s.LastPositionLogits);
        bool changed = false;
        for (int l = 0; l < rig.Layers; l++)
        {
            // One position is the same forward either way: bit-exact (pair (c)).
            Assert.Equal(rig.Read(perRun, l).K, rig.Read(single, l).K);
            Assert.Equal(rig.Read(perRun, l).V, rig.Read(single, l).V);
            // Only row 13 moved, and it did move at the deeper layers (the recompute ran).
            (float[] nk, float[] nv) = rig.Read(naive, l);
            (float[] sk, float[] sv) = rig.Read(single, l);
            for (int pos = 0; pos < L; pos++)
            {
                float d = MathF.Max(BackendSeamTests.MaxAbsDiff(nk.AsSpan(pos * rig.Stride, rig.Stride), sk.AsSpan(pos * rig.Stride, rig.Stride)),
                    BackendSeamTests.MaxAbsDiff(nv.AsSpan(pos * rig.Stride, rig.Stride), sv.AsSpan(pos * rig.Stride, rig.Stride)));
                if (pos != 13) Assert.Equal(0f, d);
                else if (l > 0 && d > 1e-3f) changed = true;
            }
        }
        Assert.True(changed, "fixture: recomputing position 13 changed nothing past layer 0");
    }

    [SkippableFact]
    public void Recompute_RandomSubset_VulkanVsCpu_WithinFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        var rng = new Random(1234);
        int[] subset = Enumerable.Range(0, L).Where(_ => rng.NextDouble() < 0.4).ToArray();
        Assert.True(subset.Length >= 4);

        List<ChunkKvEntry> cpuEntries = VulkanRecomputeTests.Encode(rig.Tiny.Model, rig.Cpu);
        using IKvCache cpu = VulkanRecomputeTests.Assembled(rig.Tiny.Config, rig.Cpu, cpuEntries);
        SelectiveRecomputer.Recompute(rig.Tiny.Model, rig.Cpu, cpu, cpuEntries, subset);
        using IKvCache vk = rig.Naive();
        SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, vk, rig.Entries, subset);
        Assert.Equal(1, r.ForwardCalls);
        using IKvCache naive = rig.Naive();

        float deep = 0, layer0 = 0, naiveGap = 0;
        for (int l = 0; l < rig.Layers; l++)
        {
            (float[] ck, float[] cv) = VulkanRecomputeTests.Read(rig.Cpu, cpu, l, L, rig.Stride);
            (float[] vk2, float[] vv) = rig.Read(vk, l);
            float d = MathF.Max(BackendSeamTests.MaxAbsDiff(ck, vk2), BackendSeamTests.MaxAbsDiff(cv, vv));
            if (l == 0) layer0 = d;
            else deep = MathF.Max(deep, d);
            if (l > 0)
            {
                (float[] nk, float[] nv) = rig.Read(naive, l);
                naiveGap = MathF.Max(naiveGap, MathF.Max(BackendSeamTests.MaxAbsDiff(nk, vk2), BackendSeamTests.MaxAbsDiff(nv, vv)));
            }
        }
        _output.WriteLine($"subset {string.Join(',', subset)}: CPU per-run vs Vulkan single-pass layer 0 {layer0:E3}, deeper {deep:E3}; naive vs recomputed {naiveGap:E3}");
        Assert.True(layer0 <= Layer0Tol, $"layer 0 {layer0}");
        Assert.True(deep <= CrossBackendDeepTol, $"deeper layers {deep}");
        Assert.True(naiveGap > 10 * CrossBackendDeepTol, $"fixture cannot discriminate: recompute moved rows by only {naiveGap}");
    }

    [SkippableFact]
    public void Recompute_NonContiguousSubset_UsesOneForwardCall()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        int[] subset = VulkanRecomputeTests.Scattered;
        using IKvCache single = rig.Naive(), perRun = rig.Naive();

        long before = rig.Model.AttentionQueryPositionsCount;
        SelectiveRecomputeResult s = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, single, rig.Entries, subset);
        long singleDispatches = rig.Model.AttentionQueryPositionsCount - before;
        before = rig.Model.AttentionQueryPositionsCount;
        SelectiveRecomputeResult p = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, perRun, rig.Entries, subset, RecomputeMode.PerRun);
        long perRunDispatches = rig.Model.AttentionQueryPositionsCount - before;

        Assert.Equal(1, s.ForwardCalls);                 // Auto on Vulkan = single pass
        Assert.Equal(rig.Layers, singleDispatches);      // one per-query-position attention dispatch per layer
        Assert.Equal(rig.Layers, s.QueryPositionAttentionLayers);
        Assert.Equal(4, p.ForwardCalls);                 // runs {2}, {7,8}, {13,14,15}, {19}
        Assert.Equal(4L * rig.Layers, perRunDispatches); // every run sits inside the filled cache
        Assert.Equal(subset.Length, s.TokensRecomputed);
        Assert.Equal(L, single.CurrentLength);

        // The single pass equals the per-run loop (one-pass layer-wise recompute either way) to F32 rounding.
        float worst = 0;
        for (int l = 0; l < rig.Layers; l++)
        {
            worst = MathF.Max(worst, BackendSeamTests.MaxAbsDiff(rig.Read(perRun, l).K, rig.Read(single, l).K));
            worst = MathF.Max(worst, BackendSeamTests.MaxAbsDiff(rig.Read(perRun, l).V, rig.Read(single, l).V));
        }
        _output.WriteLine($"single pass vs per-run on {string.Join(',', subset)}: max |d| {worst:E3}");
        Assert.True(worst <= SameBackendTol, $"single pass vs per-run {worst}");
    }

    [SkippableFact]
    public void Recompute_RowCap_SplitsIntoAscendingBatches()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        int[] subset = VulkanRecomputeTests.Scattered; // 7 rows
        using IKvCache whole = rig.Naive(), capped = rig.Naive();
        SelectiveRecomputer.Recompute(rig.Model, rig.Backend, whole, rig.Entries, subset);
        long before = rig.Model.AttentionQueryPositionsCount;
        SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, capped, rig.Entries, subset, RecomputeMode.SinglePass, 3);
        Assert.Equal(3, r.ForwardCalls); // [2,7,8] [13,14,15] [19]
        Assert.Equal(3L * rig.Layers, rig.Model.AttentionQueryPositionsCount - before);
        float worst = 0;
        for (int l = 0; l < rig.Layers; l++)
        {
            worst = MathF.Max(worst, BackendSeamTests.MaxAbsDiff(rig.Read(whole, l).K, rig.Read(capped, l).K));
            worst = MathF.Max(worst, BackendSeamTests.MaxAbsDiff(rig.Read(whole, l).V, rig.Read(capped, l).V));
        }
        Assert.True(worst <= SameBackendTol, $"capped vs whole {worst}");
    }

    [SkippableFact]
    public void Recompute_FirstAndLastPosition_Edge()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        using IKvCache full = rig.Prefill(Context, out float[] fullLast);
        int[] edges = [0, L - 1];
        using IKvCache single = rig.Naive(), perRun = rig.Naive();
        SelectiveRecomputeResult s = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, single, rig.Entries, edges);
        SelectiveRecomputeResult p = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, perRun, rig.Entries, edges, RecomputeMode.PerRun);
        Assert.Equal(1, s.ForwardCalls);
        Assert.Equal(2, p.ForwardCalls);

        // L-1 was selected: its logits come back, and both paths agree.
        Assert.NotNull(s.LastPositionLogits);
        Assert.NotNull(p.LastPositionLogits);
        float dl = BackendSeamTests.MaxAbsDiff(p.LastPositionLogits!, s.LastPositionLogits!);
        Assert.True(dl <= SameBackendTol * 10, $"last logits single vs per-run {dl}");
        // Not the full prefill's logits: the middle of the cache is stale.
        Assert.True(BackendSeamTests.MaxAbsDiff(fullLast, s.LastPositionLogits!) > 1e-3f, "fixture: stale middle should change the last logits");

        for (int l = 0; l < rig.Layers; l++)
        {
            (float[] fk, float[] fv) = rig.Read(full, l, 1);
            (float[] sk, float[] sv) = rig.Read(single, l, 1);
            // Position 0 attends only to itself: its recompute equals the full prefill's row 0 at every layer.
            Assert.True(BackendSeamTests.MaxAbsDiff(fk, sk) <= SameBackendTol && BackendSeamTests.MaxAbsDiff(fv, sv) <= SameBackendTol, $"row 0, layer {l}");
            Assert.True(BackendSeamTests.MaxAbsDiff(rig.Read(perRun, l).K, rig.Read(single, l).K) <= SameBackendTol, $"K layer {l}");
            Assert.True(BackendSeamTests.MaxAbsDiff(rig.Read(perRun, l).V, rig.Read(single, l).V) <= SameBackendTol, $"V layer {l}");
        }

        // The last position alone is a run ending at the cache length: the ordinary kernel, and the same forward as per-run.
        using IKvCache last = rig.Naive();
        long before = rig.Model.AttentionQueryPositionsCount;
        SelectiveRecomputeResult r = SelectiveRecomputer.Recompute(rig.Model, rig.Backend, last, rig.Entries, [L - 1]);
        Assert.Equal(0, rig.Model.AttentionQueryPositionsCount - before);
        Assert.Equal(0, r.QueryPositionAttentionLayers);
        Assert.NotNull(r.LastPositionLogits);
    }

    [SkippableFact]
    public void Recompute_VulkanBackend_RefusesHostModelAndHostCache()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        using (IKvCache vk = rig.Naive())
        {
            // A CPU model cannot forward over a Vulkan cache with the Vulkan backend's device id.
            Assert.Throws<NotSupportedException>(() => SelectiveRecomputer.Recompute(rig.Tiny.Model, rig.Backend, vk, rig.Entries, [3, 9]));
            using IKvCache fresh = rig.Naive();
            Assert.Equal(rig.Read(fresh, 2).K, rig.Read(vk, 2).K);
        }
        // A host cache is not a Vulkan backend cache.
        List<ChunkKvEntry> cpuEntries = VulkanRecomputeTests.Encode(rig.Tiny.Model, rig.Cpu);
        using IKvCache host = VulkanRecomputeTests.Assembled(rig.Tiny.Config, rig.Cpu, cpuEntries);
        Assert.Throws<NotSupportedException>(() => SelectiveRecomputer.Recompute(rig.Model, rig.Backend, host, cpuEntries, [3, 9]));
        // And the CPU backend refuses the Vulkan model (its forwards would run on the device with deviceId -1).
        using IKvCache vk2 = rig.Naive();
        Assert.Throws<NotSupportedException>(() => SelectiveRecomputer.Recompute(rig.Model, rig.Cpu, vk2, rig.Entries, [3, 9]));
    }

    [SkippableFact]
    public void Recompute_ModelWithoutQueryPositionKernel_IsDetected()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // A SPIR-V directory without attention_f32_qpos.spv: the model then routes a non-contiguous forward through the ordinary
        // kernel, which masks by positions[0] + row. The recomputer must notice (the per-query-position counter did not move) and throw.
        string noQpos = Path.Combine(_scratch, "spv-noqpos");
        Directory.CreateDirectory(noQpos);
        foreach (string f in Directory.EnumerateFiles(spvDir))
            if (!Path.GetFileName(f).StartsWith("attention_f32_qpos", StringComparison.Ordinal))
                File.Copy(f, Path.Combine(noQpos, Path.GetFileName(f)));
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.LoadFromGguf(device, tiny.Gguf, tiny.Config, noQpos);
        using var backend = new VulkanChunkBackend(device, spvDir, tiny.Config, ChunkModelFeatures.FromGguf(tiny.Gguf));
        List<ChunkKvEntry> entries = VulkanRecomputeTests.Encode(model, backend);
        using IKvCache cache = VulkanRecomputeTests.Assembled(tiny.Config, backend, entries);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            SelectiveRecomputer.Recompute(model, backend, cache, entries, VulkanRecomputeTests.Scattered));
        Assert.Contains("per-query-position", ex.Message, StringComparison.Ordinal);
    }

    private static void AppendSuffix(Rig rig, IKvCache cache)
    {
        int[] s = VulkanRecomputeTests.Suffix;
        using ITensor _ = ((IModel)rig.Model).Forward(s, Enumerable.Range(L, s.Length).ToArray(), 0, cache, lastTokenLogitsOnly: true);
    }
}
