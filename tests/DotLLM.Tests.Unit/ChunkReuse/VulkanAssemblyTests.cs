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
/// Phase G Task 3, CPU half: the backend-aware encoder and the resident (cache-to-cache) assembly path on the CPU backend reproduce the
/// pre-existing host path bit for bit.
/// </summary>
public sealed class VulkanAssemblyTests : IDisposable
{
    internal const string Fingerprint = BackendSeamTests.Fingerprint;
    internal static readonly int[] ChunkA = BackendSeamTests.ChunkA;
    internal static readonly int[] ChunkB = BackendSeamTests.ChunkB;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkvk-" + Guid.NewGuid().ToString("N"));

    public VulkanAssemblyTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public void CpuEncode_ThroughBackend_EqualsLegacyEncode()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry legacy = encoder.Encode(tiny.Model, ChunkA);
        var backend = new CpuChunkBackend(tiny.Config);
        Assert.Equal(-1, backend.ForwardDeviceId);
        ChunkKvEntry seam = encoder.Encode(tiny.Model, backend, ChunkA);
        Assert.Equal(ChunkBackendTags.Cpu, seam.Backend);
        Assert.Equal(ChunkA, seam.TokenIds);
        Assert.Equal(Fingerprint, seam.ModelFingerprint);
        for (int l = 0; l < tiny.Config.NumLayers; l++)
        {
            Assert.Equal(legacy.KeysPerLayer[l], seam.KeysPerLayer[l]);
            Assert.Equal(legacy.ValuesPerLayer[l], seam.ValuesPerLayer[l]);
        }
    }

    [Fact]
    public void CpuResidentAssembly_EqualsHostAssembly()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var backend = new CpuChunkBackend(tiny.Config);
        var encoder = new ChunkEncoder(Fingerprint);
        var assembler = new ChunkAssembler(tiny.Config, backend);
        ChunkKvEntry a = encoder.Encode(tiny.Model, backend, ChunkA), b = encoder.Encode(tiny.Model, backend, ChunkB);
        using ResidentChunkKvEntry ra = encoder.EncodeResident(tiny.Model, backend, ChunkA);
        using ResidentChunkKvEntry rb = encoder.EncodeResident(tiny.Model, backend, ChunkB);
        Assert.Equal(ChunkBackendTags.Cpu, ra.Backend);
        Assert.Equal(ChunkA.Length, ra.Length);

        using IKvCache host = backend.CreateCache(SyntheticDenseGguf.Context);
        using IKvCache resident = backend.CreateCache(SyntheticDenseGguf.Context);
        assembler.Assemble(host, [b, a, b], Fingerprint);
        assembler.Assemble(resident, [rb, ra, rb], Fingerprint);
        AssertCachesEqual(backend, host, backend, resident, tiny.Config.NumLayers, tiny.KvStride);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Assemble_FailureOnSecondChunk_RollsBackToStartLength(bool resident)
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry a = encoder.Encode(tiny.Model, cpu, ChunkA), b = encoder.Encode(tiny.Model, cpu, ChunkB);
        using ResidentChunkKvEntry ra = encoder.EncodeResident(tiny.Model, cpu, ChunkA);
        using ResidentChunkKvEntry rb = encoder.EncodeResident(tiny.Model, cpu, ChunkB);
        int layers = tiny.Config.NumLayers, stride = tiny.KvStride;

        // A destination that already holds a chunk (start > 0), and its rows as they were.
        using IKvCache dest = cpu.CreateCache(SyntheticDenseGguf.Context);
        new ChunkAssembler(tiny.Config, cpu).Assemble(dest, [b], Fingerprint);
        int start = dest.CurrentLength;
        var before = Enumerable.Range(0, layers).Select(l =>
        {
            float[] k = new float[start * stride], v = new float[start * stride];
            cpu.ReadRows(dest, l, 0, start, k, v);
            return (k, v);
        }).ToArray();

        // The backend fails on the second chunk's first row operation (after the first chunk was written and shifted).
        var failing = new FailingBackend(cpu, failAtRowOp: layers);
        var assembler = new ChunkAssembler(tiny.Config, failing);
        if (resident)
            Assert.Throws<IOException>(() => assembler.Assemble(dest, [ra, rb], Fingerprint));
        else
            Assert.Throws<IOException>(() => assembler.Assemble(dest, [a, b], Fingerprint));
        Assert.True(failing.RowOps > layers, "fixture: the first chunk was written before the failure");

        Assert.Equal(start, dest.CurrentLength);
        for (int l = 0; l < layers; l++)
        {
            float[] k = new float[start * stride], v = new float[start * stride];
            cpu.ReadRows(dest, l, 0, start, k, v);
            Assert.Equal(before[l].k, k);
            Assert.Equal(before[l].v, v);
        }
    }

    [Fact]
    public void Assembler_RefusesBackendBoundToOtherRope()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        Assert.Null(cpu.Rope); // the CPU backend rotates with whatever it is given
        RopeParameters rope = RopeParameters.FromConfig(tiny.Config);
        _ = new ChunkAssembler(tiny.Config, new FailingBackend(cpu, failAtRowOp: -1, rope: rope)); // same parameters: accepted
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChunkAssembler(tiny.Config, new FailingBackend(cpu, failAtRowOp: -1, rope: rope with { Theta = rope.Theta * 2 })));
        Assert.Contains("RoPE", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ResidentEntry_Disposed_IsRefused()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        var encoder = new ChunkEncoder(Fingerprint);
        using ResidentChunkKvEntry ra = encoder.EncodeResident(tiny.Model, cpu, ChunkA);
        ResidentChunkKvEntry rb = encoder.EncodeResident(tiny.Model, cpu, ChunkB);
        Assert.False(rb.IsDisposed);
        rb.Dispose();
        Assert.True(rb.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => rb.Cache);
        rb.Dispose(); // idempotent

        using IKvCache dest = cpu.CreateCache(SyntheticDenseGguf.Context);
        Assert.Throws<ObjectDisposedException>(() => new ChunkAssembler(tiny.Config, cpu).Assemble(dest, [ra, rb], Fingerprint));
        Assert.Equal(0, dest.CurrentLength);
    }

    [Fact]
    public void ResidentAssembly_RefusesShortSourceAndSelfDestination()
    {
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var cpu = new CpuChunkBackend(tiny.Config);
        var encoder = new ChunkEncoder(Fingerprint);
        var assembler = new ChunkAssembler(tiny.Config, cpu);
        using ResidentChunkKvEntry ra = encoder.EncodeResident(tiny.Model, cpu, ChunkA);
        using ResidentChunkKvEntry rb = encoder.EncodeResident(tiny.Model, cpu, ChunkB);

        // The destination is one of the chunks' own caches (a hand-built entry over a roomy cache, so capacity alone would not refuse it).
        ChunkKvEntry a = encoder.Encode(tiny.Model, cpu, ChunkA);
        IKvCache roomy = cpu.CreateCache(SyntheticDenseGguf.Context);
        for (int l = 0; l < tiny.Config.NumLayers; l++) cpu.WriteRows(roomy, l, 0, a.KeysPerLayer[l], a.ValuesPerLayer[l]);
        using var self = new ResidentChunkKvEntry(Fingerprint, ChunkA, roomy, ChunkBackendTags.Cpu);
        var ex = Assert.Throws<ArgumentException>(() => assembler.Assemble(self.Cache, [rb, self], Fingerprint));
        Assert.Contains("own cache", ex.Message, StringComparison.Ordinal);
        Assert.Equal(ChunkA.Length, self.Cache.CurrentLength);

        // A source cache truncated below the chunk length.
        using IKvCache dest = cpu.CreateCache(SyntheticDenseGguf.Context);
        rb.Cache.Rollback(ChunkB.Length - 1);
        Assert.Throws<ArgumentException>(() => assembler.Assemble(dest, [ra, rb], Fingerprint));
        Assert.Equal(0, dest.CurrentLength);
    }

    /// <summary>Delegates to a real backend and throws <see cref="IOException"/> at the given row operation (WriteRows or CopyRows, 0-based).</summary>
    private sealed class FailingBackend(IChunkBackend inner, int failAtRowOp, RopeParameters? rope = null) : IChunkBackend
    {
        public int RowOps { get; private set; }
        public string Tag => inner.Tag;
        public int ForwardDeviceId => inner.ForwardDeviceId;
        public RopeParameters? Rope => rope;
        public IKvCache CreateCache(int maxSeqLen) => inner.CreateCache(maxSeqLen);
        public void RequireCache(IKvCache cache) => inner.RequireCache(cache);
        public void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v) => inner.ReadRows(cache, layer, startPos, count, k, v);
        public void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters r) => inner.ShiftKeys(cache, startPos, count, delta, r);

        public void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
        {
            Tick();
            inner.WriteRows(cache, layer, startPos, k, v);
        }

        public void CopyRows(IKvCache source, int layer, int srcStart, IKvCache destination, int dstStart, int count)
        {
            Tick();
            inner.CopyRows(source, layer, srcStart, destination, dstStart, count);
        }

        private void Tick()
        {
            if (RowOps++ == failAtRowOp) throw new IOException("injected failure (device lost, allocation)");
        }
    }

    internal static void AssertCachesEqual(IChunkBackend eb, IKvCache expected, IChunkBackend ab, IKvCache actual, int layers, int stride)
    {
        Assert.Equal(expected.CurrentLength, actual.CurrentLength);
        int n = expected.CurrentLength;
        float[] ek = new float[n * stride], ev = new float[n * stride], ak = new float[n * stride], av = new float[n * stride];
        for (int l = 0; l < layers; l++)
        {
            eb.ReadRows(expected, l, 0, n, ek, ev);
            ab.ReadRows(actual, l, 0, n, ak, av);
            Assert.True(ek.AsSpan().SequenceEqual(ak), $"layer {l} K differs");
            Assert.True(ev.AsSpan().SequenceEqual(av), $"layer {l} V differs");
        }
    }
}

/// <summary>
/// Phase G Task 3, GPU half: encode and assembly on the Vulkan backend with the synthetic F32 model. The Q4_K floors of G0 do not apply
/// here (G0 README "Scope"). Within-Vulkan comparisons of the same forward shapes are bit-exact; a device RoPE shift against the host
/// shift is within 1e-5 (Task 1 measured 6e-7); CPU vs Vulkan is within 1e-5 at layer 0 (no attention upstream). Deeper layers use
/// <see cref="CrossBackendDeepTol"/> = 1e-2, this task's own choice from its spike (CPU vs Vulkan measured 1.1e-3 / 1.3e-3 K / V at
/// layer 1 and 1.6e-3 / 2.1e-3 at layer 2, abs, on values of magnitude ~5): about 5x headroom, while a wrong layer or offset is O(1).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanAssemblyTestsGpu : IDisposable
{
    private const string Fingerprint = VulkanAssemblyTests.Fingerprint;
    private const float ShiftTol = 1e-5f;
    private const float CrossBackendDeepTol = 1e-2f;
    private static readonly int[] ChunkA = VulkanAssemblyTests.ChunkA;
    private static readonly int[] ChunkB = VulkanAssemblyTests.ChunkB;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkvk-gpu-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public VulkanAssemblyTestsGpu(ITestOutputHelper output)
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

    /// <summary>The synthetic model on the CPU and on Vulkan, with the Vulkan chunk backend on the model's device.</summary>
    private sealed class Rig : IDisposable
    {
        public Rig(string scratch, string spvDir)
        {
            Tiny = BackendSeamTests.TinyModel.Load(scratch, "qwen3");
            Device = VulkanDevice.Create();
            Model = VulkanTransformerModel.LoadFromGguf(Device, Tiny.Gguf, Tiny.Config, spvDir);
            Backend = new VulkanChunkBackend(Device, spvDir, Tiny.Config, ChunkModelFeatures.FromGguf(Tiny.Gguf));
            Cpu = new CpuChunkBackend(Tiny.Config);
        }

        public BackendSeamTests.TinyModel Tiny { get; }
        public VulkanDevice Device { get; }
        public IModel Model { get; }
        public VulkanChunkBackend Backend { get; }
        public CpuChunkBackend Cpu { get; }
        public int Layers => Tiny.Config.NumLayers;
        public int Stride => Tiny.KvStride;

        /// <summary>Plain Vulkan prefill of <paramref name="ids"/> at positions 0.. in one forward.</summary>
        public IKvCache Prefill(int[] ids)
        {
            IKvCache cache = Backend.CreateCache(SyntheticDenseGguf.Context);
            using ITensor _ = Model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), 0, cache, lastTokenLogitsOnly: true);
            return cache;
        }

        public (float[] K, float[] V) Read(IChunkBackend backend, IKvCache cache, int layer, int start, int count)
        {
            float[] k = new float[count * Stride], v = new float[count * Stride];
            backend.ReadRows(cache, layer, start, count, k, v);
            return (k, v);
        }

        public void Dispose()
        {
            Backend.Dispose();
            Model.Dispose();
            Device.Dispose();
            Tiny.Dispose();
        }
    }

    [SkippableFact]
    public void VulkanEncode_LayerZeroV_MatchesCpuWithinFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        var encoder = new ChunkEncoder(Fingerprint);
        Assert.Equal(0, rig.Backend.ForwardDeviceId);
        ChunkKvEntry vk = encoder.Encode(rig.Model, rig.Backend, ChunkA);
        ChunkKvEntry cpu = encoder.Encode(rig.Tiny.Model, ChunkA);
        Assert.Equal(ChunkBackendTags.Vulkan, vk.Backend);
        Assert.Equal(ChunkA, vk.TokenIds);
        Assert.Equal(Fingerprint, vk.ModelFingerprint);
        Assert.Equal(rig.Layers, vk.KeysPerLayer.Length);

        // Layer 0 has no attention upstream: CPU and Vulkan agree to F32 rounding.
        float k0 = BackendSeamTests.MaxAbsDiff(vk.KeysPerLayer[0], cpu.KeysPerLayer[0]);
        float v0 = BackendSeamTests.MaxAbsDiff(vk.ValuesPerLayer[0], cpu.ValuesPerLayer[0]);
        _output.WriteLine($"layer 0 CPU vs Vulkan: K {k0:E3} V {v0:E3}");
        Assert.True(k0 <= ShiftTol && v0 <= ShiftTol, $"layer 0 K {k0} V {v0}");

        // Every layer: the entry holds that layer's rows of the plain Vulkan prefill (bit-exact: same forward), and is close to the
        // CPU encode of the same layer (layers differ from each other by O(1), so a layer mix-up fails both).
        using IKvCache plain = rig.Prefill(ChunkA);
        for (int l = 0; l < rig.Layers; l++)
        {
            (float[] k, float[] v) = rig.Read(rig.Backend, plain, l, 0, ChunkA.Length);
            Assert.Equal(k, vk.KeysPerLayer[l]);
            Assert.Equal(v, vk.ValuesPerLayer[l]);
            float dk = BackendSeamTests.MaxAbsDiff(vk.KeysPerLayer[l], cpu.KeysPerLayer[l]);
            float dv = BackendSeamTests.MaxAbsDiff(vk.ValuesPerLayer[l], cpu.ValuesPerLayer[l]);
            _output.WriteLine($"layer {l} CPU vs Vulkan: K {dk:E3} V {dv:E3}");
            Assert.True(dk <= CrossBackendDeepTol && dv <= CrossBackendDeepTol, $"layer {l} K {dk} V {dv}");
        }
    }

    [SkippableFact]
    public void VulkanAssemble_SingleChunkAtPositionZero_EqualsFullPrefillWithinFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        ChunkKvEntry a = new ChunkEncoder(Fingerprint).Encode(rig.Model, rig.Backend, ChunkA);
        var assembler = new ChunkAssembler(rig.Tiny.Config, rig.Backend);
        using IKvCache assembled = rig.Backend.CreateCache(SyntheticDenseGguf.Context);
        assembler.Assemble(assembled, [a], Fingerprint);
        using IKvCache full = rig.Prefill(ChunkA);
        // Same tokens, same positions, same forward shape: the matching floor is the rerun pair (c), which is 0.
        VulkanAssemblyTests.AssertCachesEqual(rig.Backend, full, rig.Backend, assembled, rig.Layers, rig.Stride);

        // The next-token logits over both caches are identical too.
        float[] Next(IKvCache c)
        {
            using ITensor t = rig.Model.Forward([7], [ChunkA.Length], 0, c, lastTokenLogitsOnly: true);
            return Last(t);
        }
        Assert.Equal(Next(full), Next(assembled));
    }

    [SkippableFact]
    public void VulkanAssemble_ShiftedChunk_KeysMatchFullPrefillShiftedWithinFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry a = encoder.Encode(rig.Model, rig.Backend, ChunkA), b = encoder.Encode(rig.Model, rig.Backend, ChunkB);
        var assembler = new ChunkAssembler(rig.Tiny.Config, rig.Backend);
        using IKvCache assembled = rig.Backend.CreateCache(SyntheticDenseGguf.Context);
        assembler.Assemble(assembled, [b, a], Fingerprint);
        int m = ChunkB.Length, n = ChunkA.Length;
        Assert.Equal(m + n, assembled.CurrentLength);
        RopeParameters rope = RopeParameters.FromConfig(rig.Tiny.Config);

        // Independent oracle at layer 0 (K and V depend only on the token and its RoPE position there): the plain Vulkan prefill of
        // [B, A] has A's layer-0 K at positions m.. computed directly by the model.
        using IKvCache full = rig.Prefill([.. ChunkB, .. ChunkA]);
        (float[] fk, float[] fv) = rig.Read(rig.Backend, full, 0, m, n);
        (float[] ak, float[] av) = rig.Read(rig.Backend, assembled, 0, m, n);
        float dk0 = BackendSeamTests.MaxAbsDiff(fk, ak), dv0 = BackendSeamTests.MaxAbsDiff(fv, av);
        _output.WriteLine($"layer 0 assembled vs full prefill at offset {m}: K {dk0:E3} V {dv0:E3}");
        Assert.True(dk0 <= ShiftTol && dv0 <= ShiftTol, $"layer 0 K {dk0} V {dv0}");
        Assert.True(BackendSeamTests.MaxAbsDiff(fk, a.KeysPerLayer[0]) > 1e-2f); // guard: the offset matters for K

        // Every layer: chunk B (offset 0) is the entry verbatim; chunk A's V is verbatim and its K is the entry's K rotated by m.
        float worst = 0;
        for (int l = 0; l < rig.Layers; l++)
        {
            (float[] bk, float[] bv) = rig.Read(rig.Backend, assembled, l, 0, m);
            Assert.Equal(b.KeysPerLayer[l], bk);
            Assert.Equal(b.ValuesPerLayer[l], bv);
            (float[] k, float[] v) = rig.Read(rig.Backend, assembled, l, m, n);
            Assert.Equal(a.ValuesPerLayer[l], v);
            float[] expected = (float[])a.KeysPerLayer[l].Clone();
            RopeShift.Apply(expected, rig.Tiny.Config.HeadDim, m, in rope);
            worst = MathF.Max(worst, BackendSeamTests.MaxAbsDiff(expected, k));
        }
        _output.WriteLine($"all layers: assembled K vs host-shifted entry K max abs {worst:E3}");
        Assert.True(worst <= ShiftTol, $"shifted K differs by {worst}");

        // Cross-check against the CPU path (CPU encode + CPU assembly) on every layer.
        var cpuEnc = new ChunkEncoder(Fingerprint);
        ChunkKvEntry ca = cpuEnc.Encode(rig.Tiny.Model, ChunkA), cb = cpuEnc.Encode(rig.Tiny.Model, ChunkB);
        using IKvCache cpu = rig.Cpu.CreateCache(SyntheticDenseGguf.Context);
        new ChunkAssembler(rig.Tiny.Config, rig.Cpu).Assemble(cpu, [cb, ca], Fingerprint);
        for (int l = 0; l < rig.Layers; l++)
        {
            (float[] ck, float[] cv) = rig.Read(rig.Cpu, cpu, l, 0, m + n);
            (float[] vk, float[] vv) = rig.Read(rig.Backend, assembled, l, 0, m + n);
            float dk = BackendSeamTests.MaxAbsDiff(ck, vk), dv = BackendSeamTests.MaxAbsDiff(cv, vv);
            Assert.True(dk <= CrossBackendDeepTol && dv <= CrossBackendDeepTol, $"layer {l} CPU vs Vulkan assembly K {dk} V {dv}");
        }
    }

    [SkippableFact]
    public void VulkanAssemble_RejectsCpuTaggedEntry()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry cpuA = encoder.Encode(rig.Tiny.Model, ChunkA);
        ChunkKvEntry vkB = encoder.Encode(rig.Model, rig.Backend, ChunkB);
        var assembler = new ChunkAssembler(rig.Tiny.Config, rig.Backend);
        using IKvCache cache = rig.Backend.CreateCache(SyntheticDenseGguf.Context);
        var ex = Assert.Throws<ArgumentException>(() => assembler.Assemble(cache, [vkB, cpuA], Fingerprint));
        Assert.Contains("backend", ex.Message, StringComparison.Ordinal);
        Assert.Equal(0, cache.CurrentLength); // validated before anything is written

        // Resident entries: a CPU-resident entry is refused the same way, and a Vulkan-resident one is not accepted by the CPU assembler.
        using ResidentChunkKvEntry cpuRes = encoder.EncodeResident(rig.Tiny.Model, rig.Cpu, ChunkA);
        using ResidentChunkKvEntry vkRes = encoder.EncodeResident(rig.Model, rig.Backend, ChunkB);
        Assert.Throws<ArgumentException>(() => assembler.Assemble(cache, [vkRes, cpuRes], Fingerprint));
        Assert.Equal(0, cache.CurrentLength);
        using IKvCache host = rig.Cpu.CreateCache(SyntheticDenseGguf.Context);
        Assert.Throws<ArgumentException>(() => new ChunkAssembler(rig.Tiny.Config, rig.Cpu).Assemble(host, [vkRes], Fingerprint));
        Assert.Equal(0, host.CurrentLength);

        // A resident entry whose cache this backend cannot read (host rows under a Vulkan tag) is refused before the valid chunk ahead
        // of it is copied: the whole list is vetted first.
        IKvCache hostRows = rig.Cpu.CreateCache(ChunkA.Length);
        for (int l = 0; l < rig.Layers; l++) rig.Cpu.WriteRows(hostRows, l, 0, cpuA.KeysPerLayer[l], cpuA.ValuesPerLayer[l]);
        using var mislabelled = new ResidentChunkKvEntry(Fingerprint, ChunkA, hostRows, ChunkBackendTags.Vulkan);
        Assert.Throws<NotSupportedException>(() => assembler.Assemble(cache, [vkRes, mislabelled], Fingerprint));
        Assert.Equal(0, cache.CurrentLength);
    }

    [SkippableFact]
    public void DeviceCopyAndHostWrite_AssembleIdentically()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Rig(_scratch, spvDir);
        var encoder = new ChunkEncoder(Fingerprint);
        var assembler = new ChunkAssembler(rig.Tiny.Config, rig.Backend);
        ChunkKvEntry a = encoder.Encode(rig.Model, rig.Backend, ChunkA), b = encoder.Encode(rig.Model, rig.Backend, ChunkB);
        using ResidentChunkKvEntry ra = encoder.EncodeResident(rig.Model, rig.Backend, ChunkA);
        using ResidentChunkKvEntry rb = encoder.EncodeResident(rig.Model, rig.Backend, ChunkB);
        Assert.Equal(ChunkBackendTags.Vulkan, ra.Backend);
        Assert.IsType<VulkanKvCache>(ra.Cache);

        // Both paths after a 3-row prefix already in the cache, chunks repeated: pair (c), so bit-exact on every layer.
        using IKvCache host = rig.Prefill([9, 2, 4]);
        using IKvCache device = rig.Prefill([9, 2, 4]);
        assembler.Assemble(host, [b, a, b], Fingerprint);
        assembler.Assemble(device, [rb, ra, rb], Fingerprint);
        Assert.Equal(3 + 2 * ChunkB.Length + ChunkA.Length, device.CurrentLength);
        VulkanAssemblyTests.AssertCachesEqual(rig.Backend, host, rig.Backend, device, rig.Layers, rig.Stride);

        // The resident entries are unchanged by the assembly (the copy leaves the source rows alone).
        for (int l = 0; l < rig.Layers; l++)
        {
            (float[] k, float[] v) = rig.Read(rig.Backend, ra.Cache, l, 0, ChunkA.Length);
            Assert.Equal(a.KeysPerLayer[l], k);
            Assert.Equal(a.ValuesPerLayer[l], v);
        }
    }

    private static unsafe float[] Last(ITensor t)
    {
        int rows = t.Shape[0], vocab = t.Shape[1];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
    }
}
