using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The chunk-reuse backend seam (Phase G Task 1): every backend tags what it produces, the Vulkan backend refuses models whose
/// stored keys a plain RoPE delta shift cannot re-rotate, the assembler asks the backend to vet the destination before writing,
/// and the CPU backend reproduces the pre-seam assembly bit for bit. The Vulkan round-trip test is GPU-gated.
/// </summary>
public sealed class BackendSeamTests : IDisposable
{
    internal const string Fingerprint = "test-fingerprint";
    internal static readonly int[] ChunkA = [3, 17, 5, 29, 11, 40];
    internal static readonly int[] ChunkB = [8, 22, 1, 35];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkseam-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public BackendSeamTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { /* best effort: a still-mapped file on Windows */ }
        catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public void Backend_Tag_IsInKeyAndFingerprint()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        Assert.Equal("cpu", ChunkBackendTags.Cpu);
        Assert.Equal("vulkan", ChunkBackendTags.Vulkan); // same literals as ChunkKey.Create's backend on feature/chunk-store
        Assert.Equal(ChunkBackendTags.Cpu, new CpuChunkBackend(tiny.Config).Tag);

        // Fingerprint: the backend is an identity input; the legacy 4-argument overload means the CPU backend.
        string cpu = ChunkFingerprint.Compute(tiny.Config, "org/model", "F32", "tok", ChunkBackendTags.Cpu);
        string vk = ChunkFingerprint.Compute(tiny.Config, "org/model", "F32", "tok", ChunkBackendTags.Vulkan);
        Assert.NotEqual(cpu, vk);
        Assert.Equal(cpu, ChunkFingerprint.Compute(tiny.Config, "org/model", "F32", "tok"));

        // Entry: the encoder stamps its backend; the assembler treats a backend mismatch as a miss even when the fingerprint
        // string matches (an entry must never be assembled into another backend's cache without an explicit conversion).
        ChunkKvEntry a = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);
        Assert.Equal(ChunkBackendTags.Cpu, a.Backend);
        ChunkKvEntry foreign = a with { Backend = ChunkBackendTags.Vulkan };
        var assembler = new ChunkAssembler(tiny.Config, new CpuChunkBackend(tiny.Config));
        using var cache = NewCache(tiny);
        var ex = Assert.Throws<ArgumentException>(() => assembler.Assemble(cache, [a, foreign], Fingerprint));
        Assert.Contains("backend", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, cache.CurrentLength); // validated before anything is written
        assembler.Assemble(cache, [a], Fingerprint); // the same entry with the right tag is accepted
        Assert.Equal(ChunkA.Length, cache.CurrentLength);
    }

    [Fact]
    public void VulkanBackend_RefusesYarnRopeFreqFactorsAndMla()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        RoPEConfig plain = tiny.Config.RoPEConfig!.Value;
        var none = new ChunkModelFeatures(HasRopeFreqFactors: false, HasAttentionSinks: false);

        VulkanChunkBackend.CheckModelSupported(tiny.Config, none); // plain RoPE, no factors, no sinks: accepted

        var yarn = tiny.Config with { RoPEConfig = plain with { ScalingType = RoPEScalingType.YaRN, ScalingFactor = 4f, OrigMaxSeqLen = 16 } };
        Assert.Contains("YaRN", Refusal(yarn, none), StringComparison.Ordinal);

        var mla = tiny.Config with
        {
            MlaConfig = new MlaConfig { KvLoraRank = 8, QkNopeHeadDim = 8, QkRopeHeadDim = 8, VHeadDim = 16 },
        };
        Assert.Contains("MLA", Refusal(mla, none), StringComparison.Ordinal);

        Assert.Contains("rope_freqs", Refusal(tiny.Config, none with { HasRopeFreqFactors = true }), StringComparison.Ordinal);
        Assert.Contains("sink", Refusal(tiny.Config, none with { HasAttentionSinks = true }), StringComparison.OrdinalIgnoreCase);

        // The features come from the GGUF tensors, because ModelConfig carries neither.
        Assert.Equal(none, ChunkModelFeatures.FromGguf(tiny.Gguf));
        Assert.Equal(new ChunkModelFeatures(true, false), FeaturesOf("rope_freqs.weight"));
        Assert.Equal(new ChunkModelFeatures(false, true), FeaturesOf("blk.1.attn_sinks.weight"));

        static string Refusal(ModelConfig config, ChunkModelFeatures features)
            => Assert.Throws<NotSupportedException>(() => VulkanChunkBackend.CheckModelSupported(config, features)).Message;
    }

    [Fact]
    public void Assembler_RefusesCacheWithoutPositionedIo()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        using var host = NewCache(tiny); // SimpleKvCache: exact host F32, but no IPositionedHostKvCache

        // The Vulkan backend's destination gate, which needs no device.
        Assert.Contains(nameof(IPositionedHostKvCache),
            Assert.Throws<NotSupportedException>(() => VulkanChunkBackend.CheckCacheSupported(host)).Message, StringComparison.Ordinal);

        // The assembler asks the backend before it validates or writes anything.
        ChunkKvEntry a = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);
        var backend = new PositionedOnlyBackend();
        var assembler = new ChunkAssembler(tiny.Config, backend);
        Assert.Throws<NotSupportedException>(() => assembler.Assemble(host, [a with { Backend = backend.Tag }], Fingerprint));
        Assert.Equal(0, backend.Calls);
        Assert.Equal(0, host.CurrentLength);
    }

    [Fact]
    public void CpuBackend_RoundTrip_EqualsOldPath()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry a = encoder.Encode(tiny.Model, ChunkA);
        ChunkKvEntry b = encoder.Encode(tiny.Model, ChunkB);
        ChunkKvEntry[] order = [b, a, a];

        using var seam = NewCache(tiny);
        new ChunkAssembler(tiny.Config, new CpuChunkBackend(tiny.Config)).Assemble(seam, order, Fingerprint);
        using var legacyCtor = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(legacyCtor, order, Fingerprint);
        using var oldPath = NewCache(tiny);
        OldPathAssemble(tiny.Config, oldPath, order);

        int total = ChunkB.Length + 2 * ChunkA.Length;
        Assert.Equal(total, seam.CurrentLength);
        Assert.Equal(total, oldPath.CurrentLength);
        for (int l = 0; l < tiny.Config.NumLayers; l++)
        {
            Assert.True(seam.KeysSpan(l).SequenceEqual(oldPath.KeysSpan(l)), $"layer {l} keys differ from the pre-seam assembly");
            Assert.True(seam.ValuesSpan(l).SequenceEqual(oldPath.ValuesSpan(l)), $"layer {l} values differ from the pre-seam assembly");
            Assert.True(legacyCtor.KeysSpan(l).SequenceEqual(oldPath.KeysSpan(l)));
        }

        // Positioned row IO on the CPU backend: write at an offset, read back bit for bit, neighbours untouched.
        var cpu = new CpuChunkBackend(tiny.Config);
        using var io = (SimpleKvCache)cpu.CreateCache(SyntheticDenseGguf.Context);
        int stride = tiny.KvStride;
        for (int l = 0; l < tiny.Config.NumLayers; l++)
            cpu.WriteRows(io, l, 0, new float[10 * stride], new float[10 * stride]);
        cpu.WriteRows(io, 1, 4, a.KeysPerLayer[1], a.ValuesPerLayer[1]);
        var k = new float[ChunkA.Length * stride];
        var v = new float[ChunkA.Length * stride];
        cpu.ReadRows(io, 1, 4, ChunkA.Length, k, v);
        Assert.Equal(a.KeysPerLayer[1], k);
        Assert.Equal(a.ValuesPerLayer[1], v);
        Assert.True(io.KeysSpan(1)[..(4 * stride)].ToArray().All(x => x == 0f));
        Assert.True(io.KeysSpan(0).ToArray().All(x => x == 0f)); // other layer untouched
    }

    // ---- helpers ----

    /// <summary>The assembly algorithm before the backend seam (ChunkAssembler at b99570ba), kept here as the bit-exact reference.</summary>
    private static unsafe void OldPathAssemble(ModelConfig config, SimpleKvCache destination, IReadOnlyList<ChunkKvEntry> chunks)
    {
        RopeParameters rope = RopeParameters.FromConfig(config);
        int offset = destination.CurrentLength;
        foreach (ChunkKvEntry chunk in chunks)
        {
            int n = chunk.TokenIds.Length;
            int[] positions = Enumerable.Range(offset, n).ToArray();
            for (int l = 0; l < destination.NumLayers; l++)
            {
                int stride = ((IPerLayerKvCache)destination).KvStrideOf(l);
                float[] keys = chunk.KeysPerLayer[l];
                if (offset != 0 && !config.IsNoRopeLayer(l))
                {
                    keys = (float[])keys.Clone();
                    RopeShift.Apply(keys, config.HeadDim, offset, in rope);
                }
                fixed (float* kPtr = keys)
                fixed (float* vPtr = chunk.ValuesPerLayer[l])
                {
                    destination.Update(new TensorRef(n, stride, DType.Float32, -1, (nint)kPtr),
                        new TensorRef(n, stride, DType.Float32, -1, (nint)vPtr), positions, l);
                }
            }
            offset += n;
        }
    }

    private ChunkModelFeatures FeaturesOf(string tensorName)
    {
        var w = new GgufWriter();
        w.AddString("general.architecture", "qwen3");
        w.AddUInt32("general.alignment", 32);
        w.AddTensor(tensorName, [8], (uint)QuantizationType.F32, new byte[8 * sizeof(float)]);
        string path = Path.Combine(_scratch, Guid.NewGuid().ToString("N") + ".gguf");
        File.WriteAllBytes(path, w.Build());
        using var gguf = GgufFile.Open(path);
        return ChunkModelFeatures.FromGguf(gguf);
    }

    internal static SimpleKvCache NewCache(TinyModel tiny)
        => new(KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false), SyntheticDenseGguf.Context);

    internal static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    /// <summary>A backend that, like the Vulkan one, only takes caches with positioned row IO; counts every row operation.</summary>
    private sealed class PositionedOnlyBackend : IChunkBackend
    {
        public int Calls { get; private set; }
        public string Tag => "positioned-test";
        public IKvCache CreateCache(int maxSeqLen) => throw new NotSupportedException();
        public void RequireCache(IKvCache cache)
        {
            if (cache is not IPositionedHostKvCache)
                throw new NotSupportedException($"{cache.GetType().Name} has no {nameof(IPositionedHostKvCache)}.");
        }
        public void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v) => Calls++;
        public void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v) => Calls++;
        public void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters rope) => Calls++;
    }

    /// <summary>A loaded synthetic model; keeps the mapped GGUF alive for the model's lifetime.</summary>
    internal sealed class TinyModel : IDisposable
    {
        private TinyModel(GgufFile gguf, ModelConfig config, TransformerModel model)
        {
            Gguf = gguf;
            Config = config;
            Model = model;
        }

        public GgufFile Gguf { get; }
        public ModelConfig Config { get; }
        public TransformerModel Model { get; }
        public int KvStride => Config.NumKvHeads * Config.HeadDim;

        public static TinyModel Load(string scratch, string arch)
        {
            string path = Path.Combine(scratch, arch + ".gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build(arch));
            var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            return new TinyModel(gguf, config, TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
        }

        public void Dispose()
        {
            Model.Dispose();
            Gguf.Dispose();
        }
    }
}

/// <summary>GPU half of <see cref="BackendSeamTests"/>: the Vulkan backend's row IO, device shift and destination refusals.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class BackendSeamTestsGpu : IDisposable
{
    private const string Fingerprint = BackendSeamTests.Fingerprint;
    private static readonly int[] ChunkA = BackendSeamTests.ChunkA;
    private static readonly int[] ChunkB = BackendSeamTests.ChunkB;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkseam-gpu-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public BackendSeamTestsGpu(ITestOutputHelper output)
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

    [SkippableFact]
    public void VulkanBackend_WriteThenRead_RoundTrips()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        var features = ChunkModelFeatures.FromGguf(tiny.Gguf);
        ChunkKvEntry a = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);
        int stride = tiny.KvStride, n = ChunkA.Length, layers = tiny.Config.NumLayers;

        using var device = VulkanDevice.Create();
        // Construction itself refuses a model the device shift cannot re-rotate (before any cache or kernel use).
        Assert.Throws<NotSupportedException>(() => new VulkanChunkBackend(device, spvDir, tiny.Config, features with { HasRopeFreqFactors = true }));
        Assert.Throws<NotSupportedException>(() => new VulkanChunkBackend(device, spvDir, tiny.Config, features with { HasAttentionSinks = true }));
        using var backend = new VulkanChunkBackend(device, spvDir, tiny.Config, features);
        Assert.Equal(ChunkBackendTags.Vulkan, backend.Tag);
        using var cache = (VulkanKvCache)backend.CreateCache(SyntheticDenseGguf.Context);
        Assert.Equal(VulkanKvStorage.F32, cache.Storage);

        // A shift of rows that were never written is refused (write first, then shift).
        Assert.ThrowsAny<ArgumentException>(() => backend.ShiftKeys(cache, 4, n, 4, RopeParameters.FromConfig(tiny.Config)));

        // Write every layer at offset 4 (zero rows before it), read back: F32 storage, so exact.
        for (int l = 0; l < layers; l++)
        {
            backend.WriteRows(cache, l, 0, new float[4 * stride], new float[4 * stride]);
            backend.WriteRows(cache, l, 4, a.KeysPerLayer[l], a.ValuesPerLayer[l]);
        }
        Assert.Equal(4 + n, cache.CurrentLength);
        var k = new float[n * stride];
        var v = new float[n * stride];
        for (int l = 0; l < layers; l++)
        {
            backend.ReadRows(cache, l, 4, n, k, v);
            Assert.Equal(a.KeysPerLayer[l], k);
            Assert.Equal(a.ValuesPerLayer[l], v);
        }

        // Device shift by +4 of the rows at 4..4+n matches the CPU RopeShift of the same rows (F32 rotation, rounding only),
        // V untouched; and the assembler through this backend writes the same cache as the CPU assembler (to the same tolerance).
        backend.ShiftKeys(cache, 4, n, 4, RopeParameters.FromConfig(tiny.Config));
        float worst = 0;
        for (int l = 0; l < layers; l++)
        {
            float[] expected = (float[])a.KeysPerLayer[l].Clone();
            RopeShift.Apply(expected, tiny.Config.HeadDim, 4, RopeParameters.FromConfig(tiny.Config));
            backend.ReadRows(cache, l, 4, n, k, v);
            Assert.Equal(a.ValuesPerLayer[l], v);
            Assert.True(BackendSeamTests.MaxAbsDiff(a.KeysPerLayer[l], expected) > 1e-2f); // fixture guard: the shift is not a no-op
            worst = MathF.Max(worst, BackendSeamTests.MaxAbsDiff(k, expected));
        }
        _output.WriteLine($"vulkan shift vs CPU RopeShift: max abs {worst:E3}");
        Assert.True(worst <= 1e-5f, $"device shift differs from the CPU shift by {worst}");

        // Refusals that need a device: a host cache, and an F16 Vulkan cache (not an exact F32 destination).
        var assembler = new ChunkAssembler(tiny.Config, backend);
        ChunkKvEntry vkA = a with { Backend = ChunkBackendTags.Vulkan };
        using var host = BackendSeamTests.NewCache(tiny);
        Assert.Throws<NotSupportedException>(() => assembler.Assemble(host, [vkA], Fingerprint));
        using var f16 = new VulkanKvCache(device, KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false),
            SyntheticDenseGguf.Context, VulkanKvStorage.F16);
        Assert.Contains("F16", Assert.Throws<NotSupportedException>(() => assembler.Assemble(f16, [vkA], Fingerprint)).Message,
            StringComparison.Ordinal);
        Assert.Equal(0, f16.CurrentLength);

        ChunkKvEntry b = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkB);
        using var vkAsm = (VulkanKvCache)backend.CreateCache(SyntheticDenseGguf.Context);
        assembler.Assemble(vkAsm, [b with { Backend = ChunkBackendTags.Vulkan }, vkA], Fingerprint);
        using var cpuAsm = BackendSeamTests.NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(cpuAsm, [b, a], Fingerprint);
        int total = ChunkB.Length + n;
        Assert.Equal(total, vkAsm.CurrentLength);
        var vk = new float[total * stride];
        var vv = new float[total * stride];
        float worstAsm = 0;
        for (int l = 0; l < layers; l++)
        {
            vkAsm.ReadRows(l, 0, total, vk, vv);
            Assert.True(cpuAsm.ValuesSpan(l).SequenceEqual(vv));
            worstAsm = MathF.Max(worstAsm, BackendSeamTests.MaxAbsDiff(vk, cpuAsm.KeysSpan(l)));
        }
        _output.WriteLine($"vulkan assembly vs CPU assembly keys: max abs {worstAsm:E3}");
        Assert.True(worstAsm <= 1e-5f, $"Vulkan assembly differs from the CPU assembly by {worstAsm}");
    }
}
