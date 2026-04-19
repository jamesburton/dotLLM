using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Stage D2 unit tests: end-to-end cover of
/// <see cref="Mamba3WeightLoader.Load"/> against synthetic safetensors
/// fixtures built via <see cref="SafetensorsFixtureBuilder"/>.
///
/// Dimensions are intentionally tiny (d_model=8, 2 layers, …) so every
/// per-layer + global tensor fits in a few KB, but the per-layer
/// signature and shape algebra still exercises every one of the nine
/// mapping entries. No 1.55 GB HF checkpoint is touched.
/// </summary>
public sealed class Mamba3WeightLoaderTests : IDisposable
{
    private readonly string _scratch;

    // Tiny canonical shape tuple reused across tests.
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;   // d_inner = 16 -> expand = 2
    private const int Expand = 2;
    private const int StateSize = 8; // even, pairs for RoPE
    private const int DInner = NumHeads * HeadDim;              // 16
    private const int BcDim = StateSize;                        // SISO (is_mimo=false)
    private const int DInProj = 2 * DInner + 2 * BcDim + 2 * NumHeads + StateSize / 2; // 60

    public Mamba3WeightLoaderTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-wl-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Scratch(string name) => Path.Combine(_scratch, name);

    private static ModelConfig BuildConfig(bool tieEmbeddings = false)
    {
        var m3 = new Mamba3Config
        {
            StateSize = StateSize,
            NumHeads = NumHeads,
            HeadDim = HeadDim,
            Expand = Expand,
            NumGroups = 1,
            ChunkSize = 2,
            IsMimo = false,
            MimoRank = 4,
            AFloor = 1e-4f,
            DtInitFloor = 1e-4f,
            DtMin = 1e-3f,
            DtMax = 0.1f,
            UseL2Warp = false,
            RopeFraction = 0.5f,
            IsOutProjNorm = false,
            RescalePrenormResidual = true,
            ResidualInFp32 = true,
        };
        return new ModelConfig
        {
            Architecture = Architecture.Mamba3,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = 0,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 32,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None,
            RoPEConfig = null,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = tieEmbeddings,
            SlidingWindowSize = null,
            MlaConfig = null,
            HybridLayout = null,
            SsmConfig = null,
            Mamba3Config = m3,
            ChatTemplate = null,
        };
    }

    [Fact]
    public void Load_AllTensorsPresent_PopulatesEveryHandle()
    {
        string path = Scratch("all-present.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: true, includeALog: false);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        Assert.True(w.TokenEmbedding.IsPopulated);
        Assert.True(w.FinalNorm.IsPopulated);
        Assert.True(w.LmHead.IsPopulated);
        Assert.Equal(NumLayers, w.Layers.Length);
        foreach (var layer in w.Layers)
        {
            Assert.True(layer.Norm.IsPopulated);
            Assert.True(layer.InProj.IsPopulated);
            Assert.True(layer.OutProj.IsPopulated);
            Assert.True(layer.BNorm.IsPopulated);
            Assert.True(layer.CNorm.IsPopulated);
            Assert.True(layer.BBias.IsPopulated);
            Assert.True(layer.CBias.IsPopulated);
            Assert.True(layer.D.IsPopulated);
            Assert.True(layer.DtBias.IsPopulated);
        }

        // Every required tensor loaded successfully except A_log (structural miss).
        // LoadedCount = 3 globals + 9 * numLayers per-layer = 21 for NumLayers=2.
        Assert.Equal(3 + 9 * NumLayers, w.Report.LoadedCount);
        Assert.True(w.Report.HasMissingRequired);
        Assert.Equal(1, w.Report.MissingRequiredCount); // A_log probe
    }

    [Fact]
    public void Load_ALog_Absent_EmitsMissingDiagnostic()
    {
        string path = Scratch("no-alog.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: true, includeALog: false);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        var miss = w.Report.Problems
            .Where(p => p.Kind == Mamba3TensorIssueKind.Missing &&
                        p.TensorName.EndsWith(".A_log", StringComparison.Ordinal))
            .ToArray();
        Assert.Single(miss);
        Assert.True(miss[0].IsRequired);
        Assert.Contains("A_log is absent", miss[0].Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_ALog_Present_SuppressesMissingDiagnostic()
    {
        // A_log lives on the reference but not the HF checkpoint — this test
        // proves the probe doesn't fire when the tensor *is* there.
        string path = Scratch("with-alog.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: true, includeALog: true);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        Assert.False(w.Report.HasMissingRequired);
        Assert.DoesNotContain(w.Report.Problems,
            p => p.TensorName.EndsWith(".A_log", StringComparison.Ordinal));
    }

    [Fact]
    public void Load_MissingTokenEmbedding_ReportedStructurally_NotThrown()
    {
        // Build a fixture that omits just the token-embedding tensor.
        string path = Scratch("no-emb.safetensors");
        var b = new SafetensorsFixtureBuilder()
            .AddFloat32(Mamba3TensorMapping.FinalNorm, [HiddenSize])
            .AddFloat32(Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize]);
        for (int i = 0; i < NumLayers; i++)
        {
            b.AddFloat32(Mamba3TensorMapping.LayerNorm(i), [HiddenSize]);
            b.AddFloat32(Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize]);
            b.AddFloat32(Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner]);
            b.AddFloat32(Mamba3TensorMapping.BNorm(i), [StateSize]);
            b.AddFloat32(Mamba3TensorMapping.CNorm(i), [StateSize]);
            b.AddFloat32(Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize]);
            b.AddFloat32(Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize]);
            b.AddFloat32(Mamba3TensorMapping.D(i), [NumHeads]);
            b.AddFloat32(Mamba3TensorMapping.DtBias(i), [NumHeads]);
        }
        b.WriteTo(path);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        Assert.False(w.TokenEmbedding.IsPopulated);
        Assert.True(w.FinalNorm.IsPopulated);
        Assert.True(w.Report.HasMissingRequired);
        Assert.Contains(w.Report.Problems,
            p => p.TensorName == Mamba3TensorMapping.TokenEmbedding &&
                 p.Kind == Mamba3TensorIssueKind.Missing);
    }

    [Fact]
    public void Load_ShapeMismatch_ReportedStructurally()
    {
        // in_proj with the WRONG output dim (DInProj+1 instead of DInProj).
        string path = Scratch("shape-mismatch.safetensors");
        var b = new SafetensorsFixtureBuilder()
            .AddFloat32(Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize])
            .AddFloat32(Mamba3TensorMapping.FinalNorm, [HiddenSize])
            .AddFloat32(Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize]);
        for (int i = 0; i < NumLayers; i++)
        {
            b.AddFloat32(Mamba3TensorMapping.LayerNorm(i), [HiddenSize]);
            b.AddFloat32(Mamba3TensorMapping.InProj(i), [DInProj + 1, HiddenSize]); // BAD
            b.AddFloat32(Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner]);
            b.AddFloat32(Mamba3TensorMapping.BNorm(i), [StateSize]);
            b.AddFloat32(Mamba3TensorMapping.CNorm(i), [StateSize]);
            b.AddFloat32(Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize]);
            b.AddFloat32(Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize]);
            b.AddFloat32(Mamba3TensorMapping.D(i), [NumHeads]);
            b.AddFloat32(Mamba3TensorMapping.DtBias(i), [NumHeads]);
        }
        b.WriteTo(path);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        var mismatch = w.Report.Problems
            .Where(p => p.Kind == Mamba3TensorIssueKind.ShapeMismatch)
            .ToArray();
        Assert.Equal(NumLayers, mismatch.Length); // one per layer
        Assert.All(mismatch, p =>
            Assert.Contains("in_proj.weight", p.TensorName, StringComparison.Ordinal));
    }

    [Fact]
    public void Load_UnsupportedDType_ReportedStructurally()
    {
        // Same layout but token_embedding stored as BF16 — Stage D2 must
        // surface this without throwing.
        string path = Scratch("bf16.safetensors");
        int embElems = VocabSize * HiddenSize;
        byte[] bf16Bytes = new byte[embElems * 2]; // zeroed bf16
        var b = new SafetensorsFixtureBuilder()
            .AddRaw(Mamba3TensorMapping.TokenEmbedding, "BF16", [VocabSize, HiddenSize], bf16Bytes)
            .AddFloat32(Mamba3TensorMapping.FinalNorm, [HiddenSize])
            .AddFloat32(Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize]);
        for (int i = 0; i < NumLayers; i++)
        {
            b.AddFloat32(Mamba3TensorMapping.LayerNorm(i), [HiddenSize]);
            b.AddFloat32(Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize]);
            b.AddFloat32(Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner]);
            b.AddFloat32(Mamba3TensorMapping.BNorm(i), [StateSize]);
            b.AddFloat32(Mamba3TensorMapping.CNorm(i), [StateSize]);
            b.AddFloat32(Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize]);
            b.AddFloat32(Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize]);
            b.AddFloat32(Mamba3TensorMapping.D(i), [NumHeads]);
            b.AddFloat32(Mamba3TensorMapping.DtBias(i), [NumHeads]);
        }
        b.WriteTo(path);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        Assert.False(w.TokenEmbedding.IsPopulated);
        Assert.Contains(w.Report.Problems,
            p => p.TensorName == Mamba3TensorMapping.TokenEmbedding &&
                 p.Kind == Mamba3TensorIssueKind.UnsupportedDType);
    }

    [Fact]
    public void Load_TiedEmbeddings_AliasesLmHead_WhenAbsent()
    {
        string path = Scratch("tied.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: false, includeALog: true);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(tieEmbeddings: true), sf);

        Assert.True(w.TokenEmbedding.IsPopulated);
        Assert.True(w.LmHead.IsPopulated);
        Assert.Equal(w.TokenEmbedding.Pointer, w.LmHead.Pointer);
        Assert.False(w.LmHead.OwnsMemory);
        // No required-tensor misses — A_log was also included in this fixture.
        Assert.False(w.Report.HasMissingRequired);
    }

    [Fact]
    public void Load_PointerAlignment_Is64ByteAligned()
    {
        // mmap provides page alignment (>= 4 KiB) so every tensor pointer
        // ends up 64-byte-aligned without a copy.
        string path = Scratch("alignment.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: true, includeALog: false);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        AssertAligned(w.TokenEmbedding.Pointer);
        AssertAligned(w.FinalNorm.Pointer);
        AssertAligned(w.LmHead.Pointer);
        foreach (var layer in w.Layers)
        {
            AssertAligned(layer.Norm.Pointer);
            AssertAligned(layer.InProj.Pointer);
            AssertAligned(layer.OutProj.Pointer);
            AssertAligned(layer.BNorm.Pointer);
            AssertAligned(layer.CNorm.Pointer);
            AssertAligned(layer.BBias.Pointer);
            AssertAligned(layer.CBias.Pointer);
            AssertAligned(layer.D.Pointer);
            AssertAligned(layer.DtBias.Pointer);
        }

        // Byte-count equivalence: for every handle, element_count * 4 must
        // equal the byte size derived from the descriptor.
        foreach (var layer in w.Layers)
            Assert.Equal((long)NumHeads * StateSize, layer.BBias.ElementCount);

        static void AssertAligned(nint ptr)
        {
            Assert.NotEqual(nint.Zero, ptr);
            Assert.Equal(0UL, ((ulong)ptr) & 63UL);
        }
    }

    [Fact]
    public unsafe void Load_TensorContent_MatchesFixtureRamp()
    {
        string path = Scratch("content.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: true, includeALog: false);

        using var sf = SafetensorsFile.Open(path);
        using var w = Mamba3WeightLoader.Load(BuildConfig(), sf);

        // Layer-0 norm uses startValue 10.0f + 0.0f = 10.0f, so elems are 10,11,..,17.
        var norm0 = new ReadOnlySpan<float>((void*)w.Layers[0].Norm.Pointer, HiddenSize);
        for (int i = 0; i < HiddenSize; i++)
            Assert.Equal(10.0f + i, norm0[i]);

        // Layer-1 D uses startValue 10.0f * 2 + 7.0f = 27.0f.
        var d1 = new ReadOnlySpan<float>((void*)w.Layers[1].D.Pointer, NumHeads);
        for (int i = 0; i < NumHeads; i++)
            Assert.Equal(27.0f + i, d1[i]);
    }

    [Fact]
    public void Dispose_DoesNotCrash_WhenCalledTwice()
    {
        string path = Scratch("dispose.safetensors");
        SafetensorsFixtureBuilder.WriteTinyMamba3Fixture(
            path, NumLayers, HiddenSize, VocabSize, NumHeads, HeadDim,
            StateSize, DInProj, DInner, includeLmHead: true, includeALog: false);

        using var sf = SafetensorsFile.Open(path);
        var w = Mamba3WeightLoader.Load(BuildConfig(), sf);
        w.Dispose();
        w.Dispose();
    }

    [Fact]
    public void Load_NullConfigOrFile_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Mamba3WeightLoader.Load(null!, null!));
    }

    [Fact]
    public void Load_NonMamba3Config_Throws()
    {
        var cfg = BuildConfig() with { Mamba3Config = null };
        Assert.Throws<ArgumentException>(() =>
        {
            using var sf = WriteEmptyFile(Scratch("empty.safetensors"));
            Mamba3WeightLoader.Load(cfg, sf);
        });
    }

    private static SafetensorsFile WriteEmptyFile(string path)
    {
        // Valid but empty safetensors file (header is {}).
        new SafetensorsFixtureBuilder().AddFloat32("keep", [1]).WriteTo(path);
        return SafetensorsFile.Open(path);
    }
}
