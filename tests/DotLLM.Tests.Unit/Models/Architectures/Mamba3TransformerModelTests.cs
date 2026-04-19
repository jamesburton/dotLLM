using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Stage D3 unit tests: construct a <see cref="Mamba3TransformerModel"/> from
/// a synthetic safetensors fixture, run a prefill forward pass, and verify
/// shape, finiteness, and non-degenerate variance of the logits.
/// </summary>
/// <remarks>
/// <para>
/// The fixture builder writes tiny but architecturally-valid weight tensors
/// with deterministic ramp values. The combined forward pass exercises:
/// embedding gather → per-layer RMSNorm → Mamba-3 canonical SISO block →
/// residual → final RMSNorm → LM head GEMM. Ramp weights blow up fast under
/// the Mamba-3 recurrence if anything is miswired, so "finite + non-zero
/// variance" is a strong sanity check — it would fail at load-miswire,
/// shape-mismatch, pointer-aliasing, or any NaN/Inf source in the block.
/// </para>
/// </remarks>
public sealed class Mamba3TransformerModelTests : IDisposable
{
    private readonly string _scratch;

    // Tiny canonical shape tuple — matches Mamba3WeightLoaderTests so the
    // fixture builder can be shared without modification.
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;              // d_inner = 16
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize;        // SISO (is_mimo=false), num_bc_heads=1
    private const int NumRopeAngles = 2;        // state_size * 0.5 / 2 = 2
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles; // 62

    public Mamba3TransformerModelTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3m-{Guid.NewGuid():N}");
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

    /// <summary>
    /// Writes a fixture whose ramp weight values are tiny enough that the
    /// canonical SSD recurrence does not produce overflow. The default ramp
    /// in <see cref="SafetensorsFixtureBuilder.WriteTinyMamba3Fixture"/> spans
    /// 0.0..VocabSize*HiddenSize for the embedding and up to basis+8 for the
    /// per-layer tensors; those magnitudes push in_proj outputs to O(1000)
    /// which overflows the softplus in dd_A / dd_dt. We rebuild the fixture
    /// here with smaller, mean-zero values.
    /// </summary>
    private void WriteSmallWeightFixture(string path, bool includeLmHead = true)
    {
        var b = new SafetensorsFixtureBuilder();

        // Globals — centered small ramp around zero.
        SmallRamp(b, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize],
                  amplitude: 0.05f, seed: 0);
        SmallRamp(b, Mamba3TensorMapping.FinalNorm, [HiddenSize], amplitude: 0.5f, seed: 1);
        if (includeLmHead)
            SmallRamp(b, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize],
                      amplitude: 0.05f, seed: 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            SmallRamp(b, Mamba3TensorMapping.LayerNorm(i), [HiddenSize],
                      amplitude: 0.5f, seed: sBase + 0);
            SmallRamp(b, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize],
                      amplitude: 0.02f, seed: sBase + 1);
            SmallRamp(b, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner],
                      amplitude: 0.05f, seed: sBase + 2);
            SmallRamp(b, Mamba3TensorMapping.BNorm(i), [StateSize],
                      amplitude: 0.5f, seed: sBase + 3);
            SmallRamp(b, Mamba3TensorMapping.CNorm(i), [StateSize],
                      amplitude: 0.5f, seed: sBase + 4);
            SmallRamp(b, Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize],
                      amplitude: 0.02f, seed: sBase + 5);
            SmallRamp(b, Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize],
                      amplitude: 0.02f, seed: sBase + 6);
            SmallRamp(b, Mamba3TensorMapping.D(i), [NumHeads],
                      amplitude: 0.1f, seed: sBase + 7);
            SmallRamp(b, Mamba3TensorMapping.DtBias(i), [NumHeads],
                      amplitude: 0.02f, seed: sBase + 8);
        }

        b.WriteTo(path);
    }

    /// <summary>
    /// Fills a tensor with deterministic small-magnitude values in
    /// <c>[-amplitude, +amplitude]</c>. Uses a multiplier-and-cosine
    /// scheme for seed-dependent variation without requiring a PRNG.
    /// </summary>
    private static void SmallRamp(SafetensorsFixtureBuilder b, string name,
                                  int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            // Cheap deterministic fill: cos(i * phi + seed).
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        b.AddFloat32(name, shape, values);
    }

    [Fact]
    public void Forward_ProducesCorrectShapeAndFiniteLogits()
    {
        string path = Scratch("siso.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig config = BuildConfig();

        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, config);

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f,
            $"Logits have zero variance (std={stats.StdDev}) — forward pass is degenerate.");
    }

    [Fact]
    public void Forward_TiedEmbeddings_AliasesLmHead()
    {
        // Build a fixture WITHOUT an lm_head tensor; the loader must alias it
        // to the embedding when tie_word_embeddings=true, and the forward
        // pass must still produce finite logits.
        string path = Scratch("tied.safetensors");
        WriteSmallWeightFixture(path, includeLmHead: false);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig config = BuildConfig(tieEmbeddings: true);

        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, config);

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(VocabSize, logits.Shape[1]);
        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f);
    }

    [Fact]
    public void Forward_SingleToken_ProducesVocabSizeLogits()
    {
        string path = Scratch("single.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig config = BuildConfig();

        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, config);

        using ITensor logits = model.Forward([0], [0], deviceId: -1);

        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
    }

    [Fact]
    public void Forward_MismatchedPositionsLength_Throws()
    {
        string path = Scratch("mismatch.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, BuildConfig());

        Assert.Throws<ArgumentException>(() =>
            model.Forward([0, 1], [0], deviceId: -1));
    }

    [Fact]
    public void Forward_PositionOutOfRange_Throws()
    {
        string path = Scratch("oor.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        var cfg = BuildConfig();
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, cfg);

        int oor = cfg.MaxSequenceLength; // first invalid
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            model.Forward([0], [oor], deviceId: -1));
    }

    [Fact]
    public void Forward_TokenIdOutOfRange_Throws()
    {
        string path = Scratch("oor-tok.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, BuildConfig());

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            model.Forward([VocabSize], [0], deviceId: -1));
    }

    [Fact]
    public void LoadFromSafetensors_MissingRequired_Throws()
    {
        // Fixture missing final_norm — loader produces a structured Missing
        // report, which Mamba3TransformerModel.LoadFromSafetensors converts
        // to an InvalidDataException (fail-fast on incomplete checkpoints).
        string path = Scratch("missing.safetensors");
        var b = new SafetensorsFixtureBuilder()
            .AddFloat32(Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize])
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
        Assert.Throws<InvalidDataException>(() =>
            Mamba3TransformerModel.LoadFromSafetensors(sf, BuildConfig()));
    }

    [Fact]
    public void Dispose_ReleasesWeightsAndIsIdempotent()
    {
        string path = Scratch("dispose.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        var model = Mamba3TransformerModel.LoadFromSafetensors(sf, BuildConfig());
        model.Dispose();
        // Second dispose is a no-op.
        model.Dispose();
    }

    // --------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);

        int finite = 0;
        double sum = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in span)
        {
            if (float.IsFinite(v))
            {
                finite++;
                sum += v;
                sumSq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        double mean = finite > 0 ? sum / finite : 0.0;
        double variance = finite > 0 ? (sumSq / finite) - (mean * mean) : 0.0;
        double stddev = Math.Sqrt(Math.Max(0.0, variance));
        return new LogitStats(total, finite, (float)mean, (float)stddev, min, max);
    }

    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max);
}
