using System.Buffers.Binary;
using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="ModelLoader.LoadFromSafetensors"/>
/// dispatches a <c>model_type=mamba3</c> safetensors checkpoint through
/// <see cref="Mamba3ConfigExtractor"/>, <see cref="Mamba3WeightLoader"/>,
/// and <see cref="Mamba3TransformerModel"/> into a working prefill forward
/// pass that returns finite <c>[seq_len, vocab_size]</c> logits.
/// </summary>
/// <remarks>
/// <para>
/// <b>Fixture source.</b> No tiny-random Mamba-3 checkpoint exists on the
/// HuggingFace Hub as of 2026-04-19 — only <c>ib-ssm/mamba3-370M-10BT</c>
/// (1.55 GB) and a handful of random-weight Mamba-1 / Mamba-2 derivatives.
/// Rather than pull the 370M checkpoint into every CI run, this test
/// synthesizes a deterministic miniature Mamba-3 checkpoint on disk:
/// a <c>config.json</c> matching the ib-ssm schema shape (<c>model_type</c>,
/// hidden_size, n_groups, rope_fraction, …) at tiny dimensions, plus a
/// safetensors file with all 3 global and 9 per-layer tensors populated
/// with small mean-zero values.
/// </para>
/// <para>
/// Dimensions: <c>hidden=8, vocab=16, layers=2, heads=4, head_dim=4,
/// state=8, rope_fraction=0.5, tie_word_embeddings=false</c>. Same shape
/// tuple as the unit-level <c>Mamba3TransformerModelTests</c>, mirrored
/// here so the end-to-end dispatch through <see cref="ModelLoader"/> is
/// exercised on disk rather than via the programmatic in-memory path.
/// </para>
/// <para>
/// <b>What's NOT covered here.</b> The real 1.55 GB <c>ib-ssm/mamba3-370M-10BT</c>
/// checkpoint is not downloaded — validating it end-to-end requires user
/// approval (bandwidth + disk) and is tracked as a separate follow-up. The
/// canonical-drift comparators in <c>Mamba3CanonicalReferenceCompareTests</c>
/// already prove the block math against a Python reference fixture.
/// </para>
/// </remarks>
public sealed class TinyMamba3SafetensorsLoadTests : IDisposable
{
    // Tiny config — matches the shape tuple in Mamba3TransformerModelTests
    // so the synthetic fixture exercises the same end-to-end math.
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;           // d_inner = 16
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize;     // num_bc_heads=1, is_mimo=false
    private const int NumRopeAngles = 2;     // state * 0.5 / 2
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles; // 62

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public TinyMamba3SafetensorsLoadTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-tiny-mamba3-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void LoadAndForwardPass_ProducesFiniteVocabLogits()
    {
        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMamba3Checkpoint(modelPath, configPath);

        _output.WriteLine($"Synthesised tiny Mamba-3 checkpoint at: {modelPath}");
        _output.WriteLine($"Fixture size: {new FileInfo(modelPath).Length} bytes");

        var (model, file, config) = ModelLoader.LoadFromSafetensors(modelPath);
        try
        {
            _output.WriteLine(
                $"Config: arch={config.Architecture} vocab={config.VocabSize} hidden={config.HiddenSize} "
              + $"layers={config.NumLayers} heads={config.Mamba3Config!.NumHeads} "
              + $"head_dim={config.Mamba3Config.HeadDim} d_state={config.Mamba3Config.StateSize} "
              + $"d_in_proj={config.Mamba3Config.InputProjectionDim} "
              + $"rope_fraction={config.Mamba3Config.RopeFraction} "
              + $"tied={config.TiedEmbeddings}");

            Assert.Equal(Architecture.Mamba3, config.Architecture);
            Assert.IsType<Mamba3TransformerModel>(model);
            Assert.Equal(DInProj, config.Mamba3Config.InputProjectionDim);
            Assert.Equal(NumRopeAngles, config.Mamba3Config.NumRopeAngles);

            int[] tokenIds = [0, 1, 2, 3];
            int[] positions = [0, 1, 2, 3];

            var sw = Stopwatch.StartNew();
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            sw.Stop();

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);

            var stats = ComputeStats(logits);
            _output.WriteLine(
                $"Forward: shape=[{logits.Shape[0]}, {logits.Shape[1]}] "
              + $"finite={stats.FiniteCount}/{stats.TotalCount} "
              + $"min={stats.Min:G4} max={stats.Max:G4} mean={stats.Mean:G4} stddev={stats.StdDev:G4} "
              + $"in {sw.Elapsed.TotalMilliseconds:F1} ms");

            Assert.Equal(stats.TotalCount, stats.FiniteCount);
            Assert.True(stats.StdDev > 0, "Logits have zero variance — forward pass likely degenerate.");
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    [Fact]
    public void LoadAndForwardPass_TiedEmbeddings_NoLmHeadTensor()
    {
        string modelPath = Path.Combine(_scratch, "tied-model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMamba3Checkpoint(modelPath, configPath, tieEmbeddings: true);

        var (model, file, config) = ModelLoader.LoadFromSafetensors(modelPath);
        try
        {
            Assert.Equal(Architecture.Mamba3, config.Architecture);
            Assert.True(config.TiedEmbeddings);

            using ITensor logits = model.Forward([0, 1, 2], [0, 1, 2], deviceId: -1);
            Assert.Equal(VocabSize, logits.Shape[1]);
            var stats = ComputeStats(logits);
            Assert.Equal(stats.TotalCount, stats.FiniteCount);
            Assert.True(stats.StdDev > 0);
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    // ------------------------------------------------------------------
    // Fixture synthesis
    // ------------------------------------------------------------------

    /// <summary>
    /// Writes a deterministic tiny Mamba-3 checkpoint as a pair of
    /// HuggingFace-convention files: <c>model.safetensors</c> (all 3 global
    /// and 9 per-layer tensors) plus <c>config.json</c> carrying
    /// <c>model_type=mamba3</c> and every field
    /// <see cref="Mamba3ConfigExtractor"/> reads.
    /// </summary>
    private static void WriteSyntheticMamba3Checkpoint(
        string safetensorsPath, string configPath, bool tieEmbeddings = false)
    {
        WriteConfigJson(configPath, tieEmbeddings);
        WriteSafetensorsFixture(safetensorsPath, includeLmHead: !tieEmbeddings);
    }

    private static void WriteConfigJson(string path, bool tieEmbeddings)
    {
        // Mirror the ib-ssm config.json schema, with the tiny dims.
        using var fs = File.Create(path);
        using var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteString("model_type", "mamba3");
        writer.WriteNumber("hidden_size", HiddenSize);
        writer.WriteNumber("vocab_size", VocabSize);
        writer.WriteNumber("num_hidden_layers", NumLayers);
        writer.WriteNumber("num_heads", NumHeads);
        writer.WriteNumber("head_dim", HeadDim);
        writer.WriteNumber("expand", Expand);
        writer.WriteNumber("n_groups", 1);
        writer.WriteNumber("state_size", StateSize);
        writer.WriteNumber("chunk_size", 2);
        writer.WriteNumber("mimo_rank", 1);
        writer.WriteBoolean("is_mimo", false);
        writer.WriteBoolean("is_outproj_norm", false);
        writer.WriteBoolean("use_l2warp", false);
        writer.WriteBoolean("tie_word_embeddings", tieEmbeddings);
        writer.WriteBoolean("rescale_prenorm_residual", true);
        writer.WriteBoolean("residual_in_fp32", true);
        writer.WriteNumber("A_floor", 1e-4);
        writer.WriteNumber("dt_init_floor", 1e-4);
        writer.WriteNumber("dt_min", 1e-3);
        writer.WriteNumber("dt_max", 0.1);
        writer.WriteNumber("norm_eps", 1e-5);
        writer.WriteNumber("rope_fraction", 0.5);
        writer.WriteNumber("max_position_embeddings", 32);
        writer.WriteEndObject();
    }

    private static void WriteSafetensorsFixture(string path, bool includeLmHead)
    {
        var tensors = new List<(string Name, int[] Shape, float[] Values)>();

        AddSmall(tensors, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize],
                 amplitude: 0.05f, seed: 0);
        AddSmall(tensors, Mamba3TensorMapping.FinalNorm, [HiddenSize],
                 amplitude: 0.5f, seed: 1);
        if (includeLmHead)
        {
            AddSmall(tensors, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize],
                     amplitude: 0.05f, seed: 2);
        }

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            AddSmall(tensors, Mamba3TensorMapping.LayerNorm(i), [HiddenSize],
                     amplitude: 0.5f, seed: sBase + 0);
            AddSmall(tensors, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize],
                     amplitude: 0.02f, seed: sBase + 1);
            AddSmall(tensors, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner],
                     amplitude: 0.05f, seed: sBase + 2);
            AddSmall(tensors, Mamba3TensorMapping.BNorm(i), [StateSize],
                     amplitude: 0.5f, seed: sBase + 3);
            AddSmall(tensors, Mamba3TensorMapping.CNorm(i), [StateSize],
                     amplitude: 0.5f, seed: sBase + 4);
            AddSmall(tensors, Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize],
                     amplitude: 0.02f, seed: sBase + 5);
            AddSmall(tensors, Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize],
                     amplitude: 0.02f, seed: sBase + 6);
            AddSmall(tensors, Mamba3TensorMapping.D(i), [NumHeads],
                     amplitude: 0.1f, seed: sBase + 7);
            AddSmall(tensors, Mamba3TensorMapping.DtBias(i), [NumHeads],
                     amplitude: 0.02f, seed: sBase + 8);
        }

        WriteSafetensorsFile(path, tensors);
    }

    /// <summary>
    /// Seeds one tensor in the output list with deterministic small-magnitude
    /// values in <c>[-amplitude, +amplitude]</c> (cosine-of-phi fill — same
    /// as the unit-test counterpart so results are reproducible across both
    /// test layers).
    /// </summary>
    private static void AddSmall(List<(string, int[], float[])> sink,
                                 string name, int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        sink.Add((name, shape, values));
    }

    /// <summary>
    /// Writes a safetensors binary in the minimal HuggingFace layout: LE u64
    /// header length, UTF-8 JSON header, raw row-major F32 data region.
    /// </summary>
    private static void WriteSafetensorsFile(string path, List<(string Name, int[] Shape, float[] Values)> tensors)
    {
        using var headerMs = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerMs, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var (name, shape, values) in tensors)
            {
                long byteLen = values.Length * sizeof(float);
                w.WriteStartObject(name);
                w.WriteString("dtype", "F32");
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (int d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + byteLen);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += byteLen;
            }
            w.WriteEndObject();
        }
        byte[] headerBytes = headerMs.ToArray();

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerBytes.Length);
        fs.Write(prefix);
        fs.Write(headerBytes);

        foreach (var (_, _, values) in tensors)
        {
            byte[] bytes = new byte[values.Length * sizeof(float)];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
            fs.Write(bytes);
        }
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

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
