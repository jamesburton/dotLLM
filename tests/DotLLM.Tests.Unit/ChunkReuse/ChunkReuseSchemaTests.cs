using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The persisted-KV schema version (<see cref="ChunkReuseSchema.KvOutputVersion"/>) and what the fingerprint/key hash. Stored
/// chunks are reused across builds, so a change to the encoded K/V must come with a version bump; the golden test below fails
/// when the K/V of the tiny model move and the version was not bumped.
/// </summary>
public sealed class ChunkReuseSchemaTests : IDisposable
{
    private static readonly int[] Chunk = [3, 17, 5, 29, 11, 40];

    /// <summary>
    /// Golden K/V summary per schema version: per layer (sum |K|, sum |V|, K[last row][0], V[last row][0]) of <see cref="Chunk"/>
    /// encoded on the synthetic qwen3 model (CPU, single-threaded). NEVER edit an existing version's row: when the encoded
    /// K/V change, bump <see cref="ChunkReuseSchema.KvOutputVersion"/> and add a row for the new version.
    /// </summary>
    private static readonly Dictionary<int, double[][]> Golden = new()
    {
        [1] =
        [
            // Recorded 2026-11-01 on Zen 5 (AVX-512), dotLLM feature/chunk-store @ 69646db5 + this change.
            [170.42277511488646, 225.22551880776882, -1.1586086750030518, -0.4841651916503906],
            [166.02996981237084, 204.78679384291172, 1.83122980594635, -0.8168670535087585],
            [136.0074493350694, 274.81851121783257, 0.3369029760360718, 2.3088510036468506],
        ],
    };

    // Relative tolerance: CPU SIMD paths differ by ISA (Vector512 vs Vector256, FMA) at ~1e-6 relative; a real kernel or
    // encode change moves these sums by far more. Sub-tolerance bit changes are not detected (see the report's concerns).
    private const double RelTol = 1e-4;
    private const double AbsTol = 1e-5;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkschema-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public ChunkReuseSchemaTests(ITestOutputHelper output)
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

    [Fact]
    public void Encode_TinyModel_MatchesGoldenForCurrentKvOutputVersion()
    {
        using var tiny = TinyModel.Load(_scratch);
        ChunkKvEntry entry = new ChunkEncoder("fp").Encode(tiny.Model, Chunk);
        int stride = tiny.Config.NumKvHeads * tiny.Config.HeadDim;
        var actual = new double[entry.KeysPerLayer.Length][];
        for (int l = 0; l < actual.Length; l++)
        {
            float[] k = entry.KeysPerLayer[l];
            float[] v = entry.ValuesPerLayer[l];
            actual[l] = [k.Sum(x => Math.Abs((double)x)), v.Sum(x => Math.Abs((double)x)), k[(Chunk.Length - 1) * stride], v[(Chunk.Length - 1) * stride]];
        }
        var ci = CultureInfo.InvariantCulture;
        _output.WriteLine($"KvOutputVersion {ChunkReuseSchema.KvOutputVersion}; actual:");
        foreach (double[] row in actual) _output.WriteLine("[" + string.Join(", ", row.Select(x => x.ToString("R", ci))) + "],");

        Assert.True(Golden.TryGetValue(ChunkReuseSchema.KvOutputVersion, out double[][]? golden),
            $"No golden for KvOutputVersion {ChunkReuseSchema.KvOutputVersion}: add one (values printed above).");
        Assert.Equal(golden!.Length, actual.Length);
        for (int l = 0; l < actual.Length; l++)
        {
            for (int i = 0; i < 4; i++)
            {
                double g = golden[l][i], a = actual[l][i];
                Assert.True(Math.Abs(a - g) <= AbsTol + RelTol * Math.Abs(g),
                    $"layer {l} field {i}: {a.ToString("R", ci)} vs golden {g.ToString("R", ci)}. The encoded K/V changed: bump "
                    + "ChunkReuseSchema.KvOutputVersion and add a golden row for the new version (do not edit the old row).");
            }
        }
    }

    [Fact]
    public void Fingerprint_DependsOnKvOutputVersion()
    {
        using var tiny = TinyModel.Load(_scratch);
        string fp = ChunkFingerprint.Compute(tiny.Config, "m", "F32", "t");
        Assert.Equal(fp, ChunkFingerprint.Compute(tiny.Config, "m", "F32", "t", ChunkReuseSchema.KvOutputVersion));
        Assert.NotEqual(fp, ChunkFingerprint.Compute(tiny.Config, "m", "F32", "t", ChunkReuseSchema.KvOutputVersion + 1));
    }

    [Fact]
    public void Key_RopeScalingFieldsOnlyInFingerprint_StillDifferentKeys()
    {
        // RopeParameters carries only (dim, theta, pairing). Scaling fields reach the key through the model fingerprint, so
        // two configs with equal RopeParameters but different scaling fields must give different fingerprints and keys.
        using var tiny = TinyModel.Load(_scratch);
        ModelConfig a = tiny.Config;
        ModelConfig b = a with { RoPEConfig = a.RoPEConfig!.Value with { BetaFast = 16f, OrigMaxSeqLen = 4096 } };
        RopeParameters ra = RopeParameters.FromConfig(a);
        RopeParameters rb = RopeParameters.FromConfig(b);
        Assert.Equal(ra, rb);

        string fa = ChunkFingerprint.Compute(a, "m", "F32", "t");
        string fb = ChunkFingerprint.Compute(b, "m", "F32", "t");
        Assert.NotEqual(fa, fb);
        Assert.NotEqual(ChunkKey.Create(fa, "t", ra, "cpu", KvDtype.F32, Chunk), ChunkKey.Create(fb, "t", rb, "cpu", KvDtype.F32, Chunk));
    }

    private sealed class TinyModel : IDisposable
    {
        private readonly GgufFile _gguf;

        private TinyModel(GgufFile gguf, ModelConfig config, TransformerModel model)
        {
            _gguf = gguf;
            Config = config;
            Model = model;
        }

        public ModelConfig Config { get; }

        public TransformerModel Model { get; }

        public static TinyModel Load(string scratch)
        {
            string path = Path.Combine(scratch, "qwen3-" + Guid.NewGuid().ToString("N") + ".gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build("qwen3"));
            var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            return new TinyModel(gguf, config, TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
        }

        public void Dispose()
        {
            Model.Dispose();
            _gguf.Dispose();
        }
    }
}
