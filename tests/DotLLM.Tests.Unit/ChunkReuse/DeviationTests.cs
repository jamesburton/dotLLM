using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.ChunkReuse.Graph;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// E3 per-layer K/V deviation (Task 10): <see cref="KvDeviation"/> between a full-prefill cache and an assembled chunk cache,
/// token roles (<see cref="TokenRoles"/>) and the top-k% capture share (<see cref="TopKCapture"/>). Model-backed tests run the
/// synthetic dense <c>qwen3</c> model (<see cref="SyntheticDenseGguf"/>, F32, CPU); the arithmetic tests use planted values.
/// </summary>
public sealed class DeviationTests : IDisposable
{
    private const string Fingerprint = "test-fingerprint";

    private static readonly int[] ChunkA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] ChunkB = [8, 22, 1, 35];
    private static readonly int[] ChunkC = [12, 7, 44, 2, 19];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-deviation-" + Guid.NewGuid().ToString("N"));

    public DeviationTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { /* best effort: a still-mapped file on Windows */ }
        catch (UnauthorizedAccessException) { }
    }

    // ---- deviation on the model ------------------------------------------------------------------------------------------

    [Fact]
    public void Deviation_IdenticalIsolatedAndFullKv_IsZero()
    {
        // A one-chunk context: the isolated encoding IS the full prefill, so every layer, token, K and V must be exactly 0.
        using var tiny = TinyModel.Load(_scratch);
        using SimpleKvCache full = FullPrefill(tiny, ChunkA);
        using SimpleKvCache assembled = Assembled(tiny, Encode(tiny, ChunkA));

        KvDeviation d = KvDeviation.Compute(full, assembled, ChunkA.Length);

        Assert.Equal(tiny.Config.NumLayers, d.Layers);
        Assert.Equal(ChunkA.Length, d.Tokens);
        for (int l = 0; l < d.Layers; l++)
        for (int t = 0; t < d.Tokens; t++)
        {
            Assert.Equal(0f, d.KeyL2(l, t));
            Assert.Equal(0f, d.ValueL2(l, t));
            Assert.Equal(0f, d.KeyRelative(l, t));
            Assert.Equal(0f, d.ValueRelative(l, t));
            Assert.True(d.KeyReferenceNorm(l, t) > 0f, "fixture: reference K rows must be non-zero so the relative path is exercised");
        }
    }

    [Fact]
    public void Deviation_FirstChunk_IsZeroForAllTokens()
    {
        using var tiny = TinyModel.Load(_scratch);
        int[] context = [.. ChunkA, .. ChunkB, .. ChunkC];
        using SimpleKvCache full = FullPrefill(tiny, context);
        using SimpleKvCache assembled = Assembled(tiny, Encode(tiny, ChunkA, ChunkB, ChunkC));

        KvDeviation d = KvDeviation.Compute(full, assembled, context.Length);

        // The first chunk sees the same (empty) prefix in both encodings: exact.
        for (int l = 0; l < d.Layers; l++)
        for (int t = 0; t < ChunkA.Length; t++)
        {
            Assert.Equal(0f, d.KeyL2(l, t));
            Assert.Equal(0f, d.ValueL2(l, t));
        }

        // Discrimination: later chunks deviate from layer 1 on (they missed the earlier chunks), and layer 0 only by RoPE-shift
        // float error (layer-0 K/V depend on token and position alone).
        double later = 0;
        for (int l = 1; l < d.Layers; l++)
            for (int t = ChunkA.Length; t < context.Length; t++)
                later += d.KeyL2(l, t) + d.ValueL2(l, t);
        Assert.True(later > 1e-2, $"fixture cannot discriminate: later chunks deviate by only {later}");
        for (int t = ChunkA.Length; t < context.Length; t++)
        {
            Assert.Equal(0f, d.ValueL2(0, t));
            Assert.True(d.KeyRelative(0, t) < 1e-5f, $"layer 0 token {t}: K relative deviation {d.KeyRelative(0, t)}");
        }
    }

    // ---- deviation arithmetic on planted rows -----------------------------------------------------------------------------

    [Fact]
    public void Deviation_RelativeNormalisation_UsesTheFullPrefillNorm_KAndVSeparately()
    {
        using SimpleKvCache reference = PlantedCache(
            keys: [[3f, 4f, 0f, 0f], [0f, 0f, 6f, 8f]],
            values: [[1f, 0f, 0f, 0f], [0f, 2f, 0f, 0f]]);
        using SimpleKvCache reused = PlantedCache(
            keys: [[3f, 4f, 0f, 0f], [0f, 0f, 6f, 13f]],   // token 1: ΔK = (0,0,0,5)
            values: [[1f, 0f, 12f, 0f], [0f, 2f, 0f, 0f]]); // token 0: ΔV = (0,0,12,0)

        KvDeviation d = KvDeviation.Compute(reference, reused, 2);

        Assert.Equal(0f, d.KeyL2(0, 0));
        Assert.Equal(12f, d.ValueL2(0, 0), 5);
        Assert.Equal(5f, d.KeyL2(0, 1), 5);
        Assert.Equal(0f, d.ValueL2(0, 1));

        Assert.Equal(5f, d.KeyReferenceNorm(0, 0), 5);
        Assert.Equal(10f, d.KeyReferenceNorm(0, 1), 5);
        Assert.Equal(1f, d.ValueReferenceNorm(0, 0), 5);
        Assert.Equal(2f, d.ValueReferenceNorm(0, 1), 5);

        // Relative = deviation / FULL-PREFILL norm (not the reused norm: token 1's reused K norm is 14.32, not 10).
        Assert.Equal(0.5f, d.KeyRelative(0, 1), 5);
        Assert.Equal(12f, d.ValueRelative(0, 0), 5);
        Assert.Equal(0f, d.ValueRelative(0, 1));

        // Combined: ‖(ΔK, ΔV)‖ and ‖(ΔK, ΔV)‖ / ‖(K, V)_ref‖.
        Assert.Equal(12f, d.CombinedL2(0, 0), 5);
        Assert.Equal(5f, d.CombinedL2(0, 1), 5);
        Assert.Equal(12f / MathF.Sqrt(26f), d.CombinedRelative(0, 0), 5);
        Assert.Equal(5f / MathF.Sqrt(104f), d.CombinedRelative(0, 1), 5);
    }

    [Fact]
    public void Deviation_ZeroReferenceRow_RelativeIsZeroWhenEqual_InfinityWhenNot()
    {
        using SimpleKvCache reference = PlantedCache(keys: [[0f, 0f, 0f, 0f]], values: [[0f, 0f, 0f, 0f]]);
        using SimpleKvCache same = PlantedCache(keys: [[0f, 0f, 0f, 0f]], values: [[0f, 0f, 0f, 0f]]);
        using SimpleKvCache other = PlantedCache(keys: [[1f, 0f, 0f, 0f]], values: [[0f, 0f, 0f, 0f]]);

        Assert.Equal(0f, KvDeviation.Compute(reference, same, 1).KeyRelative(0, 0));
        Assert.Equal(float.PositiveInfinity, KvDeviation.Compute(reference, other, 1).KeyRelative(0, 0));
    }

    [Fact]
    public void Deviation_RowRangeBeyondEitherCache_Throws()
    {
        using SimpleKvCache a = PlantedCache(keys: [[1f, 0f, 0f, 0f]], values: [[0f, 0f, 0f, 0f]]);
        using SimpleKvCache b = PlantedCache(keys: [[1f, 0f, 0f, 0f], [1f, 0f, 0f, 0f]], values: [[0f, 0f, 0f, 0f], [0f, 0f, 0f, 0f]]);
        Assert.Throws<ArgumentException>(() => KvDeviation.Compute(a, b, 2));
    }

    // ---- roles -----------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(TokenRoleFlags.ChunkStart1 | TokenRoleFlags.Definition | TokenRoleFlags.Use | TokenRoleFlags.Import, TokenRole.ChunkStart)]
    [InlineData(TokenRoleFlags.Definition | TokenRoleFlags.Use | TokenRoleFlags.Import, TokenRole.DefinitionSite)]
    [InlineData(TokenRoleFlags.Use | TokenRoleFlags.Import, TokenRole.UseSite)]
    [InlineData(TokenRoleFlags.Import, TokenRole.Import)]
    [InlineData(TokenRoleFlags.None, TokenRole.Other)]
    // first-k (k = 4, 16) and the bookkeeping flags never decide the primary role
    [InlineData(TokenRoleFlags.ChunkStart4 | TokenRoleFlags.ChunkStart16 | TokenRoleFlags.Use, TokenRole.UseSite)]
    [InlineData(TokenRoleFlags.ChunkStart16 | TokenRoleFlags.CrossChunk | TokenRoleFlags.InContextCross | TokenRoleFlags.QueryTarget, TokenRole.Other)]
    // query rows have their own roles, whatever else is set
    [InlineData(TokenRoleFlags.Query | TokenRoleFlags.Use, TokenRole.QuerySite)]
    [InlineData(TokenRoleFlags.Query, TokenRole.Query)]
    public void Roles_PrimaryRole_FollowsPriority_ChunkStartOverDefinitionOverUseOverImport(TokenRoleFlags flags, TokenRole expected)
        => Assert.Equal(expected, TokenRoles.Primary(flags));

    [Fact]
    public void Roles_Label_SetsFirstKFlagsPerChunk_OrsMarks_AndMarksQueryRows()
    {
        // Chunks of 20, 3 and 5 tokens, then 2 query rows (L = 28, 30 rows).
        int[] starts = [0, 20, 23];
        var marks = new (int Token, TokenRoleFlags Flags)[]
        {
            (0, TokenRoleFlags.Definition),          // chunk start AND definition
            (5, TokenRoleFlags.Use),
            (5, TokenRoleFlags.Import | TokenRoleFlags.CrossChunk),
            (21, TokenRoleFlags.Definition),
            (29, TokenRoleFlags.Use),                // query row with an edge
        };

        TokenRoleFlags[] f = TokenRoles.Label(starts, contextLength: 28, queryLength: 2, marks);

        Assert.Equal(30, f.Length);
        Assert.Equal(TokenRoleFlags.ChunkStart1 | TokenRoleFlags.ChunkStart4 | TokenRoleFlags.ChunkStart16 | TokenRoleFlags.Definition, f[0]);
        Assert.Equal(TokenRoleFlags.ChunkStart4 | TokenRoleFlags.ChunkStart16, f[3]);
        Assert.Equal(TokenRoleFlags.ChunkStart16, f[4]);
        Assert.Equal(TokenRoleFlags.ChunkStart16 | TokenRoleFlags.Use | TokenRoleFlags.Import | TokenRoleFlags.CrossChunk, f[5]);
        Assert.Equal(TokenRoleFlags.ChunkStart16, f[15]);
        Assert.Equal(TokenRoleFlags.None, f[16]);
        Assert.Equal(TokenRoleFlags.None, f[19]);
        Assert.Equal(TokenRoleFlags.ChunkStart1 | TokenRoleFlags.ChunkStart4 | TokenRoleFlags.ChunkStart16, f[20]);
        Assert.Equal(TokenRoleFlags.ChunkStart4 | TokenRoleFlags.ChunkStart16 | TokenRoleFlags.Definition, f[21]);
        Assert.Equal(TokenRoleFlags.ChunkStart1 | TokenRoleFlags.ChunkStart4 | TokenRoleFlags.ChunkStart16, f[23]);
        Assert.Equal(TokenRoleFlags.ChunkStart16, f[27]);
        Assert.Equal(TokenRoleFlags.Query, f[28]);
        Assert.Equal(TokenRoleFlags.Query | TokenRoleFlags.Use, f[29]);

        Assert.Equal(TokenRole.ChunkStart, TokenRoles.Primary(f[0]));
        Assert.Equal(TokenRole.UseSite, TokenRoles.Primary(f[5]));
        Assert.Equal(TokenRole.DefinitionSite, TokenRoles.Primary(f[21]));
        Assert.Equal(TokenRole.Other, TokenRoles.Primary(f[16]));
        Assert.Equal(TokenRole.QuerySite, TokenRoles.Primary(f[29]));
    }

    [Theory]
    [InlineData(SymbolEdgeKind.Call, true, TokenRoleFlags.Use)]
    [InlineData(SymbolEdgeKind.Use, true, TokenRoleFlags.Use)]
    [InlineData(SymbolEdgeKind.Call, false, TokenRoleFlags.Definition)]
    [InlineData(SymbolEdgeKind.Use, false, TokenRoleFlags.Definition)]
    [InlineData(SymbolEdgeKind.Definition, true, TokenRoleFlags.Definition)]   // a later partial declaration is a declaration
    [InlineData(SymbolEdgeKind.Definition, false, TokenRoleFlags.Definition)]
    [InlineData(SymbolEdgeKind.Import, true, TokenRoleFlags.Import)]
    [InlineData(SymbolEdgeKind.Import, false, TokenRoleFlags.Import)]
    public void Roles_EdgeEndFlag_MapsKindAndEnd(SymbolEdgeKind kind, bool useEnd, TokenRoleFlags expected)
        => Assert.Equal(expected, TokenRoles.EdgeEndFlag(kind, useEnd));

    [Fact]
    public void Roles_Label_RejectsMarksOutsideTheRows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TokenRoles.Label([0], 4, 0, [(4, TokenRoleFlags.Use)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => TokenRoles.Label([0], 4, 0, [(-1, TokenRoleFlags.Use)]));
    }

    // ---- top-k capture ---------------------------------------------------------------------------------------------------

    [Fact]
    public void TopK_Capture_KnownValues()
    {
        // 10 tokens, total 100.
        float[] v = [1, 40, 2, 3, 20, 4, 5, 10, 6, 9];
        Assert.Equal(0.40, TopKCapture.Share(v, 0.10), 10);   // 1 token: 40
        Assert.Equal(0.60, TopKCapture.Share(v, 0.20), 10);   // 2 tokens: 40 + 20
        Assert.Equal(0.85, TopKCapture.Share(v, 0.50), 10);   // 5 tokens: 40 + 20 + 10 + 9 + 6 (not the bottom 5: 15)
        Assert.Equal(1.00, TopKCapture.Share(v, 1.00), 10);
        Assert.Equal(0.00, TopKCapture.Share(v, 0.00), 10);

        // Count = round(n * fraction) away from zero, at least 1 for a positive fraction.
        Assert.Equal(1, TopKCapture.Count(10, 0.05));   // 0.5 -> 1
        Assert.Equal(1, TopKCapture.Count(10, 0.01));   // 0.1 -> at least 1
        Assert.Equal(2, TopKCapture.Count(30, 0.05));   // 1.5 -> 2
        Assert.Equal(0, TopKCapture.Count(10, 0));
        Assert.Equal(0, TopKCapture.Count(0, 0.5));

        // Indices: highest first, ties by lower index.
        Assert.Equal([1, 4], TopKCapture.TopIndices(v, 2));
        Assert.Equal([0, 2], TopKCapture.TopIndices([5f, 1f, 5f, 5f], 2));

        // Share of an arbitrary set (random / position baselines).
        Assert.Equal(0.41, TopKCapture.ShareOf(v, [0, 1]), 10);
        Assert.Equal(0.0, TopKCapture.ShareOf(new float[3], [0]), 10); // all-zero row: 0, not NaN
    }

    [Fact]
    public void TopK_PositionOrder_IsOffsetInChunkThenChunk()
    {
        // Chunks [0..3), [3..5), [5..9) of a 9-token range: offset 0 of every chunk, then offset 1, ...
        int[] order = TopKCapture.PositionOrder([0, 3, 5], 9);
        Assert.Equal([0, 3, 5, 1, 4, 6, 2, 7, 8], order);
    }

    [Fact]
    public void TopK_RandomBaseline_ConvergesToTheFraction_AndIsSeeded()
    {
        float[] v = Enumerable.Range(1, 200).Select(i => (float)i).ToArray();
        double a = TopKCapture.RandomShare(v, 0.2, draws: 400, seed: 7);
        double b = TopKCapture.RandomShare(v, 0.2, draws: 400, seed: 7);
        Assert.Equal(a, b);
        Assert.InRange(a, 0.18, 0.22);
        Assert.True(TopKCapture.Share(v, 0.2) > 0.3);
    }

    // ---- helpers ---------------------------------------------------------------------------------------------------------

    /// <summary>A one-layer cache with KV stride 4 holding the given rows.</summary>
    private static unsafe SimpleKvCache PlantedCache(float[][] keys, float[][] values)
    {
        var geometry = KvGeometry.Uniform(numLayers: 1, numKvHeads: 1, headDim: 4);
        var cache = new SimpleKvCache(geometry, 8);
        for (int r = 0; r < keys.Length; r++)
        {
            fixed (float* k = keys[r])
            fixed (float* v = values[r])
            {
                cache.Update(new TensorRef(1, 4, DType.Float32, -1, (nint)k), new TensorRef(1, 4, DType.Float32, -1, (nint)v), [r], 0);
            }
        }
        return cache;
    }

    private static List<ChunkKvEntry> Encode(TinyModel tiny, params int[][] chunks)
    {
        var encoder = new ChunkEncoder(Fingerprint);
        return chunks.Select(c => encoder.Encode(tiny.Model, c)).ToList();
    }

    private static SimpleKvCache NewCache(TinyModel tiny)
        => new(KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false), SyntheticDenseGguf.Context);

    private static SimpleKvCache Assembled(TinyModel tiny, IReadOnlyList<ChunkKvEntry> chunks)
    {
        var cache = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(cache, chunks, Fingerprint);
        return cache;
    }

    private static SimpleKvCache FullPrefill(TinyModel tiny, int[] ids)
    {
        var cache = NewCache(tiny);
        int[] positions = Enumerable.Range(0, ids.Length).ToArray();
        using ITensor _ = tiny.Model.Forward(ids, positions, -1, cache, lastTokenLogitsOnly: true);
        return cache;
    }

    /// <summary>A loaded synthetic qwen3 model; keeps the mapped GGUF alive for the model's lifetime.</summary>
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
            string path = Path.Combine(scratch, "qwen3.gguf");
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
