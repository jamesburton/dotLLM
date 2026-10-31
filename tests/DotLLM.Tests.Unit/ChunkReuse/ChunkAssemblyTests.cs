using DotLLM.ChunkReuse;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Chunk encoder + RoPE-shifted assembly on a synthetic dense model (<see cref="SyntheticDenseGguf"/>, F32 weights, CPU).
/// <para>
/// The oracle for re-rotated keys is layer 0 of a full prefill: there, a token's K is <c>RoPE_p(KNorm(Wk · norm(embed(t))))</c>
/// and depends only on the token and its position, so a chunk encoded alone at <c>0..n-1</c> and shifted by <c>Δ</c> must equal the
/// keys the model itself writes for the same tokens at <c>Δ..Δ+n-1</c>. Above layer 0, K depends on the attended prefix, so
/// there the expected value is <c>RopeShift(isolated, Δ)</c> (the assembler's contract), not the full-prefill key.
/// </para>
/// </summary>
public sealed class ChunkAssemblyTests : IDisposable
{
    private const string Fingerprint = "test-fingerprint";

    // Chunk A, chunk B and a prefix used to place chunks at a non-zero offset. Distinct token ids so rows are distinguishable.
    private static readonly int[] ChunkA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] ChunkB = [8, 22, 1, 35];
    private static readonly int[] Prefix = [12, 7, 44, 2, 19, 31, 26];
    private static readonly int[] Suffix = [9, 14, 38];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunkreuse-" + Guid.NewGuid().ToString("N"));

    private readonly ITestOutputHelper _output;

    public ChunkAssemblyTests(ITestOutputHelper output)
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

    [Theory]
    [InlineData("qwen3", RoPEType.NeoX)]
    [InlineData("llama", RoPEType.Norm)]
    public void RopeShift_ShiftedKeys_EqualKeysComputedAtTargetPosition(string arch, RoPEType expectedPairing)
    {
        using var tiny = TinyModel.Load(_scratch, arch);
        // Both pairing layouts are exercised: a shift with the wrong layout fails one of the two cases.
        Assert.Equal(expectedPairing, tiny.Config.RoPEConfig!.Value.Type);

        int delta = Prefix.Length;
        ChunkKvEntry isolated = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);

        // Reference: the model's own layer-0 keys for ChunkA at positions delta..delta+n-1.
        float[] expected = FullPrefillRows(tiny, [.. Prefix, .. ChunkA], layer: 0, keys: true, from: delta, count: ChunkA.Length);

        float[] shifted = (float[])isolated.KeysPerLayer[0].Clone();
        RopeShift.Apply(shifted, tiny.Config.HeadDim, delta, RopeParameters.FromConfig(tiny.Config));

        // Fixture guard: the unshifted keys must be far from the reference, or a no-op shift would pass.
        Assert.True(MaxAbsDiff(isolated.KeysPerLayer[0], expected) > 1e-2f,
            $"fixture cannot discriminate: unshifted keys already within {MaxAbsDiff(isolated.KeysPerLayer[0], expected)} of the reference");
        // Fixture guard: shifting by delta-1 or delta+1 must also miss (off-by-one in the offset).
        float[] offByOne = (float[])isolated.KeysPerLayer[0].Clone();
        RopeShift.Apply(offByOne, tiny.Config.HeadDim, delta + 1, RopeParameters.FromConfig(tiny.Config));
        Assert.True(MaxAbsDiff(offByOne, expected) > 1e-2f);

        float err = MaxAbsDiff(shifted, expected);
        _output.WriteLine($"{arch}: max |shifted - model K at target| = {err:E3}; unshifted = {MaxAbsDiff(isolated.KeysPerLayer[0], expected):E3}");
        Assert.True(err <= 1e-5f, $"shifted keys differ from the model's keys at the target position by {err}");
    }

    [Fact]
    public void SingleChunkAtOffsetZero_MatchesFullPrefillLogits()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        ChunkKvEntry entry = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);

        // Reference: one prefill of [ChunkA; Suffix], last row.
        float[] reference;
        using (var cache = NewCache(tiny))
            reference = LastRow(tiny.Model, [.. ChunkA, .. Suffix], 0, cache);

        // Assembled: ChunkA's KV placed at 0..n-1, then only the suffix forwarded at n..n+2.
        float[] assembled;
        using (var cache = NewCache(tiny))
        {
            new ChunkAssembler(tiny.Config).Assemble(cache, [entry], Fingerprint);
            Assert.Equal(ChunkA.Length, cache.CurrentLength);
            assembled = LastRow(tiny.Model, Suffix, ChunkA.Length, cache);
        }

        // Fixture guard: the chunk context must matter for the suffix logits.
        float[] noContext;
        using (var cache = NewCache(tiny))
            noContext = LastRow(tiny.Model, Suffix, 0, cache);
        Assert.True(MaxAbsDiff(noContext, reference) > 1e-2f);

        float err = MaxAbsDiff(assembled, reference);
        _output.WriteLine($"max |assembled - full-prefill logits| = {err:E3}; no-context = {MaxAbsDiff(noContext, reference):E3}");
        Assert.True(err <= 1e-4f, $"assembled-cache logits differ from full prefill by {err}");
    }

    [Fact]
    public void Assemble_WrongFingerprint_Throws()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        ChunkKvEntry entry = new ChunkEncoder("model-a").Encode(tiny.Model, ChunkA);
        using var cache = NewCache(tiny);

        var ex = Assert.Throws<ArgumentException>(() => new ChunkAssembler(tiny.Config).Assemble(cache, [entry], "model-b"));
        Assert.Contains("fingerprint", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, cache.CurrentLength); // nothing written

        // A valid chunk before the bad one must not be written either (validation precedes every write).
        ChunkKvEntry good = new ChunkEncoder("model-b").Encode(tiny.Model, ChunkB);
        Assert.Throws<ArgumentException>(() => new ChunkAssembler(tiny.Config).Assemble(cache, [good, entry], "model-b"));
        Assert.Equal(0, cache.CurrentLength);
    }

    [Fact]
    public void Assemble_EmptyChunk_ThrowsArgumentException()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        var emptyLayers = Enumerable.Range(0, tiny.Config.NumLayers).Select(_ => Array.Empty<float>()).ToArray();
        var empty = new ChunkKvEntry(Fingerprint, [], emptyLayers, emptyLayers);
        using var cache = NewCache(tiny);

        Assert.Throws<ArgumentException>(() => new ChunkAssembler(tiny.Config).Assemble(cache, [empty], Fingerprint));
        Assert.Equal(0, cache.CurrentLength);
        // The encoder refuses an empty chunk too.
        Assert.Throws<ArgumentException>(() => new ChunkEncoder(Fingerprint).Encode(tiny.Model, ReadOnlySpan<int>.Empty));
    }

    [Fact]
    public void Assemble_SameChunkTwice_ProducesDistinctPositions()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        ChunkKvEntry entry = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);
        float[][] keysBefore = entry.KeysPerLayer.Select(k => (float[])k.Clone()).ToArray();
        int n = ChunkA.Length;
        int stride = tiny.KvStride;

        using var cache = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(cache, [entry, entry], Fingerprint);
        Assert.Equal(2 * n, cache.CurrentLength);

        // The entry is not mutated (a shared entry shifted in place would be double-shifted on reuse).
        for (int l = 0; l < keysBefore.Length; l++)
            Assert.Equal(keysBefore[l], entry.KeysPerLayer[l]);

        // Layer 0: copy 1 equals the model's own keys at 0..n-1, copy 2 its keys at n..2n-1 of a full [A; A] prefill.
        float[] firstCopy = cache.KeysSpan(0)[..(n * stride)].ToArray();
        float[] secondCopy = cache.KeysSpan(0)[(n * stride)..(2 * n * stride)].ToArray();
        float[] expectedSecond = FullPrefillRows(tiny, [.. ChunkA, .. ChunkA], layer: 0, keys: true, from: n, count: n);
        Assert.Equal(entry.KeysPerLayer[0], firstCopy);
        Assert.True(MaxAbsDiff(secondCopy, expectedSecond) <= 1e-5f);
        Assert.True(MaxAbsDiff(firstCopy, secondCopy) > 1e-2f, "the two copies hold the same keys: positions were not distinct");

        // Values are position-free: both copies carry the encoded values unchanged, on every layer.
        for (int l = 0; l < tiny.Config.NumLayers; l++)
        {
            ReadOnlySpan<float> v = cache.ValuesSpan(l);
            Assert.Equal(entry.ValuesPerLayer[l], v[..(n * stride)].ToArray());
            Assert.Equal(entry.ValuesPerLayer[l], v[(n * stride)..(2 * n * stride)].ToArray());
        }
    }

    [Fact]
    public void Assemble_ReorderedChunks_ShiftsByNewOffsets()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry a = encoder.Encode(tiny.Model, ChunkA);
        ChunkKvEntry b = encoder.Encode(tiny.Model, ChunkB);
        var rope = RopeParameters.FromConfig(tiny.Config);
        int stride = tiny.KvStride;

        // Order [B, A]: B sits at 0 (unshifted), A at |B|.
        using var cache = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(cache, [b, a], Fingerprint);
        Assert.Equal(ChunkA.Length + ChunkB.Length, cache.CurrentLength);

        int bRows = ChunkB.Length * stride, aRows = ChunkA.Length * stride;
        for (int l = 0; l < tiny.Config.NumLayers; l++)
        {
            ReadOnlySpan<float> k = cache.KeysSpan(l);
            Assert.Equal(b.KeysPerLayer[l], k[..bRows].ToArray());
            float[] aShifted = (float[])a.KeysPerLayer[l].Clone();
            RopeShift.Apply(aShifted, tiny.Config.HeadDim, ChunkB.Length, rope);
            Assert.True(MaxAbsDiff(k[bRows..(bRows + aRows)].ToArray(), aShifted) <= 1e-6f, $"layer {l}: A not shifted by its new offset");
        }

        // Layer 0 also matches the model's own keys for the reordered prompt [B; A].
        float[] expectedA = FullPrefillRows(tiny, [.. ChunkB, .. ChunkA], layer: 0, keys: true, from: ChunkB.Length, count: ChunkA.Length);
        Assert.True(MaxAbsDiff(cache.KeysSpan(0)[bRows..(bRows + aRows)].ToArray(), expectedA) <= 1e-5f);

        // The other order puts A at 0 and B at |A|: A's keys differ between the two layouts.
        using var other = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(other, [a, b], Fingerprint);
        Assert.Equal(a.KeysPerLayer[0], other.KeysSpan(0)[..aRows].ToArray());
        Assert.True(MaxAbsDiff(other.KeysSpan(0)[..aRows].ToArray(), cache.KeysSpan(0)[bRows..(bRows + aRows)].ToArray()) > 1e-2f);
    }

    [Fact]
    public void Assemble_IntoNonEmptyCache_AppendsAtCurrentLength()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        var encoder = new ChunkEncoder(Fingerprint);
        ChunkKvEntry a = encoder.Encode(tiny.Model, ChunkA);
        ChunkKvEntry b = encoder.Encode(tiny.Model, ChunkB);
        int stride = tiny.KvStride, aRows = ChunkA.Length * stride, bRows = ChunkB.Length * stride;

        // Two calls: A, then B. B must land after A (offset |A|), shifted by |A|, with A left intact.
        using var cache = NewCache(tiny);
        var assembler = new ChunkAssembler(tiny.Config);
        assembler.Assemble(cache, [a], Fingerprint);
        assembler.Assemble(cache, [b], Fingerprint);
        Assert.Equal(ChunkA.Length + ChunkB.Length, cache.CurrentLength);

        Assert.Equal(a.KeysPerLayer[0], cache.KeysSpan(0)[..aRows].ToArray());
        float[] expectedB = FullPrefillRows(tiny, [.. ChunkA, .. ChunkB], layer: 0, keys: true, from: ChunkA.Length, count: ChunkB.Length);
        Assert.True(MaxAbsDiff(cache.KeysSpan(0)[aRows..(aRows + bRows)].ToArray(), expectedB) <= 1e-5f);
    }

    [Fact]
    public void Assemble_QuantizedDestination_ThrowsNotSupported()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        ChunkKvEntry entry = new ChunkEncoder(Fingerprint).Encode(tiny.Model, ChunkA);
        var assembler = new ChunkAssembler(tiny.Config);
        var geometry = KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false);

        using (var q8 = new QuantizedKvCache(geometry, 32, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0))
        {
            Assert.Throws<NotSupportedException>(() => assembler.Assemble(q8, [entry], Fingerprint));
            Assert.Equal(0, q8.CurrentLength);
        }

        // A paged cache on the default F16 pool is lossy as well.
        using var pool = new KvBlockPool(tiny.Config.NumLayers, tiny.Config.NumKvHeads, tiny.Config.HeadDim, blockSize: 4, totalBlocks: 32);
        using var paged = new PagedKvCache(pool, tiny.Config.NumLayers, tiny.KvStride, 32);
        Assert.Throws<NotSupportedException>(() => assembler.Assemble(paged, [entry], Fingerprint));
        Assert.Equal(0, paged.CurrentLength);
    }

    [Fact]
    public void RopeParameters_UnsupportedRopeVariant_ThrowsNotSupportedNamingIt()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        RoPEConfig plain = tiny.Config.RoPEConfig!.Value;

        var yarn = tiny.Config with { RoPEConfig = plain with { ScalingType = RoPEScalingType.YaRN, ScalingFactor = 4f, OrigMaxSeqLen = 16 } };
        Assert.Contains("YaRN", Assert.Throws<NotSupportedException>(() => RopeParameters.FromConfig(yarn)).Message, StringComparison.Ordinal);

        var linear = tiny.Config with { RoPEConfig = plain with { ScalingType = RoPEScalingType.Linear, ScalingFactor = 2f } };
        Assert.Contains("Linear", Assert.Throws<NotSupportedException>(() => RopeParameters.FromConfig(linear)).Message, StringComparison.Ordinal);

        var noRope = tiny.Config with { RoPEConfig = null };
        Assert.Throws<NotSupportedException>(() => RopeParameters.FromConfig(noRope));
    }

    [Fact]
    public void ChunkFingerprint_DependsOnEveryIdentityInput()
    {
        using var tiny = TinyModel.Load(_scratch, "qwen3");
        string fp = ChunkFingerprint.Compute(tiny.Config, "org/model", "F32", "tok-1");

        Assert.Equal(fp, ChunkFingerprint.Compute(tiny.Config, "org/model", "F32", "tok-1"));
        Assert.NotEqual(fp, ChunkFingerprint.Compute(tiny.Config, "org/other", "F32", "tok-1"));
        Assert.NotEqual(fp, ChunkFingerprint.Compute(tiny.Config, "org/model", "Q4_K_M", "tok-1"));
        Assert.NotEqual(fp, ChunkFingerprint.Compute(tiny.Config, "org/model", "F32", "tok-2"));
        var otherTheta = tiny.Config with { RoPEConfig = tiny.Config.RoPEConfig!.Value with { Theta = 1000f } };
        Assert.NotEqual(fp, ChunkFingerprint.Compute(otherTheta, "org/model", "F32", "tok-1"));
    }

    // ---- helpers ----

    private static SimpleKvCache NewCache(TinyModel tiny)
        => new(KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false), SyntheticDenseGguf.Context);

    /// <summary>Full prefill of <paramref name="ids"/> at 0..; returns K (or V) rows [from, from+count) of <paramref name="layer"/>.</summary>
    private static float[] FullPrefillRows(TinyModel tiny, int[] ids, int layer, bool keys, int from, int count)
    {
        using var cache = NewCache(tiny);
        using (ITensor _ = tiny.Model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, cache, lastTokenLogitsOnly: true)) { }
        ReadOnlySpan<float> all = keys ? cache.KeysSpan(layer) : cache.ValuesSpan(layer);
        return all.Slice(from * tiny.KvStride, count * tiny.KvStride).ToArray();
    }

    private static unsafe float[] LastRow(IModel model, int[] ids, int start, IKvCache cache)
    {
        using ITensor logits = model.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), -1, cache, lastTokenLogitsOnly: true);
        int vocab = logits.Shape[1];
        int row = logits.Shape[0] - 1;
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)row * vocab, vocab).ToArray();
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    /// <summary>A loaded synthetic model; keeps the mapped GGUF alive for the model's lifetime.</summary>
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
            _gguf.Dispose();
        }
    }
}
