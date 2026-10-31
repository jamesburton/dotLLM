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
/// The two additive internal <see cref="TransformerModel"/> seams Task 7 adds for the selectors: C4 (a
/// <c>ForwardLayerWindow</c> overload that writes/reads a KV-cache) and C7 (a read-only attention-query probe). Both must leave
/// the existing paths byte-identical.
/// </summary>
public sealed class ModelProbeTests : IDisposable
{
    private static readonly int[] Context = [3, 17, 5, 29, 11, 40, 8, 22, 1, 35, 12, 7, 44, 2, 19];
    private static readonly int[] Suffix = [9, 14, 38];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-probe-" + Guid.NewGuid().ToString("N"));

    public ModelProbeTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public void ForwardLayerWindow_WithCache_AllWindowsPlusOutputHead_EqualsForwardBitwise()
    {
        using var m = Load();
        int n = Context.Length, hidden = m.Config.HiddenSize, layers = m.Config.NumLayers;
        int[] pos = Enumerable.Range(0, n).ToArray();

        using SimpleKvCache refCache = NewCache(m);
        float[] reference = Rows(m.Model.Forward(Context, pos, -1, refCache));

        // Same forward as two windows [0, 1) and [1, layers) over one cache, then the output head.
        using SimpleKvCache cache = NewCache(m);
        var h1 = new float[n * hidden];
        var h2 = new float[n * hidden];
        m.Model.ForwardLayerWindow(Context, ReadOnlySpan<float>.Empty, pos, 0, 1, h1, cache);
        Assert.Equal(n, cache.CurrentLength);
        m.Model.ForwardLayerWindow(ReadOnlySpan<int>.Empty, h1, pos, 1, layers - 1, h2, cache);
        float[] windowed = Rows(m.Model.ApplyOutputHead(h2, n, -1));

        Assert.Equal(reference, windowed);
        for (int l = 0; l < layers; l++)
        {
            Assert.Equal(refCache.KeysSpan(l).ToArray(), cache.KeysSpan(l).ToArray());
            Assert.Equal(refCache.ValuesSpan(l).ToArray(), cache.ValuesSpan(l).ToArray());
        }

        // The window continues from cached context like Forward does: a suffix through the windows equals Forward's suffix.
        int[] spos = Enumerable.Range(n, Suffix.Length).ToArray();
        float[] refSuffix = Rows(m.Model.Forward(Suffix, spos, -1, refCache));
        var s1 = new float[Suffix.Length * hidden];
        var s2 = new float[Suffix.Length * hidden];
        m.Model.ForwardLayerWindow(Suffix, ReadOnlySpan<float>.Empty, spos, 0, 2, s1, cache);
        m.Model.ForwardLayerWindow(ReadOnlySpan<int>.Empty, s1, spos, 2, layers - 2, s2, cache);
        Assert.Equal(refSuffix, Rows(m.Model.ApplyOutputHead(s2, Suffix.Length, -1)));
    }

    [Fact]
    public void ForwardLayerWindow_WithoutCache_IsUnchanged_AndDiffersFromCachedSuffix()
    {
        // The original cache-less overload still attends only within the given rows (old behaviour): for a suffix after a
        // cached context its output must differ from the cached window's, and must equal the explicit kvCache: null call.
        using var m = Load();
        int hidden = m.Config.HiddenSize, layers = m.Config.NumLayers;
        int[] spos = Enumerable.Range(0, Suffix.Length).ToArray();
        var a = new float[Suffix.Length * hidden];
        var b = new float[Suffix.Length * hidden];
        m.Model.ForwardLayerWindow(Suffix, ReadOnlySpan<float>.Empty, spos, 0, layers, a);
        m.Model.ForwardLayerWindow(Suffix, ReadOnlySpan<float>.Empty, spos, 0, layers, b, kvCache: null);
        Assert.Equal(a, b);

        using SimpleKvCache cache = NewCache(m);
        int n = Context.Length;
        using (m.Model.Forward(Context, Enumerable.Range(0, n).ToArray(), -1, cache, lastTokenLogitsOnly: true)) { }
        var c = new float[Suffix.Length * hidden];
        m.Model.ForwardLayerWindow(Suffix, ReadOnlySpan<float>.Empty, Enumerable.Range(n, Suffix.Length).ToArray(), 0, layers, c, cache);
        Assert.True(a.Zip(c).Max(t => MathF.Abs(t.First - t.Second)) > 1e-3f, "cached window ignored the cached context");
    }

    [Fact]
    public void AttentionQueryProbe_SetOrUnset_LogitsAndCacheBitIdentical_AndSeesEveryLayer()
    {
        using var m = Load();
        int n = Context.Length;
        int[] pos = Enumerable.Range(0, n).ToArray();

        using SimpleKvCache c1 = NewCache(m);
        float[] plain = Rows(m.Model.Forward(Context, pos, -1, c1));

        var seen = new List<int>();
        m.Model.AttentionQueryProbe = (int layer, ReadOnlySpan<float> q, ReadOnlySpan<float> attnOut, ReadOnlySpan<int> positions,
            int numHeads, int numKvHeads, int headDim, float scale, IKvCache cache) =>
        {
            Assert.Equal(n, positions.Length);
            Assert.Equal(n * numHeads * headDim, q.Length);
            Assert.Equal(n * numHeads * headDim, attnOut.Length);
            Assert.Equal(m.Config.AttentionScoreScale(headDim), scale);
            Assert.Equal(n, cache.CurrentLength); // called after the rows were written
            seen.Add(layer);
        };
        using SimpleKvCache c2 = NewCache(m);
        float[] probed;
        try { probed = Rows(m.Model.Forward(Context, pos, -1, c2)); }
        finally { m.Model.AttentionQueryProbe = null; }

        Assert.Equal(plain, probed);
        Assert.Equal(Enumerable.Range(0, m.Config.NumLayers), seen);
        for (int l = 0; l < m.Config.NumLayers; l++)
            Assert.Equal(c1.KeysSpan(l).ToArray(), c2.KeysSpan(l).ToArray());
    }

    // ---- helpers ----

    private static unsafe float[] Rows(ITensor t)
    {
        using (t)
            return new ReadOnlySpan<float>((float*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    private static SimpleKvCache NewCache(Tiny m) => new(KvGeometry.FromConfig(m.Config, compactSlidingWindows: false), SyntheticDenseGguf.Context);

    private Tiny Load()
    {
        string path = Path.Combine(_scratch, "qwen3.gguf");
        File.WriteAllBytes(path, SyntheticDenseGguf.Build("qwen3"));
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return new Tiny(gguf, config, TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
    }

    private sealed class Tiny(GgufFile gguf, ModelConfig config, TransformerModel model) : IDisposable
    {
        public ModelConfig Config { get; } = config;

        public TransformerModel Model { get; } = model;

        public void Dispose()
        {
            Model.Dispose();
            gguf.Dispose();
        }
    }
}
