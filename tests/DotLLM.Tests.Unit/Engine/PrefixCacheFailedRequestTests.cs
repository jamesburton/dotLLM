using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1589: a prefix-cache hit followed by a failed or cancelled request must not leave the entry claiming rows the
/// request has since overwritten, and a fresh KV cache allocated beside a prefix cache must be disposed when prefill throws.
/// Runs a deterministic random-weight dense qwen3 on the CPU, so a wrong restored K/V row changes the greedy continuation.
/// </summary>
public sealed class PrefixCacheFailedRequestTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private const int MaxTokens = 6;
    private const int Shared = 12;
    private readonly TinyQwen _tiny;

    public PrefixCacheFailedRequestTests(TinyQwen tiny) => _tiny = tiny;

    private static readonly int[] PromptA = Ids(24, seed: 3);
    /// <summary>Shares the first <see cref="Shared"/> tokens with A, then diverges.</summary>
    private static readonly int[] PromptB = [.. PromptA[..Shared], .. Ids(12, seed: 11)];

    private static int[] Ids(int count, int seed)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = 2 + (seed * 7 + i * 13) % (TinyQwen.Vocab - 2);
        return ids;
    }

    private static string Text(int[] ids) => string.Join(",", ids.Select(i => i.ToString(CultureInfo.InvariantCulture)));

    private static InferenceOptions Opts() => new() { Temperature = 0f, MaxTokens = MaxTokens };

    private static TextGenerator Generator(IModel model, PrefixCache? prefix, Func<ModelConfig, int, IKvCache>? factory = null)
        => new(model, new IdsTokenizer(TinyQwen.Vocab), kvCacheFactory: factory, prefixCache: prefix, prefillChunkSize: 4);

    private static async Task<int[]> Run(TextGenerator gen, int[] prompt, bool stream, CancellationToken ct = default)
    {
        if (!stream)
            return gen.Generate(Text(prompt), Opts(), cancellationToken: ct).GeneratedTokenIds;
        var ids = new List<int>();
        await foreach (var t in gen.GenerateStreamingTokensAsync(Text(prompt), Opts(), ct))
            ids.Add(t.TokenId);
        return ids.ToArray();
    }

    private int[] Plain(int[] prompt)
    {
        using var m = new FaultModel(_tiny.Model);
        return Generator(m, null).Generate(Text(prompt), Opts()).GeneratedTokenIds;
    }

    [Theory]
    [InlineData("cancel", false)]
    [InlineData("cancel", true)]
    [InlineData("throw", false)]
    [InlineData("throw", true)]
    public async Task FailedRequestAfterHit_DoesNotLeaveStaleEntry(string failure, bool stream)
    {
        int[] plainA = Plain(PromptA);
        using var m = new FaultModel(_tiny.Model);
        using var prefix = new PrefixCache(maxEntries: 1);
        var gen = Generator(m, prefix);
        Assert.Equal(plainA, await Run(gen, PromptA, stream));        // A completes: entry claims A + generated over its rows

        using var cts = new CancellationTokenSource();
        int forwards = 0;
        m.AfterPrefillForward = () =>
        {
            if (++forwards != 1) return;                              // after B's first chunk wrote rows 12..15 with B's tokens
            if (failure == "cancel") cts.Cancel();
            else throw new InvalidOperationException("injected failure");
        };
        if (failure == "cancel")
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Run(gen, PromptB, stream, cts.Token));
        else
            await Assert.ThrowsAsync<InvalidOperationException>(() => Run(gen, PromptB, stream));
        Assert.True(forwards >= 1);                                   // the failure really fired mid-prefill, after a hit
        m.AfterPrefillForward = null;

        // C = A again: it matches the stored sequence beyond the Shared rows that are still valid.
        Assert.Equal(plainA, await Run(gen, PromptA, stream));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FailedRequestAfterHit_KeepsValidPrefix(bool stream)
    {
        using var m = new FaultModel(_tiny.Model);
        using var prefix = new PrefixCache(maxEntries: 1);
        var gen = Generator(m, prefix);
        await Run(gen, PromptA, stream);
        m.AfterPrefillForward = () => throw new InvalidOperationException("injected failure");
        await Assert.ThrowsAsync<InvalidOperationException>(() => Run(gen, PromptB, stream));

        // The Shared rows were never overwritten: they stay reusable (narrowing must not just drop the whole entry).
        var (entry, matched) = prefix.FindMatch(PromptA);
        Assert.NotNull(entry);
        Assert.Equal(Shared, matched);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task FreshKvCache_IsDisposed_WhenPrefillThrows(bool withPrefixCache, bool stream)
    {
        var caches = new List<IKvCache>();
        IKvCache Factory(ModelConfig c, int size)
        {
            var k = new SimpleKvCache(KvGeometry.FromConfig(c, false), size);
            caches.Add(k);
            return k;
        }
        using var m = new FaultModel(_tiny.Model);
        using var prefix = withPrefixCache ? new PrefixCache(maxEntries: 1) : null;
        var gen = Generator(m, prefix, Factory);
        m.AfterPrefillForward = () => throw new InvalidOperationException("injected failure");

        await Assert.ThrowsAsync<InvalidOperationException>(() => Run(gen, PromptA, stream));

        Assert.Single(caches);
        Assert.All(caches, c => Assert.Throws<ObjectDisposedException>(() => c.TryGrowTo(c.MaxLength)));
        Assert.Equal(0, prefix?.EntryCount ?? 0);
    }

    [Fact]
    public async Task SuccessfulRequest_HandsCacheToPrefixCache_NotDisposed()
    {
        var caches = new List<IKvCache>();
        using var m = new FaultModel(_tiny.Model);
        using var prefix = new PrefixCache(maxEntries: 1);
        var gen = Generator(m, prefix, (c, size) =>
        {
            var k = new SimpleKvCache(KvGeometry.FromConfig(c, false), size);
            caches.Add(k);
            return k;
        });
        await Run(gen, PromptA, stream: false);

        Assert.Single(caches);
        Assert.True(caches[0].TryGrowTo(caches[0].MaxLength));      // still live: owned by the prefix cache
        Assert.Equal(1, prefix.EntryCount);
    }

    /// <summary>Forwards to the real model; after each multi-token (prefill) forward it has written, runs a fault hook.</summary>
    internal sealed class FaultModel(TransformerModel inner) : IModel
    {
        public Action? AfterPrefillForward { get; set; }
        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
        public bool RequiresPerSequenceState => false;
        public bool SupportsMtp => false;
        public void ResetSequenceState() { }

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            var r = inner.Forward(tokenIds, positions, deviceId, kvCache);
            if (tokenIds.Length > 1 && AfterPrefillForward is { } hook)
            {
                try { hook(); } catch { r.Dispose(); throw; }
            }
            return r;
        }

        public void Dispose() { }   // the fixture owns the real model
    }

    /// <summary>Prompts are comma-separated token ids; no EOS, so every request generates <see cref="MaxTokens"/> tokens.</summary>
    private sealed class IdsTokenizer(int vocabSize) : ITokenizer
    {
        public int VocabSize => vocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Deterministic random-weight dense qwen3 (3 layers, GQA 4:2, head dim 16 or 32, RoPE base 100, F32), built in memory: every
    /// layer above 0 depends on attention, so a wrong K/V row changes later rows and the logits.
    /// </summary>
    public sealed class TinyQwen : IDisposable
    {
        public const int Hidden = 64, Layers = 3, Heads = 4, Kv = 2, Ff = 96, Vocab = 48, Context = 64;

        private readonly string _dir;
        private readonly GgufFile _gguf, _gguf32;

        public TinyQwen()
        {
            _dir = Path.Combine(Path.GetTempPath(), "dotllm-i1589-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
            (_gguf, Config, Model) = Load(_dir, headDim: 16);
            (_gguf32, _, Model32) = Load(_dir, headDim: 32);
        }

        private static (GgufFile, ModelConfig, TransformerModel) Load(string dir, int headDim)
        {
            string path = Path.Combine(dir, $"qwen3-hd{headDim}.gguf");
            File.WriteAllBytes(path, Build(headDim));
            var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            return (gguf, config, TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
        }

        public ModelConfig Config { get; }

        /// <summary>Head dim 16 (8 rotation pairs per head).</summary>
        public TransformerModel Model { get; }

        /// <summary>Head dim 32: rows fill whole 32-element blocks, as the quantised KV caches require.</summary>
        public TransformerModel Model32 { get; }

        public void Dispose()
        {
            Model.Dispose();
            Model32.Dispose();
            _gguf.Dispose();
            _gguf32.Dispose();
            try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }

        private static byte[] Build(int headDim)
        {
            const string arch = "qwen3";
            var rng = new Random(20261103);
            var w = new GgufWriter();
            w.AddString("general.architecture", arch);
            w.AddUInt32("general.alignment", 32);
            w.AddUInt32($"{arch}.context_length", Context);
            w.AddUInt32($"{arch}.embedding_length", Hidden);
            w.AddUInt32($"{arch}.block_count", Layers);
            w.AddUInt32($"{arch}.feed_forward_length", Ff);
            w.AddUInt32($"{arch}.attention.head_count", Heads);
            w.AddUInt32($"{arch}.attention.head_count_kv", Kv);
            w.AddUInt32($"{arch}.attention.key_length", (uint)headDim);
            w.AddUInt32($"{arch}.attention.value_length", (uint)headDim);
            w.AddFloat32($"{arch}.rope.freq_base", 100f);
            w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
            var tokens = new string[Vocab];
            for (int i = 0; i < Vocab; i++) tokens[i] = "t" + i.ToString(CultureInfo.InvariantCulture);
            w.AddString("tokenizer.ggml.model", "gpt2");
            w.AddStringArray("tokenizer.ggml.tokens", tokens);
            w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, Vocab).ToArray());
            w.AddStringArray("tokenizer.ggml.merges", []);

            Matrix(w, rng, "token_embd.weight", Hidden, Vocab, 0.6f);
            Norm(w, rng, "output_norm.weight", Hidden);
            Matrix(w, rng, "output.weight", Hidden, Vocab, 0.25f);
            float sh = 1.4f / MathF.Sqrt(Hidden);
            for (int i = 0; i < Layers; i++)
            {
                string p = $"blk.{i}";
                Norm(w, rng, $"{p}.attn_norm.weight", Hidden);
                Matrix(w, rng, $"{p}.attn_q.weight", Hidden, Heads * headDim, sh);
                Matrix(w, rng, $"{p}.attn_k.weight", Hidden, Kv * headDim, sh);
                Matrix(w, rng, $"{p}.attn_v.weight", Hidden, Kv * headDim, sh);
                Matrix(w, rng, $"{p}.attn_output.weight", Heads * headDim, Hidden, 1.4f / MathF.Sqrt(Heads * headDim));
                Norm(w, rng, $"{p}.attn_q_norm.weight", headDim);
                Norm(w, rng, $"{p}.attn_k_norm.weight", headDim);
                Norm(w, rng, $"{p}.ffn_norm.weight", Hidden);
                Matrix(w, rng, $"{p}.ffn_gate.weight", Hidden, Ff, sh);
                Matrix(w, rng, $"{p}.ffn_up.weight", Hidden, Ff, sh);
                Matrix(w, rng, $"{p}.ffn_down.weight", Ff, Hidden, 1.4f / MathF.Sqrt(Ff));
            }
            return w.Build();
        }

        private static double Gauss(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        private static void Matrix(GgufWriter w, Random rng, string name, int inK, int outM, float std)
        {
            var f = new float[inK * outM];
            for (int i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
            w.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());
        }

        private static void Norm(GgufWriter w, Random rng, string name, int n)
        {
            var f = new float[n];
            for (int i = 0; i < n; i++) f[i] = 1.0f + 0.2f * (float)Gauss(rng);
            w.AddTensor(name, [n], (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());
        }
    }
}
