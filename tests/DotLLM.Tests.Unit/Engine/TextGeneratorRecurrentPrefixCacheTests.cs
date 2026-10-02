using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The recurrent prefix cache snapshots (KV + recurrent state) at the prefix two consecutive requests share, then serves
/// later requests that start with it by restoring the state, rolling the KV back and prefilling only the suffix.
/// The fake model's next token is a function of its WHOLE history (a running hash), so a missed restore, a stale state or
/// a wrongly positioned suffix shows up as different tokens, and the recorded forward lengths show what was skipped.
/// </summary>
public sealed class TextGeneratorRecurrentPrefixCacheTests
{
    private const int Vocab = 64;
    private const int HiddenSize = 8, NumKvHeads = 1, HeadDim = 4, MaxSeqLen = 512;
    private const int PrefixLen = 20;

    private static string Prompt(params int[] suffix)
        => string.Join(",", Enumerable.Range(2, PrefixLen).Concat(suffix));

    private static InferenceOptions Opts => new() { Temperature = 0f, MaxTokens = 4 };

    private static (TextGenerator Cached, TextGenerator Plain, HistoryModel Model) Build()
    {
        var model = new HistoryModel();
        var tok = new CsvTokenizer();
        Func<ModelConfig, int, IKvCache> kv = (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg), size);
        return (new TextGenerator(model, tok, kv, recurrentPrefixCache: true),
                new TextGenerator(model, tok, kv), model);
    }

    [Fact]
    public void SharedPrefix_IsSkippedOnLaterRequests_AndTokensMatchTheUncachedRun()
    {
        var (cached, plain, model) = Build();
        int[][] suffixes = [[30, 31], [40, 41, 42], [50], [30, 31], [60, 61]];

        for (int i = 0; i < suffixes.Length; i++)
        {
            string prompt = Prompt(suffixes[i]);
            var want = plain.Generate(prompt, Opts);
            model.Calls.Clear();
            var got = cached.Generate(prompt, Opts);

            Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);

            int prefilled = model.Calls[0];
            if (i >= 2)
            {
                // request 0 has no previous prompt, request 1 takes the snapshot, request 2 onwards hit it
                Assert.Equal(PrefixLen, got.Timings.CachedTokenCount);
                Assert.Equal(suffixes[i].Length, prefilled);          // only the suffix ran through the model
            }
            else
            {
                Assert.Equal(0, got.Timings.CachedTokenCount);
            }
        }
    }

    [Fact]
    public void NonMatchingPrompt_FallsBackToAFullPrefill_AndStaysCorrect()
    {
        var (cached, plain, model) = Build();
        foreach (int[] s in new[] { new[] { 30 }, new[] { 31 }, new[] { 32 } })
            cached.Generate(Prompt(s), Opts);                       // establishes the snapshot

        string other = string.Join(",", Enumerable.Range(40, 25));   // different prefix entirely
        var want = plain.Generate(other, Opts);
        model.Calls.Clear();
        var got = cached.Generate(other, Opts);

        Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
        Assert.Equal(0, got.Timings.CachedTokenCount);
        Assert.Equal(25, model.Calls[0]);
    }

    [Fact]
    public void PromptEqualToTheSnapshotPrefix_IsNotReused_BecauseASuffixTokenMustProduceLogits()
    {
        var (cached, plain, model) = Build();
        foreach (int[] s in new[] { new[] { 30 }, new[] { 31 }, new[] { 32 } })
            cached.Generate(Prompt(s), Opts);

        string exactlyThePrefix = Prompt();                          // 20 tokens == snapshot length
        var want = plain.Generate(exactlyThePrefix, Opts);
        var got = cached.Generate(exactlyThePrefix, Opts);

        Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
        Assert.Equal(0, got.Timings.CachedTokenCount);
    }

    [Fact]
    public void ClearRecurrentPrefixCache_ForcesAFullPrefillAgain()
    {
        var (cached, _, model) = Build();
        foreach (int[] s in new[] { new[] { 30 }, new[] { 31 }, new[] { 32 } })
            cached.Generate(Prompt(s), Opts);
        Assert.Equal(PrefixLen, cached.Generate(Prompt(33), Opts).Timings.CachedTokenCount);

        cached.ClearRecurrentPrefixCache();
        model.Calls.Clear();
        var after = cached.Generate(Prompt(34), Opts);

        Assert.Equal(0, after.Timings.CachedTokenCount);
        Assert.Equal(PrefixLen + 1, model.Calls[0]);
    }

    [Fact]
    public void WithoutTheFlag_NothingIsCached()
    {
        var (_, plain, _) = Build();
        for (int i = 0; i < 4; i++)
            Assert.Equal(0, plain.Generate(Prompt(30 + i), Opts).Timings.CachedTokenCount);
    }

    // ── fakes ──

    private sealed class CsvTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => Vocab - 1;
        public int[] Encode(string text)
            => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => Encode(text).Length;
    }

    /// <summary>
    /// A recurrent model: state S = (S * 31 + token + 1) mod 9973 after every token; the next-token logits are one-hot on
    /// S mod (Vocab - 1) (never EOS). Supports checkpoint/restore like the GDN hybrids.
    /// </summary>
    private sealed class HistoryModel : IModel
    {
        private int _state;

        /// <summary>Token count of every Forward call, in order.</summary>
        public List<int> Calls { get; } = [];

        public ModelConfig Config => new()
        {
            VocabSize = Vocab, NumLayers = 1, NumAttentionHeads = NumKvHeads, NumKvHeads = NumKvHeads,
            HiddenSize = HiddenSize, IntermediateSize = HiddenSize * 4, HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public bool RequiresPerSequenceState => true;
        public void ResetSequenceState() => _state = 0;
        public bool SupportsRecurrentStateCheckpoint => true;
        public object? CheckpointRecurrentState() => _state;
        public void RestoreRecurrentState(object? checkpoint) { if (checkpoint is int s) _state = s; }

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Core(t, p, d, null);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => Core(t, p, d, kv);

        private unsafe ITensor Core(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            int seqLen = tokenIds.Length;
            Calls.Add(seqLen);

            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)seqLen * Vocab * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int r = 0; r < seqLen; r++)
            {
                _state = (_state * 31 + tokenIds[r] + 1) % 9973;
                var row = new Span<float>(dst + (long)r * Vocab, Vocab);
                row.Fill(-10f);
                row[_state % (Vocab - 1)] = 10f;
            }

            if (kvCache is not null)
            {
                int kvStride = NumKvHeads * HeadDim;
                nuint bytes = (nuint)(seqLen * kvStride * sizeof(float));
                nint kPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                NativeMemory.Clear((void*)kPtr, bytes);
                NativeMemory.Clear((void*)vPtr, bytes);
                kvCache.Update(new TensorRef(seqLen, kvStride, DType.Float32, -1, kPtr),
                               new TensorRef(seqLen, kvStride, DType.Float32, -1, vPtr), positions, 0);
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            return new UnmanagedTensor(new TensorShape(seqLen, Vocab), DType.Float32, deviceId, ptr);
        }
    }
}
