using System.Globalization;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Tokenizers;
using Xunit;
using TinyQwen = DotLLM.Tests.Unit.Engine.PrefixCacheFailedRequestTests.TinyQwen;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #2064: after a fully accepted round the draft model had not forwarded the last accepted draft, so the next round
/// wrote past a never-written row of the draft KV-cache and nearly every later draft was rejected. A model drafting for
/// itself under greedy decoding must have every draft accepted.
/// </summary>
public sealed class SpeculativeDraftCacheContinuityTests : IClassFixture<TinyQwen>
{
    private const int K = 3;
    private readonly TinyQwen _tiny;

    public SpeculativeDraftCacheContinuityTests(TinyQwen tiny) => _tiny = tiny;

    private static readonly int[] Prompt = [5, 11, 23, 7, 31, 2, 19, 41];

    private static InferenceOptions Opts() => new() { Temperature = 0f, MaxTokens = 24 };

    private static string Text(int[] ids) => string.Join(",", ids.Select(i => i.ToString(CultureInfo.InvariantCulture)));

    [Fact]
    public void SelfDraft_DraftCacheTracksTargetCache_AcrossRounds()
    {
        var model = _tiny.Model;
        using var target = new SimpleKvCache(KvGeometry.FromConfig(_tiny.Config, false), TinyQwen.Context);
        using var draft = new SimpleKvCache(KvGeometry.FromConfig(_tiny.Config, false), TinyQwen.Context);
        var positions = Enumerable.Range(0, Prompt.Length).ToArray();
        int first;
        using (var logits = model.Forward(Prompt, positions, -1, target))
            first = ArgMaxLastRow(logits);
        using (model.Forward(Prompt, positions, -1, draft)) { }

        var decoder = new SpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(Opts());
        var generated = new List<int> { first };
        var output = new int[K + 1];
        int position = Prompt.Length;
        for (int round = 0; round < 4; round++)
        {
            var result = decoder.DraftAndVerify(model, model, target, draft, pipeline, generated, null,
                position, TinyQwen.Vocab, TinyQwen.Vocab, K, output);

            Assert.Equal(K + 1, result.AcceptedCount);
            // Both caches must hold every row before the next round's last token: no gap on the draft side.
            Assert.Equal(target.CurrentLength, draft.CurrentLength);
            position += result.AcceptedCount;
            Assert.Equal(position, draft.CurrentLength);
            generated.AddRange(output.AsSpan(0, result.AcceptedCount).ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SelfDraft_AcceptsEveryDraft_AndMatchesPlainDecode(bool streaming)
    {
        var plain = new TextGenerator(_tiny.Model, new IdsTokenizer()).Generate(Text(Prompt), Opts()).GeneratedTokenIds;
        var generator = new TextGenerator(_tiny.Model, new IdsTokenizer(), draftModel: _tiny.Model, speculativeCandidates: K);

        var ids = new List<int>();
        InferenceTimings? timings = null;
        if (streaming)
        {
            await foreach (var t in generator.GenerateStreamingTokensAsync(Text(Prompt), Opts()))
            {
                if (t.TokenId >= 0) ids.Add(t.TokenId);
                timings = t.Timings ?? timings;
            }
        }
        else
        {
            var response = generator.Generate(Text(Prompt), Opts());
            ids.AddRange(response.GeneratedTokenIds);
            timings = response.Timings;
        }

        Assert.Equal(plain, ids);
        Assert.NotNull(timings);
        var t2 = timings.Value;
        Assert.True(t2.SpeculativeDraftTokens >= 3 * K, $"only {t2.SpeculativeDraftTokens} drafted");
        Assert.Equal(t2.SpeculativeDraftTokens, t2.SpeculativeAcceptedTokens);
    }

    private static unsafe int ArgMaxLastRow(DotLLM.Core.Tensors.ITensor logits)
    {
        var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * TinyQwen.Vocab, TinyQwen.Vocab);
        return System.Numerics.Tensors.TensorPrimitives.IndexOfMax(row);
    }

    private sealed class IdsTokenizer : ITokenizer
    {
        public int VocabSize => TinyQwen.Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }
}
