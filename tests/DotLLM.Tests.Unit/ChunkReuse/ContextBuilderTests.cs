using ChunkKvLab;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// E1 context construction: chunk token ids are fixed once per document and contexts are concatenations of those ids. The
/// fixture tokenizer merges "a"+"b" into one token, so re-tokenising decoded text at a chunk boundary gives different ids;
/// a builder that re-tokenises cannot pass.
/// </summary>
public sealed class ContextBuilderTests
{
    [Fact]
    public void ContextBuilder_TokenizesChunksOnce_ConcatenatedIdsEqualPerChunkIds()
    {
        var tokenizer = new MergingTokenizer();
        // Four 4-token chunks: two end in 'a', two start with 'b', so some orderings put "a|b" on a boundary.
        (string, string)[] docs = [("d0.cs", "cdeacdea"), ("d1.cs", "bcdebcde"), ("d2.cs", "xyz")];
        var builder = new ContextBuilder(docs, tokenizer, chunkTokens: 4);

        Assert.Equal(3, tokenizer.EncodeCalls);           // once per document
        Assert.Equal(4, builder.Chunks.Count);            // "xyz" (3 tokens) is shorter than a chunk and is dropped
        Assert.All(builder.Chunks, c => Assert.Equal(4, c.TokenIds.Length));

        bool sawBoundaryMerge = false;
        for (int seed = 0; seed < 16; seed++)
        {
            ContextPlan plan = builder.Build(seed, contextTokens: 8, continuationTokens: 2);
            Assert.Equal(2, plan.RequestOrder.Length);
            int[] expected = plan.RequestOrder.SelectMany(i => builder.Chunks[i].TokenIds).ToArray();
            Assert.Equal(expected, plan.ContextIds);
            Assert.Equal(2, plan.Continuation.Length);

            // Substitution check: the fixture can tell the two pipelines apart for at least one ordering.
            int[] retokenised = tokenizer.Encode(tokenizer.Decode(plan.ContextIds));
            if (!retokenised.AsSpan().SequenceEqual(plan.ContextIds)) sawBoundaryMerge = true;
        }

        Assert.True(sawBoundaryMerge, "fixture never put 'a|b' on a chunk boundary; it cannot detect re-tokenisation");
        Assert.Equal(3 + 16, tokenizer.EncodeCalls); // only this test's own re-encodes; Build never tokenises
    }

    /// <summary>Character tokenizer that merges the pair "ab" into a single token (id 3).</summary>
    private sealed class MergingTokenizer : ITokenizer
    {
        public int EncodeCalls { get; private set; }

        public int[] Encode(string text)
        {
            EncodeCalls++;
            var ids = new List<int>();
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == 'a' && i + 1 < text.Length && text[i + 1] == 'b') { ids.Add(3); i++; }
                else ids.Add(text[i] + 10);
            }
            return ids.ToArray();
        }

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds) sb.Append(DecodeToken(id));
            return sb.ToString();
        }

        public string DecodeToken(int tokenId) => tokenId == 3 ? "ab" : ((char)(tokenId - 10)).ToString();

        public int VocabSize => 200;

        public int BosTokenId => -1;

        public int EosTokenId => -1;

        public int CountTokens(string text) => Encode(text).Length;
    }
}
