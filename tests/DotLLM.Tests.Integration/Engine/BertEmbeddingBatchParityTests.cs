using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>#1405: <see cref="IEmbeddingModel.ForwardHiddenBatch"/> must equal one-at-a-time <c>ForwardHidden</c> bit for bit,
/// at every batch size (sequential, sub-pool split and one-thread-per-sequence regimes), and keep input order.</summary>
public sealed class BertEmbeddingBatchParityTests
{
    [SkippableTheory]
    [InlineData("minilm-q8")]
    [InlineData("nomic-q8")]
    [InlineData("minilm-f16")]
    public unsafe void Batch_is_bit_identical_to_one_at_a_time(string key)
    {
        var spec = BertEmbeddingLlamaCppParityTests.SpecFor(key);
        var loc = TestFixtureResolver.ResolveFile(spec.Env, spec.Org, spec.Repo, spec.File);
        Skip.If(!loc.Found, loc.SkipMessage(spec.Description));

        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Embeddings", spec.Fixture)));
        var baseTokens = doc.RootElement.GetProperty("inputs").EnumerateArray()
            .Select(e => e.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();

        using var gguf = GgufFile.Open(loc.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = BertEncoderModel.LoadFromGguf(gguf, config, new ThreadingConfig(8, 8));
        int h = config.HiddenSize;

        foreach (int count in new[] { 1, 2, 3, 5, 13, 40 })
        {
            // Varying lengths (prefix kept, final token kept as the last id) so length-sorting actually reorders.
            var seqs = new List<int[]>();
            for (int i = 0; i < count; i++)
            {
                var t = baseTokens[i % baseTokens.Length];
                int len = Math.Max(2, t.Length - (i * 7) % Math.Max(1, t.Length - 2));
                var s = t[..len];
                s[^1] = t[^1];
                seqs.Add(s);
            }

            var expected = new float[count][];
            for (int i = 0; i < count; i++)
            {
                using var hid = model.ForwardHidden(seqs[i], Enumerable.Range(0, seqs[i].Length).ToArray(), 0);
                expected[i] = new ReadOnlySpan<float>((void*)hid.DataPointer, seqs[i].Length * h).ToArray();
            }

            var got = new float[count][];
            ((IEmbeddingModel)model).ForwardHiddenBatch(seqs, (idx, hidden) => got[idx] = hidden.ToArray(), 0);

            for (int i = 0; i < count; i++)
                Assert.True(expected[i].AsSpan().SequenceEqual(got[i]),
                    $"batch of {count}: sequence {i} ({seqs[i].Length} tok) differs from the one-at-a-time result");
        }
    }
}
