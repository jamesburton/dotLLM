using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #2030: draft-model speculation read row <c>i</c> of the target's verify logits even when the target
/// returns only one row (dense CUDA host), reading past the tensor.
/// </summary>
public sealed class SpeculativeOneRowLogitsTests
{
    private const int V = 8;
    private const int Argmax = 3;

    [Theory]
    [InlineData(true)]   // the host declares its limit: refused before the forward
    [InlineData(false)]  // the host does not declare it: caught from the returned tensor
    public void DraftAndVerify_Throws_OnAOneRowTarget(bool declaresLimit)
    {
        using var target = new FixedModel(oneRow: true, declaresLimit);
        using var draft = new FixedModel(oneRow: false, declaresLimit: false);
        using var kvT = new SimpleKvCache(1, 1, 4, 64);
        using var kvD = new SimpleKvCache(1, 1, 4, 64);
        var output = new int[8];
        var decoder = new SpeculativeDecoder(greedy: true, seed: 1);

        var ex = Assert.Throws<InvalidOperationException>(() => decoder.DraftAndVerify(
            target, draft, kvT, kvD, new SamplerPipeline(new InferenceOptions { Temperature = 0f }),
            [1], null, 0, V, V, 3, output));

        Assert.Contains("row", ex.Message);
        // A declared limit must refuse before the target's cache or recurrent state can advance.
        Assert.Equal(declaresLimit ? 0 : 1, target.ForwardCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Generate_WithDraft_OnAOneRowTarget_FallsBackToPlainDecode(bool streaming)
    {
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 6 };
        using var target = new FixedModel(oneRow: true, declaresLimit: true);
        using var draft = new FixedModel(oneRow: false, declaresLimit: false);
        var generator = new TextGenerator(target, new StubTokenizer(), draftModel: draft);

        var tokens = new List<int>();
        if (streaming)
        {
            await Drain(generator, options, tokens);
        }
        else
        {
            var response = generator.Generate("ab", options, tokens.Add);
            Assert.Equal(0, response.Timings.SpeculativeDraftTokens);
        }

        Assert.Equal(Enumerable.Repeat(Argmax, 6), tokens);
        Assert.Equal(0, draft.ForwardCalls);
    }

    private static async Task Drain(TextGenerator generator, InferenceOptions options, List<int> tokens)
    {
        await foreach (var t in generator.GenerateStreamingTokensAsync("ab", options))
            if (t.TokenId >= 0) tokens.Add(t.TokenId);
    }

    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => V;
        public int BosTokenId => 1;
        public int EosTokenId => 7;
        public int[] Encode(string text) => [1, 2];
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(System.Globalization.CultureInfo.InvariantCulture);
        public int CountTokens(string text) => 2;
    }

    /// <summary>Fixed-argmax model. <c>oneRow</c> declares a [1, V] tensor, as the dense CUDA host does.</summary>
    private sealed unsafe class FixedModel(bool oneRow, bool declaresLimit) : IModel
    {
        public int ForwardCalls;

        public ModelConfig Config => new()
        {
            VocabSize = V, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HiddenSize = 4,
            IntermediateSize = 16, HeadDim = 4, MaxSequenceLength = 64, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public int MaxAllRowLogitsLength => declaresLimit ? 1 : int.MaxValue;
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Forward(t, p, d, null);

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv)
        {
            ForwardCalls++;
            // Allocated for 16 rows whatever the declared shape, so an unfixed over-read stays in bounds.
            int allocRows = Math.Max(16, t.Length);
            nuint bytes = (nuint)(allocRows * V * sizeof(float));
            nint ptr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
            NativeMemory.Clear((void*)ptr, bytes);
            for (int r = 0; r < allocRows; r++)
                ((float*)ptr)[r * V + Argmax] = 10f;
            return new UnmanagedTensor(new TensorShape(oneRow ? 1 : t.Length, V), DType.Float32, d, ptr);
        }

        public void Dispose() { }
    }
}
