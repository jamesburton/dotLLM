using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Verifies the <see cref="TextGenerator"/> prefill chunk size (llama.cpp <c>-ub</c> analog,
/// issue #141) reaches the model layer: with a chunk size configured, the prompt prefill is
/// split into multiple <c>Forward</c> calls of at most that many tokens with contiguous,
/// correctly-offset positions; with the default (0), the prefill stays a single forward pass.
/// </summary>
public sealed class TextGeneratorPrefillChunkTests
{
    private const int VocabSize = 16;
    private const int PromptLen = 10;

    [Fact]
    public void DefaultChunkSize_SinglePrefillForward()
    {
        var model = new RecordingModel(argmaxToken: 3);
        var generator = new TextGenerator(model, new StubTokenizer());

        generator.Generate("prompt", new InferenceOptions { MaxTokens = 2, Temperature = 0f });

        // First recorded call is the whole prompt in one pass; the rest are single-token decodes.
        Assert.Equal(PromptLen, model.Calls[0].Length);
        Assert.Equal(0, model.Calls[0].FirstPosition);
        Assert.All(model.Calls.Skip(1), c => Assert.Equal(1, c.Length));
    }

    [Fact]
    public void ChunkSize4_SplitsPrefillInto4_4_2()
    {
        var model = new RecordingModel(argmaxToken: 3);
        var generator = new TextGenerator(model, new StubTokenizer(), prefillChunkSize: 4);

        Assert.Equal(4, generator.PrefillChunkSize);
        generator.Generate("prompt", new InferenceOptions { MaxTokens = 2, Temperature = 0f });

        // Prefill: 10 tokens in chunks of ≤ 4 with contiguous positions.
        Assert.Equal(4, model.Calls[0].Length);
        Assert.Equal(0, model.Calls[0].FirstPosition);
        Assert.Equal(4, model.Calls[1].Length);
        Assert.Equal(4, model.Calls[1].FirstPosition);
        Assert.Equal(2, model.Calls[2].Length);
        Assert.Equal(8, model.Calls[2].FirstPosition);
        // Decode continues from position 10, one token at a time.
        Assert.Equal(1, model.Calls[3].Length);
        Assert.Equal(10, model.Calls[3].FirstPosition);
    }

    [Fact]
    public void ChunkSizeLargerThanPrompt_SinglePrefillForward()
    {
        var model = new RecordingModel(argmaxToken: 3);
        var generator = new TextGenerator(model, new StubTokenizer(), prefillChunkSize: 64);

        generator.Generate("prompt", new InferenceOptions { MaxTokens = 1, Temperature = 0f });

        Assert.Equal(PromptLen, model.Calls[0].Length);
    }

    [Fact]
    public void ChunkedPrefill_ProducesSameTokens_AsSinglePass()
    {
        // The fake model's logits depend only on the last input token, so chunked and
        // unchunked prefill must produce the identical greedy continuation.
        var chunked = new RecordingModel(argmaxToken: 3);
        var single = new RecordingModel(argmaxToken: 3);

        var chunkedResponse = new TextGenerator(chunked, new StubTokenizer(), prefillChunkSize: 3)
            .Generate("prompt", new InferenceOptions { MaxTokens = 4, Temperature = 0f });
        var singleResponse = new TextGenerator(single, new StubTokenizer())
            .Generate("prompt", new InferenceOptions { MaxTokens = 4, Temperature = 0f });

        Assert.Equal(singleResponse.GeneratedTokenIds, chunkedResponse.GeneratedTokenIds);
        // And the chunked run really did chunk (3+3+3+1 prefill calls before decoding).
        Assert.Equal(new[] { 3, 3, 3, 1 }, chunked.Calls.Take(4).Select(c => c.Length).ToArray());
    }

    // ── Fakes ──

    private sealed record ForwardCall(int Length, int FirstPosition);

    /// <summary>Stub tokenizer: any prompt encodes to PromptLen sequential ids; EOS is never produced.</summary>
    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => TextGeneratorPrefillChunkTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => 15;

        public int[] Encode(string text) => Enumerable.Range(2, PromptLen).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString();
        public int CountTokens(string text) => PromptLen;
    }

    /// <summary>
    /// Fake model that records every Forward call (token count + first position) and returns
    /// fixed-argmax logits for each position in the batch.
    /// </summary>
    private sealed class RecordingModel : IModel
    {
        private readonly int _argmaxToken;

        public List<ForwardCall> Calls { get; } = new();

        public RecordingModel(int argmaxToken) => _argmaxToken = argmaxToken;

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = 1,
            NumAttentionHeads = 1,
            NumKvHeads = 1,
            HiddenSize = 4,
            IntermediateSize = 16,
            HeadDim = 4,
            MaxSequenceLength = 64,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            Calls.Add(new ForwardCall(tokenIds.Length, positions[0]));

            int batchSize = tokenIds.Length;
            long totalFloats = (long)batchSize * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
            {
                var row = new Span<float>(dst + b * VocabSize, VocabSize);
                row.Fill(-10f);
                row[_argmaxToken] = 10f;
            }

            if (kvCache != null)
            {
                int kvStride = Config.NumKvHeads * Config.HeadDim;
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                NativeMemory.Clear((void*)kPtr, (nuint)(batchSize * kvStride * sizeof(float)));
                NativeMemory.Clear((void*)vPtr, (nuint)(batchSize * kvStride * sizeof(float)));
                var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                kvCache.Update(kRef, vRef, positions, 0);
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
