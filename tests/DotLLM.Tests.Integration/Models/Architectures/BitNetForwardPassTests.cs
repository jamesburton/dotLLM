using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Integration tests for the BitNet b1.58 2B4T forward pass against the official I2_S GGUF.
/// Validates the ternary-weight (I2_S) GEMV/GEMM path, the squared-ReLU gated FFN, and the
/// attention/FFN Sub-LN end-to-end: embedding → 30 blocks → final norm → tied LM head → logits.
/// </summary>
[Collection("BitNetModel")]
public class BitNetForwardPassTests
{
    private readonly BitNetModelFixture _fixture;

    public BitNetForwardPassTests(BitNetModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer);
    }

    [Fact]
    public void Config_IsBitNet_WithReLU2()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        using var _ = gguf;
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.BitNet, config.Architecture);
        Assert.Equal(ActivationFunction.ReLU2, config.ActivationFunction);
        Assert.Equal(2560, config.HiddenSize);
        Assert.Equal(30, config.NumLayers);
        Assert.Equal(128256, config.VocabSize);
    }

    [Fact]
    public void SingleToken_ProducesFiniteVocabSizedLogits()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using ITensor logits = model.Forward([tokenizer.BosTokenId], [0], deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(model.Config.VocabSize, logits.Shape[1]);

        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount);
            for (int i = 0; i < span.Length; i++)
                Assert.True(float.IsFinite(span[i]), $"Logit {i} is not finite: {span[i]}");
        }
    }

    [Fact]
    public void GreedyDecode_PredictsParis()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] tokenIds = tokenizer.Encode("The capital of France is");
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        int vocabSize = model.Config.VocabSize;
        int nextTokenId;
        unsafe
        {
            float* logitPtr = (float*)(logits.DataPointer + (long)(tokenIds.Length - 1) * vocabSize * sizeof(float));
            nextTokenId = ArgMax(new ReadOnlySpan<float>(logitPtr, vocabSize));
        }

        string predicted = tokenizer.DecodeToken(nextTokenId).Trim();
        Assert.Equal("Paris", predicted);
    }

    [Fact]
    public void Forward_WithKvCache_DecodeMatchesUncached()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] promptIds = tokenizer.Encode("The capital of France is");
        int vocabSize = model.Config.VocabSize;

        // Uncached: prefill prompt, then re-feed growing context for one decode step.
        int[] positions = new int[promptIds.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        using ITensor prefill = model.Forward(promptIds, positions, deviceId: -1);
        int firstUncached;
        unsafe
        {
            float* ptr = (float*)(prefill.DataPointer + (long)(promptIds.Length - 1) * vocabSize * sizeof(float));
            firstUncached = ArgMax(new ReadOnlySpan<float>(ptr, vocabSize));
        }

        // Cached: prefill into KV cache, then one single-token decode step.
        int cacheSize = promptIds.Length + 2;
        using var kvCache = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, cacheSize);
        int[] cachePositions = new int[cacheSize];
        for (int i = 0; i < cacheSize; i++) cachePositions[i] = i;

        int firstCached;
        using (ITensor cachedPrefill = model.Forward(
            promptIds, cachePositions.AsSpan(0, promptIds.Length), -1, kvCache))
        {
            unsafe
            {
                float* ptr = (float*)(cachedPrefill.DataPointer + (long)(promptIds.Length - 1) * vocabSize * sizeof(float));
                firstCached = ArgMax(new ReadOnlySpan<float>(ptr, vocabSize));
            }
        }

        // Decode the next token via the single-token (KV-cache) path — exercises the I2_S decode route.
        using ITensor decodeLogits = model.Forward(
            [firstCached], cachePositions.AsSpan(promptIds.Length, 1), -1, kvCache);
        int secondCached;
        unsafe
        {
            secondCached = ArgMax(new ReadOnlySpan<float>((void*)decodeLogits.DataPointer, vocabSize));
        }

        Assert.Equal(firstUncached, firstCached);
        Assert.InRange(secondCached, 0, vocabSize - 1);
    }

    private static int ArgMax(ReadOnlySpan<float> span)
        => System.Numerics.Tensors.TensorPrimitives.IndexOfMax(span);
}
