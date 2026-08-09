using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Verifies the text-level streaming callback added for issue #424: a consumer driven purely by
/// <c>onTextGenerated</c> sees exactly <see cref="InferenceResponse.Text"/>, including the kept
/// prefix of a final token whose stop-string suffix was trimmed at a character boundary, and
/// never sees the stop string itself.
/// </summary>
/// <remarks>
/// The scripted tokenizer here is what makes these tests discriminating: token 4 decodes to
/// <c>"ld&lt;|im_end|&gt;"</c>, so the stop string covers only part of it. There is no token id
/// whose text is <c>"ld"</c>, which is precisely why the id-level <c>Action&lt;int&gt;</c>
/// callback cannot express the output and these tests fail without the text-level callback.
/// </remarks>
public sealed class TextGeneratorTextStreamingCallbackTests
{
    private const int VocabSize = 16;
    private const string StopString = "<|im_end|>";

    [Fact]
    public void StopStringSplittingAToken_CallbackTextEqualsReturnedText()
    {
        // "Hello" + " wor" + "ld<|im_end|>"  →  returned text "Hello world".
        // The stop string covers only the tail of the last token; "ld" must still be streamed.
        using var model = new ScriptedModel(2, 3, 4);
        var generator = new TextGenerator(model, new ScriptedTokenizer());

        var streamed = new StringBuilder();
        var response = generator.Generate(
            "prompt",
            new InferenceOptions
            {
                MaxTokens = 8,
                Temperature = 0f,
                StopSequences = [StopString]
            },
            onTextGenerated: text => streamed.Append(text));

        Assert.Equal("Hello world", response.Text);
        Assert.Equal(FinishReason.Stop, response.FinishReason);

        // AC1: callback-assembled text is exactly the non-streaming text, trimmed token included.
        Assert.Equal(response.Text, streamed.ToString());
        // AC2: the stop string never reaches the callback.
        Assert.DoesNotContain(StopString, streamed.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void StopStringSplittingAToken_IdCallbackIsStrictlyPreTrim()
    {
        // AC3: the id-level callback keeps its existing (documented, pre-trim) semantics —
        // the stopping token is not handed over, so re-rendering the ids yields a strict prefix.
        using var model = new ScriptedModel(2, 3, 4);
        var generator = new TextGenerator(model, new ScriptedTokenizer());
        var tokenizer = new ScriptedTokenizer();

        var ids = new List<int>();
        var streamed = new StringBuilder();
        var response = generator.Generate(
            "prompt",
            new InferenceOptions
            {
                MaxTokens = 8,
                Temperature = 0f,
                StopSequences = [StopString]
            },
            onTokenGenerated: ids.Add,
            onTextGenerated: text => streamed.Append(text));

        Assert.Equal(new[] { 2, 3 }, ids);

        // The point of the issue: the id stream cannot reproduce the output, the text stream can.
        string fromIds = tokenizer.Decode(CollectionsMarshal.AsSpan(ids), stripBosSpace: false);
        Assert.Equal("Hello wor", fromIds);
        Assert.NotEqual(response.Text, fromIds);
        Assert.Equal(response.Text, streamed.ToString());
    }

    [Fact]
    public void StopStringSpanningSeveralTokens_IsNeverPartiallyEmitted()
    {
        // "Hi" + "<|im" + "_end|>" → the stop string straddles two tokens, so a callback that
        // simply forwarded each decode delta would already have emitted "<|im" and could not
        // retract it. Text that might begin a stop string must be withheld until it is safe.
        using var model = new ScriptedModel(5, 6, 7);
        var generator = new TextGenerator(model, new ScriptedTokenizer());

        var streamed = new StringBuilder();
        var response = generator.Generate(
            "prompt",
            new InferenceOptions
            {
                MaxTokens = 8,
                Temperature = 0f,
                StopSequences = [StopString]
            },
            onTextGenerated: text => streamed.Append(text));

        Assert.Equal("Hi", response.Text);
        Assert.Equal("Hi", streamed.ToString());
        Assert.DoesNotContain("<|im", streamed.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutAStopString_WithheldTextIsStillFlushedAtTheEnd()
    {
        // Generation ends on the token limit while "<|im" is being withheld as a possible stop
        // prefix. Nothing may be swallowed: the final flush has to release it.
        using var model = new ScriptedModel(5, 6);
        var generator = new TextGenerator(model, new ScriptedTokenizer());

        var streamed = new StringBuilder();
        var response = generator.Generate(
            "prompt",
            new InferenceOptions
            {
                MaxTokens = 2,
                Temperature = 0f,
                StopSequences = [StopString]
            },
            onTextGenerated: text => streamed.Append(text));

        Assert.Equal("Hi<|im", response.Text);
        Assert.Equal(response.Text, streamed.ToString());
    }

    [Fact]
    public void NoStopSequences_CallbackTextEqualsReturnedText()
    {
        using var model = new ScriptedModel(2, 3, 8);
        var generator = new TextGenerator(model, new ScriptedTokenizer());

        var streamed = new StringBuilder();
        var response = generator.Generate(
            "prompt",
            new InferenceOptions { MaxTokens = 3, Temperature = 0f },
            onTextGenerated: text => streamed.Append(text));

        Assert.Equal("Hello wor!", response.Text);
        Assert.Equal(response.Text, streamed.ToString());
    }

    // ── Fakes ──

    /// <summary>
    /// Tokenizer with a hand-written vocabulary chosen so a stop string can split a token
    /// (id 4 = <c>"ld&lt;|im_end|&gt;"</c>) or straddle two (ids 6 + 7).
    /// </summary>
    private sealed class ScriptedTokenizer : ITokenizer
    {
        private static readonly Dictionary<int, string> Pieces = new()
        {
            [1] = "",              // BOS
            [2] = "Hello",
            [3] = " wor",
            [4] = "ld" + StopString,
            [5] = "Hi",
            [6] = "<|im",
            [7] = "_end|>",
            [8] = "!",
            [9] = "prompt",
            [15] = "",             // EOS
        };

        public int VocabSize => TextGeneratorTextStreamingCallbackTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => 15;

        public int[] Encode(string text) => [9];

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new StringBuilder();
            foreach (int id in tokenIds)
                sb.Append(DecodeToken(id));
            return sb.ToString();
        }

        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);

        public string DecodeToken(int tokenId)
            => Pieces.TryGetValue(tokenId, out string? s) ? s : tokenId.ToString(CultureInfo.InvariantCulture);

        public int CountTokens(string text) => 1;
    }

    /// <summary>
    /// Fake model that makes each successive sampling step pick the next token in a script;
    /// once the script is exhausted it repeats token 8 ("!") so generation ends on the token limit.
    /// </summary>
    private sealed class ScriptedModel : IModel
    {
        private readonly int[] _script;
        private int _index;

        public ScriptedModel(params int[] script) => _script = script;

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
            int next = _index < _script.Length ? _script[_index] : 8;
            _index++;

            int batchSize = tokenIds.Length;
            long totalFloats = (long)batchSize * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
            {
                var row = new Span<float>(dst + b * VocabSize, VocabSize);
                row.Fill(-10f);
                row[next] = 10f;
            }

            if (kvCache != null)
            {
                int kvStride = Config.NumKvHeads * Config.HeadDim;
                nuint bytes = (nuint)(batchSize * kvStride * sizeof(float));
                nint kPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                NativeMemory.Clear((void*)kPtr, bytes);
                NativeMemory.Clear((void*)vPtr, bytes);
                kvCache.Update(
                    new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr),
                    new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr),
                    positions, 0);
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
