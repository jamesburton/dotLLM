using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// A constraint that is complete and permits only EOS (a one-letter classifier answer, a closed JSON object) leaves the
/// next forward pass nothing to decide, so <c>TextGenerator</c> must stop without it. On Tev1-4B / Vulkan that forward is
/// ~23 ms of a ~150 ms request. The output must be identical to running the extra pass and sampling the forced EOS.
/// </summary>
public sealed class TextGeneratorConstraintEarlyStopTests
{
    private const int VocabSize = 16;
    private const int PromptLen = 10;
    private const int HiddenSize = 8, NumKvHeads = 1, HeadDim = 4, MaxSeqLen = 128;

    [Fact]
    public void CompleteConstraintWithOnlyEosLeft_StopsAfterThePrefill()
    {
        using var model = new CountingModel();
        var generator = new TextGenerator(model, new StubTokenizer());

        // The fake model continues token t with t + 1, so the prompt's last token (11) is followed by 12; the
        // regex "12" matches the single token "12" and then permits nothing but EOS.
        var response = generator.Generate("p", new InferenceOptions
        {
            MaxTokens = 8, Temperature = 0f, ResponseFormat = new ResponseFormat.Regex { Pattern = "12" },
        });

        Assert.Equal([12], response.GeneratedTokenIds);
        Assert.Equal(FinishReason.Stop, response.FinishReason);
        Assert.Equal(1, model.ForwardCalls);   // prefill only: no decode pass for the forced EOS
    }

    [Fact]
    public void WithoutAConstraint_GenerationStillRunsDecodePasses()
    {
        using var model = new CountingModel();
        var generator = new TextGenerator(model, new StubTokenizer());

        var response = generator.Generate("p", new InferenceOptions { MaxTokens = 3, Temperature = 0f });

        Assert.Equal(3, response.GeneratedTokenIds.Length);
        Assert.Equal(3, model.ForwardCalls);   // prefill + 2 decodes
    }

    [Fact]
    public void ConstraintThatCanStillExtend_IsNotStoppedEarly()
    {
        using var model = new CountingModel();
        var generator = new TextGenerator(model, new StubTokenizer());

        // "\d+" is complete after "12" but digits remain allowed, so generation must continue.
        var response = generator.Generate("p", new InferenceOptions
        {
            MaxTokens = 3, Temperature = 0f, ResponseFormat = new ResponseFormat.Regex { Pattern = @"\d+" },
        });

        Assert.True(model.ForwardCalls > 1, "a constraint that still allows more digits must not stop after one token");
    }

    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => TextGeneratorConstraintEarlyStopTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => 15;
        public int[] Encode(string text) => Enumerable.Range(2, PromptLen).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => PromptLen;
    }

    /// <summary>Token t is always followed by t + 1; counts forward passes.</summary>
    private sealed class CountingModel : IModel
    {
        public int ForwardCalls { get; private set; }

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize, NumLayers = 1, NumAttentionHeads = NumKvHeads, NumKvHeads = NumKvHeads,
            HiddenSize = HiddenSize, IntermediateSize = HiddenSize * 4, HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Core(t, p, d, null);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => Core(t, p, d, kv);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => Core(t, p, d, kv);

        private unsafe ITensor Core(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            ForwardCalls++;
            int seqLen = tokenIds.Length;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)seqLen * VocabSize * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int r = 0; r < seqLen; r++)
            {
                var row = new Span<float>(dst + (long)r * VocabSize, VocabSize);
                row.Fill(-10f);
                row[(tokenIds[r] + 1) % VocabSize] = 10f;
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

            return new UnmanagedTensor(new TensorShape(seqLen, VocabSize), DType.Float32, deviceId, ptr);
        }
    }
}
