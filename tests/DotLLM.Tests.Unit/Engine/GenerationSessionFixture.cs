using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// CPU model fixture for the <see cref="GenerationSession"/> tests (#1919): one synthetic GGUF per supported family (attention-only
/// Llama, GDN-hybrid dense, GDN-hybrid MoE), plus a read-only forward counter hung on the residual-modifier hook so a test can
/// prove how many tokens the model actually forwarded (no prefix re-prefill).
/// </summary>
internal sealed class GenerationSessionFixture : IDisposable
{
    public const string Llama = "llama", HybridDense = "hybrid-dense", MoeHybrid = "moe-hybrid";

    public static TheoryData<string> Families => new() { Llama, HybridDense, MoeHybrid };

    private readonly string _dir;
    private readonly GgufFile _gguf;

    public IModel Model { get; }
    public ITokenizer Tokenizer { get; }
    public ForwardCounter Counter { get; } = new();

    /// <summary>A prompt and an observation made only of ordinary tokens (never BOS / EOS / unk).</summary>
    public int[] Prompt { get; }
    public int[] Observation { get; }

    public static GenerationSessionFixture Create(string family) => new(family);

    private GenerationSessionFixture(string family)
    {
        _dir = Path.Combine(Path.GetTempPath(), $"dotllm-session-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "m.gguf");
        switch (family)
        {
            case Llama:
                File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama", byteLevelVocab: true));
                break;
            case HybridDense:
                SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: false, contextLength: 256);
                break;
            case MoeHybrid:
                SyntheticQwen35MoeGguf.Write(path, contextLength: 256);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(family), family, null);
        }

        _gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        Model = ModelLoader.CreateCpuModelFromGguf(_gguf, config);
        if (family == Llama)
        {
            Tokenizer = GgufBpeTokenizerFactory.Load(_gguf.Metadata);
            Prompt = [.. Enumerable.Range(0, 12).Select(i => 97 + (i * 7) % 20)];
            Observation = [.. Enumerable.Range(0, 5).Select(i => 110 + i)];
        }
        else
        {
            Tokenizer = new IdTokenizer(config.VocabSize);
            Assert.True(Model.RequiresPerSequenceState, "hybrid fixture must be a recurrent architecture");
            Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
            Observation = [6, 3, 5];
        }
        ((IResidualModifierHost)Model).ResidualModifier = Counter;
    }

    public bool IsRecurrent => Model.RequiresPerSequenceState;

    public TextGenerator NewGenerator() => new(Model, Tokenizer);

    public static CacheProvenanceTag Tag() => new("model-a", "tok-1", "f32", "mods-0", "snap-1");

    /// <summary>Greedy, EOS masked out, so only the supplied stop condition ends a segment.</summary>
    public static InferenceOptions Greedy(int maxTokens = 64) => new()
    {
        Temperature = 0f,
        MaxTokens = maxTokens,
        IgnoreEos = true,
    };

    public void Dispose()
    {
        Model.Dispose();
        _gguf.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>Counts forward calls and tokens at layer 0 without touching the residual.</summary>
    internal sealed class ForwardCounter : IResidualModifier
    {
        public int Forwards;
        public int Tokens;
        public bool AppliesTo(int layerIndex) => layerIndex == 0;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount) { Forwards++; Tokens += tokenCount; }
        public void Reset() { Forwards = 0; Tokens = 0; }
    }

    /// <summary>Stops (including the token) after <c>n</c> generated tokens in the segment.</summary>
    internal sealed class StopAfter(int n, StopResult result = StopResult.StopInclude) : IStopCondition
    {
        public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
            => generatedTokens.Count >= n ? result : StopResult.Continue;
    }

    /// <summary>Token id i decodes to "t{i}|"; no merges, never an end-of-generation marker.</summary>
    private sealed class IdTokenizer(int vocab) : ITokenizer
    {
        public int[] Encode(string text) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds) sb.Append('t').Append(id).Append('|');
            return sb.ToString();
        }
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => Decode([tokenId]);
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int VocabSize => vocab;
        public int CountTokens(string text) => throw new NotSupportedException();
    }
}
