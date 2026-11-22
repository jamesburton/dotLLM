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

/// <summary>Backend a <see cref="GenerationSessionFixture"/> loads its synthetic model on.</summary>
internal enum SessionBackend { Cpu, Vulkan, Cuda }

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

    public static GenerationSessionFixture Create(string family, int? headDim = null, SessionBackend backend = SessionBackend.Cpu)
        => new(family, headDim, backend);

    /// <summary>
    /// Logit tolerance (absolute) versus a fresh-prefill oracle on the same backend: CPU 1e-4; GPU 1e-2 (#1919). The GPU figure is
    /// measured, not assumed: the Vulkan Llama path (F16 KV, batched prefill vs one-row decode kernels) differs by up to 5.6e-3 on
    /// logits between a rolled-back or appended session and a fresh prefill, with identical tokens, so the 5e-4 the plan guessed is
    /// below that backend's own prefill-vs-decode noise. A dropped rollback or restore moves logits by orders of magnitude more
    /// (the mutants in .docs/progress-expert-integ.md).
    /// </summary>
    public float LogitTolerance => Backend == SessionBackend.Cpu ? 1e-4f : 1e-2f;

    /// <summary>Relative part of the tolerance (unused: 0).</summary>
    public float LogitRelativeTolerance => 0f;

    public SessionBackend Backend { get; }

    private readonly Func<int, Core.Attention.IKvCache>? _gpuKvFactory;

    private GenerationSessionFixture(string family, int? headDim, SessionBackend backend)
    {
        Backend = backend;
        _dir = Path.Combine(Path.GetTempPath(), $"dotllm-session-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "m.gguf");
        switch (family)
        {
            case Llama:
                File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama", byteLevelVocab: true, headDim: headDim));
                break;
            case HybridDense:
                SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: false, contextLength: 256, headDim: headDim, ropeDim: headDim is null ? null : 8);
                break;
            case MoeHybrid:
                SyntheticQwen35MoeGguf.Write(path, contextLength: 256);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(family), family, null);
        }

        _gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        switch (backend)
        {
            case SessionBackend.Cpu:
                Model = ModelLoader.CreateCpuModelFromGguf(_gguf, config);
                break;
            case SessionBackend.Vulkan:
                (Model, _gpuKvFactory) = DotLLM.Vulkan.VulkanModelLoader.CreateOnDeviceFromGguf(_gguf, config, physicalIndex: null);
                break;
            case SessionBackend.Cuda:
                (Model, _gpuKvFactory) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(_gguf, config);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(backend));
        }
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
        if (Model is IResidualModifierHost host) host.ResidualModifier = Counter;
    }

    /// <summary>True when the model forwards through the CPU residual-modifier hook, so <see cref="Counter"/> counts real forwards.</summary>
    public bool CountsForwards => Backend == SessionBackend.Cpu;

    /// <summary>
    /// Tokenizer a constraint is built over (#1921). Llama: its byte-level tokenizer filtered to the alphabet below. The hybrid models' 12-id vocabulary has no
    /// JSON spelling, and a constraint only needs id to text, so they get a stand-in whose ids 3..7 spell <c>{ " a : }</c>
    /// (ids 0..2 are pad/bos/eos). The model is indifferent to what an id spells.
    /// </summary>
    public ITokenizer ConstraintTokenizer => _constraintTokenizer ??= new CharTokenizer(Tokenizer.VocabSize, Tokenizer is IdTokenizer ? null : Tokenizer);
    private ITokenizer? _constraintTokenizer;

    public bool IsRecurrent => Model.RequiresPerSequenceState;

    public TextGenerator NewGenerator()
        => _gpuKvFactory is null ? new(Model, Tokenizer) : new(Model, Tokenizer, kvCacheFactory: (_, size) => _gpuKvFactory(size));

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
        /// <summary>When &gt; 0: the forward whose number (1-based, since the last Reset) equals this throws at layer 1.</summary>
        public int FailOnForward;
        public bool AppliesTo(int layerIndex) => layerIndex <= 1;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount)
        {
            if (layerIndex == 0) { Forwards++; Tokens += tokenCount; }
            else if (FailOnForward > 0 && Forwards == FailOnForward)
            {
                FailOnForward = 0;
                throw new IOException("injected forward failure");
            }
        }
        public void Reset() { Forwards = 0; Tokens = 0; FailOnForward = 0; }
        public void FailNextForward() => FailOnForward = Forwards + 1;
    }

    /// <summary>Stops (including the token) after <c>n</c> generated tokens in the segment.</summary>
    internal sealed class StopAfter(int n, StopResult result = StopResult.StopInclude) : IStopCondition
    {
        public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
            => generatedTokens.Count >= n ? result : StopResult.Continue;
    }

    /// <summary>
    /// Id to text for a constraint: only the single characters <c>{ " a : }</c> keep their spelling; every other id decodes to the empty string
    /// (never allowed by a constraint, so a free-running model cannot spin on whitespace).
    /// ids 3..7 are those characters when <paramref name="inner"/> is null (the hybrid vocabularies: 8 ids for MoE).
    /// </summary>
    private sealed class CharTokenizer(int vocab, ITokenizer? inner) : ITokenizer
    {
        private const string Chars = "{\"a:}";
        public int[] Encode(string text) => inner?.Encode(text) ?? [.. text.Select(c => Chars.IndexOf(c) + 3)];
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds) sb.Append(DecodeToken(id));
            return sb.ToString();
        }
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId)
        {
            string t = inner is null
                ? (tokenId >= 3 && tokenId - 3 < Chars.Length ? Chars[tokenId - 3].ToString() : "")
                : inner.DecodeToken(tokenId);
            return t.Length == 1 && Chars.Contains(t[0]) ? t : "";
        }
        public int BosTokenId => inner?.BosTokenId ?? 1;
        public int EosTokenId => inner?.EosTokenId ?? 2;
        public int VocabSize => vocab;
        public int CountTokens(string text) => text.Length;
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
