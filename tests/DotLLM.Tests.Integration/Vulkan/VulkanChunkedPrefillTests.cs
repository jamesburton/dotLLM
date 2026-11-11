using DotLLM.Core.Configuration;
using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1723: the dense Vulkan host prefills a long prompt in engine chunks by default, so progress moves and cancellation is
/// observed between chunks. Chunking must not change the text: greedy output of the default (chunked) run is compared with
/// the single-pass run (<c>DOTLLM_VULKAN_PREFILL_CHUNK=0</c>) on a real model and real text.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public class VulkanChunkedPrefillTests
{
    private readonly ITestOutputHelper _out;
    public VulkanChunkedPrefillTests(ITestOutputHelper output) => _out = output;

    private const int PromptTokens = 1700;   // > 3 default chunks of 512, small enough to stay far inside the watchdog

    private sealed class Loaded : IDisposable
    {
        public GgufFile Gguf = null!;
        public VulkanTransformerModel Vk = null!;
        public DotLLM.Tokenizers.ITokenizer Tok = null!;
        public string PromptText = null!;
        public void Dispose() { Vk.Dispose(); Gguf.Dispose(); }
    }

    /// <summary>Loads the model with <c>DOTLLM_VULKAN_PREFILL_CHUNK</c> set to <paramref name="envChunk"/> (null = unset) while the model resolves its chunk.</summary>
    private static Loaded Load(string? envChunk)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        FixtureLocation fx = TestFixtureResolver.ResolveFile("DOTLLM_QWEN3_4B_Q4_K_M_GGUF", "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Qwen3-4B Q4_K_M GGUF"));
        var l = new Loaded { Gguf = GgufFile.Open(fx.Path!) };
        var cfg = GgufModelConfigExtractor.Extract(l.Gguf.Metadata);
        l.Vk = VulkanTransformerModel.LoadFromGguf(l.Gguf, cfg, VulkanModelLoader.ResolveSpvDir());
        string? prev = Environment.GetEnvironmentVariable(VulkanPrefillChunk.EnvVar);
        Environment.SetEnvironmentVariable(VulkanPrefillChunk.EnvVar, envChunk);
        try { _ = l.Vk.PreferredPrefillChunkTokens; }   // resolved once, on first read
        finally { Environment.SetEnvironmentVariable(VulkanPrefillChunk.EnvVar, prev); }
        l.Tok = GgufBpeTokenizerFactory.Load(l.Gguf.Metadata);
        l.PromptText = RealText(l.Tok);
        return l;
    }

    /// <summary>~<see cref="PromptTokens"/> tokens of real text: wikitext-2 when present, else repeated prose.</summary>
    private static string RealText(DotLLM.Tokenizers.ITokenizer tok)
    {
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        string text = File.Exists(corpus)
            ? File.ReadAllText(corpus)[..200_000]
            : string.Concat(Enumerable.Repeat("The history of the river valley is recorded in several chronicles, each of which disagrees about the founding of the town. ", 400));
        int[] ids = tok.Encode(text);
        Skip.If(ids.Length < PromptTokens, "corpus too short");
        return tok.Decode(ids.AsSpan(0, PromptTokens));
    }

    private static string Run(Loaded l, int newTokens, CancellationToken ct = default)
    {
        var gen = new TextGenerator(l.Vk, l.Tok, kvCacheFactory: (_, n) => l.Vk.CreateKvCache(n));
        return gen.Generate(l.PromptText, new InferenceOptions { Temperature = 0f, MaxTokens = newTokens }, cancellationToken: ct).Text;
    }

    [SkippableFact]
    public void DenseHost_ReportsAPerDeviceChunk_AndEnvZeroTurnsItOff()
    {
        using var def = Load(null);
        int chunk = def.Vk.PreferredPrefillChunkTokens;
        _out.WriteLine($"[1723] device type {def.Vk.Device.PhysicalDeviceTypeValue}, default chunk {chunk}");
        Assert.Equal(VulkanPrefillChunk.ForDeviceType(def.Vk.Device.PhysicalDeviceTypeValue), chunk);
        Assert.True(chunk > 0);
        using var off = Load("0");
        Assert.Equal(0, off.Vk.PreferredPrefillChunkTokens);
    }

    [SkippableFact]
    public void ChunkedPrefill_IsGreedyIdentical_ToSinglePass()
    {
        string chunked, single;
        using (var l = Load(null)) chunked = Run(l, 48);
        using (var l = Load("0")) single = Run(l, 48);
        _out.WriteLine($"[1723] chunked : {chunked.ReplaceLineEndings(" ")}");
        _out.WriteLine($"[1723] single  : {single.ReplaceLineEndings(" ")}");
        Assert.False(string.IsNullOrWhiteSpace(chunked));
        Assert.Equal(single, chunked);
    }

    /// <summary>Cancelling after the first chunk stops the prefill there: the model sees ONE forward, not the whole prompt.</summary>
    [SkippableFact]
    public void CancellationBetweenChunks_StopsThePrefillAfterTheFirstChunk()
    {
        using var l = Load(null);
        using var cts = new CancellationTokenSource();
        var spy = new SpyModel(l.Vk) { AfterForward = () => cts.Cancel() };
        var gen = new TextGenerator(spy, l.Tok, kvCacheFactory: (_, n) => l.Vk.CreateKvCache(n));
        Assert.ThrowsAny<OperationCanceledException>(() =>
            gen.Generate(l.PromptText, new InferenceOptions { Temperature = 0f, MaxTokens = 8 }, cancellationToken: cts.Token));
        _out.WriteLine($"[1723] forward row counts before cancel: [{string.Join(",", spy.Rows)}]");
        Assert.Equal([l.Vk.PreferredPrefillChunkTokens], spy.Rows);
    }

    private sealed class SpyModel(IModel inner) : IModel
    {
        public List<int> Rows { get; } = [];
        public Action? AfterForward { get; set; }
        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
        public int PreferredPrefillChunkTokens => inner.PreferredPrefillChunkTokens;
        public void Dispose() { }

        private ITensor Note(ReadOnlySpan<int> t, ITensor r) { Rows.Add(t.Length); AfterForward?.Invoke(); return r; }
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Note(t, inner.Forward(t, p, d));
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => Note(t, inner.Forward(t, p, d, kv));
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => Note(t, inner.Forward(t, p, d, kv, last));
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => Note(t, inner.Forward(t, p, d, kv, a));
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => Note(t, inner.Forward(t, p, d, kv, a, m));
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => Note(t, inner.Forward(t, p, d, kv, a, m, last));
    }
}
