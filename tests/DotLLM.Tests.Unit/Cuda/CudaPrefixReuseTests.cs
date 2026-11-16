using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1594: on CUDA (<c>serve --no-paged</c>) a later request adopts the previous request's device KV cache
/// (<see cref="CudaKvCache"/> or the q8_0 <see cref="CudaQuantizedKvCache"/>), cuts it back to the common prompt prefix, grows it
/// when the next turn is longer (headroom 0, so turn 2 always outgrows turn 1's allocation) and prefills only the suffix.
/// Three arms per KV type:
/// <list type="bullet">
/// <item>warm-with-growth vs warm-without-growth (a cache pre-sized by headroom): the same kernels on the same rows, so every
/// generated token AND every logprob must be bit-identical. This is the exact oracle for the growth copy and for the decode
/// graph recapture on the grown buffers.</item>
/// <item>warm vs cold prefill: the suffix prefill at a non-zero offset runs other attention kernels than one long cold prefill,
/// so tokens must match and the first-token logprob must agree closely, not bit-exactly.</item>
/// <item>growth refused by admission: a miss (0 cached tokens), still the cold result.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaPrefixReuseTests
{
    private readonly ITestOutputHelper _out;
    public CudaPrefixReuseTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;

    private const string Filler =
        "The quick brown fox jumps over the lazy dog while the committee reviews the quarterly figures in detail. ";

    private static string Turn1() =>
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + string.Concat(Enumerable.Repeat(Filler, 24)) + "Question 1: summarise that in one sentence.<|im_end|>\n<|im_start|>assistant\n";

    private static string Turn2(string reply) =>
        Turn1() + reply + "<|im_end|>\n<|im_start|>user\n" + string.Concat(Enumerable.Repeat(Filler, 6))
        + "Question 2: now name one colour.<|im_end|>\n<|im_start|>assistant\n";

    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void SecondTurn_ReusesPrefix_GrowsCache_AndMatchesColdAndUngrownPaths(string kvType)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");

        // The decode graph only engages below its depth ceiling (a perf limit, 512 by default); turn 2 decodes past it, so lift it
        // here or the graph-recapture-after-growth path under test never runs.
        int savedDepth = CudaTransformerModel.GraphCaptureMaxDepth;
        CudaTransformerModel.GraphCaptureMaxDepth = 8192;
        try { RunTwoTurn(kvType, ptxDir!, path!); }
        finally { CudaTransformerModel.GraphCaptureMaxDepth = savedDepth; }
    }

    private void RunTwoTurn(string kvType, string ptxDir, string path)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
        var kvConfig = kvType == "q8_0" ? new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0) : new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);
        Func<Core.Models.ModelConfig, int, Core.Attention.IKvCache> factory = (_, size) => model.CreateKvCache(size, kvConfig);
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 16, Logprobs = true };

        var cold = new TextGenerator(model, tokenizer, factory, mtpEnabled: false);
        var turn1 = cold.Generate(Turn1(), opts);
        string t2 = Turn2(turn1.Text);
        var expected = cold.Generate(t2, opts);
        // The retained cache holds turn 1's prompt AND its forwarded reply rows (all but the last sampled token), so the hit can
        // extend into the reply when turn 2 re-tokenizes it identically.
        int[] stored = tokenizer.Encode(Turn1()).Concat(turn1.GeneratedTokenIds[..^1]).ToArray();
        int common = tokenizer.Encode(t2).AsSpan().CommonPrefixLength(stored);

        // Warm, headroom 0: turn 2 outgrows turn 1's (prompt + 16) allocation, so the hit must grow the cache.
        using var growCache = new PrefixCache(2);
        var grown = new TextGenerator(model, tokenizer, factory, growCache, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
        Assert.Equal(turn1.GeneratedTokenIds, grown.Generate(Turn1(), opts).GeneratedTokenIds);
        long offsetRunsBefore = CudaG3Attention.OffsetRunCount;
        long growthsBefore = CudaKvGrowth.GrowthCount;
        int capturesBefore = model.GraphCaptureCount, replaysBefore = model.GraphReplayCount;
        // Poison new device allocations (NaN bytes) while the growth runs: CUDA hands back freed VRAM unchanged, and an earlier
        // cache of this same prompt left identical K/V rows there, so a growth that skipped rows passed without it (mutant m1).
        DotLLM.Cuda.Interop.CudaDriverApi.TestPoisonAllocations = true;
        InferenceResponse actual;
        try { actual = grown.Generate(t2, opts); }
        finally { DotLLM.Cuda.Interop.CudaDriverApi.TestPoisonAllocations = false; }
        long growths = CudaKvGrowth.GrowthCount - growthsBefore;
        _out.WriteLine($"{kvType}: turn2 prompt {actual.PromptTokenCount}, cached {actual.Timings.CachedTokenCount} (common {common}), " +
                       $"growths {growths}, graph captures +{model.GraphCaptureCount - capturesBefore}, replays +{model.GraphReplayCount - replaysBefore}");

        Assert.True(actual.Timings.CachedTokenCount > 0, "expected a prefix hit");
        Assert.True(actual.Timings.CachedTokenCount <= common, "cached more rows than the prompts share");
        Assert.True(growths >= 1, "turn 2 should have grown the retained cache");
        if (kvType == "f16" && CudaG3Attention.Enabled)   // GeForce Ampere default: the F16 suffix prefill must take the G3 offset path
            Assert.True(CudaG3Attention.OffsetRunCount > offsetRunsBefore, "the reused prefill did not run G3 with a position offset");
        if (kvType == "f16" && model.UseGraphCapture)
        {
            // Decode after the growth must recapture against the new buffers and then replay (not silently fall back to eager).
            Assert.True(model.GraphCaptureCount > capturesBefore, "no graph recapture after the growth");
            Assert.True(model.GraphReplayCount > replaysBefore, "decode after the reused prefill did not replay a graph");
        }

        // vs cold: same tokens, close first-token logprob (different prefill attention kernels; not bit-exact).
        Assert.Equal(expected.GeneratedTokenIds, actual.GeneratedTokenIds);
        double dCold = Math.Abs(expected.Logprobs![0].Logprob - actual.Logprobs![0].Logprob);
        {
            // Diagnostic: the same warm turn with the suffix attention on attention_f16 instead of the G3 offset path.
            using var diagCache = new PrefixCache(2);
            var diag = new TextGenerator(model, tokenizer, factory, diagCache, mtpEnabled: false, prefixCacheHeadroomTokens: 4096);
            diag.Generate(Turn1(), opts);
            CudaG3Attention.OffsetEnabled = false;
            try
            {
                var d = diag.Generate(t2, opts);
                _out.WriteLine($"{kvType}: warm logprob with attention_f16 suffix {d.Logprobs![0].Logprob:R} (cached {d.Timings.CachedTokenCount})");
            }
            finally { CudaG3Attention.OffsetEnabled = true; }
        }
        _out.WriteLine($"{kvType}: first-token logprob cold {expected.Logprobs[0].Logprob:R} warm {actual.Logprobs[0].Logprob:R} (|d| {dCold:E2})");

        // vs warm-without-growth: a big headroom sizes turn 1's cache for turn 2, so the same rollback + suffix prefill runs on an
        // ungrown cache. Every token and logprob must be bit-identical: the growth copied every live row and nothing else changed.
        using var roomyCache = new PrefixCache(2);
        var roomy = new TextGenerator(model, tokenizer, factory, roomyCache, mtpEnabled: false, prefixCacheHeadroomTokens: 4096);
        roomy.Generate(Turn1(), opts);
        long g0 = CudaKvGrowth.GrowthCount;
        var ungrown = roomy.Generate(t2, opts);
        Assert.Equal(g0, CudaKvGrowth.GrowthCount);
        Assert.Equal(actual.Timings.CachedTokenCount, ungrown.Timings.CachedTokenCount);
        Assert.Equal(ungrown.GeneratedTokenIds, actual.GeneratedTokenIds);
        for (int i = 0; i < ungrown.Logprobs!.Length; i++)
            Assert.Equal(ungrown.Logprobs[i].Logprob, actual.Logprobs[i].Logprob);
        _out.WriteLine($"{kvType}: ungrown warm logprob {ungrown.Logprobs[0].Logprob:R}");
        Assert.True(dCold < 0.05, $"first-token logprob moved {dCold} vs cold");

        // Refused growth (no room in VRAM): the hit misses and re-prefills; the result is still the cold one.
        using var refuseCache = new PrefixCache(2);
        var refused = new TextGenerator(model, tokenizer, factory, refuseCache, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
        refused.Generate(Turn1(), opts);
        CudaKvGrowth.AdmissionOverride = _ => false;
        try
        {
            var miss = refused.Generate(t2, opts);
            Assert.Equal(0, miss.Timings.CachedTokenCount);
            Assert.Equal(expected.GeneratedTokenIds, miss.GeneratedTokenIds);
        }
        finally
        {
            CudaKvGrowth.AdmissionOverride = null;
        }
    }

    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void Seed_CopiesLeadingRows_AndGrowth_KeepsThem(string kvType)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir!);
        var kvConfig = kvType == "q8_0" ? new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0) : new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);

        int[] tokens = tokenizer.Encode(Turn1());
        int n = tokens.Length, cut = n / 2;
        var positions = Enumerable.Range(0, n).ToArray();
        float[] Suffix(Core.Attention.IKvCache kv)
        {
            using var t = model.Forward(tokens.AsSpan(cut), positions.AsSpan(cut), 0, kv);
            return LastRow(t, config.VocabSize);
        }

        // Reference: prefill all, cut to `cut`, re-prefill the suffix.
        using var a = model.CreateKvCache(n, kvConfig);
        model.Forward(tokens, positions, 0, a).Dispose();
        a.Rollback(cut);
        float[] reference = Suffix(a);

        // Seeded from a filled cache, then the suffix: the same rows, so bit-identical.
        using var src = model.CreateKvCache(n, kvConfig);
        model.Forward(tokens, positions, 0, src).Dispose();
        using var seeded = model.CreateKvCache(n, kvConfig);
        Assert.True(((Core.Attention.IKvPrefixSeedable)seeded).TrySeedPrefixFrom(src, cut));
        Assert.Equal(cut, seeded.CurrentLength);
        Assert.Equal(reference, Suffix(seeded));

        // Grown from an exact-size cache holding `cut` rows, then the suffix: bit-identical too.
        using var small = model.CreateKvCache(cut, kvConfig);
        Assert.True(((Core.Attention.IKvPrefixSeedable)small).TrySeedPrefixFrom(src, cut));
        DotLLM.Cuda.Interop.CudaDriverApi.TestPoisonAllocations = true;   // see RunTwoTurn: freed VRAM still holds this prompt's rows
        try { Assert.True(small.TryGrowTo(n)); }
        finally { DotLLM.Cuda.Interop.CudaDriverApi.TestPoisonAllocations = false; }
        Assert.Equal(n, small.MaxLength);
        Assert.Equal(reference, Suffix(small));
    }

    private static unsafe float[] LastRow(Core.Tensors.ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }
}
