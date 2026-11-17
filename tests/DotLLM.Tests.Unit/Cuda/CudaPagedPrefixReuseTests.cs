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
/// Issue #1843: prompt-prefix reuse on the CUDA paged KV cache (<c>serve --paged</c>), the paged side of #1594. A later request
/// adopts the previous request's <see cref="CudaPagedKvCache"/>, cuts it back to the common prefix, grows it when the next turn is
/// longer and prefills only the suffix (G3's offset path over the gather scratch). On Qwen2.5-0.5B q8_0 weights:
/// <list type="bullet">
/// <item>warm-with-growth vs warm-without-growth: the same kernels on the same blocks, so tokens AND logprobs are bit-identical.</item>
/// <item>warm vs cold: same tokens, first-token logprob close (the suffix runs other attention kernels than one long prefill).</item>
/// <item>a block-sharing seed gives the same suffix logits as the cut-back cache, and copy-on-write keeps the source intact.</item>
/// <item>on a pool too small for everything, a hit evicts the other retained entry instead of missing or exhausting the pool.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaPagedPrefixReuseTests
{
    private const int BlockSize = 16;
    private readonly ITestOutputHelper _out;
    public CudaPagedPrefixReuseTests(ITestOutputHelper output) => _out = output;

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

    private static string Unrelated() =>
        "<|im_start|>system\nYou are a terse assistant.<|im_end|>\n<|im_start|>user\n"
        + string.Concat(Enumerable.Repeat("Seven sailors sorted silver spoons beside the harbour wall at dusk. ", 22))
        + "Count the sailors.<|im_end|>\n<|im_start|>assistant\n";

    private sealed record Loaded(GgufFile Gguf, CudaTransformerModel Model, DotLLM.Tokenizers.ITokenizer Tokenizer) : IDisposable
    {
        public void Dispose()
        {
            Model.Dispose();
            Gguf.Dispose();
        }
    }

    private static Loaded? Load()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return new Loaded(gguf, CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir!), tokenizer);
    }

    [SkippableFact]
    public void SecondTurn_ReusesPrefix_GrowsCache_AndMatchesColdAndUngrownPaths()
    {
        using var env = Load()!;
        var (model, tokenizer) = (env.Model, env.Tokenizer);
        using var pool = model.CreateKvBlockPool(BlockSize, 8192);
        Func<Core.Models.ModelConfig, int, Core.Attention.IKvCache> paged = (_, size) => model.CreatePagedKvCache(pool, size);
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 16, Logprobs = true };
        int freeAtStart = pool.FreeBlocks;

        var cold = new TextGenerator(model, tokenizer, paged, mtpEnabled: false);
        var turn1 = cold.Generate(Turn1(), opts);
        string t2 = Turn2(turn1.Text);
        var expected = cold.Generate(t2, opts);
        Assert.Equal(freeAtStart, pool.FreeBlocks);   // no prefix cache: every request's blocks went back

        // Warm, headroom 0: turn 2 outgrows turn 1's (prompt + 16) cache, so the hit must grow it.
        InferenceResponse actual;
        long growths, offsetRuns;
        using (var growCache = new PrefixCache(2))
        {
            var grown = new TextGenerator(model, tokenizer, paged, growCache, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
            Assert.Equal(turn1.GeneratedTokenIds, grown.Generate(Turn1(), opts).GeneratedTokenIds);
            Assert.Equal(1, growCache.EntryCount);   // the paged cache is retained now (it was not before #1843)
            long g0 = CudaKvGrowth.GrowthCount, o0 = CudaG3Attention.OffsetRunCount;
            // NaN-poison new allocations while the growth runs: the regrown gather scratch must be rebuilt from the blocks.
            DotLLM.Cuda.Interop.CudaDriverApi.TestPoisonAllocations = true;
            try { actual = grown.Generate(t2, opts); }
            finally { DotLLM.Cuda.Interop.CudaDriverApi.TestPoisonAllocations = false; }
            growths = CudaKvGrowth.GrowthCount - g0;
            offsetRuns = CudaG3Attention.OffsetRunCount - o0;
        }
        Assert.Equal(freeAtStart, pool.FreeBlocks);   // disposing the prefix cache returned every retained block
        _out.WriteLine($"turn2 prompt {actual.PromptTokenCount}, cached {actual.Timings.CachedTokenCount}, growths {growths}, " +
                       $"G3 enabled {CudaG3Attention.Enabled}, offset runs {offsetRuns}");
        Assert.True(actual.Timings.CachedTokenCount > 0, "expected a prefix hit");
        Assert.True(growths >= 1, "turn 2 should have grown the retained paged cache");
        if (CudaG3Attention.Enabled)
            Assert.True(offsetRuns > 0, "the reused paged prefill did not run G3 with a position offset");

        Assert.Equal(expected.GeneratedTokenIds, actual.GeneratedTokenIds);
        double dCold = Math.Abs(expected.Logprobs![0].Logprob - actual.Logprobs![0].Logprob);
        _out.WriteLine($"first-token logprob cold {expected.Logprobs[0].Logprob:R} warm {actual.Logprobs[0].Logprob:R} (|d| {dCold:E2})");

        // Warm without growth (a headroom sizes turn 1's cache for turn 2): bit-identical tokens and logprobs.
        using (var roomyCache = new PrefixCache(2))
        {
            var roomy = new TextGenerator(model, tokenizer, paged, roomyCache, mtpEnabled: false, prefixCacheHeadroomTokens: 4096);
            roomy.Generate(Turn1(), opts);
            long g0 = CudaKvGrowth.GrowthCount;
            var ungrown = roomy.Generate(t2, opts);
            Assert.Equal(g0, CudaKvGrowth.GrowthCount);
            Assert.Equal(actual.Timings.CachedTokenCount, ungrown.Timings.CachedTokenCount);
            Assert.Equal(ungrown.GeneratedTokenIds, actual.GeneratedTokenIds);
            for (int i = 0; i < ungrown.Logprobs!.Length; i++)
                Assert.Equal(ungrown.Logprobs[i].Logprob, actual.Logprobs[i].Logprob);
        }
        Assert.True(dCold < 0.05, $"first-token logprob moved {dCold} vs cold");

        // Refused growth (no room for the larger gather scratch): the hit misses and re-prefills; still the cold result.
        using (var refuseCache = new PrefixCache(2))
        {
            var refused = new TextGenerator(model, tokenizer, paged, refuseCache, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
            refused.Generate(Turn1(), opts);
            CudaKvGrowth.AdmissionOverride = _ => false;
            try
            {
                var miss = refused.Generate(t2, opts);
                Assert.Equal(0, miss.Timings.CachedTokenCount);
                Assert.Equal(expected.GeneratedTokenIds, miss.GeneratedTokenIds);
            }
            finally { CudaKvGrowth.AdmissionOverride = null; }
        }
        Assert.Equal(freeAtStart, pool.FreeBlocks);
    }

    [SkippableFact]
    public void Seed_SharesBlocks_GivesTheCutBackLogits_AndCopyOnWriteKeepsTheSource()
    {
        using var env = Load()!;
        var (model, tokenizer) = (env.Model, env.Tokenizer);
        int vocab = model.Config.VocabSize;
        using var pool = model.CreateKvBlockPool(BlockSize, 8192);
        int freeAtStart = pool.FreeBlocks;

        int[] tokens = tokenizer.Encode(Turn1());
        int n = tokens.Length;
        int cut = n / 2 / BlockSize * BlockSize + 5;   // mid-block, so the seed shares a PARTIAL block that the suffix writes into
        var positions = Enumerable.Range(0, n).ToArray();
        float[] Suffix(Core.Attention.IKvCache kv)
        {
            using var t = model.Forward(tokens.AsSpan(cut), positions.AsSpan(cut), 0, kv);
            return LastRow(t, vocab);
        }

        // Reference: prefill all, cut back, re-prefill the suffix.
        float[] reference;
        using (var a = model.CreatePagedKvCache(pool, n))
        {
            model.Forward(tokens, positions, 0, a).Dispose();
            a.Rollback(cut);
            reference = Suffix(a);
        }

        using var src = model.CreatePagedKvCache(pool, n);
        model.Forward(tokens, positions, 0, src).Dispose();
        int freeAfterSrc = pool.FreeBlocks;
        float[] srcFull;
        using (var t = model.Forward(tokens.AsSpan(n - 1), positions.AsSpan(n - 1), 0, CloneByRollback(src, n - 1)))
            srcFull = LastRow(t, vocab);

        using (var seeded = model.CreatePagedKvCache(pool, n))
        {
            Assert.True(seeded.TrySeedPrefixFrom(src, cut));
            Assert.Equal(cut, seeded.CurrentLength);
            Assert.Equal(freeAfterSrc, pool.FreeBlocks);               // shared, not copied
            Assert.Equal(reference, Suffix(seeded));                   // the same rows: bit-identical
            Assert.True(pool.FreeBlocks < freeAfterSrc, "the suffix should have copied the shared partial block and added its own");
        }

        // The seeded cache's writes into the shared partial block went to its own copy: the source still holds its original rows.
        using (var t = model.Forward(tokens.AsSpan(n - 1), positions.AsSpan(n - 1), 0, CloneByRollback(src, n - 1)))
            Assert.Equal(srcFull, LastRow(t, vocab));

        src.Dispose();
        Assert.Equal(freeAtStart, pool.FreeBlocks);
    }

    /// <summary>Rolls <paramref name="kv"/> back to <paramref name="length"/> so the last token can be re-forwarded over it.</summary>
    private static Core.Attention.IKvCache CloneByRollback(Core.Attention.IKvCache kv, int length)
    {
        kv.Rollback(length);
        return kv;
    }

    [SkippableFact]
    public void SmallPool_HitEvictsTheOtherEntry_InsteadOfMissingOrExhaustingThePool()
    {
        using var env = Load()!;
        var (model, tokenizer) = (env.Model, env.Tokenizer);
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 16 };
        static int Blocks(int tokens) => (tokens + BlockSize - 1) / BlockSize;

        // Size the pool from the prompts: A and an unrelated B both fit and are retained, but turn 2 of A then needs more blocks
        // than are free while B is held, and fewer than the whole pool.
        int a = tokenizer.Encode(Turn1()).Length, b = tokenizer.Encode(Unrelated()).Length;
        var probe = new TextGenerator(model, tokenizer, (_, size) => model.CreateKvCache(size), mtpEnabled: false);
        var turn1 = probe.Generate(Turn1(), opts);
        var reB = probe.Generate(Unrelated(), opts);
        string t2 = Turn2(turn1.Text);
        int n = tokenizer.Encode(t2).Length;
        // Rows a retained cache holds: its prompt plus every generated token but the last (never forwarded).
        int heldA = Blocks(a + turn1.GeneratedTokenIds.Length - 1), heldB = Blocks(b + reB.GeneratedTokenIds.Length - 1);
        int total = Math.Max(Blocks(n + opts.MaxTokens), heldA + Blocks(b + opts.MaxTokens));
        _out.WriteLine($"prompts A {a}, B {b}, turn 2 {n}; held A {heldA}, B {heldB}; pool {total} blocks");
        Assert.True(Blocks(n + opts.MaxTokens) - heldA > total - heldA - heldB, "fixture: turn 2 must not fit beside B");

        using var pool = model.CreateKvBlockPool(BlockSize, total * BlockSize);
        Func<Core.Models.ModelConfig, int, Core.Attention.IKvCache> paged = (_, size) => model.CreatePagedKvCache(pool, size);
        var expected = new TextGenerator(model, tokenizer, paged, mtpEnabled: false).Generate(t2, opts);

        using var prefix = new PrefixCache(2);
        var gen = new TextGenerator(model, tokenizer, paged, prefix, mtpEnabled: false, prefixCacheHeadroomTokens: 0);
        gen.Generate(Turn1(), opts);
        gen.Generate(Unrelated(), opts);
        Assert.Equal(2, prefix.EntryCount);
        long evictions = prefix.AdmissionEvictions;

        var warm = gen.Generate(t2, opts);
        _out.WriteLine($"turn 2 cached {warm.Timings.CachedTokenCount}, admission evictions +{prefix.AdmissionEvictions - evictions}");
        Assert.True(warm.Timings.CachedTokenCount > 0, "the hit was given up");
        Assert.Equal(1, prefix.AdmissionEvictions - evictions);
        Assert.Equal(expected.GeneratedTokenIds, warm.GeneratedTokenIds);
    }

    private static unsafe float[] LastRow(Core.Tensors.ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }
}
