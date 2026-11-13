using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1702 on a real model (Qwen2.5-0.5B q8_0, CPU): a live <see cref="PrefixCache"/> resume of a prompt that continues a
/// length-stopped reply must equal a cold run. Before the fix the resume reused the row of the reply's last sampled token,
/// which was never forwarded and held another request's K/V (first-token logprob -1.137 resumed vs -1.551 cold). Also prints
/// a cache-free split sweep (one prefill split into two forwards at row k): flat, so row-count numerics play no part here.
/// </summary>
public sealed class PrefixResumeParityTests
{
    private readonly ITestOutputHelper _out;
    public PrefixResumeParityTests(ITestOutputHelper output) => _out = output;

    private const string Sys =
        "You are the dotLLM coding agent. Project rules: file-scoped namespaces; readonly record structs for small values; " +
        "Span over arrays in signatures; no managed arrays for tensor data (NativeMemory.AlignedAlloc, 64 bytes); every " +
        "unmanaged buffer wrapped in IDisposable; XML doc comments on public APIs; nullable enabled; prove each correctness " +
        "test can fail (mutation or substitution) and record how in the PR body; benchmark kernels before and after; never " +
        "commit model weights; one issue, one branch, one PR to dev. Reply with the next concrete step and nothing else.";

    private static string Render(params (string Role, string Text)[] turns)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append("<|im_start|>system\n").Append(Sys).Append("<|im_end|>\n");
        foreach (var (role, text) in turns)
            sb.Append("<|im_start|>").Append(role).Append('\n').Append(text).Append("<|im_end|>\n");
        sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    private static unsafe (float MaxLogprob, float[] Row) LastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
        float max = float.NegativeInfinity;
        foreach (var v in row) max = Math.Max(max, v);
        double sum = 0;
        foreach (var v in row) sum += Math.Exp(v - max);
        return ((float)(-Math.Log(sum)), row);
    }

    /// <summary>Forwards <paramref name="ids"/>[from..to) as one batch into <paramref name="kv"/>.</summary>
    private static ITensor Fwd(IModel model, int[] ids, int from, int to, IKvCache kv)
    {
        var pos = Enumerable.Range(from, to - from).ToArray();
        return model.Forward(ids.AsSpan(from, to - from), pos, -1, kv);
    }

    [SkippableFact]
    [Trait("Category", "Model")]
    public void Qwen25_LiveResumeAfterLengthStoppedReply_MatchesColdRun()
    {
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;
        Skip.If(path is null, "Qwen2.5-0.5B-Instruct q8_0 GGUF not found.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        int vocab = config.VocabSize;
        var opts = new InferenceOptions { Temperature = 0f, MaxTokens = 24, Logprobs = true, TopLogprobs = 1 };

        string m1 = Render(("user", "Read src/DotLLM.Engine/PromptCache/PrefixCache.cs and tell me what the eviction rule is today. Tool result follows with many lines of output about entries and lengths and save points."));
        string s1a = Render(("user", "Subagent task: grep tests/ for every test that constructs a PrefixCache and list the files."));
        var plain = new TextGenerator(model, tokenizer);
        var s1aRes = plain.Generate(s1a, opts);
        string s1b = Render(("user", "Subagent task: grep tests/ for every test that constructs a PrefixCache and list the files."),
            ("assistant", s1aRes.Text.Trim()), ("user", "Tool result: 4 files found. Summarise the fixture shapes."));
        int[] a = tokenizer.Encode(s1a), b = tokenizer.Encode(s1b);
        var cold = plain.Generate(s1b, opts);
        _out.WriteLine($"s1a prompt {a.Length}, generated {s1aRes.GeneratedTokenCount} ({s1aRes.FinishReason}); s1b prompt {b.Length}; cold lp {cold.Logprobs![0].Logprob:F5}");

        using var cache = new PrefixCache(4);
        var gen = new TextGenerator(model, tokenizer, prefixCache: cache, prefixCacheHeadroomTokens: 2048);
        gen.Generate(m1, opts);
        var ra = gen.Generate(s1a, opts);
        var rb = gen.Generate(s1b, opts);
        _out.WriteLine($"cached: s1a hit {ra.Timings.CachedTokenCount}, s1b hit {rb.Timings.CachedTokenCount}, lp {rb.Logprobs![0].Logprob:F5}");

        // Cold split control (no prefix cache).
        int common = 0;
        int[] full = [.. a, .. s1aRes.GeneratedTokenIds];
        while (common < Math.Min(full.Length, b.Length) && full[common] == b[common]) common++;
        _out.WriteLine($"common prefix of (s1a+gen) and s1b = {common}");
        foreach (int k in new[] { 0, 1, 32, 64, 125, a.Length, common - 1, common, b.Length - 1 })
        {
            if (k < 0 || k >= b.Length) continue;
            using var kv = new SimpleKvCache(KvGeometry.FromConfig(config, model.SupportsCompactSlidingKv), b.Length + 8);
            if (k > 0) Fwd(model, b, 0, k, kv).Dispose();
            using var lg = Fwd(model, b, k, b.Length, kv);
            _out.WriteLine($"split k={k,4}: lp {LastRow(lg, vocab).MaxLogprob:F5}");
        }

        // Write-history replay: prefill s1a, decode gen[0..n-2] one at a time, then the s1b tail from `common` clamped to written rows.
        int written = a.Length + s1aRes.GeneratedTokenCount - 1;
        int resume = Math.Min(common, written);
        using (var kv = new SimpleKvCache(KvGeometry.FromConfig(config, model.SupportsCompactSlidingKv), b.Length + 8))
        {
            Fwd(model, a, 0, a.Length, kv).Dispose();
            for (int i = a.Length; i < resume; i++) Fwd(model, full, i, i + 1, kv).Dispose();
            using var lg = Fwd(model, b, resume, b.Length, kv);
            _out.WriteLine($"replay resume={resume}: lp {LastRow(lg, vocab).MaxLogprob:F5}");
        }

        Assert.Equal(FinishReason.Length, s1aRes.FinishReason);       // the reply's last token was never forwarded
        Assert.True(common > written, $"s1b matches only {common} rows of the stored sequence: the unwritten row is not exercised");
        Assert.Equal(written, rb.Timings.CachedTokenCount);
        Assert.Equal(cold.Text, rb.Text);
        Assert.Equal(cold.Logprobs![0].Logprob, rb.Logprobs![0].Logprob, 1e-4f);
    }
}
