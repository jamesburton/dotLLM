using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Gguf;

/// <summary>
/// Issue #756 end to end on a real checkpoint: a GGUF split into <c>-0000N-of-0000M</c> shards (by
/// <see cref="GgufSplitter"/>, byte-for-byte) must load through every weight-loader path and produce
/// BIT-IDENTICAL logits and greedy text to the merged single file.
/// </summary>
/// <remarks>
/// Both split layouts are exercised: shard 1 metadata-only (the unsloth layout) and shard 1 carrying tensors
/// (the llama-gguf-split layout). The control arm is sensitive: the same comparison against a split set whose
/// tensor bytes were perturbed MUST disagree, otherwise the equality proves nothing.
/// </remarks>
[Collection(GpuCollection.Name)]
public sealed class SplitGgufLoadEndToEndTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-split-e2e-" + Guid.NewGuid().ToString("N"));
    private const int DecodeSteps = 8;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    private static string? ResolveFixture(out string skip)
    {
        FixtureLocation fx = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM2_135M_INSTRUCT_Q8_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf");
        skip = fx.Found ? "" : fx.SkipMessage("SmolLM2-135M-Instruct Q8_0 GGUF");
        return fx.Found ? fx.Path : null;
    }

    /// <summary>Prefill + greedy decode; returns the prefill last-row logits and the greedy tokens.</summary>
    private static unsafe (float[] LastRow, int[] Tokens) Run(string path, string device)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var r = DeviceModelLoader.Load(gguf, config, device, null, new ThreadingConfig(0, 0),
            DotLLM.HuggingFace.ModelResolver.FileLength(path), Path.GetFileName(path), _ => { }, _ => { });
        using var model = r.Model;
        Assert.Equal(device, r.ResolvedDevice);
        int vocab = config.VocabSize;
        using IKvCache kv = r.KvCacheFactory is { } f ? f(64) : new SimpleKvCache(KvGeometry.FromConfig(config), 64);

        int[] ids = [1, 450, 7483, 310, 3681, 338];
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        var tokens = new List<int>();
        float[] lastRow;
        int next;
        using (ITensor logits = model.Forward(ids, pos, deviceId: -1, kv))
        {
            long rows = logits.Shape.ElementCount / vocab;
            lastRow = new ReadOnlySpan<float>((float*)logits.DataPointer + (rows - 1) * vocab, vocab).ToArray();
            next = Argmax(lastRow);
        }
        tokens.Add(next);
        for (int i = 1; i < DecodeSteps; i++)
        {
            using ITensor logits = model.Forward([next], [ids.Length + i - 1], deviceId: -1, kv);
            long rows = logits.Shape.ElementCount / vocab;
            next = Argmax(new ReadOnlySpan<float>((float*)logits.DataPointer + (rows - 1) * vocab, vocab));
            tokens.Add(next);
        }
        return (lastRow, [.. tokens]);
    }

    private static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    [SkippableTheory]
    [InlineData(3, true)]
    [InlineData(4, false)]
    public void Cpu_SplitSet_MatchesMergedFile_BitForBit(int shards, bool metadataOnlyFirst)
    {
        string? src = ResolveFixture(out string skip);
        Skip.If(src is null, skip);

        string first = GgufSplitter.Split(src!, _dir, "smol", shards, metadataOnlyFirst);
        using (var g = GgufFile.Open(first))
        {
            Assert.True(g.IsSplit);
            Assert.Equal(shards, g.ShardCount);
            Assert.Equal(!metadataOnlyFirst, g.Tensors.Any(t => g.GetTensorShardIndex(t.Name) == 0));
        }

        var (mergedRow, mergedTokens) = Run(src!, "cpu");
        var (splitRow, splitTokens) = Run(first, "cpu");

        output.WriteLine($"CPU merged vs {shards}-shard split: greedy {string.Join(",", mergedTokens)} / {string.Join(",", splitTokens)}");
        Assert.Equal(mergedTokens, splitTokens);
        Assert.True(mergedRow.AsSpan().SequenceEqual(splitRow), "split-set logits are not bit-identical to the merged file");
    }

    [SkippableFact]
    public void Cpu_SplitSetWithPerturbedTensorBytes_DoesNotMatch_ControlArmIsSensitive()
    {
        string? src = ResolveFixture(out string skip);
        Skip.If(src is null, skip);

        string first = GgufSplitter.Split(src!, _dir, "smol", 3, metadataOnlyFirst: true);
        // Corrupt the last shard's tensor region (bytes well inside the data section of the LAST shard, which holds the tail layers / output norm).
        string last = GgufSplitter.ShardPath(_dir, "smol", 3, 3);
        using (var fs = new FileStream(last, FileMode.Open, FileAccess.ReadWrite))
        {
            long mid = fs.Length / 2;
            fs.Position = mid;
            var buf = new byte[4096];
            fs.ReadExactly(buf);
            for (int i = 0; i < buf.Length; i++) buf[i] ^= 0x55;
            fs.Position = mid;
            fs.Write(buf);
        }

        var (mergedRow, _) = Run(src!, "cpu");
        var (badRow, _) = Run(first, "cpu");
        Assert.False(mergedRow.AsSpan().SequenceEqual(badRow), "perturbed split set produced identical logits - the equality arm is inert");
    }

    [SkippableFact]
    public void Vulkan_SplitSet_MatchesMergedFile_BitForBit()
    {
        Skip.If(!DeviceEndpoint.Describe().Backends!.Any(b => b.Name == "vulkan" && b.Servable), "Vulkan is not servable on this machine.");
        string? src = ResolveFixture(out string skip);
        Skip.If(src is null, skip);

        string first = GgufSplitter.Split(src!, _dir, "smol", 3, metadataOnlyFirst: true);
        var (mergedRow, mergedTokens) = Run(src!, "vulkan");
        var (splitRow, splitTokens) = Run(first, "vulkan");

        output.WriteLine($"Vulkan merged vs 3-shard split: greedy {string.Join(",", mergedTokens)} / {string.Join(",", splitTokens)}");
        Assert.Equal(mergedTokens, splitTokens);
        Assert.True(mergedRow.AsSpan().SequenceEqual(splitRow), "Vulkan split-set logits are not bit-identical to the merged file");
    }
}
