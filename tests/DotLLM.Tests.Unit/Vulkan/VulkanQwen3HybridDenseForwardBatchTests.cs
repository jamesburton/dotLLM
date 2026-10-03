using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// <c>ForwardBatch</c> on the Vulkan dense hybrid GDN model (Tev1-4B). Today it loops the single-sequence
/// <c>Forward</c> with each request's own <see cref="VulkanGdnStateCache"/> + KV cache; these tests pin that contract so
/// a fused batched implementation (stacked GEMMs, per-seq scan dispatch) has a serial oracle to be compared against.
/// </summary>
/// <remarks>Skips cleanly when the Tev1-4B GGUF is absent (<c>DOTLLM_TEV1_4B_GGUF</c> or the HF hub cache).</remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseForwardBatchTests
{
    private const int Steps = 4;

    private static readonly int[][] Prompts =
    [
        [760, 3766, 1414, 7701, 310, 381, 264, 47739, 466, 47739],
        [33, 1049, 369, 4222, 421, 279, 2144, 13, 999, 11, 430, 7701],
        [1, 100, 200, 300, 400],
    ];

    private static string? FindGguf()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_TEV1_4B_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string snaps = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--bartowski--togethercomputer_Tev1-4B-experimental-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (string s in Directory.EnumerateDirectories(snaps))
        {
            string[] hits = Directory.GetFiles(s, "*Q4_K_M.gguf");
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }

    private sealed class Fixture : IDisposable
    {
        public required GgufFile Gguf { get; init; }
        public required ModelConfig Config { get; init; }
        public required VulkanDevice Device { get; init; }
        public required VulkanQwen3HybridDenseTransformerModel Model { get; init; }
        public void Dispose() { Model.Dispose(); Device.Dispose(); Gguf.Dispose(); }
    }

    private static Fixture Open()
    {
        string? path = FindGguf();
        Skip.If(path is null, "Tev1-4B GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var device = VulkanDevice.Create();
        var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        return new Fixture { Gguf = gguf, Config = config, Device = device, Model = model };
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
    }

    private static int Argmax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>One sequence's live state while stepping it (prefill, then <see cref="Steps"/> single-token decodes).</summary>
    private sealed class Seq
    {
        public required int[] Prompt { get; init; }
        public required IKvCache Kv { get; init; }
        public required VulkanGdnStateCache Gdn { get; init; }
        public int Pos;
        public int Next = -1;
        public readonly List<float[]> Rows = [];

        public SequenceForwardRequest NextRequest()
        {
            int[] ids = Next < 0 ? Prompt : [Next];
            int[] pos = Next < 0 ? Enumerable.Range(0, Prompt.Length).ToArray() : [Pos];
            return new SequenceForwardRequest { TokenIds = ids.AsMemory(), Positions = pos.AsMemory(), KvCache = Kv, GdnState = Gdn };
        }

        public void Accept(float[] row, int consumed)
        {
            Rows.Add(row);
            Pos += consumed;
            Next = Argmax(row);
        }
    }

    private static Seq NewSeq(Fixture f, int[] prompt, VulkanGdnStateCache gdn) => new()
    {
        Prompt = prompt,
        Kv = f.Model.CreateKvCache(prompt.Length + Steps + 4),
        Gdn = gdn,
    };

    /// <summary>Serial oracle: each sequence run to completion alone, through plain <c>Forward</c>.</summary>
    private static List<float[]>[] RunSerial(Fixture f)
    {
        var all = new List<float[]>[Prompts.Length];
        for (int s = 0; s < Prompts.Length; s++)
        {
            using var gdn = f.Model.CreateGdnStateCache();
            var seq = NewSeq(f, Prompts[s], gdn);
            try
            {
                for (int step = 0; step <= Steps; step++)
                {
                    var r = seq.NextRequest();
                    using ITensor logits = f.Model.Forward(r.TokenIds.Span, r.Positions.Span, -1, r.KvCache, r.GdnState, mtpState: null);
                    seq.Accept(LastRow(logits, f.Config.VocabSize), r.TokenIds.Length);
                }
            }
            finally { (seq.Kv as IDisposable)?.Dispose(); }
            all[s] = seq.Rows;
        }
        return all;
    }

    /// <summary>Batched run: every step issues ONE <c>ForwardBatch</c> over all sequences, interleaving their states.</summary>
    private static List<float[]>[] RunBatched(Fixture f, bool shareGdn)
    {
        var slots = new List<VulkanGdnStateCache>();
        VulkanGdnStateCache? shared = shareGdn ? f.Model.CreateGdnStateCache() : null;
        if (shared is not null) slots.Add(shared);
        var seqs = Prompts.Select(p =>
        {
            var gdn = shared ?? f.Model.CreateGdnStateCache();
            if (shared is null) slots.Add(gdn);
            return NewSeq(f, p, gdn);
        }).ToArray();
        try
        {
            for (int step = 0; step <= Steps; step++)
            {
                var requests = seqs.Select(s => s.NextRequest()).ToArray();
                var results = f.Model.ForwardBatch(requests, deviceId: -1);
                try
                {
                    for (int s = 0; s < seqs.Length; s++)
                        seqs[s].Accept(LastRow(results[s], f.Config.VocabSize), requests[s].TokenIds.Length);
                }
                finally { foreach (var t in results) t.Dispose(); }
            }
            return seqs.Select(s => s.Rows).ToArray();
        }
        finally
        {
            foreach (var s in seqs) (s.Kv as IDisposable)?.Dispose();
            foreach (var g in slots) g.Dispose();
        }
    }

    [SkippableFact]
    public void ForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        using var f = Open();
        Assert.Empty(f.Model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), deviceId: -1));
    }

    [SkippableFact]
    public void ForwardBatch_MultiSeq_NullGdnState_Throws()
    {
        using var f = Open();
        int[] ids = [1, 2, 3];
        int[] pos = [0, 1, 2];
        var req = new SequenceForwardRequest { TokenIds = ids.AsMemory(), Positions = pos.AsMemory(), KvCache = null! };
        var ex = Assert.Throws<ArgumentException>(() => f.Model.ForwardBatch([req, req], deviceId: -1));
        Assert.Contains("GdnState", ex.Message, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void ForwardBatch_InterleavedSequences_MatchSerialOracle()
    {
        using var f = Open();
        var serial = RunSerial(f);
        var batched = RunBatched(f, shareGdn: false);

        double worst = 0;
        for (int s = 0; s < Prompts.Length; s++)
        {
            Assert.Equal(serial[s].Count, batched[s].Count);
            for (int step = 0; step < serial[s].Count; step++)
            {
                Assert.Equal(Argmax(serial[s][step]), Argmax(batched[s][step]));
                for (int i = 0; i < serial[s][step].Length; i++)
                    worst = Math.Max(worst, Math.Abs(serial[s][step][i] - batched[s][step][i]));
            }
        }
        // Same kernels, same per-seq state, same order of ops per sequence: expected bit-identical.
        Assert.True(worst == 0, $"batched vs serial logits differ: max |diff| = {worst:E3}");
    }

    /// <summary>
    /// Sensitivity control: if every sequence shares ONE GDN slot, interleaving must corrupt the recurrent state and the
    /// logits must move. Without this, the identity test above could pass for a ForwardBatch that ignores per-seq state.
    /// </summary>
    [SkippableFact]
    public void ForwardBatch_SharedGdnSlot_Mutant_Diverges()
    {
        using var f = Open();
        var serial = RunSerial(f);
        var mutant = RunBatched(f, shareGdn: true);

        double worst = 0;
        for (int s = 0; s < Prompts.Length; s++)
            for (int step = 0; step < serial[s].Count; step++)
                for (int i = 0; i < serial[s][step].Length; i++)
                    worst = Math.Max(worst, Math.Abs(serial[s][step][i] - mutant[s][step][i]));
        Assert.True(worst > 1e-3, $"shared-slot mutant did not diverge (max |diff| = {worst:E3}): the identity test is insensitive");
    }
}
