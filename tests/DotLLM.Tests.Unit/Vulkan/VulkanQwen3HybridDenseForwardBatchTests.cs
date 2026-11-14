using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using DotLLM.Tests.Integration.Fixtures;

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
        // Longer than MaxAllRowLogitsSeqLen: only the LAST row is returned, so the fused path's head-row selection is exercised.
        [..Enumerable.Range(0, 40).Select(i => 1000 + i * 37 % 5000)],
    ];

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_TEV1_4B_GGUF", "bartowski", "togethercomputer_Tev1-4B-experimental-GGUF", "*Q4_K_M.gguf").Path;

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

        public void Accept(float[] row, int consumed, int? forcedNext = null)
        {
            Rows.Add(row);
            Pos += consumed;
            Next = forcedNext ?? Argmax(row);
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

    /// <summary>
    /// Batched run: every step issues ONE <c>ForwardBatch</c> over all sequences, interleaving their states. With
    /// <paramref name="teacher"/>, each sequence is fed the teacher's greedy token instead of its own (#1749): a near-tie at one step
    /// must not turn the rest of the comparison into one between two different token streams.
    /// </summary>
    private static List<float[]>[] RunBatched(Fixture f, bool shareGdn, List<float[]>[]? teacher = null)
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
                        seqs[s].Accept(LastRow(results[s], f.Config.VocabSize), requests[s].TokenIds.Length,
                            teacher is null ? null : Argmax(teacher[s][step]));
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
        var batched = WithSerialBatch(() => RunBatched(f, shareGdn: false));

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
        var mutant = WithSerialBatch(() => RunBatched(f, shareGdn: true));

        double worst = 0;
        for (int s = 0; s < Prompts.Length; s++)
            for (int step = 0; step < serial[s].Count; step++)
                for (int i = 0; i < serial[s][step].Length; i++)
                    worst = Math.Max(worst, Math.Abs(serial[s][step][i] - mutant[s][step][i]));
        Assert.True(worst > 1e-3, $"shared-slot mutant did not diverge (max |diff| = {worst:E3}): the identity test is insensitive");
    }

    private static T WithEnv<T>(string name, string value, Func<T> body)
    {
        string? saved = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
        try { return body(); }
        finally { Environment.SetEnvironmentVariable(name, saved); }
    }

    // Engage the stacked path even for these short batches.
    private static T WithFusedBatch<T>(Func<T> body) => WithEnv("DOTLLM_VK_BATCH_FUSE_MIN_ROWS", "1", body);

    // The serial loop: the bit-identity oracle for the tests that pin it.
    private static T WithSerialBatch<T>(Func<T> body) => WithEnv("DOTLLM_VK_BATCH_FUSE", "0", body);

    /// <summary>
    /// Runs <paramref name="body"/> with every projection GEMM on the same route whatever the row count (#1749): the small-row GEMVs
    /// and the dp4a MMQ off, so the serial prefills (5..40 rows) and the stacked one (67 rows) all take the F16 coopmat (or, on a
    /// device without it, the scalar) GEMM. Both switches are read when the model is built, so the fixture is opened inside.
    /// </summary>
    private static T WithSingleGemmRoute<T>(Func<Fixture, T> body)
    {
        bool saved = VulkanQwen3HybridDenseTransformerModel.SmallRowGemvEnabled;
        VulkanQwen3HybridDenseTransformerModel.SmallRowGemvEnabled = false;
        try
        {
            return WithEnv(VulkanQwen3MoeHybridKernels.HybridPrefillMmqEnvVar, "0", () =>
            {
                using var f = Open();
                return body(f);
            });
        }
        finally { VulkanQwen3HybridDenseTransformerModel.SmallRowGemvEnabled = saved; }
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static double MaxAbsDiff(List<float[]>[] a, List<float[]>[] b)
    {
        double worst = 0;
        for (int s = 0; s < a.Length; s++)
        {
            Assert.Equal(a[s].Count, b[s].Count);
            for (int step = 0; step < a[s].Count; step++) worst = Math.Max(worst, MaxAbsDiff(a[s][step], b[s][step]));
        }
        return worst;
    }

    /// <summary>Bound for the strict same-route gate: measured 7.2e-3 on the 8060S (GEMV at n = 1 vs coopmat at n = 4 on the decode steps).</summary>
    private const double SameRouteBound = 0.05;

    /// <summary>
    /// Bound for the default-route check. The stacked prefill takes the F16 coopmat GEMM while a short serial prefill takes the
    /// small-row GEMVs, so the two differ by route, not by a bug (#1749). Measured max |diff|: 0.22 on the 8060S, 0.31 on Arc, 0.18 on
    /// the RTX 3060 Vulkan; against the CPU oracle both routes sit 0.22-0.68 away, with neither consistently closer.
    /// </summary>
    private const double DefaultRouteBound = 0.5;

    /// <summary>
    /// #687 / #1749, the discriminating gate: with every GEMM on one route, the stacked forward (FFN GEMMs over the rows of every
    /// sequence, token mixing per sequence) must reproduce the serial oracle. The full-history prefills (prompts up to the all-row
    /// logits length) are bit-identical; the rest differ only by the decode GEMV-vs-GEMM route, within <see cref="SameRouteBound"/>,
    /// with the same greedy token. Teacher-forced on the serial tokens.
    /// </summary>
    [SkippableFact]
    public void ForwardBatchFused_SameGemmRoute_MatchesSerialOracle()
    {
        WithSingleGemmRoute(f =>
        {
            var serial = RunSerial(f);
            var fused = WithFusedBatch(() => RunBatched(f, shareGdn: false, teacher: serial));
            for (int s = 0; s < Prompts.Length; s++)
            {
                Assert.Equal(serial[s].Count, fused[s].Count);
                for (int step = 0; step < serial[s].Count; step++)
                    Assert.Equal(Argmax(serial[s][step]), Argmax(fused[s][step]));
            }
            for (int s = 0; s < 3; s++)
            {
                double prefill = MaxAbsDiff(serial[s][0], fused[s][0]);
                Assert.True(prefill == 0, $"seq {s}: same-route stacked prefill is not bit-identical to the serial one: max |diff| = {prefill:E3}");
            }
            double worst = MaxAbsDiff(serial, fused);
            Assert.True(worst < SameRouteBound, $"same-route fused vs serial logits differ: max |diff| = {worst:E3}");
            return 0;
        });
    }

    /// <summary>
    /// #687 on the default routes: teacher-forced on the serial tokens, the stacked forward stays within
    /// <see cref="DefaultRouteBound"/> of the serial oracle. No greedy-token assertion here: with every logit within the bound, a flip
    /// can only occur where the top two are within twice the bound, which is a near-tie on either route (seq 2's prefill: margin 0.21).
    /// </summary>
    [SkippableFact]
    public void ForwardBatchFused_MatchesSerialOracle()
    {
        using var f = Open();
        var serial = RunSerial(f);
        var fused = WithFusedBatch(() => RunBatched(f, shareGdn: false, teacher: serial));
        double worst = MaxAbsDiff(serial, fused);
        Assert.True(worst < DefaultRouteBound, $"fused vs serial logits differ: max |diff| = {worst:E3}");
    }

    /// <summary>
    /// Sensitivity control for the stacked path: one shared GDN slot must corrupt it just like the serial loop, by more than the
    /// looser of the two gates above, so both could have failed.
    /// </summary>
    [SkippableFact]
    public void ForwardBatchFused_SharedGdnSlot_Mutant_Diverges()
    {
        using var f = Open();
        var serial = RunSerial(f);
        var mutant = WithFusedBatch(() => RunBatched(f, shareGdn: true, teacher: serial));
        double worst = MaxAbsDiff(serial, mutant);
        Assert.True(worst > DefaultRouteBound, $"fused shared-slot mutant stayed within the gate (max |diff| = {worst:E3}): the gate is insensitive");
    }
}
