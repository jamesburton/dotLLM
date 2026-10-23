using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #473 on the CPU reference: per-row GDN snapshots recorded by the verify forward, and the
/// MTP decoder rolling back to row <c>accepted</c> with them instead of replaying.
/// </summary>
public sealed class MtpRecurrentRowSnapshotTests : IDisposable
{
    // A batched forward and a shorter batch reduce in a different order (the CPU GEMM tiles by n),
    // so a snapshot taken inside a K+1-row verify is ~1 ULP off a replay of the prefix. Tight
    // enough that the state of a neighbouring row misses it by orders of magnitude (asserted).
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public MtpRecurrentRowSnapshotTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gdn-rowsnap-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Fixture(string name) =>
        SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, name), withMtp: true);

    /// <summary>
    /// Row selection, directly on the model. For every row <c>n</c> of a 4-row verify, restoring
    /// snapshot <c>n</c> and probing the next token must match restoring a pre-verify checkpoint and
    /// forwarding rows <c>0..n</c> — the replay path. And it must NOT match the replay of rows
    /// <c>n-1</c> or <c>n+1</c>: a snapshot index that is off by one fails this test, which is what
    /// makes the tolerance-based match above mean something.
    /// </summary>
    [Fact]
    public void RestoreRecurrentStateToRow_MatchesPrefixReplay_AndNotItsNeighbours()
    {
        using var gguf = GgufFile.Open(Fixture("rowsnap-model.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.True(model.SupportsRecurrentRowSnapshots);

        float[][] snap = RowProbes(model, config, useSnapshots: true);
        float[][] replay = RowProbes(model, config, useSnapshots: false);
        AssertRowSelection(snap, replay, AbsTol, RelTol, _out);

        _out.WriteLine($"snapshot scratch: {model.RecurrentRowSnapshotBytes} bytes for 3 rows");
    }

    /// <summary>
    /// A snapshot is valid only until the next forward: a stale restore must throw rather than
    /// silently rewind the state to a batch that no longer describes it.
    /// </summary>
    [Fact]
    public void RestoreRecurrentStateToRow_AfterAnotherForward_Throws()
    {
        using var gguf = GgufFile.Open(Fixture("rowsnap-stale.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

        using (model.ForwardWithRecurrentSnapshots([1, 2, 3], [0, 1, 2], -1, kv, null)) { }
        kv.Rollback(2);
        using (model.Forward([5], [2], -1, kv)) { }
        Assert.Throws<InvalidOperationException>(() => model.RestoreRecurrentStateToRow(1));
    }

    /// <summary>
    /// Decoder level, with drafts scripted so rounds accept 0, 1 and 2 of K=3 drafts. With
    /// snapshots on and off, the output must equal plain greedy decode; the snapshot run must
    /// roll back through snapshots (rows above 0 included) with no replay; and the next-token
    /// logits after the run must match the replay run's.
    /// </summary>
    [Fact]
    public void DraftAndVerify_ScriptedPartialRejections_SnapshotsMatchGreedyAndReplay()
    {
        string path = Fixture("rowsnap-decoder.gguf");
        const int StartToken = 3, NewTokens = 12, K = 3;

        List<int> greedy = PlainGreedy(path, StartToken, NewTokens + K + 1);
        var on = RunDecoder(path, greedy, StartToken, NewTokens, K, snapshots: true);
        var off = RunDecoder(path, greedy, StartToken, NewTokens, K, snapshots: false);

        _out.WriteLine($"greedy: [{string.Join(",", greedy.Take(NewTokens + 1))}]");
        _out.WriteLine($"on : emitted/round [{string.Join(",", on.EmittedPerRound)}] replaysAvoided={on.ReplaysAvoided} replays={on.Replays}");
        _out.WriteLine($"off: emitted/round [{string.Join(",", off.EmittedPerRound)}] replaysAvoided={off.ReplaysAvoided} replays={off.Replays}");

        Assert.Equal(greedy.Take(NewTokens + 1), on.Tokens);
        Assert.Equal(greedy.Take(NewTokens + 1), off.Tokens);

        // The script must actually have produced a rollback to every row, and a clean round.
        for (int emitted = 1; emitted <= K + 1; emitted++)
            Assert.Contains(emitted, on.EmittedPerRound);
        Assert.True(on.ReplaysAvoided > 0 && on.Replays == 0, "snapshot run must roll back without replaying");
        Assert.True(off.ReplaysAvoided == 0 && off.Replays == on.ReplaysAvoided,
            "the replay run must replay exactly where the snapshot run restored");

        AssertClose(off.NextLogits, on.NextLogits, AbsTol, RelTol, "next-token logits, snapshots vs replay");
    }

    // ── helpers shared with the Vulkan variant ───────────────────────────────

    internal static readonly int[] Prefix = [1, 2, 3];
    internal static readonly int[] Verify = [4, 5, 6, 7];
    internal const int ProbeToken = 8;

    /// <summary>
    /// For each row n of <see cref="Verify"/> (0..2), the logits of <see cref="ProbeToken"/> forwarded
    /// right after the state was rolled back to "after row n" — via a snapshot, or via
    /// checkpoint-restore + replay of rows 0..n.
    /// </summary>
    internal static float[][] RowProbes(IModel model, ModelConfig config, bool useSnapshots,
                                        Func<IKvCache>? kvFactory = null)
    {
        int vocab = config.VocabSize;
        int p0 = Prefix.Length;
        using IKvCache kv = kvFactory?.Invoke() ?? new SimpleKvCache(
            ((Qwen3HybridDenseTransformerModel)model).AttentionLayerCount,
            config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        model.ResetSequenceState();
        using (model.Forward(Prefix, Positions(0, p0), -1, kv)) { }
        object? checkpoint = model.CheckpointRecurrentState();
        try
        {
            var result = new float[Verify.Length - 1][];
            for (int n = 0; n < Verify.Length - 1; n++)
            {
                model.RestoreRecurrentState(checkpoint);
                kv.Rollback(p0);
                if (useSnapshots)
                {
                    using (model.ForwardWithRecurrentSnapshots(Verify, Positions(p0, Verify.Length), -1, kv, null)) { }
                    kv.Rollback(p0 + n + 1);
                    model.RestoreRecurrentStateToRow(n);
                }
                else
                {
                    using (model.Forward(Verify.AsSpan(0, n + 1), Positions(p0, n + 1), -1, kv)) { }
                }

                using ITensor probe = model.Forward([ProbeToken], [p0 + n + 1], -1, kv);
                result[n] = LastRow(probe, vocab);
            }
            return result;
        }
        finally
        {
            (checkpoint as IDisposable)?.Dispose();
        }
    }

    internal static void AssertRowSelection(float[][] snap, float[][] replay, float absTol, float relTol,
                                            ITestOutputHelper output)
    {
        for (int n = 0; n < snap.Length; n++)
        {
            AssertClose(replay[n], snap[n], absTol, relTol, $"row {n}");
            foreach (int m in new[] { n - 1, n + 1 })
            {
                if (m < 0 || m >= replay.Length) continue;
                float worst = WorstExcess(replay[m], snap[n], absTol, relTol);
                output.WriteLine($"snapshot row {n} vs replay row {m}: max |diff|/bar = {worst:F1}");
                Assert.True(worst > 1f,
                    $"snapshot row {n} is within tolerance of replay row {m}: this test cannot tell a " +
                    "row-selection off-by-one from a correct restore.");
            }
        }
    }

    private sealed record DecoderRun(
        List<int> Tokens, List<int> EmittedPerRound, int ReplaysAvoided, int Replays, float[] NextLogits);

    private static DecoderRun RunDecoder(string path, List<int> greedy, int start, int newTokens, int k, bool snapshots)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var real = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        // Rounds start at 0, 1, 3, 6, 10, 11: missing these drafts makes them accept 0, 1, 2, 3
        // (all), 0 and 1 drafts — a rollback to every row 0..K-1, and the no-rollback round.
        var model = new ScriptedDraftMtpModel(real, greedy, wrongAt: p => p is 1 or 3 or 6 or 11 or 13);
        using var kv = new SimpleKvCache(real.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = snapshots };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        var ids = new List<int> { start };
        var perRound = new List<int>();
        int position = 0;
        Span<int> outBuf = stackalloc int[k + 1];
        while (ids.Count - 1 < newTokens)
        {
            var r = decoder.DraftAndVerify(model, kv, mtp, pipeline, ids, null, position,
                config.VocabSize, k, outBuf);
            perRound.Add(r.AcceptedCount);
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(outBuf[i]);
            position += r.AcceptedCount;
        }

        // The state the decoder left: forward the still-pending last token.
        using ITensor next = model.Forward([ids[position]], [position], -1, kv);
        return new DecoderRun(ids.Take(newTokens + 1).ToList(), perRound,
            decoder.ReplaysAvoided, decoder.Replays, LastRow(next, config.VocabSize));
    }

    private static List<int> PlainGreedy(string path, int start, int count)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        var ids = new List<int> { start };
        for (int pos = 0; pos < count && pos < config.MaxSequenceLength; pos++)
        {
            using ITensor logits = model.Forward([ids[^1]], [pos], -1, kv);
            float[] row = LastRow(logits, config.VocabSize);
            ids.Add(Array.IndexOf(row, row.Max()));
        }
        return ids;
    }

    internal static int[] Positions(int start, int count) => Enumerable.Range(start, count).ToArray();

    internal static unsafe float[] LastRow(ITensor t, int vocab)
    {
        int rows = t.Shape[0];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
    }

    internal static float WorstExcess(float[] expected, float[] actual, float absTol, float relTol)
    {
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]) / (absTol + relTol * MathF.Abs(expected[i])));
        return worst;
    }

    internal static void AssertClose(float[] expected, float[] actual, float absTol, float relTol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float bar = absTol + relTol * MathF.Abs(expected[i]);
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= bar,
                $"{what}: index {i}: expected {expected[i]:R}, got {actual[i]:R} (bar {bar:E2})");
        }
    }
}
