using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Engine;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #478: CUDA ports of the #473 per-row GDN snapshots and the #469 pooled GDN checkpoint on
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>, mirroring the CPU
/// (<see cref="MtpRecurrentRowSnapshotTests"/>) and Vulkan tests.
/// </summary>
/// <remarks>
/// Snapshot-vs-replay comparisons use a tolerance, not bit-equality: the snapshot comes from inside
/// the K+1-row verify forward, the replay is a shorter batch (one row for row 0), and the trunk's
/// GEMMs take a different route by batch width. The snapshot copy itself is exact. The neighbour
/// check in <see cref="MtpRecurrentRowSnapshotTests.AssertRowSelection"/> is what keeps the
/// tolerance honest: a snapshot index that is off by one must miss.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseGdnRowSnapshotTests : IDisposable
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseGdnRowSnapshotTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-gdn-rowsnap-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Fixture(string name) =>
        SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, name), withMtp: true);

    /// <summary>
    /// Row selection on CUDA: restoring the snapshot of verify row <c>n</c> and probing the next
    /// token must match a checkpoint restore + replay of rows <c>0..n</c>, and must miss the replay
    /// of rows <c>n±1</c> — so an off-by-one in the state or the conv window fails.
    /// </summary>
    [SkippableFact]
    public void RestoreRecurrentStateToRow_MatchesPrefixReplay_AndNotItsNeighbours()
    {
        string ptxDir = SkipUnlessCuda();
        using var gguf = GgufFile.Open(Fixture("rowsnap-model.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.SupportsRecurrentRowSnapshots);

        float[][] snap = MtpRecurrentRowSnapshotTests.RowProbes(
            model, config, useSnapshots: true, kvFactory: () => model.CreateKvCache(config.MaxSequenceLength));
        float[][] replay = MtpRecurrentRowSnapshotTests.RowProbes(
            model, config, useSnapshots: false, kvFactory: () => model.CreateKvCache(config.MaxSequenceLength));
        MtpRecurrentRowSnapshotTests.AssertRowSelection(snap, replay, AbsTol, RelTol, _out);
        _out.WriteLine($"snapshot scratch: {model.RecurrentRowSnapshotBytes} bytes for 3 rows");
    }

    /// <summary>A snapshot is valid only until the next forward: a stale restore must throw.</summary>
    [SkippableFact]
    public void RestoreRecurrentStateToRow_AfterAnotherForward_Throws()
    {
        string ptxDir = SkipUnlessCuda();
        using var gguf = GgufFile.Open(Fixture("rowsnap-stale.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);

        using (model.ForwardWithRecurrentSnapshots([1, 2, 3], [0, 1, 2], -1, kv, null)) { }
        kv.Rollback(2);
        using (model.Forward([5], [2], -1, kv)) { }
        Assert.Throws<InvalidOperationException>(() => model.RestoreRecurrentStateToRow(1));
    }

    /// <summary>
    /// Decoder level on CUDA: drafts scripted so rounds roll back to rows 0, 1 and 2 of K=3 (and one
    /// clean round). Snapshots on and off must both reproduce plain greedy decode, restore exactly
    /// where the other replays, and leave next-token logits within tolerance.
    /// </summary>
    [SkippableFact]
    public void DraftAndVerify_ScriptedPartialRejections_SnapshotsMatchGreedyAndReplay()
    {
        string ptxDir = SkipUnlessCuda();
        string path = Fixture("rowsnap-decoder.gguf");
        const int StartToken = 3, NewTokens = 12, K = 3;

        List<int> greedy = PlainGreedy(path, ptxDir, StartToken, NewTokens + K);
        var on = RunScripted(path, ptxDir, greedy, StartToken, NewTokens, K, snapshots: true);
        var off = RunScripted(path, ptxDir, greedy, StartToken, NewTokens, K, snapshots: false);
        _out.WriteLine($"greedy: [{string.Join(",", greedy.Take(NewTokens + 1))}]");
        _out.WriteLine($"on : emitted/round [{string.Join(",", on.PerRound)}] avoided={on.Avoided} replays={on.Replays}");
        _out.WriteLine($"off: emitted/round [{string.Join(",", off.PerRound)}] avoided={off.Avoided} replays={off.Replays}");

        Assert.Equal(greedy.Take(NewTokens + 1), on.Tokens);
        Assert.Equal(greedy.Take(NewTokens + 1), off.Tokens);
        for (int emitted = 1; emitted <= K + 1; emitted++)
            Assert.Contains(emitted, on.PerRound);
        Assert.True(on.Avoided > 0 && on.Replays == 0, "snapshot run must roll back without replaying");
        Assert.True(off.Avoided == 0 && off.Replays == on.Avoided,
            "the replay run must replay exactly where the snapshot run restored");
        MtpRecurrentRowSnapshotTests.AssertClose(off.NextLogits, on.NextLogits, AbsTol, RelTol,
            "next-token logits, snapshots vs replay");
    }

    /// <summary>
    /// Issue #469 pooling on CUDA: the checkpoint is opaque and caller-disposed, and a second
    /// checkpoint taken after disposing the first reuses its buffers. The reused snapshot must hold
    /// the state at the SECOND checkpoint (a pool that skipped the copy would restore the first).
    /// </summary>
    [SkippableFact]
    public void PooledCheckpoint_ReusedBuffers_RestoreTheLatestState()
    {
        string ptxDir = SkipUnlessCuda();
        using var gguf = GgufFile.Open(Fixture("pooled-checkpoint.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);

        using (model.Forward([1, 2], [0, 1], -1, kv)) { }
        object? first = model.CheckpointRecurrentState();
        Assert.IsAssignableFrom<IDisposable>(first);
        ((IDisposable)first!).Dispose();

        using (model.Forward([3], [2], -1, kv)) { }
        object? second = model.CheckpointRecurrentState();   // reuses first's buffers
        float[] expected;
        using (ITensor probe = model.Forward([4], [3], -1, kv))
            expected = MtpRecurrentRowSnapshotTests.LastRow(probe, config.VocabSize);

        // Move the recurrent state on, restore, and probe again at the same position.
        kv.Rollback(3);
        using (model.Forward([7], [3], -1, kv)) { }
        model.RestoreRecurrentState(second);
        kv.Rollback(3);
        float[] actual;
        using (ITensor probe = model.Forward([4], [3], -1, kv))
            actual = MtpRecurrentRowSnapshotTests.LastRow(probe, config.VocabSize);
        ((IDisposable)second!).Dispose();

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);   // same kernels, same inputs: bit-exact
    }

    private sealed record ScriptedRun(List<int> Tokens, List<int> PerRound, int Avoided, int Replays, float[] NextLogits);

    private static ScriptedRun RunScripted(
        string path, string ptxDir, List<int> greedy, int start, int newTokens, int k, bool snapshots)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var real = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var model = new ScriptedDraftMtpModel(real, greedy, wrongAt: p => p is 1 or 3 or 6 or 11 or 13);
        using var kv = real.CreateKvCache(config.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = snapshots };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        // No prefill: position 0 is the start token's own slot, and the first verify forwards it.
        var ids = new List<int> { start };
        var perRound = new List<int>();
        int position = 0;
        Span<int> outBuf = stackalloc int[k + 1];
        while (ids.Count - 1 < newTokens)
        {
            var r = decoder.DraftAndVerify(model, kv, mtp, pipeline, ids, null, position, config.VocabSize, k, outBuf);
            perRound.Add(r.AcceptedCount);
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(outBuf[i]);
            position += r.AcceptedCount;
        }

        using ITensor next = model.Forward([ids[position]], [position], deviceId: -1, kv);
        float[] nextRow = MtpRecurrentRowSnapshotTests.LastRow(next, config.VocabSize);
        return new ScriptedRun(ids.Take(newTokens + 1).ToList(), perRound, decoder.ReplaysAvoided, decoder.Replays, nextRow);
    }

    private static List<int> PlainGreedy(string path, string ptxDir, int start, int count)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);
        var ids = new List<int> { start };
        for (int pos = 0; pos < count && pos < config.MaxSequenceLength; pos++)
        {
            using ITensor logits = model.Forward([ids[^1]], [pos], -1, kv);
            float[] row = MtpRecurrentRowSnapshotTests.LastRow(logits, config.VocabSize);
            ids.Add(Array.IndexOf(row, row.Max()));
        }
        return ids;
    }

    private static string SkipUnlessCuda()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        return ptxDir!;
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
