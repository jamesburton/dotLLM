using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// #1931 final-review fixes (I-1 disposal vs in-flight operation, I-2 faulted recovery, I-3 checkpoint failure, I-4 options audit,
/// I-6 lazy KV growth, G-2 <see cref="IForwardStateRestored"/>). A scripted fake model makes each failure injectable at an exact forward.
/// </summary>
public sealed class GenerationSessionReviewFixTests
{
    private sealed class RestoredException(string m) : Exception(m), IForwardStateRestored;

    private sealed class Cp(int state) : IDisposable
    {
        public int State => state;
        public bool Disposed;
        public void Dispose() => Disposed = true;
    }

    /// <summary>Logits one-hot on a token that is never BOS/EOS; recurrent mode threads a hash state, attention mode keys on (token, position).</summary>
    private sealed class ScriptedModel(bool recurrent, int maxSeq = 256) : IModel
    {
        public const int Vocab = 32, HeadDim = 4;
        private int _state;
        public int Forwards;
        public Action<int, int>? OnForward;          // (1-based forward number, token count) at entry, before any state change
        public Action? OnCheckpoint;
        public Action? OnReset;
        public readonly List<Cp> Checkpoints = [];

        public ModelConfig Config { get; } = new()
        {
            VocabSize = Vocab, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HiddenSize = 8, IntermediateSize = 32,
            HeadDim = HeadDim, MaxSequenceLength = maxSeq, Architecture = Architecture.Llama,
        };
        public long ComputeMemoryBytes => 0;
        public void Dispose() { }
        public bool RequiresPerSequenceState => recurrent;
        public bool SupportsRecurrentStateCheckpoint => recurrent;
        public long RecurrentStateCheckpointBytes => 1024;
        public void ResetSequenceState() { OnReset?.Invoke(); _state = 0; }
        public object? CheckpointRecurrentState() { OnCheckpoint?.Invoke(); var c = new Cp(_state); Checkpoints.Add(c); return c; }
        public void RestoreRecurrentState(object? checkpoint) => _state = ((Cp)checkpoint!).State;

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => throw new NotSupportedException();
        public unsafe ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv)
        {
            Forwards++;
            OnForward?.Invoke(Forwards, t.Length);
            var logits = UnmanagedTensor.Allocate(new TensorShape(t.Length, Vocab), DType.Float32, -1);
            float* dst = (float*)logits.DataPointer;
            for (int r = 0; r < t.Length; r++)
            {
                int pick;
                if (recurrent) { _state = (_state * 31 + t[r] + 1) % 9973; pick = _state; }
                else pick = t[r] * 7 + p[r];
                for (int v = 0; v < Vocab; v++) dst[r * Vocab + v] = -10f;
                dst[r * Vocab + 3 + pick % (Vocab - 3)] = 10f;
            }
            nuint bytes = (nuint)(t.Length * HeadDim * sizeof(float));
            nint k = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(bytes, 64);
            System.Runtime.InteropServices.NativeMemory.Clear((void*)k, bytes);
            kv!.Update(new TensorRef(t.Length, HeadDim, DType.Float32, -1, k), new TensorRef(t.Length, HeadDim, DType.Float32, -1, k), p, 0);
            System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)k);
            return logits;
        }
    }

    private sealed class IdTok : ITokenizer
    {
        public int[] Encode(string text) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> ids) { var sb = new System.Text.StringBuilder(); foreach (int i in ids) sb.Append('t').Append(i).Append('|'); return sb.ToString(); }
        public string Decode(ReadOnlySpan<int> ids, bool strip) => Decode(ids);
        public string DecodeToken(int id) => Decode([id]);
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int VocabSize => ScriptedModel.Vocab;
        public int CountTokens(string text) => throw new NotSupportedException();
    }

    private static TextGenerator Gen(ScriptedModel m, List<int>? sizes = null)
        => new(m, new IdTok(), kvCacheFactory: (cfg, size) =>
        {
            sizes?.Add(size);
            return new SimpleKvCache(KvGeometry.FromConfig(cfg, false), size);
        });

    private static readonly int[] Prompt = [3, 4, 5, 6, 7];

    private static async Task<int[]> Uninterrupted(bool recurrent, int tokens)
    {
        await using var s = Gen(new ScriptedModel(recurrent)).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        return [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(tokens))).TokenIds];
    }

    // ── I-1 ────────────────────────────────────────────────────────────────────

    /// <summary>A dispose from another thread while a forward is in flight must not free the buffers under it: it runs when the operation ends.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Dispose_DuringInFlightForward_IsDeferredUntilTheOperationEnds(bool recurrent)
    {
        var m = new ScriptedModel(recurrent);
        var gen = Gen(m);
        var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        m.OnForward = (n, _) => { entered.Set(); Assert.True(release.Wait(TimeSpan.FromSeconds(30))); };
        var kv = s.KvForTest;
        var op = Task.Run(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(3)));
        Assert.True(entered.Wait(TimeSpan.FromSeconds(30)));

        await s.DisposeAsync();   // concurrent with the forward
        Assert.False(GenerationSessionBackendSuite.IsDisposed(kv), "the KV-cache was freed while a forward was running on it");
        Assert.Throws<ObjectDisposedException>(() => s.NextTokenArgMax());   // new operations are refused at once
        if (recurrent)
            Assert.Throws<InvalidOperationException>(() => Gen(m).CreateSession(Greedy(), Tag()));   // still owned until the operation ends

        release.Set();
        await op;
        Assert.True(GenerationSessionBackendSuite.IsDisposed(kv));
        if (recurrent) { await using var again = Gen(m).CreateSession(Greedy(), Tag()); }   // detached after the deferred release
    }

    /// <summary>A throwing release step must not leave the recurrent model locked in the owner table.</summary>
    [Fact]
    public async Task Dispose_ThrowingRelease_StillDetachesFromTheGenerator()
    {
        var m = new ScriptedModel(recurrent: true);
        var s = Gen(m).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        m.OnReset = () => throw new IOException("reset failed");
        await Assert.ThrowsAsync<IOException>(async () => await s.DisposeAsync());
        m.OnReset = null;
        await using var again = Gen(m).CreateSession(Greedy(), Tag());   // not locked out
    }

    // ── I-2 ────────────────────────────────────────────────────────────────────

    /// <summary>The documented recovery of a faulted session: RollbackTo(0) works and clears the fault.</summary>
    [Fact]
    public async Task FaultedSession_RollbackToZero_ClearsTheFault()
    {
        var m = new ScriptedModel(recurrent: true);
        var gen = Gen(m);
        await using var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        int before = m.Forwards;
        m.OnForward = (n, _) => { if (n == before + 2) throw new IOException("boom"); };   // 2nd token of the segment: no boundary there
        await Assert.ThrowsAsync<IOException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(4)));
        m.OnForward = null;

        Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax());          // faulted
        Assert.True(s.CanRollbackTo(0));
        Assert.False(s.CanRollbackTo(Prompt.Length));
        s.RollbackTo(0);
        Assert.Equal(0, s.CurrentPosition);

        await s.PrefillAsync(Prompt);
        Assert.Equal(await Uninterrupted(true, 4), (await s.GenerateUntilBoundaryAsync(new StopAfter(4))).TokenIds.ToArray());
    }

    // ── I-3 ────────────────────────────────────────────────────────────────────

    /// <summary>A checkpoint that cannot be taken after the segment committed never throws: the position just stops being a rollback target.</summary>
    [Fact]
    public async Task FailedBoundaryCheckpoint_NeverThrowsAfterCommit()
    {
        var m = new ScriptedModel(recurrent: true);
        await using var s = Gen(m).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        Assert.Equal([Prompt.Length], s.RollbackBoundaries);
        m.OnCheckpoint = () => throw new InvalidOperationException("device memory admission refused");
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        Assert.Equal(3, r.TokenIds.Count);
        Assert.Equal(Prompt.Length + 3, s.CurrentPosition);
        Assert.Equal(1, s.SkippedBoundaries);
        Assert.DoesNotContain(s.CurrentPosition, s.RollbackBoundaries);
        Assert.True(s.CanRollbackTo(s.CurrentPosition));
        s.NextTokenArgMax();   // the session is fully usable
    }

    /// <summary>The first failure frees the oldest checkpoint and retries once, which can succeed.</summary>
    [Fact]
    public async Task FailedBoundaryCheckpoint_EvictsOldestAndRetriesOnce()
    {
        var m = new ScriptedModel(recurrent: true);
        await using var s = Gen(m).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        int calls = 0;
        m.OnCheckpoint = () => { if (calls++ == 0) throw new InvalidOperationException("admission"); };
        await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        Assert.Equal(0, s.SkippedBoundaries);
        Assert.Equal([Prompt.Length + 3], s.RollbackBoundaries);   // the old boundary paid for the new one
        Assert.True(m.Checkpoints[0].Disposed);
    }

    /// <summary>The checkpoint byte budget is a CreateSession parameter: one boundary (1152 B) fits a 1500 B budget, two do not.</summary>
    [Fact]
    public async Task CheckpointBudget_IsConfigurable()
    {
        var m = new ScriptedModel(recurrent: true);
        await using var tight = Gen(m).CreateSession(Greedy(), Tag(), checkpointBudgetBytes: 1500);
        await tight.PrefillAsync(Prompt);
        await tight.GenerateUntilBoundaryAsync(new StopAfter(2));
        await tight.AppendObservationTokensAsync(new[] { 4, 5 });
        Assert.Single(tight.RollbackBoundaries);
        await tight.DisposeAsync();

        await using var roomy = Gen(m).CreateSession(Greedy(), Tag(), checkpointBudgetBytes: 4 * 1152);
        await roomy.PrefillAsync(Prompt);
        await roomy.GenerateUntilBoundaryAsync(new StopAfter(2));
        await roomy.AppendObservationTokensAsync(new[] { 4, 5 });
        Assert.Equal(3, roomy.RollbackBoundaries.Count);
    }

    // ── I-4 ────────────────────────────────────────────────────────────────────

    /// <summary>options.StopSequences end a segment (the matching token is committed, the text is trimmed), like TextGenerator.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task StopSequences_EndTheSegment(bool recurrent)
    {
        int[] free = await Uninterrupted(recurrent, 8);
        int i = Enumerable.Range(1, 7).First(j => !free.Take(j).Contains(free[j]));
        string stop = $"t{free[i]}|";
        await using var s = Gen(new ScriptedModel(recurrent)).CreateSession(Greedy() with { StopSequences = [stop] }, Tag());
        await s.PrefillAsync(Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(8));
        Assert.Equal(FinishReason.Stop, r.FinishReason);
        Assert.Equal(free.Take(i + 1), r.TokenIds);
        Assert.DoesNotContain(stop, r.Text);
        Assert.Equal(Prompt.Length + i + 1, s.CurrentPosition);
        Assert.Equal(s.CurrentPosition, s.KvLength);
    }

    /// <summary>Options a session cannot honour are refused up front, not ignored.</summary>
    [Fact]
    public void UnsupportedOptions_AreRefused()
    {
        var gen = Gen(new ScriptedModel(false));
        Assert.Throws<NotSupportedException>(() => gen.CreateSession(Greedy() with { StopConditions = [new StopAfter(1)] }, Tag()));
        Assert.Throws<NotSupportedException>(() => gen.CreateSession(Greedy() with { Logprobs = true }, Tag()));
        Assert.Throws<NotSupportedException>(() => gen.CreateSession(Greedy() with { RepeatGuardLimit = 4 }, Tag()));
    }

    // ── I-6 ────────────────────────────────────────────────────────────────────

    /// <summary>The default session allocates a modest cache, never the model maximum, and grows on demand with identical results.</summary>
    [Fact]
    public async Task DefaultSession_AllocatesSmall_AndGrowsAcrossTheBoundaryWithParity()
    {
        var sizes = new List<int>();
        var gen = Gen(new ScriptedModel(false, maxSeq: 100_000), sizes);
        await using var s = gen.CreateSession(Greedy(), Tag());
        Assert.Equal([TextGenerator.DefaultInitialContextTokens], sizes);
        Assert.Equal(TextGenerator.DefaultInitialContextTokens, s.KvCapacity);

        int[] big = [.. Enumerable.Range(0, 5000).Select(i => 3 + i % 20)];
        await s.PrefillAsync(big[..4000]);
        await s.AppendObservationTokensAsync(big[4000..]);   // crosses 4096
        Assert.True(s.KvCapacity >= 5000);
        Assert.Equal(s.CurrentPosition, s.KvLength);
        var grown = await s.GenerateUntilBoundaryAsync(new StopAfter(6));

        await using var fixedSize = Gen(new ScriptedModel(false, maxSeq: 100_000)).CreateSession(Greedy(), Tag(), maxContextTokens: 8000);
        await fixedSize.PrefillAsync(big);
        var reference = await fixedSize.GenerateUntilBoundaryAsync(new StopAfter(6));
        Assert.Equal(reference.TokenIds, grown.TokenIds);
    }

    /// <summary>Generation itself grows the cache when the capacity is reached, and a hard ceiling ends the segment with Length.</summary>
    [Fact]
    public async Task Generation_GrowsTheCache_UntilTheCeiling()
    {
        await using var s = Gen(new ScriptedModel(false, maxSeq: 10_000)).CreateSession(Greedy(maxTokens: 100_000), Tag());
        await s.PrefillAsync(Enumerable.Range(0, 4090).Select(i => 3 + i % 20).ToArray());
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(100));
        Assert.Equal(100, r.TokenIds.Count);
        Assert.True(s.KvCapacity > 4096);

        await using var small = Gen(new ScriptedModel(false, maxSeq: 10_000)).CreateSession(Greedy(maxTokens: 100_000), Tag(), maxContextTokens: 50);
        await small.PrefillAsync(Prompt);
        var capped = await small.GenerateUntilBoundaryAsync(new StopAfter(1000));
        Assert.Equal(FinishReason.Length, capped.FinishReason);
        Assert.Equal(50, small.CurrentPosition);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await small.AppendObservationTokensAsync(new[] { 3 }));
    }

    [Fact]
    public void MaxContextTokens_AboveTheModelMaximum_IsRejected()
        => Assert.Throws<ArgumentOutOfRangeException>(() => Gen(new ScriptedModel(false, maxSeq: 256)).CreateSession(Greedy(), Tag(), maxContextTokens: 257));

    // ── G-2 ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// A forward that throws <see cref="IForwardStateRestored"/> mid-segment leaves the session clean (not faulted, no Recover), also on a recurrent
    /// model where there is no boundary at that position, and generation then continues exactly as if nothing happened.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RestoredForwardException_MidSegment_LeavesTheSessionIntact(bool recurrent)
    {
        var m = new ScriptedModel(recurrent);
        await using var s = Gen(m).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        int before = m.Forwards;
        m.OnForward = (n, _) => { if (n == before + 2) throw new RestoredException("aborted"); };
        await Assert.ThrowsAsync<RestoredException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(4)));
        m.OnForward = null;

        Assert.Equal(Prompt.Length + 1, s.CurrentPosition);   // the first token of the segment stays committed
        Assert.Equal(s.CurrentPosition, s.KvLength);
        s.NextTokenArgMax();                                  // not faulted, logits valid
        var rest = await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        int[] free = await Uninterrupted(recurrent, 4);
        Assert.Equal(free[1..], rest.TokenIds);
        if (recurrent) Assert.Contains(s.CurrentPosition, s.RollbackBoundaries);
    }

    /// <summary>The same for the single-token re-forward of an attention rollback: the session lands cleanly one token short, without logits, and an append restores them.</summary>
    [Fact]
    public async Task RestoredForwardException_DuringAttentionRollback_DoesNotFault()
    {
        var m = new ScriptedModel(false);
        await using var s = Gen(m).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        var expected = new float[s.VocabSize];
        s.RollbackTo(Prompt.Length + 2);
        s.CopyNextLogits(expected);
        await s.GenerateUntilBoundaryAsync(new StopAfter(2));

        int before = m.Forwards;
        m.OnForward = (n, _) => { if (n == before + 1) throw new RestoredException("aborted"); };
        Assert.Throws<RestoredException>(() => s.RollbackTo(Prompt.Length + 2));
        m.OnForward = null;

        Assert.Equal(Prompt.Length + 1, s.CurrentPosition);
        Assert.Equal(s.CurrentPosition, s.KvLength);
        Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax());   // no logits yet, but not the faulted message
        Assert.True(s.CanRollbackTo(0));
        await s.AppendObservationTokensAsync(new[] { (await Uninterrupted(false, 2))[1] });
        var again = new float[s.VocabSize];
        s.CopyNextLogits(again);
        Assert.Equal(expected, again);
    }

    /// <summary>An observation append whose first forward restores itself changes nothing and does not fault (recurrent, mid-segment position without a boundary).</summary>
    [Fact]
    public async Task RestoredForwardException_DuringAppend_DoesNotFault()
    {
        var m = new ScriptedModel(recurrent: true);
        await using var s = Gen(m).CreateSession(Greedy(), Tag());
        await s.PrefillAsync(Prompt);
        m.OnCheckpoint = () => throw new InvalidOperationException("no checkpoints");
        await s.GenerateUntilBoundaryAsync(new StopAfter(2));   // position now has no boundary
        m.OnCheckpoint = null;
        int pos = s.CurrentPosition;
        int before = m.Forwards;
        m.OnForward = (n, _) => { if (n == before + 1) throw new RestoredException("aborted"); };
        await Assert.ThrowsAsync<RestoredException>(async () => await s.AppendObservationTokensAsync(new[] { 5, 6 }));
        m.OnForward = null;
        Assert.Equal(pos, s.CurrentPosition);
        s.NextTokenArgMax();   // not faulted
        await s.AppendObservationTokensAsync(new[] { 5, 6 });
        Assert.Equal(pos + 2, s.CurrentPosition);
    }
}
