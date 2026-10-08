using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #840: incremental checkpoint of the Qwen4-Exp recurrent state. Stamped delta copy of the pooled indexer keys / own K/V rows,
/// the lazily-deferred ("fused") GDN checkpoint and the scan that reads its first state from a source span. Every claim has an arm
/// that would fail if the mechanism were a no-op or only handled append-only lineages.
/// </summary>
public sealed unsafe class Qwen4ExpDeltaCheckpointTests : IDisposable
{
    private const int D = 8, Block = 4;

    private static float[] Keys(int tokens, int seed)
    {
        var rng = new Random(seed);
        var a = new float[tokens * D];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static readonly float[] Gamma = Enumerable.Repeat(1f, D).ToArray();

    private static void Append(Qwen4ExpIndexerCache c, float[] raw)
        => c.Append(raw, raw.Length / D, Gamma, 1e-6f, default, default, 0);

    private static float[] Pooled(Qwen4ExpIndexerCache c) => c.Pooled.ToArray();

    // ── indexer: stamped delta copy ──

    [Fact]
    public void Indexer_SecondSync_CopiesOnlyTheNewBlocks_AndAFreshShellCopiesEverything()
    {
        using var live = new Qwen4ExpIndexerCache(D, Block);
        using var shell = new Qwen4ExpIndexerCache(D, Block);
        using var fresh = new Qwen4ExpIndexerCache(D, Block);
        Append(live, Keys(400, 1));                     // 100 pooled blocks
        shell.CopyFrom(live);
        Assert.Equal(100, shell.LastCopiedPooledRows);
        Assert.Equal(Pooled(live), Pooled(shell));

        Append(live, Keys(12, 2));                      // 3 more blocks
        shell.CopyFrom(live);
        Assert.InRange(shell.LastCopiedPooledRows, 3, 3 + Qwen4ExpRowStamps.Group);   // the new rows + at most the straddled group
        Assert.Equal(103, live.CompleteBlocks);
        Assert.Equal(Pooled(live), Pooled(shell));

        fresh.CopyFrom(live);                           // control: a never-synced shell has nothing to skip
        Assert.Equal(103, fresh.LastCopiedPooledRows);
        Assert.Equal(Pooled(live), Pooled(fresh));

        shell.CopyFrom(live);                           // nothing changed: nothing copied
        Assert.Equal(0, shell.LastCopiedPooledRows);
    }

    [Fact]
    public void Indexer_DivergentHistory_IsNotMistakenForTheSyncedOne()
    {
        // A naive "the shell already has blocks [0, n)" delta would leave the shell on X's keys after live moved to Y.
        using var live = new Qwen4ExpIndexerCache(D, Block);
        using var shell = new Qwen4ExpIndexerCache(D, Block);
        var x = Keys(400, 3); var y = Keys(400, 4);
        Append(live, x);
        shell.CopyFrom(live);
        var pooledX = Pooled(shell);

        live.Reset();
        Append(live, y);                                // same length, different content
        shell.CopyFrom(live);
        Assert.Equal(Pooled(live), Pooled(shell));
        Assert.NotEqual(pooledX, Pooled(shell));

        // Restoring live from the shell on a third cache that holds a third history also converges exactly.
        using var third = new Qwen4ExpIndexerCache(D, Block);
        Append(third, Keys(320, 5));
        third.CopyFrom(shell);
        Assert.Equal(Pooled(shell), Pooled(third));
        Assert.Equal(shell.TokenCount, third.TokenCount);
    }

    [Fact]
    public void Indexer_RollbackThenDifferentTokens_IsCopiedNotSkipped()
    {
        // Same lineage up to a point, then live rolls back (via a copy from an older shell) and re-appends different keys over blocks the
        // newer shell already holds: those blocks must be refreshed.
        using var live = new Qwen4ExpIndexerCache(D, Block);
        using var old = new Qwen4ExpIndexerCache(D, Block);
        using var shell = new Qwen4ExpIndexerCache(D, Block);
        Append(live, Keys(200, 6));
        old.CopyFrom(live);                             // checkpoint at 200 tokens
        Append(live, Keys(80, 7));
        shell.CopyFrom(live);                           // newer shell at 280
        live.CopyFrom(old);                             // roll back to 200
        Append(live, Keys(80, 8));                      // different continuation
        shell.CopyFrom(live);
        Assert.Equal(Pooled(live), Pooled(shell));
        Assert.True(shell.LastCopiedPooledRows < live.CompleteBlocks, "only the diverged tail should have been copied");
    }

    [Fact]
    public void QsaOwnKv_DeltaCopy_FollowsTheSameRules()
    {
        const int stride = 12;
        using var live = new Qwen4ExpQsaState(stride, D, Block);
        using var shell = new Qwen4ExpQsaState(stride, D, Block);
        void Step(int n, int seed)
        {
            var rng = new Random(seed);
            var k = new float[n * stride]; var v = new float[n * stride];
            for (int i = 0; i < k.Length; i++) { k[i] = (float)rng.NextDouble(); v[i] = (float)rng.NextDouble(); }
            live.AppendKv(k, v, n);
            live.Indexer.Append(Keys(n, seed), n, Gamma, 1e-6f, default, default, 0);
        }
        Step(300, 1);
        shell.CopyFrom(live);
        Step(8, 2);
        shell.CopyFrom(live);
        Assert.Equal(live.Keys.ToArray(), shell.Keys.ToArray());
        Assert.Equal(live.Values.ToArray(), shell.Values.ToArray());
        Assert.Equal(Pooled(live.Indexer), Pooled(shell.Indexer));
    }

    // ── GDN scan from a source span ──

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(7)]
    public void Scan_FromSource_IsBitIdenticalToCopyThenInPlace(int T)
    {
        const int nV = 4, nK = 2, dS = 8;
        var rng = new Random(9);
        float[] R(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * s; return a; }
        var src = R(nV * dS * dS, 1f);
        var q = R(T * nK * dS, 1f); var k = R(T * nK * dS, 1f); var v = R(T * nV * dS, 1f);
        var g = R(T * nV, 0.1f).Select(x => 0.9f + x).ToArray(); var beta = R(T * nV, 0.5f).Select(MathF.Abs).ToArray();

        var stateA = (float[])src.Clone(); var outA = new float[T * nV * dS];
        GatedDeltaNetScan.Execute(stateA, q, k, v, g, beta, outA, nV, nK, dS, T);

        var stateB = R(nV * dS * dS, 5f);                // garbage: must be ignored
        var outB = new float[T * nV * dS];
        GatedDeltaNetScan.Execute(stateB, q, k, v, g, beta, outB, nV, nK, dS, T, stateSource: src);

        Assert.Equal(stateA, stateB);
        Assert.Equal(outA, outB);
        // Control: the garbage state really differed, and the source was not modified.
        var garbageRun = R(nV * dS * dS, 5f); var outC = new float[T * nV * dS];
        GatedDeltaNetScan.Execute(garbageRun, q, k, v, g, beta, outC, nV, nK, dS, T);
        Assert.NotEqual(outA, outC);
    }

    // ── GdnStateCache lazy copy ──

    private static GdnStateCache NewGdn(int layers, int seed)
    {
        var gdn = new GatedDeltaNetConfig(FullAttnInterval: 4, NVHead: 4, NKHead: 2, DState: 8, DInner: 32, DConv: 4);
        var c = new GdnStateCache(gdn, layers);
        var rng = new Random(seed);
        for (int l = 0; l < layers; l++)
        {
            foreach (ref float f in c.GetGdnState(l)) f = (float)rng.NextDouble();
            foreach (ref float f in c.GetConvState(l)) f = (float)rng.NextDouble();
        }
        return c;
    }

    [Fact]
    public void Gdn_DeferredCopy_MaterialisesOnAnyAccessor_AndFusesOnBeginUpdate()
    {
        using var src = NewGdn(3, 1);
        using var dst = NewGdn(3, 2);
        using var want = src.Clone();
        dst.DeferCopyFrom(src);
        Assert.True(dst.HasPendingSource);

        dst.BeginUpdate(1, out var sc, out var sg);                 // layer 1 handed to the fused update
        Assert.False(sg.IsEmpty);
        Assert.Equal(want.GetGdnState(1).ToArray(), sg.ToArray());
        dst.BeginUpdate(1, out sc, out sg);                          // already consumed: now an ordinary in-place update
        Assert.True(sg.IsEmpty && sc.IsEmpty);

        Assert.Equal(want.GetGdnState(0).ToArray(), dst.GetGdnState(0).ToArray());   // plain accessor materialises layer 0
        dst.MaterializePending();
        Assert.False(dst.HasPendingSource);
        Assert.Equal(want.GetConvState(2).ToArray(), dst.GetConvState(2).ToArray());
        Assert.Equal(want.GetGdnState(2).ToArray(), dst.GetGdnState(2).ToArray());
    }

    [Fact]
    public void Gdn_CopyToFromAPendingCache_CopiesTheLogicalContent()
    {
        using var src = NewGdn(3, 3);
        using var live = NewGdn(3, 4);
        using var other = NewGdn(3, 5);
        using var want = src.Clone();
        live.DeferCopyFrom(src);
        live.BeginUpdate(2, out _, out _);                           // layer 2 consumed (stale for the test: overwrite it logically)
        live.GetGdnStateForUpdate(2).Clear();
        live.CopyTo(other);
        Assert.Equal(want.GetGdnState(0).ToArray(), other.GetGdnState(0).ToArray());
        Assert.Equal(want.GetGdnState(1).ToArray(), other.GetGdnState(1).ToArray());
        Assert.All(other.GetGdnState(2).ToArray(), f => Assert.Equal(0f, f));
    }

    [Fact]
    public void Gdn_SwapBuffers_ExchangesContent_AndRefusesPending()
    {
        using var a = NewGdn(2, 6); using var b = NewGdn(2, 7);
        var a0 = a.GetGdnState(0).ToArray(); var b0 = b.GetGdnState(0).ToArray();
        a.SwapBuffersWith(b);
        Assert.Equal(b0, a.GetGdnState(0).ToArray());
        Assert.Equal(a0, b.GetGdnState(0).ToArray());
        a.DeferCopyFrom(b);
        Assert.Throws<InvalidOperationException>(() => a.SwapBuffersWith(b));
    }

    // ── model level ──

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-delta-" + Guid.NewGuid().ToString("N"));
    private Qwen4ExpTransformerModel? _model;
    private IDisposable? _gguf, _m;

    private Qwen4ExpTransformerModel Model()
    {
        if (_model is not null) return _model;
        Directory.CreateDirectory(_dir);
        string path = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "syn.gguf"));
        var (m, g, _) = ModelLoader.LoadFromGguf(path);
        _m = m; _gguf = g;
        return _model = (Qwen4ExpTransformerModel)m;
    }

    public void Dispose()
    {
        _m?.Dispose(); _gguf?.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private static int[] Toks(int n, int seed)
    {
        var r = new Random(seed); var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = r.Next(4, SyntheticQwen4ExpGguf.VocabSize);
        return t;
    }

    private static float[] Fwd(Qwen4ExpTransformerModel m, int[] ids, int start)
    {
        using var t = m.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), -1, null);
        return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    [Fact]
    public void Checkpoints_TwoOutstanding_RestoredOutOfOrder_AndTwice_ReplayBitIdentically()
    {
        var m = Model();
        var p1 = Toks(21, 1); var p2 = Toks(9, 2); var detour = Toks(7, 3); var next = Toks(3, 4);

        m.ResetSequenceState();
        Fwd(m, p1, 0);
        var wantAfterP1 = Fwd(m, next, p1.Length);                   // reference continuation from checkpoint 1
        m.ResetSequenceState();
        Fwd(m, p1, 0); Fwd(m, p2, p1.Length);
        var wantAfterP2 = Fwd(m, next, p1.Length + p2.Length);       // reference continuation from checkpoint 2

        m.ResetSequenceState();
        Fwd(m, p1, 0);
        var cp1 = m.CheckpointRecurrentState();
        Fwd(m, p2, p1.Length);
        var cp2 = m.CheckpointRecurrentState();                      // second shell while the first is outstanding
        Fwd(m, detour, p1.Length + p2.Length);

        m.RestoreRecurrentState(cp1);                                // older checkpoint first
        Assert.Equal(wantAfterP1, Fwd(m, next, p1.Length));
        m.RestoreRecurrentState(cp2);                                // newer one after the live state moved on
        Assert.Equal(wantAfterP2, Fwd(m, next, p1.Length + p2.Length));
        m.RestoreRecurrentState(cp1);                                // and the first again: restore does not consume
        Assert.Equal(wantAfterP1, Fwd(m, next, p1.Length));
        (cp1 as IDisposable)?.Dispose(); (cp2 as IDisposable)?.Dispose();
    }

    [Fact]
    public void RestoreThenReleaseBeforeForward_ExchangesBuffersInsteadOfCopying()
    {
        var m = Model();
        var prefix = Toks(15, 11); var detour = Toks(6, 12); var next = Toks(4, 13);

        float[] Run(bool viaCheckpoint, out long copies)
        {
            using var live = m.CreateState();
            using var shell = m.CreateState();
            Row(m, live, prefix, 0);
            copies = 0;
            if (viaCheckpoint)
            {
                live.CaptureInto(shell);                        // no GDN copy: buffers exchanged, live defers from the shell
                Row(m, live, detour, prefix.Length);            // consumes the deferred copy (fused)
                live.RestoreFrom(shell);                        // deferred again
                live.ReleaseSource(shell);                      // restore-then-dispose: exchange, not copy
                copies = live.Gdn.MaterializedLayerCopies;
            }
            return Row(m, live, next, prefix.Length);
        }

        var want = Run(false, out _);
        var got = Run(true, out long copies);
        Assert.Equal(want, got);
        Assert.Equal(0, copies);
    }

    private static float[] Row(Qwen4ExpTransformerModel m, Qwen4ExpSequenceState st, int[] ids, int start)
    {
        using var t = m.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), -1, st, null, lastTokenLogitsOnly: false);
        return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    [Fact]
    public void Checkpoint_DisposedBeforeAnyForward_StillLeavesTheLiveStateIntact()
    {
        // The live state defers its GDN content from the shell: freeing / reusing the shell first must materialise it.
        var m = Model();
        var prefix = Toks(13, 5); var next = Toks(4, 6); var other = Toks(5, 7);
        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        var want = Fwd(m, next, prefix.Length);

        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        var cp = m.CheckpointRecurrentState();
        (cp as IDisposable)!.Dispose();                              // shell goes back to the pool, live still points at it
        var cpReuse = m.CheckpointRecurrentState();                  // reuses (and swaps buffers with) the pooled shell
        Fwd(m, other, prefix.Length);
        m.RestoreRecurrentState(cpReuse);
        Assert.Equal(want, Fwd(m, next, prefix.Length));
        (cpReuse as IDisposable)!.Dispose();
    }

    [Fact]
    public void RestoreWithoutForward_ThenCheckpointAgain_KeepsTheLogicalState()
    {
        var m = Model();
        var prefix = Toks(11, 8); var detour = Toks(6, 9); var next = Toks(5, 10);
        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        var want = Fwd(m, next, prefix.Length);

        m.ResetSequenceState();
        Fwd(m, prefix, 0);
        var cp = m.CheckpointRecurrentState();
        Fwd(m, detour, prefix.Length);
        m.RestoreRecurrentState(cp);                                 // live now lazily points at cp's shell
        var cp2 = m.CheckpointRecurrentState();                      // checkpoint of a state that has pending GDN content
        m.RestoreRecurrentState(cp2);
        Assert.Equal(want, Fwd(m, next, prefix.Length));
        (cp as IDisposable)!.Dispose(); (cp2 as IDisposable)!.Dispose();
    }
}
