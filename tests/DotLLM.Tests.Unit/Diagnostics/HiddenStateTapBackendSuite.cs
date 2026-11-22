using DotLLM.Core.Attention;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

/// <summary>One model of one family on one backend, plus what the suite needs around it (#1920).</summary>
public abstract class HiddenStateTapFixture : IDisposable
{
    public abstract IModel Model { get; }
    public IHiddenStateTapHost Host => (IHiddenStateTapHost)Model;
    public abstract int NumLayers { get; }
    public abstract int Hidden { get; }
    public abstract int Vocab { get; }
    /// <summary>Tokens that are valid for the model (ordinary ids, below the vocabulary).</summary>
    public abstract int[] Prompt { get; }
    /// <summary>True when the model carries recurrent state (hybrid families).</summary>
    public bool IsRecurrent => Model.RequiresPerSequenceState;
    /// <summary>True when <c>ForwardBatch</c> normally takes the packed matmul-fused loop that a tap must bypass.</summary>
    public abstract bool ExpectsPackedBatchFallback { get; }
    /// <summary>Whether bit-identical logits are expected with and without a tap (CPU: yes; a GPU may only promise tolerance).</summary>
    public virtual bool BitExact => true;
    /// <summary>
    /// Whether <c>RegisterAbortableTap</c> is available for this model (a host whose abort cannot yet restore its recurrent state refuses it).
    /// The abort tests skip when false.
    /// </summary>
    public virtual bool SupportsAbortableTaps => true;
    /// <summary>
    /// Whether the batch test expects bit-identical logits. False where the tapped run takes a different kernel path than the untapped one
    /// (a fused batch versus the per-sequence loop on Vulkan): tolerance 1e-2 absolute and identical argmax then.
    /// </summary>
    public virtual bool BatchBitExact => BitExact;
    public abstract IKvCache NewKv(int maxLen = 96);
    /// <summary>
    /// A KV-cache of the type this backend would create for a sliding-window model (compact windows, window 8 on the even layers), or
    /// <see langword="null"/> when the host has none (the windowed abort tests skip).
    /// </summary>
    public virtual IKvCache? NewWindowedKv(int maxLen) => null;
    /// <summary>
    /// A second host (owned and disposed by the fixture) over the same weights whose <see cref="ModelConfig"/> uses sliding-window
    /// attention (window 8, pattern 2), or <see langword="null"/> when the host cannot build one.
    /// </summary>
    public virtual IHiddenStateTapHost? NewSlidingWindowConfigHost() => null;
    /// <summary>Raw KV contents for the first <c>CurrentLength</c> rows, every layer that has KV, for byte comparison.</summary>
    public abstract float[] KvBytes(IKvCache kv);
    public abstract void Dispose();
}

/// <summary>
/// The #1920 tap assertions, written once for every backend: a derived class supplies <see cref="Open"/>. CPU runs them here;
/// Vulkan and CUDA derive from it when their hosts land. Families: attention-only Llama, GDN-hybrid dense, GDN-hybrid MoE.
/// </summary>
public abstract class HiddenStateTapBackendSuite
{
    public const string Llama = "llama", HybridDense = "hybrid-dense", MoeHybrid = "moe-hybrid";
    public static TheoryData<string> Families => new() { Llama, HybridDense, MoeHybrid };

    protected abstract HiddenStateTapFixture Open(string family);

    private sealed record Run(List<int> Tokens, List<float[]> Logits, IKvCache Kv);

    private static float[] LastRow(ITensor t, int vocab)
    {
        var all = new float[vocab];
        int rows = (int)(t.ElementCount / vocab);
        unsafe
        {
            new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(all);
        }
        return all;
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    /// <summary>Prefill then <paramref name="steps"/> greedy steps; keeps every logits row.</summary>
    private static Run Generate(HiddenStateTapFixture fx, int steps)
    {
        var kv = fx.NewKv();
        var tokens = new List<int>();
        var logits = new List<float[]>();
        int[] cur = fx.Prompt;
        int pos = 0;
        for (int s = 0; s <= steps; s++)
        {
            var positions = new int[cur.Length];
            for (int i = 0; i < cur.Length; i++) positions[i] = pos + i;
            using var t = fx.Model.Forward(cur, positions, -1, kv);
            var row = LastRow(t, fx.Vocab);
            logits.Add(row);
            int next = ArgMax(row);
            tokens.Add(next);
            pos += cur.Length;
            cur = [next];
        }
        return new Run(tokens, logits, kv);
    }

    private static int[] DepthLayers(int n)
    {
        var set = new SortedSet<int>();
        foreach (double f in new[] { 0.25, 0.5, 0.75, 1.0 })
            set.Add(Math.Max(0, (int)Math.Ceiling(n * f) - 1));
        return [.. set];
    }

    private static readonly HookPoint[] AllPoints =
        [HookPoint.PostEmbedding, HookPoint.PostLayer, HookPoint.PreFinalNorm, HookPoint.PreLmHead];

    /// <summary>20 greedy steps with a multi-depth tap are identical to the same run without it: tokens, logits, KV bytes.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void MultiLayerTap_DoesNotPerturb_TokensLogitsOrKv(string family)
    {
        using var plain = Open(family);
        using var tapped = Open(family);
        var a = Generate(plain, 20);

        var seen = new List<(HookPoint Point, int Layer, int Row, int Pos, int Len)>();
        int[] layers = DepthLayers(tapped.NumLayers);
        HiddenStateTapRegistry.PathCounters.Reset();
        List<int> bTokens; List<float[]> bLogits; float[] bKv;
        using (tapped.Host.RegisterTap(new HiddenStateTapSubscription(layers, AllPoints),
            s => seen.Add((s.Point, s.LayerIndex, s.RowIndex, s.TokenPosition, s.Data.Length))))
        {
            var b = Generate(tapped, 20);
            bTokens = b.Tokens; bLogits = b.Logits; bKv = tapped.KvBytes(b.Kv);
        }

        Assert.Equal(a.Tokens, bTokens);
        for (int i = 0; i < a.Logits.Count; i++)
        {
            if (plain.BitExact) Assert.True(a.Logits[i].AsSpan().SequenceEqual(bLogits[i]), $"logits differ at step {i}");
            else Assert.All(a.Logits[i].Zip(bLogits[i]), p => Assert.InRange(p.First - p.Second, -1e-2f, 1e-2f));
        }
        if (plain.BitExact) Assert.True(plain.KvBytes(a.Kv).AsSpan().SequenceEqual(bKv), "KV bytes differ");

        // 21 forwards (prefill + 20 decode); each delivers PostEmbedding + one PostLayer per subscribed layer + PreFinalNorm + PreLmHead.
        int perForward = 1 + layers.Length + 1 + 1;
        Assert.Equal(21 * perForward, seen.Count);
        Assert.Equal(21, HiddenStateTapRegistry.PathCounters.TapForwards);
        Assert.All(seen, s => Assert.Equal(tapped.Hidden, s.Len));
        Assert.Equal(layers, seen.Where(s => s.Point == HookPoint.PostLayer).Select(s => s.Layer).Distinct().OrderBy(x => x).ToArray());
        Assert.All(seen.Where(s => s.Point != HookPoint.PostLayer), s => Assert.Equal(-1, s.Layer));
        // Rows default to Last: the prefill's last row is prompt.Length - 1; decode step k is at prompt.Length + k - 1.
        int P = plain.Prompt.Length;
        var prefill = seen.Take(perForward).ToList();
        Assert.All(prefill, s => { Assert.Equal(P - 1, s.Row); Assert.Equal(P - 1, s.Pos); });
        var decode = seen.Skip(perForward).Chunk(perForward).ToList();
        for (int k = 0; k < decode.Count; k++)
            Assert.All(decode[k], s => { Assert.Equal(0, s.Row); Assert.Equal(P + k, s.Pos); });
        // Delivery order within one forward: embedding, layers ascending, pre-final-norm, pre-lm-head.
        Assert.Equal(HookPoint.PostEmbedding, prefill[0].Point);
        Assert.Equal(HookPoint.PreFinalNorm, prefill[^2].Point);
        Assert.Equal(HookPoint.PreLmHead, prefill[^1].Point);
    }

    /// <summary>Rows All delivers one sample per row with the right positions; Last delivers only the final one. The final points agree where they must differ or match.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void Rows_LastVsAll_AndFinalPointsRelate(string family)
    {
        using var fx = Open(family);
        int last = fx.NumLayers - 1;
        var lastRows = new List<(HookPoint P, int Row, int Pos, float[] D)>();
        var allRows = new List<(HookPoint P, int Row, int Pos, float[] D)>();
        using var h1 = fx.Host.RegisterTap(new HiddenStateTapSubscription([last], AllPoints),
            s => lastRows.Add((s.Point, s.RowIndex, s.TokenPosition, s.Data.ToArray())));
        using var h2 = fx.Host.RegisterTap(new HiddenStateTapSubscription([last], AllPoints, TapRows.All),
            s => allRows.Add((s.Point, s.RowIndex, s.TokenPosition, s.Data.ToArray())));
        var prompt = fx.Prompt;
        int[] positions = Enumerable.Range(0, prompt.Length).ToArray();
        using var kv = fx.NewKv();
        using (fx.Model.Forward(prompt, positions, -1, kv)) { }

        int P = prompt.Length;
        Assert.Equal(AllPoints.Length, lastRows.Count);
        Assert.All(lastRows, r => { Assert.Equal(P - 1, r.Row); Assert.Equal(P - 1, r.Pos); });
        foreach (var point in AllPoints)
        {
            var rows = allRows.Where(r => r.P == point).ToList();
            Assert.Equal(P, rows.Count);
            Assert.Equal(Enumerable.Range(0, P), rows.Select(r => r.Row));
            Assert.Equal(positions, rows.Select(r => r.Pos));
            // The Last tap's sample is the All tap's final row, bit for bit.
            Assert.True(rows[^1].D.AsSpan().SequenceEqual(lastRows.Single(r => r.P == point).D));
        }
        var lastBy = lastRows.ToDictionary(r => r.P, r => r.D);
        // PreFinalNorm is the residual leaving the last layer: the same row as PostLayer(last).
        Assert.True(lastBy[HookPoint.PostLayer].AsSpan().SequenceEqual(lastBy[HookPoint.PreFinalNorm]));
        // PreLmHead is post-norm: it must differ from the pre-norm residual, and be finite.
        Assert.False(lastBy[HookPoint.PreLmHead].AsSpan().SequenceEqual(lastBy[HookPoint.PreFinalNorm]));
        Assert.All(lastBy[HookPoint.PreLmHead], v => Assert.True(float.IsFinite(v)));
        // The embedding row differs from the deep residual (a tap that always returned one buffer would not).
        Assert.False(lastBy[HookPoint.PostEmbedding].AsSpan().SequenceEqual(lastBy[HookPoint.PostLayer]));
    }

    /// <summary>A tap on layer L never fires for another layer, and a tap on nothing delivers nothing.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void Tap_FiresOnlyOnSubscribedLayers(string family)
    {
        using var fx = Open(family);
        int target = fx.NumLayers / 2;
        var seen = new List<int>();
        using var h = fx.Host.RegisterTap(new HiddenStateTapSubscription([target], [HookPoint.PostLayer]), s => seen.Add(s.LayerIndex));
        using var kv = fx.NewKv();
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        Assert.Equal([target], seen);
    }

    /// <summary>Two taps on different layers: each sees only its own (a single tap cannot tell a per-entry filter from a union filter).</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void TwoTaps_OnDifferentLayers_EachSeeOnlyTheirOwn(string family)
    {
        using var fx = Open(family);
        var a = new List<int>(); var b = new List<int>();
        using var ha = fx.Host.RegisterTap(new HiddenStateTapSubscription([0], [HookPoint.PostLayer]), s => a.Add(s.LayerIndex));
        using var hb = fx.Host.RegisterTap(new HiddenStateTapSubscription([fx.NumLayers - 1], [HookPoint.PostLayer]), s => b.Add(s.LayerIndex));
        using var kv = fx.NewKv();
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        Assert.Equal([0], a);
        Assert.Equal([fx.NumLayers - 1], b);
    }

    /// <summary>Disposed taps stop delivering; with no tap attached nothing is dispatched, no counter moves and decode allocates nothing extra.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void Unattached_HasNoDispatch_NoCounters_NoExtraAllocation(string family)
    {
        using var never = Open(family);            // a registry that was never used
        using var used = Open(family);             // a registry that had a tap and lost it
        int calls = 0;
        used.Host.RegisterTap(new HiddenStateTapSubscription(null, AllPoints), _ => calls++).Dispose();

        HiddenStateTapRegistry.PathCounters.Reset();
        long a = Measure(never, 16);
        long b = Measure(used, 16);
        Assert.Equal(0, calls);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.TapForwards);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.PackedBatchFallbacks);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.AbortCheckpoints);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.AbortedForwards);
        Assert.Equal(a, b);
    }

    private static long Measure(HiddenStateTapFixture fx, int steps)
    {
        // Warm up (JIT, pools, first-use allocation) with the same shape, then measure N decode forwards on this thread.
        using var kv = fx.NewKv();
        var prompt = fx.Prompt;
        int pos = prompt.Length;
        using (fx.Model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kv)) { }
        int[] tok = [prompt[0]];
        int[] p = new int[1];
        for (int i = 0; i < 4; i++) { p[0] = pos++; using (fx.Model.Forward(tok, p, -1, kv)) { } }
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < steps; i++) { p[0] = pos++; using (fx.Model.Forward(tok, p, -1, kv)) { } }
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void DisposedTap_StopsDelivering(string family)
    {
        using var fx = Open(family);
        int n = 0;
        var h = fx.Host.RegisterTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => n++);
        using var kv = fx.NewKv();
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        Assert.Equal(1, n);
        h.Dispose();
        using (fx.Model.Forward([fx.Prompt[0]], [fx.Prompt.Length], -1, kv)) { }
        Assert.Equal(1, n);
    }

    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void Registration_Validates(string family)
    {
        using var fx = Open(family);
        Assert.Throws<NotSupportedException>(() =>
            fx.Host.RegisterTap(new HiddenStateTapSubscription(null, [HookPoint.PreAttention]), _ => { }));
        Assert.Throws<NotSupportedException>(() =>
            fx.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, [HookPoint.PostFfn]), _ => HookResult.Continue));
        Assert.Throws<ArgumentException>(() =>
            fx.Host.RegisterTap(new HiddenStateTapSubscription([fx.NumLayers], [HookPoint.PostLayer]), _ => { }));
        Assert.Throws<ArgumentException>(() =>
            fx.Host.RegisterTap(new HiddenStateTapSubscription(null, []), _ => { }));
    }

    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void BufferProvider_ReceivesTheRow(string family)
    {
        using var fx = Open(family);
        var mine = new float[fx.Hidden];
        float[]? copy = null;
        using var h = fx.Host.RegisterTap(
            new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding], TapRows.Last, _ => mine),
            s => copy = s.Data.ToArray());
        using var kv = fx.NewKv();
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        Assert.Equal(copy, mine);
        Assert.Contains(mine, v => v != 0f);
    }

    // ── Abort ─────────────────────────────────────────────────────────────────

    private sealed class Aborter(HookPoint point, int layer)
    {
        public bool Armed;
        public int Hits;
        public HookResult OnSample(HiddenStateSample s)
        {
            if (Armed && s.Point == point && s.LayerIndex == layer) { Hits++; return HookResult.AbortForward; }
            return HookResult.Continue;
        }
    }

    public static TheoryData<string, string> AbortCases()
    {
        var d = new TheoryData<string, string>();
        foreach (var f in new[] { Llama, HybridDense, MoeHybrid })
            foreach (var where in new[] { "last-layer", "embedding", "pre-lm-head" })
                d.Add(f, where);
        return d;
    }

    private static (HookPoint, int) AbortSite(string where, int numLayers) => where switch
    {
        "last-layer" => (HookPoint.PostLayer, numLayers - 1),
        "embedding" => (HookPoint.PostEmbedding, -1),
        _ => (HookPoint.PreLmHead, -1),
    };

    /// <summary>
    /// An aborted decode forward returns false without throwing, rolls KV back, restores recurrent state (the abort at the last layer
    /// happens after every GDN layer advanced it), and the next forward is bit-identical to a model that never ran the aborted one.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(AbortCases))]
    public void Abort_TryForward_LeavesModelAsIfNeverStarted(string family, string where)
    {
        using var aborted = Open(family);
        Skip.IfNot(aborted.SupportsAbortableTaps, "abortable taps are not available on this host");
        using var control = Open(family);
        var (point, layer) = AbortSite(where, aborted.NumLayers);
        var sut = new Aborter(point, layer);
        var kvA = aborted.NewKv();
        var kvC = control.NewKv();
        var prompt = aborted.Prompt;
        int[] pp = Enumerable.Range(0, prompt.Length).ToArray();
        using (aborted.Model.Forward(prompt, pp, -1, kvA)) { }
        using (control.Model.Forward(prompt, pp, -1, kvC)) { }

        using var h = aborted.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, AllPoints), sut.OnSample);
        HiddenStateTapRegistry.PathCounters.Reset();
        sut.Armed = true;
        int[] tok = [prompt[1]];
        int[] pos = [prompt.Length];
        bool ok = aborted.Host.TryForward(tok, pos, -1, kvA, out var logits, out var abort);
        Assert.False(ok);
        Assert.Null(logits);
        Assert.Equal(new ForwardAbortInfo(layer, point), abort);
        Assert.Equal(1, sut.Hits);
        Assert.Equal(prompt.Length, kvA.CurrentLength);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.AbortedForwards);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.AbortCheckpoints);

        // The legacy entry point throws instead, with the same site.
        var ex = Assert.Throws<ForwardAbortedException>(() => aborted.Model.Forward(tok, pos, -1, kvA));
        Assert.Equal(layer, ex.Layer);
        Assert.Equal(point, ex.Point);
        Assert.Equal(prompt.Length, kvA.CurrentLength);

        // Disarm and continue: three more steps must equal the control that never aborted anything.
        sut.Armed = false;
        for (int step = 0; step < 3; step++)
        {
            int[] t = [prompt[(step + 2) % prompt.Length]];
            int[] p = [prompt.Length + step];
            using var la = aborted.Model.Forward(t, p, -1, kvA);
            using var lc = control.Model.Forward(t, p, -1, kvC);
            Assert.True(LastRow(la, aborted.Vocab).AsSpan().SequenceEqual(LastRow(lc, control.Vocab)), $"logits differ at step {step}");
        }
        Assert.Equal(kvC.CurrentLength, kvA.CurrentLength);
        Assert.True(control.KvBytes(kvC).AsSpan().SequenceEqual(aborted.KvBytes(kvA)), "KV bytes differ after abort");
        kvA.Dispose(); kvC.Dispose();
    }

    /// <summary>An abort during a multi-token prefill leaves KV empty and the next full prefill equal to a fresh one.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void Abort_DuringPrefill_RollsBackToEmpty(string family)
    {
        using var aborted = Open(family);
        Skip.IfNot(aborted.SupportsAbortableTaps, "abortable taps are not available on this host");
        using var control = Open(family);
        var sut = new Aborter(HookPoint.PostLayer, aborted.NumLayers - 1) { Armed = true };
        using var h = aborted.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, AllPoints), sut.OnSample);
        using var kvA = aborted.NewKv();
        using var kvC = control.NewKv();
        int[] pp = Enumerable.Range(0, aborted.Prompt.Length).ToArray();
        Assert.False(aborted.Host.TryForward(aborted.Prompt, pp, -1, kvA, out _, out _));
        Assert.Equal(0, kvA.CurrentLength);
        sut.Armed = false;
        using var la = aborted.Model.Forward(aborted.Prompt, pp, -1, kvA);
        using var lc = control.Model.Forward(control.Prompt, pp, -1, kvC);
        Assert.True(LastRow(la, aborted.Vocab).AsSpan().SequenceEqual(LastRow(lc, control.Vocab)));
        Assert.True(control.KvBytes(kvC).AsSpan().SequenceEqual(aborted.KvBytes(kvA)));
    }

    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void AbortableTap_ReturningReplace_IsRejected(string family)
    {
        using var fx = Open(family);
        Skip.IfNot(fx.SupportsAbortableTaps, "abortable taps are not available on this host");
        using var h = fx.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]),
            _ => HookResult.Replace(new float[fx.Hidden]));
        using var kv = fx.NewKv();
        Assert.Throws<InvalidOperationException>(() => fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv));
    }

    /// <summary>
    /// Ruling 3 (#1920): a windowed KV-cache (compact sliding-window layers) refuses an abortable tap at the FIRST abortable forward,
    /// independent of its length and of whether this forward would compact: NotSupportedException naming the cache, before any KV,
    /// recurrent or tap state is touched (the next forward equals a control that never saw the refused call).
    /// </summary>
    [SkippableFact]
    public void Abort_WindowedCache_RefusedAtFirstAbortableForward_EvenContiguous_NoStateChange()
    {
        using var fx = Open(Llama);
        Skip.IfNot(fx.SupportsAbortableTaps, "abortable taps are not available on this host");
        using var kvA = fx.NewWindowedKv(1024);
        Skip.If(kvA is null, "no sliding-window cache on this host");
        using var control = Open(Llama);
        using var kvC = control.NewWindowedKv(1024)!;
        int[] pp = Enumerable.Range(0, fx.Prompt.Length).ToArray();
        using (fx.Model.Forward(fx.Prompt, pp, -1, kvA)) { }
        using (control.Model.Forward(control.Prompt, pp, -1, kvC)) { }

        var sut = new Aborter(HookPoint.PostLayer, fx.NumLayers - 1) { Armed = true };
        using var h = fx.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, AllPoints), sut.OnSample);
        HiddenStateTapRegistry.PathCounters.Reset();
        // A contiguous decode step: with only 8 rows in a 1024-row cache nothing would compact, yet it is refused (not lazily).
        var ex = Assert.Throws<NotSupportedException>(() => fx.Host.TryForward([fx.Prompt[1]], [fx.Prompt.Length], -1, kvA, out _, out _));
        Assert.Contains(kvA!.GetType().Name, ex.Message);
        Assert.Contains("sliding-window", ex.Message);
        Assert.Equal(fx.Prompt.Length, kvA.CurrentLength);
        Assert.Equal(0, sut.Hits);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.AbortedForwards);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.AbortCheckpoints);

        sut.Armed = false;
        h.Dispose();   // the refusal is per forward while an abortable tap is attached; detach to continue
        int[] t = [fx.Prompt[2]];
        int[] p = [fx.Prompt.Length];
        using var la = fx.Model.Forward(t, p, -1, kvA);
        using var lc = control.Model.Forward(t, p, -1, kvC);
        Assert.True(LastRow(la, fx.Vocab).AsSpan().SequenceEqual(LastRow(lc, control.Vocab)), "logits differ after the refused forward");
        Assert.Equal(kvC.CurrentLength, kvA.CurrentLength);
    }

    /// <summary>A read-only tap is fine on a windowed cache (only an abort needs the rollback).</summary>
    [SkippableFact]
    public void ReadOnlyTap_OnWindowedCache_Works()
    {
        using var fx = Open(Llama);
        using var kv = fx.NewWindowedKv(1024);
        Skip.If(kv is null, "no sliding-window cache on this host");
        int n = 0;
        using var h = fx.Host.RegisterTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => n++);
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        Assert.Equal(1, n);
    }

    /// <summary>Ruling 3, registration half: a model whose config uses sliding-window attention refuses an abortable tap at registration (a read-only tap still registers).</summary>
    [SkippableFact]
    public void RegisterAbortableTap_OnSlidingWindowConfigModel_IsRefusedAtRegistration()
    {
        using var fx = Open(Llama);
        var host = fx.NewSlidingWindowConfigHost();
        Skip.If(host is null, "no sliding-window config model on this host");
        var ex = Assert.Throws<NotSupportedException>(() =>
            host!.RegisterAbortableTap(new HiddenStateTapSubscription(null, AllPoints), _ => HookResult.Continue));
        Assert.Contains("sliding-window", ex.Message);
        host.RegisterTap(new HiddenStateTapSubscription(null, AllPoints), _ => { }).Dispose();
    }

    /// <summary>Read-only taps never checkpoint: only an abortable tap pays for entry state.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void ReadOnlyTap_TakesNoCheckpoint(string family)
    {
        using var fx = Open(family);
        using var h = fx.Host.RegisterTap(new HiddenStateTapSubscription(null, AllPoints), _ => { });
        using var kv = fx.NewKv();
        HiddenStateTapRegistry.PathCounters.Reset();
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.AbortCheckpoints);
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.TapForwards);
    }

    // ── ForwardBatch ──────────────────────────────────────────────────────────

    /// <summary>With a tap attached a multi-sequence ForwardBatch goes per sequence (counter on the packed model), each sequence reporting its own id.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void ForwardBatch_WithTap_FallsBackPerSequence(string family)
    {
        using var plain = Open(family);
        using var tapped = Open(family);

        IReadOnlyList<ITensor> Batch(HiddenStateTapFixture fx, IKvCache k0, IKvCache k1, IModel m)
        {
            int n = fx.Prompt.Length;
            var reqs = new[]
            {
                new SequenceForwardRequest { TokenIds = fx.Prompt, Positions = Enumerable.Range(0, n).ToArray(), KvCache = k0,
                    GdnState = m.CreateSequenceState() as IGdnState },
                new SequenceForwardRequest { TokenIds = fx.Prompt[..(n - 2)], Positions = Enumerable.Range(0, n - 2).ToArray(), KvCache = k1,
                    GdnState = m.CreateSequenceState() as IGdnState },
            };
            return m.ForwardBatch(reqs, -1);
        }

        using var a0 = plain.NewKv(); using var a1 = plain.NewKv();
        using var b0 = tapped.NewKv(); using var b1 = tapped.NewKv();
        HiddenStateTapRegistry.PathCounters.Reset();
        var expect = Batch(plain, a0, a1, plain.Model);
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.PackedBatchFallbacks);

        var ids = new HashSet<int>();
        using var h = tapped.Host.RegisterTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), s => ids.Add(s.SequenceId));
        var got = Batch(tapped, b0, b1, tapped.Model);
        Assert.Equal(tapped.ExpectsPackedBatchFallback ? 1 : 0, HiddenStateTapRegistry.PathCounters.PackedBatchFallbacks);
        Assert.Equal([0, 1], ids.OrderBy(x => x));
        for (int i = 0; i < 2; i++)
        {
            var e = LastRow(expect[i], plain.Vocab);
            var g = LastRow(got[i], tapped.Vocab);
            if (plain.BatchBitExact) Assert.True(e.AsSpan().SequenceEqual(g), $"seq {i} logits differ");
            else
            {
                Assert.All(e.Zip(g), p => Assert.InRange(p.First - p.Second, -1e-2f, 1e-2f));
                Assert.Equal(ArgMax(e), ArgMax(g));
            }
        }
        foreach (var t in expect) t.Dispose();
        foreach (var t in got) t.Dispose();
    }
}
