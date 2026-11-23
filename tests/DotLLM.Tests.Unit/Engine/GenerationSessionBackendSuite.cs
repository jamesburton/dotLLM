using System.Reflection;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Engine;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The #1919 backend-parity assertions, written once and run on every backend by a derived class that supplies
/// <see cref="Open"/>: CPU always, Vulkan when a device is present, CUDA only where a CUDA GPU is (written, not run, on the
/// Strix Halo dev box). Families: attention-only Llama, GDN-hybrid dense, GDN-hybrid MoE.
/// </summary>
public abstract partial class GenerationSessionBackendSuite
{
    internal abstract GenerationSessionFixture Open(string family, int? headDim = null);

    private static float[] Logits(GenerationSession s)
    {
        var buf = new float[s.VocabSize];
        s.CopyNextLogits(buf);
        return buf;
    }

    /// <summary>
    /// Safety factor on the control: the bound is <see cref="ControlFactor"/> x the control's max-abs logit difference. 2 because the
    /// control is one realisation of the noise on one token sequence while the compared pair is another, so the two are the same
    /// order but not equal; 2 absorbs that without letting a real defect through (a dropped rollback or restore moves logits by
    /// orders of magnitude, see the mutants in the PR body). When the control is exactly 0 the comparison must be bit-identical:
    /// there is no floor.
    /// </summary>
    private const float ControlFactor = 2f;

    /// <summary>A control above this means the instrument itself is broken (or noisy beyond any parity claim): the run fails instead of deriving a runaway bound.</summary>
    private const float ControlSanityCeiling = 1e-1f;

    /// <summary>
    /// Derived tolerance (James 2026-11-22, consistent with #1928): no fixed number, and the control does NOT run through the code under
    /// test. Everything here is raw <c>IModel.Forward</c> on a fresh cache from the generator's KV factory, never <see cref="GenerationSession"/>,
    /// so a defect in the session's append or prefill path cannot inflate the control and the compared diff together.
    /// Two arms, same model, same tokens, max |logit diff| against one batched prefill of all tokens:
    /// (1) the genuine 1-ULP arm (<see cref="ResidualUlpArm"/>, the same instrument on CPU, Vulkan and CUDA): the same prefill with every element
    /// of the last token's residual row moved by exactly one ULP of the residual's storage dtype (F32; F16 on the CUDA dense model) after layer 0, the
    /// "one rounding of one input" sensitivity of #1928, through each backend's own PostLayer seam;
    /// (2) the prefill-vs-decode path split (prefill of the first half, then every remaining token as a single decode row), the path pair
    /// every comparison here crosses. Control = max of the arms; the bound is <see cref="ControlFactor"/> x control. A 1-ULP perturbation can be
    /// absorbed on one particular sequence, so observability is asserted bitwise once per backend and family by
    /// <see cref="OneUlpArm_IsObservable"/> (an inert arm must fail the run), not per call.
    /// </summary>
    private static float ControlMaxAbsDiff(GenerationSessionFixture oracle, int[] tokens)
    {
        float[] full = RawLastLogits(oracle, tokens, tokens.Length, perturb: false);
        float raw = MaxAbsDiff(full, RawLastLogits(oracle, tokens, Math.Max(1, tokens.Length / 2), perturb: false));
        float ulp = oracle.Arm is null ? 0f : MaxAbsDiff(full, RawLastLogits(oracle, tokens, tokens.Length, perturb: true));   // may be 0 on one sequence; OneUlpArm_IsObservable proves the arm can disagree
        float control = Math.Max(raw, ulp);
        string? log = Environment.GetEnvironmentVariable("DOTLLM_SESSION_CONTROL_LOG");
        if (!string.IsNullOrEmpty(log))
            File.AppendAllText(log, $"{oracle.Backend} n={tokens.Length} ARMS ulp1={ulp:E3} raw={raw:E3}{Environment.NewLine}");
        Assert.True(control <= ControlSanityCeiling, $"{oracle.Backend}: control {control:E3} exceeds the sanity ceiling {ControlSanityCeiling:E1}");
        return control;
    }

    /// <summary>Last-row logits after a raw forward of <paramref name="tokens"/>: the first <paramref name="firstChunk"/> tokens batched, the rest one row at a time.</summary>
    private static unsafe float[] RawLastLogits(GenerationSessionFixture fx, int[] tokens, int firstChunk, bool perturb)
    {
        var model = fx.Model;
        using var kv = fx.NewKvCache(tokens.Length + 1);
        if (model.RequiresPerSequenceState) model.ResetSequenceState();
        var arm = perturb ? fx.Arm! : null;
        arm?.Attach();
        try
        {
            var pos = new int[tokens.Length];
            for (int i = 0; i < pos.Length; i++) pos[i] = i;
            float[]? last = null;
            for (int start = 0, len = firstChunk; start < tokens.Length; start += len, len = 1)
            {
                using var logits = model.Forward(tokens.AsSpan(start, len), pos.AsSpan(start, len), -1, kv, null, null, true);
                if (start + len == tokens.Length)
                {
                    int vocab = model.Config.VocabSize;
                    last = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
                }
                arm?.Detach();   // the perturbation is one-shot; later rows run clean
            }
            return last!;
        }
        finally
        {
            arm?.Detach();
            if (model.RequiresPerSequenceState) model.ResetSequenceState();
        }
    }

    /// <summary>
    /// The control's 1-ULP arm must be able to disagree (James 2026-11-22: a control that cannot disagree proves nothing): on every backend and
    /// family the model really called the perturber exactly once, and at least one logit differs BITWISE from the clean forward.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public void OneUlpArm_IsObservable(string family)
    {
        using var fx = Open(family);
        var arm = fx.Arm;
        Assert.NotNull(arm);
        int[] tokens = [.. fx.Prompt, .. fx.Observation];
        float[] clean = RawLastLogits(fx, tokens, tokens.Length, perturb: false);
        int firedBefore = arm.Fired;
        float[] moved = RawLastLogits(fx, tokens, tokens.Length, perturb: true);
        Assert.Equal(1, arm.Fired - firedBefore);   // the hook ran (an uncalled hook is a dead seam, not an absorbed perturbation)
        bool anyBitwise = false;
        for (int i = 0; i < clean.Length && !anyBitwise; i++)
            anyBitwise = BitConverter.SingleToInt32Bits(clean[i]) != BitConverter.SingleToInt32Bits(moved[i]);
        Assert.True(anyBitwise, $"{fx.Backend}/{family}: a 1-ULP residual perturbation left every logit bit-identical (inert control arm)");
    }

    /// <summary>
    /// The seam is inert when unset (James 2026-11-22). With the modifier cleared (the hook never attached): (1) after an attach-perturb-detach
    /// cycle the logits are BIT-identical to before it, on the prefill and the decode path (no residue: a latched bypass, a stale descriptor or
    /// graph, a left-behind modifier); (2) managed bytes per forward and the backend's path fingerprint (Vulkan fence waits, CUDA eager-decode /
    /// graph-launch counters) are identical before and after; (3) an unset forward allocates no more managed bytes, and waits on the device no more
    /// often, than a forward with a modifier attached that never applies (that one pays an AppliesTo call per layer and, on the GPU backends, leaves
    /// the fused / graphed decode), so a hook that allocates or syncs on the unset path is caught on the backends where attaching changes nothing
    /// else (CPU), and by (1)-(2) everywhere it leaves residue.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public void UnsetHook_IsInert(string family)
    {
        using var fx = Open(family);
        var arm = fx.Arm!;
        int[] tokens = [.. fx.Prompt, .. fx.Observation];
        object? saved = arm.ClearModifier();
        try
        {
            foreach (int firstChunk in new[] { tokens.Length, tokens.Length / 2 })   // prefill-only, then prefill + decode rows
            {
                for (int w = 0; w < 2; w++) RawLastLogits(fx, tokens, firstChunk, perturb: false);   // warm: pipelines, scratch, JIT
                var (before, bytesBefore, fpBefore) = MeasureUnset(fx, tokens, firstChunk);
                RawLastLogits(fx, tokens, firstChunk, perturb: true);
                var (after, bytesAfter, fpAfter) = MeasureUnset(fx, tokens, firstChunk);
                for (int i = 0; i < before.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(before[i]) == BitConverter.SingleToInt32Bits(after[i]), $"{fx.Backend}/{family} chunk={firstChunk} logit {i}: {before[i]} vs {after[i]} after an attach/detach cycle");
                if (fx.Backend != SessionBackend.Cuda) Assert.Equal(bytesBefore, bytesAfter);   // CUDA: bounded by the same-run spread in (3) instead (allocation noise there is real)
                Assert.Equal(fpBefore, fpAfter);

                // (3) the unset path against a modifier that is attached and never applies. Arms alternate (inapplicable, unset, inapplicable),
                // each arm warmed and measured as a minimum over several forwards. The noise floor is a same-run control: the spread of the
                // repeated identical-arm minimums (the two attached-inapplicable runs, the three unset runs). It is 0 on CPU (exact), so a hook
                // that allocates on the unset path dies there; on a GPU, first-call / pooled-buffer / ordering noise (observed: CUDA hybrid-dense
                // unset 8624 B vs inapplicable 7192 B) is bounded by what identical arms themselves disagree by.
                long inapplicableBytes, inapplicable2; string fpInapplicable;
                arm.Attach(layer: int.MaxValue);
                try
                {
                    RawLastLogits(fx, tokens, firstChunk, perturb: false);
                    (_, inapplicableBytes, fpInapplicable) = MeasureUnset(fx, tokens, firstChunk);
                }
                finally { arm.Detach(); }
                var (_, unset3, fpUnset3) = MeasureUnset(fx, tokens, firstChunk);
                arm.Attach(layer: int.MaxValue);
                try { (_, inapplicable2, _) = MeasureUnset(fx, tokens, firstChunk); }
                finally { arm.Detach(); }
                long[] unsetRuns = [bytesBefore, bytesAfter, unset3];
                long[] inappRuns = [inapplicableBytes, inapplicable2];
                long spread = Math.Max(unsetRuns.Max() - unsetRuns.Min(), inappRuns.Max() - inappRuns.Min());
                Assert.True(unsetRuns.Min() <= inappRuns.Min() + spread, $"{fx.Backend}/{family}: the unset forward allocates {unsetRuns.Min()} B (runs {string.Join("/", unsetRuns)}), more than the attached-inapplicable {inappRuns.Min()} B (runs {string.Join("/", inappRuns)}) plus the same-run spread {spread} B");
                if (fx.Backend == SessionBackend.Cpu) Assert.Equal(0, spread);   // the CPU path is deterministic: the control must be exactly 0, so the comparison is exact there
                Assert.True(FenceWaits(fpBefore) <= FenceWaits(fpInapplicable), $"{fx.Backend}/{family}: the unset forward waited more on the device ({fpBefore}) than the attached-inapplicable one ({fpInapplicable})");
                string? log = Environment.GetEnvironmentVariable("DOTLLM_SESSION_CONTROL_LOG");
                if (!string.IsNullOrEmpty(log))
                    File.AppendAllText(log, $"{fx.Backend}/{family} UNSET chunk={firstChunk} bytes={bytesBefore} fp[{fpBefore}] | attached-inapplicable bytes={inapplicableBytes} fp[{fpInapplicable}]{Environment.NewLine}");
            }
        }
        finally { arm.RestoreModifier(saved); }
    }

    /// <summary>Min managed bytes over 6 raw forwards (the allocation count of a deterministic path is exact), the last logits, and the path fingerprint delta of one forward.</summary>
    private static (float[] Logits, long Bytes, string Fingerprint) MeasureUnset(GenerationSessionFixture fx, int[] tokens, int firstChunk)
    {
        long min = long.MaxValue;
        float[] logits = [];
        string fp = "";
        for (int i = 0; i < 6; i++)
        {
            string fp0 = PathFingerprint(fx.Model);
            long b0 = GC.GetAllocatedBytesForCurrentThread();
            logits = RawLastLogits(fx, tokens, firstChunk, perturb: false);
            long b = GC.GetAllocatedBytesForCurrentThread() - b0;
            min = Math.Min(min, b);
            fp = Delta(fp0, PathFingerprint(fx.Model));
        }
        return (logits, min, fp);
    }

    /// <summary>Counters that move when a forward syncs or leaves a fast path: Vulkan device fence waits; CUDA eager-decode and decode-graph launch counts.</summary>
    private static string PathFingerprint(IModel model) => model switch
    {
        DotLLM.Cuda.CudaTransformerModel cuda => $"eager={cuda.EagerDecodeCount}",
        DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel hd => $"graph={hd.DecodeGraphCounts}",
        _ when model.GetType().Namespace == "DotLLM.Vulkan" => $"fence={DotLLM.Vulkan.VulkanDevice.SubmitContext.DiagFenceWaits}",
        _ => "",
    };

    /// <summary>Per-counter difference of two fingerprints ("fence=3" then "fence=5" gives "fence=2"; a tuple "(a, b)" is diffed element-wise).</summary>
    private static string Delta(string a, string b)
    {
        if (a == "") return "";
        static string[] Parts(string x) => x.Split(['=', '(', ')', ',', ' '], StringSplitOptions.RemoveEmptyEntries);
        string[] pa = Parts(a), pb = Parts(b);
        return string.Join(",", pb.Select((v, i) => long.TryParse(v, out long n) ? (n - long.Parse(pa[i])).ToString() : v));
    }

    /// <summary>Vulkan fence waits out of a "fence=n" fingerprint delta; 0 when the backend exposes none.</summary>
    private static long FenceWaits(string fp) => fp.StartsWith("fence,", StringComparison.Ordinal) ? long.Parse(fp["fence,".Length..]) : 0;

    private static float MaxAbsDiff(float[] x, float[] y)
    {
        Assert.Equal(x.Length, y.Length);
        float m = 0;
        for (int i = 0; i < x.Length; i++) m = Math.Max(m, Math.Abs(x[i] - y[i]));
        return m;
    }

    /// <summary>Asserts <paramref name="actual"/> is within <see cref="ControlFactor"/> x the control of the fresh-prefill <paramref name="expected"/> (exact when the control is 0).</summary>
    private static Task AssertLogitsMatchFresh(GenerationSessionFixture fx, GenerationSessionFixture oracle, int[] tokens, float[] expected, float[] actual)
    {
        AssertWithinControl(fx, tokens.Length, ControlMaxAbsDiff(oracle, tokens), expected, actual);
        return Task.CompletedTask;
    }

    /// <summary>The assertion half, for a caller that must measure the control while no session is open on the oracle (a recurrent model owns one session's state).</summary>
    private static void AssertWithinControl(GenerationSessionFixture fx, int n, float control, float[] expected, float[] actual)
    {
        float bound = ControlFactor * control;
        float diff = MaxAbsDiff(expected, actual);
        string? log = Environment.GetEnvironmentVariable("DOTLLM_SESSION_CONTROL_LOG");
        if (!string.IsNullOrEmpty(log))
            File.AppendAllText(log, $"{fx.Backend} n={n} control={control:E3} bound={bound:E3} diff={diff:E3}{Environment.NewLine}");
        Assert.True(diff <= bound, $"{fx.Backend}: max logit diff {diff:E3} exceeds {ControlFactor}x control {control:E3} = {bound:E3} (n={n})");
    }

    private static async Task<(float[] Logits, int[] Next)> Fresh(GenerationSessionFixture oracle, IEnumerable<int> tokens)
    {
        await using var s = oracle.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(tokens.ToArray());
        var logits = Logits(s);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
        return (logits, [.. r.TokenIds]);
    }

    private static async Task<int[]> Build(GenerationSession s, GenerationSessionFixture fx)
    {
        await s.PrefillAsync(fx.Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        await s.AppendObservationTokensAsync(fx.Observation);
        return [.. fx.Prompt, .. r.TokenIds, .. fx.Observation];
    }

    /// <summary>[prefill A, generate, append B] leaves the same distribution and continuation as one prefill of A+g+B; the KV length tracks the position.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task MultiSegment_MatchesSinglePass(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);   // recurrent state is model-owned: the oracle needs its own model
        int[] g1, g2, g2Single;
        float[] multiLogits;
        int multiArgMax;
        await using (var multi = fx.NewGenerator().CreateSession(Greedy(), Tag()))
        {
            await multi.PrefillAsync(fx.Prompt);
            Assert.Equal(fx.Prompt.Length, multi.KvLength);
            var r1 = await multi.GenerateUntilBoundaryAsync(new StopAfter(3));
            g1 = [.. r1.TokenIds];
            Assert.Equal(3, g1.Length);
            Assert.Equal(multi.CurrentPosition, multi.KvLength);
            await multi.AppendObservationTokensAsync(fx.Observation);
            Assert.Equal(multi.CurrentPosition, multi.KvLength);
            multiLogits = Logits(multi);
            multiArgMax = multi.NextTokenArgMax();
            g2 = [.. (await multi.GenerateUntilBoundaryAsync(new StopAfter(4))).TokenIds];
        }

        float control = ControlMaxAbsDiff(oracle, [.. fx.Prompt, .. g1, .. fx.Observation]);   // before `single` opens: one session per recurrent model
        await using (var single = oracle.NewGenerator().CreateSession(Greedy(), Tag()))
        {
            await single.PrefillAsync(fx.Prompt.Concat(g1).Concat(fx.Observation).ToArray());
            AssertWithinControl(fx, fx.Prompt.Length + g1.Length + fx.Observation.Length, control, Logits(single), multiLogits);
            Assert.Equal(multiArgMax, single.NextTokenArgMax());
            g2Single = [.. (await single.GenerateUntilBoundaryAsync(new StopAfter(4))).TokenIds];
        }
        Assert.Equal(g2Single, g2);
    }

    /// <summary>After RollbackTo(k) the logits and the continuation equal a fresh session over tokens[0..k): nothing at or past k survives in KV or recurrent state.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackToABoundary_EqualsFreshPrefillOfThePrefix(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        int a = fx.Prompt.Length;

        foreach (int k in new[] { a + 4, a })
        {
            Assert.True(s.CanRollbackTo(k));
            s.RollbackTo(k);
            Assert.Equal(k, s.CurrentPosition);
            Assert.Equal(k, s.KvLength);
            var (freshLogits, freshNext) = await Fresh(oracle, all.Take(k));
            await AssertLogitsMatchFresh(fx, oracle, all[..k], freshLogits, Logits(s));
            var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
            Assert.Equal(freshNext, r.TokenIds);
            s.RollbackTo(k);
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task AfterRollback_AppendingOtherTokens_MatchesFresh(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        s.RollbackTo(fx.Prompt.Length);
        int[] other = [7, 5, 6, 4];
        await s.AppendObservationTokensAsync(other);
        var (freshLogits, freshNext) = await Fresh(oracle, fx.Prompt.Concat(other));
        await AssertLogitsMatchFresh(fx, oracle, [.. fx.Prompt, .. other], freshLogits, Logits(s));
        Assert.Equal(freshNext, (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackToZero_ResetsEverything(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        s.RollbackTo(0);
        Assert.Equal(0, s.CurrentPosition);
        Assert.Equal(0, s.KvLength);
        int[] other = [4, 6, 3, 5, 7];
        await s.PrefillAsync(other);
        var (freshLogits, freshNext) = await Fresh(oracle, other);
        await AssertLogitsMatchFresh(fx, oracle, other, freshLogits, Logits(s));
        Assert.Equal(freshNext, (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    /// <summary>
    /// #1931 review M-6 (James 2026-11-22, rollback exactness): on a recurrent model RollbackTo(boundary) restores the stored logits and
    /// recurrent state, so (a) the restored logits are BIT-identical to the ones captured when the boundary was recorded, and (b) one decoded
    /// token afterwards, which reads the restored GDN state, matches a fresh prefill of the prefix plus that token within the derived bound.
    /// (b) is what proves the state, not just the logits copy, was restored: a stale state leaves (a) intact.
    /// </summary>
    [SkippableTheory]
    [InlineData(HybridDense)]
    [InlineData(MoeHybrid)]
    public async Task Recurrent_Rollback_RestoresBoundaryLogitsBitExactly_AndTheStateDecodesLikeTheOracle(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var atPrompt = Logits(s);
        var seg = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        var atSegment = Logits(s);
        int a = fx.Prompt.Length;
        await s.AppendObservationTokensAsync(fx.Observation);

        foreach (var (k, captured) in new[] { (a + 4, atSegment), (a, atPrompt) })
        {
            s.RollbackTo(k);
            var restored = Logits(s);
            for (int i = 0; i < restored.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(captured[i]) == BitConverter.SingleToInt32Bits(restored[i]), $"k={k} logit {i}: restored {restored[i]} != captured {captured[i]} (must be bit-exact)");

            int next = s.NextTokenArgMax();
            await s.AppendObservationTokensAsync(new[] { next });
            int[] prefix = [.. fx.Prompt, .. seg.TokenIds];
            int[] expectedTokens = [.. prefix.Take(k), next];
            var (freshLogits, _) = await Fresh(oracle, expectedTokens);
            await AssertLogitsMatchFresh(fx, oracle, expectedTokens, freshLogits, Logits(s));
            s.RollbackTo(k);
        }
    }

    /// <summary>An attention-only model rolls back to any position (not just a pause) by re-forwarding one token.</summary>
    [SkippableFact]
    public async Task AttentionModel_RollsBackToAnyPosition()
    {
        using var fx = Open(Llama);
        using var oracle = Open(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        int k = fx.Prompt.Length + 2;   // mid-segment: never a pause
        Assert.True(s.CanRollbackTo(k));
        s.RollbackTo(k);
        Assert.Equal(k, s.KvLength);
        var (freshLogits, freshNext) = await Fresh(oracle, all.Take(k));
        await AssertLogitsMatchFresh(fx, oracle, all[..k], freshLogits, Logits(s));
        Assert.Equal(freshNext, (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    /// <summary>
    /// Path-matched exactness of an attention rollback: RollbackTo(k) re-forwards token k-1 as one decode row over rows [0, k-1) kept from
    /// the earlier forwards, which is exactly what a fresh session does for [prefill k-1 tokens, append token k-1]. Both go through the
    /// same kernels, so they agree to 5e-4 on every backend even where prefill-vs-decode noise (see <c>ControlMaxAbsDiff</c>) is larger.
    /// Bound (James 2026-11-23, after the first CUDA run): DERIVED from the same-run control, no fixed number on GPU. CUDA llama emits F16 logits and
    /// its oracle is not bit-matched (the rolled-back KV rows [0, k-1) came from the ORIGINAL longer prefill, the oracle prefills only k-1 tokens:
    /// different tile shapes, legitimately different F16 roundings; observed k=14 logit 0 -1.6220703 vs -1.6210938 = one F16 ULP). So the bound is
    /// <see cref="ControlFactor"/> x <see cref="ControlMaxAbsDiff"/> (1-ULP residual arm and prefill-vs-decode split), never below the control's own diff; on CPU
    /// the 5e-4 floor holds as well (max of the two). A rollback that keeps an extra KV row or re-forwards at a wrong position moves logits by orders of
    /// magnitude more (mutants in the PR body).
    /// </summary>
    [SkippableFact]
    public async Task AttentionModel_Rollback_MatchesAPathMatchedOracle()
    {
        using var fx = Open(Llama);
        using var oracle = Open(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        foreach (int k in new[] { fx.Prompt.Length + 2, fx.Prompt.Length })
        {
            float control = ControlMaxAbsDiff(oracle, all[..k]);
            float bound = ControlFactor * control;
            if (fx.Backend == SessionBackend.Cpu) bound = Math.Max(bound, 5e-4f);   // the 5e-4 floor holds on CPU only
            s.RollbackTo(k);
            Assert.Equal(k, s.KvLength);   // the logit comparison alone cannot see a skipped truncate (the re-forward rewrites its own row): surviving mutant, killed here
            await using var o = oracle.NewGenerator().CreateSession(Greedy(), Tag());
            await o.PrefillAsync(all.Take(k - 1).ToArray());
            await o.AppendObservationTokensAsync(new[] { all[k - 1] });
            float diff = MaxAbsDiff(Logits(s), Logits(o));
            Assert.True(diff <= bound, $"{fx.Backend} k={k}: rolled back vs path-matched oracle max diff {diff:E3} exceeds {ControlFactor}x control {control:E3} (bound {bound:E3})");
        }
    }

    /// <summary>A recurrent model refuses a position that is not a recorded pause, changing nothing.</summary>
    [SkippableTheory]
    [InlineData(HybridDense)]
    [InlineData(MoeHybrid)]
    public async Task Recurrent_RollbackBetweenBoundaries_IsRefused(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        int between = fx.Prompt.Length + 2;
        Assert.False(s.CanRollbackTo(between));
        Assert.Throws<NotSupportedException>(() => s.RollbackTo(between));
        Assert.Equal(s.CurrentPosition, s.KvLength);
    }

    /// <summary>The sampler's history is session-wide: two segments of 4 sample what one segment of 8 does under a repetition penalty.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task SamplerHistory_IsSessionWide(string family)
    {
        using var fx = Open(family);
        var opts = Greedy() with { RepetitionPenalty = 50f };
        int[] split, whole;
        await using (var s = fx.NewGenerator().CreateSession(opts, Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            var a = await s.GenerateUntilBoundaryAsync(new StopAfter(12));
            var b = await s.GenerateUntilBoundaryAsync(new StopAfter(12));
            split = [.. a.TokenIds, .. b.TokenIds];
        }
        await using (var s = fx.NewGenerator().CreateSession(opts, Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            whole = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(24))).TokenIds];
        }
        Assert.Equal(whole, split);
    }

    /// <summary>Provenance drift refuses every operation on this backend's session too, and restoring the identity makes it usable again.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task Provenance_DriftRefusesEveryOperation(string family)
    {
        using var fx = Open(family);
        var live = Tag();
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag(), () => live);
        await s.PrefillAsync(fx.Prompt);
        live = live with { ActiveModifiersDigest = "mods-1" };
        Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax());
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.AppendObservationTokensAsync(fx.Observation));
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(2)));
        Assert.Throws<InvalidOperationException>(() => s.RollbackTo(0));
        live = Tag();
        s.NextTokenArgMax();   // identity restored: usable again, nothing was disturbed
    }

    /// <summary>
    /// #1919 T6: disposal releases the owned KV-cache and every boundary checkpoint, lets a new session open on the same model, and
    /// the new session starts from clean recurrent state (it reproduces the first session's tokens).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task Dispose_ReleasesKvAndCheckpoints_AndASecondSessionStartsClean(string family)
    {
        using var fx = Open(family);
        var gen = fx.NewGenerator();
        var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int[] first = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds];
        IKvCache kv = s.KvForTest;
        IReadOnlyList<object?> checkpoints = s.CheckpointsForTest;
        if (fx.IsRecurrent) Assert.NotEmpty(checkpoints);
        await s.DisposeAsync();
        await s.DisposeAsync();   // idempotent

        Assert.NotEqual(false, IsDisposed(kv));   // true, or null for a length-only handle that owns nothing
        foreach (object? cp in checkpoints)
            Assert.Throws<ObjectDisposedException>(() => fx.Model.RestoreRecurrentState(cp));
        Assert.Throws<ObjectDisposedException>(() => s.NextTokenArgMax());

        // The generator is usable again: a Generate call is no longer blocked by an open session.
        if (family == Llama) Assert.NotNull(gen.Generate("ab", Greedy(2)));

        await using var s2 = gen.CreateSession(Greedy(), Tag());
        await s2.PrefillAsync(fx.Prompt);
        Assert.Equal(first, (await s2.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    /// <summary>
    /// A prompt and a repeat-guard limit L (2..6) for which the model's own greedy output (60 tokens, guard off) contains a run of L identical
    /// tokens: <c>J</c> is the index of the token that completes the first such run. Synthetic models differ (one cycles, one is constant, one
    /// never repeats a token twice in a row), so the prompt is searched; a family with no repeating prompt skips rather than pass vacuously.
    /// </summary>
    private static async Task<(int[] Prompt, int[] Greedy, int L, int J)> FindRepeat(GenerationSessionFixture fx)
    {
        var candidates = new List<int[]> { fx.Prompt };
        if (fx.Prompt.Length > 0 && fx.Tokenizer.VocabSize < 1000 && fx.Prompt[0] < 20)   // the id-token families: any small ids are valid
            foreach (int a in Enumerable.Range(3, 5)) { candidates.Add([a]); candidates.Add([a, a]); foreach (int b in Enumerable.Range(3, 5)) candidates.Add([a, b, a]); }
        foreach (int[] prompt in candidates)
        {
            await using var s = fx.NewGenerator().CreateSession(Greedy(60) with { RepeatGuardLimit = 0 }, Tag());
            await s.PrefillAsync(prompt);
            int[] g = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(int.MaxValue))).TokenIds];
            for (int limit = 2; limit <= 6; limit++)
                for (int j = limit - 1; j < g.Length; j++)
                    if (Enumerable.Range(j - limit + 1, limit).All(i => g[i] == g[j])) return (prompt, g, limit, j);
        }
        Skip.If(true, "no prompt makes this family's greedy output repeat a token 2-6 times in a row");
        return default;
    }

    /// <summary>
    /// options.RepeatGuardLimit on a session stops where the plain generator stops (#1291 semantics, James 2026-11-23): the same prompt and
    /// limit end after the same tokens, the tripping token kept, <see cref="FinishReason.Length"/>.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RepeatGuard_StopsWhereThePlainGeneratorDoes(string family)
    {
        using var fx = Open(family);
        var (prompt, g, limit, j) = await FindRepeat(fx);
        var opts = Greedy(60) with { RepeatGuardLimit = limit };
        var plain = fx.NewGenerator().Generate(fx.TextOf(prompt), opts);
        Assert.Equal(g[..(j + 1)], plain.GeneratedTokenIds);   // the plain path really tripped, at J
        await using var s = fx.NewGenerator().CreateSession(opts, Tag());
        await s.PrefillAsync(prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(int.MaxValue));
        Assert.Equal(plain.GeneratedTokenIds, r.TokenIds);
        Assert.Equal(FinishReason.Length, r.FinishReason);
    }

    /// <summary>The run survives a pause: the segment ends one token before the limit-th repeat, and the next segment trips on its first token.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RepeatGuard_RunSurvivesAPause_OnEveryBackend(string family)
    {
        using var fx = Open(family);
        var (prompt, g, limit, j) = await FindRepeat(fx);
        await using var s = fx.NewGenerator().CreateSession(Greedy(60) with { RepeatGuardLimit = limit }, Tag());
        await s.PrefillAsync(prompt);
        Assert.Equal(g[..j], (await s.GenerateUntilBoundaryAsync(new StopAfter(j))).TokenIds);   // j tokens: not tripped, run = limit - 1
        Assert.Equal([g[j]], (await s.GenerateUntilBoundaryAsync(new StopAfter(int.MaxValue))).TokenIds);
    }

    /// <summary>
    /// RollbackTo restores the guard: after a trip, rolling back to the pause before the run and sampling again reproduces the same run (a stale
    /// run would trip on its first token). The pause is a recorded boundary, so recurrent models roll back to it as well.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RepeatGuard_RollbackRestoresTheRun_OnEveryBackend(string family)
    {
        using var fx = Open(family);
        var (prompt, g, limit, j) = await FindRepeat(fx);
        int start = j - limit + 1;   // g[start..j] is the run
        await using var s = fx.NewGenerator().CreateSession(Greedy(60) with { RepeatGuardLimit = limit }, Tag());
        await s.PrefillAsync(prompt);
        if (start > 0) await s.GenerateUntilBoundaryAsync(new StopAfter(start));
        int pause = prompt.Length + start;
        Assert.Equal(pause, s.CurrentPosition);
        Assert.Equal(g[start..(j + 1)], (await s.GenerateUntilBoundaryAsync(new StopAfter(int.MaxValue))).TokenIds);   // tripped
        Assert.True(s.CanRollbackTo(pause));
        s.RollbackTo(pause);
        Assert.Equal(g[start..(j + 1)], (await s.GenerateUntilBoundaryAsync(new StopAfter(int.MaxValue))).TokenIds);
    }

    /// <summary>true/false from the cache's private disposed flag; null when the cache has none (a length-only handle that owns no storage).</summary>
    internal static bool? IsDisposed(IKvCache kv)
    {
        var f = kv.GetType().GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic);
        return f is null ? null : (bool)f.GetValue(kv)!;
    }
}

/// <summary>CPU run of <see cref="GenerationSessionBackendSuite"/>.</summary>
public sealed class GenerationSessionCpuSuiteTests : GenerationSessionBackendSuite
{
    internal override GenerationSessionFixture Open(string family, int? headDim = null)
        => Create(family, headDim, SessionBackend.Cpu);
}
