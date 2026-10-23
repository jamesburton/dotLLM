using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Tokenizers;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The Vulkan qwen4exp MTP draft head, per-row recurrent snapshots and verify path (issue #820, stage 2) against the CPU oracle
/// (<c>Qwen4ExpTransformerModel</c> with the same head file). Fixtures are 512-expert / top-10 random checkpoints (the released router class)
/// with a standalone head file shaped like the released <c>mtp-*.gguf</c>.
/// </summary>
/// <remarks>
/// There is no external reference for the head's draft quality (HF ignores <c>mtp.*</c>): these tests pin STRUCTURE and CPU-oracle agreement,
/// and prove by counters that each fast path ran. Speculative decode always emits the verify rows' argmax, so a draft can never change the text.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpMtpTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-vkmtp-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _disposables = [];

    public VulkanQwen4ExpMtpTests(ITestOutputHelper output) { _out = output; Directory.CreateDirectory(_dir); }

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly Q4eGeometry Geo = Qwen4ExpRandomGguf.Experts512;

    /// <summary>Trunk + head, loaded on both backends. <paramref name="quant"/> picks the weights' quantisation (the head's Q8_0 projections are the release's).</summary>
    private Q4eRig Rig(string spvDir, Q4eQuant quant, uint headSeed = 0xBEEF01u, Q4eGeometry? geo = null)
    {
        geo ??= Geo;
        var rig = new Q4eRig(Qwen4ExpRandomGguf.Build(geo, quant), spvDir);
        _disposables.Add(rig);
        string head = Path.Combine(_dir, $"mtp-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(head, Qwen4ExpRandomGguf.BuildMtpOnly(geo, quant, headSeed));
        rig.Cpu.AttachMtpHead(head);
        rig.Vk.AttachMtpHead(head);
        return rig;
    }

    private static int Argmax(ReadOnlySpan<float> row) { int b = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[b]) b = i; return b; }

    private static int[] Prompt(int n, int seed, int vocab) { var r = new Random(seed); return Enumerable.Range(0, n).Select(_ => r.Next(6, vocab)).ToArray(); }

    private static int[] Pos(int n, int start = 0) => Enumerable.Range(start, n).ToArray();

    private static unsafe float[] Rows(ITensor t) { using (t) return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray(); }

    private static float RelL2(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)a[i] * a[i]; }
        return (float)Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static readonly Q4eQuant[] Quants = [Q4eQuant.F32, Q4eQuant.Q8Q51];

    public static IEnumerable<object[]> QuantIndex() { yield return [0]; yield return [1]; }

    [SkippableTheory]
    [MemberData(nameof(QuantIndex))]
    public void AttachAndAbsorb_Prefill_MatchesTheOracleCapturedState(int q)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Quants[q]);
        Assert.True(rig.Vk.SupportsMtp && rig.Vk.HasMtpHead && rig.Vk.SupportsMtpArgMax);
        Assert.Equal(long.MaxValue, rig.Vk.MtpGatePriorBytes);
        int V = rig.Config.VocabSize;
        int[] ids = Prompt(19, 1, V);

        rig.Cpu.ResetSequenceState();
        using var kvC = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(rig.Config), 96);
        using var stC = (Qwen4ExpMtpState)rig.Cpu.CreateMtpState(96)!;
        var cpuLogits = Rows(rig.Cpu.Forward(ids, Pos(ids.Length), -1, kvC, null, stC));

        rig.Vk.ResetSequenceState();
        using var kvV = rig.Vk.CreateKvCache(96);
        using var stV = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(96)!;
        long absorbedBefore = rig.Vk.AbsorbedCells;
        var vkRows = Rows(rig.Vk.Forward(ids, Pos(ids.Length), -1, kvV, null, stV));   // 19 rows -> all rows (verify-sized batch)

        Assert.Equal(ids.Length, stV.CurrentLength);
        Assert.Equal(ids.Length - 1, stV.CellCount);   // token 0 owns no cell
        Assert.Equal(ids.Length, stV.CapturedRowCount);
        Assert.Equal(ids.Length - 1, rig.Vk.AbsorbedCells - absorbedBefore);   // path counter: every cell went through the absorb path
        // the trunk logits agree with the oracle (the MTP side effects must not move them)
        Assert.Equal(cpuLogits.Length, vkRows.Length);
        float rel = RelL2(cpuLogits.AsSpan((ids.Length - 1) * V, V), vkRows.AsSpan((ids.Length - 1) * V, V));
        _out.WriteLine($"quant {q}: last-row relL2 vs oracle {rel:E3}");
        Assert.True(rel < 0.08f);
        // the captured trunk residual equals the oracle's (it is what the head pairs with)
        var cap = stC.CapturedHiddenRows.Slice((ids.Length - 1) * stC.HiddenSize, stC.HiddenSize).ToArray();
        var capV = stV.DownloadCapturedRow(ids.Length - 1);
        float relCap = RelL2(cap, capV);
        _out.WriteLine($"quant {q}: captured residual relL2 {relCap:E3}");
        Assert.True(relCap < 0.08f);
    }

    [SkippableTheory]
    [MemberData(nameof(QuantIndex))]
    public void DraftChain_MatchesTheCpuOracle(int q)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Quants[q]);
        int V = rig.Config.VocabSize;
        int agree = 0, total = 0;
        double worstRel = 0;
        foreach (int seed in new[] { 3, 4, 5, 6 })
        {
            int[] ids = Prompt(14, seed, V);
            rig.Cpu.ResetSequenceState();
            using var kvC = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(rig.Config), 96);
            using var stC = (Qwen4ExpMtpState)rig.Cpu.CreateMtpState(96)!;
            rig.Cpu.Forward(ids, Pos(ids.Length), -1, kvC, null, stC).Dispose();

            rig.Vk.ResetSequenceState();
            using var kvV = rig.Vk.CreateKvCache(96);
            using var stV = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(96)!;
            rig.Vk.Forward(ids, Pos(ids.Length), -1, kvV, null, stV).Dispose();

            long stepsBefore = rig.Vk.DraftSteps;
            int token = 9;   // an arbitrary "last committed token" at position 14
            for (int i = 0; i < 4; i++)
            {
                float[] c = Rows(rig.Cpu.ForwardMtp(stC, token, ids.Length + i));
                float[] v = Rows(rig.Vk.ForwardMtp(stV, token, ids.Length + i));
                worstRel = Math.Max(worstRel, RelL2(c, v));
                total++;
                if (Argmax(c) == Argmax(v)) agree++;
                token = Argmax(c);   // teacher-force the oracle's chain so one flipped token does not cascade
            }
            Assert.Equal(4, rig.Vk.DraftSteps - stepsBefore);   // path counter
        }
        _out.WriteLine($"quant {q}: draft argmax agreement {agree}/{total}, worst logits relL2 {worstRel:E3}");
        Assert.True(worstRel < 0.15, $"draft logits off the oracle: {worstRel:E3}");
        Assert.True(agree >= total - 1, $"draft argmax agreement {agree}/{total}");
    }

    [SkippableFact]
    public void DraftStep_WritesTheSameCellAsAbsorbingThatToken_AndDependsOnTheSeed()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Q4eQuant.F32);
        int V = rig.Config.VocabSize;
        int[] ids = Prompt(9, 7, V);
        int kvStride = Geo.KvHeads * Geo.HeadDim;

        float[] Cell(VulkanQwen4ExpMtpState s, int cell)
        {
            var all = new float[(cell + 1) * kvStride];
            rig.Device.Download(s.Kv.GetKeysBuffer(0), all);
            return all.AsSpan(cell * kvStride, kvStride).ToArray();
        }

        rig.Vk.ResetSequenceState();
        using var kvA = rig.Vk.CreateKvCache(64);
        using var full = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
        rig.Vk.Forward(ids, Pos(9), -1, kvA, null, full).Dispose();      // cells for tokens 1..8

        rig.Vk.ResetSequenceState();
        using var kvB = rig.Vk.CreateKvCache(64);
        using var part = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
        rig.Vk.Forward(ids[..8], Pos(8), -1, kvB, null, part).Dispose(); // cells for tokens 1..7, carry = R_7
        Assert.Equal(7, part.CellCount);
        rig.Vk.ForwardMtp(part, ids[8], 8).Dispose();                     // pairs (R_7, token_8): writes cell 7
        Assert.Equal(8, part.CellCount);

        float[] a = Cell(full, 7), b = Cell(part, 7);
        float rel = RelL2(a, b);
        _out.WriteLine($"draft-step cell vs absorbed cell relL2 {rel:E3}");
        Assert.True(rel < 1e-3f, "a draft step must write the cell the absorb path writes for the same pair");

        // sensitivity control: a different captured row as the seed must change the draft (else the pairing check above is blind)
        rig.Vk.ResetSequenceState();
        using var kvC = rig.Vk.CreateKvCache(64);
        using var st = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
        rig.Vk.Forward(ids, Pos(9), -1, kvC, null, st).Dispose();
        st.SeedFromCapturedRow(8);
        float[] atLast = Rows(rig.Vk.ForwardMtp(st, 5, 9));
        st.Rollback(9);
        st.SeedFromCapturedRow(7);
        float[] atPrev = Rows(rig.Vk.ForwardMtp(st, 5, 9));
        Assert.True(RelL2(atLast, atPrev) > 1e-3f, "the draft must depend on which trunk residual it is seeded with");
    }

    [SkippableFact]
    public void Rollback_DiscardsSpeculativeCells_AndRedoingThemIsDeterministic()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Q4eQuant.F32);
        int V = rig.Config.VocabSize;
        int[] ids = Prompt(8, 5, V);
        rig.Vk.ResetSequenceState();
        using var kv = rig.Vk.CreateKvCache(64);
        using var st = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
        rig.Vk.Forward(ids, Pos(8), -1, kv, null, st).Dispose();
        st.SeedFromCapturedRow(7);
        float[] first = Rows(rig.Vk.ForwardMtp(st, 6, 8));
        rig.Vk.ForwardMtp(st, 7, 9).Dispose();
        rig.Vk.ForwardMtp(st, 8, 10).Dispose();
        Assert.Equal(11, st.CurrentLength);
        st.Rollback(8);
        Assert.Equal(8, st.CurrentLength);
        Assert.Equal(7, st.CellCount);
        st.SeedFromCapturedRow(7);
        float[] again = Rows(rig.Vk.ForwardMtp(st, 6, 8));
        Assert.Equal(first, again);
        Assert.Throws<InvalidOperationException>(() => rig.Vk.ForwardMtp(st, 6, 12));   // a gap: the head holds no cell 11
    }

    [SkippableFact]
    public void RowSnapshots_RestoreToRow_EqualsAShorterForward()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Q4eQuant.Q8Q51);
        Skip.IfNot(rig.Vk.SupportsRecurrentRowSnapshots, "snapshot scan kernel not built");
        int V = rig.Config.VocabSize;
        int[] ids = Prompt(24, 11, V);
        const int Pre = 12;

        // arm A: prefill 12, a 5-row snapshot forward, restore to row r, then the next token
        // arm B: prefill 12, a plain forward of r + 1 rows, then the same next token
        foreach (int r in new[] { 0, 1, 2, 3 })
        {
            rig.Vk.ResetSequenceState();
            using var kvA = rig.Vk.CreateKvCache(64);
            using var stA = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
            rig.Vk.Forward(ids.AsSpan(0, Pre), Pos(Pre), -1, kvA, null, stA).Dispose();
            long restoresBefore = rig.Vk.SnapshotRestores, snapsBefore = rig.Vk.SnapshotForwards;
            rig.Vk.ForwardWithRecurrentSnapshots(ids.AsSpan(Pre, 5), Pos(5, Pre), -1, kvA, stA).Dispose();
            Assert.Equal(1, rig.Vk.SnapshotForwards - snapsBefore);
            rig.Vk.RestoreRecurrentStateToRow(r);
            Assert.Equal(1, rig.Vk.SnapshotRestores - restoresBefore);
            kvA.Rollback(Pre + r + 1);
            stA.Rollback(Pre + r + 1);
            float[] a = Rows(rig.Vk.Forward(ids.AsSpan(Pre + 5, 1), Pos(1, Pre + r + 1), -1, kvA, null, stA, false));

            rig.Vk.ResetSequenceState();
            using var kvB = rig.Vk.CreateKvCache(64);
            using var stB = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
            rig.Vk.Forward(ids.AsSpan(0, Pre), Pos(Pre), -1, kvB, null, stB).Dispose();
            rig.Vk.Forward(ids.AsSpan(Pre, r + 1), Pos(r + 1, Pre), -1, kvB, null, stB).Dispose();
            float[] b = Rows(rig.Vk.Forward(ids.AsSpan(Pre + 5, 1), Pos(1, Pre + r + 1), -1, kvB, null, stB, false));

            float rel = RelL2(b, a);
            _out.WriteLine($"restore to row {r}: next-token logits relL2 vs the shorter forward {rel:E3}, top-1 {(Argmax(a) == Argmax(b) ? "same" : "DIFF")}");
            Assert.True(rel < 0.05f, $"restoring to row {r} left a state that is not the shorter forward's: relL2 {rel:E3}");
        }

        // control: skipping the restore leaves the state after ALL rows, which must be visibly different for r < 4
        rig.Vk.ResetSequenceState();
        using var kvC = rig.Vk.CreateKvCache(64);
        using var stC = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
        rig.Vk.Forward(ids.AsSpan(0, Pre), Pos(Pre), -1, kvC, null, stC).Dispose();
        rig.Vk.ForwardWithRecurrentSnapshots(ids.AsSpan(Pre, 5), Pos(5, Pre), -1, kvC, stC).Dispose();
        Assert.Throws<InvalidOperationException>(() => rig.Vk.RestoreRecurrentStateToRow(9));
        rig.Vk.Forward(ids.AsSpan(Pre + 5, 1), Pos(1, Pre + 5), -1, kvC, null, stC, false).Dispose();
        Assert.Throws<InvalidOperationException>(() => rig.Vk.RestoreRecurrentStateToRow(1));   // a later forward invalidated the snapshots
    }

    private (List<int> Tokens, List<int> PerRound) Speculate(IModel model, IKvCache kv, int[] prompt, int n, int k, int vocab)
    {
        model.ResetSequenceState();
        using var st = model.CreateMtpState(kv.MaxLength)!;
        var dec = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var gen = new List<int>();
        float[] first = Rows(model.Forward(prompt, Pos(prompt.Length), -1, kv, null, st, true));
        gen.Add(Argmax(first.AsSpan(first.Length - vocab, vocab)));
        var buf = new int[k + 1];
        var perRound = new List<int>();
        while (gen.Count < n)
        {
            int pos = prompt.Length + gen.Count - 1;
            var r = dec.DraftAndVerify(model, kv, st, pipeline, gen, null, pos, vocab, Math.Min(k, n - gen.Count), buf);
            perRound.Add(r.AcceptedCount);
            for (int i = 0; i < r.AcceptedCount && gen.Count < n; i++) gen.Add(buf[i]);
        }
        return (gen, perRound);
    }

    private List<int> PlainGreedy(IModel model, IKvCache kv, int[] prompt, int n, int vocab, List<float[]>? logitsOut = null)
    {
        model.ResetSequenceState();
        var gen = new List<int>();
        var f = Rows(model.Forward(prompt, Pos(prompt.Length), -1, kv, true));
        var row = f.AsSpan(f.Length - vocab, vocab).ToArray();
        logitsOut?.Add(row);
        gen.Add(Argmax(row));
        while (gen.Count < n)
        {
            row = Rows(model.Forward([gen[^1]], [prompt.Length + gen.Count - 1], -1, kv, true));
            logitsOut?.Add(row);
            gen.Add(Argmax(row));
        }
        return gen;
    }

    [SkippableTheory]
    [InlineData(1, 11, 3, 0)]
    [InlineData(2, 9, 4, 1)]
    [InlineData(3, 17, 2, 0)]
    [InlineData(4, 8, 4, 1)]
    public void SpeculativeDecode_EmitsThePlainGreedySequence_ThroughTheSnapshotPath(int seed, int promptLen, int k, int q)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Quants[q]);
        Skip.IfNot(rig.Vk.SupportsRecurrentRowSnapshots, "snapshot scan kernel not built");
        int V = rig.Config.VocabSize;
        int[] prompt = Prompt(promptLen, seed, V);
        const int N = 28;

        using var kvP = rig.Vk.CreateKvCache(96);
        var plainLogits = new List<float[]>();
        var plain = PlainGreedy(rig.Vk, kvP, prompt, N, V, plainLogits);

        long draft0 = rig.Vk.DraftSteps, snap0 = rig.Vk.SnapshotForwards, restore0 = rig.Vk.SnapshotRestores, absorb0 = rig.Vk.AbsorbedCells;
        using var kvS = rig.Vk.CreateKvCache(96);
        var (spec, perRound) = Speculate(rig.Vk, kvS, prompt, N, k, V);
        _out.WriteLine($"seed {seed} k {k} quant {q}: rounds={perRound.Count} tokens/round={(double)(N - 1) / perRound.Count:F2} perRound=[{string.Join(",", perRound)}]; " +
                       $"draftSteps={rig.Vk.DraftSteps - draft0} snapshotForwards={rig.Vk.SnapshotForwards - snap0} restores={rig.Vk.SnapshotRestores - restore0} absorbed={rig.Vk.AbsorbedCells - absorb0}");

        // path counters: the draft head, the snapshot verify and (with a random head: many rejections) the restore all ran
        Assert.True(rig.Vk.DraftSteps - draft0 >= perRound.Count, "the draft head never ran");
        Assert.True(rig.Vk.SnapshotForwards - snap0 >= perRound.Count, "the verify forward never recorded row snapshots");
        Assert.True(rig.Vk.AbsorbedCells - absorb0 > 0, "the head never absorbed a trunk batch");
        Assert.True(rig.Vk.SnapshotRestores - restore0 > 0 || perRound.All(x => x == k + 1), "a rejection must roll back through the recurrent snapshots");

        // the output is the verify path's greedy; on the random tiny model it also equals the 1-row greedy (a near-tie would show here and be reported)
        int firstDiff = Enumerable.Range(0, N).FirstOrDefault(i => plain[i] != spec[i], -1);
        if (firstDiff < 0) _out.WriteLine("identical to 1-row greedy");
        else
        {
            // The verify rows run the multi-row kernels (KL 2e-3..1.4e-2 vs the 1-row path, top-1 usually identical): a divergence is only
            // legitimate when it is a NEAR-TIE of the 1-row logits, i.e. the verify path's token is within numeric noise of the 1-row argmax.
            var l = plainLogits[firstDiff];
            float range = l.Max() - l.Min(), gap = l[plain[firstDiff]] - l[spec[firstDiff]];
            _out.WriteLine($"first divergence from 1-row greedy at token {firstDiff}: 1-row logit gap between its pick and the verify path's pick {gap:E3} (logit range {range:F2}, ratio {gap / range:E2})");
            Assert.True(gap >= 0 && gap < 0.02f * range, $"divergence at token {firstDiff} is not a near-tie: gap {gap:E3} of range {range:F2}");
        }
    }

    [SkippableTheory]
    [InlineData(1, 11, 3)]
    [InlineData(2, 9, 4)]
    public void SpeculativeDecode_BeyondTheDenseLimit_EmitsThePlainGreedySequence(int seed, int promptLen, int k)
    {
        // #819: a 16-token indexer budget makes everything past 19 tokens sparse QSA, so the verify forwards, the recurrent-snapshot
        // restores and the position rewrites after a rejection all run against the sparse indexer cache.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Q4eQuant.F32, geo: Geo with { Budget = 16, Context = 512 });
        Skip.IfNot(rig.Vk.SupportsRecurrentRowSnapshots, "snapshot scan kernel not built");
        int V = rig.Config.VocabSize;
        int[] prompt = Prompt(promptLen, seed, V);
        const int N = 36;
        using var kvP = rig.Vk.CreateKvCache(128);
        var plainLogits = new List<float[]>();
        var plain = PlainGreedy(rig.Vk, kvP, prompt, N, V, plainLogits);
        long restore0 = rig.Vk.SnapshotRestores, sparse0 = rig.Vk.Qsa.SparseLayerRecordings;
        using var kvS = rig.Vk.CreateKvCache(128);
        var (spec, perRound) = Speculate(rig.Vk, kvS, prompt, N, k, V);
        _out.WriteLine($"seed {seed} k {k}: rounds={perRound.Count} perRound=[{string.Join(",", perRound)}] restores={rig.Vk.SnapshotRestores - restore0} sparseLayers={rig.Vk.Qsa.SparseLayerRecordings - sparse0}");
        Assert.True(rig.Vk.Qsa.SparseLayerRecordings > sparse0, "the sparse path never ran");
        Assert.True(rig.Vk.SnapshotRestores - restore0 > 0 || perRound.All(x => x == k + 1), "a rejection must roll back");
        int firstDiff = Enumerable.Range(0, N).FirstOrDefault(i => plain[i] != spec[i], -1);
        if (firstDiff >= 0)
        {
            var l = plainLogits[firstDiff];
            float range = l.Max() - l.Min(), gap = l[plain[firstDiff]] - l[spec[firstDiff]];
            _out.WriteLine($"first divergence at token {firstDiff}: gap {gap:E3} of range {range:F2}");
            Assert.True(gap >= 0 && gap < 0.02f * range, $"divergence at token {firstDiff} is not a near-tie: gap {gap:E3} of range {range:F2}");
        }
    }

    [SkippableFact]
    public void SpeculativeDecode_AcceptanceMatchesTheCpuOracle_WhenTheHeadIsSharp()
    {
        // The oracle's draft chain and the Vulkan one drive the same decoder on the same prompt; per-round accepted counts agree.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Q4eQuant.F32);
        int V = rig.Config.VocabSize;
        int[] prompt = Prompt(13, 21, V);
        const int N = 24, K = 3;
        using var kvC = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(rig.Config), 96);
        var (cpuGen, cpuRounds) = Speculate(rig.Cpu, kvC, prompt, N, K, V);
        using var kvV = rig.Vk.CreateKvCache(96);
        var (vkGen, vkRounds) = Speculate(rig.Vk, kvV, prompt, N, K, V);
        _out.WriteLine($"cpu rounds [{string.Join(",", cpuRounds)}]; vulkan rounds [{string.Join(",", vkRounds)}]");
        Assert.Equal(cpuGen, vkGen);
        Assert.Equal(cpuRounds, vkRounds);
    }

    [SkippableFact]
    public void NgramPrefetch_IsIssuedForForwardsAndForDraftPrefixes()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = Rig(spvDir, Q4eQuant.F32);
        if (rig.Vk.PleTableStats is null) { _out.WriteLine("prefetch service is off in this environment"); return; }
        int V = rig.Config.VocabSize;
        int[] ids = Prompt(10, 31, V);
        rig.Vk.ResetSequenceState();
        using var kv = rig.Vk.CreateKvCache(64);
        using var st = (VulkanQwen4ExpMtpState)rig.Vk.CreateMtpState(64)!;
        long r0 = rig.Vk.PleTableStats!.Value.Requests;
        rig.Vk.Forward(ids, Pos(10), -1, kv, null, st).Dispose();
        long r1 = rig.Vk.PleTableStats!.Value.Requests;
        Assert.True(r1 > r0, "the forward never asked the table service to prefetch");
        st.SeedFromCapturedRow(9);
        rig.Vk.ForwardMtp(st, 5, 10).Dispose();
        rig.Vk.ForwardMtp(st, 6, 11).Dispose();
        long r2 = rig.Vk.PleTableStats!.Value.Requests;
        Assert.True(r2 > r1, "draft steps never pre-requested the coming verify rows");
    }

    [SkippableFact]
    public void TextGenerator_UsesTheVulkanHead_WhenAttachedThroughTheResolver_AndMatchesNoMtp()
    {
        // The wiring `dotllm run` uses: the resolver attaches the sibling mtp-*.gguf to the loaded Vulkan model, the generator then speculates.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rig = new Q4eRig(Qwen4ExpRandomGguf.Build(Geo, Q4eQuant.Q8Q51), spvDir);
        _disposables.Add(rig);
        File.WriteAllBytes(Path.Combine(Path.GetDirectoryName(rig.FilePath)!, "mtp-syn-Q8_0.gguf"), Qwen4ExpRandomGguf.BuildMtpOnly(Geo, Q4eQuant.Q8Q51));
        Assert.False(rig.Vk.SupportsMtp);
        string? attached = Qwen4ExpMtpHeadResolver.TryAttach(rig.Vk, rig.FilePath);
        Assert.NotNull(attached);
        Assert.True(rig.Vk.SupportsMtp);
        Assert.Null(Qwen4ExpMtpHeadResolver.TryAttach(rig.Vk, rig.FilePath));   // already attached

        using var gguf = GgufFile.Open(rig.FilePath);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Func<ModelConfig, int, IKvCache> kv = (_, len) => rig.Vk.CreateKvCache(len);
        var opts = new InferenceOptions { Temperature = 0f, MaxTokens = 26 };
        const string prompt = "tok12 tok13 tok14 tok15 tok16 tok17 tok18";

        var plain = new TextGenerator(rig.Vk, tokenizer, kv, mtpEnabled: false).Generate(prompt, opts);
        long d0 = rig.Vk.DraftSteps, s0 = rig.Vk.SnapshotForwards;
        var gen = new TextGenerator(rig.Vk, tokenizer, kv, speculativeCandidates: 3, mtpEnabled: true, mtpAdaptive: false);
        var spec = gen.Generate(prompt, opts);
        _out.WriteLine($"plain: [{string.Join(",", plain.GeneratedTokenIds)}]  spec: [{string.Join(",", spec.GeneratedTokenIds)}]  draftSteps {rig.Vk.DraftSteps - d0}, snapshot verifies {rig.Vk.SnapshotForwards - s0}");
        Assert.True(rig.Vk.DraftSteps - d0 > 0 && rig.Vk.SnapshotForwards - s0 > 0, "the generator never took the MTP path");
        Assert.Equal(plain.GeneratedTokenIds.Length, spec.GeneratedTokenIds.Length);
        // multi-row verify numerics may flip a near-tie (see the decode test); require agreement over a long prefix
        int agree = plain.GeneratedTokenIds.Zip(spec.GeneratedTokenIds).TakeWhile(t => t.First == t.Second).Count();
        Assert.True(agree >= Math.Min(plain.GeneratedTokenIds.Length, 8), $"speculative output diverges from plain after {agree} tokens");
    }
}
