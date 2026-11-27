using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Architectures;
using DotLLM.Models.Architectures.DSpark;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The Inkling 2 DSpark drafter on Vulkan (issue #2046): greedy identity with the unmodified MTP decoder, draft parity against the cached llama.cpp oracle, row-snapshot restore,
/// windowed-context equivalence (#2060), and the target/drafter mutants. Heavy (17.6 GB target + 0.5 GB drafter on the iGPU); the caller holds the Strix GPU lock.
/// </summary>
public sealed class Inkling2VulkanDSparkTests : IClassFixture<VkInklingFixture>
{
    private readonly ITestOutputHelper _out;
    private readonly VkInklingFixture _fx;
    public Inkling2VulkanDSparkTests(ITestOutputHelper o, VkInklingFixture fx) { _out = o; _fx = fx; }

    private VulkanInkling2TransformerModel Model()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        Skip.If(_fx.TargetPath is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        Skip.If(_fx.DraftPath is null, "dspark-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        var m = _fx.Get();
        if (!m.HasDFlashDrafter) m.AttachDFlashDrafter(_fx.DraftPath!);
        return m;
    }

    private static int ArgMax(ReadOnlySpan<float> row) => System.Numerics.Tensors.TensorPrimitives.IndexOfMax(row);
    private static unsafe ReadOnlySpan<float> LastRow(ITensor t, int vocab) => new((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab);

    private int[] Plain(VulkanInkling2TransformerModel m, int[] prompt, int n)
    {
        int V = m.Config.VocabSize;
        m.ResetSequenceState();
        var got = new List<int>();
        using var pre = m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true);
        int next = ArgMax(LastRow(pre, V));
        for (int j = 0; j < n; j++)
        {
            got.Add(next);
            if (j + 1 == n) break;
            using var step = m.Forward([next], [prompt.Length + j], -1, null, lastTokenLogitsOnly: true);
            next = ArgMax(LastRow(step, V));
        }
        return got.ToArray();
    }

    internal sealed record SpecStats(int Rounds, int Emitted, int Drafted)
    {
        public double TokensPerRound => Rounds == 0 ? 0 : (double)Emitted / Rounds;
        public double AcceptRate => Drafted == 0 ? 0 : (double)(Emitted - Rounds) / Drafted;
    }

    private (int[] Tokens, SpecStats Stats) Spec(VulkanInkling2TransformerModel m, int[] prompt, int n, int k)
    {
        int V = m.Config.VocabSize;
        m.ResetSequenceState();
        using var mtp = m.CreateMtpState()!;
        var kv = m.CreateKvCacheHandle(prompt.Length + n + 32);
        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = true };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var gen = new List<int>();
        using (var pre = m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kv, null, mtp, true))
            gen.Add(ArgMax(LastRow(pre, V)));
        int rounds = 0, emitted = 1, drafted = 0;
        var buf = new int[k + 1];
        while (gen.Count < n)
        {
            int position = prompt.Length + gen.Count - 1;
            var r = decoder.DraftAndVerify(m, kv, mtp, pipeline, gen, null, position, V, Math.Min(k, n - gen.Count), buf);
            if (r.AcceptedCount == 0) throw new InvalidOperationException("round produced nothing");
            for (int i = 0; i < r.AcceptedCount; i++) gen.Add(buf[i]);
            rounds++; emitted += r.AcceptedCount; drafted += r.DraftedCount;
        }
        return (gen.Take(n).ToArray(), new SpecStats(rounds, emitted - 1, drafted));
    }

    [SkippableFact]
    public void GreedySpeculative_EqualsPlainGreedy_OnEightPrompts()
    {
        var m = Model();
        var prompts = Inkling2DSparkTests.GreedyPrompts();
        Skip.If(prompts.Count == 0, "greedy oracle prompts missing");
        int rounds = 0, emitted = 0, drafted = 0;
        var bad = new List<string>();
        foreach (var (name, prompt, _) in prompts)
        {
            const int n = 40;
            var plain = Plain(m, prompt, n);
            var (spec, st) = Spec(m, prompt, n, k: 3);
            rounds += st.Rounds; emitted += st.Emitted; drafted += st.Drafted;
            int same = 0; while (same < n && plain[same] == spec[same]) same++;
            _out.WriteLine($"{name,-8} identical prefix {same}/{n}; rounds {st.Rounds}, tokens/round {st.TokensPerRound:F2}, draft acceptance {st.AcceptRate:P0}");
            if (same != n) bad.Add($"{name}: first difference at {same}");
        }
        _out.WriteLine($"TOTAL K=3: rounds {rounds}, tokens/round {(double)emitted / rounds:F2}, acceptance {(double)(emitted - rounds) / Math.Max(1, drafted):P1}");
        Assert.True(bad.Count == 0, string.Join("; ", bad));
    }

    // ───────────── row snapshots ─────────────

    private float[] AfterVerifyRestore(VulkanInkling2TransformerModel m, int[] prompt, int[] verify, int keep, DSparkTargetMutant mutant)
    {
        int V = m.Config.VocabSize;
        m.ResetSequenceState();
        using (m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true)) { }
        using (m.ForwardWithRecurrentSnapshots(verify, Enumerable.Range(prompt.Length, verify.Length).ToArray(), -1, null, null)) { }
        m.TargetMutant = mutant;
        try
        {
            m.RestoreRecurrentStateToRow(keep);
            if (mutant != DSparkTargetMutant.None) m.ForceLengthForTests(prompt.Length + keep + 1);   // the KV length rewinds either way; only the conv state differs
        }
        finally { m.TargetMutant = DSparkTargetMutant.None; }
        using var t = m.Forward([777], [prompt.Length + keep + 1], -1, null, lastTokenLogitsOnly: true);
        return LastRow(t, V).ToArray();
    }

    /// <summary>The conv windows restored from a verify row equal sequential decoding bit for bit, across ALL 48 slots; each target mutant must change the next logits.</summary>
    [SkippableFact]
    public void RowSnapshotRestore_EqualsSingleRowDecode_BitForBit_AndMutantsDie()
    {
        var m = Model();
        int V = m.Config.VocabSize;
        int[] prompt = [371, 151451, 149879, 151194, 149884, 152717, 13, 68014];
        int[] verify = [32, 13, 220, 44781, 68014, 33];
        const int keep = 2;
        var viaRestore = AfterVerifyRestore(m, prompt, verify, keep, DSparkTargetMutant.None);

        m.ResetSequenceState();
        using (m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true)) { }
        for (int i = 0; i <= keep; i++) using (m.Forward([verify[i]], [prompt.Length + i], -1, null, lastTokenLogitsOnly: true)) { }
        float[] viaSingles;
        using (var t = m.Forward([777], [prompt.Length + keep + 1], -1, null, lastTokenLogitsOnly: true)) viaSingles = LastRow(t, V).ToArray();
        Assert.True(viaRestore.AsSpan().SequenceEqual(viaSingles), "state restored from a verify row differs from sequential decoding");

        foreach (var mutant in new[] { DSparkTargetMutant.SkipRestore, DSparkTargetMutant.RestoreRowPlusOne, DSparkTargetMutant.RestorePass0Only })
        {
            var got = AfterVerifyRestore(m, prompt, verify, keep, mutant);
            bool survived = got.AsSpan().SequenceEqual(viaSingles);
            _out.WriteLine($"{mutant}: {(survived ? "SURVIVED" : "killed")}");
            Assert.False(survived, $"{mutant} survived");
        }
        _out.WriteLine($"snapshot bytes {m.SnapshotBytes} (6 verify rows, 48 slots)");
    }

    // ───────────── probes (teacher-forced drafts) ─────────────

    private (int[][] Drafts, float[][] Sig) Probe(VulkanInkling2TransformerModel m, int[] tokens, int firstProbe, int stride, int k, int chunk, int ring = 0, DSparkMutant[]? draftMutants = null)
    {
        m.ResetSequenceState();
        using var st = ring > 0 ? m.CreateDSparkState(ring) : (VulkanInkling2DSparkState)m.CreateMtpState()!;
        var all = new[] { DSparkMutant.None }.Concat(draftMutants ?? []).ToArray();
        var drafts = new List<int[]>(); var sigs = new List<float[]>();
        int p = 0;
        while (p < tokens.Length)
        {
            int next = p < firstProbe ? Math.Min(chunk, firstProbe - p) : 1;
            using (m.Forward(tokens[p..(p + next)], Enumerable.Range(p, next).ToArray(), -1, null, null, st, true)) { }
            p += next;
            if (p >= firstProbe && (p - firstProbe) % stride == 0 && p < tokens.Length)
                foreach (var mut in all)
                {
                    m.DrafterMutant = mut;
                    try
                    {
                        var d = new int[k];
                        st.Rollback(p);
                        for (int i = 0; i < k; i++) d[i] = st.NextDraft(tokens[p], p + i);
                        drafts.Add(d); sigs.Add(Signature(st, p, k));
                    }
                    finally { m.DrafterMutant = DSparkMutant.None; }
                }
        }
        return (drafts.ToArray(), sigs.ToArray());
    }

    private static float[] Signature(VulkanInkling2DSparkState st, int p, int k)
    {
        var sig = new List<float>();
        for (int i = 0; i < k; i++)
        {
            var row = st.DraftLogits(p + i).ToArray();
            // top 16 by (logit desc, id asc): a stable order, so exact ties (a softcap-saturated 30.0) cannot reorder the fingerprint between two runs
            var top = Enumerable.Range(0, row.Length).OrderByDescending(t => row[t]).ThenBy(t => t).Take(16);
            foreach (int t in top) { sig.Add(row[t]); sig.Add(t); }
        }
        return sig.ToArray();
    }

    private static double MaxDiff(float[][] a, float[][] b)
    {
        double d = 0;
        for (int r = 0; r < a.Length; r++) for (int i = 0; i < a[r].Length; i++) d = Math.Max(d, Math.Abs(a[r][i] - b[r][i]));
        return d;
    }

    private static float[][] Pick(float[][] sigs, int slot, int stride) => sigs.Where((_, i) => i % stride == slot).ToArray();

    private int[] ShortSequence()
    {
        var (_, prompt, gen) = Inkling2DSparkTests.GreedyPrompts().First(x => x.Name == "history");
        return prompt.Concat(gen).ToArray();
    }

    [SkippableFact]
    public void DraftProbe_ShortContext_TapAndMarkovMutantsAreObservable()
    {
        var m = Model();
        var seq = ShortSequence();
        const int k = 3, first = 5;
        var draftMut = new[] { DSparkMutant.NoMarkovState, DSparkMutant.SoftcapBeforeBias, DSparkMutant.FrnnWrongPos };
        var (d0, s0) = Probe(m, seq, first, 1, k, 8, 0, draftMut);
        int per = 1 + draftMut.Length;
        var baseSig = Pick(s0, 0, per);
        // agreement with the greedy continuation (the baseline must be a working drafter)
        int hit = 0, total = 0;
        var baseDrafts = d0.Where((_, i) => i % per == 0).ToArray();
        for (int r = 0; r < baseDrafts.Length; r++) for (int i = 0; i < k && first + r + 1 + i < seq.Length; i++) { total++; if (baseDrafts[r][i] == seq[first + r + 1 + i]) hit++; }
        _out.WriteLine($"baseline: {baseDrafts.Length} probes, agreement with the greedy continuation {(double)hit / total:P1}");
        Assert.True((double)hit / total > 0.10);
        for (int j = 0; j < draftMut.Length; j++)
        {
            double diff = MaxDiff(baseSig, Pick(s0, j + 1, per));
            _out.WriteLine($"{draftMut[j]}: max |draft logit diff| {diff:E2}");
            Assert.True(diff > 1e-3, $"{draftMut[j]} is not observable (diff {diff})");
        }
        foreach (var mutant in new[] { DSparkTargetMutant.TapBySlot, DSparkTargetMutant.TapEarlyPrelude, DSparkTargetMutant.TapLate, DSparkTargetMutant.TapPreNorm })
        {
            m.TargetMutant = mutant;
            try
            {
                var (_, s) = Probe(m, seq, first, 1, k, 8);
                double diff = MaxDiff(baseSig, s);
                _out.WriteLine($"{mutant}: max |draft logit diff| {diff:E2}");
                Assert.True(diff > 1e-3, $"{mutant} is not observable (diff {diff})");
            }
            finally { m.TargetMutant = DSparkTargetMutant.None; }
        }
    }

    /// <summary>
    /// #2060 on Vulkan: with the windowed ring (window + chunk slack) the drafts equal those of an effectively unbounded ring (no wrap) at 1300 and 4000 tokens of context,
    /// and a ring one row below the minimum (1024 + the rows a rejected verify may write ahead) changes them. Window mutants are observable on non-repetitive data.
    /// </summary>
    [SkippableTheory]
    [InlineData(1330, 1300)]
    [InlineData(4030, 4000)]
    public void WindowedRing_DraftsEqualUnboundedRing_AndAShortRingChangesThem(int length, int first)
    {
        var m = Model();
        var rng = new Random(2060);
        int[] tokens = Enumerable.Range(0, length).Select(_ => rng.Next(1000, 120000)).ToArray();
        const int k = 3;
        int shipped = m.RingPositionsFor();
        var (_, sShip) = Probe(m, tokens, first, 10, k, 64, shipped);
        var (_, sBig) = Probe(m, tokens, first, 10, k, 64, 8192);
        double eq = MaxDiff(sShip, sBig);
        _out.WriteLine($"ring {shipped} vs unbounded 8192 at ~{first} tokens: max |draft logit diff| {eq:E2}, resident K/V {shipped * 1024L * 2 * 2 * 2 / 1024} KiB for the two windowed layers");
        Assert.True(eq == 0, "a ring of window + chunk slack must reproduce the unbounded drafts exactly");
        // one row below the minimum: 1024 window keys + 1 row the verify wrote ahead = 1025; 1024 loses the oldest visible key once a row is written ahead
        var (_, sShort) = Probe(m, tokens, first, 10, k, 64, 1023);
        double shortDiff = MaxDiff(sBig, sShort);
        _out.WriteLine($"ring 1023 (one row short of the 1024-key window): max |draft logit diff| {shortDiff:E2}");
        Assert.True(shortDiff > 0, "a ring shorter than the window must change the drafts");
        if (length < 2000)
        {
            var mutants = new[] { DSparkMutant.AllLayersLocal, DSparkMutant.PerQueryWindow };
            var (_, sm) = Probe(m, tokens, first, 10, k, 64, 8192, mutants);
            int per = 1 + mutants.Length;
            for (int j = 0; j < mutants.Length; j++)
            {
                double diff = MaxDiff(Pick(sm, 0, per), Pick(sm, j + 1, per));
                _out.WriteLine($"{mutants[j]}: max |draft logit diff| {diff:E2} " + (diff > 1e-5 ? "(killed)" : "(SURVIVED)"));
                if (mutants[j] != DSparkMutant.AllLayersLocal) Assert.True(diff > 1e-5, $"{mutants[j]} not observable");
            }
        }
    }

    /// <summary>Draft parity against llama.cpp (<c>dspark-oracle-k3.json</c>): teacher-forced on its committed tokens, drafts at each round's anchor. The CPU port reached 96.0%.</summary>
    [SkippableFact]
    public void Drafts_MatchLlamaCppDsparkOracle()
    {
        var m = Model();
        string path = Path.Combine(Inkling2RealModelTests.OracleDir, "dspark-oracle-k3.json");
        Skip.If(!File.Exists(path), "dspark-oracle-k3.json missing (see docs/INKLING2.md section 13)");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        int total = 0, hit = 0, pos1 = 0, pos1Hit = 0, rounds = 0, roundsAllHit = 0;
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            int[] prompt = prop.Value.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] T = prop.Value.GetProperty("tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            m.ResetSequenceState();
            using var st = (VulkanInkling2DSparkState)m.CreateMtpState()!;
            using (m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, null, st, true)) { }
            int fed = 0;
            foreach (var r in prop.Value.GetProperty("rounds").EnumerateArray())
            {
                int a = r.GetProperty("anchor").GetInt32();
                while (fed < a) { using (m.Forward([T[fed]], [prompt.Length + fed], -1, null, null, st, true)) { } fed++; }
                var cands = r.GetProperty("drafts").EnumerateArray().Select(d => d.EnumerateArray().Select(c => (Id: c[0].GetInt32(), P: c[1].GetDouble())).ToArray()).ToArray();
                int k = cands.Length;
                bool all = true;
                for (int i = 0; i < k; i++)
                {
                    int mine = st.NextDraft(T[a], prompt.Length + a + i);
                    total++; bool ok = mine == cands[i][0].Id; if (ok) hit++; else all = false;
                    if (i == 0) { pos1++; if (ok) pos1Hit++; }
                }
                rounds++; if (all) roundsAllHit++;
            }
        }
        _out.WriteLine($"draft top-1 agreement with llama.cpp: {hit}/{total} = {(double)hit / total:P1} (position 1: {(double)pos1Hit / pos1:P1}); whole rounds identical {roundsAllHit}/{rounds}");
        Assert.True((double)hit / total > 0.90, $"top-1 agreement {(double)hit / total:P1} too low (CPU: 96.0%)");
    }

    /// <summary>Resident bytes: windowed K/V of the drafter against the old unbounded layout (12 KiB per position), the conv snapshots and the drafter's weights + scratch.</summary>
    [SkippableFact]
    public void Report_ResidentBytes()
    {
        var m = Model();
        using var st = (VulkanInkling2DSparkState)m.CreateMtpState()!;
        long perPos = 3L * 2 * 8 * 128 * 2;
        _out.WriteLine($"drafter weights + scratch {m.DSparkBytes / 1048576.0:F1} MiB; fresh state K/V {st.KvBytes / 1024.0:F0} KiB (2 ring layers x {st.RingPositions} positions + the global layer's first 1024), steady-state {(8L * 128 * 2 * 2) / 1024} KiB per position (was {perPos / 1024} KiB)");
        _out.WriteLine($"at 2K context {((long)st.RingPositions * 2 * 2 * 2 * 8 * 128 + 2048L * 8 * 128 * 2 * 2) / 1024} KiB, at 100K {((long)st.RingPositions * 2 * 2 * 2 * 8 * 128 + 100000L * 8 * 128 * 2 * 2) / 1048576} MiB (old layout {100000L * perPos / 1048576} MiB)");
    }

    /// <summary>Two identical prefills must leave bit-identical K/V, f_rnn ring and drafts (guards the race of rows r and r + 16 writing the same f_rnn slot in one dispatch, found while testing #2060).</summary>
    [SkippableFact]
    public void TwoIdenticalPrefills_GiveBitIdenticalContextRingAndDrafts()
    {
        var m = Model();
        var rng = new Random(2060);
        int[] tokens = Enumerable.Range(0, 1330).Select(_ => rng.Next(1000, 120000)).ToArray();
        float[] Down(VulkanInkling2DSparkState st, int layer, bool v)
        {
            var b = v ? st.V[layer] : st.K[layer];
            var raw = new float[b.Size / 4];
            m.DeviceForTests.Download(b, raw);
            return raw;
        }
        var kvs = new List<float[][]>(); var drafts = new List<float[]>(); var frnnSums = new List<double>();
        for (int rep = 0; rep < 3; rep++)
        {
            m.ResetSequenceState();
            using var st = (VulkanInkling2DSparkState)m.CreateMtpState()!;
            int p = 0;
            while (p < 1300) { int n = Math.Min(64, 1300 - p); using (m.Forward(tokens[p..(p + n)], Enumerable.Range(p, n).ToArray(), -1, null, null, st, true)) { } p += n; }
            kvs.Add(Enumerable.Range(0, 3).SelectMany(l => new[] { Down(st, l, false), Down(st, l, true) }).ToArray());
            var d1 = new float[3]; var sigs = new List<float>();
            for (int k = 0; k < 2; k++)
            {
                st.Rollback(1300);
                for (int i = 0; i < 3; i++) st.NextDraft(tokens[1300], 1300 + i);
                sigs.AddRange(Signature(st, 1300, 3));
            }
            drafts.Add(sigs.ToArray());
            var fr = new float[st.Frnn.Size / 4]; m.DeviceForTests.Download(st.Frnn, fr);
            frnnSums.Add(fr.Sum(x => (double)BitConverter.SingleToInt32Bits(x)));
            _out.WriteLine($"rep {rep}: frnn checksum {fr.Sum(x => (double)BitConverter.SingleToInt32Bits(x)):E6}; first draft top: id {sigs[1]} logit {sigs[0]}");
        }
        for (int i = 0; i < 6; i++)
        {
            int diff = 0; for (int j = 0; j < kvs[0][i].Length; j++) if (BitConverter.SingleToInt32Bits(kvs[0][i][j]) != BitConverter.SingleToInt32Bits(kvs[1][i][j]) || BitConverter.SingleToInt32Bits(kvs[0][i][j]) != BitConverter.SingleToInt32Bits(kvs[2][i][j])) diff++;
            _out.WriteLine($"{(i % 2 == 0 ? "K" : "V")} layer {i / 2}: {diff} of {kvs[0][i].Length} words differ between two identical prefills");
            Assert.Equal(0, diff);
        }
        Assert.True(frnnSums.Distinct().Count() == 1, "the f_rnn ring differs between two identical prefills (a race in the ring store)");
        int dd = 0; for (int j = 0; j < drafts[0].Length; j++) if (drafts[0][j] != drafts[1][j] || drafts[0][j] != drafts[2][j]) dd++;
        Assert.Equal(0, dd);
        _out.WriteLine($"draft signatures differing between two prefills: {dd} of {drafts[0].Length}; within one prefill, draft run 1 vs 2: {Enumerable.Range(0, drafts[0].Length / 2).Count(j => drafts[0][j] != drafts[0][j + drafts[0].Length / 2])}");
    }
}
