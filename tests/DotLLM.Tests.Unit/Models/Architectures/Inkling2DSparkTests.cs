using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Architectures.DSpark;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Loads the NVFP4 target and the DSpark drafter once for the whole class (a 17.6 GB CPU load).</summary>
public sealed class DSparkFixture : IDisposable
{
    public string? TargetPath { get; } = Inkling2RealModelTests.FindGguf();
    public string? DraftPath { get; } = Inkling2DSparkTests.FindDraft();
    public GgufFile? File { get; private set; }
    public Inkling2TransformerModel? Model { get; private set; }

    public Inkling2TransformerModel Get()
    {
        if (Model is not null) return Model;
        File = GgufFile.Open(TargetPath!);
        var cfg = GgufModelConfigExtractor.Extract(File.Metadata);
        Model = Inkling2TransformerModel.LoadFromGguf(File, cfg, ThreadingConfig.Auto, Inkling2Options.FromEnvironment());
        Model.AttachDFlashDrafter(DraftPath!);
        return Model;
    }

    public void Dispose() { Model?.Dispose(); File?.Dispose(); }
}

/// <summary>
/// The Inkling 2 DSpark drafter (issue #2039) on the CPU: greedy identity with the unmodified MTP decoder, draft-quality probes, mutants, allocation, integration.
/// Heavy (17.6 GB target); skipped when the model or the drafter is missing from the HF cache.
/// </summary>
public sealed class Inkling2DSparkTests : IClassFixture<DSparkFixture>
{
    private readonly ITestOutputHelper _out;
    private readonly DSparkFixture _fx;
    public Inkling2DSparkTests(ITestOutputHelper output, DSparkFixture fx) { _out = output; _fx = fx; }

    internal static string? FindDraft()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_DSPARK_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } h ? h : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--unsloth--Inkling-2-Mini-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var d in Directory.GetDirectories(snaps))
        {
            string p = Path.Combine(d, "dspark-Inkling-2-Mini-Q8_0.gguf");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private Inkling2TransformerModel Model()
    {
        Skip.If(_fx.TargetPath is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        Skip.If(_fx.DraftPath is null, "dspark-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        return _fx.Get();
    }

    private static int ArgMax(ReadOnlySpan<float> row) => System.Numerics.Tensors.TensorPrimitives.IndexOfMax(row);

    private static unsafe ReadOnlySpan<float> LastRow(ITensor t, int vocab) => new((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab);

    internal static List<(string Name, int[] Prompt, int[] Gen)> GreedyPrompts()
    {
        string path = Path.Combine(Inkling2RealModelTests.OracleDir, "greedy-nvfp4-fa-off.json");
        var list = new List<(string, int[], int[])>();
        if (!File.Exists(path)) return list;
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        foreach (var p in doc.RootElement.EnumerateObject())
            list.Add((p.Name, p.Value.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                      p.Value.GetProperty("gen_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray()));
        return list;
    }

    private int[] Plain(Inkling2TransformerModel m, int[] prompt, int n)
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

    internal sealed record SpecStats(int Rounds, int Emitted, int Drafted, int ZeroRounds)
    {
        public double TokensPerRound => Rounds == 0 ? 0 : (double)Emitted / Rounds;
        public double AcceptRate => Drafted == 0 ? 0 : (double)(Emitted - Rounds) / Drafted;
    }

    private (int[] Tokens, SpecStats Stats) Spec(Inkling2TransformerModel m, int[] prompt, int n, int k, bool rowSnapshots = true)
    {
        int V = m.Config.VocabSize;
        m.ResetSequenceState();
        using var mtp = m.CreateMtpState()!;
        using var kv = new SimpleKvCache(1, 1, 1, prompt.Length + n + 32);
        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = rowSnapshots };
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
        return (gen.Take(n).ToArray(), new SpecStats(rounds, emitted - 1, drafted, 0));
    }

    [SkippableFact]
    public void GreedySpeculative_EqualsPlainGreedy_OnEightPrompts()
    {
        var m = Model();
        var prompts = GreedyPrompts();
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

    [SkippableFact]
    public unsafe void RowSnapshotRestore_EqualsSingleRowDecode_BitForBit()
    {
        var m = Model();
        int V = m.Config.VocabSize;
        int[] prompt = [371, 151451, 149879, 151194, 149884, 152717, 13, 68014];
        int[] verify = [32, 13, 220, 44781, 68014, 33];
        const int keep = 2;   // rows 0..2 accepted
        float[] viaRestore, viaSingles;
        m.ResetSequenceState();
        using (m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true)) { }
        using (m.ForwardWithRecurrentSnapshots(verify, Enumerable.Range(prompt.Length, verify.Length).ToArray(), -1, null, null)) { }
        m.RestoreRecurrentStateToRow(keep);
        using (var t = m.Forward([777], [prompt.Length + keep + 1], -1, null, lastTokenLogitsOnly: true)) viaRestore = LastRow(t, V).ToArray();

        m.ResetSequenceState();
        using (m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true)) { }
        for (int i = 0; i <= keep; i++) using (m.Forward([verify[i]], [prompt.Length + i], -1, null, lastTokenLogitsOnly: true)) { }
        using (var t = m.Forward([777], [prompt.Length + keep + 1], -1, null, lastTokenLogitsOnly: true)) viaSingles = LastRow(t, V).ToArray();
        Assert.True(viaRestore.AsSpan().SequenceEqual(viaSingles), "state restored from a verify row differs from sequential decoding");

        // Mutant: a restore that does nothing leaves the state advanced through the rejected rows.
        m.ResetSequenceState();
        using (m.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true)) { }
        using (m.ForwardWithRecurrentSnapshots(verify, Enumerable.Range(prompt.Length, verify.Length).ToArray(), -1, null, null)) { }
        m.Mutant = Inkling2Mutant.SkipRestore;
        try
        {
            m.RestoreRecurrentStateToRow(keep);
            m.DefaultState.Length = prompt.Length + keep + 1;   // the KV length rewinds either way; only the conv state is skipped
            using var t = m.Forward([777], [prompt.Length + keep + 1], -1, null, lastTokenLogitsOnly: true);
            Assert.False(LastRow(t, V).SequenceEqual(viaSingles), "SkipRestore mutant survived");
        }
        finally { m.Mutant = Inkling2Mutant.None; }
    }

    /// <summary>Teacher-forced drafts at every position in <paramref name="probe"/>: the draft tokens and the first draft's logits.</summary>
    private (int[][] Drafts, float[][] FirstLogits) Probe(Inkling2TransformerModel m, int[] tokens, int firstProbe, int stride, int k, int chunk)
    {
        int V = m.Config.VocabSize;
        m.ResetSequenceState();
        using var st = m.CreateDSparkState();
        var drafts = new List<int[]>(); var logits = new List<float[]>();
        int p = 0;
        while (p < tokens.Length)
        {
            int next = p < firstProbe ? Math.Min(chunk, firstProbe - p) : 1;
            int[] seg = tokens[p..(p + next)];
            using (m.Forward(seg, Enumerable.Range(p, next).ToArray(), -1, null, null, st, true)) { }
            p += next;
            if (p >= firstProbe && (p - firstProbe) % stride == 0 && p < tokens.Length)
            {
                var d = new int[k];
                for (int i = 0; i < k; i++) d[i] = st.NextDraft(m, tokens[p], p + i, k);
                drafts.Add(d);
                logits.Add(st.DraftLogits(p).ToArray());
            }
        }
        return (drafts.ToArray(), logits.ToArray());
    }

    private static double Agreement(int[][] drafts, int[] tokens, int firstProbe, int stride)
    {
        int hit = 0, total = 0;
        for (int r = 0; r < drafts.Length; r++)
        {
            int p = firstProbe + r * stride;
            for (int i = 0; i < drafts[r].Length && p + 1 + i < tokens.Length; i++) { total++; if (drafts[r][i] == tokens[p + 1 + i]) hit++; }
        }
        return total == 0 ? 0 : (double)hit / total;
    }

    private static double MaxDiff(float[][] a, float[][] b)
    {
        double d = 0;
        for (int r = 0; r < a.Length; r++) for (int i = 0; i < a[r].Length; i++) d = Math.Max(d, Math.Abs(a[r][i] - b[r][i]));
        return d;
    }

    private int[] ShortSequence(Inkling2TransformerModel m)
    {
        var (_, prompt, gen) = GreedyPrompts().First(x => x.Name == "history");
        return prompt.Concat(gen).ToArray();
    }

    [SkippableFact]
    public void DraftProbe_ShortContext_Baseline_AndTapAndInjectionMutants()
    {
        var m = Model();
        var seq = ShortSequence(m);
        const int k = 3, first = 5;
        var (d0, l0) = Probe(m, seq, first, 1, k, 8);
        double a0 = Agreement(d0, seq, first, 1);
        _out.WriteLine($"baseline: {d0.Length} probes, draft-token agreement with the greedy continuation {a0:P1}");
        Assert.True(a0 > 0.10, "baseline drafts agree with the target's continuation on almost nothing");

        var results = new List<string>();
        foreach (var mutant in new[] { Inkling2Mutant.TapBySlot, Inkling2Mutant.TapOffByOne, Inkling2Mutant.TapPreNorm })
        {
            m.Mutant = mutant;
            try
            {
                var (_, l) = Probe(m, seq, first, 1, k, 8);
                double diff = MaxDiff(l0, l);
                results.Add($"{mutant}: max |draft logit diff| {diff:E2}");
                Assert.True(diff > 1e-4, $"{mutant} is not observable (diff {diff})");
            }
            finally { m.Mutant = Inkling2Mutant.None; }
        }
        var dr = m.DSparkDrafter!;
        foreach (var mutant in new[] { DSparkMutant.DoubleSoftmaxCtxMix })
        {
            dr.Mutant = mutant;
            try
            {
                var (_, l) = Probe(m, seq, first, 1, k, 8);
                double diff = MaxDiff(l0, l);
                results.Add($"{mutant}: max |draft logit diff| {diff:E2}");
                Assert.True(diff > 1e-4, $"{mutant} is not observable (diff {diff})");
            }
            finally { dr.Mutant = DSparkMutant.None; }
        }
        // Draft-time mutants reuse one injection pass: flip the drafter's mutant between draws of the same state.
        foreach (var mutant in new[] { DSparkMutant.NoAnchorNorm, DSparkMutant.DoubleAnchorNorm, DSparkMutant.NoMarkovState, DSparkMutant.SoftcapBeforeBias, DSparkMutant.FrnnWrongPos })
        {
            dr.Mutant = mutant;
            try
            {
                var (_, l) = Probe(m, seq, first, 1, k, 8);
                double diff = MaxDiff(l0, l);
                results.Add($"{mutant}: max |draft logit diff| {diff:E2}");
                Assert.True(diff > 1e-4, $"{mutant} is not observable (diff {diff})");
            }
            finally { dr.Mutant = DSparkMutant.None; }
        }
        foreach (var r in results) _out.WriteLine(r);
    }

    [SkippableFact]
    public void DraftProbe_LongContext_WindowMutantsAreObservable()
    {
        var m = Model();
        var kl = Inkling2RealModelTests.LoadKlRef();
        Skip.If(kl is null, "oracle passage tokens missing");
        var seq = new List<int>();
        while (seq.Count < 1330) seq.AddRange(kl!.Tokens.Take(256));
        int[] tokens = seq.Take(1330).ToArray();
        const int k = 3, first = 1300;
        var (d0, l0) = Probe(m, tokens, first, 10, k, 32);
        _out.WriteLine($"long baseline: {d0.Length} probes, agreement {Agreement(d0, tokens, first, 10):P1}");
        var dr = m.DSparkDrafter!;
        foreach (var mutant in new[] { DSparkMutant.AllLayersGlobal, DSparkMutant.AllLayersLocal, DSparkMutant.PerQueryWindow })
        {
            dr.Mutant = mutant;
            try
            {
                var (_, l) = Probe(m, tokens, first, 10, k, 32);
                double diff = MaxDiff(l0, l);
                _out.WriteLine($"{mutant}: max |draft logit diff| {diff:E2}");
                Assert.True(diff > 1e-5, $"{mutant} is not observable at 1300 tokens of context (diff {diff})");
            }
            finally { dr.Mutant = DSparkMutant.None; }
        }
    }

    [SkippableFact]
    public void DraftHotPath_AllocatesNoManagedTensorMemory()
    {
        var m = Model();
        var seq = ShortSequence(m);
        m.ResetSequenceState();
        using var st = m.CreateDSparkState();
        using (m.Forward(seq, Enumerable.Range(0, seq.Length).ToArray(), -1, null, null, st, true)) { }
        int p = seq.Length;
        for (int w = 0; w < 3; w++) st.NextDraft(m, 1234, p, 3);   // JIT + scratch
        GC.Collect(); GC.WaitForPendingFinalizers();
        long a0 = GC.GetTotalAllocatedBytes(precise: true);
        const int n = 8;
        for (int i = 0; i < n; i++) st.NextDraft(m, 1234, p, 7);
        long per = (GC.GetTotalAllocatedBytes(precise: true) - a0) / n;
        _out.WriteLine($"managed allocation per draft (K=7): {per} B");
        Assert.True(per < 16 * 1024, $"a draft allocated {per} managed bytes");
    }
}
