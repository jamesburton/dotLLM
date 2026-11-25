using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>Oracle / reference data of the Olala tests (<c>scripts/olala-oracle/olala_oracle.py</c>; <c>~/.dotllm/test-cache/olala-oracle</c>).</summary>
internal static class OlalaOracleData
{
    internal sealed record Case(string Name, int Vocab, int[] Ids, int[] GreedyIds, float[] Full, float[] Greedy)
    {
        public int L => Ids.Length;
        public int N => GreedyIds.Length;
    }

    public static string? Dir()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_OLALA_ORACLE_DIR");
        string dir = !string.IsNullOrEmpty(env) ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "olala-oracle");
        return Directory.Exists(dir) ? dir : null;
    }

    public static Case? Load(string name)
    {
        string? dir = Dir();
        if (dir is null) return null;
        string meta = Path.Combine(dir, name + ".meta.json"), bin = Path.Combine(dir, name + ".logits.f32");
        if (!File.Exists(meta) || !File.Exists(bin)) return null;
        using var doc = JsonDocument.Parse(File.ReadAllText(meta));
        var r = doc.RootElement;
        int vocab = r.GetProperty("vocab").GetInt32();
        int[] ids = r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[] gids = r.GetProperty("greedy_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        byte[] raw = File.ReadAllBytes(bin);
        var all = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, all, 0, raw.Length);
        Assert.Equal((long)(ids.Length + gids.Length) * vocab, all.Length);
        return new Case(name, vocab, ids, gids, all[..(ids.Length * vocab)], all[(ids.Length * vocab)..]);
    }

    public static string? TinyGguf()
    {
        string? dir = Dir();
        string? p = dir is null ? null : Path.Combine(dir, "tiny.gguf");
        return p is not null && File.Exists(p) ? p : null;
    }

    public static string? Gguf(string envVar, string file)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } h ? h : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--OVHaiLLM--Olala-v1.0-mini-7A1B-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var d in Directory.GetDirectories(snaps))
        {
            string p = Path.Combine(d, file);
            if (File.Exists(p)) return p;
        }
        return null;
    }

    public static string? Q8Safe2() => Gguf("DOTLLM_OLALA_Q8_GGUF", "Olala-v1.0-mini-7A1B-q8_0-safe2.gguf");

    /// <summary>KL(reference || test) in nats from two logit rows.</summary>
    public static double Kl(ReadOnlySpan<float> reference, ReadOnlySpan<float> test)
    {
        double mr = double.NegativeInfinity, mt = double.NegativeInfinity;
        for (int i = 0; i < reference.Length; i++) { mr = Math.Max(mr, reference[i]); mt = Math.Max(mt, test[i]); }
        double zr = 0, zt = 0;
        for (int i = 0; i < reference.Length; i++) { zr += Math.Exp(reference[i] - mr); zt += Math.Exp(test[i] - mt); }
        double lzr = Math.Log(zr) + mr, lzt = Math.Log(zt) + mt, kl = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double lp = reference[i] - lzr, lq = test[i] - lzt;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }

    public static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    public static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        var a = new float[rows * vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a);
        return a;
    }

    public static double MaxAbs(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }
}

/// <summary>
/// CUDA Olala (#1984) on the tiny HF-oracle model (hidden 64, layers MVMMV, window 5, soft cap 3, BF16 GGUF): CUDA against the HF oracle and
/// against the in-process CPU reference. CUDA stores activations in BF16 (the CPU keeps F32), so the agreement is bounded by BF16 noise rather
/// than float rounding; the gates below were set from measured values, and the mutation record is in docs/OLALA.md.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CudaOlalaParityTests
{
    private readonly ITestOutputHelper _out;
    public CudaOlalaParityTests(ITestOutputHelper output) => _out = output;

    private static (GgufFile File, ModelConfig Config) OpenTiny()
    {
        var f = GgufFile.Open(OlalaOracleData.TinyGguf()!);
        return (f, GgufModelConfigExtractor.Extract(f.Metadata));
    }

    private static CudaOlalaTransformerModel LoadCuda(GgufFile f, ModelConfig c)
        => Assert.IsType<CudaOlalaTransformerModel>(CudaModelLoader.CreateFromGguf(f, c).Model);

    private static void RequireTiny(out OlalaOracleData.Case c, string name = "tiny")
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        Skip.If(OlalaOracleData.TinyGguf() is null, "tiny Olala oracle not generated (scripts/olala-oracle/olala_oracle.py tiny)");
        var loaded = OlalaOracleData.Load(name);
        Skip.If(loaded is null, $"oracle case {name} missing");
        c = loaded!;
    }

    // ---- gates (measured; see the PR) ----
    // Measured clean (RTX 3060, 2026-11-25): worst KL 1.3e-5 vs HF, 1.5e-5 vs CPU, worst |dlogit| 1.7e-2. Gates sit just above those: the soft-cap
    // mutant (KL 4.7e-3), the V-side token-shift mask mutant (2.6e-3) and the other logic mutants die; effects below the BF16 noise (K-side token shift,
    // 1e-8) are gated by the F32-activation control arm below instead.
    private const double HfKlGate = 5e-5;      // CUDA (BF16 activations) vs HF F32-query oracle, per row
    private const double CpuKlGate = 5e-5;     // CUDA vs the in-process CPU reference, per row
    private const double HfMaxAbsGate = 4e-2;  // |dlogit|

    [SkippableTheory]
    [InlineData("tiny")]
    [InlineData("tiny-hfbf16")]
    public void Tiny_Prefill_AllRows_VsHf(string name)
    {
        RequireTiny(out var c, name);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var model = LoadCuda(f, cfg);
        using var logits = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1);
        float[] got = OlalaOracleData.Rows(logits, c.L, c.Vocab);
        double worstKl = 0, worstAbs = 0; int same = 0;
        for (int t = 0; t < c.L; t++)
        {
            var r = c.Full.AsSpan(t * c.Vocab, c.Vocab); var g = got.AsSpan(t * c.Vocab, c.Vocab);
            double kl = OlalaOracleData.Kl(r, g), ma = OlalaOracleData.MaxAbs(r, g);
            worstKl = Math.Max(worstKl, kl); worstAbs = Math.Max(worstAbs, ma);
            if (OlalaOracleData.Argmax(r) == OlalaOracleData.Argmax(g)) same++;
            _out.WriteLine($"{name} row {t}: KL {kl:E2} max|dlogit| {ma:E2} argmax {OlalaOracleData.Argmax(r)}/{OlalaOracleData.Argmax(g)}");
        }
        _out.WriteLine($"{name}: prefill {c.L} rows worst KL {worstKl:E2}, worst |dlogit| {worstAbs:E2}, top-1 {same}/{c.L}");
        Assert.True(worstKl < HfKlGate, $"{name}: worst KL {worstKl:E3}");
        Assert.True(worstAbs < HfMaxAbsGate, $"{name}: worst |dlogit| {worstAbs:E3}");
        Assert.True(same >= c.L - 1, $"{name}: top-1 {same}/{c.L}");
    }

    [SkippableFact]
    public void Tiny_Prefill_VsCpuReference_AndDecodeTeacherForced()
    {
        RequireTiny(out var c);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var cpu = Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(f, cfg));
        using var cuda = LoadCuda(f, cfg);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        using var a = cpu.Forward(c.Ids, pos, -1); using var b = cuda.Forward(c.Ids, pos, -1);
        float[] ra = OlalaOracleData.Rows(a, c.L, c.Vocab), rb = OlalaOracleData.Rows(b, c.L, c.Vocab);
        double worst = 0; int same = 0;
        for (int t = 0; t < c.L; t++)
        {
            worst = Math.Max(worst, OlalaOracleData.Kl(ra.AsSpan(t * c.Vocab, c.Vocab), rb.AsSpan(t * c.Vocab, c.Vocab)));
            if (OlalaOracleData.Argmax(ra.AsSpan(t * c.Vocab, c.Vocab)) == OlalaOracleData.Argmax(rb.AsSpan(t * c.Vocab, c.Vocab))) same++;
        }
        _out.WriteLine($"prefill vs CPU: worst KL {worst:E2}, top-1 {same}/{c.L}");
        Assert.True(worst < CpuKlGate, $"prefill KL {worst:E3}");
        Assert.True(same >= c.L - 1);

        // teacher-forced decode: the trapezoid / token-shift / angle carries and the scalable-softmax position
        cpu.ResetSequenceState(); cuda.ResetSequenceState();
        using (cpu.Forward(c.Ids, pos, -1, null, true)) { }
        using (cuda.Forward(c.Ids, pos, -1, null, true)) { }
        double worstDec = 0; int sameDec = 0;
        for (int j = 1; j < c.N; j++)
        {
            using var sa = cpu.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
            using var sb = cuda.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
            var xa = OlalaOracleData.Rows(sa, 1, c.Vocab); var xb = OlalaOracleData.Rows(sb, 1, c.Vocab);
            double kl = OlalaOracleData.Kl(xa, xb); worstDec = Math.Max(worstDec, kl);
            if (OlalaOracleData.Argmax(xa) == OlalaOracleData.Argmax(xb)) sameDec++;
        }
        _out.WriteLine($"decode vs CPU: worst KL {worstDec:E2}, top-1 {sameDec}/{c.N - 1}");
        Assert.True(worstDec < CpuKlGate, $"decode KL {worstDec:E3}");
        Assert.True(sameDec >= c.N - 2);
    }

    /// <summary>
    /// F32-activation CONTROL arm (<c>olala_f32act.ptx</c>, internal, measurement only; never the shipped storage): the same kernels with F32 stored
    /// activations agree with the CPU reference at float-rounding level. This is what lets the suite see effects far below BF16 noise (a K-side token-shift
    /// mask error is ~1e-8 in KL; the K RMSNorm cancels the uniform scale up to its eps) and what bounds the cost of BF16 storage.
    /// </summary>
    [SkippableFact]
    public void Tiny_F32ActivationControl_MatchesCpuAtFloatLevel()
    {
        RequireTiny(out var c);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var cpu = Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(f, cfg));
        using var ctl = CudaOlalaTransformerModel.LoadF32ActivationControl(f, cfg);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        using var a = cpu.Forward(c.Ids, pos, -1); using var b = ctl.Forward(c.Ids, pos, -1);
        float[] ra = OlalaOracleData.Rows(a, c.L, c.Vocab), rb = OlalaOracleData.Rows(b, c.L, c.Vocab);
        double worstKl = 0, worstAbs = 0;
        for (int t = 0; t < c.L; t++)
        {
            worstKl = Math.Max(worstKl, OlalaOracleData.Kl(ra.AsSpan(t * c.Vocab, c.Vocab), rb.AsSpan(t * c.Vocab, c.Vocab)));
            worstAbs = Math.Max(worstAbs, OlalaOracleData.MaxAbs(ra.AsSpan(t * c.Vocab, c.Vocab), rb.AsSpan(t * c.Vocab, c.Vocab)));
        }
        cpu.ResetSequenceState(); ctl.ResetSequenceState();
        using (cpu.Forward(c.Ids, pos, -1, null, true)) { }
        using (ctl.Forward(c.Ids, pos, -1, null, true)) { }
        for (int j = 1; j < c.N; j++)
        {
            using var sa = cpu.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
            using var sb = ctl.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
            var xa = OlalaOracleData.Rows(sa, 1, c.Vocab); var xb = OlalaOracleData.Rows(sb, 1, c.Vocab);
            worstKl = Math.Max(worstKl, OlalaOracleData.Kl(xa, xb)); worstAbs = Math.Max(worstAbs, OlalaOracleData.MaxAbs(xa, xb));
        }
        _out.WriteLine($"F32-activation control vs CPU: worst KL {worstKl:E2}, worst |dlogit| {worstAbs:E2}");
        Assert.True(worstKl < ControlKlGate, $"control KL {worstKl:E3}");
        Assert.True(worstAbs < ControlAbsGate, $"control |dlogit| {worstAbs:E3}");
    }

    private const double ControlKlGate = 1e-11, ControlAbsGate = 1e-5;   // measured clean: KL 3.5e-14, |dlogit| 9e-7 (Vulkan control arm: 7e-14)

    [SkippableFact]
    public void Tiny_GreedyContinuation_MatchesHf()
    {
        RequireTiny(out var c);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var model = LoadCuda(f, cfg);
        using var pre = model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, true);
        var generated = new List<int> { OlalaOracleData.Argmax(OlalaOracleData.Rows(pre, 1, c.Vocab)) };
        for (int j = 1; j < c.N; j++)
        {
            using var step = model.Forward([generated[^1]], [c.L + j - 1], -1, null, true);
            generated.Add(OlalaOracleData.Argmax(OlalaOracleData.Rows(step, 1, c.Vocab)));
        }
        int firstDiff = Enumerable.Range(0, c.N).FirstOrDefault(i => generated[i] != c.GreedyIds[i], -1);
        _out.WriteLine($"greedy: {(firstDiff < 0 ? "identical" : $"first divergence at token {firstDiff}")} over {c.N} tokens");
        Assert.Equal(c.GreedyIds, generated);
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void Tiny_ChunkedPrefill_IsBitIdenticalToOneShot(int chunk)
    {
        // Per-token arithmetic does not depend on how a prompt is split: the recurrence is sequential, the K/V history continues and the GEMV
        // accumulation order per row is fixed. A chunk boundary that drops the SSM / angle / trapezoid / token-shift carry changes the bits.
        RequireTiny(out var c);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var model = LoadCuda(f, cfg);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        float[] whole;
        using (var a = model.Forward(c.Ids, pos, -1)) whole = OlalaOracleData.Rows(a, c.L, c.Vocab);
        model.ChunkTokens = chunk;
        float[] pieces;
        using (var b = model.Forward(c.Ids, pos, -1)) pieces = OlalaOracleData.Rows(b, c.L, c.Vocab);
        Assert.Equal(whole, pieces);
    }

    [SkippableFact]
    public void Tiny_PositionZero_RestartsTheSequenceState()
    {
        RequireTiny(out var c);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var model = LoadCuda(f, cfg);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        float[] a, b;
        using (var x = model.Forward(c.Ids, pos, -1)) a = OlalaOracleData.Rows(x, c.L, c.Vocab);
        using (var y = model.Forward(c.Ids, pos, -1)) b = OlalaOracleData.Rows(y, c.L, c.Vocab);
        Assert.Equal(a, b);
    }

    [SkippableFact]
    public void Tiny_RejectsNonContinuingPositions()
    {
        RequireTiny(out var c);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var model = LoadCuda(f, cfg);
        using (model.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1)) { }
        Assert.Throws<ArgumentException>(() => model.Forward([c.Ids[0]], [c.L + 3], -1));
    }

    [SkippableFact]
    public void Tiny_RecurrentStateAllocation_IsF32()
    {
        // The allocator sizes the on-device state from CudaOlalaStateLayout (policy-driven, F32). Narrowing any buffer shrinks this total.
        RequireTiny(out var unusedCase);
        var (f, cfg) = OpenTiny();
        using var _ = f;
        using var model = LoadCuda(f, cfg);
        var o = cfg.Olala!;
        long expected = CudaOlalaStateLayout.Ssm(o).Total * o.SsmLayerCount
                      + (CudaOlalaStateLayout.Attn(o, cfg).LastKBytes + CudaOlalaStateLayout.Attn(o, cfg).LastVBytes) * o.AttentionLayerCount;
        Assert.Equal(expected, model.RecurrentStateBytes);
        // F32 per element: the SSM matrices alone must be H*P*N*4 bytes per M layer.
        Assert.True(model.RecurrentStateBytes >= (long)o.NumSsmHeads * o.MambaHeadDim * o.StateSize * sizeof(float) * o.SsmLayerCount);
    }
}
