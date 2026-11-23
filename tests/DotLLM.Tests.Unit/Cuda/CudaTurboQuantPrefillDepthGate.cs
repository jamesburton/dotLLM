using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1924 gate for the bit-specialised (E1, <c>_tq{bits}</c>) and staged (E3, <c>_tq{bits}_cpa</c>) TurboQuant prefill entries, set by
/// James 2026-11-22 in place of the SmolLM step-1 MinCos bound (which keeps gating the legacy entry). On Qwen3-4B (hd128) and
/// Qwen2.5-0.5B (hd64), at tq4/3/2/8, after a long chunked prefill, each new entry at a width whose default is not legacy must have
/// mean KL(f16 KV || arm) no worse than the legacy entry's plus the drift band, and top-1 agreement with f16 KV no worse than legacy's
/// minus the control arms' top-1 spread (James 2026-11-23; it was a fixed minus 1). The drift band (James 2026-11-22, "measure the noise, then decide"; widened per the final review) is the largest
/// mean-KL movement over the legacy entry run with one deliberate 1-ulp perturbation per arm: softmax scale +-1 and +-2 F32 ulps, and
/// the last bit of one stored K or V norm flipped after the first chunk. Arms inert by construction or that move no logit bitwise are
/// dropped with the reason printed; at least two arms must move. E1 and E3 must be bit-identical.
/// <para>A 4096-token wikitext prompt (<c>DOTLLM_TQ_PREFILL_DEPTH_TOKENS</c> overrides) is prefilled in 1024-row chunks, so every
/// chunk after the first runs the tensor-core TQ prefill at depth; 32 teacher-forced decode rows follow (scalar kernel, unchanged).
/// 33 rows are compared. The scratch oracle is reported for context, not gated. tq-gate's quality and 1-layer parity modes cannot see
/// this change on CUDA (the CUDA forward returns one row, so quality runs one token per forward; one layer cannot move the codes).
/// Models: Qwen3-4B-Instruct-2507 Q4_K_M (<c>DOTLLM_QWEN3_4B_GGUF</c>), Qwen2.5-0.5B-Instruct Q8_0 (<c>DOTLLM_QWEN25_05B_GGUF</c>), or
/// the model store.</para>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantPrefillDepthGate
{
    private const ulong Seed = 0x5DEECE66DUL;   // the CLI's default TurboQuant seed
    private readonly ITestOutputHelper _out;
    public CudaTurboQuantPrefillDepthGate(ITestOutputHelper output) => _out = output;

    /// <summary>Models of the gate: a hd128 model (Qwen3-4B) and a hd64 model (Qwen2.5-0.5B), so every head dim the defaults cover has a
    /// model-level gate (final review I2).</summary>
    public static TheoryData<string> Models => new() { "qwen3-4b", "qwen2.5-0.5b" };

    private static string ModelPath(string model) => model switch
    {
        "qwen3-4b" => Environment.GetEnvironmentVariable("DOTLLM_QWEN3_4B_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
                                               ["Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"], null).Path ?? "",
        "qwen2.5-0.5b" => Environment.GetEnvironmentVariable("DOTLLM_QWEN25_05B_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", ["qwen2.5-0.5b-instruct-q8_0.gguf"], null).Path ?? "",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>A drift-control arm: the LEGACY entry with one deliberate 1-ulp perturbation.</summary>
    private sealed record Ctrl(string Name, int ScaleUlps, int FlipNormSide);   // FlipNormSide: 0 none, 1 K, 2 V

    private static readonly Ctrl[] Controls =
    [
        new("scale+1ulp", +1, 0), new("scale-1ulp", -1, 0), new("scale+2ulp", +2, 0), new("scale-2ulp", -2, 0),
        new("Knorm-lsb", 0, 1), new("Vnorm-lsb", 0, 2),
    ];

    /// <summary>The kernel's F32 <c>scale * PK_LOG2E</c> for a scale moved <paramref name="ulps"/> F32 ulps from 1/sqrt(hd).</summary>
    private static float ScaleLog2(int hd, int ulps)
    {
        float sc = 1f / MathF.Sqrt(hd);
        for (int k = 0; k < Math.Abs(ulps); k++) sc = ulps > 0 ? MathF.BitIncrement(sc) : MathF.BitDecrement(sc);
        return sc * 1.4426950408889634f;
    }

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void LongPrefill_NewEntries_NoFurtherFromF16ThanLegacy(string modelName)
    {
        string modelPath = ModelPath(modelName);
        Skip.If(!File.Exists(modelPath), $"{modelName} GGUF not found ({modelPath})");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        int n = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TQ_PREFILL_DEPTH_TOKENS"), out int nn) ? nn : 4096;
        const int chunk = 1024, steps = 32;
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora",
                                     "wikitext-2-raw", "wiki.test.lf.raw");
        string ptx = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "ptx"));

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] all = tokenizer.Encode(File.ReadAllText(corpus).Substring(0, 400_000));
        Assert.True(all.Length > n + steps + 1, $"corpus too short: {all.Length}");
        int[] toks = all[..(n + steps + 1)];
        int vocab = config.VocabSize, maxSeq = n + steps + 8, hd = config.HeadDim;
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
        Assert.True(CudaKernels.IsAttentionFlashPrefillKvqShape(config.NumAttentionHeads, config.NumKvHeads, hd),
            $"{modelName}: {config.NumAttentionHeads}/{config.NumKvHeads} hd{hd} is not a tensor-core TQ prefill shape");
        _out.WriteLine($"DEPTH model {modelName}: hd{hd} {config.NumAttentionHeads}/{config.NumKvHeads} heads, {config.NumLayers} layers, n={n}");

        List<float[]> Run(DotLLM.Core.Attention.IKvCache cache, Action? afterFirstChunk = null)
        {
            var rows = new List<float[]>();
            float[]? last = null;
            for (int s = 0; s < n; s += chunk)
            {
                int m = Math.Min(chunk, n - s);
                var pos = Enumerable.Range(s, m).ToArray();
                using var l = model.Forward(toks.AsSpan(s, m).ToArray(), pos, 0, cache);
                last = Row(l, vocab);
                if (s == 0) afterFirstChunk?.Invoke();
            }
            rows.Add(last!);
            for (int k = 0; k < steps; k++)
            {
                using var l = model.Forward([toks[n + k]], [n + k], 0, cache);
                rows.Add(Row(l, vocab));
            }
            return rows;
        }
        static bool Same(List<float[]> a, List<float[]> b) => a.Zip(b).All(p => p.First.AsSpan().SequenceEqual(p.Second));

        var gate = new List<(int bits, Dictionary<CudaKernels.TqPrefillVariant, (double meanKl, int agree)> stats, double band, int topBand)>();
        List<float[]> f16;
        using (var c = model.CreateKvCache(maxSeq)) f16 = Run(c);
        foreach (int bits in new[] { 4, 3, 2, 8 })
        {
            var codecK = new TurboQuantCodec(hd, bits, Seed);
            var codecV = new TurboQuantCodec(hd, bits, Seed ^ TurboQuantCodec.VSeedMask);
            List<float[]> oracle;
            using (var c = model.CreateTurboQuantKvCache(maxSeq, codecK.MseBits, codecK.Centroids, codecK.RotationSigns,
                                                         codecV.RotationSigns, codecK.InvSqrtD)) oracle = Run(c);
            var kv = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, bits, Seed);
            Report(bits, "oracle", oracle, f16, oracle, 0);
            bool hasCpa = hd * bits / 32 % 4 == 0;   // staged needs 16-byte code rows (no hd64 tq3)
            var entries = new List<CudaKernels.TqPrefillVariant> { CudaKernels.TqPrefillVariant.Legacy, CudaKernels.TqPrefillVariant.Specialized };
            if (hasCpa) entries.Add(CudaKernels.TqPrefillVariant.Staged);
            var arms = new Dictionary<CudaKernels.TqPrefillVariant, List<float[]>>();
            var stats = new Dictionary<CudaKernels.TqPrefillVariant, (double meanKl, int agree)>();
            foreach (var v in entries)
            {
                long l0 = CudaKernels.AttentionFlashPrefillKvTqLaunches, s0 = CudaKernels.AttentionFlashPrefillKvTqSpecLaunches,
                     c0 = CudaKernels.AttentionFlashPrefillKvTqCpaLaunches;
                CudaKernels.TqPrefillVariantOverride = v;
                try { using var c = model.CreateKvCache(maxSeq, kv); arms[v] = Run(c); }
                finally { CudaKernels.TqPrefillVariantOverride = null; }
                long flash = CudaKernels.AttentionFlashPrefillKvTqLaunches - l0;
                long spec = CudaKernels.AttentionFlashPrefillKvTqSpecLaunches - s0, cpa = CudaKernels.AttentionFlashPrefillKvTqCpaLaunches - c0;
                Assert.True(flash > 0, "the tensor-core TQ prefill never ran");
                Assert.Equal(v == CudaKernels.TqPrefillVariant.Specialized ? flash : 0, spec);
                Assert.Equal(v == CudaKernels.TqPrefillVariant.Staged ? flash : 0, cpa);
                stats[v] = Report(bits, v.ToString(), arms[v], f16, oracle, flash);
            }
            if (hasCpa)
            {
                // E1 and E3 decode the same codes through the same table with the same arithmetic: bit-identical by construction. This is
                // the only zero-tolerance check on the staging pipeline over a long prefill (final review M1).
                Assert.True(Same(arms[CudaKernels.TqPrefillVariant.Specialized], arms[CudaKernels.TqPrefillVariant.Staged]),
                    $"{modelName} tq{bits}: Specialized and Staged logits are not bit-identical");
            }

            // Drift control (James 2026-11-22, "measure the noise, then decide"; widened after final review I1): the LEGACY entry with one
            // deliberate 1-ulp perturbation per arm. The run is deterministic, so how far each arm moves mean KL(f16||arm) samples the
            // noise of two correct implementations; the band is the max over every arm that moved. An arm that cannot or did not move
            // any logit bitwise is dropped with its reason. At hd128 the +1-ulp scale arm is inert BY CONSTRUCTION (1/sqrt(128) * log2e
            // rounds to the same F32: the product's ulp is twice the input's), which is why the 2-ulp and norm-LSB arms exist.
            double band = 0;
            int topBand = 0;   // top-1 spread of the same control arms (James 2026-11-23)
            var moved = new List<string>();
            float baseScale = ScaleLog2(hd, 0);
            var scalesRun = new Dictionary<int, string>();   // scale*log2e bits -> the arm that already ran that kernel
            foreach (var ctl in Controls)
            {
                if (ctl.ScaleUlps != 0)
                {
                    int sb = BitConverter.SingleToInt32Bits(ScaleLog2(hd, ctl.ScaleUlps));
                    if (sb == BitConverter.SingleToInt32Bits(baseScale))
                    {
                        _out.WriteLine($"DEPTH b{bits} ctrl {ctl.Name}: DROPPED, inert by construction (scale*log2e is bit-identical at hd{hd})");
                        continue;
                    }
                    // Two input steps that round to the same scale*log2e run the identical kernel: one sample, not two (at hd128 the
                    // -1 and -2 ulp arms coincide).
                    if (scalesRun.TryGetValue(sb, out string? same))
                    {
                        _out.WriteLine($"DEPTH b{bits} ctrl {ctl.Name}: DROPPED, same scale*log2e as {same} (identical kernel)");
                        continue;
                    }
                    scalesRun[sb] = ctl.Name;
                }
                List<float[]> ctrl;
                CudaKernels.TqPrefillVariantOverride = CudaKernels.TqPrefillVariant.Legacy;
                CudaKernels.TqPrefillScaleUlpNudge = ctl.ScaleUlps;
                try
                {
                    using var c = (CudaTurboQuantInPlaceKvCache)model.CreateKvCache(maxSeq, kv);
                    ctrl = Run(c, ctl.FlipNormSide == 0 ? null : () => FlipNormLsb(c, ctl.FlipNormSide));
                }
                finally { CudaKernels.TqPrefillVariantOverride = null; CudaKernels.TqPrefillScaleUlpNudge = 0; }
                if (Same(arms[CudaKernels.TqPrefillVariant.Legacy], ctrl))
                {
                    _out.WriteLine($"DEPTH b{bits} ctrl {ctl.Name}: DROPPED, no logit moved bitwise");
                    continue;
                }
                var (klC, agC) = Report(bits, $"ctrl {ctl.Name}", ctrl, f16, oracle, 0);
                band = Math.Max(band, Math.Abs(klC - stats[CudaKernels.TqPrefillVariant.Legacy].meanKl));
                topBand = Math.Max(topBand, Math.Abs(agC - stats[CudaKernels.TqPrefillVariant.Legacy].agree));
                moved.Add(ctl.Name);
            }
            // A control that cannot move is no control (CLAUDE.md: prove the test can fail), and one moving arm is a single sample.
            Assert.True(moved.Count >= 2, $"{modelName} tq{bits}: only {moved.Count} drift-control arm(s) moved the logits; the band is not measured");
            _out.WriteLine($"DEPTH b{bits}: drift band on mean KL = {band:E4} ({100 * band / stats[CudaKernels.TqPrefillVariant.Legacy].meanKl:F2}% of legacy) over {moved.Count} arms: {string.Join(", ", moved)}; top-1 spread {topBand}/33");
            gate.Add((bits, stats, band, topBand));
        }
        // The gate (James 2026-11-22), evaluated after every arm is reported so a failure still prints all the numbers: each new entry
        // at a width whose default is not legacy must have mean KL no worse than legacy + band and top-1 no worse than legacy - 1.
        var failures = new List<string>();
        foreach (var (bits, st, band, topBand) in gate)
        {
            var (klL, agL) = st[CudaKernels.TqPrefillVariant.Legacy];
            if (CudaKernels.DefaultTqPrefillVariantFor(bits) == CudaKernels.TqPrefillVariant.Legacy)
            {
                _out.WriteLine($"DEPTH b{bits}: default entry is Legacy; spec / cpa not gated at this width (opt-in only)");
                continue;
            }
            foreach (var v in st.Keys.Where(k => k != CudaKernels.TqPrefillVariant.Legacy))
            {
                var (kl, ag) = st[v];
                // Top-1 floor (James 2026-11-23, "band top-1 too"): legacy's top-1 minus the top-1 spread of the same control arms that
                // set the KL band, derived per run. It replaces the fixed "legacy minus 1", which had no noise band.
                int floor = agL - topBand;
                bool ok = kl <= klL + band && ag >= floor;
                _out.WriteLine($"DEPTH GATE {modelName} tq{bits} {v}: mean KL {kl:E4} vs legacy {klL:E4} + band {band:E4}; top-1 {ag} vs {agL} - spread {topBand} = {floor}: {(ok ? "PASS" : "FAIL")}");
                if (!ok) failures.Add($"tq{bits} {v} (KL {kl:E4} vs limit {klL + band:E4}, top-1 {ag} vs floor {floor})");
            }
        }
        Assert.True(failures.Count == 0, $"{modelName}: " + string.Join("; ", failures));
    }

    /// <summary>Flips the last bit of one F16 norm (layer 0, position 0, kv head 0) of side 1 = K or 2 = V: a 1-ulp change of a stored value.</summary>
    private static void FlipNormLsb(CudaTurboQuantInPlaceKvCache c, int side)
    {
        nint ptr = side == 1 ? c.GetKeyNormsPtr(0) : c.GetValueNormsPtr(0);
        var h = new ushort[1];
        fixed (ushort* hp = h)
        {
            DotLLM.Cuda.Interop.CudaDriverApi.cuMemcpyDtoH_v2((nint)hp, ptr, 2).ThrowOnError();
            h[0] ^= 1;
            DotLLM.Cuda.Interop.CudaDriverApi.cuMemcpyHtoD_v2(ptr, (nint)hp, 2).ThrowOnError();
        }
    }

    private (double meanKl, int agree) Report(int bits, string arm, List<float[]> rows, List<float[]> f16, List<float[]> oracle, long flash)
    {
        var kl = rows.Select((r, i) => Kl(f16[i], r)).ToArray();
        var (minCosOr, meanCosOr) = CosStats(rows, oracle);
        int agree = rows.Select((r, i) => Argmax(r) == Argmax(f16[i]) ? 1 : 0).Sum();
        _out.WriteLine($"DEPTH b{bits} {arm,-11}: KL(f16||arm) mean {kl.Average():E4} max {kl.Max():E4} prefill-row {kl[0]:E4}; " +
                       $"cos(arm,oracle) min {minCosOr:F6} mean {meanCosOr:F6}; argmax==f16 {agree}/{rows.Count}; flash launches {flash}");
        return (kl.Average(), agree);
    }

    private static (double min, double mean) CosStats(List<float[]> a, List<float[]> b)
    {
        var c = a.Select((r, i) => Cos(r, b[i])).ToArray();
        return (c.Min(), c.Average());
    }

    private static double Cos(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp) + mp, lzq = Math.Log(zq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++) { double lp = p[i] - lzp; kl += Math.Exp(lp) * (lp - (q[i] - lzq)); }
        return kl;
    }

    private static int Argmax(float[] r) { int b = 0; for (int i = 1; i < r.Length; i++) if (r[i] > r[b]) b = i; return b; }

    private static float[] Row(DotLLM.Core.Tensors.ITensor logits, int vocab)
    {
        var row = new float[vocab];
        float* p = (float*)logits.DataPointer;
        int off = (logits.Shape[0] - 1) * vocab;
        for (int i = 0; i < vocab; i++) row[i] = p[off + i];
        return row;
    }
}
