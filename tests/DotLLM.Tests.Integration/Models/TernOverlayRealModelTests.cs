using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models;

/// <summary>
/// <c>--tensor-overlay</c> with TERN_B64_F16 on REAL cached models (issue #1706): OLMoE-1B-7B Q4_K_M (indexed per-expert mixed formats inside one layer,
/// norm_topk_prob = false) and SmolLM2-135M f16 (dense FFN; an F16 base has no activation quantisation elsewhere, so the F32-activation reference can be held to float noise). Experts / FFNs are ternarised from their OWN dequantised weights (absmean per 64 columns), then
/// the model is run (a) with the ternary overlay (packed trits, int8 activations), (b) with the same numbers stored as an exact F16 dense copy
/// (trit * fp16-scale, stored as exact F32 so the F32 dense kernels see identical numbers), and (c) unmodified. (a) vs (b) isolates the kernel (int8 activation quantisation only); (c) is the control that
/// ternarising changes the model visibly. Gated on the cached GGUFs; nothing is downloaded.
/// </summary>
[Collection("TernKernelSerial")]
public sealed class TernOverlayRealModelTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-ternreal-{Guid.NewGuid():N}");

    public TernOverlayRealModelTests(ITestOutputHelper output) { _out = output; Directory.CreateDirectory(_dir); }

    public void Dispose() { try { Directory.Delete(_dir, true); } catch { /* best effort */ } }

    private const string Text =
        "The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the banks of the Tiber into the "
        + "capital of a vast realm. Its legions marched across Europe, and its roads linked provinces from Britain to Syria. Trade flourished along the "
        + "Mediterranean, and aqueducts carried water to cities. In the third century the empire faced civil war, plague and invasion, yet it endured.";

    private sealed record Ov(string Name, int In, int Out, QuantizationType Qt, byte[] Data);

    private static (sbyte[] Trits, float[] Scales, float[] Dense) Ternarize(float[] w)
    {
        var trits = new sbyte[w.Length]; var scales = new float[w.Length / 64]; var f16 = new float[w.Length];
        for (int b = 0; b < scales.Length; b++)
        {
            double mean = 0;
            for (int j = 0; j < 64; j++) mean += Math.Abs(w[b * 64 + j]);
            float s = (float)(Half)(mean / 64);
            scales[b] = s;
            for (int j = 0; j < 64; j++)
            {
                int i = b * 64 + j;
                int t = s == 0 ? 0 : (int)Math.Clamp(Math.Round(w[i] / s), -1, 1);
                trits[i] = (sbyte)t;
                f16[i] = t * s;
            }
        }
        return (trits, scales, f16);
    }

    private static unsafe (Ov Tern, Ov Dense) Pair(string name, float[] w, int inDim, int outDim)
    {
        var (trits, scales, f16) = Ternarize(w);
        var packed = TernB64F16.Pack(trits, scales);
        var viaDequant = new float[w.Length];
        unsafe { fixed (byte* pp = packed) TernB64F16.Dequantize(pp, w.Length, viaDequant); }
        for (int i = 0; i < w.Length; i++) if (viaDequant[i] != f16[i]) throw new InvalidOperationException($"{name}: dense copy differs from dequantised packed at {i}: {f16[i]} vs {viaDequant[i]}");
        return (new Ov(name, inDim, outDim, QuantizationType.TERN_B64_F16, packed),
                new Ov(name, inDim, outDim, QuantizationType.F32, MemoryMarshal.AsBytes(f16.AsSpan()).ToArray()));
    }

    private string Write(string baseSha, IEnumerable<Ov> tensors, string tag)
    {
        var w = new GgufWriter();
        w.AddString("general.architecture", "dotllm-overlay");
        w.AddString(TensorOverlay.BaseSha256Key, baseSha);
        foreach (var t in tensors) w.AddTensor(t.Name, [t.In, t.Out], (uint)t.Qt, t.Data);
        string path = Path.Combine(_dir, tag + ".gguf");
        File.WriteAllBytes(path, w.Build());
        return path;
    }

    private static float[] Dequant(GgufFile g, string tensor, long elemOffset, int count)
    {
        var d = g.TensorsByName[tensor];
        nint basePtr = g.TensorDataPointer(d);
        long blockElems = QuantFormat.TryGetInfo(d.QuantizationType)?.GroupSize ?? 1;
        Assert.Equal(0, elemOffset % blockElems);
        long byteOff = d.QuantizationType.ComputeByteCount(elemOffset);
        var f = new float[count];
        Dequantize.ToFloat32(basePtr + (nint)byteOff, count, d.QuantizationType, f);
        return f;
    }

    private static unsafe float[][] Logits(string basePath, string? overlay, int[]? ids, out int[] usedIds)
    {
        using var gguf = GgufFile.Open(basePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        usedIds = ids ?? GgufBpeTokenizerFactory.Load(gguf.Metadata).Encode(Text);
        if (overlay is not null) TensorOverlay.Attach(gguf, basePath, overlay, verifyBaseHash: true);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        gguf.Overlay?.VerifyAllApplied();
        using ITensor t = model.Forward(usedIds, Enumerable.Range(0, usedIds.Length).ToArray(), deviceId: -1);
        int rows = t.Shape[0], vocab = t.Shape[1];
        var res = new float[rows][];
        for (int r = 0; r < rows; r++) res[r] = new ReadOnlySpan<float>((float*)t.DataPointer + (long)r * vocab, vocab).ToArray();
        return res;
    }

    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x)
        {
            double m = x.Max(); double s = 0; foreach (float v in x) s += Math.Exp(v - m);
            double lz = m + Math.Log(s);
            return x.Select(v => v - lz).ToArray();
        }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < p.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private static int ArgMax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    private void Report(string label, float[][] baseL, float[][] tern, float[][] dense, float[][] ternF32, bool exactF32Reference)
    {
        int n = baseL.Length;
        double klKernel = 0, klEffect = 0, klF32 = 0, maxKernel = 0; int agree = 0, agreeBase = 0;
        for (int r = 0; r < n; r++)
        {
            double k1 = Kl(dense[r], tern[r]); klKernel += k1; maxKernel = Math.Max(maxKernel, k1);
            klEffect += Kl(baseL[r], dense[r]);
            klF32 += Kl(dense[r], ternF32[r]);
            if (ArgMax(dense[r]) == ArgMax(tern[r])) agree++;
            if (ArgMax(baseL[r]) == ArgMax(dense[r])) agreeBase++;
        }
        _out.WriteLine($"{label}: rows {n}; mean KL(dense-copy || ternary int8) {klKernel / n:E3} (max {maxKernel:E3}); mean KL(base || dense-copy) {klEffect / n:E3}; "
                       + $"top-1 agree ternary/dense-copy {agree}/{n}, base/dense-copy {agreeBase}/{n}; mean KL(dense-copy || ternary F32-act) {klF32 / n:E3}");
        // A quantised base (OLMoE Q4_K) re-quantises activations in every other matmul, which digitises float noise (1e-7 -> a different int8 step), so the
        // F32-activation reference only equals the dense copy up to that floor; an F16 base has no such floor and must match to float noise.
        Assert.True(klEffect / n > (exactF32Reference ? 20 : 10) * (klKernel / n), $"{label}: control failed - ternarising (KL {klEffect / n:E3}) must dwarf the kernel gap (KL {klKernel / n:E3})");
        Assert.True(agree >= n - Math.Max(2, n / 8), $"{label}: top-1 agreement ternary vs dense copy only {agree}/{n}");
        Assert.True(exactF32Reference ? klF32 / n < 1e-6 : klF32 / n < klEffect / n / 10, $"{label}: F32-activation reference must equal the dense copy (KL {klF32 / n:E3})");
    }

    private float[][] LogitsF32Act(string basePath, string ov, int[] ids)
    {
        bool prev = TernB64F16.UseF32ActivationReference;
        try { TernB64F16.UseF32ActivationReference = true; return Logits(basePath, ov, ids, out _); }
        finally { TernB64F16.UseF32ActivationReference = prev; }
    }

    [SkippableFact]
    public void Olmoe_PerExpertTernaryOverlay_MatchesDenseCopy_RealTensors()
    {
        var fx = TestFixtureResolver.ResolveFile("DOTLLM_OLMOE_GGUF", "bartowski", "OLMoE-1B-7B-0924-Instruct-GGUF", "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("OLMoE-1B-7B Q4_K_M GGUF"));
        string basePath = fx.Path!;
        string sha = TensorOverlay.BaseHash.Compute(basePath);

        var tern = new List<Ov>(); var dense = new List<Ov>();
        using (var g = GgufFile.Open(basePath))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            Assert.False(cfg.Moe!.NormTopKProb);   // OLMoE: raw softmax probabilities, no renormalisation (the overlay must not disturb it)
            int hidden = cfg.HiddenSize, ff = cfg.Moe.MoeIntermediateSize;
            // Two layers x 8 experts, all projections, beside 56 untouched original-format experts per layer (Q4_K / Q6_K banks).
            foreach (int L in new[] { 2, 9 })
                for (int E = 0; E < 8; E++)
                {
                    long per = (long)ff * hidden;
                    var gate = Pair($"blk.{L}.ffn_gate.{E}.weight", Dequant(g, $"blk.{L}.ffn_gate_exps.weight", E * per, (int)per), hidden, ff);
                    var up = Pair($"blk.{L}.ffn_up.{E}.weight", Dequant(g, $"blk.{L}.ffn_up_exps.weight", E * per, (int)per), hidden, ff);
                    var down = Pair($"blk.{L}.ffn_down.{E}.weight", Dequant(g, $"blk.{L}.ffn_down_exps.weight", E * per, (int)per), ff, hidden);
                    tern.AddRange([gate.Tern, up.Tern, down.Tern]); dense.AddRange([gate.Dense, up.Dense, down.Dense]);
                }
        }
        string ovT = Write(sha, tern, "olmoe-tern"), ovD = Write(sha, dense, "olmoe-dense");

        var baseL = Logits(basePath, null, null, out int[] ids);
        var denseL = Logits(basePath, ovD, ids, out _);
        var ternL = Logits(basePath, ovT, ids, out _);
        Report("OLMoE-1B-7B Q4_K_M, 2 layers x 8 experts x 3 projections ternary", baseL, ternL, denseL, LogitsF32Act(basePath, ovT, ids), exactF32Reference: false);
    }

    [SkippableFact]
    public void SmolLM2_DenseTernaryOverlay_MatchesDenseCopy_RealTensors()
    {
        var fx = TestFixtureResolver.ResolveFile("DOTLLM_SMOLLM2_135M_INSTRUCT_F16_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-f16.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("SmolLM2-135M-Instruct f16 GGUF"));
        string basePath = fx.Path!;
        string sha = TensorOverlay.BaseHash.Compute(basePath);

        var tern = new List<Ov>(); var dense = new List<Ov>();
        using (var g = GgufFile.Open(basePath))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            int hidden = cfg.HiddenSize, ff = cfg.IntermediateSize;
            Assert.Equal((576, 1536), (hidden, ff));
            foreach (int L in new[] { 12, 14, 16 })   // a few layers: an untrained all-layer ternarisation is garbage and amplifies any engine difference
            {
                int n = hidden * ff;
                var gate = Pair($"blk.{L}.ffn_gate.weight", Dequant(g, $"blk.{L}.ffn_gate.weight", 0, n), hidden, ff);
                var up = Pair($"blk.{L}.ffn_up.weight", Dequant(g, $"blk.{L}.ffn_up.weight", 0, n), hidden, ff);
                var down = Pair($"blk.{L}.ffn_down.weight", Dequant(g, $"blk.{L}.ffn_down.weight", 0, n), ff, hidden);
                tern.AddRange([gate.Tern, up.Tern, down.Tern]); dense.AddRange([gate.Dense, up.Dense, down.Dense]);
            }
        }
        string ovT = Write(sha, tern, "smol-tern"), ovD = Write(sha, dense, "smol-dense");
        var baseL = Logits(basePath, null, null, out int[] ids);
        var denseL = Logits(basePath, ovD, ids, out _);
        var ternL = Logits(basePath, ovT, ids, out _);
        Report("SmolLM2-135M f16, FFN of layers 12/14/16 ternary", baseL, ternL, denseL, LogitsF32Act(basePath, ovT, ids), exactF32Reference: true);
    }
}
