using System.Runtime.InteropServices;
using System.Security.Cryptography;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// <c>--tensor-overlay</c> (issue #1706) on tiny synthetic OLMoE-style (stacked-expert MoE, norm_topk_prob = false) and dense (SmolLM3/Llama-style) models:
/// loader contract (unknown names, shapes, hash, invalid codes, "never silently ignored") and end-to-end behaviour.
/// The end-to-end claims each have a discriminating control: an identity overlay (F32 copies of the original tensors) must reproduce the base BIT-EXACTLY
/// (so any wrong expert/projection index or format dispatch shows), and a ternary overlay must equal the same overlay stored as the dequantized dense copy
/// (up to int8 activation quantization) while differing visibly from the unmodified base.
/// </summary>
[Collection("TernKernelSerial")]   // toggles TernB64F16.UseF32ActivationReference, so it must not run beside the kernel tests
public sealed class TensorOverlayTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-overlay-{Guid.NewGuid():N}");
    private readonly string _hashCache;

    public TensorOverlayTests()
    {
        Directory.CreateDirectory(_dir);
        _hashCache = Path.Combine(_dir, "hash-cache");
        Environment.SetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR", _hashCache);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR", null);
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static readonly int[] Prompt = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 3, 40];

    private static SyntheticGraniteConfig MoeCfg() => new()
    {
        Arch = "olmoe", Layers = 3, Experts = 8, ExpertsUsed = 2, Hidden = 128, FeedForward = 128,
        EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0,
    };

    private static SyntheticGraniteConfig DenseCfg() => new()
    {
        Arch = "smollm3", Layers = 4, Hidden = 128, FeedForward = 192,
        EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0,
    };

    // ───────────────────────── helpers ─────────────────────────

    private (string Path, SyntheticGraniteGguf.Weights Weights) WriteBase(SyntheticGraniteConfig cfg)
    {
        string path = Path.Combine(_dir, $"{cfg.Arch}-{Guid.NewGuid():N}.gguf");
        var wts = SyntheticGraniteGguf.BuildWeights(cfg);
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        return (path, wts);
    }

    private static string Sha256Hex(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();

    internal sealed record OvTensor(string Name, int In, int Out, QuantizationType Qt, byte[] Data);

    private string WriteOverlay(string? baseSha, IEnumerable<OvTensor> tensors)
    {
        var w = new GgufWriter();
        w.AddString("general.architecture", "dotllm-overlay");
        if (baseSha is not null) w.AddString(TensorOverlay.BaseSha256Key, baseSha);
        foreach (var t in tensors) w.AddTensor(t.Name, [t.In, t.Out], (uint)t.Qt, t.Data);
        string path = Path.Combine(_dir, $"overlay-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, w.Build());
        return path;
    }

    private static byte[] F32Bytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    /// <summary>Absmean ternarisation per 64-column block (scale rounded to fp16): (trits, scales, dequantised values).</summary>
    internal static (sbyte[] Trits, float[] Scales, float[] Dequant) Ternarize(float[] w, int k)
    {
        var trits = new sbyte[w.Length];
        var scales = new float[w.Length / 64];
        var deq = new float[w.Length];
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
                deq[i] = t * s;
            }
        }
        return (trits, scales, deq);
    }

    private static float[] Slice(float[] all, int index, int perItem) => all.AsSpan(index * perItem, perItem).ToArray();

    private (OvTensor Tern, OvTensor Dense) TernPair(string name, float[] w, int inDim, int outDim)
    {
        var (trits, scales, deq) = Ternarize(w, inDim);
        return (new OvTensor(name, inDim, outDim, QuantizationType.TERN_B64_F16, TernB64F16.Pack(trits, scales)),
                new OvTensor(name, inDim, outDim, QuantizationType.F32, F32Bytes(deq)));
    }

    private static unsafe float[] LastRow(ITensor t)
    {
        int vocab = t.Shape[1];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static float[] Logits(string basePath, string? overlayPath, bool verifyHash = false)
    {
        using var gguf = GgufFile.Open(basePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (overlayPath is not null) TensorOverlay.Attach(gguf, basePath, overlayPath, verifyBaseHash: verifyHash);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        gguf.Overlay?.VerifyAllApplied();
        using ITensor logits = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), deviceId: -1);
        return LastRow(logits);
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    // ───────────────────────── end-to-end: identity (plumbing) ─────────────────────────

    [Fact]
    public void Moe_IdentityOverlay_F32CopiesOfOriginalExperts_IsBitExact_AndWrongIndexIsNot()
    {
        var cfg = MoeCfg();
        var (basePath, wts) = WriteBase(cfg);
        int perGateUp = cfg.FeedForward * cfg.Hidden;
        var tensors = new List<OvTensor>();
        // A scattered subset: different experts for different projections, on two layers.
        (int L, int E, int P)[] picks = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 2, 1), (1, 5, 0), (2, 7, 2), (2, 0, 0)];
        foreach (var (L, E, P) in picks)
        {
            var lw = wts.Layers[L];
            string pn = TensorOverlay.ProjectionNames[P];
            float[] src = P == 0 ? lw.GateExps! : P == 1 ? lw.UpExps! : lw.DownExps!;
            int inDim = P == 2 ? cfg.FeedForward : cfg.Hidden, outDim = P == 2 ? cfg.Hidden : cfg.FeedForward;
            tensors.Add(new OvTensor($"blk.{L}.ffn_{pn}.{E}.weight", inDim, outDim, QuantizationType.F32, F32Bytes(Slice(src, E, perGateUp))));
        }
        string ov = WriteOverlay(Sha256Hex(basePath), tensors);

        var baseLogits = Logits(basePath, null);
        var ovLogits = Logits(basePath, ov, verifyHash: true);
        Assert.Equal(baseLogits, ovLogits);   // bit-exact: same F32 kernel, same bytes, only the side-table routing differs

        // Control (the test can fail): the SAME overlay but one expert's gate bank swapped for another expert's weights must change the logits.
        var wrong = new List<OvTensor>(tensors);
        var lw0 = wts.Layers[0];
        wrong[0] = new OvTensor("blk.0.ffn_gate.1.weight", cfg.Hidden, cfg.FeedForward, QuantizationType.F32, F32Bytes(Slice(lw0.GateExps!, 4, perGateUp)));
        string wrongPath = WriteOverlay(Sha256Hex(basePath), wrong);
        Assert.NotEqual(baseLogits, Logits(basePath, wrongPath, verifyHash: true));
    }

    [Fact]
    public void Dense_IdentityOverlay_F32CopiesOfOriginalWeights_IsBitExact()
    {
        var cfg = DenseCfg();
        var (basePath, wts) = WriteBase(cfg);
        var tensors = new List<OvTensor>
        {
            new("blk.1.ffn_gate.weight", cfg.Hidden, cfg.FeedForward, QuantizationType.F32, F32Bytes(wts.Layers[1].Gate!)),
            new("blk.1.ffn_up.weight", cfg.Hidden, cfg.FeedForward, QuantizationType.F32, F32Bytes(wts.Layers[1].Up!)),
            new("blk.3.ffn_down.weight", cfg.FeedForward, cfg.Hidden, QuantizationType.F32, F32Bytes(wts.Layers[3].Down!)),
        };
        string ov = WriteOverlay(Sha256Hex(basePath), tensors);
        Assert.Equal(Logits(basePath, null), Logits(basePath, ov, verifyHash: true));
    }

    // ───────────────────────── end-to-end: ternary vs its own dequantised dense copy ─────────────────────────

    [Fact]
    public void Moe_TernaryOverlay_EqualsDequantisedDenseCopy_AndDiffersFromBase()
    {
        var cfg = MoeCfg();
        var (basePath, wts) = WriteBase(cfg);
        int per = cfg.FeedForward * cfg.Hidden;
        var tern = new List<OvTensor>(); var dense = new List<OvTensor>();
        // Every expert of every layer, all three projections => every routed token exercises the ternary kernel.
        for (int L = 0; L < cfg.Layers; L++)
            for (int E = 0; E < cfg.Experts; E++)
                for (int P = 0; P < 3; P++)
                {
                    var lw = wts.Layers[L];
                    float[] src = P == 0 ? lw.GateExps! : P == 1 ? lw.UpExps! : lw.DownExps!;
                    int inDim = P == 2 ? cfg.FeedForward : cfg.Hidden, outDim = P == 2 ? cfg.Hidden : cfg.FeedForward;
                    var (t, d) = TernPair($"blk.{L}.ffn_{TensorOverlay.ProjectionNames[P]}.{E}.weight", Slice(src, E, per), inDim, outDim);
                    tern.Add(t); dense.Add(d);
                }
        string sha = Sha256Hex(basePath);
        string ovT = WriteOverlay(sha, tern), ovD = WriteOverlay(sha, dense);

        var baseLogits = Logits(basePath, null);
        var ternLogits = Logits(basePath, ovT, verifyHash: true);
        var denseLogits = Logits(basePath, ovD, verifyHash: true);

        AssertTernMatchesDenseCopy(basePath, ovT, ovD, baseLogits, ternLogits, denseLogits);
    }

    [Fact]
    public void Dense_TernaryOverlay_EqualsDequantisedDenseCopy_AndDiffersFromBase()
    {
        var cfg = DenseCfg();
        var (basePath, wts) = WriteBase(cfg);
        var tern = new List<OvTensor>(); var dense = new List<OvTensor>();
        for (int L = 0; L < cfg.Layers; L++)
        {
            var lw = wts.Layers[L];
            var g = TernPair($"blk.{L}.ffn_gate.weight", lw.Gate!, cfg.Hidden, cfg.FeedForward);
            var u = TernPair($"blk.{L}.ffn_up.weight", lw.Up!, cfg.Hidden, cfg.FeedForward);
            var d = TernPair($"blk.{L}.ffn_down.weight", lw.Down!, cfg.FeedForward, cfg.Hidden);
            tern.AddRange([g.Tern, u.Tern, d.Tern]); dense.AddRange([g.Dense, u.Dense, d.Dense]);
        }
        string sha = Sha256Hex(basePath);
        var baseLogits = Logits(basePath, null);
        var ternLogits = Logits(basePath, WriteOverlay(sha, tern), verifyHash: true);
        var denseLogits = Logits(basePath, WriteOverlay(sha, dense), verifyHash: true);
        AssertTernMatchesDenseCopy(basePath, WriteOverlay(sha, tern), WriteOverlay(sha, dense), baseLogits, ternLogits, denseLogits);
    }

    [Fact]
    public void Moe_MixedFormats_OneExpertTernaryGate_OriginalUpDown_OtherExpertsOriginal()
    {
        // Per-(expert, projection) independence: ternary gate on one expert while its up/down and every other expert stay F32 originals.
        var cfg = MoeCfg();
        var (basePath, wts) = WriteBase(cfg);
        int per = cfg.FeedForward * cfg.Hidden;
        var (t, d) = TernPair("blk.1.ffn_gate.3.weight", Slice(wts.Layers[1].GateExps!, 3, per), cfg.Hidden, cfg.FeedForward);
        string sha = Sha256Hex(basePath);
        var a = Logits(basePath, WriteOverlay(sha, [t]), true);
        var b = Logits(basePath, WriteOverlay(sha, [d]), true);
        Assert.True(MaxAbsDiff(a, b) < 0.02 * Math.Max(1.0, MaxAbs(b)));
    }


    private static double Rms(float[] a, float[] b) { double s = 0; for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; s += d * d; } return Math.Sqrt(s / a.Length); }
    private static double Rms(float[] a) => Rms(a, new float[a.Length]);

    /// <summary>
    /// (1) With F32 activations the ternary kernel and the dequantised dense copy are the SAME function of the same numbers: logits agree to float noise.
    /// (2) With the production int8 activations the gap is pure activation quantisation: a few percent RMS of the logits on a random-weight toy, and far below
    /// what ternarising does to the model (the control: overlay vs unmodified base).
    /// </summary>
    private void AssertTernMatchesDenseCopy(string basePath, string ovT, string ovD, float[] baseLogits, float[] ternInt8, float[] denseLogits)
    {
        double scale = Rms(denseLogits);
        double effect = Rms(ternInt8, baseLogits);
        double int8Gap = Rms(ternInt8, denseLogits);
        bool prev = TernB64F16.UseF32ActivationReference;
        try
        {
            TernB64F16.UseF32ActivationReference = true;
            var ternF32 = Logits(basePath, ovT, verifyHash: true);
            double f32Gap = Rms(ternF32, denseLogits);
            Assert.True(f32Gap < 2e-4 * scale, $"F32-activation ternary vs dequantised dense copy: rms gap {f32Gap} (logit rms {scale})");
        }
        finally { TernB64F16.UseF32ActivationReference = prev; }
        Assert.True(effect > 0.3 * scale, $"ternarising must visibly change the model: rms effect {effect} vs logit rms {scale}");
        Assert.True(int8Gap < 0.05 * scale, $"int8-activation ternary vs dequantised dense copy: rms gap {int8Gap} (logit rms {scale}, effect {effect})");
        Assert.True(effect > 8 * int8Gap, $"control: effect {effect} must dwarf the kernel gap {int8Gap}");
    }

    private static double MaxAbs(float[] a) => a.Max(v => Math.Abs((double)v));
    private static int ArgMax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    // ───────────────────────── loader contract: errors are hard errors ─────────────────────────

    private OvTensor ZeroTern(string name, int inDim, int outDim)
        => new(name, inDim, outDim, QuantizationType.TERN_B64_F16, TernB64F16.Pack(new sbyte[inDim * outDim], new float[inDim * outDim / 64]));

    [Theory]
    [InlineData("blk.0.ffn_gate_exps.weight")]
    [InlineData("blk.0.attn_q.weight")]
    [InlineData("blk.0.ffn_gate.1.2.weight")]
    [InlineData("ffn_gate.weight")]
    [InlineData("blk.x.ffn_up.weight")]
    [InlineData("blk.0.ffn_gate.1.bias")]
    public void UnknownTensorName_IsRejectedAtOpen(string name)
    {
        var (basePath, _) = WriteBase(MoeCfg());
        string ov = WriteOverlay(Sha256Hex(basePath), [ZeroTern(name, 128, 128)]);
        var ex = Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(ov, basePath, verifyBaseHash: false));
        Assert.Contains(name, ex.Message);
    }

    [Fact]
    public void NameGrammar_ParsesLayerProjectionExpert()
    {
        Assert.True(TensorOverlay.TryParseName("blk.12.ffn_down.63.weight", out int l, out int p, out int e));
        Assert.Equal((12, 2, 63), (l, p, e));
        Assert.True(TensorOverlay.TryParseName("blk.3.ffn_up.weight", out l, out p, out e));
        Assert.Equal((3, 1, -1), (l, p, e));
        Assert.False(TensorOverlay.TryParseName("blk.3.ffn_gate.weight.bak", out _, out _, out _));
    }

    [Fact]
    public void ShapeMismatch_IsAnError_ExpertAndDense()
    {
        var (moePath, _) = WriteBase(MoeCfg());
        // gate expert should be [in=128, out=128]; give [64, 128].
        string ov = WriteOverlay(Sha256Hex(moePath), [ZeroTern("blk.0.ffn_gate.0.weight", 64, 128)]);
        var ex = Assert.Throws<InvalidDataException>(() => Logits(moePath, ov));
        Assert.Contains("shape", ex.Message);
        Assert.Contains("blk.0.ffn_gate.0.weight", ex.Message);

        // transposed [out, in] orientation for a non-square projection must also fail.
        var (densePath, _) = WriteBase(DenseCfg());
        string ov2 = WriteOverlay(Sha256Hex(densePath), [ZeroTern("blk.0.ffn_down.weight", 128, 192)]);   // down is [in=192, out=128]
        Assert.Throws<InvalidDataException>(() => Logits(densePath, ov2));
    }

    [Fact]
    public void LayerOrExpertOutOfRange_AndKindMismatch_AreErrors()
    {
        var (moePath, _) = WriteBase(MoeCfg());
        string sha = Sha256Hex(moePath);
        Assert.Throws<InvalidDataException>(() => Logits(moePath, WriteOverlay(sha, [ZeroTern("blk.9.ffn_gate.0.weight", 128, 128)])));     // layer
        Assert.Throws<InvalidDataException>(() => Logits(moePath, WriteOverlay(sha, [ZeroTern("blk.0.ffn_gate.8.weight", 128, 128)])));     // expert (8 experts)
        Assert.Throws<InvalidDataException>(() => Logits(moePath, WriteOverlay(sha, [ZeroTern("blk.0.ffn_gate.weight", 128, 128)])));       // dense name on MoE layer
        var (densePath, _) = WriteBase(DenseCfg());
        Assert.Throws<InvalidDataException>(() => Logits(densePath, WriteOverlay(Sha256Hex(densePath), [ZeroTern("blk.0.ffn_gate.0.weight", 128, 192)]))); // expert name on dense layer
    }

    [Fact]
    public void InDimensionNotMultipleOf64_IsRejectedForTernary()
    {
        var (basePath, _) = WriteBase(MoeCfg() with { FeedForward = 96 });
        // down expert in-dim = 96 (not a multiple of 64).
        var bad = new OvTensor("blk.0.ffn_down.0.weight", 96, 128, QuantizationType.TERN_B64_F16, new byte[18 * 2 * 128]);
        string ov = WriteOverlay(Sha256Hex(basePath), [bad]);
        var ex = Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(ov, basePath, verifyBaseHash: false));
        Assert.Contains("multiple of 64", ex.Message);
    }

    [Fact]
    public void InvalidTritCode_IsRejectedAtOpen_NamingTheTensor()
    {
        var (basePath, _) = WriteBase(MoeCfg());
        var good = ZeroTern("blk.2.ffn_up.5.weight", 128, 128);
        var data = (byte[])good.Data.Clone();
        data[18 * 40 + 2 + 7] = 0xC3;   // block 40, byte 7 -> codes 11,00,00,11
        var bad = good with { Data = data };
        string ov = WriteOverlay(Sha256Hex(basePath), [bad]);
        var ex = Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(ov, basePath, verifyBaseHash: false));
        Assert.Contains("blk.2.ffn_up.5.weight", ex.Message);
        Assert.Contains("0b11", ex.Message);
    }

    [Fact]
    public void BaseHash_Mismatch_Missing_AndOptOut()
    {
        var (basePath, _) = WriteBase(MoeCfg());
        var t = ZeroTern("blk.0.ffn_gate.0.weight", 128, 128);

        string wrongHash = WriteOverlay(new string('0', 64), [t]);
        var ex = Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(wrongHash, basePath));
        Assert.Contains("different base model", ex.Message);
        Assert.Contains(Sha256Hex(basePath), ex.Message);

        string noHash = WriteOverlay(null, [t]);
        Assert.Contains(TensorOverlay.BaseSha256Key, Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(noHash, basePath)).Message);

        // Explicit opt-out works for both, and the correct hash passes with verification on (uppercase hex accepted).
        using (TensorOverlay.Open(wrongHash, basePath, verifyBaseHash: false)) { }
        using (TensorOverlay.Open(noHash, basePath, verifyBaseHash: false)) { }
        using (TensorOverlay.Open(WriteOverlay(Sha256Hex(basePath).ToUpperInvariant(), [t]), basePath)) { }
    }

    [Fact]
    public void BaseHash_IsCached_AndInvalidatedWhenFileChanges()
    {
        var (basePath, _) = WriteBase(MoeCfg());
        string h1 = TensorOverlay.BaseHash.Compute(basePath);
        Assert.Equal(Sha256Hex(basePath), h1);
        Assert.Single(Directory.GetFiles(_hashCache));
        // Poison the cache entry: a second call must return the cached value (proves the cache is consulted), not re-hash.
        string entry = Directory.GetFiles(_hashCache)[0];
        string fake = new('a', 64);
        File.WriteAllText(entry, fake);
        Assert.Equal(fake, TensorOverlay.BaseHash.Compute(basePath));
        // Changing the file (length/mtime) changes the key => recomputed from content.
        File.AppendAllText(basePath, "x");
        Assert.Equal(Sha256Hex(basePath), TensorOverlay.BaseHash.Compute(basePath));
    }

    [Fact]
    public void OverlayNotConsumedByLoader_IsAnError_NeverSilentlyIgnored()
    {
        var (basePath, _) = WriteBase(MoeCfg());
        string ov = WriteOverlay(Sha256Hex(basePath), [ZeroTern("blk.0.ffn_gate.0.weight", 128, 128)]);
        using var gguf = GgufFile.Open(basePath);
        TensorOverlay.Attach(gguf, basePath, ov);
        // A loader that never saw the overlay (any backend without support) leaves it unapplied.
        var ex = Assert.Throws<NotSupportedException>(() => gguf.Overlay!.VerifyAllApplied());
        Assert.Contains("not applied", ex.Message);
    }

    // ───────────────────────── I8 container (python gguf writer) + hash convention ─────────────────────────

    private string WriteI8Overlay(string baseSha, string name, int inDim, int outDim, byte[] packed, bool declareDtype = true, string dtype = "TERN_B64_F16", int? declaredIn = null,
                                  IEnumerable<OvTensor>? more = null)
    {
        // Shape as the python writer stores it: GGUF dims [in/64*18, out] (reversed [out, in/64*18]), I8 bytes; logical dtype + [out, in] in metadata.
        var w = new GgufWriter();
        w.AddString("general.architecture", "dotllm-overlay");
        w.AddString(TensorOverlay.BaseSha256Key, baseSha);
        if (declareDtype)
        {
            w.AddString(TensorOverlay.DtypeKeyPrefix + name, dtype);
            w.AddInt32Array(TensorOverlay.ShapeKeyPrefix + name, [outDim, declaredIn ?? inDim]);
        }
        w.AddTensor(name, [inDim / 64 * 18, outDim], (uint)QuantizationType.I8, packed);
        foreach (var o in more ?? [])
        {
            w.AddString(TensorOverlay.DtypeKeyPrefix + o.Name, "TERN_B64_F16");
            w.AddInt32Array(TensorOverlay.ShapeKeyPrefix + o.Name, [o.Out, o.In]);
            w.AddTensor(o.Name, [o.In / 64 * 18, o.Out], (uint)QuantizationType.I8, o.Data);
        }
        string path = Path.Combine(_dir, $"i8-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, w.Build());
        return path;
    }

    [Fact]
    public void I8Container_WithMetadata_IsEquivalentToNativeTernType_BitExact()
    {
        var cfg = MoeCfg();
        var (basePath, wts) = WriteBase(cfg);
        int per = cfg.FeedForward * cfg.Hidden;
        // every expert's up projection of layer 1 (so routing certainly hits them), the first one carried in the I8 container's primary slot
        var ts = Enumerable.Range(0, cfg.Experts).Select(e => TernPair($"blk.1.ffn_up.{e}.weight", Slice(wts.Layers[1].UpExps!, e, per), cfg.Hidden, cfg.FeedForward).Tern).ToList();
        var t = ts[0];
        string sha = Sha256Hex(basePath);
        var native = Logits(basePath, WriteOverlay(sha, ts), true);
        var viaI8 = Logits(basePath, WriteI8Overlay(sha, t.Name, t.In, t.Out, t.Data, more: ts.Skip(1)), true);
        Assert.Equal(native, viaI8);
        Assert.NotEqual(Logits(basePath, null), viaI8);   // and the overlay really changed something
    }

    [Fact]
    public void I8Container_ByteCountShapeAndDtypeErrors()
    {
        var cfg = MoeCfg();
        var (basePath, _) = WriteBase(cfg);
        string sha = Sha256Hex(basePath);
        var good = ZeroTern("blk.0.ffn_gate.0.weight", 128, 128);
        // missing metadata => the I8 bytes would be read as a plain [in/64*18, out] I8 tensor: a shape error at apply time, never silently accepted
        Assert.Throws<InvalidDataException>(() => Logits(basePath, WriteI8Overlay(sha, good.Name, 128, 128, good.Data, declareDtype: false)));
        // byte count != out*(in/64)*18 (declared [out,in] larger than the bytes)
        var ex = Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(WriteI8Overlay(sha, good.Name, 128, 128, good.Data, declaredIn: 192), basePath, false));
        Assert.Contains("bytes", ex.Message);
        // unknown dtype string
        Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(WriteI8Overlay(sha, good.Name, 128, 128, good.Data, dtype: "TERN_B65"), basePath, false));
        // invalid trit code inside an I8 container is rejected like a native tensor
        var bad = (byte[])good.Data.Clone(); bad[18 * 3 + 2 + 5] = 0x0C;
        Assert.Contains("0b11", Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(WriteI8Overlay(sha, good.Name, 128, 128, bad), basePath, false)).Message);
    }

    [Fact]
    public void UnspecifiedBaseHash_IsAWarningNotAnError_OtherMismatchStillFails()
    {
        var (basePath, _) = WriteBase(MoeCfg());
        var t = ZeroTern("blk.0.ffn_gate.0.weight", 128, 128);
        using (var ov = TensorOverlay.Open(WriteOverlay("UNSPECIFIED", [t]), basePath))
            Assert.Contains(ov.Notes, n => n.Contains("UNSPECIFIED", StringComparison.Ordinal) && n.Contains("NOT verified", StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => TensorOverlay.Open(WriteOverlay("unspecified-ish", [t]), basePath));
        // base that is not a single GGUF file (e.g. an HF directory): hash skipped with a note
        using var ov2 = TensorOverlay.Open(WriteOverlay(new string('0', 64), [t]), baseModelPath: null);
        Assert.Contains(ov2.Notes, n => n.Contains("cannot be checked", StringComparison.Ordinal));
    }
}
