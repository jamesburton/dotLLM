using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using System.Numerics.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Lora;

/// <summary>
/// CPU-only error-budget companion to <see cref="LoraStackCudaParityTests"/> (#488).
/// <para>
/// The CUDA stack-parity test gates <c>cosine(cpu_stack, gpu_stack) &gt; 0.999</c> and measures
/// 0.998909 on an RTX 3060. This class answers — with no GPU — whether that deficit can be a
/// composition defect at all, by measuring on CPU (a) the cosine signature each plausible
/// composition defect would leave, and (b) how far the FP16 staging round-trip alone moves the
/// logits. Both are pure-CPU forwards of the same SmolLM-135M Q8_0 model with the same two
/// synthetic rank-8 adapters, so the numbers are directly comparable to the CUDA test's.
/// </para>
/// <para>
/// Measured on this machine (CPU forward is bit-deterministic run to run — see
/// <see cref="Repeat_Forward_Determinism_Control"/>):
/// <list type="bullet">
/// <item><c>||delta|| / ||logits|| = 1.48e-2</c> — the whole stacked-LoRA perturbation. A
/// maximally wrong GPU delta could therefore cost at most <c>(2·1.48e-2)²/2 = 4.4e-4</c> of
/// whole-logit cosine, which is 2.5× SMALLER than the 1.09e-3 deficit in #488.</item>
/// <item>Defect signatures in whole-logit cosine: adapter dropped 1.6e-4, weights halved 2.4e-4,
/// alpha/r omitted 2.4e-4, rank blocks mis-ordered 4.1e-4, FP16 staging round-trip 1.0e-4.</item>
/// <item>Delta-space (<c>logits − base</c>) cosine separates the mutants loudly (0.23 / −0.11 /
/// −0.38) but is NOT usable as a gate of any kind: an FP16 staging round-trip alone rotates the
/// delta to cosine 0.39 (single) / −0.06 (stack), and a 1e-4 relative weight nudge already moves
/// the logits by 0.42·||delta|| with a non-monotone response (1e-3 → 1.47·||delta||, 1e-2 →
/// 0.83·||delta||). That discreteness is the CPU reference's own Q8_0 <i>activation</i>
/// quantization (see <c>MatMulVnni</c> / <c>MatMul.Q8_0Sse</c>, which quantize the activation to
/// int8 with a per-32-block scale): a 1.48% perturbation sits at the Q8 activation-quant noise
/// floor, so no whole-logit metric can see the delta's <i>direction</i>.</item>
/// </list>
/// </para>
/// <para>
/// Scope, stated plainly: the bound above covers composition defects that preserve or shrink the
/// delta magnitude. A magnitude-<i>inflating</i> defect (e.g. a double-applied scale or a wrong
/// leading dimension) is unbounded and cannot be excluded from CPU — that class is what
/// <c>LoraStackCudaParityTests</c> GATE D now catches. The four composition mutants below are
/// documented signatures, not a discriminating gate; the discriminating coverage for composition
/// maths is <c>LoraComposerTests</c> (F32, elementwise, 1e-4). Making GATE D discriminating for
/// them too would need <c>SyntheticLoraAdapterFactory</c>'s ±0.01 weights enlarged (±0.03 gives a
/// ~13%-of-norm delta, clear of the Q8 floor) — that touches every sibling parity test and needs
/// a CUDA run, so it is a follow-up, not part of #488.
/// </para>
/// </summary>
public sealed unsafe class LoraStackCpuErrorBudgetTests
{
    private readonly ITestOutputHelper _output;

    public LoraStackCpuErrorBudgetTests(ITestOutputHelper output) => _output = output;

    private const string ModelPath =
        @"C:\Users\james\.dotllm\models\QuantFactory\SmolLM-135M-GGUF\SmolLM-135M.Q8_0.gguf";

    /// <summary>
    /// Establishes the defect-signature floor: every plausible composition defect (dropped
    /// adapter, missing alpha/r scale, mis-ordered rank blocks, halved weight) perturbs the
    /// logits by orders of magnitude more than the 1.1e-3 cosine deficit seen on CUDA, so
    /// 0.998909 cannot be any of them. Also pins the composition plumbing at model scale.
    /// </summary>
    [SkippableFact]
    public void Composition_Defect_Signatures_Are_Orders_Of_Magnitude_Larger_Than_Cuda_Deficit()
    {
        Skip.If(!File.Exists(ModelPath), $"SmolLM-135M Q8_0 GGUF not found at {ModelPath}.");

        using var gguf = GgufFile.Open(ModelPath);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        using var a1 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);
        using var a2 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 99);
        using var composite = LoraComposer.Compose([(a1, 1f), (a2, 1f)], cfg);

        using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));

        float[] refStack = Logits(cpu, tok, pos, composite, cfg.VocabSize);
        float[] baseVec = Logits(cpu, tok, pos, null, cfg.VocabSize);
        float[] single = Logits(cpu, tok, pos, a1, cfg.VocabSize);

        // Reference perturbations: how far the adapters move the model at all.
        double cosBaseSingle = Cosine(baseVec, single);
        double cosBaseStack = Cosine(baseVec, refStack);

        // Plumbing: a one-element stack must reproduce the adapter exactly.
        using var solo = LoraComposer.Compose([(a1, 1f)], cfg);
        double cosSoloA1 = Cosine(Logits(cpu, tok, pos, solo, cfg.VocabSize), single);

        // Defect signatures.
        using var dropped = LoraComposer.Compose([(a1, 1f)], cfg);                 // adapter2 dropped
        using var halved = LoraComposer.Compose([(a1, 0.5f), (a2, 0.5f)], cfg);    // wrong stack weights
        using var noScale = ComposeWithoutAlphaOverRank(a1, a2, cfg);              // alpha/r omitted
        using var misordered = ComposeWithMisorderedABlocks(a1, a2, cfg);          // A cols vs B rows swapped

        float[] lDropped = Logits(cpu, tok, pos, dropped, cfg.VocabSize);
        float[] lHalved = Logits(cpu, tok, pos, halved, cfg.VocabSize);
        float[] lNoScale = Logits(cpu, tok, pos, noScale, cfg.VocabSize);
        float[] lMisordered = Logits(cpu, tok, pos, misordered, cfg.VocabSize);

        // Delta space: subtract the adapter-free logits so the comparison sees only the LoRA
        // contribution. This is where a composition defect is visible; in whole-logit space the
        // delta is a ~1.3% perturbation and every defect hides under the base-model noise.
        double dDropped = DeltaCosine(baseVec, refStack, lDropped);
        double dHalved = DeltaCosine(baseVec, refStack, lHalved);
        double dNoScale = DeltaCosine(baseVec, refStack, lNoScale);
        double dMisordered = DeltaCosine(baseVec, refStack, lMisordered);

        _output.WriteLine($"cosine(base, single)          = {cosBaseSingle:F6}  (adapter perturbation)");
        _output.WriteLine($"cosine(base, stack)           = {cosBaseStack:F6}  (adapter perturbation)");
        _output.WriteLine($"cosine(Compose([a1]), a1 direct) = {cosSoloA1:F8}");
        _output.WriteLine("-- whole-logit signatures vs the correct stack (what GATE D measures) --");
        _output.WriteLine($"cosine(stack, adapter2 dropped)     = {Cosine(refStack, lDropped):F6}");
        _output.WriteLine($"cosine(stack, weights 0.5/0.5)      = {Cosine(refStack, lHalved):F6}");
        _output.WriteLine($"cosine(stack, alpha/r omitted)      = {Cosine(refStack, lNoScale):F6}");
        _output.WriteLine($"cosine(stack, A blocks misordered)  = {Cosine(refStack, lMisordered):F6}");
        _output.WriteLine("-- delta-space signatures (logits - base) --");
        _output.WriteLine($"deltaCosine(adapter2 dropped)       = {dDropped:F6}");
        _output.WriteLine($"deltaCosine(weights 0.5/0.5)        = {dHalved:F6}");
        _output.WriteLine($"deltaCosine(alpha/r omitted)        = {dNoScale:F6}");
        _output.WriteLine($"deltaCosine(A blocks misordered)    = {dMisordered:F6}");

        // Plumbing must be exact (F32 both sides).
        Assert.True(cosSoloA1 > 0.9999999,
            $"Compose([a1]) diverges from a1 at model scale: cosine={cosSoloA1:F9}.");

        // (1) The whole adapter perturbation is ~1.3% of the logit norm, so even a MAXIMALLY wrong
        //     GPU delta (sign-reversed) could only cost 1 - cos ≈ (2·1.33e-2)²/2 ≈ 3.6e-4 of
        //     whole-logit cosine. The CUDA deficit under investigation is 1.09e-3 — 3x larger than
        //     the entire LoRA path can produce. Pin that bound so the argument cannot silently rot.
        double maxLoraAttributableDeficit = 2.0 * (1.0 - cosBaseStack) * 2.0;
        Assert.True(maxLoraAttributableDeficit < (1.0 - 0.998909),
            $"The LoRA delta perturbation grew: a fully-reversed delta could now cost " +
            $"{maxLoraAttributableDeficit:E2} of whole-logit cosine, which reaches the 1.09e-3 CUDA " +
            "deficit. #488's 'the deficit cannot be the LoRA path' argument no longer holds.");

        // (2) In delta space every defect is loud. This is NOT a usable GPU gate (FP16 staging
        //     alone rotates the delta as far — see the class remarks); it pins the signatures so
        //     the numbers quoted in #488 stay honest as the model/adapters evolve.
        AssertDeltaSignature(dDropped, "adapter2 dropped");
        AssertDeltaSignature(dHalved, "weights halved");
        AssertDeltaSignature(dNoScale, "alpha/r omitted");
        AssertDeltaSignature(dMisordered, "A blocks misordered");

        static void AssertDeltaSignature(double cos, string what)
            => Assert.True(cos < 0.99,
                $"Defect '{what}' leaves delta-cosine {cos:F6} >= 0.99 — the delta-space gate in " +
                "LoraStackCudaParityTests would not catch it.");
    }

    /// <summary>
    /// Lower bound on the CUDA deficit from LoRA FP16 staging alone: re-runs the CPU forward with
    /// every A/B factor round-tripped through <see cref="Half"/>, exactly as
    /// <c>CudaLoraWeights.Stage</c> does, for the single adapter and for the composed stack.
    /// Reports the stack/single amplification factor. Diagnostic only — no threshold is asserted,
    /// because this omits the FP16 activations, FP16 <c>LoraTmp</c>, FP16 accumulate and FP16
    /// base model that the GPU also carries.
    /// </summary>
    [SkippableFact]
    public void Fp16_Staging_Roundtrip_Error_Floor_Single_Vs_Stack()
    {
        Skip.If(!File.Exists(ModelPath), $"SmolLM-135M Q8_0 GGUF not found at {ModelPath}.");

        using var gguf = GgufFile.Open(ModelPath);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        using var a1 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);
        using var a2 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 99);
        using var composite = LoraComposer.Compose([(a1, 1f), (a2, 1f)], cfg);

        using var a1Rounded = RoundTripF16(a1, cfg);
        using var compositeRounded = RoundTripF16(composite, cfg);

        using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));

        float[] baseVec = Logits(cpu, tok, pos, null, cfg.VocabSize);
        float[] lSingle = Logits(cpu, tok, pos, a1, cfg.VocabSize);
        float[] lSingleRt = Logits(cpu, tok, pos, a1Rounded, cfg.VocabSize);
        float[] lStack = Logits(cpu, tok, pos, composite, cfg.VocabSize);
        float[] lStackRt = Logits(cpu, tok, pos, compositeRounded, cfg.VocabSize);

        double cosSingle = Cosine(lSingle, lSingleRt);
        double cosStack = Cosine(lStack, lStackRt);
        double dSingle = DeltaCosine(baseVec, lSingle, lSingleRt);
        double dStack = DeltaCosine(baseVec, lStack, lStackRt);

        _output.WriteLine($"cosine(single f32, single f16-staged) = {cosSingle:F8}  deficit {1 - cosSingle:E3}");
        _output.WriteLine($"cosine(stack  f32, stack  f16-staged) = {cosStack:F8}  deficit {1 - cosStack:E3}");
        _output.WriteLine($"stack/single staging-deficit ratio     = {(1 - cosStack) / Math.Max(1 - cosSingle, 1e-12):F2}x");
        _output.WriteLine($"deltaCosine(single f32 vs f16-staged) = {dSingle:F8}  deficit {1 - dSingle:E3}");
        _output.WriteLine($"deltaCosine(stack  f32 vs f16-staged) = {dStack:F8}  deficit {1 - dStack:E3}");
    }

    /// <summary>
    /// Control: how much does the CPU forward move run-to-run with no change at all? Any
    /// cosine/delta-cosine measured above is only meaningful relative to this floor.
    /// </summary>
    [SkippableFact]
    public void Repeat_Forward_Determinism_Control()
    {
        Skip.If(!File.Exists(ModelPath), $"SmolLM-135M Q8_0 GGUF not found at {ModelPath}.");

        using var gguf = GgufFile.Open(ModelPath);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        using var a1 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);
        using var a2 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 99);
        using var composite = LoraComposer.Compose([(a1, 1f), (a2, 1f)], cfg);
        using var cpu = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));

        float[] b1 = Logits(cpu, tok, pos, null, cfg.VocabSize);
        float[] b2 = Logits(cpu, tok, pos, null, cfg.VocabSize);
        float[] s1 = Logits(cpu, tok, pos, composite, cfg.VocabSize);
        float[] s2 = Logits(cpu, tok, pos, composite, cfg.VocabSize);

        double cosBase = Cosine(b1, b2);
        double cosStack = Cosine(s1, s2);
        double dRepeat = DeltaCosine(b1, s1, s2);

        _output.WriteLine($"cosine(base run1, base run2)   = {cosBase:F9}");
        _output.WriteLine($"cosine(stack run1, stack run2) = {cosStack:F9}");
        _output.WriteLine($"deltaCosine(stack run1, run2)  = {dRepeat:F9}");
        _output.WriteLine($"max |base1-base2| = {MaxAbsDiff(b1, b2):E3}   max |stack1-stack2| = {MaxAbsDiff(s1, s2):E3}");
        _output.WriteLine($"||delta||/||logits|| = {Norm(Sub(s1, b1)) / Norm(s1):E3}");

        // Sensitivity probe: how does the logit response scale with a tiny, smooth change to the
        // stack weights? Tells us whether the end-to-end LoRA delta is a smooth function of the
        // adapter weights (delta-space gating viable) or chaotic (it is not).
        foreach (float eps in (float[])[1e-4f, 1e-3f, 1e-2f])
        {
            using var perturbed = LoraComposer.Compose([(a1, 1f), (a2, 1f + eps)], cfg);
            float[] sp = Logits(cpu, tok, pos, perturbed, cfg.VocabSize);
            _output.WriteLine(
                $"eps={eps:E0}: cosine(stack, perturbed)={Cosine(s1, sp):F9}  " +
                $"deltaCosine={DeltaCosine(b1, s1, sp):F6}  " +
                $"||Δlogits||/||delta||={Norm(Sub(sp, s1)) / Norm(Sub(s1, b1)):E3}");
        }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static float[] Sub(float[] a, float[] b)
    {
        float[] r = new float[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] - b[i];
        return r;
    }

    private static double Norm(float[] a)
    {
        double s = 0;
        foreach (float v in a) s += (double)v * v;
        return Math.Sqrt(s);
    }

    // ── Mutant composers (deliberately wrong; used only as signatures) ─────────────

    /// <summary>Composes without the per-adapter <c>alpha/rank</c> scale (scale = weight only).</summary>
    private static LoraAdapter ComposeWithoutAlphaOverRank(ILoraAdapter a1, ILoraAdapter a2, ModelConfig cfg)
        => ComposeMutant(a1, a2, cfg, scaleOverride: (_, w) => w, misorderA: false);

    /// <summary>Composes with the A column blocks in the opposite order to the B row blocks.</summary>
    private static LoraAdapter ComposeWithMisorderedABlocks(ILoraAdapter a1, ILoraAdapter a2, ModelConfig cfg)
        => ComposeMutant(a1, a2, cfg, scaleOverride: null, misorderA: true);

    private static LoraAdapter ComposeMutant(
        ILoraAdapter a1, ILoraAdapter a2, ModelConfig cfg,
        Func<ILoraAdapter, float, float>? scaleOverride, bool misorderA)
    {
        var stack = new (ILoraAdapter a, float w)[] { (a1, 1f), (a2, 1f) };
        int rSum = a1.Rank + a2.Rank;
        var composite = new LoraAdapter("mutant", rSum, rSum, ["q_proj", "v_proj"]);

        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            foreach (string proj in (string[])["q_proj", "v_proj"])
            {
                if (a1.GetLayerWeights(layer, proj) is null) continue;
                int inputDim = a1.GetLayerWeights(layer, proj)!.Value.InputDim;
                int outputDim = a1.GetLayerWeights(layer, proj)!.Value.OutputDim;

                nint bConcat = LoraAdapter.AllocAligned((long)rSum * inputDim);
                nint aConcat = LoraAdapter.AllocAligned((long)outputDim * rSum);
                float* bDst = (float*)bConcat;
                float* aDst = (float*)aConcat;

                int rowOffset = 0;
                for (int i = 0; i < stack.Length; i++)
                {
                    var (a, weight) = stack[i];
                    var w = a.GetLayerWeights(layer, proj)!.Value;
                    int rank = a.Rank;
                    float scale = scaleOverride is null ? weight * (a.Alpha / a.Rank) : scaleOverride(a, weight);

                    Buffer.MemoryCopy((void*)w.BHandle, bDst + (long)rowOffset * inputDim,
                        (long)rank * inputDim * sizeof(float), (long)rank * inputDim * sizeof(float));

                    // A column offset: correct = rowOffset; misordered = the OTHER adapter's slot.
                    int colOffset = misorderA ? (rSum - rowOffset - rank) : rowOffset;
                    float* aSrc = (float*)w.AHandle;
                    for (int o = 0; o < outputDim; o++)
                    {
                        float* src = aSrc + (long)o * rank;
                        float* dst = aDst + (long)o * rSum + colOffset;
                        for (int r = 0; r < rank; r++) dst[r] = src[r] * scale;
                    }
                    rowOffset += rank;
                }

                composite.AddLayerWeights(layer, proj,
                    new LoraLayerWeights(aConcat, bConcat, inputDim, outputDim));
            }
        }
        return composite;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    /// <summary>Copies an F32 adapter with every A/B element round-tripped through FP16.</summary>
    private static LoraAdapter RoundTripF16(ILoraAdapter src, ModelConfig cfg)
    {
        var dst = new LoraAdapter(src.Name + "-f16rt", src.Rank, src.Alpha, src.TargetModules.ToArray());
        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            foreach (string proj in src.TargetModules)
            {
                if (src.GetLayerWeights(layer, proj) is not { } w) continue;
                long bN = (long)src.Rank * w.InputDim;
                long aN = (long)w.OutputDim * src.Rank;
                nint bH = LoraAdapter.AllocAligned(bN);
                nint aH = LoraAdapter.AllocAligned(aN);
                Round((float*)w.BHandle, (float*)bH, bN);
                Round((float*)w.AHandle, (float*)aH, aN);
                dst.AddLayerWeights(layer, proj, new LoraLayerWeights(aH, bH, w.InputDim, w.OutputDim));
            }
        }
        return dst;

        static void Round(float* src, float* dst, long n)
        {
            for (long i = 0; i < n; i++) dst[i] = (float)(Half)src[i];
        }
    }

    private static float[] Logits(TransformerModel model, int[] tok, int[] pos,
                                  ILoraAdapter? adapter, int vocabSize)
    {
        using ITensor logits = model.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: adapter);
        long lastRow = (long)(tok.Length - 1) * vocabSize;
        float[] vec = new float[vocabSize];
        new ReadOnlySpan<float>((float*)logits.DataPointer + lastRow, vocabSize).CopyTo(vec);
        return vec;
    }

    /// <summary>Cosine between two LoRA deltas, each taken relative to the adapter-free logits.</summary>
    private static double DeltaCosine(float[] baseVec, float[] a, float[] b)
    {
        float[] da = new float[a.Length];
        float[] db = new float[b.Length];
        for (int i = 0; i < a.Length; i++) { da[i] = a[i] - baseVec[i]; db[i] = b[i] - baseVec[i]; }
        return Cosine(da, db);
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }
}
