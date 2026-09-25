using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// dotLLM-side LOAD + FORWARD gate for a trained identity-MoTE BitNet exported by
/// trackM-mote's <c>scripts/lora/mote_export.py</c>. The exported checkpoint is a
/// self-contained HF-style directory (config.json + model.safetensors, bf16 master).
/// The CONFIG already loads with no changes (see <see cref="HfConfigExtractorMoteTests"/>).
/// These four tests cover the WEIGHT-load + BitNet-MoE forward build items — a BitNet-MoE
/// layer differs from dotLLM's SwiGLU MoE in three numerically load-bearing ways: ternary
/// I2_S experts, a relu² gate, and a per-expert FFN Sub-LN.
///
/// Design note: .planning/2026-07-08-mote-dotllm-export-design.md
/// I2_S MoE expert kernel: <c>MatMul.MoeIndexedMatmulI2_S</c> (MoeIndexedMatmulI2STests).
/// </summary>
public sealed class MoteBitNetMoeLoaderScaffoldTests : IDisposable
{
    private const float Eps = 1e-5f;
    private readonly string _scratch;

    public MoteBitNetMoeLoaderScaffoldTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-mote-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ─────────────────────────── shared tiny BitNet-MoE dims ───────────────────────────
    // hidden / intermediate must be multiples of 128 (I2_S block); attention head dims chosen
    // so every projection element count is a multiple of 128.
    private const int Hidden = 128;
    private const int Intermediate = 256;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;   // qDim = 128, kvDim = 64
    private const int Vocab = 32;
    private const int NumExperts = 3; // skip expert 0 + 2 capability experts
    private const int TopK = 1;

    // ════════════════════════════ BUILD ITEM 1 — loader ════════════════════════════
    // LoadBitNetLayer dispatches MoE layers to LoadBitNetMoeLayer, which resolves per-expert
    // I2_S {gate,up,down}_proj into contiguous packed-trit banks + per-expert α, per-expert
    // ffn_sub_norm, and the F32 router gate.weight + gate.bias.
    [Fact]
    public void BitNetMoeLayer_LoadsPerExpertTernaryFfnAndRouter()
    {
        string path = BuildBitNetMoeFixture(numLayers: 2, moeLayers: [1], withGateBias: true);
        using var file = SafetensorsFile.Open(path);
        var config = BuildBitNetMoeConfig(numLayers: 2, mlpOnlyLayers: [0]);

        using var weights = TransformerWeightsSafetensorsLoader.Load(file, config);

        // Layer 0 is force-dense (mlp_only_layers=[0]); layer 1 is MoE.
        var dense = weights.Layers[0];
        Assert.Null(dense.Moe);
        Assert.Equal(QuantizationType.I2_S, dense.GateQuantType); // dense BitNet FFN is ternary

        var moe = weights.Layers[1].Moe;
        Assert.NotNull(moe);
        Assert.True(moe!.IsBitNetI2S);
        Assert.Equal(QuantizationType.I2_S, moe.RoutedExpertQuantType);
        Assert.Equal(NumExperts, moe.NumExperts);
        Assert.Equal(TopK, moe.NumExpertsPerTok);

        // Contiguous packed-trit banks + per-expert scale vectors.
        Assert.NotEqual(0, moe.GateExpsI2SBase);
        Assert.NotEqual(0, moe.UpExpsI2SBase);
        Assert.NotEqual(0, moe.DownExpsI2SBase);
        Assert.Equal((long)Intermediate * Hidden / 4, moe.GateExpsI2SRowBytes);
        Assert.Equal((long)Hidden * Intermediate / 4, moe.DownExpsI2SRowBytes);
        Assert.Equal(NumExperts, moe.GateExpsI2SScales!.Length);
        Assert.Equal(NumExperts, moe.UpExpsI2SScales!.Length);
        Assert.Equal(NumExperts, moe.DownExpsI2SScales!.Length);
        Assert.All(moe.GateExpsI2SScales!, s => Assert.True(s > 0f));

        // Per-expert FFN Sub-LN + router bias.
        Assert.NotNull(moe.ExpertFfnSubNorm);
        Assert.Equal(NumExperts, moe.ExpertFfnSubNorm!.Length);
        Assert.Equal(Intermediate, moe.ExpertFfnSubNorm[0].Length);
        Assert.NotNull(moe.GateBias);
        Assert.Equal(NumExperts, moe.GateBias!.Length);
        Assert.Equal(NumExperts * Hidden, moe.Gate.Length);
    }

    // ════════════════════════════ BUILD ITEM 2 — router bias ════════════════════════════
    // The additive router bias shifts the top-1 argmax expert selection.
    [Fact]
    public unsafe void MoeRoute_AppliesRouterBias()
    {
        const int hidden = 8, numExperts = 4;
        var rng = new Random(13);
        float[] hiddenAct = RandomVec(rng, hidden, 0.5f);
        float[] gate = RandomVec(rng, numExperts * hidden, 0.5f);

        // Unbiased top-1 selection.
        int chosenNoBias = Route1(hiddenAct, gate, hidden, numExperts, ReadOnlySpan<float>.Empty);

        // Bias a DIFFERENT expert up by a large margin; it must now win.
        int target = (chosenNoBias + 1) % numExperts;
        float[] bias = new float[numExperts];
        bias[target] = 100f;
        int chosenBiased = Route1(hiddenAct, gate, hidden, numExperts, bias);

        Assert.NotEqual(chosenNoBias, chosenBiased);
        Assert.Equal(target, chosenBiased);
    }

    // ════════════════════════════ BUILD ITEM 3 — expert forward ════════════════════════════
    // ExecuteBitNetMoe = down( ffn_sub_norm( relu²(gate(x)) * up(x) ) ) over ternary I2_S
    // experts, weighted by the top-1 router. Verified against an independent reference that
    // reuses the proven dense I2_S GEMV + the same Route decision, isolating the NEW composition.
    // Also verifies the identity-MoTE skip expert (all-zero down_proj) outputs exactly 0.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void BitNetMoeExpert_UsesRelu2AndFfnSubNorm(bool zeroSkipExpert)
    {
        const int hidden = 128, intermediate = 256, numExperts = 3, seqLen = 5;
        var rng = new Random(2026);

        // Router (F32) + bias, activations.
        float[] gate = RandomVec(rng, numExperts * hidden, 0.3f);
        float[] gateBias = RandomVec(rng, numExperts, 0.5f);
        float[] hiddenAct = RandomVec(rng, seqLen * hidden, 0.5f);

        // Per-expert ternary weights (payload-only banks) + per-expert absmean α.
        long gateUpRowBytes = (long)intermediate * hidden / 4;
        long downRowBytes = (long)hidden * intermediate / 4;
        byte* gateBank = AllocZeroed(gateUpRowBytes * numExperts);
        byte* upBank = AllocZeroed(gateUpRowBytes * numExperts);
        byte* downBank = AllocZeroed(downRowBytes * numExperts);
        float[] gateScales = new float[numExperts];
        float[] upScales = new float[numExperts];
        float[] downScales = new float[numExperts];
        var ffnSubNorm = new float[numExperts][];

        // Dense (tail-scale) packs of the SAME trits for the reference path.
        byte*[] denseGate = new byte*[numExperts];
        byte*[] denseUp = new byte*[numExperts];
        byte*[] denseDown = new byte*[numExperts];

        try
        {
            for (int e = 0; e < numExperts; e++)
            {
                gateScales[e] = 0.02f + 0.01f * e;
                upScales[e] = 0.03f + 0.01f * e;
                downScales[e] = 0.04f + 0.01f * e;
                ffnSubNorm[e] = RandomVec(rng, intermediate, 0.2f);
                for (int i = 0; i < intermediate; i++) ffnSubNorm[e][i] += 1.0f; // keep norm weight ~1

                sbyte[] gT = RandomTernary(rng, intermediate * hidden);
                sbyte[] uT = RandomTernary(rng, intermediate * hidden);
                sbyte[] dT = (zeroSkipExpert && e == 0)
                    ? new sbyte[hidden * intermediate]      // skip expert: all-zero down_proj
                    : RandomTernary(rng, hidden * intermediate);

                PackPayload(gT, gateBank + e * gateUpRowBytes);
                PackPayload(uT, upBank + e * gateUpRowBytes);
                PackPayload(dT, downBank + e * downRowBytes);
                denseGate[e] = PackDense(gT, gateScales[e]);
                denseUp[e] = PackDense(uT, upScales[e]);
                denseDown[e] = PackDense(dT, downScales[e]);
            }

            float[] actual = new float[seqLen * hidden];
            fixed (float* gatePtr = gate, biasPtr = gateBias, hiddenPtr = hiddenAct, outPtr = actual)
            fixed (float* gsc = gateScales, usc = upScales, dsc = downScales)
            {
                MoeSwiGluMlp.ExecuteBitNetMoe(
                    hidden: new ReadOnlySpan<float>(hiddenPtr, seqLen * hidden),
                    gateWeights: new ReadOnlySpan<float>(gatePtr, numExperts * hidden),
                    gateBias: new ReadOnlySpan<float>(biasPtr, numExperts),
                    gateBank, gateUpRowBytes, new ReadOnlySpan<float>(gsc, numExperts),
                    upBank, gateUpRowBytes, new ReadOnlySpan<float>(usc, numExperts),
                    downBank, downRowBytes, new ReadOnlySpan<float>(dsc, numExperts),
                    ffnSubNorm,
                    output: new Span<float>(outPtr, seqLen * hidden),
                    numExperts, TopK, hidden, intermediate, seqLen,
                    normTopKProb: true, rmsEps: Eps, threadPool: null);
            }

            // ── Reference: same Route decision, dense I2_S GEMV per projection. ──
            float[] expected = ReferenceBitNetMoe(
                hiddenAct, gate, gateBias, seqLen, hidden, intermediate, numExperts,
                denseGate, denseUp, denseDown, ffnSubNorm, out int[] routedExpert);

            for (int i = 0; i < actual.Length; i++)
            {
                float tol = 1e-3f + 1e-3f * MathF.Abs(expected[i]);
                Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                    $"[{i}] expected {expected[i]} got {actual[i]}");
            }

            // Skip-expert: any token routed to expert 0 (zero down_proj) must output exactly 0.
            if (zeroSkipExpert)
            {
                bool sawSkip = false;
                for (int t = 0; t < seqLen; t++)
                {
                    if (routedExpert[t] != 0) continue;
                    sawSkip = true;
                    for (int j = 0; j < hidden; j++)
                        Assert.Equal(0f, actual[t * hidden + j]);
                }
                Assert.True(sawSkip, "expected at least one token routed to the skip expert");
            }
        }
        finally
        {
            NativeMemory.Free(gateBank); NativeMemory.Free(upBank); NativeMemory.Free(downBank);
            for (int e = 0; e < numExperts; e++)
            {
                NativeMemory.Free(denseGate[e]); NativeMemory.Free(denseUp[e]); NativeMemory.Free(denseDown[e]);
            }
        }
    }

    // ════════════════════════════ BUILD ITEM 4 — end-to-end ════════════════════════════
    // Full BitNet-MoE model load + forward: finite vocab logits, deterministic across runs.
    // NOTE: numeric parity vs the *PyTorch* identity-MoTE reference requires a fixture emitted
    // by mote_export.py (config.json + model.safetensors) — that cross-framework gate is the
    // remaining TODO. This asserts the dotLLM path runs end-to-end and is stable.
    [Fact]
    public void ExportedIdentityMote_ForwardMatchesReferenceLogits()
    {
        string path = BuildBitNetMoeFixture(numLayers: 2, moeLayers: [1], withGateBias: true);
        using var file = SafetensorsFile.Open(path);
        var config = BuildBitNetMoeConfig(numLayers: 2, mlpOnlyLayers: [0]);

        using var model = TransformerModel.LoadFromSafetensors(file, config);

        int[] tokens = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using var logits1 = model.Forward(tokens, positions, deviceId: -1);
        Assert.Equal(2, logits1.Shape.Rank);
        Assert.Equal(tokens.Length, logits1.Shape[0]);
        Assert.Equal(Vocab, logits1.Shape[1]);
        AssertAllFinite(logits1);

        using var logits2 = model.Forward(tokens, positions, deviceId: -1);
        AssertTensorsEqual(logits1, logits2);
    }

    // ─────────────────────────── reference + helpers ───────────────────────────

    /// <summary>Top-1 route via the production kernel, then dense I2_S GEMV per projection:
    /// <c>down( ffn_sub_norm( relu²(gate) * up ) )</c>, scaled by the (renormalised) router weight.</summary>
    private static unsafe float[] ReferenceBitNetMoe(
        float[] hiddenAct, float[] gate, float[] gateBias, int seqLen, int hidden, int intermediate,
        int numExperts, byte*[] denseGate, byte*[] denseUp, byte*[] denseDown, float[][] ffnSubNorm,
        out int[] routedExpert)
    {
        int[] assignExpert = new int[seqLen];
        float[] assignWeight = new float[seqLen];
        int[] bc = new int[numExperts + 1];
        int[] bt = new int[seqLen];
        int[] bs = new int[seqLen];
        int[] uniq = new int[seqLen];

        fixed (float* hp = hiddenAct, gp = gate, bp = gateBias)
        {
            MoeSwiGluMlp.Route(
                new ReadOnlySpan<float>(hp, seqLen * hidden),
                new ReadOnlySpan<float>(gp, numExperts * hidden),
                assignExpert, assignWeight, bc, bt, bs, uniq,
                numExperts, TopK, hidden, seqLen, normTopKProb: true,
                new ReadOnlySpan<float>(bp, numExperts));
        }

        routedExpert = assignExpert;
        float[] output = new float[seqLen * hidden];
        float[] g = new float[intermediate];
        float[] u = new float[intermediate];
        float[] inter = new float[intermediate];
        float[] d = new float[hidden];

        fixed (float* hp = hiddenAct)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int e = assignExpert[t];
                float w = assignWeight[t];
                float* x = hp + t * hidden;

                fixed (float* gPtr = g, uPtr = u, iPtr = inter, dPtr = d)
                {
                    MatMul.GemvI2_S(denseGate[e], x, gPtr, intermediate, hidden, threadPool: null);
                    MatMul.GemvI2_S(denseUp[e], x, uPtr, intermediate, hidden, threadPool: null);
                    FusedOps.ReLU2GLU(
                        new ReadOnlySpan<float>(gPtr, intermediate),
                        new ReadOnlySpan<float>(uPtr, intermediate),
                        new Span<float>(iPtr, intermediate));
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(iPtr, intermediate), ffnSubNorm[e], Eps,
                        new Span<float>(iPtr, intermediate));
                    MatMul.GemvI2_S(denseDown[e], iPtr, dPtr, hidden, intermediate, threadPool: null);
                    for (int j = 0; j < hidden; j++)
                        output[t * hidden + j] = w * dPtr[j];
                }
            }
        }
        return output;
    }

    private static unsafe int Route1(float[] hiddenAct, float[] gate, int hidden, int numExperts, ReadOnlySpan<float> bias)
    {
        int[] assignExpert = new int[1];
        float[] assignWeight = new float[1];
        int[] bc = new int[numExperts + 1];
        int[] bt = new int[1];
        int[] bs = new int[1];
        int[] uniq = new int[1];
        fixed (float* hp = hiddenAct, gp = gate)
        {
            MoeSwiGluMlp.Route(
                new ReadOnlySpan<float>(hp, hidden),
                new ReadOnlySpan<float>(gp, numExperts * hidden),
                assignExpert, assignWeight, bc, bt, bs, uniq,
                numExperts, 1, hidden, 1, normTopKProb: true, bias);
        }
        return assignExpert[0];
    }

    // ── I2_S packing (mirrors MoeIndexedMatmulI2STests) ──
    private static unsafe void PackPayload(sbyte[] ternary, byte* dest)
    {
        int n = ternary.Length;
        for (int e = 0; e < n; e++)
        {
            int block = e / 128, j = e % 128, groupIdx = j / 32, groupPos = j % 32;
            dest[block * 32 + groupPos] |= (byte)((ternary[e] + 1) << (6 - 2 * groupIdx));
        }
    }

    private static unsafe byte* PackDense(sbyte[] ternary, float scale)
    {
        int n = ternary.Length;
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)(n / 4 + 4));
        PackPayload(ternary, buf);
        *(float*)(buf + n / 4) = scale;
        return buf;
    }

    private static sbyte[] RandomTernary(Random rng, int n)
    {
        var v = new sbyte[n];
        for (int i = 0; i < n; i++) v[i] = (sbyte)(rng.Next(3) - 1);
        return v;
    }

    private static unsafe byte* AllocZeroed(long bytes) => (byte*)NativeMemory.AllocZeroed((nuint)bytes);

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }

    // ── Full-model BitNet-MoE safetensors fixture ──
    private string BuildBitNetMoeFixture(int numLayers, int[] moeLayers, bool withGateBias)
    {
        var rng = new Random(99);
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [Vocab, Hidden], RandomVec(rng, Vocab * Hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));
        b.AddFloat32("lm_head.weight", [Vocab, Hidden], RandomVec(rng, Vocab * Hidden, 0.05f));

        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight", [qDim, Hidden], RandomVec(rng, qDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight", [kvDim, Hidden], RandomVec(rng, kvDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight", [kvDim, Hidden], RandomVec(rng, kvDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight", [Hidden, qDim], RandomVec(rng, Hidden * qDim, 0.05f));
            b.AddFloat32($"{p}.self_attn.attn_sub_norm.weight", [Hidden], Ones(Hidden));

            if (Array.IndexOf(moeLayers, i) >= 0)
            {
                // BitNet-MoE FFN: router + per-expert ternary {gate,up,down}_proj + ffn_sub_norm.
                b.AddFloat32($"{p}.mlp.gate.weight", [NumExperts, Hidden], RandomVec(rng, NumExperts * Hidden, 0.05f));
                if (withGateBias)
                    b.AddFloat32($"{p}.mlp.gate.bias", [NumExperts], RandomVec(rng, NumExperts, 0.1f));
                for (int e = 0; e < NumExperts; e++)
                {
                    b.AddFloat32($"{p}.mlp.experts.{e}.gate_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                    b.AddFloat32($"{p}.mlp.experts.{e}.up_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                    // Skip expert 0: all-zero down_proj (identity-MoTE path).
                    float[] down = e == 0 ? new float[Hidden * Intermediate] : RandomVec(rng, Hidden * Intermediate, 0.05f);
                    b.AddFloat32($"{p}.mlp.experts.{e}.down_proj.weight", [Hidden, Intermediate], down);
                    b.AddFloat32($"{p}.mlp.experts.{e}.ffn_sub_norm.weight", [Intermediate], Ones(Intermediate));
                }
            }
            else
            {
                // Dense BitNet FFN.
                b.AddFloat32($"{p}.mlp.gate_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.up_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.down_proj.weight", [Hidden, Intermediate], RandomVec(rng, Hidden * Intermediate, 0.05f));
                b.AddFloat32($"{p}.mlp.ffn_sub_norm.weight", [Intermediate], Ones(Intermediate));
            }
        }

        string path = Path.Combine(_scratch, "bitnet-mote.safetensors");
        b.WriteTo(path);
        return path;
    }

    private static ModelConfig BuildBitNetMoeConfig(int numLayers, int[] mlpOnlyLayers)
        => new()
        {
            Architecture = Architecture.BitNet,
            ActivationFunction = ActivationFunction.ReluSquared,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Intermediate,
            NumLayers = numLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 128,
            NormEpsilon = Eps,
            TiedEmbeddings = false,
            RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX),
            Moe = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = TopK,
                MoeIntermediateSize = Intermediate,
                NormTopKProb = true,
                DecoderSparseStep = 1,
                MlpOnlyLayers = mlpOnlyLayers,
            },
        };

    private static unsafe void AssertAllFinite(ITensor logits)
    {
        int n = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) n *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        for (int i = 0; i < span.Length; i++)
            Assert.True(float.IsFinite(span[i]), $"logit[{i}] = {span[i]} is not finite");
    }

    private static unsafe void AssertTensorsEqual(ITensor a, ITensor b)
    {
        int na = 1, nb = 1;
        for (int i = 0; i < a.Shape.Rank; i++) na *= a.Shape[i];
        for (int i = 0; i < b.Shape.Rank; i++) nb *= b.Shape[i];
        Assert.Equal(na, nb);
        var sa = new ReadOnlySpan<float>((void*)a.DataPointer, na);
        var sb = new ReadOnlySpan<float>((void*)b.DataPointer, nb);
        for (int i = 0; i < sa.Length; i++)
            Assert.Equal(sa[i], sb[i]);
    }
}
