using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>Post-recurrence output-gate activation of the Gated-DeltaNet token mixer: <c>norm(core) * act(z)</c>.</summary>
internal enum GdnOutputGate
{
    /// <summary><c>z * sigmoid(z)</c>: Qwen3.5 / Qwen3.6 (llama.cpp <c>qwen35moe.cpp</c>).</summary>
    SiLu,

    /// <summary><c>sigmoid(z)</c>: Qwen4-Exp (llama.cpp <c>qwen4exp.cpp build_norm_gated</c>, HF <c>output_gate_type="sigmoid"</c>).</summary>
    Sigmoid,
}

/// <summary>Which of the mixer's five dense projections a <see cref="GdnGemm"/> call computes (LoRA sites, #845).</summary>
internal enum GdnProjection
{
    /// <summary><c>attn_qkv</c> (HF <c>in_proj_qkv</c>).</summary>
    Qkv,

    /// <summary><c>attn_gate</c> (HF <c>in_proj_z</c>).</summary>
    Gate,

    /// <summary><c>ssm_alpha</c> (HF <c>in_proj_a</c>).</summary>
    Alpha,

    /// <summary><c>ssm_beta</c> (HF <c>in_proj_b</c>).</summary>
    Beta,

    /// <summary><c>ssm_out</c> (HF <c>out_proj</c>).</summary>
    Out,
}

/// <summary>A model's own GEMM dispatch (<c>output[T, outDim] = input[T, inDim] x weight^T</c>) for one projection.</summary>
internal delegate void GdnGemm(GdnProjection projection, nint weight, QuantizationType quantType, ReadOnlySpan<float> input, Span<float> output,
                               int outDim, int inDim, int seqLen);

/// <summary>Caller-owned scratch of one mixer call (all fully overwritten before being read).</summary>
/// <param name="Qkv">[T, convDim]</param><param name="Z">[T, vDim]</param><param name="Alpha">[T, nV]</param><param name="Beta">[T, nV]</param>
/// <param name="ConvIn">[dConv-1+T, convDim]</param><param name="Q">[T, kDim]</param><param name="K">[T, kDim]</param>
/// <param name="V">[T, vDim]</param><param name="Core">[T, vDim]</param>
internal readonly ref struct GdnMixerScratch(Span<float> Qkv, Span<float> Z, Span<float> Alpha, Span<float> Beta, Span<float> ConvIn,
                                             Span<float> Q, Span<float> K, Span<float> V, Span<float> Core)
{
    public readonly Span<float> Qkv = Qkv, Z = Z, Alpha = Alpha, Beta = Beta, ConvIn = ConvIn, Q = Q, K = K, V = V, Core = Core;
}

/// <summary>
/// Optional per-row recording for speculative verification (Qwen4-Exp, #842): the conv windows after each of the first
/// <see cref="Rows"/> rows plus the L2-normalised keys, decays and scan deltas that <see cref="GatedDeltaNetScan.Replay"/> rebuilds the
/// recurrent state from. Spans are this layer's regions.
/// </summary>
internal readonly ref struct GdnRowRecording(int rows, Span<float> conv, Span<float> keys, Span<float> decays, Span<float> deltas)
{
    public readonly int Rows = rows;
    public readonly Span<float> Conv = conv, Keys = keys, Decays = decays, Deltas = deltas;
}

/// <summary>
/// The Gated-DeltaNet token mixer shared by Qwen3.5 / Qwen3.6 (<see cref="Qwen3MoeHybridTransformerModel"/>) and Qwen4-Exp
/// (<see cref="Qwen4ExpTransformerModel"/>): projections, decay / write gate, causal conv, L2-normalised Q/K, the delta-rule scan,
/// per-head RMSNorm and the output gate, then <c>ssm_out</c>. The ONLY numerical difference between the two models is the output-gate
/// activation (<see cref="GdnOutputGate"/>), and each gate keeps the exact arithmetic of the implementation it replaced (silu is the
/// scalar per-element form, sigmoid the vectorised <see cref="TensorPrimitives"/> pass), so logits are bit-identical to before.
/// </summary>
internal static unsafe class GdnTokenMixer
{
    /// <summary>Runs one GDN layer over <paramref name="seqLen"/> tokens and advances <paramref name="cache"/> in place.</summary>
    /// <param name="w">Layer weights.</param>
    /// <param name="gdn">GDN geometry.</param>
    /// <param name="absoluteLayer">Block index (tensor-dump labels only).</param>
    /// <param name="ordinal">GDN-layer ordinal inside <paramref name="cache"/>.</param>
    /// <param name="seqLen">Tokens.</param>
    /// <param name="hiddenSize">Model width (dump labels only).</param>
    /// <param name="eps">Norm epsilon.</param>
    /// <param name="gate">Output-gate activation.</param>
    /// <param name="x">Pre-normed input <c>[T, hidden]</c>.</param>
    /// <param name="y">Output <c>[T, hidden]</c> (may alias <paramref name="x"/>: it is not read after the output projection starts).</param>
    /// <param name="gemm">The model's GEMM dispatch.</param>
    /// <param name="s">Scratch.</param>
    /// <param name="cache">Recurrent state.</param>
    /// <param name="rec">Optional per-row recording (rows &gt; 0 to record).</param>
    [SkipLocalsInit]
    public static void Forward(GdnTokenMixingWeights w, in GatedDeltaNetConfig gdn, int absoluteLayer, int ordinal, int seqLen,
                               int hiddenSize, float eps, GdnOutputGate gate, ReadOnlySpan<float> x, Span<float> y, GdnGemm gemm,
                               in GdnMixerScratch s, GdnStateCache cache, in GdnRowRecording rec = default)
    {
        int nV = gdn.NVHead, nK = gdn.NKHead, dS = gdn.DState, dC = gdn.DConv;
        int convDim = (2 * nK + nV) * dS, vDim = nV * dS, kDim = nK * dS, T = seqLen;
        var qkv = s.Qkv; var z = s.Z; var alpha = s.Alpha; var beta = s.Beta; var convIn = s.ConvIn;
        var q = s.Q; var k = s.K; var v = s.V; var core = s.Core;

        // ── 1. projections from the normed input ──
        gemm(GdnProjection.Qkv, w.QkvWeight, w.QkvQuantType, x, qkv, w.QkvOutputDim, w.QkvInputDim, T);
        gemm(GdnProjection.Gate, w.GateWeight, w.GateQuantType, x, z, w.GateOutputDim, w.GateInputDim, T);
        gemm(GdnProjection.Alpha, w.AlphaWeight, w.AlphaQuantType, x, alpha, w.AlphaOutputDim, w.AlphaInputDim, T);
        gemm(GdnProjection.Beta, w.BetaWeight, w.BetaQuantType, x, beta, w.BetaOutputDim, w.BetaInputDim, T);
        if (TensorDump.Enabled)
        {
            Dump2D($"blk.{absoluteLayer}.linear_attn_qkv_mixed", qkv, T, convDim);
            Dump2D($"blk.{absoluteLayer}.z", z, T, vDim);
            Dump2D($"blk.{absoluteLayer}.alpha_proj", alpha, T, nV);
            Dump2D($"blk.{absoluteLayer}.beta_proj", beta, T, nV);
        }

        // ── 2. g = exp(softplus(alpha + dt_bias) * A); beta = sigmoid(beta) ──
        for (int t = 0; t < T; t++)
            for (int vh = 0; vh < nV; vh++)
            {
                float a = alpha[t * nV + vh] + w.DtBias[vh];
                float sp = MathF.Log(1f + MathF.Exp(a));   // softplus
                alpha[t * nV + vh] = MathF.Exp(sp * w.A[vh]);
            }
        TensorPrimitives.Sigmoid(beta.Slice(0, T * nV), beta.Slice(0, T * nV));
        if (TensorDump.Enabled)
        {
            Dump2D($"blk.{absoluteLayer}.g", alpha, T, nV);
            Dump2D($"blk.{absoluteLayer}.beta_sigmoid", beta, T, nV);
        }

        // ── 3. causal conv over [conv_state | qkv], SiLU ──
        // A layer that is still lazily reading a checkpoint (BeginUpdate) takes its input from the source cache here and in the scan's
        // first token and writes the result into its own buffers: the state copy is fused into work done anyway (#840).
        cache.BeginUpdate(ordinal, out var srcConv, out var srcGdn);
        var convState = cache.GetConvStateForUpdate(ordinal);
        (srcConv.IsEmpty ? convState : srcConv).Slice(0, (dC - 1) * convDim).CopyTo(convIn.Slice(0, (dC - 1) * convDim));
        qkv.Slice(0, T * convDim).CopyTo(convIn.Slice((dC - 1) * convDim));
        Conv1dCausal.Execute(convIn.Slice(0, (dC - 1 + T) * convDim), w.Conv1dWeight, w.Conv1dBias, qkv.Slice(0, T * convDim), dC, convDim, T);
        SiLu.Execute(qkv.Slice(0, T * convDim), qkv.Slice(0, T * convDim));
        if (TensorDump.Enabled) Dump2D($"blk.{absoluteLayer}.conv_output_silu", qkv, T, convDim);
        for (int r = 0; r < dC - 1; r++)
            convIn.Slice((T + r) * convDim, convDim).CopyTo(convState.Slice(r * convDim, convDim));
        // Row recording: after row t the rolling window is convIn rows t+1 .. t+dC-1.
        int recRows = rec.Rows > 0 ? Math.Min(rec.Rows, T - 1) : 0;
        for (int t = 0; t < recRows; t++)
            convIn.Slice((t + 1) * convDim, (dC - 1) * convDim).CopyTo(rec.Conv.Slice(t * (dC - 1) * convDim, (dC - 1) * convDim));

        // ── 4. de-interleave [Q | K | V], L2-normalise Q and K ──
        for (int t = 0; t < T; t++)
        {
            qkv.Slice(t * convDim, kDim).CopyTo(q.Slice(t * kDim, kDim));
            qkv.Slice(t * convDim + kDim, kDim).CopyTo(k.Slice(t * kDim, kDim));
            qkv.Slice(t * convDim + 2 * kDim, vDim).CopyTo(v.Slice(t * vDim, vDim));
        }
        if (TensorDump.Enabled)
        {
            Dump3D($"blk.{absoluteLayer}.q_conv", q, T, nK, dS);
            Dump3D($"blk.{absoluteLayer}.k_conv", k, T, nK, dS);
            Dump3D($"blk.{absoluteLayer}.v_conv", v, T, nV, dS);
        }
        GatedDeltaNetScan.L2NormalizeHeads(q.Slice(0, T * kDim), dS);
        GatedDeltaNetScan.L2NormalizeHeads(k.Slice(0, T * kDim), dS);
        if (TensorDump.Enabled)
        {
            Dump3D($"blk.{absoluteLayer}.q_conv_predelta", q, T, nK, dS);
            Dump3D($"blk.{absoluteLayer}.k_conv_predelta", k, T, nK, dS);
        }
        if (recRows > 0)
        {
            // what GatedDeltaNetScan.Replay needs to rebuild the state after any of the first recRows tokens (the scan records the deltas)
            k.Slice(0, recRows * kDim).CopyTo(rec.Keys);
            alpha.Slice(0, recRows * nV).CopyTo(rec.Decays);
        }

        // ── 5. delta-rule scan ──
        GatedDeltaNetScan.Execute(cache.GetGdnStateForUpdate(ordinal), q.Slice(0, T * kDim), k.Slice(0, T * kDim), v.Slice(0, T * vDim),
                                  alpha.Slice(0, T * nV), beta.Slice(0, T * nV), core.Slice(0, T * vDim), nV, nK, dS, T,
                                  snapshotRows: recRows, stateSource: srcGdn, deltaRecord: recRows > 0 ? rec.Deltas : default);
        if (TensorDump.Enabled) Dump3D($"blk.{absoluteLayer}.attn_output", core, T, nV, dS);

        // ── 6. per-head RMSNorm(core, ssm_norm) * act(z) ──
        if (gate == GdnOutputGate.SiLu)
        {
            for (int t = 0; t < T; t++)
            {
                int tBase = t * vDim;
                for (int vh = 0; vh < nV; vh++)
                {
                    int headOff = tBase + vh * dS;
                    RmsNorm.Execute(core.Slice(headOff, dS), w.SsmNormWeight, eps, core.Slice(headOff, dS));
                    var zHead = z.Slice(headOff, dS);
                    var outHead = core.Slice(headOff, dS);
                    for (int i = 0; i < dS; i++)
                    {
                        float zi = zHead[i];
                        outHead[i] *= zi * (1f / (1f + MathF.Exp(-zi)));   // silu(z) = z * sigmoid(z)
                    }
                }
            }
        }
        else
        {
            for (int t = 0; t < T; t++)
                for (int vh = 0; vh < nV; vh++)
                {
                    int off = t * vDim + vh * dS;
                    RmsNorm.Execute(core.Slice(off, dS), w.SsmNormWeight, eps, core.Slice(off, dS));
                }
            var zs = z.Slice(0, T * vDim);
            TensorPrimitives.Sigmoid(zs, zs);
            TensorPrimitives.Multiply(core.Slice(0, T * vDim), zs, core.Slice(0, T * vDim));
        }
        if (TensorDump.Enabled) Dump3D($"blk.{absoluteLayer}.final_output", core, T, nV, dS);

        // ── 7. ssm_out ──
        gemm(GdnProjection.Out, w.OutWeight, w.OutQuantType, core.Slice(0, T * vDim), y, w.OutOutputDim, w.OutInputDim, T);
        if (TensorDump.Enabled) Dump2D($"blk.{absoluteLayer}.linear_attn_out", y, T, hiddenSize);
    }

    private static void Dump2D(string name, Span<float> data, int d0, int d1)
    {
        fixed (float* p = data) TensorDump.Dump2D(name, p, d0, d1);
    }

    private static void Dump3D(string name, Span<float> data, int d0, int d1, int d2)
    {
        fixed (float* p = data) TensorDump.Dump3D(name, p, d0, d1, d2);
    }
}
