using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Qwen4-Exp (Qwen3.8-Flash-Next) gated residual ("hyper-connection", GR) pure-F32 ops. The residual is
/// <c>hcCount</c> parallel streams per token; every layer READS the streams into one block input and WRITES the block
/// output back into all of them with a per-stream gain. Verified against HF <c>transformers</c>
/// <c>modeling_qwen4_exp.py</c> <c>Qwen4ExpTextGatedResidual</c> and the decoder-layer wiring (and llama.cpp
/// <c>src/models/qwen4exp.cpp</c>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Layout.</b> The residual is <c>[seqLen, hcCount, hiddenSize]</c> row-major, i.e. each token owns one flat
/// <c>hcCount * hiddenSize</c> row — exactly HF's <c>unflatten(-1, (hc, hidden))</c>. The three GEMVs of the read path
/// (<c>down</c>: <c>S*H -&gt; lowRank</c>, <c>up</c>: <c>lowRank -&gt; S*H</c>, <c>inject</c>: <c>S*H -&gt; S</c>) consume that flat
/// row with no copies; they are NOT in this class because the model owns quant-aware GEMM dispatch (the real GGUF stores
/// down/up as Q8_0 and inject/norm as F32).
/// </para>
/// <para>
/// <b>Read</b> (token mixer / FFN input), per token:
/// <code>
/// xn   = groupRMS(R) * gamma                       // RMS over each stream's H channels; gamma [S*H] (1+w already folded)
/// d    = silu(down(xn) / S)                        // ActivateLowRank
/// mix  = sigmoid(up(d))                            // [S, H]
/// h    = mean_s(mix * xn)                          // MixAndMean: mean over streams, NOT sum
/// inj  = 2 * sigmoid(inject(xn) / S)               // InjectGains: [S] per-stream write gain (absent for the final head mixer)
/// </code>
/// <b>Write</b>: <c>R[s,:] += inj[s] * y[:]</c> against the RAW (un-normalised) residual — HF returns <c>hyper_input</c>.
/// </para>
/// <para>
/// The final head mixer (HF <c>hyper_connection_mixer</c>, <c>use_combine=False</c>) is the same read path with no inject
/// weight; its output replaces the final RMSNorm. Every op has a scalar reference (<c>*Scalar</c>, internal) and a
/// <see cref="TensorPrimitives"/> implementation, cross-checked in the unit tests.
/// </para>
/// </remarks>
public static class Qwen4ExpGatedResidual
{
    /// <summary>
    /// Per-stream RMS norm: for each token and each of the <paramref name="hcCount"/> streams, normalises the stream's
    /// <paramref name="hiddenSize"/> channels and scales by the matching slice of <paramref name="gamma"/>
    /// (a per-stream gamma, not shared across streams).
    /// </summary>
    /// <param name="residual">Raw residual <c>[seqLen, hcCount * hiddenSize]</c>.</param>
    /// <param name="gamma">GGUF-convention gain <c>[hcCount * hiddenSize]</c> (HF <c>1 + weight</c> already folded by the converter).</param>
    /// <param name="hcCount">Number of residual streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="eps">RMS epsilon.</param>
    /// <param name="normed">Destination <c>[seqLen, hcCount * hiddenSize]</c>; may alias <paramref name="residual"/>.</param>
    /// <param name="seqLen">Token count.</param>
    [SkipLocalsInit]
    public static void GroupRmsNorm(ReadOnlySpan<float> residual, ReadOnlySpan<float> gamma,
                                    int hcCount, int hiddenSize, float eps, Span<float> normed, int seqLen)
    {
        int row = hcCount * hiddenSize;
        Validate(residual.Length, row, seqLen, nameof(residual));
        if (gamma.Length < row) throw new ArgumentException("gamma must hold hcCount*hiddenSize entries.", nameof(gamma));
        if (normed.Length < (long)seqLen * row) throw new ArgumentException("normed too small.", nameof(normed));

        for (int t = 0; t < seqLen; t++)
        {
            for (int s = 0; s < hcCount; s++)
            {
                int off = t * row + s * hiddenSize;
                RmsNorm.Execute(residual.Slice(off, hiddenSize), gamma.Slice(s * hiddenSize, hiddenSize), eps,
                                normed.Slice(off, hiddenSize));
            }
        }
    }

    /// <summary>Scalar reference for <see cref="GroupRmsNorm"/>.</summary>
    internal static void GroupRmsNormScalar(ReadOnlySpan<float> residual, ReadOnlySpan<float> gamma,
                                            int hcCount, int hiddenSize, float eps, Span<float> normed, int seqLen)
    {
        int row = hcCount * hiddenSize;
        for (int t = 0; t < seqLen; t++)
        for (int s = 0; s < hcCount; s++)
        {
            int off = t * row + s * hiddenSize;
            double ss = 0;
            for (int i = 0; i < hiddenSize; i++) ss += (double)residual[off + i] * residual[off + i];
            float scale = (float)(1.0 / Math.Sqrt(ss / hiddenSize + eps));
            for (int i = 0; i < hiddenSize; i++)
                normed[off + i] = residual[off + i] * scale * gamma[s * hiddenSize + i];
        }
    }

    /// <summary>
    /// In place <c>v = silu(v / hcCount)</c> over the low-rank <c>down</c> projection output — the explicit
    /// <c>/ hc_count</c> of HF's <c>F.silu(input_mix_weight_down(x) / hc_count)</c> (easy to drop).
    /// </summary>
    /// <param name="downOut">Down projection output <c>[seqLen * lowRank]</c>.</param>
    /// <param name="hcCount">Number of residual streams (the divisor).</param>
    [SkipLocalsInit]
    public static void ActivateLowRank(Span<float> downOut, int hcCount)
    {
        TensorPrimitives.Multiply(downOut, 1.0f / hcCount, downOut);
        SiLu.Execute(downOut, downOut);
    }

    /// <summary>Scalar reference for <see cref="ActivateLowRank"/>.</summary>
    internal static void ActivateLowRankScalar(Span<float> downOut, int hcCount)
    {
        for (int i = 0; i < downOut.Length; i++)
        {
            float a = downOut[i] / hcCount;
            downOut[i] = a / (1f + MathF.Exp(-a));
        }
    }

    /// <summary>
    /// Block input: <c>h[t] = mean_s( sigmoid(upOut[t,s,:]) * normed[t,s,:] )</c> — a per-(stream, channel) mix weight,
    /// then the MEAN over streams (divide by <paramref name="hcCount"/>), not the sum.
    /// </summary>
    /// <param name="upOut">Up projection output <c>[seqLen, hcCount * hiddenSize]</c>, PRE-sigmoid. Overwritten (used as scratch).</param>
    /// <param name="normed">Output of <see cref="GroupRmsNorm"/> <c>[seqLen, hcCount * hiddenSize]</c>.</param>
    /// <param name="hcCount">Number of residual streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="blockInput">Destination <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="seqLen">Token count.</param>
    [SkipLocalsInit]
    public static void MixAndMean(Span<float> upOut, ReadOnlySpan<float> normed,
                                  int hcCount, int hiddenSize, Span<float> blockInput, int seqLen)
    {
        int row = hcCount * hiddenSize;
        Validate(upOut.Length, row, seqLen, nameof(upOut));
        Validate(normed.Length, row, seqLen, nameof(normed));
        if (blockInput.Length < (long)seqLen * hiddenSize) throw new ArgumentException("blockInput too small.", nameof(blockInput));

        Span<float> all = upOut.Slice(0, seqLen * row);
        TensorPrimitives.Sigmoid(all, all);
        TensorPrimitives.Multiply(all, normed.Slice(0, seqLen * row), all);
        float inv = 1.0f / hcCount;
        for (int t = 0; t < seqLen; t++)
        {
            Span<float> dst = blockInput.Slice(t * hiddenSize, hiddenSize);
            upOut.Slice(t * row, hiddenSize).CopyTo(dst);
            for (int s = 1; s < hcCount; s++)
                TensorPrimitives.Add(dst, upOut.Slice(t * row + s * hiddenSize, hiddenSize), dst);
            TensorPrimitives.Multiply(dst, inv, dst);
        }
    }

    /// <summary>Scalar reference for <see cref="MixAndMean"/> (does not modify <paramref name="upOut"/>).</summary>
    internal static void MixAndMeanScalar(ReadOnlySpan<float> upOut, ReadOnlySpan<float> normed,
                                          int hcCount, int hiddenSize, Span<float> blockInput, int seqLen)
    {
        int row = hcCount * hiddenSize;
        for (int t = 0; t < seqLen; t++)
        for (int i = 0; i < hiddenSize; i++)
        {
            double acc = 0;
            for (int s = 0; s < hcCount; s++)
            {
                int k = t * row + s * hiddenSize + i;
                acc += (1.0 / (1.0 + Math.Exp(-(double)upOut[k]))) * normed[k];
            }
            blockInput[t * hiddenSize + i] = (float)(acc / hcCount);
        }
    }

    /// <summary>
    /// Per-stream write gains: in place <c>g = 2 * sigmoid(g / hcCount)</c> over the inject projection output
    /// (<c>[seqLen, hcCount]</c>). Absent for the final head mixer.
    /// </summary>
    /// <param name="injectOut">Inject projection output <c>[seqLen * hcCount]</c>; becomes the gains.</param>
    /// <param name="hcCount">Number of residual streams (the divisor).</param>
    [SkipLocalsInit]
    public static void InjectGains(Span<float> injectOut, int hcCount)
    {
        TensorPrimitives.Multiply(injectOut, 1.0f / hcCount, injectOut);
        TensorPrimitives.Sigmoid(injectOut, injectOut);
        TensorPrimitives.Multiply(injectOut, 2.0f, injectOut);
    }

    /// <summary>Scalar reference for <see cref="InjectGains"/>.</summary>
    internal static void InjectGainsScalar(Span<float> injectOut, int hcCount)
    {
        for (int i = 0; i < injectOut.Length; i++)
            injectOut[i] = 2f / (1f + MathF.Exp(-(injectOut[i] / hcCount)));
    }

    /// <summary>
    /// Write: <c>residual[t, s, :] += gains[t, s] * blockOutput[t, :]</c> — the block output is broadcast to every stream
    /// and scaled by that stream's gain, added to the RAW residual.
    /// </summary>
    /// <param name="residual">Raw residual <c>[seqLen, hcCount * hiddenSize]</c>, updated in place.</param>
    /// <param name="blockOutput">Token-mixer / FFN output <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="gains">Output of <see cref="InjectGains"/> <c>[seqLen, hcCount]</c>.</param>
    /// <param name="hcCount">Number of residual streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="seqLen">Token count.</param>
    [SkipLocalsInit]
    public static void Write(Span<float> residual, ReadOnlySpan<float> blockOutput, ReadOnlySpan<float> gains,
                             int hcCount, int hiddenSize, int seqLen)
    {
        int row = hcCount * hiddenSize;
        Validate(residual.Length, row, seqLen, nameof(residual));
        if (blockOutput.Length < (long)seqLen * hiddenSize) throw new ArgumentException("blockOutput too small.", nameof(blockOutput));
        if (gains.Length < (long)seqLen * hcCount) throw new ArgumentException("gains too small.", nameof(gains));

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> y = blockOutput.Slice(t * hiddenSize, hiddenSize);
            for (int s = 0; s < hcCount; s++)
            {
                Span<float> r = residual.Slice(t * row + s * hiddenSize, hiddenSize);
                TensorPrimitives.MultiplyAdd(y, gains[t * hcCount + s], r, r);
            }
        }
    }

    /// <summary>Scalar reference for <see cref="Write"/>.</summary>
    internal static void WriteScalar(Span<float> residual, ReadOnlySpan<float> blockOutput, ReadOnlySpan<float> gains,
                                     int hcCount, int hiddenSize, int seqLen)
    {
        int row = hcCount * hiddenSize;
        for (int t = 0; t < seqLen; t++)
        for (int s = 0; s < hcCount; s++)
        for (int i = 0; i < hiddenSize; i++)
            residual[t * row + s * hiddenSize + i] += gains[t * hcCount + s] * blockOutput[t * hiddenSize + i];
    }

    /// <summary>
    /// Embedding broadcast (HF <c>hidden_states.repeat(1, 1, hc_count)</c>): copies each token's embedding into all
    /// <paramref name="hcCount"/> streams of the residual.
    /// </summary>
    /// <param name="embedding">Token embeddings <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="hcCount">Number of residual streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="residual">Destination <c>[seqLen, hcCount * hiddenSize]</c>.</param>
    /// <param name="seqLen">Token count.</param>
    public static void Broadcast(ReadOnlySpan<float> embedding, int hcCount, int hiddenSize, Span<float> residual, int seqLen)
    {
        int row = hcCount * hiddenSize;
        for (int t = 0; t < seqLen; t++)
        for (int s = 0; s < hcCount; s++)
            embedding.Slice(t * hiddenSize, hiddenSize).CopyTo(residual.Slice(t * row + s * hiddenSize, hiddenSize));
    }

    private static void Validate(int length, int row, int seqLen, string name)
    {
        if (seqLen < 0 || length < (long)seqLen * row)
            throw new ArgumentException($"{name} holds {length} floats, need seqLen*hcCount*hiddenSize = {(long)seqLen * row}.", name);
    }
}
