using System.Runtime.CompilerServices;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels.Experimental;

// NanoQuant single-path factorised linear (issue #866).
//
// Functional contract (the community "llama.cpp-i32-lsb-v1" GGUF layout, which is also what the Samsung NanoQuant
// export packs; the maths is public in the format documentation, this file is an independent implementation):
//
//   x'      = x .* scale_pre                                    (d_in)
//   latent  = scale_mid .* (V x')                               (r)      V in {+-1}^{r x d_in}
//   base    = scale_post .* (U latent)                          (d_out)  U in {+-1}^{d_out x r}
//   salient = sum_s  salient_weight[o, s] * x[salient_idx[s]]   (RAW x: the reference runtime does not mask; the stored
//                                                                scale_pre at the salient indices must be exactly 0)
//   y       = base + salient
//
// Sign words: int32, LSB-first, bit set = -1, tail bits clear (= +1). U is [d_out x ceil(r/32)] words, V is
// [r x ceil(d_in/32)] words. That is byte-identical to the LittleBit spike's layout (#832), so a NanoQuant layer is
// exactly one LittleBit path (h = scale_post, l = scale_mid, g = scale_pre) plus the salient column side path, and the
// spike's AVX2 / AVX-VNNI kernels are reused unchanged.
//
// Scales are stored bf16 in the community files and widened exactly to F32 (the LittleBit kernels hold F32 scales since #864).

/// <summary>
/// One NanoQuant linear layer: a single binary-factor path plus an optional FP "salient column" side path.
/// A stacked layer (community <c>attn_qkv</c>: q, k, v rows concatenated, one shared V) is just a layer with
/// <c>DOut = dq + dk + dv</c>; the caller slices the output.
/// </summary>
public sealed unsafe class NanoQuantLayer : IDisposable
{
    private readonly LittleBitLayer _base;
    private readonly NanoQuantSalient? _salient;

    /// <summary>The underlying single-path LittleBit layer (h = scale_post, l = scale_mid, g = scale_pre).</summary>
    public LittleBitLayer Base => _base;
    /// <summary>The salient-column side path, or null when the layer has none.</summary>
    public NanoQuantSalient? Salient => _salient;

    /// <summary>Output features.</summary>
    public int DOut => _base.DOut;
    /// <summary>Input features.</summary>
    public int DIn => _base.DIn;
    /// <summary>Latent rank.</summary>
    public int R => _base.Paths[0].R;
    /// <summary>Number of salient (outlier) input columns kept in floating point.</summary>
    public int SalientCount => _salient?.Count ?? 0;

    private NanoQuantLayer(LittleBitLayer b, NanoQuantSalient? sal) { _base = b; _salient = sal; }

    /// <summary>Builds a layer from the on-disk packed form.</summary>
    /// <param name="dOut">Output features.</param>
    /// <param name="dIn">Input features.</param>
    /// <param name="r">Latent rank.</param>
    /// <param name="uWords"><c>nq_u</c> words, [dOut x ceil(r/32)] row-major.</param>
    /// <param name="vWords"><c>nq_v</c> words, [r x ceil(dIn/32)] row-major.</param>
    /// <param name="scalePre">Length dIn (zero at every salient index).</param>
    /// <param name="scaleMid">Length r.</param>
    /// <param name="scalePost">Length dOut.</param>
    /// <param name="salientIdx">Strictly increasing input-column indices, length k (may be empty).</param>
    /// <param name="salientWeight">GGUF order <c>[k, dOut]</c>, i.e. <c>w[o * k + s]</c> (ggml dim0 = k is contiguous).</param>
    /// <exception cref="ArgumentException">Shapes inconsistent.</exception>
    /// <exception cref="InvalidDataException">Salient indices invalid, or scale_pre non-zero at a salient index.</exception>
    public static NanoQuantLayer FromPacked(int dOut, int dIn, int r,
        ReadOnlySpan<int> uWords, ReadOnlySpan<int> vWords,
        ReadOnlySpan<float> scalePre, ReadOnlySpan<float> scaleMid, ReadOnlySpan<float> scalePost,
        ReadOnlySpan<int> salientIdx, ReadOnlySpan<float> salientWeight)
    {
        if (dOut <= 0 || dIn <= 0 || r <= 0) throw new ArgumentOutOfRangeException(nameof(r), "dimensions must be positive");
        int uw = (r + 31) / 32, vw = (dIn + 31) / 32, k = salientIdx.Length;
        if (uWords.Length != checked(dOut * uw)) throw new ArgumentException($"nq_u needs {dOut}x{uw} words, got {uWords.Length}");
        if (vWords.Length != checked(r * vw)) throw new ArgumentException($"nq_v needs {r}x{vw} words, got {vWords.Length}");
        if (scalePre.Length != dIn || scaleMid.Length != r || scalePost.Length != dOut)
            throw new ArgumentException("scale vector length mismatch");
        if (salientWeight.Length != checked(k * dOut)) throw new ArgumentException($"salient weight needs {k}x{dOut} values, got {salientWeight.Length}");

        for (int s = 0; s < k; s++)
        {
            int c = salientIdx[s];
            if (c < 0 || c >= dIn) throw new InvalidDataException($"salient index {c} out of range [0,{dIn})");
            if (s > 0 && c <= salientIdx[s - 1]) throw new InvalidDataException("salient indices must be strictly increasing");
            if (scalePre[c] != 0f) throw new InvalidDataException($"scale_pre[{c}] must be exactly 0 at a salient index, got {scalePre[c]}");
        }

        var path = LittleBitPath.FromPackedWords(dOut, dIn, r, uWords, vWords, scalePost, scalePre, scaleMid);
        return new NanoQuantLayer(new LittleBitLayer(path),
            k == 0 ? null : new NanoQuantSalient(dOut, salientIdx.ToArray(), salientWeight.ToArray()));
    }

    /// <summary>Scratch sized for this layer.</summary>
    public LittleBitScratch CreateScratch() => new(DIn, R);

    /// <summary>y[0..DOut) = layer(x[0..DIn)).</summary>
    [SkipLocalsInit]
    public void Gemv(float* x, float* y, LittleBitScratch scratch, ComputeThreadPool? pool,
                     LittleBitKernel kernel = LittleBitKernel.Avx2Float)
    {
        _base.Gemv(x, y, scratch, pool, kernel);
        _salient?.AddTo(x, y, 1, DIn);
    }

    /// <summary>
    /// Dense F32 decode of the whole layer (base factorisation plus the salient columns) as a 64-byte aligned native
    /// [DOut x DIn] buffer the caller frees with <c>NativeMemory.AlignedFree</c>: the "F32-decoded control".
    /// </summary>
    public float* DecodeDense()
    {
        float* w = LittleBitReference.DecodeDenseFast(_base);
        _salient?.AddToDense(w, DIn);
        return w;
    }

    /// <inheritdoc/>
    public void Dispose() => _base.Dispose();
}

/// <summary>
/// The NanoQuant "salient column" side path: <c>y[o] += sum_s W[o,s] * x[idx[s]]</c> on the RAW input (the base path's
/// <c>scale_pre</c> is exactly 0 at those columns). W is stored [dOut][k] row-major.
/// </summary>
public sealed unsafe class NanoQuantSalient
{
    private readonly int[] _idx;
    private readonly float[] _w;

    /// <summary>Output features.</summary>
    public int DOut { get; }
    /// <summary>Number of salient columns.</summary>
    public int Count => _idx.Length;
    /// <summary>Bytes held (F32 weights + int32 indices).</summary>
    public long Bytes => 4L * _w.Length + 4L * _idx.Length;

    /// <summary>Creates the side path; <paramref name="w"/> is [dOut x k] row-major.</summary>
    public NanoQuantSalient(int dOut, int[] idx, float[] w)
    {
        if (w.Length != checked(dOut * idx.Length)) throw new ArgumentException("salient weight size mismatch");
        DOut = dOut; _idx = idx; _w = w;
    }

    /// <summary>y[t*DOut + o] += salient(x[t*dIn ..]) for n tokens.</summary>
    public void AddTo(float* x, float* y, int n, int dIn)
    {
        int k = _idx.Length, dOut = DOut;
        fixed (float* w = _w)
            for (int t = 0; t < n; t++)
            {
                float* xt = x + (long)t * dIn, yt = y + (long)t * dOut;
                for (int o = 0; o < dOut; o++)
                {
                    float acc = 0;
                    float* wr = w + (long)o * k;
                    for (int s = 0; s < k; s++) acc += wr[s] * xt[_idx[s]];
                    yt[o] += acc;
                }
            }
    }

    /// <summary>Adds the salient columns into a dense [DOut x dIn] matrix (base decode has 0 in those columns).</summary>
    public void AddToDense(float* dense, int dIn)
    {
        int k = _idx.Length;
        for (int o = 0; o < DOut; o++)
            for (int s = 0; s < k; s++) dense[(long)o * dIn + _idx[s]] += _w[o * k + s];
    }
}

/// <summary>Scalar float64 reference for <see cref="NanoQuantLayer"/>, decoding the int32 words directly (independent of the spike's byte packing).</summary>
public static class NanoQuantReference
{
    /// <summary>y = NanoQuant(x) in double precision straight from the on-disk words and scales.</summary>
    public static double[] Gemv(int dOut, int dIn, int r,
        ReadOnlySpan<int> uWords, ReadOnlySpan<int> vWords,
        ReadOnlySpan<float> scalePre, ReadOnlySpan<float> scaleMid, ReadOnlySpan<float> scalePost,
        ReadOnlySpan<int> salientIdx, ReadOnlySpan<float> salientWeight, ReadOnlySpan<float> x)
    {
        int uw = (r + 31) / 32, vw = (dIn + 31) / 32, k = salientIdx.Length;
        var xp = new double[dIn];
        for (int i = 0; i < dIn; i++) xp[i] = (double)x[i] * scalePre[i];
        var lat = new double[r];
        for (int j = 0; j < r; j++)
        {
            double s = 0;
            for (int i = 0; i < dIn; i++) s += ((vWords[j * vw + (i >> 5)] >> (i & 31)) & 1) != 0 ? -xp[i] : xp[i];
            lat[j] = s * scaleMid[j];
        }
        var y = new double[dOut];
        for (int o = 0; o < dOut; o++)
        {
            double s = 0;
            for (int j = 0; j < r; j++) s += ((uWords[o * uw + (j >> 5)] >> (j & 31)) & 1) != 0 ? -lat[j] : lat[j];
            y[o] = s * scalePost[o];
            for (int q = 0; q < k; q++) y[o] += (double)salientWeight[o * k + q] * x[salientIdx[q]];
        }
        return y;
    }
}
