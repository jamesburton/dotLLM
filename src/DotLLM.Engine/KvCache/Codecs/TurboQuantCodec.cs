using System;

namespace DotLLM.Engine.KvCache.Codecs;

/// <summary>
/// TurboQuant data-oblivious KV-vector codec — the MSE stage (Algorithm 1) of
/// <i>TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate</i>
/// (arXiv:2504.19874, Google Research / NYU). This is the calibration-free,
/// per-vector codec that the KV cache's history region can use instead of Q8_0/Q4_0.
/// </summary>
/// <remarks>
/// <para><b>Pipeline (encode).</b> For each K (or V) vector <c>x ∈ ℝ^d</c>:
/// (1) split out the norm <c>‖x‖₂</c> (stored fp32, out-of-band) and work on the unit
/// direction; (2) apply a fixed, seeded <b>orthogonal rotation</b> so the coordinates
/// become near-i.i.d. and approximately Gaussian (<c>N(0, 1/d)</c> in high dimension —
/// Lemma 1 of the paper); (3) quantize each coordinate independently with an
/// <b>optimal Lloyd–Max scalar quantizer</b> for that distribution. Decode inverts each
/// step: centroid lookup → inverse rotation → rescale by the stored norm.</para>
///
/// <para><b>Rotation — deviation from the paper, by design.</b> The paper uses a dense
/// Haar-random rotation (QR of a Gaussian matrix), which is <c>O(d²)</c> per vector — far
/// too expensive on the decode hot path (the explicit risk called out in the KV plan). We
/// substitute the <b>Randomized Hadamard Transform</b> (a seeded ±1 sign flip followed by
/// a normalized Walsh–Hadamard transform), which is <c>O(d log d)</c> and yields the same
/// coordinate marginal in the large-d limit — the standard structured rotation in
/// QuaRot/QuIP#-style quantizers. It requires <c>d</c> to be a power of two (every KV
/// head dim we target — 128/256/512 — is). The substitution is validated empirically by
/// reconstruction MSE vs the paper's per-bit bound.</para>
///
/// <para><b>Codebook.</b> Because a rotated unit vector's coordinates are <c>N(0, 1/d)</c>
/// in high dimension, the optimal per-coordinate codebook is the standard normal
/// Lloyd–Max codebook scaled by <c>1/√d</c>. This reproduces the paper's stated centroids
/// exactly: b=1 → <c>±√(2/(πd))</c>, b=2 → <c>±0.453/√d, ±1.51/√d</c>. We compute the
/// N(0,1) Lloyd–Max levels numerically once per <c>(d, bits)</c>.</para>
///
/// <para><b>Not yet included:</b> the optional QJL 1-bit residual stage (Algorithm 2) that
/// debiases inner-product/attention-score estimates, and the mixed per-channel
/// bit allocation for non-integer bit-widths (2.5/3.5). Those layer on top of this MSE
/// codec; this stage is correct and testable on its own (reconstruction quality).</para>
/// </remarks>
public sealed class TurboQuantCodec
{
    private readonly int _headDim;
    private readonly int _bits;
    private readonly float[] _signs;       // [headDim] randomized Hadamard sign flips (±1)
    private readonly float[] _centroids;   // [1<<bits] Lloyd–Max centroids, ascending, scaled by 1/√d
    private readonly float[] _boundaries;  // [(1<<bits)-1] decision boundaries (midpoints), ascending
    private readonly float _invSqrtD;      // 1/√d — RHT normalization

    /// <summary>Per-head dimension this codec was built for (a power of two).</summary>
    public int HeadDim => _headDim;

    /// <summary>Bits per quantized coordinate.</summary>
    public int Bits => _bits;

    /// <summary>Number of scalar quantization levels (<c>2^Bits</c>).</summary>
    public int LevelCount => 1 << _bits;

    /// <summary>Packed code bytes for one vector (<c>ceil(headDim * bits / 8)</c>).</summary>
    public int CodeBytesPerVector => (_headDim * _bits + 7) / 8;

    /// <summary>The Lloyd–Max centroids (scaled by <c>1/√d</c>), ascending. Exposed for tests.</summary>
    public ReadOnlySpan<float> Centroids => _centroids;

    /// <summary>
    /// Builds a codec for one KV layer-class.
    /// </summary>
    /// <param name="headDim">Vector dimension (KV head dim); must be a positive power of two.</param>
    /// <param name="bits">Bits per coordinate (1–8).</param>
    /// <param name="seed">Deterministic seed for the rotation sign flips. The SAME seed must be
    /// used to decode; persist it with the cache so rollback / prefix reuse stay valid.</param>
    public TurboQuantCodec(int headDim, int bits, ulong seed)
    {
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0)
            throw new ArgumentException($"headDim must be a positive power of two; got {headDim}.", nameof(headDim));
        if (bits < 1 || bits > 8)
            throw new ArgumentOutOfRangeException(nameof(bits), bits, "bits must be in [1, 8].");

        _headDim = headDim;
        _bits = bits;
        _invSqrtD = 1.0f / MathF.Sqrt(headDim);

        _signs = BuildSigns(headDim, seed);

        double[] normalLevels = BuildNormalLloydMax(1 << bits);
        _centroids = new float[1 << bits];
        for (int i = 0; i < _centroids.Length; i++)
            _centroids[i] = (float)(normalLevels[i] * _invSqrtD);

        _boundaries = new float[_centroids.Length - 1];
        for (int i = 0; i < _boundaries.Length; i++)
            _boundaries[i] = 0.5f * (_centroids[i] + _centroids[i + 1]);
    }

    /// <summary>
    /// Encodes one vector into packed per-coordinate codes and returns its L2 norm
    /// (stored separately, fp32). The direction is rotated then quantized; a zero vector
    /// encodes to the centroid nearest 0 with norm 0.
    /// </summary>
    public float Encode(ReadOnlySpan<float> vector, Span<byte> codes)
    {
        if (vector.Length != _headDim)
            throw new ArgumentException($"vector length {vector.Length} != headDim {_headDim}.", nameof(vector));
        if (codes.Length < CodeBytesPerVector)
            throw new ArgumentException($"codes too small: need {CodeBytesPerVector}, got {codes.Length}.", nameof(codes));

        double sumSq = 0;
        for (int i = 0; i < _headDim; i++) sumSq += (double)vector[i] * vector[i];
        float norm = (float)Math.Sqrt(sumSq);

        Span<float> buf = _headDim <= 1024 ? stackalloc float[_headDim] : new float[_headDim];
        if (norm > 0)
        {
            float inv = 1.0f / norm;
            for (int i = 0; i < _headDim; i++) buf[i] = vector[i] * inv;
        }
        else
        {
            buf.Clear();
        }

        Rotate(buf);                 // unit direction → rotated coordinates (~N(0,1/d))

        codes[..CodeBytesPerVector].Clear();
        for (int i = 0; i < _headDim; i++)
            WriteCode(codes, i, NearestIndex(buf[i]));

        return norm;
    }

    /// <summary>
    /// Decodes one vector from packed codes + the stored norm into <paramref name="output"/>.
    /// </summary>
    public void Decode(ReadOnlySpan<byte> codes, float norm, Span<float> output)
    {
        if (output.Length != _headDim)
            throw new ArgumentException($"output length {output.Length} != headDim {_headDim}.", nameof(output));
        if (codes.Length < CodeBytesPerVector)
            throw new ArgumentException($"codes too small: need {CodeBytesPerVector}, got {codes.Length}.", nameof(codes));

        for (int i = 0; i < _headDim; i++)
            output[i] = _centroids[ReadCode(codes, i)];

        InverseRotate(output);       // rotated coordinates → unit direction

        if (norm != 1.0f)
            for (int i = 0; i < _headDim; i++) output[i] *= norm;
    }

    // ── Rotation (Randomized Hadamard Transform) ─────────────────────────

    // Forward: y = (1/√d) · H · (D · x)  — sign flip, then Walsh–Hadamard, then normalize.
    private void Rotate(Span<float> x)
    {
        for (int i = 0; i < _headDim; i++) x[i] *= _signs[i];
        FastWalshHadamard(x);
        for (int i = 0; i < _headDim; i++) x[i] *= _invSqrtD;
    }

    // Inverse: x = D · ((1/√d) · H · y). Because H is symmetric with H·H = d·I and D·D = I,
    // applying the normalized WHT then the sign flip inverts the forward transform exactly.
    private void InverseRotate(Span<float> y)
    {
        FastWalshHadamard(y);
        for (int i = 0; i < _headDim; i++) y[i] *= _invSqrtD * _signs[i];
    }

    // In-place unnormalized Walsh–Hadamard transform (H·x); H·H = n·I.
    private static void FastWalshHadamard(Span<float> a)
    {
        int n = a.Length;
        for (int len = 1; len < n; len <<= 1)
        {
            for (int i = 0; i < n; i += len << 1)
            {
                for (int j = i; j < i + len; j++)
                {
                    float u = a[j];
                    float v = a[j + len];
                    a[j] = u + v;
                    a[j + len] = u - v;
                }
            }
        }
    }

    private static float[] BuildSigns(int d, ulong seed)
    {
        var signs = new float[d];
        ulong s = seed == 0 ? 0x9E3779B97F4A7C15UL : seed; // avoid the all-zero SplitMix fixpoint
        for (int i = 0; i < d; i++)
        {
            // SplitMix64 — deterministic, well-distributed sign bits.
            s += 0x9E3779B97F4A7C15UL;
            ulong z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            signs[i] = (z & 1UL) == 0 ? 1.0f : -1.0f;
        }
        return signs;
    }

    // ── Scalar quantization ──────────────────────────────────────────────

    private int NearestIndex(float value)
    {
        // Boundaries are ascending midpoints; the index is the count of boundaries < value.
        int lo = 0, hi = _boundaries.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (_boundaries[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private void WriteCode(Span<byte> codes, int coord, int index)
    {
        int bitPos = coord * _bits;
        for (int b = 0; b < _bits; b++)
        {
            if (((index >> b) & 1) != 0)
            {
                int p = bitPos + b;
                codes[p >> 3] |= (byte)(1 << (p & 7));
            }
        }
    }

    private int ReadCode(ReadOnlySpan<byte> codes, int coord)
    {
        int bitPos = coord * _bits;
        int index = 0;
        for (int b = 0; b < _bits; b++)
        {
            int p = bitPos + b;
            if ((codes[p >> 3] & (1 << (p & 7))) != 0)
                index |= 1 << b;
        }
        return index;
    }

    // ── Lloyd–Max codebook for the standard normal ───────────────────────

    /// <summary>
    /// Computes the MSE-optimal <paramref name="levels"/>-level scalar quantizer centroids
    /// for the standard normal distribution via Lloyd's algorithm on a fine grid. Symmetric
    /// about 0. Caller scales by <c>1/√d</c> for the rotated-coordinate distribution.
    /// </summary>
    private static double[] BuildNormalLloydMax(int levels)
    {
        // Fine grid over [-8, 8] carrying the standard-normal weight (the 1/√(2π) constant
        // cancels in the conditional means, so we omit it).
        const int gridN = 1 << 16;
        const double lo = -8.0, hi = 8.0;
        double step = (hi - lo) / gridN;

        var x = new double[gridN];
        var w = new double[gridN];
        for (int i = 0; i < gridN; i++)
        {
            double xi = lo + (i + 0.5) * step;
            x[i] = xi;
            w[i] = Math.Exp(-0.5 * xi * xi) * step;
        }

        // Initialise centroids at uniform interior points (Lloyd's converges to the global
        // optimum for a log-concave density regardless of a reasonable init).
        var c = new double[levels];
        for (int k = 0; k < levels; k++)
            c[k] = -3.0 + 6.0 * (k + 0.5) / levels;

        var bnd = new double[levels - 1];
        for (int iter = 0; iter < 200; iter++)
        {
            for (int k = 0; k < levels - 1; k++) bnd[k] = 0.5 * (c[k] + c[k + 1]);

            var num = new double[levels];
            var den = new double[levels];
            int cell = 0;
            for (int i = 0; i < gridN; i++)
            {
                while (cell < levels - 1 && x[i] >= bnd[cell]) cell++;
                num[cell] += x[i] * w[i];
                den[cell] += w[i];
            }

            double maxShift = 0;
            for (int k = 0; k < levels; k++)
            {
                if (den[k] > 0)
                {
                    double nc = num[k] / den[k];
                    maxShift = Math.Max(maxShift, Math.Abs(nc - c[k]));
                    c[k] = nc;
                }
            }
            if (maxShift < 1e-9) break;
        }

        Array.Sort(c);
        return c;
    }
}
