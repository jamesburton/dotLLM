using System;
using System.Buffers.Binary;

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
/// <para><b>QJL residual stage (Algorithm 2), opt-in.</b> The MSE stage is a slight
/// contraction (<c>‖x̃‖ &lt; ‖x‖</c>), so attention scores <c>⟨q,k̃⟩</c> read off the
/// reconstruction are biased low by ≈<c>‖r‖²</c> (the squared quantization error). Enabling
/// <c>useQjl</c> spends one of the per-coordinate bits on a 1-bit <b>quantized Johnson–
/// Lindenstrauss</b> sketch of the residual <c>r = x − x̃_mse</c>: the MSE stage runs at
/// <c>bits−1</c> and we additionally store, per vector, <c>d</c> sign bits
/// <c>q = sign(S·r)</c> (a fixed seeded Gaussian sketch <c>S ∈ ℝ^{d×d}</c>) plus the residual
/// norm <c>γ = ‖r‖</c>. Decode folds an unbiased residual reconstruction
/// <c>x̃_qjl = (√(π/2)/d)·γ·Sᵀ·q</c> back into the output, so for any query <c>y</c>,
/// <c>E_S[⟨y, x̃_mse + x̃_qjl⟩] = ⟨y, x⟩</c> — the score bias is removed. (This is a
/// deliberate trade: <c>x̃_qjl</c> is a JL-noisy estimate of <c>r</c>, so the per-vector ℓ2
/// reconstruction error <i>rises</i> while the inner-product/score bias falls to ≈0 — exactly
/// the QJL design point.) The sketch is O(d²) per vector on encode and decode; structuring it
/// (Hadamard) is a later optimisation.</para>
///
/// <para><b>Not yet included:</b> the mixed per-channel bit allocation for non-integer
/// bit-widths (2.5/3.5). That layers on top of this codec.</para>
/// </remarks>
public sealed class TurboQuantCodec
{
    private readonly int _headDim;
    private readonly int _bits;            // total per-coordinate bit budget
    private readonly int _mseBits;         // bits the MSE stage uses (= _bits, or _bits-1 when QJL)
    private readonly bool _useQjl;
    private readonly float[] _signs;       // [headDim] randomized Hadamard sign flips (±1)
    private readonly float[] _centroids;   // [1<<mseBits] Lloyd–Max centroids, ascending, scaled by 1/√d
    private readonly float[] _boundaries;  // [(1<<mseBits)-1] decision boundaries (midpoints), ascending
    private readonly float _invSqrtD;      // 1/√d — RHT normalization
    private readonly float[]? _sketch;     // [headDim*headDim] row-major Gaussian QJL sketch S (null if !QJL)
    private readonly float _qjlScale;      // √(π/2)/d — QJL residual reconstruction scale
    private readonly int _mseCodeBytes;    // packed MSE codes per vector
    private readonly int _qjlSignBytes;    // packed QJL sign bits per vector (0 if !QJL)

    /// <summary>Per-head dimension this codec was built for (a power of two).</summary>
    public int HeadDim => _headDim;

    /// <summary>Total per-coordinate bit budget (MSE bits + the 1 QJL residual bit when enabled).</summary>
    public int Bits => _bits;

    /// <summary>Whether the QJL 1-bit residual stage (Algorithm 2) is active.</summary>
    public bool UseQjl => _useQjl;

    /// <summary>Number of scalar quantization levels the MSE stage uses (<c>2^(QJL ? Bits-1 : Bits)</c>).</summary>
    public int LevelCount => 1 << _mseBits;

    /// <summary>Packed code bytes for one vector: MSE codes, plus (when QJL) the sign bits and a 4-byte residual norm.</summary>
    public int CodeBytesPerVector => _mseCodeBytes + _qjlSignBytes + (_useQjl ? sizeof(float) : 0);

    /// <summary>The Lloyd–Max centroids (scaled by <c>1/√d</c>), ascending. Exposed for tests.</summary>
    public ReadOnlySpan<float> Centroids => _centroids;

    /// <summary>Bits the MSE stage uses per coordinate (<c>Bits</c>, or <c>Bits-1</c> when QJL). The
    /// packed MSE codes use this width.</summary>
    public int MseBits => _mseBits;

    /// <summary>The RHT sign flips (±1), length <c>HeadDim</c>. Exposed so a GPU dequant can
    /// reproduce the inverse rotation; the same seed reproduces them.</summary>
    public ReadOnlySpan<float> RotationSigns => _signs;

    /// <summary>The RHT normalization factor <c>1/√HeadDim</c> (applied once on encode/decode).</summary>
    public float InvSqrtD => _invSqrtD;

    /// <summary>
    /// Builds a codec for one KV layer-class.
    /// </summary>
    /// <param name="headDim">Vector dimension (KV head dim); must be a positive power of two.</param>
    /// <param name="bits">Bits per coordinate (1–8).</param>
    /// <param name="seed">Deterministic seed for the rotation sign flips (and, when enabled, the
    /// QJL sketch). The SAME seed must be used to decode; persist it with the cache so rollback /
    /// prefix reuse stay valid.</param>
    /// <param name="useQjl">Enable the QJL 1-bit residual stage (Algorithm 2). When set, the MSE
    /// stage runs at <c>bits-1</c> and one bit per coordinate funds the unbiased inner-product
    /// correction; requires <paramref name="bits"/> ≥ 2.</param>
    public TurboQuantCodec(int headDim, int bits, ulong seed, bool useQjl = false)
    {
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0)
            throw new ArgumentException($"headDim must be a positive power of two; got {headDim}.", nameof(headDim));
        if (bits < 1 || bits > 8)
            throw new ArgumentOutOfRangeException(nameof(bits), bits, "bits must be in [1, 8].");
        if (useQjl && bits < 2)
            throw new ArgumentOutOfRangeException(nameof(bits), bits, "QJL needs bits >= 2 (one bit funds the residual sketch).");

        _headDim = headDim;
        _bits = bits;
        _useQjl = useQjl;
        _mseBits = useQjl ? bits - 1 : bits;
        _invSqrtD = 1.0f / MathF.Sqrt(headDim);

        _signs = BuildSigns(headDim, seed);

        double[] normalLevels = BuildNormalLloydMax(1 << _mseBits);
        _centroids = new float[1 << _mseBits];
        for (int i = 0; i < _centroids.Length; i++)
            _centroids[i] = (float)(normalLevels[i] * _invSqrtD);

        _boundaries = new float[_centroids.Length - 1];
        for (int i = 0; i < _boundaries.Length; i++)
            _boundaries[i] = 0.5f * (_centroids[i] + _centroids[i + 1]);

        _mseCodeBytes = (_headDim * _mseBits + 7) / 8;

        if (useQjl)
        {
            _qjlSignBytes = (_headDim + 7) / 8;
            _qjlScale = MathF.Sqrt(MathF.PI / 2.0f) / _headDim;
            // Independent of the rotation: derive a distinct sketch seed so the sign flips and
            // the Gaussian sketch don't share randomness.
            _sketch = BuildGaussianSketch(headDim, seed ^ 0x2545F4914F6CDD1DUL);
        }
    }

    /// <summary>
    /// Encodes one vector into packed per-coordinate codes and returns its L2 norm
    /// (stored separately, fp32). The direction is rotated then quantized; a zero vector
    /// encodes to the centroid nearest 0 with norm 0. When QJL is enabled the per-vector
    /// residual sign bits and residual norm are packed into the tail of <paramref name="codes"/>.
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

        if (_useQjl)
            EncodeQjlResidual(vector, norm, codes);

        return norm;
    }

    // Reconstructs x̃_mse from the codes just written, forms the residual r = x − x̃_mse, and
    // packs q = sign(S·r) (d sign bits) followed by γ = ‖r‖ (fp32) into the tail of `codes`.
    private void EncodeQjlResidual(ReadOnlySpan<float> vector, float norm, Span<byte> codes)
    {
        float[] sketch = _sketch!;

        Span<float> recon = _headDim <= 1024 ? stackalloc float[_headDim] : new float[_headDim];
        DecodeMse(codes, norm, recon);

        Span<float> r = _headDim <= 1024 ? stackalloc float[_headDim] : new float[_headDim];
        double rSumSq = 0;
        for (int i = 0; i < _headDim; i++)
        {
            float ri = vector[i] - recon[i];
            r[i] = ri;
            rSumSq += (double)ri * ri;
        }
        float gamma = (float)Math.Sqrt(rSumSq);

        Span<byte> signs = codes.Slice(_mseCodeBytes, _qjlSignBytes); // already zeroed by caller
        for (int i = 0; i < _headDim; i++)
        {
            // Project the residual onto row i of the Gaussian sketch and keep only the sign.
            int rowOff = i * _headDim;
            double proj = 0;
            for (int j = 0; j < _headDim; j++) proj += (double)sketch[rowOff + j] * r[j];
            if (proj >= 0) signs[i >> 3] |= (byte)(1 << (i & 7));
        }

        BinaryPrimitives.WriteSingleLittleEndian(codes.Slice(_mseCodeBytes + _qjlSignBytes, sizeof(float)), gamma);
    }

    /// <summary>
    /// Decodes one vector from packed codes + the stored norm into <paramref name="output"/>.
    /// When QJL is enabled the unbiased residual reconstruction is folded in on top of the MSE
    /// reconstruction.
    /// </summary>
    public void Decode(ReadOnlySpan<byte> codes, float norm, Span<float> output)
    {
        if (output.Length != _headDim)
            throw new ArgumentException($"output length {output.Length} != headDim {_headDim}.", nameof(output));
        if (codes.Length < CodeBytesPerVector)
            throw new ArgumentException($"codes too small: need {CodeBytesPerVector}, got {codes.Length}.", nameof(codes));

        DecodeMse(codes, norm, output);

        if (_useQjl)
            DecodeQjlResidual(codes, output);
    }

    // MSE reconstruction only: centroid lookup → inverse rotation → rescale by norm.
    private void DecodeMse(ReadOnlySpan<byte> codes, float norm, Span<float> output)
    {
        for (int i = 0; i < _headDim; i++)
            output[i] = _centroids[ReadCode(codes, i)];

        InverseRotate(output);       // rotated coordinates → unit direction

        if (norm != 1.0f)
            for (int i = 0; i < _headDim; i++) output[i] *= norm;
    }

    // Adds x̃_qjl = (√(π/2)/d)·γ·Sᵀ·q to `output`, where q is the stored ±1 sign vector and γ the
    // stored residual norm. ⟨y, x̃_qjl⟩ is an unbiased estimate of ⟨y, r⟩ over the random sketch,
    // so the MSE contraction bias in the score ⟨y, x̃⟩ is removed in expectation.
    private void DecodeQjlResidual(ReadOnlySpan<byte> codes, Span<float> output)
    {
        float gamma = BinaryPrimitives.ReadSingleLittleEndian(codes.Slice(_mseCodeBytes + _qjlSignBytes, sizeof(float)));
        if (gamma == 0) return;

        float[] sketch = _sketch!;
        ReadOnlySpan<byte> signs = codes.Slice(_mseCodeBytes, _qjlSignBytes);

        Span<float> acc = _headDim <= 1024 ? stackalloc float[_headDim] : new float[_headDim];
        acc.Clear();

        // acc = Sᵀ·q = Σ_i q_i · S[i,:]   (row i of S is column i of Sᵀ).
        for (int i = 0; i < _headDim; i++)
        {
            float qi = (signs[i >> 3] & (1 << (i & 7))) != 0 ? 1.0f : -1.0f;
            int rowOff = i * _headDim;
            for (int j = 0; j < _headDim; j++) acc[j] += qi * sketch[rowOff + j];
        }

        float scale = _qjlScale * gamma;
        for (int j = 0; j < _headDim; j++) output[j] += scale * acc[j];
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

    // Builds a fixed d×d row-major standard-normal sketch S for the QJL residual stage. Computed
    // once per codec; the SAME seed reproduces it on decode. Box–Muller over SplitMix64 uniforms.
    private static float[] BuildGaussianSketch(int d, ulong seed)
    {
        var s = new float[(long)d * d];
        ulong state = seed == 0 ? 0x9E3779B97F4A7C15UL : seed;
        for (int idx = 0; idx < s.Length; idx += 2)
        {
            double u1 = 1.0 - NextUnitDouble(ref state); // in (0,1]
            double u2 = NextUnitDouble(ref state);
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            s[idx] = (float)(mag * Math.Cos(2.0 * Math.PI * u2));
            if (idx + 1 < s.Length)
                s[idx + 1] = (float)(mag * Math.Sin(2.0 * Math.PI * u2));
        }
        return s;
    }

    // SplitMix64 → uniform double in [0,1).
    private static double NextUnitDouble(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 11) * (1.0 / 9007199254740992.0); // 53-bit mantissa
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
        int bitPos = coord * _mseBits;
        for (int b = 0; b < _mseBits; b++)
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
        int bitPos = coord * _mseBits;
        int index = 0;
        for (int b = 0; b < _mseBits; b++)
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
