using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// CPU kernels of the OVHcloud Olala architecture (issue #1970): the geodesic residual update, the Mamba-3 MIMO single-token
/// recurrence (F32 state, see <c>OlalaStatePolicy</c>), and small element-wise helpers. Semantics follow HF <c>modeling_olala.py</c>
/// (<c>OlalaGeodesicNorm.forward</c>, <c>_mamba3_step_torch</c>, <c>_mamba3_rotary_step_torch</c>); correctness first, no GEMM here.
/// </summary>
public static class OlalaOps
{
    /// <summary>Largest rotation angle of the geodesic update (pi / 4).</summary>
    public const float GeodesicClamp = MathF.PI / 4f;

    /// <summary>Floor on the (negative) SSM decay rate: <c>A = min(-softplus(a), -AFloor)</c>.</summary>
    public const float AFloor = 1e-4f;

    /// <summary>
    /// <c>log(1 + exp(x))</c> with PyTorch's linear threshold at 20. Evaluated in double (PyTorch uses <c>log1p(exp(x))</c>): in float,
    /// <c>1 + exp(x)</c> rounds away the small term, so <c>dt = softplus(dt_raw + dt_bias)</c> with Olala's <c>dt_bias</c> down to -8.6 lost
    /// 1e-4..1e-1 relative precision, and returned exactly 0 below x = -17. Below -20 exp(x) itself is exact to ~exp(x)/2 relative, where
    /// 1 + exp(x) in double would start losing digits.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Softplus(float x) => x > 20f ? x : x < -20f ? MathF.Exp(x) : (float)Math.Log(1.0 + Math.Exp(x));

    /// <summary>Logistic sigmoid.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary><c>x * sigmoid(x)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    /// <summary>
    /// <c>F.normalize(x, dim=-1)</c>: <c>x / max(||x||, eps)</c> in place. The token embedding row of Olala is normalised to unit length
    /// before block 0 (<c>normalize_embeddings_ngpt</c>), which also fixes the residual norm at 1 (the geodesic update is a rotation).
    /// </summary>
    public static void L2Normalize(Span<float> x, float eps = 1e-12f)
    {
        double ss = 0;
        for (int i = 0; i < x.Length; i++) ss += (double)x[i] * x[i];
        float norm = MathF.Max((float)Math.Sqrt(ss), eps);
        TensorPrimitives.Divide(x, norm, x);
    }

    /// <summary>
    /// <c>OlalaGeodesicNorm.forward</c> for one row: rotates the residual <paramref name="x"/> towards the component of the mixer/MLP
    /// output <paramref name="g"/> that is orthogonal to it, <c>out = x cos(t) + unit(g_perp) * |x| * sin(t)</c> with
    /// <c>t = min((min(|g_perp| / |x|, pi/4) * scale + bias) / (layer + 1), pi/4)</c>. Reductions run in double: <paramref name="g"/> carries
    /// activation outliers up to ~1e14 on this model, whose squares overflow nothing in F32 but whose sums must not lose the unit tangent.
    /// <paramref name="output"/> may alias <paramref name="x"/>.
    /// </summary>
    /// <param name="x">Residual row (length = hidden).</param>
    /// <param name="g">Mixer or MLP output row (same length).</param>
    /// <param name="output">Result row; may alias <paramref name="x"/>.</param>
    /// <param name="scale">Learned scalar <c>geodesic_*.scale</c>.</param>
    /// <param name="bias">Learned scalar <c>geodesic_*.bias</c>.</param>
    /// <param name="layerIndex">Zero-based block index (the angle is divided by <c>layerIndex + 1</c>).</param>
    public static void GeodesicUpdate(ReadOnlySpan<float> x, ReadOnlySpan<float> g, Span<float> output,
                                      float scale, float bias, int layerIndex)
    {
        int n = x.Length;
        if (g.Length != n || output.Length != n) throw new ArgumentException("x, g and output must have equal length.");
        double xx = 0, xg = 0;
        for (int i = 0; i < n; i++) { xx += (double)x[i] * x[i]; xg += (double)x[i] * g[i]; }
        double xNormSq = Math.Max(xx, 1e-12);
        double proj = xg / xNormSq;
        double tt = 0;
        for (int i = 0; i < n; i++) { double d = g[i] - proj * x[i]; tt += d * d; }
        double tangentNorm = Math.Sqrt(tt);
        double safeTangent = Math.Max(tangentNorm, 1e-8);
        double r = Math.Sqrt(xx);
        double safeR = Math.Max(r, 1e-6);
        double theta = Math.Min(safeTangent / safeR, GeodesicClamp);
        theta = Math.Min((theta * scale + bias) / (layerIndex + 1), GeodesicClamp);
        double cos = Math.Cos(theta), sin = Math.Sin(theta);
        double tangentCoeff = safeR * sin / safeTangent;
        for (int i = 0; i < n; i++)
        {
            double d = g[i] - proj * x[i];
            output[i] = (float)(x[i] * cos + d * tangentCoeff);
        }
    }

    /// <summary>
    /// Per-head RMSNorm of consecutive <paramref name="headDim"/> slices with a shared gamma (zero-centred gamma is already folded into
    /// the stored weight by the GGUF converter): <c>y = x / sqrt(mean(x^2) + eps) * gamma</c>, in place.
    /// </summary>
    public static void RmsNormSlices(Span<float> x, ReadOnlySpan<float> gamma, int headDim, float eps)
    {
        if (gamma.Length != headDim || x.Length % headDim != 0) throw new ArgumentException("Slice geometry mismatch.");
        for (int o = 0; o < x.Length; o += headDim)
        {
            var s = x.Slice(o, headDim);
            RmsNorm.Execute(s, gamma, eps, s);
        }
    }

    /// <summary>Round-to-nearest-even conversion of <paramref name="v"/> to BF16 bits (NaN stays NaN).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort ToBf16(float v)
    {
        uint b = BitConverter.SingleToUInt32Bits(v);
        if ((b & 0x7fffffffu) > 0x7f800000u) return (ushort)((b >> 16) | 0x40);
        b += 0x7fffu + ((b >> 16) & 1u);
        return (ushort)(b >> 16);
    }

    /// <summary>Expands BF16 bits to <see cref="float"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FromBf16(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    /// <summary>Geometry and per-layer constants of one Mamba-3 MIMO mixer (Olala 7A1B: H 48, P 64, N 128, R 4, 32 angles).</summary>
    /// <param name="Heads">SSM heads H.</param>
    /// <param name="HeadDim">Per-head value width P.</param>
    /// <param name="State">State size N.</param>
    /// <param name="Rank">MIMO rank R.</param>
    /// <param name="Angles">Rotary angle pairs (dims i and i + N/2 for i &lt; Angles).</param>
    /// <param name="NormEps">RMSNorm epsilon of the B/C norms.</param>
    public readonly record struct Mamba3Geometry(int Heads, int HeadDim, int State, int Rank, int Angles, float NormEps)
    {
        /// <summary>Width of the per-head interleaved projection <c>[z(P), x(P), dt, A, trap]</c>.</summary>
        public int PerHead => 2 * HeadDim + 3;

        /// <summary>Width of the dynamic projection <c>[B(R*N), C(R*N), angles]</c> (one group).</summary>
        public int DynWidth => 2 * Rank * State + Angles;

        /// <summary>Elements of the SSM matrix <c>[H][P][N]</c>.</summary>
        public int SsmElements => Heads * HeadDim * State;
    }

    /// <summary>Learned per-layer constants of one Mamba-3 MIMO mixer, all F32.</summary>
    /// <param name="BNorm">B RMSNorm gamma, <c>[N]</c> (1 + w already folded).</param>
    /// <param name="CNorm">C RMSNorm gamma, <c>[N]</c>.</param>
    /// <param name="BBias">K bias <c>[H][R][N]</c>.</param>
    /// <param name="CBias">Q bias <c>[H][R][N]</c>.</param>
    /// <param name="DtBias">dt bias <c>[H]</c>.</param>
    /// <param name="D">Skip <c>[H]</c>.</param>
    /// <param name="MimoX">x up-projection <c>[H][R][P]</c>.</param>
    /// <param name="MimoZ">z gate projection <c>[H][R][P]</c>.</param>
    /// <param name="MimoO">Output down-projection <c>[H][R][P]</c>.</param>
    public sealed record Mamba3Params(float[] BNorm, float[] CNorm, float[] BBias, float[] CBias, float[] DtBias, float[] D,
                                      float[] MimoX, float[] MimoZ, float[] MimoO);

    /// <summary>
    /// One decode/prefill step of the Mamba-3 MIMO mixer for a single token (HF <c>_mamba3_step_torch</c> +
    /// <c>_mamba3_rotary_step_torch</c>; the same recurrence the chunked TileLang kernel evaluates):
    /// <code>
    ///   h_t = a_t h_{t-1} + b_t u_{t-1} + g_t u_t,   a = exp(A dt), b = (1 - trap) dt a, g = trap dt
    ///   u_t = sum_r k_{t,r} (x_t * MimoX_r)^T,       o_r = h_t q_r + D (x * MimoX_r)
    ///   y   = sum_r MimoO_r * o_r * silu(z * MimoZ_r)
    /// </code>
    /// with <c>q, k</c> the RMS-normed, biased C and B rotated by the cumulative data-dependent angle. All state buffers are F32
    /// (<c>OlalaStatePolicy</c>) and updated in place; the previous token's rotated K and raw x are carried for the trapezoid term.
    /// </summary>
    /// <param name="g">Geometry.</param>
    /// <param name="p">Learned constants.</param>
    /// <param name="zxdt">This token's <c>ssm_in</c> output, <c>[H * (2P+3)]</c>, per head <c>[z, x, dt, A, trap]</c>.</param>
    /// <param name="dyn">This token's <c>ssm_in_dyn</c> output <c>[2RN + Angles]</c>: B <c>[R][N]</c>, C <c>[R][N]</c>, raw angles.</param>
    /// <param name="ssm">SSM matrix <c>[H][P][N]</c>, updated in place.</param>
    /// <param name="angle">Wrapped rotary phase <c>[H][Angles]</c>, updated in place.</param>
    /// <param name="kState">Previous token's rotated K <c>[R][H][N]</c>, updated in place.</param>
    /// <param name="vState">Previous token's raw x <c>[H][P]</c>, updated in place.</param>
    /// <param name="y">Output <c>[H * P]</c> (head-major).</param>
    /// <param name="pool">Optional compute pool; heads are partitioned across its workers (no per-call allocation). Null runs serially.</param>
    public static void Mamba3MimoStep(in Mamba3Geometry g, Mamba3Params p, ReadOnlySpan<float> zxdt, ReadOnlySpan<float> dyn,
                                      Span<float> ssm, Span<float> angle, Span<float> kState, Span<float> vState,
                                      Span<float> y, DotLLM.Cpu.Threading.ComputeThreadPool? pool = null)
    {
        int H = g.Heads, P = g.HeadDim, N = g.State, R = g.Rank, A = g.Angles, ph = g.PerHead;
        if (zxdt.Length < H * ph || dyn.Length < g.DynWidth || ssm.Length < g.SsmElements || angle.Length < H * A
            || kState.Length < R * H * N || vState.Length < H * P || y.Length < H * P)
            throw new ArgumentException("Mamba3MimoStep buffer too small.");

        // B/C RMSNorm (per rank row, shared across heads) and the raw angle activation (shared across heads).
        Span<float> bn = stackalloc float[R * N], cn = stackalloc float[R * N], tanhA = stackalloc float[A];
        for (int r = 0; r < R; r++)
        {
            RmsNorm.Execute(dyn.Slice(r * N, N), p.BNorm, g.NormEps, bn.Slice(r * N, N));
            RmsNorm.Execute(dyn.Slice(R * N + r * N, N), p.CNorm, g.NormEps, cn.Slice(r * N, N));
        }
        for (int a = 0; a < A; a++) tanhA[a] = MathF.Tanh(dyn[2 * R * N + a]);

        unsafe
        {
            fixed (float* zp = zxdt, bnp = bn, cnp = cn, tp = tanhA, sp = ssm, ap = angle, kp = kState, vp = vState, yp = y)
            fixed (float* dtb = p.DtBias, dd = p.D, mx = p.MimoX, mz = p.MimoZ, mo = p.MimoO, bb = p.BBias, cb = p.CBias)
            {
                var ctx = new StepCtx(g, zp, bnp, cnp, tp, sp, ap, kp, vp, yp, dtb, dd, mx, mz, mo, bb, cb);
                if (pool is not null && pool.ThreadCount > 1 && H > 1)
                    pool.Dispatch((nint)(&ctx), &HeadWorker);
                else
                    for (int h = 0; h < H; h++) HeadStep(in ctx, h);
            }
        }
    }

    private readonly unsafe struct StepCtx(Mamba3Geometry g, float* zxdt, float* bn, float* cn, float* tanhA,
                                           float* ssm, float* angle, float* kState, float* vState, float* y,
                                           float* dtBias, float* d, float* mimoX, float* mimoZ, float* mimoO, float* bBias, float* cBias)
    {
        public readonly Mamba3Geometry G = g;
        public readonly float* Zxdt = zxdt, Bn = bn, Cn = cn, TanhA = tanhA, Ssm = ssm, Angle = angle, KState = kState, VState = vState, Y = y;
        public readonly float* DtBias = dtBias, D = d, MimoX = mimoX, MimoZ = mimoZ, MimoO = mimoO, BBias = bBias, CBias = cBias;
    }

    private static unsafe void HeadWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref readonly var ctx = ref Unsafe.AsRef<StepCtx>((void*)ctxPtr);
        int H = ctx.G.Heads;
        int per = (H + threadCount - 1) / threadCount;
        int start = threadIdx * per, end = Math.Min(H, start + per);
        for (int h = start; h < end; h++) HeadStep(in ctx, h);
    }

    [SkipLocalsInit]
    private static unsafe void HeadStep(in StepCtx c, int h)
    {
        var g = c.G;
        int H = g.Heads, P = g.HeadDim, N = g.State, R = g.Rank, A = g.Angles;
        float* z = c.Zxdt + h * g.PerHead;
        float* x = z + P;
        float dtRaw = z[2 * P], aRaw = z[2 * P + 1], trapRaw = z[2 * P + 2];

        float decayRate = MathF.Min(-Softplus(aRaw), -AFloor);
        float dt = Softplus(dtRaw + c.DtBias[h]);
        float trap = Sigmoid(trapRaw);
        float alpha = MathF.Exp(decayRate * dt);
        float beta = (1f - trap) * dt * alpha;
        float gamma = trap * dt;

        // Cumulative rotary phase, wrapped to [0, 2pi) like the reference state (angle_state = remainder(angle, 2pi)).
        float* ang = c.Angle + h * A;
        Span<float> cos = stackalloc float[A], sin = stackalloc float[A];
        for (int a = 0; a < A; a++)
        {
            double v = ang[a] + (double)c.TanhA[a] * dt * Math.PI;
            v -= Math.Floor(v / (2 * Math.PI)) * (2 * Math.PI);
            ang[a] = (float)v;
            cos[a] = MathF.Cos((float)v);
            sin[a] = MathF.Sin((float)v);
        }

        // q = rot(C_norm + C_bias), k = rot(B_norm + B_bias), per rank.
        Span<float> q = stackalloc float[R * N], k = stackalloc float[R * N];
        int half = N / 2;
        for (int r = 0; r < R; r++)
        {
            for (int n = 0; n < N; n++)
            {
                q[r * N + n] = c.Cn[r * N + n] + c.CBias[(h * R + r) * N + n];
                k[r * N + n] = c.Bn[r * N + n] + c.BBias[(h * R + r) * N + n];
            }
            for (int a = 0; a < A; a++)
            {
                int i0 = r * N + a, i1 = r * N + half + a;
                float q0 = q[i0], q1 = q[i1], k0 = k[i0], k1 = k[i1];
                q[i0] = q0 * cos[a] - q1 * sin[a]; q[i1] = q0 * sin[a] + q1 * cos[a];
                k[i0] = k0 * cos[a] - k1 * sin[a]; k[i1] = k0 * sin[a] + k1 * cos[a];
            }
        }

        // kPrev[r][n] for this head (state layout [R][H][N]).
        Span<float> kPrev = stackalloc float[R * N];
        for (int r = 0; r < R; r++)
            for (int n = 0; n < N; n++) kPrev[r * N + n] = c.KState[(r * H + h) * N + n];

        float d = c.D[h];
        Span<float> o = stackalloc float[R], xv = stackalloc float[R], xs = stackalloc float[R];
        Span<float> coef = stackalloc float[N];
        for (int pIdx = 0; pIdx < P; pIdx++)
        {
            float xp = x[pIdx], zp = z[pIdx], vPrev = c.VState[h * P + pIdx];
            for (int r = 0; r < R; r++)
            {
                float mx = c.MimoX[(h * R + r) * P + pIdx];
                xv[r] = xp * mx;
                xs[r] = vPrev * mx;
            }
            float* s = c.Ssm + ((long)h * P + pIdx) * N;
            // s = s * alpha + sum_r (xv_r gamma k_r + xs_r beta kPrev_r);  o_r = s . q_r
            for (int r = 0; r < R; r++) o[r] = 0f;
            int n0 = 0;
            int vw = Vector<float>.Count;
            for (; n0 + vw <= N; n0 += vw)
            {
                var sv = new Vector<float>(new ReadOnlySpan<float>(s + n0, vw)) * alpha;
                for (int r = 0; r < R; r++)
                {
                    sv += new Vector<float>(k.Slice(r * N + n0, vw)) * (xv[r] * gamma)
                        + new Vector<float>(kPrev.Slice(r * N + n0, vw)) * (xs[r] * beta);
                }
                sv.CopyTo(new Span<float>(s + n0, vw));
                for (int r = 0; r < R; r++) o[r] += Vector.Dot(sv, new Vector<float>(q.Slice(r * N + n0, vw)));
            }
            for (; n0 < N; n0++)
            {
                float sv = s[n0] * alpha;
                for (int r = 0; r < R; r++) sv += xv[r] * gamma * k[r * N + n0] + xs[r] * beta * kPrev[r * N + n0];
                s[n0] = sv;
                for (int r = 0; r < R; r++) o[r] += sv * q[r * N + n0];
            }
            float acc = 0f;
            for (int r = 0; r < R; r++)
            {
                float mz = c.MimoZ[(h * R + r) * P + pIdx];
                float mo = c.MimoO[(h * R + r) * P + pIdx];
                acc += (o[r] + xv[r] * d) * Silu(zp * mz) * mo;
            }
            c.Y[h * P + pIdx] = acc;
        }

        // Trapezoid boundary: this token's rotated K and raw x feed the next token's b-term.
        for (int r = 0; r < R; r++)
            for (int n = 0; n < N; n++) c.KState[(r * H + h) * N + n] = k[r * N + n];
        for (int pIdx = 0; pIdx < P; pIdx++) c.VState[h * P + pIdx] = x[pIdx];
    }
}
