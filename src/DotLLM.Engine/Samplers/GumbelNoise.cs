using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Counter-based Gumbel noise for position-seeded Gumbel-max sampling (#1358).
/// </summary>
/// <remarks>
/// <para>
/// The noise for vocabulary entry <c>i</c> of the token at generation index <c>n</c> is a pure function
/// <c>G(seed, n, i)</c> (Philox4x32-10, key = seed, counter = (i, n, 0, 0)), so a token drawn as
/// <c>argmax_i (prepared_logit_i + G(seed, n, i))</c> is a deterministic function of (context, seed, n). Plain decoding and every
/// row of a speculative verify batch therefore draw the same token whenever their contexts agree, which makes the speculative
/// output identical to the plain output for a fixed seed, and the draw is exact: Gumbel-max over the prepared logits is a sample
/// from softmax(prepared) (entries masked to -inf by top-k / top-p / min-p / constraints can never win).
/// </para>
/// <para>
/// <b>The uniform is 24-bit.</b> <c>u = max(w &gt;&gt; 8, 1) * 2^-24</c> lies in <c>[2^-24, 1 - 2^-24]</c> and is exactly representable in
/// an F32, so <c>-log(-log(u))</c> is always finite. Converting the full 32-bit word to F32 rounds the top ~128 values up to 2^32, i.e.
/// <c>u = 1</c>, whose noise is <c>+inf</c>: that token wins every time (a foreign-script token about once in 200-400 draws at a
/// 150k vocabulary). <see cref="Uniform32Unsafe"/> is that broken form, kept only so a test can prove it catches the bug.
/// </para>
/// <para>The same arithmetic (integer Philox, 24-bit uniform, the deterministic F32 <see cref="Log"/>) is what the Vulkan kernel (#1382) reproduces bit for bit.</para>
/// </remarks>
public static class GumbelNoise
{
    private const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    private const uint W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;

    /// <summary>First 32-bit Philox4x32-10 output word for counter (<paramref name="index"/>, <paramref name="tokenIndex"/>, 0, 0) and key <paramref name="seed"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Word(ulong seed, int tokenIndex, int index)
    {
        uint c0 = (uint)index, c1 = (uint)tokenIndex, c2 = 0, c3 = 0;
        uint k0 = (uint)seed, k1 = (uint)(seed >> 32);
        for (int r = 0; r < 10; r++)
        {
            ulong p0 = (ulong)M0 * c0, p1 = (ulong)M1 * c2;
            uint n0 = (uint)(p1 >> 32) ^ c1 ^ k0;
            uint n1 = (uint)p1;
            uint n2 = (uint)(p0 >> 32) ^ c3 ^ k1;
            uint n3 = (uint)p0;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            k0 += W0; k1 += W1;
        }
        return c0;
    }

    /// <summary>24-bit uniform in <c>[2^-24, 1 - 2^-24]</c>, exactly representable in F32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Uniform24(uint word)
    {
        uint m = word >> 8;
        if (m < 1) m = 1;
        return m * (1.0f / 16777216.0f);
    }

    /// <summary>The BROKEN 32-bit conversion (can return exactly 1.0). Test mutant only; never use for sampling.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Uniform32Unsafe(uint word) => word * (1.0f / 4294967296.0f);

    // Degree-9 minimax fit of log1p(f)/f on [sqrt(1/2)-1, sqrt(2)-1] (max relative error 1.2e-7 in F32). Shared, bit for bit, with
    // native/vulkan/shaders/gumbel_argmax_f32.comp.
    private const float P0 = 1.0f, P1 = -0.49999988f, P2 = 0.33333325f, P3 = -0.25001585f, P4 = 0.20002072f,
        P5 = -0.16609126f, P6 = 0.14180080f, P7 = -0.13240842f, P8 = 0.12914170f, P9 = -0.07638592f;
    private const float Ln2Hi = 6.9313812256e-01f, Ln2Lo = 9.0580006145e-06f, Sqrt2 = 1.41421356f;

    /// <summary>
    /// Deterministic natural logarithm of a positive normal F32 built only from IEEE add/multiply and bit operations (exponent split,
    /// <c>f = m - 1</c> exact, a fixed degree-9 polynomial, fixed order, no FMA, no division), so a device kernel that does the same
    /// operations in the same order returns the same bits. <c>MathF.Log</c> and GLSL <c>log</c> both differ from each other by a few ULP
    /// (GLSL only promises an absolute error near 1), which near u = 1 is several times the value and would flip the argmax between the
    /// CPU (plain decode) and the GPU (verify). Relative error ~1.2e-7. Non-positive input returns -inf / NaN; subnormals are not supported.
    /// </summary>
    public static float Log(float x)
    {
        if (!(x > 0f)) return x == 0f ? float.NegativeInfinity : float.NaN;
        uint bits = BitConverter.SingleToUInt32Bits(x);
        int e = (int)(bits >> 23) - 127;
        float m = BitConverter.UInt32BitsToSingle((bits & 0x007FFFFFu) | 0x3F800000u);   // [1, 2)
        if (m > Sqrt2) { m *= 0.5f; e++; }                                                // [sqrt(1/2), sqrt(2))
        float f = m - 1.0f;                                                                // exact (Sterbenz)
        float acc = P9;
        acc = acc * f + P8; acc = acc * f + P7; acc = acc * f + P6; acc = acc * f + P5;
        acc = acc * f + P4; acc = acc * f + P3; acc = acc * f + P2; acc = acc * f + P1; acc = acc * f + P0;
        float l1p = f * acc;
        float de = e;
        return de * Ln2Hi + (l1p + de * Ln2Lo);
    }

    /// <summary>Standard Gumbel(0,1) variate for a uniform in (0, 1]: <c>-log(-log(u))</c> (+inf at u = 1), via the deterministic <see cref="Log"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FromUniform(float u) => -Log(-Log(u));

    /// <summary>Gumbel noise <c>G(seed, tokenIndex, index)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Noise(ulong seed, int tokenIndex, int index) => FromUniform(Uniform24(Word(seed, tokenIndex, index)));

    /// <summary>
    /// Gumbel-max draw over already prepared logits: <c>argmax_i (logits[i] + G(seed, tokenIndex, i))</c> over the finite entries
    /// (lowest index wins an exact tie). Falls back to the plain argmax when no entry is finite.
    /// </summary>
    /// <param name="preparedLogits">Output of <c>SamplerPipeline.PrepareDistribution</c> (temperature, top-k/p, min-p, penalties, constraint mask applied; excluded entries are -inf).</param>
    /// <param name="seed">Request noise seed.</param>
    /// <param name="tokenIndex">Generation index of the token being drawn.</param>
    public static int Sample(ReadOnlySpan<float> preparedLogits, ulong seed, int tokenIndex)
        => Sample(preparedLogits, seed, tokenIndex, unsafeUniform32: false);

    internal static int Sample(ReadOnlySpan<float> preparedLogits, ulong seed, int tokenIndex, bool unsafeUniform32)
    {
        int best = -1;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < preparedLogits.Length; i++)
        {
            float l = preparedLogits[i];
            if (float.IsNegativeInfinity(l)) continue;
            uint w = Word(seed, tokenIndex, i);
            float g = FromUniform(unsafeUniform32 ? Uniform32Unsafe(w) : Uniform24(w));
            float s = l + g;
            if (best < 0 || s > bestScore) { best = i; bestScore = s; }
        }
        return best >= 0 ? best : CategoricalSampler.ArgMax(preparedLogits);
    }
}
