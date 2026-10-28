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
/// <para>The same arithmetic (integer Philox, 24-bit uniform, F32 <c>log</c>) is what a device kernel must reproduce.</para>
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

    /// <summary>Standard Gumbel(0,1) variate for a uniform in (0, 1]: <c>-log(-log(u))</c> (+inf at u = 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FromUniform(float u) => -MathF.Log(-MathF.Log(u));

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
