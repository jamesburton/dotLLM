using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Kernels.Experimental;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1623: every kernel family that has instruction-set tiers, every tier the host can run, against the scalar oracle, on
/// shapes that discriminate (block counts that are not multiples of the unroll width, the single-block case, odd tails).
///
/// <para>Tier availability uses the RAW hardware flags (<c>Avx2.IsSupported</c> ...), not <see cref="CpuFeatures"/>: the kernels
/// are called directly, so a <c>DOTLLM_CPU_ISA</c> cap on the test process does not hide a tier the CPU can run. Under
/// <c>DOTNET_EnableAVX2=0</c> and friends the runtime reports the flag false and the tier is skipped (correctly).</para>
///
/// <para>Tolerance. Integer reductions are exact in every tier; tiers differ only in float accumulation order, so the bound is
/// <c>1e-5 * sqrt(blocks) * sum_b |dot of block b|</c> where the per-block dots come from the scalar kernel itself. A dropped or
/// mis-strided block moves the result by a whole block contribution, orders of magnitude above that bound.</para>
/// </summary>
public sealed unsafe class CpuTierMatrixTests
{
    private delegate float DotFn(byte* w, byte* x, int blocks);
    private delegate void QuantFn(float* src, byte* dst, int count);

    private sealed record Tier(string Name, Func<bool> Available, DotFn Fn);

    private sealed record Family(string Name, int WBytes, int XBytes, int ElemsPerBlock, Action<byte[], Random> Fixup,
                                 QuantFn QuantX, DotFn Scalar, Tier[] Tiers, bool AvoidMinus128 = false);

    private static void SetHalf(byte[] b, int off, float v) => BitConverter.TryWriteBytes(b.AsSpan(off, 2), (Half)v);

    private static Action<byte[], Random> Scales(params int[] halfOffsets) => (blk, rng) =>
    {
        foreach (int o in halfOffsets) SetHalf(blk, o, 0.002f + 0.05f * (float)rng.NextDouble());
    };

    private static readonly Dictionary<string, Family> Families = BuildFamilies();

    private static Dictionary<string, Family> BuildFamilies()
    {
        var f = new Dictionary<string, Family>();
        void Add(Family x) => f[x.Name] = x;

        QuantFn q80 = MatMul.QuantizeF32ToQ8_0, q81 = MatMul.QuantizeF32ToQ8_1, q8k = MatMul.QuantizeF32ToQ8_K;
        static bool Ssse3Hw() => Ssse3.IsSupported;
        static bool Avx2Hw() => Avx2.IsSupported;

        Add(new Family("Q8_0", 34, 34, 32, Scales(0), q80, MatMul.VecDotQ8_0Scalar,
        [
            new("Sse", Ssse3Hw, MatMul.VecDotQ8_0Sse),
            new("Avx2", Avx2Hw, MatMul.VecDotQ8_0Avx2),
            new("Avx512", () => Avx512BW.IsSupported && Avx512F.IsSupported, MatMul.VecDotQ8_0Avx512),
            new("Vnni", () => AvxVnni.IsSupported && Fma.IsSupported, MatMul.VecDotQ8_0Vnni),
        ], AvoidMinus128: true));
        Add(new Family("Q4_0", 18, 36, 32, Scales(0), q81, MatMul.VecDotQ4_0Q8_1Scalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ4_0Q8_1Sse), new("Avx2", Avx2Hw, MatMul.VecDotQ4_0Q8_1Avx2)]));
        Add(new Family("Q4_1", 20, 36, 32, Scales(0, 2), q81, MatMul.VecDotQ4_1Q8_1Scalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ4_1Q8_1Sse), new("Avx2", Avx2Hw, MatMul.VecDotQ4_1Q8_1Avx2)]));
        Add(new Family("Q5_1", 24, 36, 32, Scales(0, 2), q81, MatMul.VecDotQ5_1Q8_1Scalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ5_1Q8_1Sse), new("Avx2", Avx2Hw, MatMul.VecDotQ5_1Q8_1Avx2)]));
        Add(new Family("IQ4_NL", 18, 36, 32, Scales(0), q81, MatMul.VecDotIQ4_NLQ8_1Scalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotIQ4_NLQ8_1Sse), new("Avx2", Avx2Hw, MatMul.VecDotIQ4_NLQ8_1Avx2)]));
        Add(new Family("Q5_0", 22, 36, 32, Scales(0), q81, MatMul.VecDotQ5_0Q8_1Scalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ5_0Q8_1Sse), new("Avx2", Avx2Hw, MatMul.VecDotQ5_0Q8_1Avx2)]));

        Add(new Family("Q4_K", 144, 292, 256, Scales(0, 2), q8k, MatMul.VecDotQ4_K_Q8_KScalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ4_K_Q8_KSse), new("Avx2", Avx2Hw, MatMul.VecDotQ4_K_Q8_KAvx2)]));
        Add(new Family("Q5_K", 176, 292, 256, Scales(0, 2), q8k, MatMul.VecDotQ5_K_Q8_KScalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ5_K_Q8_KSse), new("Avx2", Avx2Hw, MatMul.VecDotQ5_K_Q8_KAvx2)]));
        Add(new Family("Q6_K", 210, 292, 256, Scales(208), q8k, MatMul.VecDotQ6_K_Q8_KScalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ6_K_Q8_KSse), new("Avx2", Avx2Hw, MatMul.VecDotQ6_K_Q8_KAvx2)]));
        Add(new Family("Q2_K", 84, 292, 256, Scales(80, 82), q8k, MatMul.VecDotQ2_K_Q8_KScalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ2_K_Q8_KSse), new("Avx2", Avx2Hw, MatMul.VecDotQ2_K_Q8_KAvx2)]));
        Add(new Family("Q3_K", 110, 292, 256, Scales(108), q8k, MatMul.VecDotQ3_K_Q8_KScalar,
            [new("Sse", Ssse3Hw, MatMul.VecDotQ3_K_Q8_KSse), new("Avx2", Avx2Hw, MatMul.VecDotQ3_K_Q8_KAvx2)]));
        return f;
    }

    public static IEnumerable<object[]> DotCases()
    {
        foreach (var fam in Families.Values)
        {
            int[] counts = fam.ElemsPerBlock == 256 ? [1, 2, 3, 5] : [1, 2, 3, 5, 7, 8, 9, 13];
            foreach (var t in fam.Tiers)
                foreach (int n in counts)
                    yield return [fam.Name, t.Name, n];
        }
    }

    /// <summary>Every available tier of every dot-product family agrees with the scalar tier (see class summary for the bound).</summary>
    [SkippableTheory]
    [MemberData(nameof(DotCases))]
    public void Dot_Tier_MatchesScalar(string family, string tier, int blocks)
    {
        Family fam = Families[family];
        Tier t = fam.Tiers.Single(x => x.Name == tier);
        Skip.IfNot(t.Available(), $"{family}/{tier}: ISA not available on this host (or disabled)");

        var rng = new Random(unchecked(StringComparer.Ordinal.GetHashCode(family) * 31 + blocks));
        byte[] wBytes = new byte[fam.WBytes * blocks];
        for (int b = 0; b < blocks; b++)
        {
            byte[] blk = new byte[fam.WBytes];
            rng.NextBytes(blk);
            if (fam.AvoidMinus128) for (int i = 2; i < blk.Length; i++) if (blk[i] == 0x80) blk[i] = 0x81;
            fam.Fixup(blk, rng);
            blk.CopyTo(wBytes, b * fam.WBytes);
        }

        int elems = blocks * fam.ElemsPerBlock;
        float[] x = new float[elems];
        for (int i = 0; i < elems; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);

        byte* xq = (byte*)NativeMemory.AlignedAlloc((nuint)(fam.XBytes * blocks), 64);
        try
        {
            fixed (float* px = x) fam.QuantX(px, xq, elems);
            fixed (byte* w = wBytes)
            {
                float expected = fam.Scalar(w, xq, blocks);
                float actual = t.Fn(w, xq, blocks);

                double mag = 0;
                for (int b = 0; b < blocks; b++) mag += Math.Abs(fam.Scalar(w + b * fam.WBytes, xq + b * fam.XBytes, 1));
                double tol = 5e-5 * Math.Sqrt(blocks) * mag + 1e-12;
                Assert.True(Math.Abs((double)expected - actual) <= tol,
                    $"{family}/{tier} blocks={blocks}: scalar={expected:R} tier={actual:R} |diff|={Math.Abs((double)expected - actual):R} tol={tol:R} [{CpuFeatures.Describe()}]");
                Assert.True(mag > 0, "degenerate fixture: all block dots are zero, the comparison cannot discriminate");
            }
        }
        finally { NativeMemory.AlignedFree(xq); }
    }

    // ──────────────────── Activation quantizers: dispatcher and per-tier entries ────────────────────

    public static IEnumerable<object[]> QuantCases()
    {
        foreach (int n in new[] { 32, 64, 96, 160, 224, 288 }) yield return ["Q8_0", n];
        foreach (int n in new[] { 32, 96, 160 }) yield return ["Q8_1", n];
        foreach (int n in new[] { 256, 512, 768 }) yield return ["Q8_K", n];
    }

    /// <summary>
    /// The public quantizer (which dispatches AVX-512 / AVX2 / scalar) produces the same bytes as the scalar quantizer, and so does
    /// the AVX2 entry when the host has AVX2. Quantization is the one place a tier difference changes WHICH integers are stored.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(QuantCases))]
    public void Quantize_DispatchAndTiers_MatchScalarBytes(string fmt, int count)
    {
        int bytes = fmt switch { "Q8_0" => count / 32 * 34, "Q8_1" => count / 32 * 36, _ => count / 256 * 292 };
        var rng = new Random(count + fmt.Length);
        float[] x = new float[count];
        for (int i = 0; i < count; i++) x[i] = (float)(rng.NextDouble() * 8 - 4);
        if (count >= 64) x[40] = 0f; // a zero in the middle of a block

        byte[] want = new byte[bytes], got = new byte[bytes];
        fixed (float* px = x) fixed (byte* pw = want) fixed (byte* pg = got)
        {
            switch (fmt)
            {
                case "Q8_0":
                    MatMul.QuantizeF32ToQ8_0Scalar(px, pw, count);
                    MatMul.QuantizeF32ToQ8_0(px, pg, count);
                    Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_0 dispatch != scalar n={count} [{CpuFeatures.Describe()}]");
                    if (Avx2.IsSupported)
                    {
                        Array.Clear(got); MatMul.QuantizeF32ToQ8_0Avx2(px, pg, count);
                        Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_0 Avx2 != scalar n={count}");
                    }
                    if (Avx512BW.IsSupported && Avx512F.IsSupported)
                    {
                        Array.Clear(got); MatMul.QuantizeF32ToQ8_0Avx512(px, pg, count);
                        Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_0 Avx512 != scalar n={count}");
                    }
                    break;
                case "Q8_1":
                    MatMul.QuantizeF32ToQ8_1Scalar(px, pw, count);
                    MatMul.QuantizeF32ToQ8_1(px, pg, count);
                    Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_1 dispatch != scalar n={count} [{CpuFeatures.Describe()}]");
                    if (Avx2.IsSupported)
                    {
                        Array.Clear(got); MatMul.QuantizeF32ToQ8_1Avx2(px, pg, count);
                        Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_1 Avx2 != scalar n={count}");
                    }
                    break;
                default:
                    MatMul.QuantizeF32ToQ8_KScalar(px, pw, count);
                    MatMul.QuantizeF32ToQ8_K(px, pg, count);
                    Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_K dispatch != scalar n={count} [{CpuFeatures.Describe()}]");
                    if (Avx2.IsSupported)
                    {
                        Array.Clear(got); MatMul.QuantizeF32ToQ8_KAvx2(px, pg, count);
                        Assert.True(want.AsSpan().SequenceEqual(got), $"Q8_K Avx2 != scalar n={count}");
                    }
                    break;
            }
        }
    }

    // ──────────────────── Dispatcher: Q8_0 GEMV rows under the tier in force ────────────────────

    /// <summary>
    /// <c>ComputeRows</c> picks VNNI / AVX-512 / AVX2 / SSSE3 / scalar from <see cref="CpuFeatures"/>. Whatever tier this process is
    /// in (default, or capped by <c>DOTLLM_CPU_ISA</c> / <c>DOTNET_Enable*</c>), it must agree with the scalar loop; <c>m</c> values
    /// exercise the 4-row unroll and its 1..3 row remainder.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 2)]
    [InlineData(4, 5)]
    [InlineData(5, 7)]
    [InlineData(7, 9)]
    [InlineData(9, 3)]
    public void Q8_0_ComputeRows_TierInForce_MatchesScalar(int m, int blocks)
    {
        var rng = new Random(m * 100 + blocks);
        int rowBytes = blocks * 34;
        byte[] w = new byte[m * rowBytes];
        for (int r = 0; r < m; r++)
            for (int b = 0; b < blocks; b++)
            {
                byte[] blk = new byte[34];
                rng.NextBytes(blk);
                for (int i = 2; i < 34; i++) if (blk[i] == 0x80) blk[i] = 0x81;
                SetHalf(blk, 0, 0.002f + 0.05f * (float)rng.NextDouble());
                blk.CopyTo(w, r * rowBytes + b * 34);
            }
        float[] x = new float[blocks * 32];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);

        byte[] xq = new byte[rowBytes];
        float[] got = new float[m], want = new float[m];
        fixed (float* px = x) fixed (byte* pxq = xq) fixed (byte* pw = w) fixed (float* pg = got)
        {
            MatMul.QuantizeF32ToQ8_0(px, pxq, x.Length);
            MatMul.ComputeRows(pw, pxq, pg, m, blocks);
            for (int r = 0; r < m; r++)
            {
                want[r] = MatMul.VecDotQ8_0Scalar(pw + (long)r * rowBytes, pxq, blocks);
                double mag = 0;
                for (int b = 0; b < blocks; b++) mag += Math.Abs(MatMul.VecDotQ8_0Scalar(pw + (long)r * rowBytes + b * 34, pxq + b * 34, 1));
                Assert.True(Math.Abs((double)want[r] - got[r]) <= 1e-5 * Math.Sqrt(blocks) * mag + 1e-12,
                    $"row {r}/{m} blocks={blocks}: scalar={want[r]:R} dispatched={got[r]:R} [{CpuFeatures.Describe()}]");
            }
        }
    }

    // ──────────────────── LittleBit sign dots: scalar fallback vs SIMD ────────────────────

    /// <summary>
    /// LittleBit's 8-row sign-dots used to throw <see cref="PlatformNotSupportedException"/> without AVX2 / AVX-VNNI; they now fall back
    /// to scalar loops. Integer path: bit-exact. Float path: the same value up to summation order. <c>valid &lt; 8</c> aliases rows.
    /// </summary>
    [SkippableTheory]
    [InlineData(32, 8)]
    [InlineData(64, 5)]
    [InlineData(160, 1)]
    [InlineData(96, 7)]
    public void LittleBit_SignedDot8_Tiers_MatchScalar(int n, int valid)
    {
        var rng = new Random(n * 10 + valid);
        int stride = n / 8;
        byte[] bits = new byte[8 * stride];
        rng.NextBytes(bits);
        float[] v = new float[n];
        sbyte[] a = new sbyte[n];
        for (int i = 0; i < n; i++) { v[i] = (float)(rng.NextDouble() * 2 - 1); a[i] = (sbyte)rng.Next(-127, 128); }

        double sumAbs = v.Sum(e => Math.Abs((double)e));
        fixed (byte* pb = bits) fixed (float* pv = v) fixed (sbyte* pa = a)
        {
            float* fs = stackalloc float[8], fd = stackalloc float[8], fa = stackalloc float[8];
            int* ie = stackalloc int[8], id = stackalloc int[8], ia = stackalloc int[8];

            LittleBitLayer.SignedDot8FScalar(pb, stride, valid, pv, n, fs);
            LittleBitLayer.SignedDot8FDispatch(pb, stride, valid, pv, n, fd);
            for (int k = 0; k < 8; k++)
                Assert.True(Math.Abs(fs[k] - fd[k]) <= 1e-5 * Math.Sqrt(n) * sumAbs, $"F row {k}: scalar={fs[k]} dispatched={fd[k]} [{CpuFeatures.Describe()}]");
            if (Avx2.IsSupported)
            {
                LittleBitLayer.SignedDot8FAvx2(pb, stride, valid, pv, n, fa);
                for (int k = 0; k < 8; k++)
                    Assert.True(Math.Abs(fs[k] - fa[k]) <= 1e-5 * Math.Sqrt(n) * sumAbs, $"F row {k}: scalar={fs[k]} avx2={fa[k]}");
            }

            LittleBitLayer.SignedDot8IScalar(pb, stride, valid, pa, n, ie);
            LittleBitLayer.SignedDot8IDispatch(pb, stride, valid, pa, n, id);
            for (int k = 0; k < 8; k++) Assert.True(ie[k] == id[k], $"I row {k}: scalar={ie[k]} dispatched={id[k]} [{CpuFeatures.Describe()}]");
            if (Avx2.IsSupported && AvxVnni.IsSupported)
            {
                LittleBitLayer.SignedDot8IVnni(pb, stride, valid, pa, n, ia);
                for (int k = 0; k < 8; k++) Assert.True(ie[k] == ia[k], $"I row {k}: scalar={ie[k]} vnni={ia[k]}");
            }

            // The aliasing contract: rows >= valid repeat the last valid row.
            for (int k = valid; k < 8; k++) Assert.True(ie[k] == ie[valid - 1] && fs[k] == fs[valid - 1], $"row {k} should alias row {valid - 1}");
        }
    }

    // ──────────────────── LoRA stage 2 (rank 16): dispatcher and AVX-512 entry vs scalar ────────────────────

    /// <summary>
    /// <c>ApplyF32_R16</c> used to throw <see cref="PlatformNotSupportedException"/> without AVX-512F; it now dispatches to a scalar loop.
    /// <c>outputDim</c> values that are not a multiple of 16 exercise the AVX-512 body's scalar tail, and a non-zero starting <c>y</c>
    /// pins the read-modify-write contract.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 17)]
    [InlineData(2, 32)]
    [InlineData(5, 23)]
    [InlineData(4, 257)]
    public void LoraStage2_R16_DispatchAndAvx512_MatchScalar(int seqLen, int outputDim)
    {
        const int Rank = 16;
        var rng = new Random(seqLen * 1000 + outputDim);
        float[] aT = new float[Rank * outputDim], tmp = new float[seqLen * Rank], y0 = new float[seqLen * outputDim];
        for (int i = 0; i < aT.Length; i++) aT[i] = (float)(rng.NextDouble() * 0.8 - 0.4);
        for (int i = 0; i < tmp.Length; i++) tmp[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < y0.Length; i++) y0[i] = (float)(rng.NextDouble() - 0.5);
        float[] want = (float[])y0.Clone(), viaDispatch = (float[])y0.Clone(), viaAvx = (float[])y0.Clone();

        fixed (float* pa = aT) fixed (float* pt = tmp) fixed (float* pw = want) fixed (float* pd = viaDispatch) fixed (float* pv = viaAvx)
        {
            LoraStage2.ApplyF32_R16Scalar(pa, pt, pw, seqLen, outputDim, 0.5f);
            LoraStage2.ApplyF32_R16(pa, pt, pd, seqLen, outputDim, 0.5f);
            if (Avx512F.IsSupported) LoraStage2.ApplyF32_R16Avx512(pa, pt, pv, seqLen, outputDim, 0.5f);
        }

        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(Math.Abs(want[i] - viaDispatch[i]) <= 1e-5, $"dispatch [{i}] scalar={want[i]} got={viaDispatch[i]} [{CpuFeatures.Describe()}]");
            if (Avx512F.IsSupported) Assert.True(Math.Abs(want[i] - viaAvx[i]) <= 1e-5, $"avx512 [{i}] scalar={want[i]} got={viaAvx[i]}");
        }
        Assert.NotEqual(y0, want); // the kernel actually wrote something
    }
}
