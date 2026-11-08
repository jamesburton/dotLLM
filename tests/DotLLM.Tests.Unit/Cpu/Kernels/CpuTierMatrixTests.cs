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
/// <para>Tier availability uses the RAW hardware flags (<c>Avx2.IsSupported</c> ...): the kernels are called directly, so a
/// <c>DOTLLM_CPU_ISA</c> cap on the test process cannot change which kernel runs. A tier ABOVE the cap is therefore skipped, not
/// compared (#1692): the pair would be exactly the one the uncapped lane already checks on the same data, and a failure message
/// saying "cap=Scalar" next to "tier Avx2" reads as a dispatch bug when it is not one. Under <c>DOTNET_EnableAVX2=0</c> and
/// friends the runtime reports the flag false and the tier is skipped too.</para>
///
/// <para>Tolerance (#1692). Integer reductions are exact in every tier; tiers differ only in WHERE the float rounding happens
/// (the scalar oracle rounds once per sub-block term; a SIMD tier may integer-accumulate an unsigned form and subtract a separate
/// bias term, two large values that nearly cancel). Each rounding is relative to the magnitude of the term it sits on, never to
/// the final result, so the bound is derived from the operands: <c>ulp(1) * sqrt(K) * sum_i |w_i * x_i|</c> over the K dequantized
/// weight/activation pairs, plus an absolute floor. A dot whose result nearly cancels keeps a bound proportional to what was
/// summed. A dropped or mis-decoded sub-block, block or element moves the result by whole-term amounts, orders of magnitude above
/// that bound; see the mutants in the PR body.</para>
///
/// <para>The fixture is seeded from the family NAME and block count through <see cref="StableSeed"/>, a fixed FNV-1a hash, not
/// <c>string.GetHashCode</c>, which .NET randomises per process: that is why the #1692 failures appeared on unrelated PRs and did
/// not reproduce. Set <c>DOTLLM_TIER_SWEEP_SEEDS=N</c> to run <see cref="Dot_Tier_SeedSweep_WorstRatio"/> over N seeds per case
/// and report the worst |diff|/bound seen.</para>
/// </summary>
public sealed unsafe class CpuTierMatrixTests
{
    private delegate float DotFn(byte* w, byte* x, int blocks);
    private delegate void QuantFn(float* src, byte* dst, int count);
    private delegate void DequantFn(nint src, long count, Span<float> dest);

    private sealed record Tier(string Name, CpuIsaTier IsaTier, Func<bool> Available, DotFn Fn);

    private sealed record Family(string Name, int WBytes, int XBytes, int ElemsPerBlock, Action<byte[], Random> Fixup,
                                 QuantFn QuantX, DequantFn DequantW, DotFn Scalar, Tier[] Tiers, bool AvoidMinus128 = false);

    /// <summary>Spacing of floats at 1 (2^-23): one unit of relative rounding on a float term.</summary>
    private const double FloatUlp = 1.1920928955078125e-7;

    /// <summary>Absolute floor of the bound, for fixtures whose every product is tiny.</summary>
    private const double AbsFloor = 1e-12;

    /// <summary>FNV-1a over the UTF-16 code units: the same value in every process, unlike <c>string.GetHashCode</c>.</summary>
    private static int StableSeed(string family, int blocks, int salt = 0)
    {
        uint h = 2166136261;
        foreach (char c in family) { h ^= c; h *= 16777619; }
        h ^= (uint)blocks; h *= 16777619;
        h ^= (uint)salt; h *= 16777619;
        return unchecked((int)h);
    }

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
        const CpuIsaTier V128 = CpuIsaTier.Vector128, V256 = CpuIsaTier.Avx2, V512 = CpuIsaTier.Avx512;

        Add(new Family("Q8_0", 34, 34, 32, Scales(0), q80, Dequantize.DequantizeQ8_0Scalar, MatMul.VecDotQ8_0Scalar,
        [
            new("Sse", V128, Ssse3Hw, MatMul.VecDotQ8_0Sse),
            new("Avx2", V256, Avx2Hw, MatMul.VecDotQ8_0Avx2),
            new("Avx512", V512, () => Avx512BW.IsSupported && Avx512F.IsSupported, MatMul.VecDotQ8_0Avx512),
            new("Vnni", V256, () => AvxVnni.IsSupported && Fma.IsSupported, MatMul.VecDotQ8_0Vnni),
        ], AvoidMinus128: true));
        Add(new Family("Q4_0", 18, 36, 32, Scales(0), q81, Dequantize.DequantizeQ4_0Scalar, MatMul.VecDotQ4_0Q8_1Scalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ4_0Q8_1Sse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ4_0Q8_1Avx2)]));
        Add(new Family("Q4_1", 20, 36, 32, Scales(0, 2), q81, Dequantize.DequantizeQ4_1Scalar, MatMul.VecDotQ4_1Q8_1Scalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ4_1Q8_1Sse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ4_1Q8_1Avx2)]));
        Add(new Family("Q5_1", 24, 36, 32, Scales(0, 2), q81, Dequantize.DequantizeQ5_1Scalar, MatMul.VecDotQ5_1Q8_1Scalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ5_1Q8_1Sse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ5_1Q8_1Avx2)]));
        Add(new Family("IQ4_NL", 18, 36, 32, Scales(0), q81, Dequantize.DequantizeIQ4_NL, MatMul.VecDotIQ4_NLQ8_1Scalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotIQ4_NLQ8_1Sse), new("Avx2", V256, Avx2Hw, MatMul.VecDotIQ4_NLQ8_1Avx2)]));
        Add(new Family("Q5_0", 22, 36, 32, Scales(0), q81, Dequantize.DequantizeQ5_0Scalar, MatMul.VecDotQ5_0Q8_1Scalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ5_0Q8_1Sse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ5_0Q8_1Avx2)]));

        Add(new Family("Q4_K", 144, 292, 256, Scales(0, 2), q8k, Dequantize.DequantizeQ4_KScalar, MatMul.VecDotQ4_K_Q8_KScalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ4_K_Q8_KSse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ4_K_Q8_KAvx2)]));
        Add(new Family("Q5_K", 176, 292, 256, Scales(0, 2), q8k, Dequantize.DequantizeQ5_KScalar, MatMul.VecDotQ5_K_Q8_KScalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ5_K_Q8_KSse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ5_K_Q8_KAvx2)]));
        Add(new Family("Q6_K", 210, 292, 256, Scales(208), q8k, Dequantize.DequantizeQ6_KScalar, MatMul.VecDotQ6_K_Q8_KScalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ6_K_Q8_KSse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ6_K_Q8_KAvx2)]));
        Add(new Family("Q2_K", 84, 292, 256, Scales(80, 82), q8k, Dequantize.DequantizeQ2_K, MatMul.VecDotQ2_K_Q8_KScalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ2_K_Q8_KSse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ2_K_Q8_KAvx2)]));
        Add(new Family("Q3_K", 110, 292, 256, Scales(108), q8k, Dequantize.DequantizeQ3_KScalar, MatMul.VecDotQ3_K_Q8_KScalar,
            [new("Sse", V128, Ssse3Hw, MatMul.VecDotQ3_K_Q8_KSse), new("Avx2", V256, Avx2Hw, MatMul.VecDotQ3_K_Q8_KAvx2)]));
        return f;
    }

    /// <summary>
    /// Dequantizes the activation side exactly as the kernels read it: Q8_0 <c>half d@0, qs@2</c>; Q8_1 <c>half d@0, half s@2, qs@4</c>;
    /// Q8_K <c>float d@0, qs@4</c>. The integer values are what every tier multiplies, so <c>|w_i x_i|</c> built from them is the
    /// magnitude every tier actually summed.
    /// </summary>
    private static void DequantX(Family fam, byte* xq, int blocks, Span<float> dest)
    {
        for (int b = 0; b < blocks; b++)
        {
            byte* blk = xq + b * fam.XBytes;
            float d; sbyte* qs;
            switch (fam.XBytes)
            {
                case 34: d = (float)BitConverter.ToHalf(new ReadOnlySpan<byte>(blk, 2)); qs = (sbyte*)(blk + 2); break;
                case 36: d = (float)BitConverter.ToHalf(new ReadOnlySpan<byte>(blk, 2)); qs = (sbyte*)(blk + 4); break;
                default: d = BitConverter.ToSingle(new ReadOnlySpan<byte>(blk, 4)); qs = (sbyte*)(blk + 4); break;
            }
            for (int i = 0; i < fam.ElemsPerBlock; i++) dest[b * fam.ElemsPerBlock + i] = d * qs[i];
        }
    }

    private readonly record struct DotOutcome(float Scalar, float Tier, double Diff, double Bound, double SumAbsProducts);

    /// <summary>Builds the seeded fixture, runs the scalar oracle and the tier kernel, and returns the operand-derived bound.</summary>
    private static DotOutcome RunDot(Family fam, Tier t, int blocks, int seed)
    {
        var rng = new Random(seed);
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

                float[] wf = new float[elems], xf = new float[elems];
                fam.DequantW((nint)w, elems, wf);
                DequantX(fam, xq, blocks, xf);
                double sumAbs = 0;
                for (int i = 0; i < elems; i++) sumAbs += Math.Abs((double)wf[i] * xf[i]);

                double bound = FloatUlp * Math.Sqrt(elems) * sumAbs + AbsFloor;
                return new DotOutcome(expected, actual, Math.Abs((double)expected - actual), bound, sumAbs);
            }
        }
        finally { NativeMemory.AlignedFree(xq); }
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

    private static void SkipUnlessTierRunnable(string family, Tier t)
    {
        Skip.IfNot(t.Available(), $"{family}/{t.Name}: ISA not available on this host (or disabled by DOTNET_Enable*)");
        Skip.If(CpuFeatures.Cap < t.IsaTier,
            $"{family}/{t.Name}: tier {t.IsaTier} is above the {CpuFeatures.EnvVar} cap ({CpuFeatures.Cap}). The kernel is called directly, so the " +
            "cap cannot change what runs; the uncapped lane checks this exact pair on the same seeded data. Not compared here (#1692).");
    }

    /// <summary>Every available tier of every dot-product family agrees with the scalar tier (see class summary for the bound).</summary>
    [SkippableTheory]
    [MemberData(nameof(DotCases))]
    public void Dot_Tier_MatchesScalar(string family, string tier, int blocks)
    {
        Family fam = Families[family];
        Tier t = fam.Tiers.Single(x => x.Name == tier);
        SkipUnlessTierRunnable(family, t);

        int seed = StableSeed(family, blocks);
        DotOutcome r = RunDot(fam, t, blocks, seed);
        Assert.True(r.SumAbsProducts > 0, "degenerate fixture: every w_i*x_i is zero, the comparison cannot discriminate");
        Assert.True(r.Diff <= r.Bound,
            $"{family}/{tier} blocks={blocks} seed={seed}: scalar={r.Scalar:R} tier={r.Tier:R} |diff|={r.Diff:R} bound={r.Bound:R} " +
            $"(ulp*sqrt({blocks * fam.ElemsPerBlock})*sum|w*x|={r.SumAbsProducts:R}) [{CpuFeatures.Describe()}]");
    }

    /// <summary>
    /// Local worst-case finder for the bound (#1692): opt in with <c>DOTLLM_TIER_SWEEP_SEEDS=N</c>. Runs every case over N seeds and
    /// fails if any seed exceeds the bound, printing the worst |diff|/bound per family/tier so the margin is visible. Skipped in CI.
    /// </summary>
    [SkippableFact]
    public void Dot_Tier_SeedSweep_WorstRatio()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_TIER_SWEEP_SEEDS");
        Skip.IfNot(int.TryParse(env, out int seeds) && seeds > 0, "set DOTLLM_TIER_SWEEP_SEEDS=N to sweep N seeds per case");

        var report = new System.Text.StringBuilder();
        var failures = new List<string>();
        foreach (var fam in Families.Values)
            foreach (var t in fam.Tiers)
            {
                if (!t.Available() || CpuFeatures.Cap < t.IsaTier) continue;
                double worst = 0; string worstCase = "";
                foreach (object[] c in DotCases().Where(c => (string)c[0] == fam.Name && (string)c[1] == t.Name))
                {
                    int blocks = (int)c[2];
                    for (int s = 0; s < seeds; s++)
                    {
                        int seed = StableSeed(fam.Name, blocks, s + 1);
                        DotOutcome r = RunDot(fam, t, blocks, seed);
                        double ratio = r.Diff / r.Bound;
                        if (ratio > worst) { worst = ratio; worstCase = $"blocks={blocks} seed={seed} scalar={r.Scalar:R} tier={r.Tier:R} |diff|={r.Diff:R} bound={r.Bound:R}"; }
                        if (r.Diff > r.Bound) failures.Add($"{fam.Name}/{t.Name} {worstCase}");
                    }
                }
                report.AppendLine($"{fam.Name}/{t.Name}: worst |diff|/bound = {worst:F4} ({worstCase})");
            }

        Console.WriteLine(report.ToString());
        Assert.True(failures.Count == 0, $"{failures.Count} seed(s) over the bound:\n{string.Join('\n', failures)}\n\n{report}");
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
