using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1340: the CPU activation quantizers must agree with each other (scalar vs AVX2 vs AVX-512 dispatch) on exact rounding
/// ties and on one-ULP-from-a-tie values, and with llama.cpp's reference quantizers
/// (<c>quantize_row_q8_0_ref</c>/<c>quantize_row_q8_1_ref</c>: <c>d = amax/127</c>, <c>id = 1/d</c>, round-to-nearest-even in the SIMD
/// paths). Corpus shared with the Vulkan test (<see cref="ActivationQuantizerCorpus.BuildCorpus"/>).
/// </summary>
public sealed unsafe class ActivationQuantizerRoundingTests
{
    private readonly ITestOutputHelper _out;
    public ActivationQuantizerRoundingTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Q8_1_Scalar_And_Avx2_AreBitIdentical_OnTieCorpus()
    {
        float[] x = ActivationQuantizerCorpus.BuildCorpus(out string[] labels);
        int blocks = labels.Length;
        var a = new byte[blocks * 36];
        var b = new byte[blocks * 36];
        fixed (float* src = x)
        fixed (byte* pa = a)
        fixed (byte* pb = b)
        {
            MatMul.QuantizeF32ToQ8_1Scalar(src, pa, x.Length);
            MatMul.QuantizeF32ToQ8_1(src, pb, x.Length);
        }
        for (int i = 0; i < blocks; i++)
            Assert.True(a.AsSpan(i * 36, 36).SequenceEqual(b.AsSpan(i * 36, 36)), $"block {i} '{labels[i]}' differs scalar vs dispatched");
    }

    [Fact]
    public void Q8_0_Scalar_Avx2_Avx512_AreBitIdentical_OnTieCorpus()
    {
        float[] x = ActivationQuantizerCorpus.BuildCorpus(out string[] labels);
        int blocks = labels.Length;
        var a = new byte[blocks * 34];
        var b = new byte[blocks * 34];
        var c = new byte[blocks * 34];
        fixed (float* src = x)
        fixed (byte* pa = a)
        fixed (byte* pb = b)
        fixed (byte* pc = c)
        {
            MatMul.QuantizeF32ToQ8_0Scalar(src, pa, x.Length);
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) MatMul.QuantizeF32ToQ8_0Avx2(src, pb, x.Length);
            else Buffer.BlockCopy(a, 0, b, 0, a.Length);
            if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported) MatMul.QuantizeF32ToQ8_0Avx512(src, pc, x.Length);
            else Buffer.BlockCopy(a, 0, c, 0, a.Length);
        }
        for (int i = 0; i < blocks; i++)
        {
            Assert.True(a.AsSpan(i * 34, 34).SequenceEqual(b.AsSpan(i * 34, 34)), $"block {i} '{labels[i]}' differs scalar vs AVX2");
            Assert.True(a.AsSpan(i * 34, 34).SequenceEqual(c.AsSpan(i * 34, 34)), $"block {i} '{labels[i]}' differs scalar vs AVX-512");
        }
    }

    /// <summary>Tie corpus for 256-wide Q8_K blocks: 8 corpus blocks per super-block would mix scales, so build exact-tie super-blocks directly.</summary>
    private static float[] Q8KCorpus(out int n)
    {
        var list = new List<float>();
        foreach (int e in new[] { -8, -3, 0, 4 })
            for (int sign = 0; sign < 2; sign++)
            {
                float d = MathF.Pow(2f, e), s = sign == 0 ? 1f : -1f;
                for (int rep = 0; rep < 4; rep++)
                    for (int i = 0; i < 256; i++)
                        list.Add(i == 0 ? s * 127f * d : s * ((rep * 31 + i) % 126 + 0.5f) * d);
            }
        var rng = new Random(1340);
        for (int b = 0; b < 200; b++)
        {
            float amax = (float)Math.Exp(rng.NextDouble() * 12 - 6), d = amax / 127f, id = 1f / d;
            for (int i = 0; i < 256; i++)
            {
                if (i == 0) { list.Add(amax); continue; }
                float bd = (rng.Next(0, 126) + 0.5f) / id;
                int m = rng.Next(3);
                float v = m == 0 ? bd : m == 1 ? MathF.BitIncrement(bd) : MathF.BitDecrement(bd);
                list.Add(rng.Next(2) == 0 ? v : -v);
            }
        }
        n = list.Count / 256;
        return list.ToArray();
    }

    [Fact]
    public void Q8_K_Scalar_And_Avx2_AreBitIdentical_OnTieCorpus()
    {
        float[] x = Q8KCorpus(out int blocks);
        var a = new byte[blocks * MatMul.Q8_K_BlockBytes];
        var b = new byte[blocks * MatMul.Q8_K_BlockBytes];
        fixed (float* src = x)
        fixed (byte* pa = a)
        fixed (byte* pb = b)
        {
            MatMul.QuantizeF32ToQ8_KScalar(src, pa, x.Length);
            MatMul.QuantizeF32ToQ8_K(src, pb, x.Length);
        }
        for (int i = 0; i < blocks; i++)
            Assert.True(a.AsSpan(i * MatMul.Q8_K_BlockBytes, MatMul.Q8_K_BlockBytes).SequenceEqual(b.AsSpan(i * MatMul.Q8_K_BlockBytes, MatMul.Q8_K_BlockBytes)),
                $"Q8_K block {i} differs scalar vs dispatched");
    }

    /// <summary>
    /// llama.cpp's <c>quantize_row_q8_K_ref</c> uses a SIGNED scale (<c>iscale = -127/max</c>, <c>d = 1/iscale</c>), which differs from
    /// the old dotLLM form (<c>d = amax/127</c>, <c>id = 1/d</c>) in the final ULP of the multiplier, which moves lanes across ties. Our blocks keep d positive and fold the sign into q (q_ours == -q_ref, d_ours == |d_ref|), so every product d*q is bit-equal.
    /// </summary>
    [Fact]
    public void Q8_K_MatchesLlamaCppReference_OnTieCorpus()
    {
        float[] x = Q8KCorpus(out int blocks);
        var ours = new byte[blocks * MatMul.Q8_K_BlockBytes];
        fixed (float* src = x)
        fixed (byte* po = ours)
            MatMul.QuantizeF32ToQ8_K(src, po, x.Length);

        int diffLanes = 0, diffBlocks = 0, dBitDiff = 0;
        for (int b = 0; b < blocks; b++)
        {
            // llama.cpp quantize_row_q8_K_ref
            float max = 0, amax = 0;
            for (int j = 0; j < 256; j++) { float ax = MathF.Abs(x[b * 256 + j]); if (ax > amax) { amax = ax; max = x[b * 256 + j]; } }
            float iscale = -127f / max;
            float dRef = 1f / iscale;
            int off = b * MatMul.Q8_K_BlockBytes;
            float dOurs = BitConverter.ToSingle(ours, off);
            bool blockDiff = false;
            if (MathF.Abs(dRef) != dOurs) dBitDiff++;
            for (int j = 0; j < 256; j++)
            {
                int v = (int)MathF.Round(iscale * x[b * 256 + j]);   // nearest_int
                v = Math.Min(127, v);
                // ours = positive-d convention: q_ours == -q_ref
                int qOurs = (sbyte)ours[off + 4 + j];
                if (qOurs != (max > 0 ? -v : v)) { diffLanes++; blockDiff = true; }
            }
            if (blockDiff) diffBlocks++;
        }
        _out.WriteLine($"Q8_K vs llama.cpp ref: blocks={blocks} blocksWithDiff={diffBlocks} lanesDiff={diffLanes} scaleBitsDiff={dBitDiff}");
        Assert.True(diffLanes == 0 && dBitDiff == 0,
            $"Q8_K diverges from llama.cpp quantize_row_q8_K_ref: {diffLanes} lanes in {diffBlocks}/{blocks} blocks, {dBitDiff} scales differ in bits.");
    }
}
