using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class KvQuantizeTests
{
    private const int BlockSize = 32;

    [Fact]
    public void Q4_0_Scalar_RoundTrip_WithinTolerance()
    {
        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 4];
        float[] output = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, qp, input.Length);
            KvQuantize.Q4_0ToF32Scalar(qp, op, input.Length);
        }

        float maxErr = 0;
        for (int i = 0; i < input.Length; i++)
        {
            float err = MathF.Abs(input[i] - output[i]);
            if (err > maxErr) maxErr = err;
        }

        // Q4_0 has 4-bit precision — expect ~1/7 of max value error per element
        float maxVal = input.Max(MathF.Abs);
        float expectedMaxErr = maxVal / 7.0f; // one quantization step
        Assert.True(maxErr <= expectedMaxErr * 1.1f,
            $"Max round-trip error {maxErr} exceeds expected {expectedMaxErr}");
    }

    [Fact]
    public void Q4_0_Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return; // Skip on non-AVX2 hardware

        float[] input = GenerateTestData(BlockSize * 8);
        byte[] quantScalar = new byte[KvQuantize.Q4_0BlockBytes * 8];
        byte[] quantAvx2 = new byte[KvQuantize.Q4_0BlockBytes * 8];

        fixed (float* ip = input)
        fixed (byte* sp = quantScalar)
        fixed (byte* ap = quantAvx2)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, sp, input.Length);
            KvQuantize.F32ToQ4_0Avx2(ip, ap, input.Length);
        }

        Assert.Equal(quantScalar, quantAvx2);
    }

    [Fact]
    public void Q4_0_ZeroInput_AllZeroOutput()
    {
        float[] input = new float[BlockSize];
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes];
        float[] output = new float[BlockSize];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ4_0(ip, qp, BlockSize);
            KvQuantize.Q4_0ToF32(qp, op, BlockSize);
        }

        for (int i = 0; i < BlockSize; i++)
            Assert.Equal(0f, output[i]);
    }

    [Fact]
    public void Q4_0_Dequant_Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;

        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 4];
        float[] outScalar = new float[input.Length];
        float[] outAvx2 = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, qp, input.Length);
        }

        fixed (byte* qp = quantized)
        fixed (float* sp = outScalar)
        fixed (float* ap = outAvx2)
        {
            KvQuantize.Q4_0ToF32Scalar(qp, sp, input.Length);
            KvQuantize.Q4_0ToF32Avx2(qp, ap, input.Length);
        }

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(outScalar[i], outAvx2[i], 5);
    }

    [Fact]
    public void Q8_0_RoundTrip_WithinTolerance()
    {
        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q8_0BlockBytes * 4];
        float[] output = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ8_0(ip, qp, input.Length);
            KvQuantize.Q8_0ToF32(qp, op, input.Length);
        }

        float maxErr = 0;
        for (int i = 0; i < input.Length; i++)
        {
            float err = MathF.Abs(input[i] - output[i]);
            if (err > maxErr) maxErr = err;
        }

        // Q8_0 has 8-bit precision — much tighter tolerance
        float maxVal = input.Max(MathF.Abs);
        float expectedMaxErr = maxVal / 127.0f;
        Assert.True(maxErr <= expectedMaxErr * 1.1f,
            $"Max round-trip error {maxErr} exceeds expected {expectedMaxErr}");
    }

    [Fact]
    public void Q8_0_Dequant_Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;

        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q8_0BlockBytes * 4];
        float[] outScalar = new float[input.Length];
        float[] outAvx2 = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        {
            KvQuantize.F32ToQ8_0(ip, qp, input.Length);
        }

        fixed (byte* qp = quantized)
        fixed (float* sp = outScalar)
        fixed (float* ap = outAvx2)
        {
            KvQuantize.Q8_0ToF32Scalar(qp, sp, input.Length);
            KvQuantize.Q8_0ToF32Avx2(qp, ap, input.Length);
        }

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(outScalar[i], outAvx2[i], 5);
    }

    [Fact]
    public void QuantizedRowBytes_ReturnsCorrectSize()
    {
        int kvStride = 1024;
        Assert.Equal(kvStride / BlockSize * KvQuantize.Q8_0BlockBytes,
            KvQuantize.QuantizedRowBytes(kvStride, Core.Configuration.KvCacheDType.Q8_0));
        Assert.Equal(kvStride / BlockSize * KvQuantize.Q4_0BlockBytes,
            KvQuantize.QuantizedRowBytes(kvStride, Core.Configuration.KvCacheDType.Q4_0));
    }

    /// <summary>
    /// Random-data round-trip with per-element error bound — verifies that quantize → dequant
    /// preserves element ORDER within each block, not just block-level statistics.
    /// </summary>
    [Fact]
    public void Q4_0_RandomData_RoundTrip_PreservesElementOrder()
    {
        float[] input = GenerateTestData(BlockSize * 8);
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 8];
        float[] output = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ4_0(ip, qp, input.Length);
            KvQuantize.Q4_0ToF32(qp, op, input.Length);
        }

        for (int i = 0; i < input.Length; i++)
        {
            int block = i / BlockSize;
            float[] blockSlice = input[(block * BlockSize)..((block + 1) * BlockSize)];
            float maxAbs = blockSlice.Max(MathF.Abs);
            float step = maxAbs / 7.0f;
            Assert.True(MathF.Abs(input[i] - output[i]) <= step * 1.1f,
                $"Element {i}: input={input[i]}, output={output[i]}, err={MathF.Abs(input[i] - output[i])}, step={step}");
        }
    }

    /// <summary>
    /// Negative control: deliberately permute lo/hi nibbles in Q4_0 dequant output.
    /// The test MUST fail — if it passes, the test data is degenerate and can't see permutations.
    /// </summary>
    [Fact]
    public void Q4_0_NegativeControl_PermutedUnpack_DetectedByRandomData()
    {
        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 4];
        float[] correct = new float[input.Length];
        float[] permuted = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* cp = correct)
        {
            KvQuantize.F32ToQ4_0(ip, qp, input.Length);
            KvQuantize.Q4_0ToF32Scalar(qp, cp, input.Length);
        }

        // Permuted dequant: swap even/odd elements within each block (lo ↔ hi nibble)
        fixed (byte* qp = quantized)
        fixed (float* pp = permuted)
        {
            int blockCount = input.Length / BlockSize;
            for (int block = 0; block < blockCount; block++)
            {
                byte* blockSrc = qp + block * KvQuantize.Q4_0BlockBytes;
                float* blockDst = pp + block * BlockSize;
                float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
                byte* qs = blockSrc + 2;
                for (int j = 0; j < 16; j++)
                {
                    byte packed = qs[j];
                    int lo = (packed & 0x0F) - 8;
                    int hi = (packed >> 4) - 8;
                    blockDst[2 * j] = d * hi;      // swapped
                    blockDst[2 * j + 1] = d * lo;  // swapped
                }
            }
        }

        // The permuted output should differ from the correct output for random data
        int mismatches = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (MathF.Abs(correct[i] - permuted[i]) > 1e-6f)
                mismatches++;
        }
        Assert.True(mismatches > input.Length / 4,
            $"Permuted Q4_0 unpack only differed at {mismatches}/{input.Length} elements — test data is too degenerate to catch permutation bugs.");
    }

    /// <summary>
    /// Negative control: deliberately swap adjacent pairs in Q8_0 dequant output.
    /// </summary>
    [Fact]
    public void Q8_0_NegativeControl_PermutedUnpack_DetectedByRandomData()
    {
        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q8_0BlockBytes * 4];
        float[] correct = new float[input.Length];
        float[] permuted = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* cp = correct)
        {
            KvQuantize.F32ToQ8_0(ip, qp, input.Length);
            KvQuantize.Q8_0ToF32Scalar(qp, cp, input.Length);
        }

        // Permuted dequant: swap adjacent pairs (i, i+1) within each block
        fixed (byte* qp = quantized)
        fixed (float* pp = permuted)
        {
            int blockCount = input.Length / BlockSize;
            for (int block = 0; block < blockCount; block++)
            {
                byte* blockSrc = qp + block * KvQuantize.Q8_0BlockBytes;
                float* blockDst = pp + block * BlockSize;
                float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
                sbyte* qs = (sbyte*)(blockSrc + 2);
                for (int i = 0; i < BlockSize; i += 2)
                {
                    blockDst[i] = d * qs[i + 1];       // swapped
                    blockDst[i + 1] = d * qs[i];        // swapped
                }
            }
        }

        int mismatches = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (MathF.Abs(correct[i] - permuted[i]) > 1e-6f)
                mismatches++;
        }
        Assert.True(mismatches > input.Length / 4,
            $"Permuted Q8_0 unpack only differed at {mismatches}/{input.Length} elements — test data is too degenerate to catch permutation bugs.");
    }

    /// <summary>
    /// Demonstrates that RAMP data fails the negative control: a monotone ramp is too degenerate
    /// to distinguish permuted from correct unpack within many blocks. This test documents the
    /// vulnerability that #974 fixes.
    /// </summary>
    [Fact]
    public void Q4_0_RampData_IsDegenerate_FailsToDetectPermutation()
    {
        // Ramp: 0, 0.01, 0.02, ... — within a 32-element block adjacent pairs differ by ~0.01
        // which is well below the Q4_0 quantization step for most blocks.
        float[] ramp = new float[BlockSize * 4];
        for (int i = 0; i < ramp.Length; i++)
            ramp[i] = i * 0.01f;

        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 4];
        float[] correct = new float[ramp.Length];
        float[] permuted = new float[ramp.Length];

        fixed (float* ip = ramp)
        fixed (byte* qp = quantized)
        fixed (float* cp = correct)
        {
            KvQuantize.F32ToQ4_0(ip, qp, ramp.Length);
            KvQuantize.Q4_0ToF32Scalar(qp, cp, ramp.Length);
        }

        // Same lo/hi swap as the negative control
        fixed (byte* qp = quantized)
        fixed (float* pp = permuted)
        {
            int blockCount = ramp.Length / BlockSize;
            for (int block = 0; block < blockCount; block++)
            {
                byte* blockSrc = qp + block * KvQuantize.Q4_0BlockBytes;
                float* blockDst = pp + block * BlockSize;
                float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
                byte* qs = blockSrc + 2;
                for (int j = 0; j < 16; j++)
                {
                    byte packed = qs[j];
                    int lo = (packed & 0x0F) - 8;
                    int hi = (packed >> 4) - 8;
                    blockDst[2 * j] = d * hi;
                    blockDst[2 * j + 1] = d * lo;
                }
            }
        }

        // Count how many elements differ between correct and permuted
        int mismatches = 0;
        for (int i = 0; i < ramp.Length; i++)
        {
            if (MathF.Abs(correct[i] - permuted[i]) > 1e-6f)
                mismatches++;
        }

        // Ramp data: most adjacent pairs quantize to the same nibble → permutation is invisible.
        // This proves ramp data is degenerate.
        Assert.True(mismatches < ramp.Length / 4,
            $"Ramp data detected {mismatches}/{ramp.Length} permutation mismatches — " +
            $"ramp is NOT degenerate on this system (this is unexpected; the test assumption is wrong).");
    }

    private static float[] GenerateTestData(int count)
    {
        var rng = new Random(42);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1) * 5.0f; // range [-5, 5]
        return data;
    }
}
