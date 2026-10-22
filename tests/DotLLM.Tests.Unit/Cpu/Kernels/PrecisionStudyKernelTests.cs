using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #1109 (CPU precision study) prototypes, each checked against an independent scalar oracle:
/// the float-activation Q8_0 / Q4_0 / BF16 GEMV tiers, the AVX-VNNI legacy-quant dot, the integer
/// QK^T over Q8_0 KV, the simulated FP16/BF16 KV rounding, and the F32-activation reference switch.
/// All switches are process-global, hence the isolated collection.
/// </summary>
[Collection("PrecisionStudyStatics")]
public sealed unsafe class PrecisionStudyKernelTests
{
    private static float[] Gaussianish(int n, int seed, float scale)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (rng.NextSingle() + rng.NextSingle() + rng.NextSingle() - 1.5f) * scale;
        return a;
    }

    private static byte[] Encode(float[] w, QuantizationType qt)
    {
        if (qt == QuantizationType.Q8_0)
        {
            var dst = new byte[Quantize.RowByteSize(w.Length, qt)];
            Quantize.FromFloat32(w, w.Length, qt, dst);
            return dst;
        }

        // Q4_0 (the shared quantizer does not cover it): llama.cpp quantize_row_q4_0_ref.
        var q = new byte[w.Length / 32 * 18];
        for (int b = 0; b < w.Length / 32; b++)
        {
            float amax = 0, max = 0;
            for (int i = 0; i < 32; i++)
            {
                if (MathF.Abs(w[b * 32 + i]) > amax) { amax = MathF.Abs(w[b * 32 + i]); max = w[b * 32 + i]; }
            }

            float d = max / -8f, id = d != 0 ? 1f / d : 0f;
            BitConverter.TryWriteBytes(q.AsSpan(b * 18), (Half)d);
            for (int i = 0; i < 16; i++)
            {
                int lo = Math.Min(15, (int)(w[b * 32 + i] * id + 8.5f));
                int hi = Math.Min(15, (int)(w[b * 32 + i + 16] * id + 8.5f));
                q[b * 18 + 2 + i] = (byte)(lo | (hi << 4));
            }
        }

        return q;
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q8_0, 3072)]
    [InlineData(QuantizationType.Q8_0, 96)]
    [InlineData(QuantizationType.Q4_0, 3072)]
    [InlineData(QuantizationType.Q4_0, 160)]
    public void FloatActivationTier_4Rows_MatchesScalarAndDequantOracle(QuantizationType qt, int k)
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "needs AVX2+FMA");
        const int rows = 4;
        float[] w = Gaussianish(rows * k, 11, 0.05f);
        float[] x = Gaussianish(k, 12, 1f);

        // Make the rows distinct in magnitude so a row mix-up cannot pass.
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < k; i++) w[r * k + i] *= r + 1;

        byte[] q = Encode(w, qt);
        int rowBytes = q.Length / rows;
        var got = new float[rows];
        var deq = new float[k];
        fixed (byte* qp = q)
        fixed (float* xp = x, gp = got)
        {
            if (qt == QuantizationType.Q8_0) MatMul.VecDotQ8_0F32Avx2_4Rows(qp, rowBytes, xp, k / 32, gp);
            else MatMul.VecDotQ4_0F32Avx2_4Rows(qp, rowBytes, xp, k / 32, gp);

            for (int r = 0; r < rows; r++)
            {
                float scalar = MatMul.VecDotQuantF32Scalar(qt, qp + r * rowBytes, xp, k / 32);

                // Independent oracle: the production dequantizer, then a double-precision dot.
                Dequantize.ToFloat32((nint)(qp + r * rowBytes), k, qt, deq);
                double oracle = 0;
                for (int i = 0; i < k; i++) oracle += (double)deq[i] * x[i];

                double tol = 1e-4 * Math.Max(1.0, Math.Abs(oracle));
                Assert.True(Math.Abs(got[r] - oracle) < tol, $"{qt} row {r}: kernel {got[r]} vs dequant oracle {oracle}");
                Assert.True(Math.Abs(scalar - oracle) < tol, $"{qt} row {r}: scalar {scalar} vs dequant oracle {oracle}");
            }
        }
    }

    [SkippableFact]
    public void Bf16Tier_4Rows_MatchesScalar()
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "needs AVX2+FMA");
        const int rows = 4, k = 1000;
        float[] w = Gaussianish(rows * k, 21, 0.05f);
        float[] x = Gaussianish(k, 22, 1f);
        var bits = new ushort[rows * k];
        for (int i = 0; i < bits.Length; i++)
            bits[i] = (ushort)(BitConverter.SingleToUInt32Bits(w[i] * (1 + i / k)) >> 16);

        var got = new float[rows];
        fixed (ushort* bp = bits)
        fixed (float* xp = x, gp = got)
        {
            MatMul.VecDotBf16F32Avx2_4Rows((byte*)bp, k * 2, xp, k, gp);
            for (int r = 0; r < rows; r++)
            {
                double oracle = 0;
                for (int i = 0; i < k; i++) oracle += BitConverter.Int32BitsToSingle(bp[r * k + i] << 16) * (double)x[i];
                float scalar = MatMul.VecDotBf16F32Scalar(bp + r * k, xp, k);
                Assert.True(Math.Abs(got[r] - oracle) < 1e-4 * Math.Max(1, Math.Abs(oracle)), $"row {r}: {got[r]} vs {oracle}");
                Assert.True(Math.Abs(scalar - oracle) < 1e-4 * Math.Max(1, Math.Abs(oracle)), $"scalar row {r}: {scalar} vs {oracle}");
            }
        }
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q4_1)]
    [InlineData(QuantizationType.Q5_1)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void LegacyVnni_BitExactWithAvx2AndScalar(QuantizationType qt)
    {
        Skip.IfNot(AvxVnni.IsSupported, "needs AVX-VNNI");
        const int rows = 4, blocks = 37;   // odd block count
        var rng = new Random(31 + (int)qt);
        int blockBytes = qt switch { QuantizationType.Q4_1 => 20, QuantizationType.Q5_1 => 24, _ => 18 };
        var w = new byte[rows * blocks * blockBytes];
        rng.NextBytes(w);
        for (int b = 0; b < rows * blocks; b++)
        {
            BitConverter.TryWriteBytes(w.AsSpan(b * blockBytes), (Half)(0.02f + 0.0001f * b));
            if (qt is QuantizationType.Q4_1 or QuantizationType.Q5_1)
                BitConverter.TryWriteBytes(w.AsSpan(b * blockBytes + 2), (Half)(-0.1f));
        }

        // Raw Q8_1 bytes over the quantizer's range [-127, 127]. (-128 is excluded: the IQ4_NL sign
        // trick wraps it in both AVX2 forms alike, a pre-existing difference from scalar that no
        // quantizer can trigger.)
        var q8 = new byte[blocks * MatMul.Q8_1BlockBytes];
        rng.NextBytes(q8);
        for (int i = 0; i < q8.Length; i++)
        {
            if (q8[i] == 0x80) q8[i] = 0x81;
        }
        for (int b = 0; b < blocks; b++)
        {
            BitConverter.TryWriteBytes(q8.AsSpan(b * MatMul.Q8_1BlockBytes), (Half)(0.01f + 0.001f * b));
            BitConverter.TryWriteBytes(q8.AsSpan(b * MatMul.Q8_1BlockBytes + 2), (Half)(0.5f * b));
        }

        int rowBytes = w.Length / rows;
        var avx2 = new float[rows];
        var vnni = new float[rows];
        try
        {
            fixed (byte* wp = w, xp = q8)
            fixed (float* a = avx2, v = vnni)
            {
                MatMul.LegacyVnniEnabled = false;
                MatMul.VecDotLegacyQuantAvx2_4Rows(qt, wp, rowBytes, xp, blocks, a);
                float single0 = MatMul.VecDotLegacyQuantRow(qt, wp, xp, blocks);
                MatMul.LegacyVnniEnabled = true;
                MatMul.VecDotLegacyQuantAvx2_4Rows(qt, wp, rowBytes, xp, blocks, v);
                float single1 = MatMul.VecDotLegacyQuantRow(qt, wp, xp, blocks);

                for (int r = 0; r < rows; r++)
                {
                    Assert.Equal(BitConverter.SingleToInt32Bits(avx2[r]), BitConverter.SingleToInt32Bits(vnni[r]));
                    float scalar = MatMul.VecDotLegacyQuantScalar(qt, wp + r * rowBytes, xp, blocks);
                    Assert.Equal(BitConverter.SingleToInt32Bits(scalar), BitConverter.SingleToInt32Bits(vnni[r]));
                }

                Assert.Equal(BitConverter.SingleToInt32Bits(single0), BitConverter.SingleToInt32Bits(single1));
            }
        }
        finally
        {
            MatMul.LegacyVnniEnabled = false;
        }
    }

    [Theory]
    [InlineData(8, 2, 64, 130)]
    [InlineData(24, 8, 128, 300)]
    [InlineData(32, 4, 128, 77)]
    public void Int8Qk_MatchesQuantizedQReference_AndDiffersFromDequantPath(int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        int kvStride = numKvHeads * headDim, qStride = numHeads * headDim;

        // K with a few outlier channels per head, as real K caches have.
        float[] k = Gaussianish(seqKv * kvStride, 41, 1f);
        for (int p = 0; p < seqKv; p++)
            for (int h = 0; h < numKvHeads; h++)
                k[p * kvStride + h * headDim + (h * 7 % headDim)] *= 12f;
        float[] v = Gaussianish(seqKv * kvStride, 42, 1f);
        float[] q = Gaussianish(qStride, 43, 1f);

        using var cache = new QuantizedKvCache(1, numKvHeads, headDim, seqKv, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        int[] pos = Enumerable.Range(0, seqKv).ToArray();
        fixed (float* kp = k, vp = v)
            cache.Update(new TensorRef(seqKv, kvStride, DType.Float32, -1, (nint)kp),
                new TensorRef(seqKv, kvStride, DType.Float32, -1, (nint)vp), pos, 0);

        var outInt = new float[qStride];
        var outDeq = new float[qStride];
        try
        {
            fixed (float* qp = q, oi = outInt, od = outDeq)
            {
                Attention.KvInt8QkEnabled = true;
                Attention.Execute(qp, cache, 0, oi, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool: null);
                Attention.KvInt8QkEnabled = false;
                Attention.Execute(qp, cache, 0, od, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool: null);

                // Oracle: Q rounded through Q8_0, K and V dequantized, softmax in double.
                byte* kq = (byte*)cache.GetQuantizedKeysPtr(0), vq = (byte*)cache.GetQuantizedValuesPtr(0);
                int rowBytes = cache.KeyQuantizedRowBytesOf(0), headBytes = headDim / 32 * 34;
                var qq = new byte[headBytes];
                var qd = new float[headDim];
                var kd = new float[headDim];
                var vd = new float[headDim];
                var s = new double[seqKv];
                double maxErrVsOracle = 0, maxDiffPaths = 0;
                for (int h = 0; h < numHeads; h++)
                {
                    int kvh = h / (numHeads / numKvHeads);
                    fixed (byte* qqp = qq)
                    fixed (float* qdp = qd, kdp = kd, vdp = vd)
                    {
                        MatMul.QuantizeF32ToQ8_0(qp + h * headDim, qqp, headDim);
                        KvQuantize.Q8_0ToF32(qqp, qdp, headDim);
                        double mx = double.NegativeInfinity;
                        for (int j = 0; j < seqKv; j++)
                        {
                            KvQuantize.Q8_0ToF32(kq + (long)j * rowBytes + kvh * headBytes, kdp, headDim);
                            double dot = 0;
                            for (int d = 0; d < headDim; d++) dot += (double)qd[d] * kd[d];
                            s[j] = dot / Math.Sqrt(headDim);
                            mx = Math.Max(mx, s[j]);
                        }

                        double sum = 0;
                        for (int j = 0; j < seqKv; j++) sum += s[j] = Math.Exp(s[j] - mx);
                        var o = new double[headDim];
                        for (int j = 0; j < seqKv; j++)
                        {
                            KvQuantize.Q8_0ToF32(vq + (long)j * rowBytes + kvh * headBytes, vdp, headDim);
                            for (int d = 0; d < headDim; d++) o[d] += s[j] / sum * vd[d];
                        }

                        for (int d = 0; d < headDim; d++)
                        {
                            maxErrVsOracle = Math.Max(maxErrVsOracle, Math.Abs(oi[h * headDim + d] - o[d]));
                            maxDiffPaths = Math.Max(maxDiffPaths, Math.Abs(oi[h * headDim + d] - od[h * headDim + d]));
                        }
                    }
                }

                Assert.True(maxErrVsOracle < 2e-4, $"int8 QK^T vs quantized-Q oracle: max abs err {maxErrVsOracle:E3}");

                // The prototype must actually be the dispatched path: quantizing Q moves the output.
                Assert.True(maxDiffPaths > 1e-5, $"int8 QK^T output identical to dequant path ({maxDiffPaths:E3}) - flag not honoured?");
            }
        }
        finally
        {
            Attention.KvInt8QkEnabled = false;
        }
    }

    [Fact]
    public void SimulatedKvRounding_F16AndBf16_RoundToNearestEven()
    {
        float[] row = [1.00390625f, 1.01171875f, 3.14159265f, -70000f, 1e-8f, float.PositiveInfinity, float.NaN, 65519f];
        var f16 = (float[])row.Clone();
        var bf16 = (float[])row.Clone();
        fixed (float* a = f16, b = bf16)
        {
            SimpleKvCache.RoundRowInPlace(a, row.Length, SimpleKvCache.SimulatedKvRounding.F16);
            SimpleKvCache.RoundRowInPlace(b, row.Length, SimpleKvCache.SimulatedKvRounding.BF16);
        }

        for (int i = 0; i < row.Length; i++)
            Assert.Equal((float)(Half)row[i], f16[i]);

        Assert.Equal(1.0f, bf16[0]);          // tie -> even mantissa
        Assert.Equal(1.015625f, bf16[1]);     // tie -> even mantissa (rounds up)
        Assert.Equal(3.140625f, bf16[2]);
        Assert.Equal(float.PositiveInfinity, bf16[5]);
        Assert.True(float.IsNaN(bf16[6]));
        Assert.Equal(float.NegativeInfinity, f16[3]);   // FP16 overflow saturates to inf
    }

    [Fact]
    public void SimpleKvCache_SimulatedRounding_AppliesOnWrite_OnlyWhenEnabled()
    {
        const int stride = 64;
        float[] k = Gaussianish(stride * 3, 51, 1f);
        float[] v = Gaussianish(stride * 3, 52, 1f);
        try
        {
            foreach (var mode in new[] { SimpleKvCache.SimulatedKvRounding.None, SimpleKvCache.SimulatedKvRounding.F16 })
            {
                SimpleKvCache.SimulatedRounding = mode;
                using var cache = new SimpleKvCache(1, 1, stride, 4);
                fixed (float* kp = k, vp = v)
                    cache.Update(new TensorRef(3, stride, DType.Float32, -1, (nint)kp),
                        new TensorRef(3, stride, DType.Float32, -1, (nint)vp), [0, 1, 2], 0);
                var stored = cache.KeysSpan(0);
                for (int i = 0; i < 3 * stride; i++)
                    Assert.Equal(mode == SimpleKvCache.SimulatedKvRounding.None ? k[i] : (float)(Half)k[i], stored[i]);
            }
        }
        finally
        {
            SimpleKvCache.SimulatedRounding = SimpleKvCache.SimulatedKvRounding.None;
        }
    }

    [Fact]
    public void ReferenceF32Activations_DefaultOff_AndDisablesFusedDecodeWhenOn()
    {
        Assert.False(MatMul.ReferenceF32Activations);
        Assert.True(MatMul.SupportsFusedDecode(QuantizationType.Q8_0));
        try
        {
            MatMul.ReferenceF32Activations = true;
            Assert.True(MatMul.UsesReferenceF32Path(QuantizationType.Q4_K));
            Assert.False(MatMul.UsesReferenceF32Path(QuantizationType.F16));
            Assert.False(MatMul.SupportsFusedDecode(QuantizationType.Q8_0));
            Assert.True(MatMul.SupportsFusedDecode(QuantizationType.F16));
        }
        finally
        {
            MatMul.ReferenceF32Activations = false;
        }
    }
}
