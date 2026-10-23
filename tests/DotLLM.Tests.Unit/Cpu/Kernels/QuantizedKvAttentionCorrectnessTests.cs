using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Correctness tests comparing quantized-KV attention (Q8_0, Q4_0, K8V4) against
/// F32-KV attention at real GQA shapes (numHeads > numKvHeads, headDim >= 32).
/// Random data, discriminating group sizes, exercising the tiled quantized attention path.
/// </summary>
public sealed unsafe class QuantizedKvAttentionCorrectnessTests
{
    private const int NumLayers = 1;

    [Theory]
    [InlineData(8, 2, 64, 128, "Q8_0 decode, Llama-style GQA 8:2, headDim=64")]
    [InlineData(32, 4, 128, 256, "Q8_0 decode, Qwen-style GQA 32:4, headDim=128")]
    [InlineData(8, 1, 64, 96, "Q8_0 decode, MQA 8:1, headDim=64")]
    [InlineData(16, 4, 64, 200, "Q8_0 decode, GQA 16:4, headDim=64")]
    public void Decode_Q8_0_MatchesF32(int numHeads, int numKvHeads, int headDim, int seqKv, string label)
    {
        AssertQuantizedMatchesF32(
            numHeads, numKvHeads, headDim, seqKv, seqQ: 1,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0,
            tolerance: 0.05f, label);
    }

    [Theory]
    [InlineData(8, 2, 64, 128, "Q4_0 decode, Llama-style GQA 8:2, headDim=64")]
    [InlineData(32, 4, 128, 256, "Q4_0 decode, Qwen-style GQA 32:4, headDim=128")]
    [InlineData(8, 1, 64, 96, "Q4_0 decode, MQA 8:1, headDim=64")]
    public void Decode_Q4_0_MatchesF32(int numHeads, int numKvHeads, int headDim, int seqKv, string label)
    {
        AssertQuantizedMatchesF32(
            numHeads, numKvHeads, headDim, seqKv, seqQ: 1,
            KvCacheDType.Q4_0, KvCacheDType.Q4_0,
            tolerance: 0.15f, label);
    }

    [Theory]
    [InlineData(8, 2, 64, 128, "K8V4 decode, Llama-style GQA 8:2")]
    [InlineData(32, 4, 128, 256, "K8V4 decode, Qwen-style GQA 32:4")]
    [InlineData(16, 4, 64, 200, "K8V4 decode, GQA 16:4")]
    public void Decode_K8V4_MatchesF32(int numHeads, int numKvHeads, int headDim, int seqKv, string label)
    {
        AssertQuantizedMatchesF32(
            numHeads, numKvHeads, headDim, seqKv, seqQ: 1,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0,
            tolerance: 0.15f, label);
    }

    [Theory]
    [InlineData(8, 2, 64, 128, 16, "Q8_0 prefill, Llama-style GQA 8:2")]
    [InlineData(32, 4, 128, 96, 32, "Q8_0 prefill, Qwen-style GQA 32:4")]
    public void Prefill_Q8_0_MatchesF32(int numHeads, int numKvHeads, int headDim, int seqKv, int seqQ, string label)
    {
        AssertQuantizedMatchesF32(
            numHeads, numKvHeads, headDim, seqKv, seqQ,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0,
            tolerance: 0.05f, label);
    }

    [Theory]
    [InlineData(8, 2, 64, 128, 16, "K8V4 prefill, Llama-style GQA 8:2")]
    [InlineData(16, 4, 64, 200, 8, "K8V4 prefill, GQA 16:4")]
    public void Prefill_K8V4_MatchesF32(int numHeads, int numKvHeads, int headDim, int seqKv, int seqQ, string label)
    {
        AssertQuantizedMatchesF32(
            numHeads, numKvHeads, headDim, seqKv, seqQ,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0,
            tolerance: 0.15f, label);
    }

    [Fact]
    public void Decode_Q8_0_HeadDim32_SmallGQA()
    {
        AssertQuantizedMatchesF32(
            numHeads: 4, numKvHeads: 2, headDim: 32, seqKv: 96, seqQ: 1,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0,
            tolerance: 0.05f, "Q8_0 decode, headDim=32 (smallest valid), GQA 4:2");
    }

    [Fact]
    public void Decode_Q8_0_HeadDim256_WideHead()
    {
        AssertQuantizedMatchesF32(
            numHeads: 4, numKvHeads: 2, headDim: 256, seqKv: 64, seqQ: 1,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0,
            tolerance: 0.05f, "Q8_0 decode, headDim=256 (wide), GQA 4:2");
    }

    [Fact]
    public void Decode_WindowedQuantizedCache_MatchesF32()
    {
        const int numHeads = 8, numKvHeads = 2, headDim = 64;
        const int seqKv = 160;
        const int windowSize = 32;

        AssertQuantizedMatchesF32WithWindow(
            numHeads, numKvHeads, headDim, seqKv, seqQ: 1, windowSize,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0,
            tolerance: 0.05f,
            "Q8_0 decode with window=32, GQA 8:2, seqKv=160 (128 quantized + 32 window)");
    }

    [Fact]
    public void Decode_WindowedK8V4_MatchesF32()
    {
        const int numHeads = 8, numKvHeads = 2, headDim = 64;
        const int seqKv = 160;
        const int windowSize = 32;

        AssertQuantizedMatchesF32WithWindow(
            numHeads, numKvHeads, headDim, seqKv, seqQ: 1, windowSize,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0,
            tolerance: 0.15f,
            "K8V4 decode with window=32, GQA 8:2, seqKv=160");
    }

    [Fact]
    public void GroupBroadcast_Discriminating_GQA8to2()
    {
        const int numHeads = 8, numKvHeads = 2, headDim = 64;
        const int seqKv = 96;
        var rng = new Random(77);
        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;

        float[] kData = RandomArray(rng, seqKv * kvStride);
        float[] vData = RandomArray(rng, seqKv * kvStride);
        float[] qData = RandomArray(rng, 1 * qStride);

        using var fpCache = new SimpleKvCache(NumLayers, numKvHeads, headDim, seqKv);
        FillCacheF32(fpCache, kData, vData, seqKv, kvStride);

        using var quantCache = new QuantizedKvCache(NumLayers, numKvHeads, headDim, seqKv,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);
        FillCacheF32(quantCache, kData, vData, seqKv, kvStride);

        float* qPtr = (float*)NativeMemory.AlignedAlloc((nuint)(qStride * sizeof(float)), 64);
        float* outFp = (float*)NativeMemory.AlignedAlloc((nuint)(qStride * sizeof(float)), 64);
        float* outQuant = (float*)NativeMemory.AlignedAlloc((nuint)(qStride * sizeof(float)), 64);
        try
        {
            qData.AsSpan().CopyTo(new Span<float>(qPtr, qStride));

            var kRef = fpCache.GetKeysRef(0);
            var vRef = fpCache.GetValuesRef(0);
            Attention.Execute(qPtr, (float*)kRef.DataPointer, (float*)vRef.DataPointer, outFp,
                1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool: null);

            Attention.Execute(qPtr, quantCache, 0, outQuant,
                1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool: null);

            for (int h = 0; h < numHeads; h++)
            {
                int kvHead = h / (numHeads / numKvHeads);
                int otherKvHead = (kvHead + 1) % numKvHeads;

                for (int d = 0; d < headDim; d++)
                {
                    int idx = h * headDim + d;
                    float fpVal = outFp[idx];
                    float quantVal = outQuant[idx];
                    float err = MathF.Abs(fpVal - quantVal);

                    Assert.True(err < 0.05f,
                        $"Head {h} (kvHead={kvHead}) dim {d}: FP32={fpVal:G6}, quant={quantVal:G6}, err={err:G6}");
                }
            }

            // Verify that different KV head groups produce different outputs (discriminating)
            float maxDiffAcrossGroups = 0f;
            for (int d = 0; d < headDim; d++)
            {
                float h0Val = outQuant[0 * headDim + d]; // head 0, maps to kvHead 0
                float h4Val = outQuant[4 * headDim + d]; // head 4, maps to kvHead 1
                float diff = MathF.Abs(h0Val - h4Val);
                if (diff > maxDiffAcrossGroups) maxDiffAcrossGroups = diff;
            }
            Assert.True(maxDiffAcrossGroups > 0.01f,
                $"Heads in different KV groups should produce different outputs (maxDiff={maxDiffAcrossGroups:G6})");
        }
        finally
        {
            NativeMemory.AlignedFree(qPtr);
            NativeMemory.AlignedFree(outFp);
            NativeMemory.AlignedFree(outQuant);
        }
    }

    private static void AssertQuantizedMatchesF32(
        int numHeads, int numKvHeads, int headDim, int seqKv, int seqQ,
        KvCacheDType keyDType, KvCacheDType valueDType,
        float tolerance, string label)
    {
        var rng = new Random(42);
        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;

        float[] kData = RandomArray(rng, seqKv * kvStride);
        float[] vData = RandomArray(rng, seqKv * kvStride);
        float[] qData = RandomArray(rng, seqQ * qStride);

        using var fpCache = new SimpleKvCache(NumLayers, numKvHeads, headDim, seqKv);
        FillCacheF32(fpCache, kData, vData, seqKv, kvStride);

        using var quantCache = new QuantizedKvCache(NumLayers, numKvHeads, headDim, seqKv,
            keyDType, valueDType, windowSize: 0);
        FillCacheF32(quantCache, kData, vData, seqKv, kvStride);

        int outputSize = seqQ * qStride;
        float* qPtr = (float*)NativeMemory.AlignedAlloc((nuint)(seqQ * qStride * sizeof(float)), 64);
        float* outFp = (float*)NativeMemory.AlignedAlloc((nuint)(outputSize * sizeof(float)), 64);
        float* outQuant = (float*)NativeMemory.AlignedAlloc((nuint)(outputSize * sizeof(float)), 64);
        try
        {
            qData.AsSpan().CopyTo(new Span<float>(qPtr, seqQ * qStride));

            var kRef = fpCache.GetKeysRef(0);
            var vRef = fpCache.GetValuesRef(0);
            int positionOffset = seqKv - seqQ;
            Attention.Execute(qPtr, (float*)kRef.DataPointer, (float*)vRef.DataPointer, outFp,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool: null);

            Attention.Execute(qPtr, quantCache, 0, outQuant,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool: null);

            float maxErr = 0f;
            int worstIdx = 0;
            for (int i = 0; i < outputSize; i++)
            {
                float err = MathF.Abs(outFp[i] - outQuant[i]);
                if (err > maxErr) { maxErr = err; worstIdx = i; }
            }

            Assert.True(maxErr < tolerance,
                $"[{label}] Max error {maxErr:G6} at index {worstIdx} exceeds tolerance {tolerance}. " +
                $"FP32={outFp[worstIdx]:G6}, quant={outQuant[worstIdx]:G6}");
        }
        finally
        {
            NativeMemory.AlignedFree(qPtr);
            NativeMemory.AlignedFree(outFp);
            NativeMemory.AlignedFree(outQuant);
        }
    }

    private static void AssertQuantizedMatchesF32WithWindow(
        int numHeads, int numKvHeads, int headDim, int seqKv, int seqQ, int windowSize,
        KvCacheDType keyDType, KvCacheDType valueDType,
        float tolerance, string label)
    {
        var rng = new Random(42);
        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;

        float[] kData = RandomArray(rng, seqKv * kvStride);
        float[] vData = RandomArray(rng, seqKv * kvStride);
        float[] qData = RandomArray(rng, seqQ * qStride);

        using var fpCache = new SimpleKvCache(NumLayers, numKvHeads, headDim, seqKv);
        FillCacheF32(fpCache, kData, vData, seqKv, kvStride);

        using var quantCache = new QuantizedKvCache(NumLayers, numKvHeads, headDim, seqKv,
            keyDType, valueDType, windowSize);
        FillCacheF32(quantCache, kData, vData, seqKv, kvStride);

        int outputSize = seqQ * qStride;
        float* qPtr = (float*)NativeMemory.AlignedAlloc((nuint)(seqQ * qStride * sizeof(float)), 64);
        float* outFp = (float*)NativeMemory.AlignedAlloc((nuint)(outputSize * sizeof(float)), 64);
        float* outQuant = (float*)NativeMemory.AlignedAlloc((nuint)(outputSize * sizeof(float)), 64);
        try
        {
            qData.AsSpan().CopyTo(new Span<float>(qPtr, seqQ * qStride));

            var kRef = fpCache.GetKeysRef(0);
            var vRef = fpCache.GetValuesRef(0);
            int positionOffset = seqKv - seqQ;
            Attention.Execute(qPtr, (float*)kRef.DataPointer, (float*)vRef.DataPointer, outFp,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool: null);

            Attention.Execute(qPtr, quantCache, 0, outQuant,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool: null);

            float maxErr = 0f;
            int worstIdx = 0;
            for (int i = 0; i < outputSize; i++)
            {
                float err = MathF.Abs(outFp[i] - outQuant[i]);
                if (err > maxErr) { maxErr = err; worstIdx = i; }
            }

            Assert.True(maxErr < tolerance,
                $"[{label}] Max error {maxErr:G6} at index {worstIdx} exceeds tolerance {tolerance}. " +
                $"FP32={outFp[worstIdx]:G6}, quant={outQuant[worstIdx]:G6}");
        }
        finally
        {
            NativeMemory.AlignedFree(qPtr);
            NativeMemory.AlignedFree(outFp);
            NativeMemory.AlignedFree(outQuant);
        }
    }

    private static void FillCacheF32(IKvCache cache, float[] kData, float[] vData, int seqLen, int kvStride)
    {
        int rowBytes = kvStride * sizeof(float);
        for (int pos = 0; pos < seqLen; pos++)
        {
            nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
            nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
            try
            {
                kData.AsSpan(pos * kvStride, kvStride).CopyTo(new Span<float>((void*)kPtr, kvStride));
                vData.AsSpan(pos * kvStride, kvStride).CopyTo(new Span<float>((void*)vPtr, kvStride));

                var kRef = new TensorRef(1, kvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, kvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [pos], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }
    }

    private static float[] RandomArray(Random rng, int length)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = (rng.NextSingle() * 2f - 1f) * 2f;
        return data;
    }
}
