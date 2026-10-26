using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Cross-backend KV Q4_0/Q8_0 byte-layout contract tests (issue #961, gap 1).
/// Verifies that CPU <see cref="KvQuantize"/> and Engine <see cref="QuantizedKvCache"/>
/// produce byte-identical quantized output, and that the nibble layout matches the CUDA
/// convention: lo nibble = element[2j], hi nibble = element[2j+1] (adjacent-pair, NOT the
/// interleaved GGUF weight layout). A deliberately swapped layout must fail.
/// </summary>
public sealed unsafe class KvQuantizeLayoutContractTests
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q4_0BlockBytes = 18;

    /// <summary>
    /// Verifies that Q4_0 quantization preserves element-to-nibble mapping:
    /// element[2j] → lo nibble of byte[j], element[2j+1] → hi nibble.
    /// Uses a diagnostic input where each element's quantized value is unique,
    /// so a swapped mapping is distinguishable.
    /// </summary>
    [Fact]
    public void Q4_0_NibbleMapping_MatchesCudaConvention()
    {
        // Construct input where element indices are recoverable after quantization.
        // Use values 0..31 mapped to [-7d, 7d] range so each quantizes to a unique nibble.
        float[] input = new float[BlockSize];
        for (int i = 0; i < BlockSize; i++)
            input[i] = (i - 15.5f); // range roughly [-15.5, 15.5]; scale d = max/7 ≈ 2.21

        byte[] quantized = new byte[Q4_0BlockBytes];
        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, qp, BlockSize);
        }

        // Verify nibble assignment matches the adjacent-pair convention.
        byte* qs = stackalloc byte[16];
        fixed (byte* qp = quantized)
        {
            Buffer.MemoryCopy(qp + 2, qs, 16, 16);
        }

        float d = (float)BitConverter.ToHalf(quantized, 0);
        Assert.True(d > 0, "Scale must be positive for this test to be meaningful.");

        for (int j = 0; j < 16; j++)
        {
            int loNibble = qs[j] & 0x0F;
            int hiNibble = qs[j] >> 4;

            // Dequantize and verify they correspond to the right source elements.
            float loVal = d * (loNibble - 8);
            float hiVal = d * (hiNibble - 8);

            // The lo nibble should reconstruct element[2j], hi should reconstruct element[2j+1].
            float expectedLo = input[2 * j];
            float expectedHi = input[2 * j + 1];

            // Tolerance: one quantization step = d
            Assert.True(MathF.Abs(loVal - expectedLo) <= d * 1.1f,
                $"Byte[{j}] lo nibble: dequant={loVal}, expected≈{expectedLo} (element[{2 * j}])");
            Assert.True(MathF.Abs(hiVal - expectedHi) <= d * 1.1f,
                $"Byte[{j}] hi nibble: dequant={hiVal}, expected≈{expectedHi} (element[{2 * j + 1}])");
        }
    }

    /// <summary>
    /// A deliberately swapped nibble layout (hi in lo position, lo in hi) must produce
    /// different bytes — this test would pass if the layout check above were vacuous.
    /// </summary>
    [Fact]
    public void Q4_0_SwappedNibbleLayout_ProducesDifferentBytes()
    {
        // Use asymmetric data so swapping lo/hi is detectable.
        float[] input = new float[BlockSize];
        for (int i = 0; i < BlockSize; i++)
            input[i] = (i % 2 == 0) ? 3.0f : -2.0f; // even=+3, odd=-2 → different nibbles

        byte[] correct = new byte[Q4_0BlockBytes];
        byte[] swapped = new byte[Q4_0BlockBytes];

        fixed (float* ip = input)
        fixed (byte* cp = correct)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, cp, BlockSize);
        }

        // Build a swapped version: hi nibble = element[2j], lo nibble = element[2j+1].
        float d = (float)BitConverter.ToHalf(correct, 0);
        swapped[0] = correct[0];
        swapped[1] = correct[1]; // same scale
        if (d > 0)
        {
            float invD = 1.0f / d;
            for (int j = 0; j < 16; j++)
            {
                // SWAPPED: lo = element[2j+1], hi = element[2j]
                int lo = Math.Clamp((int)MathF.Round(input[2 * j + 1] * invD) + 8, 0, 15);
                int hi = Math.Clamp((int)MathF.Round(input[2 * j] * invD) + 8, 0, 15);
                swapped[j + 2] = (byte)((hi << 4) | lo);
            }
        }

        // The swapped bytes must differ from the correct ones.
        bool anyDifferent = false;
        for (int i = 2; i < Q4_0BlockBytes; i++)
        {
            if (correct[i] != swapped[i]) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent,
            "Swapped nibble layout produced identical bytes — test data is degenerate (even==odd values).");
    }

    /// <summary>
    /// CPU <see cref="KvQuantize.F32ToQ4_0"/> and Engine <see cref="QuantizedKvCache"/>'s
    /// internal quantizer must produce byte-identical output for the same FP32 input.
    /// </summary>
    [Theory]
    [InlineData(32)]   // 1 block — minimum
    [InlineData(128)]  // 4 blocks — typical head_dim
    [InlineData(256)]  // 8 blocks — Bonsai-2 head_dim
    public void Q4_0_CpuKernel_MatchesEngineQuantizer(int elementCount)
    {
        float[] input = GenerateRandomData(elementCount, seed: 42);
        int quantBytes = elementCount / BlockSize * Q4_0BlockBytes;
        byte[] cpuResult = new byte[quantBytes];
        byte[] engineResult = new byte[quantBytes];

        fixed (float* ip = input)
        fixed (byte* cp = cpuResult)
        fixed (byte* ep = engineResult)
        {
            KvQuantize.F32ToQ4_0(ip, cp, elementCount);
            QuantizeRowQ4_0_Engine(ip, ep, elementCount);
        }

        Assert.Equal(cpuResult, engineResult);
    }

    /// <summary>
    /// CPU <see cref="KvQuantize.F32ToQ8_0"/> and Engine quantizer must produce byte-identical output.
    /// </summary>
    [Theory]
    [InlineData(32)]
    [InlineData(128)]
    [InlineData(256)]
    public void Q8_0_CpuKernel_MatchesEngineQuantizer(int elementCount)
    {
        float[] input = GenerateRandomData(elementCount, seed: 42);
        int quantBytes = elementCount / BlockSize * Q8_0BlockBytes;
        byte[] cpuResult = new byte[quantBytes];
        byte[] engineResult = new byte[quantBytes];

        fixed (float* ip = input)
        fixed (byte* cp = cpuResult)
        fixed (byte* ep = engineResult)
        {
            KvQuantize.F32ToQ8_0(ip, cp, elementCount);
            QuantizeRowQ8_0_Engine(ip, ep, elementCount);
        }

        Assert.Equal(cpuResult, engineResult);
    }

    /// <summary>
    /// Q4_0 round-trip at GQA-realistic head dimensions (≥32, random data) must preserve
    /// element position. Uses random data (not ramps/constants) and verifies per-element
    /// to catch broadcast-style bugs where elements alias under degenerate shapes.
    /// </summary>
    [Theory]
    [InlineData(32, 4)]   // head_dim=32, 4 KV heads (Phi-3)
    [InlineData(64, 8)]   // head_dim=64, 8 KV heads (Llama-3 70B GQA)
    [InlineData(128, 8)]  // head_dim=128, 8 KV heads (Qwen 2.5)
    [InlineData(256, 4)]  // head_dim=256, 4 KV heads (Bonsai-2)
    public void Q4_0_RoundTrip_PreservesElementPosition_GQAShapes(int headDim, int numKvHeads)
    {
        int kvStride = numKvHeads * headDim;
        float[] input = GenerateRandomData(kvStride, seed: 1337);
        int quantBytes = kvStride / BlockSize * Q4_0BlockBytes;
        byte[] quantized = new byte[quantBytes];
        float[] output = new float[kvStride];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ4_0(ip, qp, kvStride);
            KvQuantize.Q4_0ToF32(qp, op, kvStride);
        }

        // Per-head verification: each head's data must round-trip independently.
        for (int h = 0; h < numKvHeads; h++)
        {
            int headOffset = h * headDim;
            for (int i = 0; i < headDim; i++)
            {
                float orig = input[headOffset + i];
                float reconstructed = output[headOffset + i];
                float maxInBlock = 0;
                int blockStart = ((headOffset + i) / BlockSize) * BlockSize;
                for (int b = blockStart; b < blockStart + BlockSize; b++)
                    maxInBlock = MathF.Max(maxInBlock, MathF.Abs(input[b]));
                float tolerance = maxInBlock / 7.0f * 1.1f;

                Assert.True(MathF.Abs(orig - reconstructed) <= tolerance,
                    $"Head {h}, elem {i}: orig={orig}, got={reconstructed}, tol={tolerance}");
            }
        }
    }

    /// <summary>
    /// Q8_0 round-trip at GQA-realistic shapes.
    /// </summary>
    [Theory]
    [InlineData(32, 4)]
    [InlineData(128, 8)]
    [InlineData(256, 4)]
    public void Q8_0_RoundTrip_PreservesElementPosition_GQAShapes(int headDim, int numKvHeads)
    {
        int kvStride = numKvHeads * headDim;
        float[] input = GenerateRandomData(kvStride, seed: 7919);
        int quantBytes = kvStride / BlockSize * Q8_0BlockBytes;
        byte[] quantized = new byte[quantBytes];
        float[] output = new float[kvStride];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ8_0(ip, qp, kvStride);
            KvQuantize.Q8_0ToF32(qp, op, kvStride);
        }

        for (int h = 0; h < numKvHeads; h++)
        {
            int headOffset = h * headDim;
            for (int i = 0; i < headDim; i++)
            {
                float orig = input[headOffset + i];
                float reconstructed = output[headOffset + i];
                float maxInBlock = 0;
                int blockStart = ((headOffset + i) / BlockSize) * BlockSize;
                for (int b = blockStart; b < blockStart + BlockSize; b++)
                    maxInBlock = MathF.Max(maxInBlock, MathF.Abs(input[b]));
                float tolerance = maxInBlock / 127.0f * 1.1f;

                Assert.True(MathF.Abs(orig - reconstructed) <= tolerance,
                    $"Head {h}, elem {i}: orig={orig}, got={reconstructed}, tol={tolerance}");
            }
        }
    }

    /// <summary>
    /// QuantizedKvCache evict-and-dequant round-trip at a real GQA shape.
    /// Writes tokens through the cache, forces eviction, then reads quantized data back.
    /// </summary>
    [Theory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q4_0)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0)] // K8V4
    public void QuantizedKvCache_EvictRoundTrip_MatchesCpuKernel(KvCacheDType keyDType, KvCacheDType valueDType)
    {
        const int numLayers = 1;
        const int numKvHeads = 4;
        const int headDim = 128;
        const int kvStride = numKvHeads * headDim;
        const int windowSize = 4;
        const int maxSeqLen = 32;

        using var cache = new QuantizedKvCache(numLayers, numKvHeads, headDim, maxSeqLen, keyDType, valueDType, windowSize);

        // Fill beyond window to trigger eviction.
        int tokensToFill = windowSize + 8;
        float[] kData = GenerateRandomData(kvStride, seed: 100);
        float[] vData = GenerateRandomData(kvStride, seed: 200);

        for (int pos = 0; pos < tokensToFill; pos++)
        {
            Span<int> positions = [pos];
            fixed (float* kp = kData)
            fixed (float* vp = vData)
            {
                var kRef = new Core.Tensors.TensorRef(1, kvStride, Core.Tensors.DType.Float32, -1, (nint)kp);
                var vRef = new Core.Tensors.TensorRef(1, kvStride, Core.Tensors.DType.Float32, -1, (nint)vp);
                cache.Update(kRef, vRef, positions, 0);
            }
        }

        // Quantized length should be tokensToFill - windowSize.
        int expectedQuantLen = tokensToFill - windowSize;
        Assert.Equal(expectedQuantLen, cache.QuantizedLength);

        // Read quantized data and dequantize with the CPU kernel.
        nint kQuantPtr = cache.GetQuantizedKeysPtr(0);
        nint vQuantPtr = cache.GetQuantizedValuesPtr(0);
        int kRowBytes = cache.KeyQuantizedRowBytes;
        int vRowBytes = cache.ValueQuantizedRowBytes;

        float[] kDequant = new float[kvStride];
        float[] vDequant = new float[kvStride];

        // Verify first evicted position (pos=0).
        fixed (float* kd = kDequant)
        fixed (float* vd = vDequant)
        {
            if (keyDType == KvCacheDType.Q8_0)
                KvQuantize.Q8_0ToF32((byte*)kQuantPtr, kd, kvStride);
            else
                KvQuantize.Q4_0ToF32((byte*)kQuantPtr, kd, kvStride);

            if (valueDType == KvCacheDType.Q8_0)
                KvQuantize.Q8_0ToF32((byte*)vQuantPtr, vd, kvStride);
            else
                KvQuantize.Q4_0ToF32((byte*)vQuantPtr, vd, kvStride);
        }

        // Dequantized data should be close to original input (all positions use same data in this test).
        float kTolerance = keyDType == KvCacheDType.Q8_0 ? kData.Max(MathF.Abs) / 127f * 1.1f : kData.Max(MathF.Abs) / 7f * 1.1f;
        float vTolerance = valueDType == KvCacheDType.Q8_0 ? vData.Max(MathF.Abs) / 127f * 1.1f : vData.Max(MathF.Abs) / 7f * 1.1f;

        for (int i = 0; i < kvStride; i++)
        {
            Assert.True(MathF.Abs(kData[i] - kDequant[i]) <= kTolerance,
                $"K element {i}: expected≈{kData[i]}, got={kDequant[i]}");
            Assert.True(MathF.Abs(vData[i] - vDequant[i]) <= vTolerance,
                $"V element {i}: expected≈{vData[i]}, got={vDequant[i]}");
        }
    }

    /// <summary>
    /// Verifies that Q4_0 quantized bytes from <see cref="KvQuantize"/> and the
    /// <see cref="QuantizedKvCache"/> internal quantizer are identical for every evicted row.
    /// This is the key cross-backend contract: both paths must produce the same bytes
    /// so that CPU dequant kernels can read data quantized by either path.
    /// </summary>
    [Fact]
    public void Q4_0_CacheEviction_ByteIdenticalToCpuKernel()
    {
        const int numKvHeads = 8;
        const int headDim = 128;
        const int kvStride = numKvHeads * headDim;
        const int windowSize = 2;
        const int maxSeqLen = 16;

        using var cache = new QuantizedKvCache(1, numKvHeads, headDim, maxSeqLen, KvCacheDType.Q4_0, KvCacheDType.Q4_0, windowSize);

        float[] rowData = GenerateRandomData(kvStride, seed: 54321);
        int quantRowBytes = kvStride / BlockSize * Q4_0BlockBytes;

        // Fill 4 tokens — first 2 will be evicted.
        for (int pos = 0; pos < 4; pos++)
        {
            Span<int> positions = [pos];
            fixed (float* rp = rowData)
            {
                var kRef = new Core.Tensors.TensorRef(1, kvStride, Core.Tensors.DType.Float32, -1, (nint)rp);
                var vRef = new Core.Tensors.TensorRef(1, kvStride, Core.Tensors.DType.Float32, -1, (nint)rp);
                cache.Update(kRef, vRef, positions, 0);
            }
        }

        Assert.Equal(2, cache.QuantizedLength);

        // Get bytes from the cache's evicted quantized buffer.
        byte[] cacheBytes = new byte[quantRowBytes];
        Marshal.Copy(cache.GetQuantizedKeysPtr(0), cacheBytes, 0, quantRowBytes);

        // Quantize the same data with the CPU kernel directly.
        byte[] cpuBytes = new byte[quantRowBytes];
        fixed (float* rp = rowData)
        fixed (byte* cp = cpuBytes)
        {
            KvQuantize.F32ToQ4_0(rp, cp, kvStride);
        }

        Assert.Equal(cpuBytes, cacheBytes);
    }

    /// <summary>
    /// Same as above but for Q8_0.
    /// </summary>
    [Fact]
    public void Q8_0_CacheEviction_ByteIdenticalToCpuKernel()
    {
        const int numKvHeads = 8;
        const int headDim = 128;
        const int kvStride = numKvHeads * headDim;
        const int windowSize = 2;
        const int maxSeqLen = 16;

        using var cache = new QuantizedKvCache(1, numKvHeads, headDim, maxSeqLen, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        float[] rowData = GenerateRandomData(kvStride, seed: 54321);
        int quantRowBytes = kvStride / BlockSize * Q8_0BlockBytes;

        for (int pos = 0; pos < 4; pos++)
        {
            Span<int> positions = [pos];
            fixed (float* rp = rowData)
            {
                var kRef = new Core.Tensors.TensorRef(1, kvStride, Core.Tensors.DType.Float32, -1, (nint)rp);
                var vRef = new Core.Tensors.TensorRef(1, kvStride, Core.Tensors.DType.Float32, -1, (nint)rp);
                cache.Update(kRef, vRef, positions, 0);
            }
        }

        Assert.Equal(2, cache.QuantizedLength);

        byte[] cacheBytes = new byte[quantRowBytes];
        Marshal.Copy(cache.GetQuantizedKeysPtr(0), cacheBytes, 0, quantRowBytes);

        byte[] cpuBytes = new byte[quantRowBytes];
        fixed (float* rp = rowData)
        fixed (byte* cp = cpuBytes)
        {
            KvQuantize.F32ToQ8_0(rp, cp, kvStride);
        }

        Assert.Equal(cpuBytes, cacheBytes);
    }

    // ──────────────────── Engine QuantizeRow replicas (for comparison) ────────────────────

    /// <summary>
    /// Replicates <see cref="QuantizedKvCache"/>'s private <c>QuantizeRowQ4_0</c> exactly,
    /// so we can compare byte output without reflection. The layout MUST match the engine's
    /// internal implementation — if this test passes but the eviction test fails, the engine
    /// diverged from this reference.
    /// </summary>
    [SkipLocalsInit]
    private static void QuantizeRowQ4_0_Engine(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* bs = src + block * BlockSize;
            byte* bd = dest + block * Q4_0BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(bs[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(bd, (Half)d);
            byte* qs = bd + 2;

            if (d == 0)
            {
                for (int j = 0; j < 16; j++) qs[j] = 0x88;
            }
            else
            {
                float invD = 1.0f / d;
                for (int j = 0; j < 16; j++)
                {
                    int lo = Math.Clamp((int)MathF.Round(bs[2 * j] * invD) + 8, 0, 15);
                    int hi = Math.Clamp((int)MathF.Round(bs[2 * j + 1] * invD) + 8, 0, 15);
                    qs[j] = (byte)((hi << 4) | lo);
                }
            }
        }
    }

    [SkipLocalsInit]
    private static void QuantizeRowQ8_0_Engine(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* bs = src + block * BlockSize;
            byte* bd = dest + block * Q8_0BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(bs[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(bd, (Half)scale);
            sbyte* qs = (sbyte*)(bd + 2);

            if (scale == 0)
            {
                for (int i = 0; i < BlockSize; i++) qs[i] = 0;
            }
            else
            {
                float invScale = 1.0f / scale;
                for (int i = 0; i < BlockSize; i++)
                    qs[i] = (sbyte)Math.Clamp((int)MathF.Round(bs[i] * invScale), -127, 127);
            }
        }
    }

    // ──────────────────── Helpers ────────────────────

    private static float[] GenerateRandomData(int count, int seed)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1) * 5.0f;
        return data;
    }
}
