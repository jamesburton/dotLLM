using System.Buffers.Binary;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Hip;
using DotLLM.Hip.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
[Trait("Category", "GPU")]
public sealed class HipPackedKernelTests : IDisposable
{
    private readonly HipContext? _context;
    private readonly HipStream? _stream;
    private readonly DenseF16Kernels? _kernels;

    public HipPackedKernelTests()
    {
        if (!HipDevice.IsAvailable()) return;
        _context = HipContext.Create(0);
        try
        {
            _stream = HipStream.Create();
            _kernels = new DenseF16Kernels(_context, _stream);
        }
        catch { _stream?.Dispose(); _context.Dispose(); throw; }
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 32)]
    [InlineData(QuantizationType.Q4_0, 64)]
    [InlineData(QuantizationType.Q4_0, 128)]
    [InlineData(QuantizationType.Q4_0, 256)]
    [InlineData(QuantizationType.Q8_0, 32)]
    [InlineData(QuantizationType.Q8_0, 64)]
    [InlineData(QuantizationType.Q8_0, 128)]
    [InlineData(QuantizationType.Q8_0, 256)]
    public void PackedWeights_RespectSignedScalesNibbleOrderAndNonsquareRows(QuantizationType type, int columns)
    {
        RequireGpu();
        const int rows = 7, embeddingToken = 3;
        byte[] packed = GoldenWeights(type, rows * columns);
        float[] decoded = DecodeWeights(packed, type, rows * columns);
        Half[] input = Enumerable.Range(0, columns).Select(i => (Half)(MathF.Sin(i * 0.37f + 0.2f) * 0.7f)).ToArray();
        using var weights = Buffer(packed);
        using var x = Buffer(input);
        using var logits = Buffer(new Half[rows]);
        using var embedding = Buffer(new Half[columns]);
        _kernels!.Linear(logits.Pointer, x.Pointer, weights.Pointer, rows, columns, type);
        _kernels.Embed(embedding.Pointer, weights.Pointer, embeddingToken, columns, type);
        _stream!.Synchronize();

        Half[] expected = new Half[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int col = 0; col < columns; col++) sum += decoded[row * columns + col] * (double)input[col];
            expected[row] = (Half)sum;
        }
        Close(expected, logits.Read());
        Assert.Equal(decoded.AsSpan(embeddingToken * columns, columns).ToArray().Select(value => (Half)value).ToArray(), embedding.Read());
        Assert.Equal(packed, weights.Read());
        Assert.Equal((long)rows * columns / 32 * (type == QuantizationType.Q4_0 ? 18 : 34), weights.Bytes);
    }

    public static IEnumerable<object[]> KvFormats()
    {
        foreach (int headDim in new[] { 32, 64, 128, 256 })
        {
            yield return [headDim, KvCacheDType.Q8_0, KvCacheDType.Q8_0];
            yield return [headDim, KvCacheDType.F32, KvCacheDType.Q8_0];
            yield return [headDim, KvCacheDType.Q8_0, KvCacheDType.F32];
        }
        yield return [64, KvCacheDType.F32, KvCacheDType.F32];
    }

    [SkippableTheory]
    [MemberData(nameof(KvFormats))]
    public void MixedPagedKv_WritesExactBlocksAndAttendsOnlyVisibleGroupedHistory(
        int headDim, KvCacheDType keyType, KvCacheDType valueType)
    {
        RequireGpu();
        const int queryHeads = 6, kvHeads = 2, pageTokens = 3, length = 8, pageCount = 3;
        int kvDim = kvHeads * headDim;
        int keyRowBytes = RowBytes(kvDim, keyType), valueRowBytes = RowBytes(kvDim, valueType);
        using var keyPages = new Pages(_context!, pageCount, pageTokens * keyRowBytes);
        using var valuePages = new Pages(_context!, pageCount, pageTokens * valueRowBytes);
        var referenceKeys = new float[length][];
        var referenceValues = new float[length][];
        for (int token = 0; token < length; token++)
        {
            Half[] key = SourceRow(token, kvDim, false), value = SourceRow(token, kvDim, true);
            byte[] encodedKey = EncodeKv(key, keyType), encodedValue = EncodeKv(value, valueType);
            referenceKeys[token] = DecodeKv(encodedKey, keyType, kvDim);
            referenceValues[token] = DecodeKv(encodedValue, valueType, kvDim);
            using var k = Buffer(key);
            using var v = Buffer(value);
            _kernels!.WriteKv(keyPages.Table.Pointer, valuePages.Table.Pointer, k.Pointer, v.Pointer,
                pageTokens, kvHeads, headDim, token, keyType, valueType);
            _stream!.Synchronize();
            Assert.Equal(encodedKey, keyPages.ReadRow(token / pageTokens, token % pageTokens, keyRowBytes));
            Assert.Equal(encodedValue, valuePages.ReadRow(token / pageTokens, token % pageTokens, valueRowBytes));
        }

        // The ninth row is poison, so reading capacity rather than visible length corrupts logits.
        Assert.All(keyPages.ReadRow(2, 2, keyRowBytes), value => Assert.Equal(0xff, value));
        Assert.All(valuePages.ReadRow(2, 2, valueRowBytes), value => Assert.Equal(0xff, value));
        Half[] query = Enumerable.Range(0, queryHeads * headDim)
            .Select(i => (Half)(0.45f * MathF.Sin(i * 0.19f + 0.4f))).ToArray();
        using var q = Buffer(query);
        using var output = Buffer(new Half[queryHeads * headDim]);
        float scale = 1f / MathF.Sqrt(headDim);
        _kernels!.Attention(output.Pointer, q.Pointer, keyPages.Table.Pointer, valuePages.Table.Pointer,
            pageTokens, queryHeads, kvHeads, headDim, length, scale, keyType, valueType);
        _stream!.Synchronize();
        Close(AttentionReference(query, referenceKeys, referenceValues, queryHeads, kvHeads, headDim, scale), output.Read());
        keyPages.CheckGuards();
        valuePages.CheckGuards();
        Assert.Equal((long)pageCount * pageTokens * (keyRowBytes + valueRowBytes),
            keyPages.PayloadBytes + valuePages.PayloadBytes);
    }

    [SkippableFact]
    public void PackedBoundariesRejectUnsupportedTypesAndPartialBlocksBeforeLaunch()
    {
        RequireGpu();
        Assert.Throws<ArgumentException>(() => _kernels!.Linear(1, 1, 1, 7, 48, QuantizationType.Q4_0));
        Assert.Throws<ArgumentException>(() => _kernels!.Embed(1, 1, 1, 31, QuantizationType.Q8_0));
        Assert.Throws<NotSupportedException>(() => _kernels!.Linear(1, 1, 1, 7, 32, QuantizationType.BF16));
        Assert.Throws<NotSupportedException>(() => _kernels!.Embed(1, 1, 1, 32, QuantizationType.Q5_0));
        Assert.Throws<NotSupportedException>(() => _kernels!.WriteKv(1, 1, 1, 1, 3, 2, 40, 0, KvCacheDType.Q8_0, KvCacheDType.F32));
        Assert.Throws<NotSupportedException>(() => _kernels!.WriteKv(1, 1, 1, 1, 3, 2, 64, 0, KvCacheDType.Q4_0, KvCacheDType.F32));
        Assert.Throws<NotSupportedException>(() => _kernels!.Attention(1, 1, 1, 1, 3, 6, 2, 288, 8, 0.1f, KvCacheDType.Q8_0, KvCacheDType.Q8_0));
        Assert.Throws<NotSupportedException>(() => _kernels!.Attention(1, 1, 1, 1, 3, 6, 2, 64, 8, 0.1f, KvCacheDType.F32, KvCacheDType.TurboQuant));
        Assert.Throws<ArgumentException>(() => _kernels!.Attention(1, 1, 1, 1, 3, 5, 2, 64, 8, 0.1f, KvCacheDType.Q8_0, KvCacheDType.Q8_0));
    }

    private static byte[] GoldenWeights(QuantizationType type, int count)
    {
        int blockBytes = type == QuantizationType.Q4_0 ? 18 : 34;
        byte[] result = new byte[count / 32 * blockBytes];
        for (int block = 0; block < count / 32; block++)
        {
            Half scale = (Half)(block % 5 == 0 ? 0 : (block % 2 == 0 ? -1 : 1) * (0.004f + block % 7 * 0.003f));
            BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(block * blockBytes), BitConverter.HalfToUInt16Bits(scale));
            if (type == QuantizationType.Q4_0)
                for (int i = 0; i < 16; i++)
                    result[block * blockBytes + 2 + i] = (byte)(((block * 3 + i * 5) % 16) | (((block * 7 + i * 11 + 3) % 16) << 4));
            else
                for (int i = 0; i < 32; i++)
                    result[block * blockBytes + 2 + i] = unchecked((byte)(sbyte)(i == 0 ? -128 : (block * 17 + i * 29) % 255 - 127));
        }
        return result;
    }

    private static float[] DecodeWeights(byte[] bytes, QuantizationType type, int count)
    {
        var result = new float[count];
        int stride = type == QuantizationType.Q4_0 ? 18 : 34;
        for (int block = 0; block < count / 32; block++)
        {
            float scale = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(block * stride)));
            for (int i = 0; i < 32; i++)
            {
                int quant = type == QuantizationType.Q8_0 ? unchecked((sbyte)bytes[block * stride + 2 + i])
                    : ((bytes[block * stride + 2 + i % 16] >> (i < 16 ? 0 : 4)) & 15) - 8;
                result[block * 32 + i] = scale * quant;
            }
        }
        return result;
    }

    private static Half[] SourceRow(int token, int dim, bool value)
    {
        Half[] row = Enumerable.Range(0, dim).Select(i => (Half)(MathF.Sin((i + 7 * token) * (value ? 0.17f : 0.11f))
            * (0.4f + token * 0.07f) + (i / 32 % 3 - 1) * 0.13f)).ToArray();
        if (token == 2) Array.Clear(row, 0, 32);
        if (token == 0 && value)
        {
            // Exactly representable ties, endpoints, signs and a zero block exercise the Q8 writer.
            row[0] = (Half)127; row[1] = (Half)(-127); row[2] = (Half)63.5f;
            row[3] = (Half)(-63.5f); row[4] = (Half)126.5f; row[5] = (Half)(-126.5f);
        }
        return row;
    }

    private static int RowBytes(int dim, KvCacheDType type) => type == KvCacheDType.F32 ? dim * 2 : dim / 32 * 34;

    private static byte[] EncodeKv(Half[] source, KvCacheDType type)
    {
        if (type == KvCacheDType.F32) return MemoryMarshal.AsBytes(source.AsSpan()).ToArray();
        var bytes = new byte[source.Length / 32 * 34];
        for (int block = 0; block < source.Length / 32; block++)
        {
            float maximum = 0;
            for (int i = 0; i < 32; i++) maximum = Math.Max(maximum, Math.Abs((float)source[block * 32 + i]));
            float scale = maximum / 127f, inverse = scale == 0 ? 0 : 1f / scale;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(block * 34), BitConverter.HalfToUInt16Bits((Half)scale));
            for (int i = 0; i < 32; i++)
            {
                int quant = Math.Clamp((int)MathF.Round((float)source[block * 32 + i] * inverse), -127, 127);
                bytes[block * 34 + 2 + i] = unchecked((byte)(sbyte)quant);
            }
        }
        return bytes;
    }

    private static float[] DecodeKv(byte[] bytes, KvCacheDType type, int dim)
        => type == KvCacheDType.F32 ? MemoryMarshal.Cast<byte, Half>(bytes.AsSpan()).ToArray().Select(value => (float)value).ToArray()
            : DecodeWeights(bytes, QuantizationType.Q8_0, dim);

    private static Half[] AttentionReference(Half[] query, float[][] keys, float[][] values,
        int qHeads, int kvHeads, int headDim, float scale)
    {
        var output = new Half[qHeads * headDim];
        for (int head = 0; head < qHeads; head++)
        {
            int kvHead = head / (qHeads / kvHeads);
            var scores = new double[keys.Length];
            for (int token = 0; token < keys.Length; token++)
            {
                for (int d = 0; d < headDim; d++) scores[token] += (double)query[head * headDim + d] * keys[token][kvHead * headDim + d];
                scores[token] *= scale;
            }
            double maximum = scores.Max();
            double denominator = scores.Sum(score => Math.Exp(score - maximum));
            for (int d = 0; d < headDim; d++)
            {
                double sum = 0;
                for (int token = 0; token < keys.Length; token++)
                    sum += Math.Exp(scores[token] - maximum) / denominator * values[token][kvHead * headDim + d];
                output[head * headDim + d] = (Half)sum;
            }
        }
        return output;
    }

    private static void Close(Half[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float error = Math.Abs((float)expected[i] - (float)actual[i]);
            float tolerance = 0.001f + 0.002f * Math.Abs((float)expected[i]);
            Assert.True(float.IsFinite((float)actual[i]) && error <= tolerance,
                $"Element {i}: expected {(float)expected[i]:R}, actual {(float)actual[i]:R}, tolerance {tolerance:R}");
        }
    }

    private void RequireGpu()
    {
        if (Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_REQUIRED") is "1") Assert.NotNull(_kernels);
        Skip.If(_kernels is null, "HIP GPU is unavailable.");
    }

    private DeviceBuffer<T> Buffer<T>(T[] values) where T : unmanaged => new(_context!, values);

    private sealed class Pages : IDisposable
    {
        private const int Guard = 32;
        private readonly DeviceBuffer<byte>[] _physical;
        private readonly int _bytes;
        public DeviceBuffer<nint> Table { get; }
        public long PayloadBytes => (long)_physical.Length * _bytes;

        public Pages(HipContext context, int count, int bytes)
        {
            _bytes = bytes;
            _physical = new DeviceBuffer<byte>[count];
            try
            {
                for (int page = 0; page < count; page++)
                {
                    byte[] data = Enumerable.Repeat((byte)0xcd, bytes + 2 * Guard).ToArray();
                    data.AsSpan(Guard, bytes).Fill(0xff);
                    _physical[page] = new DeviceBuffer<byte>(context, data);
                }
                // Reversed, individually allocated pages disallow assumptions of contiguous physical history.
                Table = new DeviceBuffer<nint>(context, Enumerable.Range(0, count).Select(page => _physical[count - page - 1].Pointer + Guard).ToArray());
            }
            catch { foreach (var page in _physical) page?.Dispose(); throw; }
        }

        public byte[] ReadRow(int page, int row, int rowBytes)
            => _physical[_physical.Length - page - 1].Read().AsSpan(Guard + row * rowBytes, rowBytes).ToArray();

        public void CheckGuards()
        {
            foreach (var page in _physical)
            {
                byte[] bytes = page.Read();
                Assert.All(bytes[..Guard], value => Assert.Equal(0xcd, value));
                Assert.All(bytes[(Guard + _bytes)..], value => Assert.Equal(0xcd, value));
            }
        }

        public void Dispose() { Table.Dispose(); foreach (var page in _physical) page.Dispose(); }
    }

    private sealed unsafe class DeviceBuffer<T> : IDisposable where T : unmanaged
    {
        private readonly HipContext _context;
        private readonly int _length;
        private nint _pointer;
        public nint Pointer => _pointer;
        public long Bytes => (long)_length * sizeof(T);

        public DeviceBuffer(HipContext context, T[] values)
        {
            _context = context; _length = values.Length;
            context.MakeCurrent();
            _pointer = HipDevice.Allocate(Bytes);
            try { fixed (T* p = values) HipDevice.Upload(_pointer, (nint)p, Bytes); }
            catch { Dispose(); throw; }
        }

        public T[] Read()
        {
            _context.MakeCurrent();
            T[] result = new T[_length];
            fixed (T* p = result) HipDevice.Download((nint)p, _pointer, Bytes);
            return result;
        }

        public void Dispose()
        {
            nint pointer = Interlocked.Exchange(ref _pointer, 0);
            if (pointer == 0) return;
            _context.MakeCurrent();
            HipDevice.Free(pointer);
        }
    }

    public void Dispose() { _kernels?.Dispose(); _stream?.Dispose(); _context?.Dispose(); }
}
