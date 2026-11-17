using DotLLM.Core.Configuration;
using DotLLM.Hip;
using DotLLM.Hip.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
[Trait("Category", "GPU")]
public sealed class HipSafeTensorsKernelTests : IDisposable
{
    private readonly HipContext? _context;
    private readonly HipStream? _stream;
    private readonly DenseF16Kernels? _kernels;

    public HipSafeTensorsKernelTests()
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
    [InlineData(37)]
    [InlineData(257)]
    public void Bf16Weights_EmbeddingProjectionAndNormReadOriginalBits(int columns)
    {
        RequireGpu();
        const int rows = 7, token = 3;
        ushort[] bits = Enumerable.Range(0, rows * columns)
            .Select(i => (ushort)(BitConverter.SingleToUInt32Bits(MathF.Sin(i * 0.31f + 0.2f) * 0.8f) >> 16)).ToArray();
        float[] weights = bits.Select(b => BitConverter.UInt32BitsToSingle((uint)b << 16)).ToArray();
        Half[] input = Input(columns);
        using var x = Buffer(input);
        using var w = Buffer(bits);
        using var output = Buffer(new Half[rows]);
        using var embedding = Buffer(new Half[columns]);
        using var norm = Buffer(new Half[columns]);
        _kernels!.Linear(output.Pointer, x.Pointer, w.Pointer, rows, columns, QuantizationType.BF16);
        _kernels.Embed(embedding.Pointer, w.Pointer, token, columns, QuantizationType.BF16);
        _kernels.RmsNorm(norm.Pointer, x.Pointer, w.Pointer, columns, 1e-5f, QuantizationType.BF16);
        _stream!.Synchronize();
        Half[] expected = Project(input, rows, (row, col) => weights[row * columns + col]);
        Close(expected, output.Read());
        Assert.Equal(weights.AsSpan(token * columns, columns).ToArray().Select(v => (Half)v), embedding.Read());
        double inverse = 1 / Math.Sqrt(input.Sum(v => (double)v * (double)v) / columns + 1e-5);
        Half[] expectedNorm = input.Select((v, i) => (Half)((double)v * inverse * weights[i])).ToArray();
        Close(expectedNorm, norm.Read());
        Assert.Equal(bits, w.Read());

        // Replacing BF16 storage with the numerically same F16 values must break all three typed reads.
        using var wrong = Buffer(weights.Select(v => BitConverter.HalfToUInt16Bits((Half)v)).ToArray());
        _kernels.Linear(output.Pointer, x.Pointer, wrong.Pointer, rows, columns, QuantizationType.BF16);
        _kernels.Embed(embedding.Pointer, wrong.Pointer, token, columns, QuantizationType.BF16);
        _kernels.RmsNorm(norm.Pointer, x.Pointer, wrong.Pointer, columns, 1e-5f, QuantizationType.BF16);
        _stream.Synchronize();
        Assert.Throws<Xunit.Sdk.TrueException>(() => Close(expected, output.Read()));
        Assert.Throws<Xunit.Sdk.TrueException>(() => Close(weights.AsSpan(token * columns, columns).ToArray().Select(v => (Half)v).ToArray(), embedding.Read()));
        Assert.Throws<Xunit.Sdk.TrueException>(() => Close(expectedNorm, norm.Read()));
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Awq4, 32)]
    [InlineData(QuantizationType.Awq4, 64)]
    [InlineData(QuantizationType.Awq4, 128)]
    [InlineData(QuantizationType.Awq4, -1)]
    [InlineData(QuantizationType.Gptq4, 32)]
    [InlineData(QuantizationType.Gptq4, 64)]
    [InlineData(QuantizationType.Gptq4, 128)]
    [InlineData(QuantizationType.Gptq4, -1)]
    public void Packed4_ReadsOriginalProducerLayoutGroupsAndZeroPoints(QuantizationType type, int declaredGroup)
    {
        RequireGpu();
        const int rows = 16, columns = 256;
        int groupSize = declaredGroup == -1 ? columns : declaredGroup;
        int groups = columns / groupSize;
        bool awq = type == QuantizationType.Awq4;
        Half[] input = Input(columns);
        int[] groupIndices = Enumerable.Range(0, columns).Select(i => ((i * 37 + 17) % columns) / groupSize).ToArray();
        Half[] scales = Enumerable.Range(0, groups * rows).Select(i => (Half)(0.01f + i * 0.0017f)).ToArray();
        int[,] zeros = new int[groups, rows], codes = new int[columns, rows];
        for (int group = 0; group < groups; group++)
            for (int row = 0; row < rows; row++) zeros[group, row] = (group * 3 + row * 5) % 16 + (awq ? 0 : 1);
        for (int col = 0; col < columns; col++)
            for (int row = 0; row < rows; row++) codes[col, row] = (col * 7 + row * 11 + col / 3) % 16;
        Half[] expected = Project(input, rows, (row, col) =>
        {
            int group = awq ? col / groupSize : groupIndices[col];
            return (codes[col, row] - zeros[group, row]) * (float)scales[group * rows + row];
        });

        uint[] qweight = new uint[columns * rows / 8], qzeros = new uint[groups * rows / 8];
        // This is the producer's packed-lane -> logical-lane table, independent of the kernel's inverse formula.
        int[] outputOrder = awq ? [0, 2, 4, 6, 1, 3, 5, 7] : [0, 1, 2, 3, 4, 5, 6, 7];
        for (int group = 0; group < groups; group++)
            for (int block = 0; block < rows / 8; block++)
                for (int lane = 0; lane < 8; lane++)
                    qzeros[group * (rows / 8) + block] |= (uint)(zeros[group, block * 8 + outputOrder[lane]] - (awq ? 0 : 1)) << (lane * 4);
        if (awq)
            for (int col = 0; col < columns; col++)
                for (int block = 0; block < rows / 8; block++)
                    for (int lane = 0; lane < 8; lane++)
                        qweight[col * (rows / 8) + block] |= (uint)codes[col, block * 8 + outputOrder[lane]] << (lane * 4);
        else
            for (int block = 0; block < columns / 8; block++)
                for (int row = 0; row < rows; row++)
                    for (int lane = 0; lane < 8; lane++)
                        qweight[block * rows + row] |= (uint)codes[block * 8 + lane, row] << (lane * 4);

        using var x = Buffer(input);
        using var w = Buffer(qweight);
        using var s = Buffer(scales);
        using var z = Buffer(qzeros);
        using var g = Buffer(groupIndices);
        using var output = Buffer(new Half[rows]);
        Run(w.Pointer, z.Pointer, g.Pointer);
        Close(expected, output.Read());
        Assert.Equal(qweight, w.Read()); Assert.Equal(qzeros, z.Read());
        Assert.Equal(scales, s.Read()); Assert.Equal(groupIndices, g.Read());

        if (awq)
        {
            uint[] wrong = new uint[qweight.Length];
            for (int col = 0; col < columns; col++)
                for (int row = 0; row < rows; row++)
                    wrong[col * (rows / 8) + row / 8] |= (uint)codes[col, row] << ((row % 8) * 4);
            using var naiveOutputPacking = Buffer(wrong);
            Run(naiveOutputPacking.Pointer, z.Pointer, g.Pointer);
            Assert.Throws<Xunit.Sdk.TrueException>(() => Close(expected, output.Read()));
        }
        else
        {
            // +1 must be applied after unpacking, including stored nibble 15 decoding to zero point 16.
            uint[] wrong = new uint[qzeros.Length];
            for (int group = 0; group < groups; group++)
                for (int row = 0; row < rows; row++)
                    wrong[group * (rows / 8) + row / 8] |= (uint)(zeros[group, row] & 15) << ((row % 8) * 4);
            using var missingZeroOffset = Buffer(wrong);
            Run(w.Pointer, missingZeroOffset.Pointer, g.Pointer);
            Assert.Throws<Xunit.Sdk.TrueException>(() => Close(expected, output.Read()));
            if (groups > 1)
            {
                using var canonicalGroups = Buffer(Enumerable.Range(0, columns).Select(i => i / groupSize).ToArray());
                Run(w.Pointer, z.Pointer, canonicalGroups.Pointer);
                Assert.Throws<Xunit.Sdk.TrueException>(() => Close(expected, output.Read()));
            }
        }

        void Run(nint weight, nint zero, nint indices)
        {
            _kernels!.LinearPacked4(output.Pointer, x.Pointer, weight, s.Pointer, zero, awq ? 0 : indices,
                rows, columns, declaredGroup, type);
            _stream!.Synchronize();
        }
    }

    [SkippableFact]
    public void TypedBoundaries_RejectInvalidGeometryAndMissingComponentsBeforeLaunch()
    {
        RequireGpu();
        Assert.Throws<NotSupportedException>(() => _kernels!.LinearPacked4(1, 1, 1, 1, 1, 0, 16, 256, 32, QuantizationType.Q4_0));
        Assert.Throws<ArgumentException>(() => _kernels!.LinearPacked4(1, 1, 1, 1, 1, 0, 15, 256, 32, QuantizationType.Awq4));
        Assert.Throws<ArgumentException>(() => _kernels!.LinearPacked4(1, 1, 1, 1, 1, 0, 16, 256, 48, QuantizationType.Awq4));
        Assert.Throws<ArgumentException>(() => _kernels!.LinearPacked4(1, 1, 1, 1, 1, 0, 16, 256, 32, QuantizationType.Gptq4));
        Assert.Throws<ArgumentException>(() => _kernels!.LinearPacked4(1, 1, 1, 1, 0, 0, 16, 256, 32, QuantizationType.Awq4));
        Assert.Throws<ArgumentOutOfRangeException>(() => _kernels!.Embed(1, 1, -1, 37, QuantizationType.BF16));
        Assert.Throws<ArgumentOutOfRangeException>(() => _kernels!.Linear(1, 1, 1, 7, 0, QuantizationType.BF16));
        Assert.Throws<ArgumentOutOfRangeException>(() => _kernels!.RmsNorm(1, 1, 1, 37, float.NaN, QuantizationType.BF16));
        Assert.Throws<NotSupportedException>(() => _kernels!.RmsNorm(1, 1, 1, 37, 1e-5f, QuantizationType.Q4_0));
    }

    private static Half[] Input(int count) => Enumerable.Range(0, count)
        .Select(i => (Half)(MathF.Sin(i * 0.23f) * 0.7f + 0.11f)).ToArray();

    private static Half[] Project(Half[] input, int rows, Func<int, int, float> weight)
    {
        Half[] result = new Half[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int col = 0; col < input.Length; col++) sum += (double)input[col] * weight(row, col);
            result[row] = (Half)sum;
        }
        return result;
    }

    private static void Close(Half[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tolerance = 0.001 + Math.Abs((double)expected[i]) * 0.002;
            Assert.True(Half.IsFinite(actual[i]) && Math.Abs((double)expected[i] - (double)actual[i]) <= tolerance,
                $"At {i}: expected {expected[i]}, actual {actual[i]}, tolerance {tolerance}.");
        }
    }

    private void RequireGpu()
    {
        if (string.Equals(Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_REQUIRED"), "1", StringComparison.Ordinal)) Assert.NotNull(_kernels);
        Skip.If(_kernels is null, "HIP GPU is unavailable.");
    }

    private DeviceBuffer<T> Buffer<T>(T[] values) where T : unmanaged => new(_context!, values);

    private sealed unsafe class DeviceBuffer<T> : IDisposable where T : unmanaged
    {
        private readonly HipContext _context;
        private readonly int _length;
        private nint _pointer;
        public nint Pointer => _pointer;

        public DeviceBuffer(HipContext context, T[] values)
        {
            _context = context; _length = values.Length;
            context.MakeCurrent();
            _pointer = HipDevice.Allocate((long)_length * sizeof(T));
            try { fixed (T* p = values) HipDevice.Upload(_pointer, (nint)p, (long)_length * sizeof(T)); }
            catch { Dispose(); throw; }
        }

        public T[] Read()
        {
            _context.MakeCurrent();
            T[] result = new T[_length];
            fixed (T* p = result) HipDevice.Download((nint)p, _pointer, (long)_length * sizeof(T));
            return result;
        }

        public void Dispose()
        {
            nint pointer = Interlocked.Exchange(ref _pointer, 0);
            if (pointer == 0) return;
            _context.MakeCurrent(); HipDevice.Free(pointer);
        }
    }

    public void Dispose() { _kernels?.Dispose(); _stream?.Dispose(); _context?.Dispose(); }
}
