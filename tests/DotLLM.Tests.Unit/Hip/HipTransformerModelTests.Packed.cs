using System.Buffers.Binary;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

public sealed partial class HipTransformerModelTests
{
    [Fact]
    public unsafe void Q4FixturePacker_UsesGgufHalfBlockNibblesAndCpuDecodesExactly()
    {
        var source = new Half[32];
        for (int i = 0; i < 16; i++) { source[i] = (Half)(i - 8); source[i + 16] = (Half)(7 - i); }
        byte[] packed = PackFixtureWeights(source, QuantizationType.Q4_0);
        Assert.Equal(18, packed.Length);
        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)1), BinaryPrimitives.ReadUInt16LittleEndian(packed));
        for (int i = 0; i < 16; i++) Assert.Equal((byte)(i | (15 - i) << 4), packed[i + 2]);
        var decoded = new float[32];
        fixed (byte* pointer = packed) Dequantize.ToFloat32((nint)pointer, 32, QuantizationType.Q4_0, decoded);
        Assert.Equal(source.Select(value => (float)value), decoded);
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 32, false)]
    [InlineData(QuantizationType.Q4_0, 64, false)]
    [InlineData(QuantizationType.Q4_0, 32, true)]
    [InlineData(QuantizationType.Q4_0, 64, true)]
    [InlineData(QuantizationType.Q8_0, 32, false)]
    [InlineData(QuantizationType.Q8_0, 64, false)]
    [InlineData(QuantizationType.Q8_0, 32, true)]
    [InlineData(QuantizationType.Q8_0, 64, true)]
    [Trait("Category", "GPU")]
    public void PackedModel_WithF16KvMatchesCpuAcrossPrefillDecodeAndRollback(QuantizationType type, int headDim, bool tied)
    {
        RequireGpu();
        using var environment = new HipQ8Environment(null);
        Fixture fixture = CreateFixture(tied, matrixType: type, headDimension: headDim);
        using var model = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(DecodedReferenceFixture(fixture));
        using var packedCpu = new CpuOracle(fixture);
        using var cache = model.CreateKvCache(LogicalContext, PageTokens);
        long compute = model.ComputeMemoryBytes;
        AssertPackedWeightBytes(model, fixture, type, tied);
        int[] history = Tokens(17, 3);
        using (var actual = model.Forward(history, Positions(0, history.Length), 0, cache, true))
            Compare(ReadLast(actual), history, "prefill");
        AssertPackedCacheBytes(cache, fixture.Config, 17, 8);
        for (int token = 47; token <= 49; token++)
        {
            int position = history.Length;
            history = [.. history, token];
            using var actual = model.Forward([token], [position], 0, cache);
            Compare(ReadLast(actual), history, $"decode {position}");
        }
        cache.Rollback(6);
        AssertPackedCacheBytes(cache, fixture.Config, 6, 8);
        int[] replacement = Tokens(11, 19);
        history = [.. history[..6], .. replacement];
        using (var actual = model.Forward(replacement, Positions(6, replacement.Length), 0, cache, true))
            Compare(ReadLast(actual), history, "rollback");
        Assert.Equal(compute, model.ComputeMemoryBytes);
        AssertPackedWeightBytes(model, fixture, type, tied);
        cache.Rollback(0);
        Assert.Equal(0, cache.AllocatedBytes);
        using (var actual = model.Forward([31], [0], 0, cache))
            Compare(ReadLast(actual), [31], "reuse");
        output.WriteLine($"Packed fixture: {fixture.Path}; stored={model.StoredWeightBytes}; allocated={model.WeightMemoryBytes}; compute={compute}.");

        void Compare(float[] actual, int[] tokens, string phase)
        {
            // Packed embedding values need an extra Half rounding that the exact F32
            // CPU reference does not perform. Allow reduction-length-scaled F16
            // roundoff at cancellation logits; tied heads use the larger embedding
            // weight amplitude. The global relative-L2 limit stays 0.003 for all cases.
            double absolute = 0.003 + Math.ScaleB(1, -10) * Math.Sqrt(fixture.Config.HiddenSize) * (tied ? 0.4 : 0.045);
            Close(oracle.Last(tokens), actual, $"{type} head={headDim} tied={tied} {phase}, exact decoded weights", absolute);
            PackedCpuQuality(packedCpu.Last(tokens), actual, $"{type} head={headDim} tied={tied} {phase}, CPU Q8 activations");
        }
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(QuantizationType.Q4_0, KvCacheDType.Q8_0, KvCacheDType.F32)]
    [InlineData(QuantizationType.Q4_0, KvCacheDType.F32, KvCacheDType.Q8_0)]
    [InlineData(QuantizationType.Q8_0, KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(QuantizationType.Q8_0, KvCacheDType.Q8_0, KvCacheDType.F32)]
    [InlineData(QuantizationType.Q8_0, KvCacheDType.F32, KvCacheDType.Q8_0)]
    [Trait("Category", "GPU")]
    public void PackedModel_MixedKvTracksCpuQuantizedCacheAndIsolatesRequests(
        QuantizationType type, KvCacheDType keyType, KvCacheDType valueType)
    {
        RequireGpu();
        using var environment = new HipQ8Environment("1");
        Fixture fixture = CreateFixture(false, matrixType: type, headDimension: 64);
        using var model = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(DecodedReferenceFixture(fixture));
        using var packedCpu = new CpuOracle(fixture);
        using var a = model.CreateKvCache(LogicalContext, PageTokens, keyType, valueType);
        using var b = model.CreateKvCache(LogicalContext, PageTokens, keyType, valueType);
        Assert.Equal(keyType, a.KeyType);
        Assert.Equal(valueType, a.ValueType);
        Assert.Equal(0, a.AllocatedBytes + b.AllocatedBytes);
        long compute = model.ComputeMemoryBytes;
        int[] historyA = Tokens(17, 3), historyB = Tokens(7, 29);
        using (var first = model.Forward(historyA, Positions(0, historyA.Length), 0, a, true))
            CheckQuantizedHistory(ReadLast(first), historyA, "A prefill");
        using (var second = model.Forward(historyB, Positions(0, historyB.Length), 0, b, true))
            CheckQuantizedHistory(ReadLast(second), historyB, "B prefill");
        AssertPackedCacheBytes(a, fixture.Config, 17, 8);
        AssertPackedCacheBytes(b, fixture.Config, 7, 2);

        for (int step = 0; step < 3; step++)
        {
            int positionA = historyA.Length, positionB = historyB.Length;
            int tokenA = 47 + step, tokenB = 59 + step;
            historyA = [.. historyA, tokenA]; historyB = [.. historyB, tokenB];
            using var nextB = model.Forward([tokenB], [positionB], 0, b);
            using var nextA = model.Forward([tokenA], [positionA], 0, a);
            CheckQuantizedHistory(ReadLast(nextA), historyA, $"A decode {step}");
            CheckQuantizedHistory(ReadLast(nextB), historyB, $"B decode {step}");
        }
        AssertPackedCacheBytes(a, fixture.Config, 20, 8);
        AssertPackedCacheBytes(b, fixture.Config, 10, 4);
        a.Rollback(6);
        int[] replacement = Tokens(11, 71);
        historyA = [.. historyA[..6], .. replacement];
        using (var actual = model.Forward(replacement, Positions(6, replacement.Length), 0, a, true))
            CheckQuantizedHistory(ReadLast(actual), historyA, "rollback overwrite");
        AssertPackedCacheBytes(a, fixture.Config, 17, 8);
        Assert.Equal(compute, model.ComputeMemoryBytes);
        AssertPackedWeightBytes(model, fixture, type, tied: false);
        a.Rollback(0);
        using (var actual = model.Forward([31], [0], 0, a))
            CheckQuantizedHistory(ReadLast(actual), [31], "reuse");
        AssertPackedCacheBytes(a, fixture.Config, 1, 1);
        Assert.Equal(historyB.Length, b.CurrentLength);
        output.WriteLine($"{type} K={keyType} V={valueType}: KV A={a.AllocatedBytes}, B={b.AllocatedBytes}, fixed compute={compute}, packed weights={model.StoredWeightBytes}.");

        void CheckQuantizedHistory(float[] actual, int[] history, string phase)
        {
            // The exact decoded-weight CPU model avoids its packed path's additional
            // Q8 activation quantization. The test cache independently narrows each plane.
            QuantizedClose(oracle.Last(history, keyType, valueType, referenceKv: true), actual,
                $"{type} K={keyType} V={valueType} {phase}, independently quantized KV", 0.006);
            QuantizedClose(oracle.Last(history), actual,
                $"{type} K={keyType} V={valueType} {phase}, unquantized KV quality", 0.012);
            PackedCpuQuality(packedCpu.Last(history), actual,
                $"{type} K={keyType} V={valueType} {phase}, CPU Q8 activations");
        }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void PackedCache_RejectsQ4AndUnalignedHeadGeometryBeforeAllocatingPages()
    {
        RequireGpu();
        using var environment = new HipQ8Environment("1");
        Fixture aligned = CreateFixture(false, matrixType: QuantizationType.Q8_0, headDimension: 32);
        using var model = LoadHipAndCloseMapping(aligned);
        Assert.Throws<NotSupportedException>(() => { using var rejected = model.CreateKvCache(128, PageTokens, KvCacheDType.Q4_0, KvCacheDType.F32); });
        Assert.Throws<NotSupportedException>(() => { using var rejected = model.CreateKvCache(128, PageTokens, KvCacheDType.F32, KvCacheDType.Q4_0); });
        // kvDim=96 is block-aligned, but each 48-channel head is not.
        Fixture unaligned = CreateFixture(false, headDimension: 48);
        using var unalignedModel = LoadHipAndCloseMapping(unaligned);
        Assert.Throws<NotSupportedException>(() => { using var rejected = unalignedModel.CreateKvCache(128, PageTokens, KvCacheDType.Q8_0, KvCacheDType.F32); });
        using var valid = model.CreateKvCache(128, PageTokens, KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        Assert.Equal(0, valid.AllocatedBytes);
    }

    [Theory]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q8_0)]
    public void PackedMatrix_RejectsInputRowsThatCrossBlockBoundariesBeforeDeviceInitialization(QuantizationType type)
    {
        Fixture fixture = CreateFixture(false, matrixType: type, headDimension: 32, intermediate: 240);
        using var file = GgufFile.Open(fixture.Path);
        var error = Assert.Throws<NotSupportedException>(() => { using var rejected = HipTransformerModel.LoadFromGguf(file, fixture.Config, deviceId: -1); });
        Assert.Contains("input dimension divisible by 32", error.Message, StringComparison.Ordinal);
    }

    private void QuantizedClose(float[] expected, float[] actual, string context, double relativeLimit)
    {
        double squaredError = 0, squaredReference = 0, maxError = 0;
        double maximumExpected = expected.Max(), maximumActual = actual.Max();
        double expectedSum = expected.Sum(value => Math.Exp(value - maximumExpected));
        double actualSum = actual.Sum(value => Math.Exp(value - maximumActual));
        double kl = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{context}: nonfinite logit {i}");
            double error = Math.Abs(expected[i] - actual[i]);
            maxError = Math.Max(maxError, error);
            squaredError += error * error;
            squaredReference += expected[i] * expected[i];
            Assert.True(error <= 0.006 + 0.006 * Math.Abs(expected[i]), $"{context}: logit {i}, expected {expected[i]:R}, actual {actual[i]:R}");
            double p = Math.Exp(expected[i] - maximumExpected) / expectedSum;
            double q = Math.Exp(actual[i] - maximumActual) / actualSum;
            kl += p * Math.Log(p / q);
        }
        double relative = Math.Sqrt(squaredError / squaredReference);
        Assert.True(relative <= relativeLimit, $"{context}: relative L2 {relative:G6} exceeds {relativeLimit:G6}");
        Assert.True(kl <= 0.00002, $"{context}: KL {kl:G6}");
        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
        output.WriteLine($"{context}: max abs={maxError:G6}, relative L2={relative:G6}, KL={kl:G6}");
    }

    private void PackedCpuQuality(float[] expected, float[] actual, string context)
    {
        // The production CPU packed GEMV dynamically quantizes activations to Q8_0;
        // HIP instead reads F16 activations. This is a separate quality comparison,
        // not the tight implementation oracle above.
        double error = 0, reference = 0, maxError = 0, kl = 0;
        double expectedMax = expected.Max(), actualMax = actual.Max();
        double expectedLogSum = Math.Log(expected.Sum(value => Math.Exp(value - expectedMax)));
        double actualLogSum = Math.Log(actual.Sum(value => Math.Exp(value - actualMax)));
        for (int i = 0; i < expected.Length; i++)
        {
            double delta = actual[i] - expected[i];
            error += delta * delta;
            reference += expected[i] * expected[i];
            maxError = Math.Max(maxError, Math.Abs(delta));
            double logP = expected[i] - expectedMax - expectedLogSum;
            double logQ = actual[i] - actualMax - actualLogSum;
            kl += Math.Exp(logP) * (logP - logQ);
        }
        double relative = Math.Sqrt(error / reference);
        Assert.True(relative <= 0.025, $"{context}: relative L2 {relative:G6}");
        Assert.True(kl <= 0.001, $"{context}: KL {kl:G6}");
        output.WriteLine($"{context}: max abs={maxError:G6}, relative L2={relative:G6}, KL={kl:G6}, top1 equal={Array.IndexOf(expected, expected.Max()) == Array.IndexOf(actual, actual.Max())}");
    }

    private static Fixture DecodedReferenceFixture(Fixture packed)
    {
        // Test-only expansion provides the exact same decoded weights while avoiding
        // the CPU packed matmul's additional Q8_0 activation quantization. Production
        // loading and inference retain the original packed payloads throughout.
        using var source = GgufFile.Open(packed.Path);
        var writer = new GgufWriter().AddString("general.architecture", "llama");
        long bytes = 0;
        foreach (GgufTensorDescriptor tensor in source.Tensors)
        {
            int count = checked((int)tensor.Shape.ElementCount);
            var values = new float[count];
            Dequantize.ToFloat32(source.TensorDataPointer(tensor.Name), count, tensor.QuantizationType, values);
            byte[] payload = MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
            bytes += payload.Length;
            Assert.Equal(0, payload.Length % 32);
            writer.AddTensor(tensor.Name, tensor.Shape.Dimensions.ToArray(), (uint)QuantizationType.F32, payload);
        }
        string path = packed.Path + ".decoded-reference.gguf";
        byte[] data = writer.Build();
        if (!File.Exists(path) || !File.ReadAllBytes(path).AsSpan().SequenceEqual(data)) File.WriteAllBytes(path, data);
        return new Fixture(path, packed.Config, bytes);
    }

    private sealed unsafe class ReferenceKvCache : IKvCache
    {
        private readonly SimpleKvCache _inner;
        private readonly int _stride;
        private readonly KvCacheDType _keyType, _valueType;

        public ReferenceKvCache(ModelConfig config, KvCacheDType keyType, KvCacheDType valueType)
        {
            _inner = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 128);
            _stride = config.NumKvHeads * config.HeadDim;
            _keyType = keyType;
            _valueType = valueType;
        }

        public int CurrentLength => _inner.CurrentLength;
        public int MaxLength => _inner.MaxLength;
        public ITensor GetKeys(int layerIndex) => _inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => _inner.GetValues(layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => _inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => _inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => _inner.Rollback(length);
        public void Dispose() => _inner.Dispose();

        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
            => Update(new TensorRef(positions.Length, _stride, keys.DType, keys.DeviceId, keys.DataPointer),
                new TensorRef(positions.Length, _stride, values.DType, values.DeviceId, values.DataPointer), positions, layerIndex);

        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        {
            using var narrowedKeys = UnmanagedTensor.Allocate(new TensorShape(positions.Length, _stride), DType.Float32);
            using var narrowedValues = UnmanagedTensor.Allocate(new TensorShape(positions.Length, _stride), DType.Float32);
            Narrow(keys, narrowedKeys, _keyType, positions.Length);
            Narrow(values, narrowedValues, _valueType, positions.Length);
            _inner.Update(narrowedKeys, narrowedValues, positions, layerIndex);
        }

        private void Narrow(TensorRef source, UnmanagedTensor destination, KvCacheDType type, int rows)
        {
            // F32 storage exists only in this independent CPU oracle. Values first
            // traverse the F16 activation boundary, then the selected cache format.
            var input = new ReadOnlySpan<float>((void*)source.DataPointer, rows * _stride);
            var output = new Span<float>((void*)destination.DataPointer, rows * _stride);
            for (int i = 0; i < input.Length; i++) output[i] = (float)(Half)input[i];
            if (type == KvCacheDType.F32) return;
            Assert.Equal(KvCacheDType.Q8_0, type);
            var blockBytes = new byte[_stride / 32 * 34];
            for (int row = 0; row < rows; row++)
            {
                Span<float> values = output.Slice(row * _stride, _stride);
                Quantize.FromFloat32(values, _stride, QuantizationType.Q8_0, blockBytes);
                fixed (byte* pointer = blockBytes)
                    Dequantize.ToFloat32((nint)pointer, _stride, QuantizationType.Q8_0, values);
            }
        }
    }

    private static void AssertPackedCacheBytes(HipPagedKvCache cache, ModelConfig config, int length, int tableCapacity)
    {
        int pages = (length + PageTokens - 1) / PageTokens;
        int elements = config.NumKvHeads * config.HeadDim;
        long keyRow = cache.KeyType == KvCacheDType.F32 ? elements * 2L : elements / 32L * 34;
        long valueRow = cache.ValueType == KvCacheDType.F32 ? elements * 2L : elements / 32L * 34;
        Assert.Equal(length, cache.CurrentLength);
        Assert.Equal(pages, cache.AllocatedPages);
        Assert.Equal(pages * PageTokens * config.NumLayers * (keyRow + valueRow)
            + (long)tableCapacity * config.NumLayers * 2 * IntPtr.Size, cache.AllocatedBytes);
    }

    private static void AssertPackedWeightBytes(HipTransformerModel model, Fixture fixture, QuantizationType type, bool tied)
    {
        ModelConfig c = fixture.Config;
        long matrixElements = (long)c.HiddenSize * c.VocabSize * (tied ? 1 : 2)
            + c.NumLayers * (2L * c.HiddenSize * c.HiddenSize + 2L * c.HiddenSize * c.NumKvHeads * c.HeadDim
                + 3L * c.HiddenSize * c.IntermediateSize);
        long normBytes = (long)c.HiddenSize * (2 * c.NumLayers + 1) * sizeof(ushort);
        long expected = matrixElements / 32 * (type == QuantizationType.Q4_0 ? 18 : 34) + normBytes;
        Assert.Equal(expected, fixture.StoredBytes);
        Assert.Equal(expected, model.StoredWeightBytes);
        Assert.True(expected < matrixElements * sizeof(ushort) + normBytes);
        Assert.True(model.WeightMemoryBytes >= expected);
    }

    private static byte[] PackFixtureWeights(Half[] values, QuantizationType type)
    {
        float[] source = values.Select(value => (float)value).ToArray();
        if (type != QuantizationType.Q4_0) return Quantize.FromFloat32(source, source.Length, type);
        // Quantize.FromFloat32 has no GGUF Q4_0 arm. This reference-only packer follows
        // GGUF's signed max / -8 scale and low16/high16 nibble arrangement.
        var data = new byte[source.Length / 32 * 18];
        for (int block = 0; block < source.Length / 32; block++)
        {
            float maxSigned = 0;
            for (int i = 0; i < 32; i++)
                if (Math.Abs(source[block * 32 + i]) > Math.Abs(maxSigned)) maxSigned = source[block * 32 + i];
            float scale = maxSigned / -8f, inverse = scale == 0 ? 0 : 1f / scale;
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(block * 18), BitConverter.HalfToUInt16Bits((Half)scale));
            for (int i = 0; i < 16; i++)
            {
                int low = Math.Clamp((int)(source[block * 32 + i] * inverse + 8.5f), 0, 15);
                int high = Math.Clamp((int)(source[block * 32 + i + 16] * inverse + 8.5f), 0, 15);
                data[block * 18 + 2 + i] = (byte)(low | high << 4);
            }
        }
        return data;
    }
}
