using DotLLM.Hip;
using DotLLM.Hip.Interop;
using DotLLM.Hip.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

[Trait("Category", "GPU")]
[Collection("HipGpu")]
public sealed class HipDenseF16KernelTests : IDisposable
{
    private readonly HipContext? _context;
    private readonly HipStream? _stream;
    private readonly DenseF16Kernels? _kernels;

    public HipDenseF16KernelTests()
    {
        if (!HipDevice.IsAvailable()) return;
        _context = HipContext.Create(0);
        try
        {
            _stream = HipStream.Create();
            _kernels = new DenseF16Kernels(_context, _stream);
        }
        catch
        {
            _stream?.Dispose();
            _context.Dispose();
            throw;
        }
    }

    [SkippableFact]
    public void LinearAndEmbedding_RespectNonsquareRowMajorLayout()
    {
        RequireGpu();
        const int inputDim = 37, outputDim = 7;
        Half[] input = Values(inputDim, 0.6f);
        Half[] weights = Values(inputDim * outputDim, 0.2f);
        using var x = Buffer(input);
        using var w = Buffer(weights);
        using var output = Buffer(new Half[outputDim]);
        _kernels!.Linear(output.Pointer, x.Pointer, w.Pointer, outputDim, inputDim);
        _stream!.Synchronize();
        var expected = new Half[outputDim];
        for (int row = 0; row < outputDim; row++)
        {
            double sum = 0;
            for (int col = 0; col < inputDim; col++) sum += (double)input[col] * (double)weights[row * inputDim + col];
            expected[row] = (Half)sum;
        }
        Close(expected, output.Read());

        using var embedding = Buffer(new Half[inputDim]);
        _kernels.Embed(embedding.Pointer, w.Pointer, 3, inputDim);
        _stream.Synchronize();
        Assert.Equal(weights.AsSpan(3 * inputDim, inputDim).ToArray(), embedding.Read());

        Half[] transposed = new Half[weights.Length];
        for (int row = 0; row < outputDim; row++)
            for (int col = 0; col < inputDim; col++)
                transposed[col * outputDim + row] = weights[row * inputDim + col];
        using var wrongWeights = Buffer(transposed);
        _kernels.Linear(output.Pointer, x.Pointer, wrongWeights.Pointer, outputDim, inputDim);
        _stream.Synchronize();
        Assert.Throws<Xunit.Sdk.TrueException>(() => Close(expected, output.Read()));
    }

    [SkippableFact]
    public void NormResidualSwiGluAndLoadConversion_MatchReferenceIncludingTail()
    {
        RequireGpu();
        const int count = 257;
        Half[] input = Values(count, 1.4f);
        Half[] weight = Values(count, 0.3f).Select(x => (Half)((float)x + 0.8f)).ToArray();
        using var x = Buffer(input);
        using var w = Buffer(weight);
        using var output = Buffer(new Half[count]);
        _kernels!.RmsNorm(output.Pointer, x.Pointer, w.Pointer, count, 1e-5f);
        _stream!.Synchronize();
        double mean = input.Sum(value => (double)value * (double)value) / count;
        Half[] expected = input.Select((value, i) => (Half)((double)value * (double)weight[i] / Math.Sqrt(mean + 1e-5))).ToArray();
        Close(expected, output.Read());

        _kernels.Add(x.Pointer, x.Pointer, w.Pointer, count);
        _stream.Synchronize();
        Half[] added = input.Select((value, i) => (Half)((float)value + (float)weight[i])).ToArray();
        Assert.Equal(added, x.Read());

        _kernels.SwiGlu(output.Pointer, x.Pointer, w.Pointer, count);
        _stream.Synchronize();
        expected = added.Select((value, i) => (Half)((double)value / (1 + Math.Exp(-(double)value)) * (double)weight[i])).ToArray();
        Close(expected, output.Read());

        float[] source = Enumerable.Range(0, count).Select(i => MathF.Sin(i * 0.13f) * 3.7f).ToArray();
        using var sourceDevice = Buffer(source);
        _kernels.ConvertF32ToF16(output.Pointer, sourceDevice.Pointer, count);
        _stream.Synchronize();
        Assert.Equal(source.Select(value => (Half)value), output.Read());
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Rope_PartialRotationPreservesTailAndUsesNonzeroPosition(bool neox)
    {
        RequireGpu();
        const int qHeads = 4, kvHeads = 2, headDim = 10, rotaryDim = 6, position = 7;
        Half[] queries = Values(qHeads * headDim, 0.7f), keys = Values(kvHeads * headDim, 0.9f);
        using var q = Buffer(queries);
        using var k = Buffer(keys);
        _kernels!.Rope(q.Pointer, k.Pointer, qHeads, kvHeads, headDim, rotaryDim, position, 10000, 0.5f, neox);
        _stream!.Synchronize();
        Close(Rotate(queries, neox), q.Read());
        Close(Rotate(keys, neox), k.Read());
        Assert.Throws<Xunit.Sdk.TrueException>(() => Close(Rotate(queries, !neox), q.Read()));

        static Half[] Rotate(Half[] input, bool neox)
        {
            Half[] result = (Half[])input.Clone();
            for (int head = 0; head < input.Length / headDim; head++)
                for (int pair = 0; pair < rotaryDim / 2; pair++)
                {
                    int first = head * headDim + (neox ? pair : 2 * pair);
                    int second = neox ? first + rotaryDim / 2 : first + 1;
                    double angle = position * 0.5 * Math.Pow(10000, -2.0 * pair / rotaryDim);
                    result[first] = (Half)((double)input[first] * Math.Cos(angle) - (double)input[second] * Math.Sin(angle));
                    result[second] = (Half)((double)input[first] * Math.Sin(angle) + (double)input[second] * Math.Cos(angle));
                }
            return result;
        }
    }

    [SkippableTheory]
    [InlineData(40)]
    [InlineData(256)]
    public void PagedAttention_UsesGroupedHeadsPartialPagesAndVisibleLength(int headDim)
    {
        RequireGpu();
        const int qHeads = 6, kvHeads = 2, pageTokens = 3, length = 8;
        int kvDim = kvHeads * headDim;
        Half[] query = Values(qHeads * headDim, 1.1f);
        Half[][] keys = Enumerable.Range(0, length).Select(t => Values(kvDim, 0.8f, t * 13)).ToArray();
        Half[][] values = Enumerable.Range(0, length).Select(t => Values(kvDim, 1.3f, t * 71)).ToArray();
        using var q = Buffer(query);
        using var output = Buffer(new Half[query.Length]);
        var kPages = new List<DeviceBuffer<Half>>();
        var vPages = new List<DeviceBuffer<Half>>();
        try
        {
            for (int page = 0; page < 3; page++)
            {
                kPages.Add(Buffer(Enumerable.Repeat(Half.NaN, pageTokens * kvDim).ToArray()));
                vPages.Add(Buffer(Enumerable.Repeat(Half.NaN, pageTokens * kvDim).ToArray()));
            }
            using var kTable = Buffer(kPages.Select(page => page.Pointer).ToArray());
            using var vTable = Buffer(vPages.Select(page => page.Pointer).ToArray());
            for (int token = 0; token < length; token++)
            {
                using var k = Buffer(keys[token]);
                using var v = Buffer(values[token]);
                _kernels!.WriteKv(kTable.Pointer, vTable.Pointer, k.Pointer, v.Pointer, pageTokens, kvDim, token);
                _stream!.Synchronize();
            }
            float scale = 1 / MathF.Sqrt(headDim);
            _kernels!.Attention(output.Pointer, q.Pointer, kTable.Pointer, vTable.Pointer,
                pageTokens, qHeads, kvHeads, headDim, length, scale);
            _stream!.Synchronize();
            Half[] expected = AttentionReference(query, keys, values, qHeads, kvHeads, headDim, scale, false);
            Close(expected, output.Read());
            Half[] wrongGrouping = AttentionReference(query, keys, values, qHeads, kvHeads, headDim, scale, true);
            Assert.Throws<Xunit.Sdk.TrueException>(() => Close(wrongGrouping, output.Read()));

            // The unwritten ninth row contains NaNs: reading page capacity instead of length cannot pass.
            Assert.All(output.Read(), value => Assert.True(Half.IsFinite(value)));
            Assert.Equal(keys[7], kPages[2].Read().AsSpan(kvDim, kvDim).ToArray());
        }
        finally
        {
            foreach (var page in vPages) page.Dispose();
            foreach (var page in kPages) page.Dispose();
        }
    }

    [SkippableFact]
    public void InvalidGeometryAndDisposedResources_FailBeforeLaunch()
    {
        RequireGpu();
        Assert.Throws<ArgumentException>(() => _kernels!.Attention(1, 1, 1, 1, 16, 3, 2, 64, 1, 1));
        Assert.Throws<NotSupportedException>(() => _kernels!.Attention(1, 1, 1, 1, 16, 4, 2, 288, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => _kernels!.Attention(1, 1, 1, 1, 16, 4, 2, 64, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => _kernels!.Rope(1, 1, 4, 2, 64, 63, 0, 10000, 1, false));
        Assert.Throws<ArgumentOutOfRangeException>(() => _kernels!.WriteKv(1, 1, 1, 1, 16, 64, -1));
        _kernels!.Dispose();
        Assert.Throws<ObjectDisposedException>(() => _kernels.Add(1, 1, 1, 1));
        _stream!.Dispose();
        Assert.Throws<ObjectDisposedException>(() => _stream.Synchronize());
        _context!.Dispose();
        Assert.Throws<ObjectDisposedException>(() => _context.MakeCurrent());
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public unsafe void HipBlasGemmEx_UsesLegacyAbiAndPaddedRectangularMatrices(bool transposeA, bool transposeB)
    {
        RequireGpu();
        const int m = 3, n = 5, k = 7, ldc = 7;
        int lda = (transposeA ? k : m) + 2, ldb = (transposeB ? n : k) + 3;
        Half[] a = Enumerable.Repeat(Half.NaN, lda * (transposeA ? m : k)).ToArray();
        Half[] b = Enumerable.Repeat(Half.NaN, ldb * (transposeB ? k : n)).ToArray();
        for (int row = 0; row < m; row++)
            for (int inner = 0; inner < k; inner++)
                a[(transposeA ? inner : row) + (transposeA ? row : inner) * lda] = (Half)((row + 1) * 0.2 - inner * 0.13);
        for (int inner = 0; inner < k; inner++)
            for (int col = 0; col < n; col++)
                b[(transposeB ? col : inner) + (transposeB ? inner : col) * ldb] = (Half)((col + 1) * 0.17 + inner * 0.09);
        using var da = Buffer(a);
        using var db = Buffer(b);
        using var dc = Buffer(Enumerable.Repeat(Half.NaN, ldc * n).ToArray());
        Assert.Equal(0, HipBlasApi.hipblasCreate(out nint handle));
        try
        {
            Assert.Equal(0, HipBlasApi.hipblasSetStream(handle, _stream!.Handle));
            float alpha = 1, beta = 0;
            int status = HipBlasApi.hipblasGemmEx(handle,
                transposeA ? HipBlasApi.HIPBLAS_OP_T : HipBlasApi.HIPBLAS_OP_N,
                transposeB ? HipBlasApi.HIPBLAS_OP_T : HipBlasApi.HIPBLAS_OP_N,
                m, n, k, (nint)(&alpha), da.Pointer, HipBlasApi.HIPBLAS_R_16F, lda,
                db.Pointer, HipBlasApi.HIPBLAS_R_16F, ldb, (nint)(&beta), dc.Pointer,
                HipBlasApi.HIPBLAS_R_16F, ldc, HipBlasApi.HIPBLAS_COMPUTE_32F, HipBlasApi.HIPBLAS_GEMM_DEFAULT);
            Assert.Equal(0, status);
            _stream.Synchronize();
            Half[] actual = dc.Read();
            for (int col = 0; col < n; col++)
            {
                Half[] expected = new Half[m];
                for (int row = 0; row < m; row++)
                {
                    double sum = 0;
                    for (int inner = 0; inner < k; inner++)
                        sum += (double)a[(transposeA ? inner : row) + (transposeA ? row : inner) * lda]
                            * (double)b[(transposeB ? col : inner) + (transposeB ? inner : col) * ldb];
                    expected[row] = (Half)sum;
                }
                Close(expected, actual.AsSpan(col * ldc, m).ToArray());
                Assert.All(actual.AsSpan(col * ldc + m, ldc - m).ToArray(), value => Assert.True(Half.IsNaN(value)));
            }
        }
        finally
        {
            Assert.Equal(0, HipBlasApi.hipblasDestroy(handle));
        }
    }

    [SkippableFact]
    public void MissingOrWrongKernelManifest_FailsClearlyAndAllowsRetry()
    {
        RequireGpu();
        string directory = Path.Combine(Path.GetTempPath(), "dotllm-hip-manifest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            Assert.Throws<FileNotFoundException>(() => new DenseF16Kernels(_context!, _stream!, directory));
            File.Copy(Path.Combine(AppContext.BaseDirectory, "co", "rmsnorm.co"), Path.Combine(directory, "dense_f16.co"));
            for (int i = 0; i < 3; i++)
                Assert.Throws<HipException>(() => new DenseF16Kernels(_context!, _stream!, directory));
            using var valid = new DenseF16Kernels(_context!, _stream!);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [SkippableFact]
    public void DeviceAttributesAndExplicitContextSelection_WorkWithVersionedRuntime()
    {
        RequireGpu();
        HipDevice device = HipDevice.GetDevice(0);
        Assert.True(device.ComputeUnitCount > 0);
        Assert.Contains(device.WarpSize, new[] { 32, 64 });
        if (HipDevice.GetDeviceCount() > 1)
        {
            using var second = HipContext.Create(1);
            HipDriverApi.hipGetDevice(out int current).ThrowOnError();
            Assert.Equal(1, current);
            _context!.MakeCurrent();
            HipDriverApi.hipGetDevice(out current).ThrowOnError();
            Assert.Equal(0, current);
        }
    }

    private void RequireGpu()
    {
        if (Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_REQUIRED") == "1")
            Assert.NotNull(_kernels);
        Skip.If(_kernels is null, "No HIP GPU/runtime available.");
    }

    private DeviceBuffer<T> Buffer<T>(T[] data) where T : unmanaged => new(_context!, data);

    private static Half[] Values(int length, float scale, int offset = 0) =>
        Enumerable.Range(0, length).Select(i => (Half)(Math.Sin((i + offset + 1) * 0.37) * scale)).ToArray();

    private static void Close(Half[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tolerance = 0.002 * (1 + Math.Abs((double)expected[i]));
            Assert.True(double.IsFinite((double)actual[i]) && Math.Abs((double)expected[i] - (double)actual[i]) <= tolerance,
                $"At {i}: expected {expected[i]}, actual {actual[i]}, tolerance {tolerance}.");
        }
    }

    private static Half[] AttentionReference(Half[] q, Half[][] keys, Half[][] values,
        int qHeads, int kvHeads, int headDim, float scale, bool wrongGrouping)
    {
        Half[] output = new Half[q.Length];
        for (int head = 0; head < qHeads; head++)
        {
            int kvHead = wrongGrouping ? head % kvHeads : head / (qHeads / kvHeads);
            double[] scores = new double[keys.Length];
            for (int token = 0; token < keys.Length; token++)
            {
                for (int d = 0; d < headDim; d++) scores[token] += (double)q[head * headDim + d] * (double)keys[token][kvHead * headDim + d];
                scores[token] *= scale;
            }
            double maximum = scores.Max();
            double denominator = scores.Sum(score => Math.Exp(score - maximum));
            for (int d = 0; d < headDim; d++)
            {
                double sum = 0;
                for (int token = 0; token < keys.Length; token++)
                    sum += Math.Exp(scores[token] - maximum) / denominator * (double)values[token][kvHead * headDim + d];
                output[head * headDim + d] = (Half)sum;
            }
        }
        return output;
    }

    private sealed unsafe class DeviceBuffer<T> : IDisposable where T : unmanaged
    {
        private readonly HipContext _context;
        private readonly int _length;
        private nint _pointer;
        public nint Pointer => _pointer;

        public DeviceBuffer(HipContext context, T[] data)
        {
            _context = context;
            _length = data.Length;
            context.MakeCurrent();
            _pointer = HipDevice.Allocate((long)data.Length * sizeof(T));
            try
            {
                fixed (T* source = data) HipDevice.Upload(_pointer, (nint)source, (long)data.Length * sizeof(T));
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public T[] Read()
        {
            _context.MakeCurrent();
            T[] result = new T[_length];
            fixed (T* target = result) HipDevice.Download((nint)target, _pointer, (long)_length * sizeof(T));
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

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _context?.Dispose();
    }
}
