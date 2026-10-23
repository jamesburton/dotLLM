using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Collection(CudaCollection.Name)]
[Trait("Category", "GPU")]
public sealed unsafe class CudaLfm2KernelTests
{
    private static nint Upload(float[] values, List<nint> buffers)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)(values.Length * 4)).ThrowOnError(); buffers.Add(p);
        fixed (float* src = values) CudaDriverApi.cuMemcpyHtoD_v2(p, (nint)src, (nuint)(values.Length * 4)).ThrowOnError(); return p;
    }
    private static float[] Read(nint p, int count)
    {
        var values = new float[count]; fixed (float* dst = values) CudaDriverApi.cuMemcpyDtoH_v2((nint)dst, p, (nuint)(count * 4)).ThrowOnError(); return values;
    }
    [SkippableTheory]
    [InlineData(3, 33, 1)]
    [InlineData(3, 64, 2)]
    [InlineData(4, 257, 19)]
    [InlineData(3, 2048, 17)]
    public void ShortConvAndRolledStateMatchScalarAcrossChainedChunks(int taps, int hidden, int tokens)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "CUDA unavailable");
        using var context = CudaContext.Create(0); using var stream = CudaStream.Create();
        using var kernels = new CudaLfm2Kernels(Path.Combine(AppContext.BaseDirectory, "ptx"));
        var buffers = new List<nint>(); var rng = new Random(1032 + tokens);
        float[] Random(int count) => Enumerable.Range(0, count).Select(_ => rng.NextSingle() * 2 - 1).ToArray();
        float[] state = Random((taps - 1) * hidden), w = Random(taps * hidden);
        try
        {
            nint ds = Upload(state, buffers), dw = Upload(w, buffers);
            foreach (int t in new[] { tokens, 1 + tokens / 3 })
            {
                float[] proj = Random(t * 3 * hidden), ext = new float[(taps - 1 + t) * hidden], expected = new float[t * hidden];
                state.CopyTo(ext, 0);
                for (int row = 0; row < t; row++) for (int c = 0; c < hidden; c++) ext[(taps - 1 + row) * hidden + c] = proj[row * 3 * hidden + c] * proj[row * 3 * hidden + 2 * hidden + c];
                for (int row = 0; row < t; row++) for (int c = 0; c < hidden; c++)
                {
                    float sum = 0; for (int j = 0; j < taps; j++) sum += w[j * hidden + c] * ext[(row + j) * hidden + c];
                    expected[row * hidden + c] = sum * proj[row * 3 * hidden + hidden + c];
                }
                Array.Copy(ext, t * hidden, state, 0, state.Length);
                nint dp = Upload(proj, buffers), output = Upload(new float[t * hidden], buffers);
                kernels.Conv(ds, dp, dw, output, taps, hidden, t, stream.Handle); stream.Synchronize();
                float[] actual = Read(output, expected.Length), rolled = Read(ds, state.Length);
                for (int i = 0; i < actual.Length; i++) Assert.True(MathF.Abs(expected[i] - actual[i]) < 2e-6f);
                Assert.Equal(state, rolled);
            }
        }
        finally { foreach (nint p in buffers) CudaDriverApi.cuMemFree_v2(p); }
    }
    [SkippableTheory]
    [InlineData(4, 2, 2)]
    [InlineData(64, 4, 9)]
    public void SigmoidSelectionBiasDoesNotContaminateNormalizedWeights(int experts, int topk, int tokens)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "CUDA unavailable");
        using var context = CudaContext.Create(0); using var stream = CudaStream.Create();
        using var kernels = new CudaLfm2Kernels(Path.Combine(AppContext.BaseDirectory, "ptx"));
        var buffers = new List<nint>();
        float[] logits = Enumerable.Range(0, experts * tokens).Select(i => MathF.Sin(i * 1.7f) * 3).ToArray();
        float[] bias = Enumerable.Range(0, experts).Select(i => i == 0 ? 1.5f : MathF.Cos(i) * 0.3f).ToArray();
        try
        {
            nint dl = Upload(logits, buffers), db = Upload(bias, buffers), di = Upload(new float[tokens * topk], buffers), dw = Upload(new float[tokens * topk], buffers);
            kernels.Route(dl, db, di, dw, experts, topk, tokens, stream.Handle); stream.Synchronize();
            int[] actualIndices = Read(di, tokens * topk).Select(BitConverter.SingleToInt32Bits).ToArray(); float[] actual = Read(dw, tokens * topk);
            for (int t = 0; t < tokens; t++)
            {
                float[] sigmoid = Enumerable.Range(0, experts).Select(e => 1f / (1f + MathF.Exp(-logits[t * experts + e]))).ToArray();
                int[] selected = Enumerable.Range(0, experts).OrderByDescending(e => sigmoid[e] + bias[e]).ThenBy(e => e).Take(topk).ToArray();
                float sum = selected.Sum(e => sigmoid[e]);
                for (int j = 0; j < topk; j++) { Assert.Equal(selected[j], actualIndices[t * topk + j]); Assert.True(MathF.Abs(actual[t * topk + j] - sigmoid[selected[j]] / sum) < 2e-7f); }
            }
        }
        finally { foreach (nint p in buffers) CudaDriverApi.cuMemFree_v2(p); }
    }
    [SkippableTheory]
    [InlineData(17, 32)]
    [InlineData(33, 256)]
    [InlineData(2048, 2048)]
    public void PackedQ4GemvMatchesUnquantizedF32Dot(int rows, int cols)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "CUDA unavailable");
        using var context = CudaContext.Create(0); using var stream = CudaStream.Create();
        using var kernels = new CudaLfm2Kernels(Path.Combine(AppContext.BaseDirectory, "ptx"));
        var buffers = new List<nint>(); var rng = new Random(1032 + rows);
        float[] x = Enumerable.Range(0, cols).Select(_ => rng.NextSingle() * 2 - 1).ToArray();
        byte[] quant = new byte[rows * (cols / 32) * 18]; rng.NextBytes(quant);
        for (int b = 0; b < rows * (cols / 32); b++) BitConverter.TryWriteBytes(quant.AsSpan(b * 18, 2), BitConverter.HalfToInt16Bits((Half)((b % 3 - 1.3f) * 0.007f)));
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out nint qw, (nuint)quant.Length).ThrowOnError(); buffers.Add(qw);
            float[] dequant = new float[rows * cols];
            fixed (byte* src = quant)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(qw, (nint)src, (nuint)quant.Length).ThrowOnError();
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)src, rows * cols, DotLLM.Core.Configuration.QuantizationType.Q4_0, dequant);
            }
            nint dx = Upload(x, buffers), dy = Upload(new float[rows], buffers);
            kernels.GemvQ4(qw, dx, dy, rows, cols, stream.Handle); stream.Synchronize(); float[] actual = Read(dy, rows);
            for (int row = 0; row < rows; row++)
            {
                float expected = 0; for (int c = 0; c < cols; c++) expected += dequant[row * cols + c] * x[c];
                Assert.True(MathF.Abs(expected - actual[row]) < 2e-5f + 2e-5f * MathF.Abs(expected), $"row {row}: CPU={expected}, CUDA={actual[row]}");
            }
            // Pin the production Q4_0 x Q8_1 path to the CPU packed arithmetic, including Half d/s headers.
            int tokens = 2, blocks = tokens * cols / 32;
            float[] multi = Enumerable.Range(0, tokens * cols).Select(_ => rng.NextSingle() * 2 - 1).ToArray();
            Array.Clear(multi, 0, 32); // a zero activation block exercises the scale=0 branch
            nint mx = Upload(multi, buffers), qx = Upload(new float[blocks * 9], buffers), my = Upload(new float[tokens * rows], buffers);
            kernels.QuantizeQ8(mx, qx, blocks, stream.Handle);
            kernels.GemmQ4Q8(qw, qx, my, rows, cols, tokens, stream.Handle); stream.Synchronize();
            byte[] cpuQ8 = new byte[blocks * 36]; float[] cpuPacked = new float[tokens * rows];
            fixed (float* input = multi)
            fixed (byte* q8 = cpuQ8)
            fixed (byte* w = quant)
            fixed (float* expected = cpuPacked)
            {
                DotLLM.Cpu.Kernels.MatMul.QuantizeF32ToQ8_1(input, q8, tokens * cols);
                DotLLM.Cpu.Kernels.MatMul.GemmLegacyQuant(w, DotLLM.Core.Configuration.QuantizationType.Q4_0, input, expected, rows, cols, tokens);
            }
            Assert.Equal(cpuQ8, System.Runtime.InteropServices.MemoryMarshal.AsBytes(Read(qx, blocks * 9).AsSpan()).ToArray());
            float[] cudaPacked = Read(my, cpuPacked.Length);
            for (int i = 0; i < cpuPacked.Length; i++) Assert.True(MathF.Abs(cpuPacked[i] - cudaPacked[i]) < 2e-5f + 2e-5f * MathF.Abs(cpuPacked[i]));
        }
        finally { foreach (nint p in buffers) CudaDriverApi.cuMemFree_v2(p); }
    }

}
