using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>DFlash on the CUDA <c>qwen35</c> trunk (issue #1175): GPU verify, CPU drafter, token-identical to greedy.</summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaDFlashTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-dflash-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;

    public CudaDFlashTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    private static string? PtxDir()
    {
        foreach (var dir in new[] { Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx") })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static string RequireCuda()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptx = PtxDir();
        Skip.If(ptx is null, "PTX files not found");
        return ptx!;
    }

    private sealed unsafe class DeviceMemory : DFlashDrafterTests.ITargetMemory
    {
        public void Up(float* src, nint dst, int count)
            => DotLLM.Cuda.Interop.CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)src, (nuint)(count * sizeof(float))).ThrowOnError();

        public void Down(nint src, float* dst, int count)
            => DotLLM.Cuda.Interop.CudaDriverApi.cuMemcpyDtoH_v2((nint)dst, src, (nuint)(count * sizeof(float))).ThrowOnError();
    }

    /// <summary>The GPU drafter (device weights, ring, ring attention, convolutions) against the stateless reference.</summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Draft_OnDevice_MatchesStatelessReference(bool dflash2)
    {
        string ptx = RequireCuda();
        string trunk = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_dir, "trunk.gguf"), withMtp: false);
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptx);
        // F32 fixture weights run through cuBLAS; the tolerance covers float RoPE angles and reduction order.
        DFlashDrafterTests.AssertMatchesReference(_dir, dflash2, model.CreateDFlashBackend,
            t => new DFlashDrafterTests.StagedTarget(t, new DeviceMemory()), 1e-3f, _out);
    }

    /// <summary>
    /// The drafter's multi-column K-quant GEMV against the CPU dequantization of the same blocks: random blocks (finite
    /// scales), an output count that is not a multiple of the 4 rows per CTA, and every column count 1..8.
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q6_K)]
    public unsafe void GemvMulti_MatchesDequantizedReference(QuantizationType qt)
    {
        string ptx = RequireCuda();
        const int N = 37, K = 512;
        int blockBytes = qt == QuantizationType.Q4_K ? 144 : 210, rowBytes = K / 256 * blockBytes;
        var rng = new Random(11);
        var w = new byte[N * rowBytes];
        rng.NextBytes(w);
        for (int b = 0; b < w.Length / blockBytes; b++)
        {
            // Finite, modest scales: Q4_K d/dmin lead the block, Q6_K d trails it.
            int dOff = qt == QuantizationType.Q4_K ? 0 : 208;
            BitConverter.TryWriteBytes(w.AsSpan(b * blockBytes + dOff, 2), (Half)(0.01f + 0.02f * rng.NextSingle()));
            if (qt == QuantizationType.Q4_K)
                BitConverter.TryWriteBytes(w.AsSpan(b * blockBytes + 2, 2), (Half)(0.01f * rng.NextSingle()));
        }
        var wf = new float[N * K];
        fixed (byte* wp = w) DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)wp, (long)N * K, qt, wf);

        string trunk = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_dir, "trunk.gguf"), withMtp: false);
        using var gguf = GgufFile.Open(trunk);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), deviceId: 0, ptx);
        using var kernels = CudaDFlashKernels.TryLoad(ptx)!;
        CudaDriverApi.cuMemAlloc_v2(out nint dw, (nuint)w.Length).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dx, (nuint)(8 * K * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dy, (nuint)(8 * N * sizeof(float))).ThrowOnError();
        try
        {
            var x = new float[8 * K];
            for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2 - 1;
            fixed (byte* wp = w) CudaDriverApi.cuMemcpyHtoD_v2(dw, (nint)wp, (nuint)w.Length).ThrowOnError();
            fixed (float* xp = x) CudaDriverApi.cuMemcpyHtoD_v2(dx, (nint)xp, (nuint)(x.Length * sizeof(float))).ThrowOnError();
            for (int cols = 1; cols <= 8; cols++)
            {
                kernels.LaunchGemvMulti(qt, dw, dx, dy, N, K, cols, 0);
                var y = new float[cols * N];
                fixed (float* yp = y) CudaDriverApi.cuMemcpyDtoH_v2((nint)yp, dy, (nuint)(y.Length * sizeof(float))).ThrowOnError();
                for (int col = 0; col < cols; col++)
                    for (int r = 0; r < N; r++)
                    {
                        double want = 0;
                        for (int i = 0; i < K; i++) want += (double)wf[r * K + i] * x[col * K + i];
                        Assert.True(Math.Abs(y[col * N + r] - want) <= 1e-3 + 1e-4 * Math.Abs(want), $"{qt} cols {cols} [{col},{r}]: {y[col * N + r]} vs {want}");
                    }
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dw);
            CudaDriverApi.cuMemFree_v2(dx);
            CudaDriverApi.cuMemFree_v2(dy);
        }
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecode_WithDFlash_IsTokenIdenticalToGreedy(bool snapshots)
    {
        string? ptx = RequireCuda();
        DFlashDrafterTests.AssertTokenIdentical(_dir, (gguf, config) =>
        {
            var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptx!);
            return (m, m.CreateKvCache(config.MaxSequenceLength));
        }, snapshots, _out);
    }
}
