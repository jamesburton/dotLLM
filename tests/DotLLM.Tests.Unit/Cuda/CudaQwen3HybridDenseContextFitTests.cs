using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1068: <see cref="CudaQwen3HybridDenseTransformerModel.MeasureServingContextFit"/> divides the free VRAM left after
/// the reserve and the fixed buffers by the per-token cost, and <see cref="CudaQwen3HybridDenseTransformerModel.LimitContextLength"/>
/// makes that the window every position is checked against. Driven with a fake free-VRAM figure.
/// </summary>
/// <remarks>
/// Mutants: dropping the MTP term from <c>ContextBytesPerToken</c> fails the MTP comparison; ignoring the reserve in the
/// fit (passing 0) fails the exact-fit assertion; making <c>LimitContextLength</c> a no-op fails the position check.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseContextFitTests : IDisposable
{
    private const int NativeContext = 8192;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-ctxfit-{Guid.NewGuid():N}");

    public CudaQwen3HybridDenseContextFitTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Fit_IsTheFreeBudgetOverThePerTokenCost_AndTheLimitBoundsPositions()
    {
        string? ptxDir = SkipUnlessCuda();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "fit.gguf"), withMtp: true, contextLength: NativeContext);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);

        long perToken = model.ContextBytesPerToken(mtp: true);
        Assert.True(perToken > model.ContextBytesPerToken(mtp: false), "the MTP head's KV must be counted");
        Assert.True(perToken >= model.ComputeKvBytes(1));

        // Fixed bytes: one more recurrent checkpoint, which the measurement reports back.
        long reserve = CudaQwen3HybridDenseTransformerModel.VramReserveBytes;
        var probe = model.MeasureServingContextFit(mtp: true, verifyRows: 4);
        Assert.Null(probe.SizingError);
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = reserve + probe.FixedBytes + perToken * 3000;
        var fit = model.MeasureServingContextFit(mtp: true, verifyRows: 4);
        Assert.Equal(2048, fit.MaxContext);   // 3000 tokens of budget, rounded down to the 1024-token granule
        Assert.Equal(0, model.KvAllocatedBytes);   // the sizing pass releases what it grew

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        model.LimitContextLength(fit.MaxContext);
        Assert.Equal(2048, model.Config.MaxSequenceLength);
        Assert.Equal(NativeContext, model.NativeMaxSequenceLength);

        using var kv = model.CreateKvCache(2049);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => model.Forward([1], [2048], deviceId: -1, kv, lastTokenLogitsOnly: true));
        using (ITensor _ = model.Forward([1, 2, 3], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }

        Assert.Throws<ArgumentOutOfRangeException>(() => model.LimitContextLength(NativeContext + 1));
    }

    private static string? SkipUnlessCuda()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        Skip.If(true, "PTX files not found");
        return null;
    }
}
