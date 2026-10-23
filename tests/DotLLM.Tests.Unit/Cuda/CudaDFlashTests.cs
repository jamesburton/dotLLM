using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
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

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecode_WithDFlash_IsTokenIdenticalToGreedy(bool snapshots)
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptx = PtxDir();
        Skip.If(ptx is null, "PTX files not found");
        DFlashDrafterTests.AssertTokenIdentical(_dir, (gguf, config) =>
        {
            var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptx!);
            return (m, m.CreateKvCache(config.MaxSequenceLength));
        }, snapshots, _out);
    }
}
