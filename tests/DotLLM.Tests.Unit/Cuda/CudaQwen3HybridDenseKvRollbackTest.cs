using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #478: after a speculative-decoding rollback of the length-only KV handle, the CUDA hybrid
/// dense model's own F16 cursor must shrink to the committed position so the next forward converts
/// only the live range.
/// </summary>
/// <remarks>
/// Uses the tiny F32 <see cref="SyntheticQwen35HybridDenseMtpGguf"/> fixture (layer 0 GDN, layer 1
/// full attention), so no model download is needed.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseKvRollbackTest : IDisposable
{
    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseKvRollbackTest(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-kv-rollback-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void RollbackThenAppend_ShrinksKvCursor()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "kv-rollback.gguf"), withMtp: false);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
        Assert.True(model.AttentionLayerCount >= 1, "fixture must have a full-attention layer");
        using var kv = model.CreateKvCache(config.MaxSequenceLength);

        using (model.Forward([1, 2, 3], [0, 1, 2], deviceId: -1, kv)) { }
        using (model.Forward([4, 5, 6], [3, 4, 5], deviceId: -1, kv)) { }   // a verify-shaped batch
        int cursorBefore = model.DebugF16CacheCurrentLengthForTest;
        Assert.Equal(6, kv.CurrentLength);

        kv.Rollback(4);   // a partial rejection: positions 4 and 5 are dropped
        float[] logits;
        using (ITensor t = model.Forward([7], [4], deviceId: -1, kv))
            logits = Copy(t, config.VocabSize);
        int cursorAfter = model.DebugF16CacheCurrentLengthForTest;

        _out.WriteLine($"cursor before rollback {cursorBefore}, after append {cursorAfter}");

        Assert.Equal(6, cursorBefore);
        Assert.True(cursorAfter == 5,
            $"F16 cursor is {cursorAfter} after rollback-to-4 + append at 4, expected 5 " +
            "(6 means the cursor did not shrink on rollback -- issue #478).");
    }

    private static unsafe float[] Copy(ITensor t, int count)
        => new ReadOnlySpan<float>((void*)t.DataPointer, count).ToArray();

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
