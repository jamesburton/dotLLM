using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1025: a recurrent-prefix hit grows its KV handle instead of re-prefilling. The CUDA hybrid's KV rows live in
/// model-owned storage, which used to be reallocated empty whenever a handle's <c>MaxLength</c> grew. The rows must
/// survive the growth, or the hit attends over zeros.
/// </summary>
/// <remarks>
/// Mutant: resetting the live length (or skipping the row copy) in <c>EnsureF16KvCache</c>'s growth path makes the
/// second forward's logits diverge from the never-grown reference by O(1).
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseKvGrowthTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-kv-grow-{Guid.NewGuid():N}");

    public CudaQwen3HybridDenseKvGrowthTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void GrownHandle_KeepsItsRows_AndMatchesANeverGrownCache()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "m.gguf"), withMtp: false);

        float[] reference = Run(path, ptxDir!, initialLength: 16, grow: false);
        float[] grown = Run(path, ptxDir!, initialLength: 6, grow: true);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(reference[i]) == BitConverter.SingleToInt32Bits(grown[i]),
                $"logits[{i}]: never-grown {reference[i]:R} vs grown {grown[i]:R}");
        Assert.Contains(grown, v => v != 0f);
    }

    private static unsafe float[] Run(string path, string ptxDir, int initialLength, bool grow)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(initialLength);

        int[] first = [1, 4, 7, 2, 9, 3];
        using (ITensor _ = model.Forward(first, [0, 1, 2, 3, 4, 5], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        if (grow) Assert.True(kv.TryGrowTo(16));
        Assert.Equal(16, kv.MaxLength);

        using ITensor logits = model.Forward([5, 8, 6], [6, 7, 8], deviceId: -1, kv, lastTokenLogitsOnly: true);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, config.VocabSize).ToArray();
    }

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
