using System.Runtime.InteropServices;
using DotLLM.Core.Models;
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
/// Mutant: skipping the row copy in <c>EnsureF16KvCache</c>'s growth path moves the second forward's logits away from the
/// never-grown reference (measured: logits[0] -0.1567 vs -0.1656). Resetting the live length alone is an equivalent
/// mutant, because the next write's <c>LengthAfterWrite</c> re-covers the copied rows.
/// Issue #1080 adds the in-place growth (CUDA virtual memory), the default where the device supports it. Each test about rows
/// surviving a growth runs on both paths.
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

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void GrownHandle_KeepsItsRows_AndMatchesANeverGrownCache(bool inPlace)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "m.gguf"), withMtp: false);

        float[] reference = Run(path, ptxDir!, initialLength: 16, grow: false, inPlace);
        float[] grown = Run(path, ptxDir!, initialLength: 6, grow: true, inPlace);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(reference[i]) == BitConverter.SingleToInt32Bits(grown[i]),
                $"logits[{i}]: never-grown {reference[i]:R} vs grown {grown[i]:R}");
        Assert.Contains(grown, v => v != 0f);
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public unsafe void FailedGrowthAllocation_LeavesTheOldStorageIntact_AndARetryKeepsTheRows(bool inPlace)
    {
        // PR #1026 review (SF1): the new storage arrays used to be swapped in before the allocation loop. An OOM partway
        // through leaked the old KV and left half-zero arrays, so the retry "kept" rows copied out of never-filled
        // buffers: a silently corrupted prefix.
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "f.gguf"), withMtp: false);
        float[] reference = Run(path, ptxDir!, initialLength: 16, grow: false, inPlace);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Load(gguf, config, ptxDir!, inPlace);
        using var kv = model.CreateKvCache(6);
        using (ITensor _ = model.Forward([1, 4, 7, 2, 9, 3], [0, 1, 2, 3, 4, 5], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        Assert.True(kv.TryGrowTo(16));
        object? state = model.CheckpointRecurrentState();

        CudaQwen3HybridDenseTransformerModel.KvAllocFailureAfter = 1;   // the 2nd KV buffer allocation fails
        try
        {
            Assert.ThrowsAny<Exception>(() => model.Forward([5, 8, 6], [6, 7, 8], deviceId: -1, kv, lastTokenLogitsOnly: true));
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.KvAllocFailureAfter = null;
        }

        // The engine's view of a failed request: roll the KV and the recurrent state back, then run it again.
        kv.Rollback(6);
        model.RestoreRecurrentState(state);
        (state as IDisposable)?.Dispose();
        using ITensor logits = model.Forward([5, 8, 6], [6, 7, 8], deviceId: -1, kv, lastTokenLogitsOnly: true);
        float[] retried = new ReadOnlySpan<float>((void*)logits.DataPointer, config.VocabSize).ToArray();
        for (int i = 0; i < reference.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(reference[i]) == BitConverter.SingleToInt32Bits(retried[i]),
                $"logits[{i}]: never-grown {reference[i]:R} vs retried after a failed growth {retried[i]:R}");
    }

    [SkippableFact]
    public void Growth_IsRefused_WhenOldPlusNewStorageWouldNotFitInFreeVram()
    {
        // PR #1026 review (SF2): a copying growth holds old and new storage at once, where the pre-#1025 miss freed first.
        // When that would not fit, TryGrowTo says no, and the engine takes the miss path (free, then allocate).
        // Issue #1080: this is the copying fallback. The in-place growth (tests below) allocates only the added pages.
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "v.gguf"), withMtp: false);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Load(gguf, config, ptxDir!, inPlace: false);
        using var kv = model.CreateKvCache(6);
        using (ITensor _ = model.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = 0;
        try
        {
            Assert.False(kv.TryGrowTo(16));
            Assert.Equal(6, kv.MaxLength);
            Assert.True(kv.TryGrowTo(6));   // no growth needed: always fine
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        }

        Assert.True(kv.TryGrowTo(16));      // the real card has room for a few KiB
        Assert.Equal(16, kv.MaxLength);
    }

    [SkippableTheory]
    [InlineData(262_144)]   // the window covers the growth: the address range is reserved once and pages are added behind it
    [InlineData(16)]        // past the reservation: the same pages move to a larger range, still without a copy
    public void InPlaceGrowth_KeepsTheRows_WhenFreeVramCannotHoldACopy(int contextLength)
    {
        // Issue #1080: near the fitted window a copying growth needs old + new KV at once, which the admission refuses, so every
        // agent turn re-prefilled its whole prompt. In place, a growth needs only the added pages. The synthetic model keeps
        // 32 bytes of K (and of V) per token, so 3 x 65,536 tokens spans three 2 MiB pages per side.
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, $"g{contextLength}.gguf"), withMtp: false,
            contextLength: contextLength);
        float[] reference = Run(path, ptxDir!, initialLength: Grown, grow: false, inPlace: true, grownLength: Grown);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Load(gguf, config, ptxDir!, inPlace: true);
        Assert.True(model.KvGrowsInPlace, "the device must support CUDA virtual memory");
        using var kv = model.CreateKvCache(6);
        using (ITensor _ = model.Forward([1, 4, 7, 2, 9, 3], [0, 1, 2, 3, 4, 5], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        nint before = model.DebugKeyStoragePointerForTest(0);

        long growth = model.KvStorageGrowthBytes(Grown);
        Assert.True(growth > 0 && growth < model.ComputeKvBytes(Grown), $"growth {growth} must be only the added pages");
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = CudaQwen3HybridDenseTransformerModel.VramReserveBytes + growth;
        try
        {
            Assert.True(kv.TryGrowTo(Grown));
            float[] grown = Logits(model, kv, config);   // the growth runs here, admitted against the same small free figure
            AssertBitIdentical(reference, grown);
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        }

        nint after = model.DebugKeyStoragePointerForTest(0);
        if (contextLength >= Grown) Assert.Equal(before, after);
        else Assert.NotEqual(before, after);
        Assert.Equal(0, model.KvStorageGrowthBytes(Grown));
    }

    [SkippableFact]
    public void CopyingGrowth_IsRefused_UnderTheFreeVramAnInPlaceGrowthNeeds()
    {
        // The other side of the test above: the fallback (no virtual memory) still needs the whole new storage beside the old.
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "c.gguf"), withMtp: false, contextLength: 262_144);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Load(gguf, config, ptxDir!, inPlace: true);
        using var kv = model.CreateKvCache(6);
        using (ITensor _ = model.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        long inPlaceGrowth = model.KvStorageGrowthBytes(Grown);

        using var copying = Load(gguf, config, ptxDir!, inPlace: false);
        using var copyingKv = copying.CreateKvCache(6);
        using (ITensor _ = copying.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, copyingKv, lastTokenLogitsOnly: true)) { }
        Assert.Equal(copying.ComputeKvBytes(Grown), copying.KvStorageGrowthBytes(Grown));
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = CudaQwen3HybridDenseTransformerModel.VramReserveBytes + inPlaceGrowth;
        try
        {
            Assert.False(copyingKv.TryGrowTo(Grown));
            Assert.True(kv.TryGrowTo(Grown));
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        }
    }

    /// <summary>Tokens the in-place tests grow to: three 2 MiB pages of the synthetic model's 32-byte K (or V) rows.</summary>
    private const int Grown = 3 * 65_536;

    private static CudaQwen3HybridDenseTransformerModel Load(GgufFile gguf, ModelConfig config, string ptxDir, bool inPlace)
    {
        var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        if (!inPlace) model.UseCopyingKvGrowthForTest();
        return model;
    }

    private static float[] Run(string path, string ptxDir, int initialLength, bool grow, bool inPlace, int grownLength = 16)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Load(gguf, config, ptxDir, inPlace);
        using var kv = model.CreateKvCache(initialLength);

        int[] first = [1, 4, 7, 2, 9, 3];
        using (ITensor _ = model.Forward(first, [0, 1, 2, 3, 4, 5], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        if (grow) Assert.True(kv.TryGrowTo(grownLength));
        Assert.Equal(grownLength, kv.MaxLength);
        return Logits(model, kv, config);
    }

    private static unsafe float[] Logits(CudaQwen3HybridDenseTransformerModel model, CudaHybridKvCacheHandle kv, ModelConfig config)
    {
        using ITensor logits = model.Forward([5, 8, 6], [6, 7, 8], deviceId: -1, kv, lastTokenLogitsOnly: true);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, config.VocabSize).ToArray();
    }

    private static void AssertBitIdentical(float[] reference, float[] actual)
    {
        for (int i = 0; i < reference.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(reference[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"logits[{i}]: never-grown {reference[i]:R} vs grown {actual[i]:R}");
        Assert.Contains(actual, v => v != 0f);
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
