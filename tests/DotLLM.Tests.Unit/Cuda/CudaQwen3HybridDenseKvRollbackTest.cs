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
/// dense model's own F16 cursor and per-slot F32 staging valid lengths must shrink to the committed
/// position, so the next append takes the incremental #182 conversion.
/// </summary>
/// <remarks>
/// <para>
/// Pre-fix both lengths were grow-only. In the scenario below the cursor then reads 6 instead of 5
/// (<c>WriteF16KvRows</c>'s <c>if (newLength &gt; _f16CacheCurrentLength)</c>: 5 &gt; 6 is false),
/// and the append at position 4 misses the incremental path (<c>positions[0] == prevValid</c> with
/// <c>prevValid</c> still 6), so the full-reconversion counter grows by one per attention slot.
/// Correctness is unaffected either way — the causal mask is positional — which is why this test
/// asserts on the internal lengths rather than on logits. It does ALSO check that the incremental
/// path's logits equal the forced full-reconversion path's bit for bit, which is what would break
/// if the shrink left stale F32 staging rows in place.
/// </para>
/// <para>
/// Uses the tiny F32 <see cref="SyntheticQwen35HybridDenseMtpGguf"/> fixture (layer 0 GDN, layer 1
/// full attention → one KV slot), so no model download is needed.
/// </para>
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
    public void RollbackThenAppend_ShrinksKvLengths_AndTakesIncrementalConversion()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "kv-rollback.gguf"), withMtp: false);

        var incremental = Run(path, ptxDir!, forceFullOnLastStep: false);
        var forcedFull = Run(path, ptxDir!, forceFullOnLastStep: true);

        _out.WriteLine($"cursor before rollback {incremental.CursorBeforeRollback}, after append {incremental.CursorAfterAppend}; " +
                       $"full reconversions on the post-rollback append: {incremental.FullReconvertsOnAppend} " +
                       $"(attention slots: {incremental.AttentionSlots})");

        Assert.Equal(6, incremental.CursorBeforeRollback);
        Assert.True(incremental.CursorAfterAppend == 5,
            $"F16 cursor is {incremental.CursorAfterAppend} after rollback-to-4 + append at 4, expected 5 " +
            "(6 means the cursor did not shrink on rollback — issue #478).");
        Assert.True(incremental.FullReconvertsOnAppend == 0,
            $"The post-rollback append took {incremental.FullReconvertsOnAppend} full-range F16->F32 " +
            "reconversion(s); expected the incremental #182 path (issue #478).");
        Assert.Equal(forcedFull.AttentionSlots, forcedFull.FullReconvertsOnAppend);   // control: the counter works

        Assert.Equal(forcedFull.Logits.Length, incremental.Logits.Length);
        for (int i = 0; i < forcedFull.Logits.Length; i++)
            Assert.Equal(forcedFull.Logits[i], incremental.Logits[i]);   // bit-exact
    }

    private sealed record Result(int CursorBeforeRollback, int CursorAfterAppend, int FullReconvertsOnAppend,
                                 int AttentionSlots, float[] Logits);

    private static Result Run(string path, string ptxDir, bool forceFullOnLastStep)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.AttentionLayerCount >= 1, "fixture must have a full-attention layer");
        using var kv = model.CreateKvCache(config.MaxSequenceLength);

        using (model.Forward([1, 2, 3], [0, 1, 2], deviceId: -1, kv)) { }
        using (model.Forward([4, 5, 6], [3, 4, 5], deviceId: -1, kv)) { }   // a verify-shaped batch
        int before = model.DebugF16CacheCurrentLengthForTest;
        Assert.Equal(6, kv.CurrentLength);

        kv.Rollback(4);   // a partial rejection: positions 4 and 5 are dropped
        model.ForceFullKvReconvertForTest = forceFullOnLastStep;
        int reconverts0 = model.DebugFullKvReconvertCountForTest;
        float[] logits;
        using (ITensor t = model.Forward([7], [4], deviceId: -1, kv))
            logits = Copy(t, config.VocabSize);
        int reconverts = model.DebugFullKvReconvertCountForTest - reconverts0;

        return new Result(before, model.DebugF16CacheCurrentLengthForTest, reconverts,
                          model.AttentionLayerCount, logits);
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
