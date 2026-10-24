using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #966 on CUDA: the MTP head KV is no longer capped at 4096 positions and allocates lazily in
/// 1024-position steps up to the trunk KV-cache's length. CPU twin (runnable anywhere):
/// <c>MtpLongContextTests</c>.
/// </summary>
/// <remarks>
/// Verified output equals plain greedy however poor the drafts are, so the greedy comparison alone
/// cannot see a corrupted head KV; <see cref="DraftLogits_AfterLazyGrowth_AreBitIdenticalToAFullySizedState"/>
/// observes the head directly against an eagerly sized state (same kernels, same data, same launch
/// shapes, hence exact equality).
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMtpLongContextTests : IDisposable
{
    private const int ContextLength = 5200;
    private const int NewTokens = 8;
    private const int K = 3;

    private readonly string _scratch;

    public CudaQwen3HybridDenseMtpLongContextTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mtp-longctx-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Fixture(string name)
        => SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, name), withMtp: true, contextLength: ContextLength);

    /// <summary>Prompt lengths straddling the first growth step (1024) and the old cap (4096), as in the CPU twin.</summary>
    [SkippableTheory]
    [InlineData(1023)]
    [InlineData(1024)]
    [InlineData(1025)]
    [InlineData(4095)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void MtpGreedy_AcrossGrowthAndTheOldCap_MatchesPlainGreedy(int promptLength)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(Fixture($"longctx-{promptLength}.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = MtpLongContextHarness.Prompt(promptLength, config.VocabSize);

        int[] expected;
        using (var plain = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
        using (var kv = plain.CreateKvCache(config.MaxSequenceLength))
            expected = MtpLongContextHarness.PlainGreedy(plain, kv, prompt, NewTokens, config.VocabSize);

        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
        using var kvCache = model.CreateKvCache(config.MaxSequenceLength);
        using var mtp = (CudaMtpState)model.CreateMtpState(kvCache.MaxLength)!;
        Assert.Equal(1024, mtp.Capacity);

        int[] actual = MtpLongContextHarness.MtpGreedy(model, kvCache, mtp, prompt, NewTokens, K, config.VocabSize);

        Assert.Equal(expected, actual);
        Assert.True(mtp.Capacity >= promptLength + 1, $"capacity {mtp.Capacity} does not cover the sequence");
        Assert.True(mtp.Capacity <= mtp.MaxSteps);
        Assert.True(mtp.CurrentLength >= promptLength);
    }

    [SkippableTheory]
    [InlineData(1023)]   // drafts cross the first growth step inside ForwardMtp
    [InlineData(2500)]   // several growths inside the batched prefill absorb
    [InlineData(4095)]   // drafts cross the old 4096 cap inside ForwardMtp
    public void DraftLogits_AfterLazyGrowth_AreBitIdenticalToAFullySizedState(int promptLength)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(Fixture($"longctx-draft-{promptLength}.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = MtpLongContextHarness.Prompt(promptLength, config.VocabSize);

        float[][] eager;
        using (var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
        using (var kv = model.CreateKvCache(config.MaxSequenceLength))
        using (var mtp = (CudaMtpState)model.CreateMtpState(kv.MaxLength)!)
        {
            mtp.EnsureCapacity(kv.MaxLength, stream: 0);   // the pre-#966 eager allocation; nothing is in flight yet
            Assert.Equal(kv.MaxLength, mtp.Capacity);
            eager = MtpLongContextHarness.DraftLogits(model, kv, mtp, prompt, steps: 5, config.VocabSize);
            Assert.Equal(kv.MaxLength, mtp.Capacity);
        }

        float[][] lazy;
        long lazyBytes;
        using (var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
        using (var kv = model.CreateKvCache(config.MaxSequenceLength))
        using (var mtp = (CudaMtpState)model.CreateMtpState(kv.MaxLength)!)
        {
            lazy = MtpLongContextHarness.DraftLogits(model, kv, mtp, prompt, steps: 5, config.VocabSize);
            Assert.True(mtp.Capacity < kv.MaxLength, "the lazy state must not have been sized up front");
            lazyBytes = mtp.AllocatedBytes;
        }
        Assert.True(lazyBytes > 0);

        for (int s = 0; s < eager.Length; s++)
            Assert.Equal(eager[s], lazy[s]);
    }

    /// <summary>Above the ceiling the state refuses with a clear exception before any device write.</summary>
    [SkippableFact]
    public void ExceedingTheCeiling_ThrowsInvalidOperation_AndNeverGrowsPastIt()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(Fixture("longctx-ceiling.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
        using var kvCache = model.CreateKvCache(config.MaxSequenceLength);

        const int ceiling = 1030;
        using var mtp = (CudaMtpState)model.CreateMtpState(ceiling)!;
        int len = ceiling + 1;
        int[] prompt = MtpLongContextHarness.Prompt(len, config.VocabSize);

        var ex = Assert.Throws<InvalidOperationException>(() =>
        {
            using ITensor _ = model.Forward(prompt, Enumerable.Range(0, len).ToArray(), deviceId: -1, kvCache, adapter: null, mtp);
        });
        Assert.Contains("exhausted", ex.Message);
        Assert.True(mtp.Capacity <= ceiling);
        Assert.True(mtp.CurrentLength <= ceiling);   // a tiled prefill may have committed earlier chunks

        // Direct single-step path: position == ceiling is one past the last slot.
        using var small = (CudaMtpState)model.CreateMtpState(4)!;
        int[] shortPrompt = MtpLongContextHarness.Prompt(4, config.VocabSize);
        using var smallKv = model.CreateKvCache(config.MaxSequenceLength);
        using (ITensor _ = model.Forward(shortPrompt, [0, 1, 2, 3], deviceId: -1, smallKv, adapter: null, small)) { }
        Assert.Throws<InvalidOperationException>(() => model.ForwardMtp(small, tokenId: 1, position: 4));
        Assert.Equal(4, small.Capacity);
    }

    /// <summary>The parameterless overload no longer caps at 4096; memory follows the positions reached.</summary>
    [SkippableFact]
    public void ParameterlessCreateMtpState_NoLongerCapsAt4096_AndAllocatesOnlyTheFirstStep()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(Fixture("longctx-default.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
        using var kvCache = model.CreateKvCache(config.MaxSequenceLength);
        using var mtp = (CudaMtpState)model.CreateMtpState()!;

        Assert.Equal(Math.Max(CudaQwen3HybridDenseTransformerModel.MtpDefaultMaxSequenceLength, ContextLength), mtp.MaxSteps);
        Assert.Equal(1024, mtp.Capacity);
        long before = mtp.AllocatedBytes;

        int len = 4100;
        int[] prompt = MtpLongContextHarness.Prompt(len, config.VocabSize);
        using (ITensor _ = model.Forward(prompt, Enumerable.Range(0, len).ToArray(), deviceId: -1, kvCache, adapter: null, mtp)) { }
        Assert.Equal(len, mtp.CurrentLength);
        Assert.Equal(5120, mtp.Capacity);
        // Issue #1095: grown in place, memory is whole mapped granules, and this tiny head's 5120 rows fit the first ones; a
        // copying growth allocates the new rows outright.
        if (mtp.GrowsInPlace)
            Assert.Equal(before, mtp.AllocatedBytes);
        else
            Assert.True(mtp.AllocatedBytes > before);
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
