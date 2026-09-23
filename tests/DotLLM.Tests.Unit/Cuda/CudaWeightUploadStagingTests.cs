using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #509 — the pinned, chunked, double-buffered weight-upload path must put <b>exactly</b> the
/// same bytes on the device as the single synchronous <c>cuMemcpyHtoD_v2</c> it replaces.
/// </summary>
/// <remarks>
/// <para>
/// The oracle is the <i>host source buffer</i>, not a second device copy — comparing two device
/// buffers would pass if both were wrong in the same way. Every case memsets the destination to a
/// 0xCD sentinel first, so a short write (the classic chunking off-by-one) shows up as sentinel
/// bytes in the tail rather than as whatever VRAM happened to contain.
/// </para>
/// <para>
/// Sizes are chosen around the chunk boundary, which is forced down to 64 KiB so the loop runs many
/// iterations cheaply: below one chunk, exactly one chunk, one byte over (two chunks, the second
/// tiny), one byte under two chunks, a multi-chunk non-multiple, and an exact multi-chunk multiple.
/// <see cref="CudaWeightUploadStaging.StagedChunks"/> is asserted against <c>ceil(bytes/chunk)</c>
/// so the loop's trip count is pinned too, not just the final bytes.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaWeightUploadStagingTests
{
    private const long TestChunk = 64L * 1024;

    public static TheoryData<long> Sizes() => new()
    {
        TestChunk - 1,        // smaller than one chunk
        TestChunk,            // exactly one chunk
        TestChunk + 1,        // spills into a second, one-byte chunk
        (2 * TestChunk) - 1,  // one byte under an exact two-chunk multiple
        (3 * TestChunk) + 17, // several chunks, non-multiple remainder
        4 * TestChunk,        // several chunks, exact multiple
    };

    [SkippableTheory]
    [MemberData(nameof(Sizes))]
    public unsafe void StagedUpload_PutsByteIdenticalContentOnDevice(long bytes)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        using var ctx = CudaContext.Create(0);

        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        byte* back = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        nint dev = 0;
        try
        {
            var rng = new Random(509);
            for (long i = 0; i < bytes; i++) src[i] = (byte)rng.Next(256);
            new Span<byte>(back, checked((int)bytes)).Clear();

            CudaDriverApi.cuMemAlloc_v2(out dev, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(dev, 0xCD, (nuint)bytes).ThrowOnError();

            long stagedChunks;
            long stagedBytes;
            long? savedChunk = CudaWeightUploadStaging.ChunkBytesOverride;
            long? savedMin = CudaWeightUploadStaging.MinStagedBytesOverride;
            bool? savedEnabled = CudaWeightUploadStaging.EnabledOverride;
            try
            {
                CudaWeightUploadStaging.ChunkBytesOverride = TestChunk;
                CudaWeightUploadStaging.MinStagedBytesOverride = 0;   // force even a sub-chunk tensor through the loop
                CudaWeightUploadStaging.EnabledOverride = true;

                using var scope = CudaWeightUploadStaging.BeginScope();
                Assert.True(scope.ShouldStage(bytes), "the override should force even a sub-chunk tensor to stage");
                Assert.Equal(0, scope.Upload(dev, (nint)src, bytes));   // -1 would mean the pinned pair failed
                stagedChunks = scope.StagedChunks;
                stagedBytes = scope.StagedBytes;
            }
            finally
            {
                CudaWeightUploadStaging.ChunkBytesOverride = savedChunk;
                CudaWeightUploadStaging.MinStagedBytesOverride = savedMin;
                CudaWeightUploadStaging.EnabledOverride = savedEnabled;
            }

            CudaDriverApi.cuMemcpyDtoH_v2((nint)back, dev, (nuint)bytes).ThrowOnError();

            Assert.Equal(bytes, stagedBytes);
            Assert.Equal((bytes + TestChunk - 1) / TestChunk, stagedChunks);
            Assert.True(
                new ReadOnlySpan<byte>(src, checked((int)bytes))
                    .SequenceEqual(new ReadOnlySpan<byte>(back, checked((int)bytes))),
                $"device contents differ from the host source for {bytes} bytes " +
                $"(first mismatch at {FirstMismatch(src, back, bytes)})");
        }
        finally
        {
            if (dev != 0) CudaDriverApi.cuMemFree_v2(dev);
            NativeMemory.AlignedFree(src);
            NativeMemory.AlignedFree(back);
        }
    }

    /// <summary>
    /// The default threshold is exactly one chunk: at or below it there is no second chunk to
    /// overlap with, so staging would only add a host memcpy. Pins the dispatch, not the transfer.
    /// </summary>
    [Fact]
    public void DefaultThreshold_StagesOnlyAboveOneChunk()
    {
        long? chunk = CudaWeightUploadStaging.ChunkBytesOverride;
        long? min = CudaWeightUploadStaging.MinStagedBytesOverride;
        bool? enabled = CudaWeightUploadStaging.EnabledOverride;
        try
        {
            CudaWeightUploadStaging.ChunkBytesOverride = TestChunk;
            CudaWeightUploadStaging.MinStagedBytesOverride = null;   // default = chunk size
            CudaWeightUploadStaging.EnabledOverride = true;

            using var scope = CudaWeightUploadStaging.BeginScope();
            Assert.False(scope.ShouldStage(1));
            Assert.False(scope.ShouldStage(TestChunk - 1));
            Assert.False(scope.ShouldStage(TestChunk));
            Assert.True(scope.ShouldStage(TestChunk + 1));
        }
        finally
        {
            CudaWeightUploadStaging.ChunkBytesOverride = chunk;
            CudaWeightUploadStaging.MinStagedBytesOverride = min;
            CudaWeightUploadStaging.EnabledOverride = enabled;
        }
    }

    /// <summary>
    /// The <c>DOTLLM_CUDA_PINNED_UPLOAD=0</c> escape hatch must genuinely disable the path, and the
    /// counters must show it — that is what makes the orchestrator's A/B provably an A/B rather
    /// than two runs of the same code.
    /// </summary>
    [Fact]
    public void EscapeHatch_DisablesStaging_AndTheCountersSaySo()
    {
        bool? enabled = CudaWeightUploadStaging.EnabledOverride;
        long? chunk = CudaWeightUploadStaging.ChunkBytesOverride;
        long? min = CudaWeightUploadStaging.MinStagedBytesOverride;
        try
        {
            CudaWeightUploadStaging.ChunkBytesOverride = TestChunk;
            CudaWeightUploadStaging.MinStagedBytesOverride = 0;
            CudaWeightUploadStaging.EnabledOverride = false;

            using var scope = CudaWeightUploadStaging.BeginScope();
            Assert.False(scope.ShouldStage(100L * 1024 * 1024));
            Assert.Equal(0, scope.StagedBytes);
            Assert.Equal(0, scope.StagedChunks);

            // The direct counter is what the benchmark reads to confirm the legacy path ran.
            scope.RecordDirect(4096);
            Assert.Equal(4096, scope.DirectBytes);
        }
        finally
        {
            CudaWeightUploadStaging.EnabledOverride = enabled;
            CudaWeightUploadStaging.ChunkBytesOverride = chunk;
            CudaWeightUploadStaging.MinStagedBytesOverride = min;
        }
    }

    /// <summary>Scopes nest and restore, so a load inside a load cannot orphan the pinned pair.</summary>
    [Fact]
    public void Scopes_NestAndRestore()
    {
        Assert.Null(CudaWeightUploadStaging.Current);
        using (var outer = CudaWeightUploadStaging.BeginScope())
        {
            Assert.Same(outer, CudaWeightUploadStaging.Current);
            using (var inner = CudaWeightUploadStaging.BeginScope())
            {
                Assert.Same(inner, CudaWeightUploadStaging.Current);
            }
            Assert.Same(outer, CudaWeightUploadStaging.Current);
        }
        Assert.Null(CudaWeightUploadStaging.Current);
    }

    private static unsafe string FirstMismatch(byte* a, byte* b, long n)
    {
        for (long i = 0; i < n; i++)
            if (a[i] != b[i]) return $"offset {i}: expected 0x{a[i]:X2}, got 0x{b[i]:X2}";
        return "none";
    }
}
