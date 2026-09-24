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
    public unsafe void BothArms_PutByteIdenticalContentOnDevice(long bytes)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        using var ctx = CudaContext.Create(0);

        int n = checked((int)bytes);
        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        byte* direct = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        byte* staged = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        nint devDirect = 0, devStaged = 0;
        try
        {
            var rng = new Random(509);
            for (long i = 0; i < bytes; i++) src[i] = (byte)rng.Next(256);
            new Span<byte>(direct, n).Clear();
            new Span<byte>(staged, n).Clear();

            // ── Arm A: the DEFAULT path (opt-in off) — one synchronous pageable copy. ──
            CudaDriverApi.cuMemAlloc_v2(out devDirect, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(devDirect, 0xCD, (nuint)bytes).ThrowOnError();
            using (var offScope = OpenScope(enabled: false))
            {
                Assert.False(offScope.ShouldStage(bytes), "the opt-in is off, so nothing may stage");
                CudaDriverApi.cuMemcpyHtoD_v2(devDirect, (nint)src, (nuint)bytes).ThrowOnError();
                Assert.Equal(0, offScope.StagedChunks);
            }
            CudaDriverApi.cuMemcpyDtoH_v2((nint)direct, devDirect, (nuint)bytes).ThrowOnError();

            // ── Arm B: the OPT-IN pinned, chunked, double-buffered path. ──
            long stagedChunks;
            long stagedBytes;
            CudaDriverApi.cuMemAlloc_v2(out devStaged, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(devStaged, 0xCD, (nuint)bytes).ThrowOnError();
            using (var onScope = OpenScope(enabled: true))
            {
                Assert.True(onScope.ShouldStage(bytes), "the override should force even a sub-chunk tensor to stage");
                Assert.Equal(0, onScope.Upload(devStaged, (nint)src, bytes));  // -1 would mean the pinned pair failed
                stagedChunks = onScope.StagedChunks;
                stagedBytes = onScope.StagedBytes;
            }
            CudaDriverApi.cuMemcpyDtoH_v2((nint)staged, devStaged, (nuint)bytes).ThrowOnError();

            Assert.Equal(bytes, stagedBytes);
            Assert.Equal((bytes + TestChunk - 1) / TestChunk, stagedChunks);

            var source = new ReadOnlySpan<byte>(src, n);
            Assert.True(source.SequenceEqual(new ReadOnlySpan<byte>(direct, n)),
                $"default arm differs from the host source for {bytes} bytes " +
                $"(first mismatch at {FirstMismatch(src, direct, bytes)})");
            Assert.True(source.SequenceEqual(new ReadOnlySpan<byte>(staged, n)),
                $"staged arm differs from the host source for {bytes} bytes " +
                $"(first mismatch at {FirstMismatch(src, staged, bytes)})");
            Assert.True(
                new ReadOnlySpan<byte>(direct, n).SequenceEqual(new ReadOnlySpan<byte>(staged, n)),
                $"the two arms disagree for {bytes} bytes " +
                $"(first mismatch at {FirstMismatch(direct, staged, bytes)})");
        }
        finally
        {
            if (devDirect != 0) CudaDriverApi.cuMemFree_v2(devDirect);
            if (devStaged != 0) CudaDriverApi.cuMemFree_v2(devStaged);
            NativeMemory.AlignedFree(src);
            NativeMemory.AlignedFree(direct);
            NativeMemory.AlignedFree(staged);
        }
    }

    /// <summary>
    /// Opens a scope with the test chunk size and a zero threshold (so even a sub-chunk tensor
    /// exercises the chunking loop) and the given opt-in state. The static overrides are restored
    /// by <see cref="OverrideScope.Dispose"/> so a failing case cannot leak 64 KiB chunks into a
    /// sibling test.
    /// </summary>
    private static OverrideScope OpenScope(bool enabled) => new(TestChunk, minStagedBytes: 0, enabled);

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
    /// The path is <b>opt-in</b>: with no <c>DOTLLM_CUDA_PINNED_UPLOAD</c> set, nothing stages, no
    /// matter how large the tensor. Read against the real environment (no override), so this fails
    /// if the switch is ever silently flipped to default-on without a measurement.
    /// </summary>
    [SkippableFact]
    public void OptIn_IsOffByDefault()
    {
        Skip.If(Environment.GetEnvironmentVariable(CudaWeightUploadStaging.EnabledEnvVar) is not null,
            $"{CudaWeightUploadStaging.EnabledEnvVar} is set in this environment");
        bool? saved = CudaWeightUploadStaging.EnabledOverride;
        try
        {
            CudaWeightUploadStaging.EnabledOverride = null;   // fall through to the env-var read
            Assert.False(CudaWeightUploadStaging.Enabled);

            using var scope = CudaWeightUploadStaging.BeginScope();
            Assert.False(scope.ShouldStage(1024L * 1024 * 1024));
            Assert.Equal(0, scope.StagedBytes);
            Assert.Equal(0, scope.StagedChunks);

            // The direct counter is what a benchmark reads to confirm the legacy path ran.
            scope.RecordDirect(4096);
            Assert.Equal(4096, scope.DirectBytes);
        }
        finally
        {
            CudaWeightUploadStaging.EnabledOverride = saved;
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

    /// <summary>
    /// A <see cref="CudaWeightUploadStaging"/> scope that also sets and restores the three static
    /// test overrides around it.
    /// </summary>
    private sealed class OverrideScope : IDisposable
    {
        private readonly CudaWeightUploadStaging _inner;
        private readonly long? _savedChunk;
        private readonly long? _savedMin;
        private readonly bool? _savedEnabled;

        public OverrideScope(long chunkBytes, long minStagedBytes, bool enabled)
        {
            _savedChunk = CudaWeightUploadStaging.ChunkBytesOverride;
            _savedMin = CudaWeightUploadStaging.MinStagedBytesOverride;
            _savedEnabled = CudaWeightUploadStaging.EnabledOverride;
            CudaWeightUploadStaging.ChunkBytesOverride = chunkBytes;
            CudaWeightUploadStaging.MinStagedBytesOverride = minStagedBytes;
            CudaWeightUploadStaging.EnabledOverride = enabled;
            _inner = CudaWeightUploadStaging.BeginScope();
        }

        public long StagedBytes => _inner.StagedBytes;

        public long StagedChunks => _inner.StagedChunks;

        public bool ShouldStage(long bytes) => _inner.ShouldStage(bytes);

        public int Upload(nint devPtr, nint hostPtr, long bytes) => _inner.Upload(devPtr, hostPtr, bytes);

        public void Dispose()
        {
            _inner.Dispose();
            CudaWeightUploadStaging.ChunkBytesOverride = _savedChunk;
            CudaWeightUploadStaging.MinStagedBytesOverride = _savedMin;
            CudaWeightUploadStaging.EnabledOverride = _savedEnabled;
        }
    }
}
