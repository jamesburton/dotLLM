using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Hip;
using DotLLM.Hip.DirectStorage;
using DotLLM.Hip.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
public sealed class HipDirectWeightStoreTests(ITestOutputHelper output)
{
    [Fact]
    public void InteropLayouts_MatchHip64WindowsHeaders()
    {
        Assert.Equal(1472, Marshal.SizeOf<HipExternalMemoryApi.DeviceProperties>());
        Assert.Equal(272, (int)Marshal.OffsetOf<HipExternalMemoryApi.DeviceProperties>(nameof(HipExternalMemoryApi.DeviceProperties.Luid)));
        Assert.Equal(104, Marshal.SizeOf<HipExternalMemoryApi.MemoryHandleDesc>());
        Assert.Equal(8, (int)Marshal.OffsetOf<HipExternalMemoryApi.MemoryHandleDesc>(nameof(HipExternalMemoryApi.MemoryHandleDesc.Handle)));
        Assert.Equal(24, (int)Marshal.OffsetOf<HipExternalMemoryApi.MemoryHandleDesc>(nameof(HipExternalMemoryApi.MemoryHandleDesc.Size)));
        Assert.Equal(32, (int)Marshal.OffsetOf<HipExternalMemoryApi.MemoryHandleDesc>(nameof(HipExternalMemoryApi.MemoryHandleDesc.Flags)));
        Assert.Equal(88, Marshal.SizeOf<HipExternalMemoryApi.BufferDesc>());
        Assert.Equal(8, (int)Marshal.OffsetOf<HipExternalMemoryApi.BufferDesc>(nameof(HipExternalMemoryApi.BufferDesc.Size)));
        Assert.Equal(4, HipExternalMemoryApi.D3D12Heap);
        Assert.Equal(104, Marshal.SizeOf<DStorageNative.Request>());
        Assert.Equal(40, (int)Marshal.OffsetOf<DStorageNative.Request>(nameof(DStorageNative.Request.DestResource)));
        Assert.Equal(48, Marshal.SizeOf<D3dNative.HeapDesc>());
        Assert.Equal(56, Marshal.SizeOf<D3dNative.ResourceDesc>());
    }

    [Theory]
    [InlineData(100, -1, 1)]
    [InlineData(100, 0, 0)]
    [InlineData(100, 0, -1)]
    [InlineData(100, 100, 1)]
    [InlineData(100, 90, 11)]
    [InlineData(100, long.MaxValue, 2)]
    [InlineData(long.MaxValue, long.MaxValue - 1, 2)]
    public void ValidateRange_RejectsInvalidAndOverflowingRanges(long fileLength, long offset, long length)
        => Assert.Throws<ArgumentOutOfRangeException>(() => HipDirectWeightStore.ValidateRange(fileLength, offset, length));

    [Fact]
    public void ValidateRange_AcceptsExactEndAndUnalignedRange()
    {
        HipDirectWeightStore.ValidateRange(260, 3, 257);
        HipDirectWeightStore.ValidateRange(long.MaxValue, long.MaxValue - 2, 2);
    }

    [SkippableTheory]
    [InlineData(257, 65536L)]
    [InlineData(34603027, 34668544L)]
    [Trait("Category", "GPU")]
    public unsafe void Load_UnalignedRangeIsByteIdenticalAndOutlivesStore(int length, long allocatedBytes)
    {
        SkipIfUnavailable();
        using var context = HipContext.Create(0);
        using var store = CreateStore(context);
        const int offset = 3;
        byte[] bytes = new byte[offset + length + 17];
        new Random(1783).NextBytes(bytes);
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-hip-dstorage-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, bytes);
            var timer = Stopwatch.StartNew();
            using HipDirectWeightBuffer buffer = store.Load(path, offset, length);
            timer.Stop();
            Assert.Equal(length, buffer.ByteLength);
            Assert.Equal(allocatedBytes, buffer.AllocatedBytes);
            store.Dispose();

            byte[] actual = new byte[length];
            context.MakeCurrent();
            fixed (byte* destination = actual)
                HipDriverApi.hipMemcpyDtoH((nint)destination, buffer.Pointer, (nuint)length).ThrowOnError();
            Assert.True(bytes.AsSpan(offset, length).SequenceEqual(actual), "DirectStorage-loaded bytes differ from the requested file range.");
            Assert.False(bytes.AsSpan(offset + 1, length).SequenceEqual(actual), "Fixture must distinguish a wrong source offset.");
            output.WriteLine($"HIP DirectStorage: {length} file bytes, {buffer.AllocatedBytes} allocated GPU bytes, {timer.Elapsed.TotalMilliseconds:F3} ms. Application host staging: 0 bytes; DirectStorage's shared staging configuration is retained (32 MiB default).");

            context.Dispose();
            buffer.Dispose();
            buffer.Dispose();
            Assert.Throws<ObjectDisposedException>(() => buffer.Pointer);
        }
        finally { File.Delete(path); }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void Load_InvalidRangeOrMissingFileDoesNotPoisonTheStore()
    {
        SkipIfUnavailable();
        using var context = HipContext.Create(0);
        using var store = CreateStore(context);
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-hip-dstorage-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, new byte[257]);
            using var otherStore = CreateStore(context);
            Assert.Throws<ArgumentOutOfRangeException>(() => store.Load(path, 1, 257));
            Assert.Throws<ArgumentOutOfRangeException>(() => store.Load(path, long.MaxValue, 2));
            Assert.Throws<FileNotFoundException>(() => store.Load(path + ".missing", 0, 1));
            using var buffer = store.Load(path, 0, 257);
            Assert.Equal(257, buffer.ByteLength);
            store.Dispose();
            Assert.Throws<ObjectDisposedException>(() => store.Load(path, 0, 1));
            using var otherBuffer = otherStore.Load(path, 128, 129);
            Assert.Equal(129, otherBuffer.ByteLength);
            Assert.NotEqual(buffer.Pointer, otherBuffer.Pointer);
        }
        finally { File.Delete(path); }
    }

    private static void SkipIfUnavailable()
    {
        Skip.IfNot(OperatingSystem.IsWindows() && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "HIP DirectStorage qualification requires Windows x64.");
        Skip.IfNot(HipDevice.IsAvailable(), "HIP runtime/GPU unavailable.");
    }

    private static HipDirectWeightStore CreateStore(HipContext context)
    {
        try { return new HipDirectWeightStore(context); }
        catch (DllNotFoundException ex)
        {
            Skip.If(true, ex.Message);
            throw;
        }
    }
}
