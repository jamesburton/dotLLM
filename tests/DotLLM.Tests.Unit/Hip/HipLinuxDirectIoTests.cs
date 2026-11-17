using System.Runtime.InteropServices;
using DotLLM.Hip;
using DotLLM.Hip.DirectStorage;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
public sealed class HipLinuxDirectIoTests
{
    [Fact]
    public void InteropLayouts_MatchLinuxX64AndAmdStorageHeaders()
    {
        Assert.Equal(8, Marshal.SizeOf<HipLinuxDirectIo.AmdFileHandle>());
        Assert.Equal(0, (int)Marshal.OffsetOf<HipLinuxDirectIo.AmdFileHandle>(nameof(HipLinuxDirectIo.AmdFileHandle.FileDescriptor)));
        Assert.Equal(256, Marshal.SizeOf<HipLinuxDirectIo.FileStat>());
        Assert.Equal(40, (int)Marshal.OffsetOf<HipLinuxDirectIo.FileStat>(nameof(HipLinuxDirectIo.FileStat.Size)));
        Assert.Equal(152, (int)Marshal.OffsetOf<HipLinuxDirectIo.FileStat>(nameof(HipLinuxDirectIo.FileStat.MemoryAlignment)));
        Assert.Equal(156, (int)Marshal.OffsetOf<HipLinuxDirectIo.FileStat>(nameof(HipLinuxDirectIo.FileStat.OffsetAlignment)));
        Assert.Equal(120, Marshal.SizeOf<HipLinuxDirectIo.FileSystemStat>());
    }

    [Theory]
    [InlineData(260L, 3L, 257L, 512u, 512u, 0L, 4096L, 260L, 3L, 8191L)]
    [InlineData(16384L, 8192L, 8192L, 4096u, 4096u, 8192L, 8192L, 8192L, 0L, 12287L)]
    [InlineData(1000000L, 65537L, 65536L, 65536u, 512u, 65536L, 131072L, 65537L, 1L, 196607L)]
    [InlineData(4097L, 4096L, 1L, 0u, 0u, 4096L, 4096L, 1L, 0L, 8191L)]
    public void Plan_CoversUnalignedRangesWithoutCopyingOrChangingFiles(long fileLength, long offset, long length,
        uint memoryAlignment, uint offsetAlignment, long expectedStart, long expectedRead, long expectedRequired,
        long expectedOffset, long expectedAllocation)
    {
        var plan = HipLinuxDirectIo.Plan(fileLength, offset, length, memoryAlignment, offsetAlignment);
        Assert.Equal(expectedStart, plan.FileOffset);
        Assert.Equal(expectedRead, plan.ReadBytes);
        Assert.Equal(expectedRequired, plan.RequiredBytes);
        Assert.Equal(expectedOffset, plan.TensorOffset);
        Assert.Equal(expectedAllocation, plan.AllocationBytes);
    }

    [Theory]
    [InlineData(3u, 512u)]
    [InlineData(4096u, 511u)]
    [InlineData(67108864u, 4096u)]
    public void Plan_RejectsUnsupportedAlignment(uint memoryAlignment, uint offsetAlignment)
        => Assert.Throws<NotSupportedException>(() => HipLinuxDirectIo.Plan(100, 0, 100, memoryAlignment, offsetAlignment));

    [Fact]
    public void Plan_RejectsRangeAndAlignmentOverflow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => HipLinuxDirectIo.Plan(100, 99, 2, 0, 0));
        Assert.Throws<OverflowException>(() => HipLinuxDirectIo.Plan(long.MaxValue, long.MaxValue - 1, 1, 0, 0));
        Assert.Throws<OverflowException>(() => HipLinuxDirectIo.Plan(long.MaxValue, 0, long.MaxValue, 0, 0));
    }

    [Fact]
    public void ValidateRead_AllowsPaddingOnlyShortReadAndRejectsTruncationOrPartialErrors()
    {
        HipLinuxDirectIo.ValidateRead(0, 0, 260, 4096, 260);
        HipLinuxDirectIo.ValidateRead(0, 0, 4096, 4096, 260);
        Assert.Throws<EndOfStreamException>(() => HipLinuxDirectIo.ValidateRead(0, 0, 259, 4096, 260));
        Assert.Throws<EndOfStreamException>(() => HipLinuxDirectIo.ValidateRead(0, 0, 0, 4096, 260));
        Assert.Throws<EndOfStreamException>(() => HipLinuxDirectIo.ValidateRead(0, 0, 4097, 4096, 260));
        Assert.Throws<IOException>(() => HipLinuxDirectIo.ValidateRead(999, 0, 260, 4096, 260));
        Assert.Throws<IOException>(() => HipLinuxDirectIo.ValidateRead(0, -5, 260, 4096, 260));
    }

    [SkippableFact]
    public void LinuxOpenFile_PreservesUtf8PathAndExactLength()
    {
        Skip.IfNot(OperatingSystem.IsLinux() && RuntimeInformation.ProcessArchitecture == Architecture.X64, "Linux x64 file ABI test.");
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-hip-é-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, new byte[513]);
            using var file = HipLinuxDirectIo.OpenFile(path, out var stat);
            Assert.False(file.IsInvalid);
            Assert.Equal(513UL, stat.Size);
        }
        finally { File.Delete(path); }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
#pragma warning disable IDISP016, IDISP017 // Exercise early and repeated disposal while retaining failure-safe cleanup.
    public unsafe void LinuxDirectRead_ExactTailAndIndependentBufferLifetime()
    {
        Skip.IfNot(OperatingSystem.IsLinux() && RuntimeInformation.ProcessArchitecture == Architecture.X64, "Linux x64 AIS hardware test.");
        Skip.IfNot(HipDevice.IsAvailable(), "Linux HIP GPU unavailable.");
        string? directory = Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_LINUX_DIR");
        Skip.If(string.IsNullOrWhiteSpace(directory), "Set DOTLLM_TEST_HIP_LINUX_DIR to an AIS-capable ext4/XFS test directory.");
        string path = Path.Combine(directory!, $"dotllm-hip-ais-{Guid.NewGuid():N}.bin");
        byte[] data = new byte[32 * 1024 * 1024 + 257];
        new Random(1797).NextBytes(data);
        try
        {
            File.WriteAllBytes(path, data);
            using var context = HipContext.Create(0);
            using var store = new HipDirectWeightStore(context);
            using var buffer = store.Load(path, 3, data.Length - 3);
            Assert.Equal(data.Length - 3, buffer.ByteLength);
            store.Dispose();
            byte[] actual = new byte[data.Length - 3];
            fixed (byte* destination = actual)
                HipDriverApi.hipMemcpyDtoH((nint)destination, buffer.Pointer, (nuint)actual.Length).ThrowOnError();
            Assert.Equal(data[3..], actual);
            Assert.NotEqual(data[2..^1], actual);
            context.Dispose();
            buffer.Dispose();
            buffer.Dispose();
            Assert.Throws<ObjectDisposedException>(() => buffer.Pointer);
        }
        finally { File.Delete(path); }
    }
#pragma warning restore IDISP016, IDISP017
}
