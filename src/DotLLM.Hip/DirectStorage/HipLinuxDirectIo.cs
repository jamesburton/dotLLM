using System.ComponentModel;
using System.Runtime.InteropServices;
using DotLLM.Hip.Interop;
using Microsoft.Win32.SafeHandles;

namespace DotLLM.Hip.DirectStorage;

/// <summary>Linux x64 AIS reads into VRAM; deliberately has no host-copy compatibility backend.</summary>
internal static unsafe partial class HipLinuxDirectIo
{
    private const int OpenDirect = 0x4000, OpenCloseOnExec = 0x80000;
    private const uint StatxBasicStats = 0x7ff, StatxDioAlign = 0x2000;
    private const long MaxReadBytes = 32 * 1024 * 1024;

    // ABI: ROCm hip/amd_detail/hip_storage.h. Resolve as hipFile's fastpath does.
    [StructLayout(LayoutKind.Explicit, Size = 8)]
    internal struct AmdFileHandle
    {
        [FieldOffset(0)] internal int FileDescriptor;
    }

    // Linux x64 UAPI linux/stat.h and glibc bits/statfs.h; unused fields retain their ABI space.
    [StructLayout(LayoutKind.Explicit, Size = 256)]
    internal struct FileStat
    {
        [FieldOffset(0)] internal uint Mask;
        [FieldOffset(28)] internal ushort Mode;
        [FieldOffset(40)] internal ulong Size;
        [FieldOffset(152)] internal uint MemoryAlignment;
        [FieldOffset(156)] internal uint OffsetAlignment;
    }

    [StructLayout(LayoutKind.Explicit, Size = 120)]
    internal struct FileSystemStat
    {
        [FieldOffset(0)] internal long Type;
    }

    [LibraryImport("libc", EntryPoint = "open", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int Open(string path, int flags);

    [LibraryImport("libc", EntryPoint = "statx", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int Stat(SafeFileHandle file, string path, int flags, uint mask, out FileStat stat);

    [LibraryImport("libc", EntryPoint = "fstatfs", SetLastError = true)]
    private static partial int StatFileSystem(SafeFileHandle file, out FileSystemStat stat);

    internal static nint Resolve(HipContext context)
    {
        context.MakeCurrent();
        HipDriverApi.hipInit(0).ThrowOnError();
        HipDriverApi.hipRuntimeGetVersion(out int version).ThrowOnError();
        try
        {
            int error = HipDriverApi.hipGetProcAddress("hipAmdFileRead", out nint read, version, 0, out int status);
            if (error == 0 && status == 0 && read != 0) return read;
        }
        catch (EntryPointNotFoundException) { }
        throw new PlatformNotSupportedException("The Linux HIP runtime does not expose hipAmdFileRead. Install an AMD Infinity Storage-capable ROCm runtime and amdgpu/KFD driver; host-copy fallback is disabled.");
    }

    internal static SafeFileHandle OpenFile(string path, out FileStat stat)
    {
        int fd = Open(path, OpenDirect | OpenCloseOnExec);
        if (fd < 0) throw IoError("O_DIRECT open", Marshal.GetLastPInvokeError());
        var file = new SafeFileHandle(fd, ownsHandle: true);
        try
        {
            if (Stat(file, "", 0x1000 /* AT_EMPTY_PATH */, StatxBasicStats | StatxDioAlign, out stat) != 0)
                throw IoError("statx", Marshal.GetLastPInvokeError());
            if ((stat.Mode & 0xf000) != 0x8000)
                throw new NotSupportedException("Linux HIP weight loading requires a regular model file.");
            if (StatFileSystem(file, out var fs) != 0)
                throw IoError("fstatfs", Marshal.GetLastPInvokeError());
            if (fs.Type is not (0xef53 or 0x58465342))
                throw new NotSupportedException("Linux HIP direct loading requires an ext4 or XFS model volume with a GPU-accessible storage path. Host-copy fallback is disabled.");
            return file;
        }
        catch { file.Dispose(); throw; }
    }

    internal readonly record struct ReadPlan(long FileOffset, long ReadBytes, long RequiredBytes,
        long TensorOffset, long AllocationBytes, long Alignment);

    internal static ReadPlan Plan(long fileLength, long offset, long byteLength, uint memoryAlignment, uint offsetAlignment)
    {
        HipDirectWeightStore.ValidateRange(fileLength, offset, byteLength);
        if ((memoryAlignment != 0 && (memoryAlignment & (memoryAlignment - 1)) != 0)
            || (offsetAlignment != 0 && (offsetAlignment & (offsetAlignment - 1)) != 0))
            throw new NotSupportedException("Linux direct-I/O alignment must be a power of two.");
        // Older AIS drivers require page alignment even when statx reports a smaller sector size.
        long alignment = Math.Max(4096L, Math.Max(memoryAlignment, offsetAlignment));
        if ((alignment & (alignment - 1)) != 0 || alignment > MaxReadBytes)
            throw new NotSupportedException($"Unsupported Linux direct-I/O alignment {alignment}.");
        long start = offset & -alignment;
        long displacement = offset - start;
        long required = checked(displacement + byteLength);
        long readBytes = checked(required + alignment - 1) & -alignment;
        _ = checked(start + readBytes);
        long allocation = checked(readBytes + alignment - 1);
        return new(start, readBytes, required, displacement, allocation, alignment);
    }

    internal static HipDirectWeightBuffer Load(HipContext context, nint readFunction, string path, long offset, long byteLength)
    {
        using var file = OpenFile(path, out var stat);
        if (stat.Size > long.MaxValue) throw new NotSupportedException("Model file exceeds the supported size.");
        var plan = Plan((long)stat.Size, offset, byteLength,
            (stat.Mask & StatxDioAlign) != 0 ? stat.MemoryAlignment : 0,
            (stat.Mask & StatxDioAlign) != 0 ? stat.OffsetAlignment : 0);
        HipDriverApi.hipMemGetInfo(out nuint free, out _).ThrowOnError();
        if ((ulong)plan.AllocationBytes > free)
            throw new OutOfMemoryException($"HIP direct loading needs {plan.AllocationBytes} device bytes; only {free} are free.");
        HipDriverApi.hipMalloc(out nint allocation, checked((nuint)plan.AllocationBytes)).ThrowOnError();
        try
        {
            nint aligned = checked((nint)(((long)allocation + plan.Alignment - 1) & -plan.Alignment));
            var read = (delegate* unmanaged<AmdFileHandle, nint, ulong, long, ulong*, int*, int>)readFunction;
            var handle = new AmdFileHandle { FileDescriptor = checked((int)file.DangerousGetHandle()) };
            for (long copied = 0; copied < plan.ReadBytes;)
            {
                long count = Math.Min(MaxReadBytes, plan.ReadBytes - copied);
                ulong received = 0;
                int status = 0;
                int error = read(handle, aligned + checked((nint)copied), (ulong)count,
                    checked(plan.FileOffset + copied), &received, &status);
                ValidateRead(error, status, received, count, Math.Min(count, plan.RequiredBytes - copied));
                copied += (long)received;
                if (received < (ulong)count) break; // A successful EOF tail must still contain every requested tensor byte.
            }
            return new HipDirectWeightBuffer(context.DeviceId, aligned + checked((nint)plan.TensorOffset),
                allocation, byteLength, plan.AllocationBytes);
        }
        catch { HipDriverApi.hipFree(allocation); throw; }
    }

    internal static void ValidateRead(int error, int status, ulong received, long requested, long required)
    {
        if (error != 0 || status != 0)
            throw new IOException($"Linux HIP direct read failed (HIP {error}, storage status {status}). Verify AIS/P2PDMA support in amdgpu/KFD and the storage topology. Host-copy fallback is disabled.");
        if (received > (ulong)requested || received < (ulong)required)
            throw new EndOfStreamException($"Linux HIP direct read returned {received} bytes; at least {required} of {requested} requested bytes were required.");
    }

    private static IOException IoError(string operation, int error)
        => new($"Linux HIP {operation} failed: {new Win32Exception(error).Message} (errno {error}).", new Win32Exception(error));
}
