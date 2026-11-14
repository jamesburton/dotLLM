using System.Runtime.InteropServices;

namespace DotLLM.Hip.DirectStorage;

/// <summary>HIP 6's versioned device-properties ABI and external-memory descriptors on Windows x64.</summary>
internal static unsafe partial class HipExternalMemoryApi
{
    // sizeof/offsetof verified against HIP SDK 6.4.2's hip_runtime_api.h.
    [StructLayout(LayoutKind.Explicit, Size = 1472)]
    internal struct DeviceProperties
    {
        [FieldOffset(272)] public ulong Luid;
    }

    [StructLayout(LayoutKind.Explicit, Size = 104)]
    internal struct MemoryHandleDesc
    {
        [FieldOffset(0)] public int Type;
        [FieldOffset(8)] public nint Handle;
        [FieldOffset(24)] public ulong Size;
        [FieldOffset(32)] public uint Flags;
    }

    [StructLayout(LayoutKind.Explicit, Size = 88)]
    internal struct BufferDesc
    {
        [FieldOffset(0)] public ulong Offset;
        [FieldOffset(8)] public ulong Size;
        [FieldOffset(16)] public uint Flags;
    }

    internal const int D3D12Heap = 4;

    [LibraryImport("amdhip64", EntryPoint = "hipGetDevicePropertiesR0600")]
    internal static partial int GetDeviceProperties(out DeviceProperties properties, int device);

    [LibraryImport("amdhip64", EntryPoint = "hipImportExternalMemory")]
    internal static partial int ImportExternalMemory(out nint memory, in MemoryHandleDesc descriptor);

    [LibraryImport("amdhip64", EntryPoint = "hipExternalMemoryGetMappedBuffer")]
    internal static partial int GetMappedBuffer(out nint devicePtr, nint memory, in BufferDesc descriptor);

    [LibraryImport("amdhip64", EntryPoint = "hipDestroyExternalMemory")]
    internal static partial int DestroyExternalMemory(nint memory);
}
