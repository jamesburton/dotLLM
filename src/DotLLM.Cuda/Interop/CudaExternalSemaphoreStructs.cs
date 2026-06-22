using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

// Layouts mirror the CUDA Driver API headers (cuda.h, CUDA 12/13) field-for-field.
// LayoutKind.Sequential with default Pack so natural 8-byte alignment inserts the
// pad after the leading `int type` exactly as the C compiler does. Do NOT set
// Pack=1 — a wrong size yields CUDA_ERROR_INVALID_VALUE that is indistinguishable
// from "unsupported handle type".

/// <summary>
/// Mirrors <c>CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC</c>. Describes the external
/// semaphore handle to import. For a Win32 handle type
/// (OPAQUE_WIN32 / D3D12_FENCE) populate <see cref="Handle"/> and leave
/// <see cref="Name"/> zero.
/// </summary>
/// <remarks>
/// C layout: <c>int type</c> + 4-byte pad + union{ <c>int fd</c> | struct{ <c>void* handle; const void* name</c> } | <c>void* nvSciSyncObj</c> } (16 bytes) + <c>uint flags</c> + <c>uint reserved[16]</c>.
/// The union's largest member is the 16-byte win32 struct, so we model the union
/// inline as the two pointers (handle, name).
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
internal struct CudaExternalSemaphoreHandleDesc
{
    /// <summary>CUexternalSemaphoreHandleType (e.g. <see cref="CudaDriverApi.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32"/>).</summary>
    internal int Type;

    // 4 bytes of implicit padding here (handle pointers are 8-byte aligned on x64).

    /// <summary>Win32 NT HANDLE referencing the semaphore object. Must be set for Win32 handle types; mutually exclusive with <see cref="Name"/>.</summary>
    internal nint Handle;

    /// <summary>Named-object alternative to <see cref="Handle"/>; left zero for handle-based import.</summary>
    internal nint Name;

    /// <summary>Reserved — must be zero.</summary>
    internal uint Flags;

    // unsigned int reserved[16] — 64 bytes, must be zero. Modelled as a fixed buffer.
    private unsafe fixed uint _reserved[16];
}

/// <summary>
/// Mirrors <c>CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS</c>. For a binary semaphore
/// the entire struct is zeroed; the fence value applies only to timeline /
/// D3D12-fence semaphores.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct CudaExternalSemaphoreWaitParams
{
    /// <summary>Fence value to wait on (timeline / D3D12-fence semaphores only; zero for binary).</summary>
    internal ulong FenceValue;

    // union { void* fence; unsigned long long reserved; } nvSciSync — 8 bytes.
    private ulong _nvSciSync;

    // struct { unsigned long long key; unsigned int timeoutMs; } keyedMutex — 16 bytes (with pad).
    private ulong _keyedMutexKey;
    private uint _keyedMutexTimeoutMs;

    // unsigned int reserved[10] within params — 40 bytes.
    private unsafe fixed uint _paramsReserved[10];

    /// <summary>Reserved — must be zero for non-NvSciSync semaphores.</summary>
    internal uint Flags;

    // unsigned int reserved[16] — 64 bytes.
    private unsafe fixed uint _reserved[16];
}

/// <summary>
/// Mirrors <c>CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS</c>. For a binary semaphore
/// the entire struct is zeroed; the fence value applies only to timeline /
/// D3D12-fence semaphores.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct CudaExternalSemaphoreSignalParams
{
    /// <summary>Fence value to signal (timeline / D3D12-fence semaphores only; zero for binary).</summary>
    internal ulong FenceValue;

    // union { void* fence; unsigned long long reserved; } nvSciSync — 8 bytes.
    private ulong _nvSciSync;

    // struct { unsigned long long key; } keyedMutex — 8 bytes.
    private ulong _keyedMutexKey;

    // unsigned int reserved[12] within params — 48 bytes.
    private unsafe fixed uint _paramsReserved[12];

    /// <summary>Reserved — must be zero for non-NvSciSync semaphores.</summary>
    internal uint Flags;

    // unsigned int reserved[16] — 64 bytes.
    private unsafe fixed uint _reserved[16];
}
