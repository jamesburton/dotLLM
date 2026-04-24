using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Minimal P/Invoke declarations against AMD's HIP runtime API.
/// libamdhip64.so (Linux) / amdhip64.dll (Windows) — installed with the ROCm stack
/// on Linux and with the ROCm / AMD software stack on Windows.
/// All functions return hipError_t (int): 0 = hipSuccess, non-zero = error.
/// Mirrors <c>DotLLM.Cuda.Interop.CudaDriverApi</c> 1:1 — HIP is source-level
/// compatible with the CUDA Driver API, so the symbol mapping is mechanical.
/// </summary>
internal static partial class HipDriverApi
{
    // .NET resolves "amdhip64" to libamdhip64.so (Linux) / amdhip64.dll (Windows)
    // via HipLibraryResolver registered at startup.
    private const string LibName = "amdhip64";

    // ── Initialization ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipInit(uint flags);

    // ── Device ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    internal static partial int hipGetDeviceCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceTotalMem(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int hipMemGetInfo(out nuint free, out nuint total);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ─────────────────────────────────────────────────────
    //
    // Modern HIP exposes a primary-context model via the runtime API (hipSetDevice
    // / hipGetDevice). hipCtxCreate also exists (driver-style) but is deprecated on
    // Windows ROCm builds. We prefer hipSetDevice — it binds the current thread to
    // a primary context the runtime manages internally, and is what hipModuleLoadData
    // expects when it initializes code objects.

    [LibraryImport(LibName)]
    internal static partial int hipSetDevice(int device);

    [LibraryImport(LibName)]
    internal static partial int hipGetDevice(out int device);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceSynchronize();

    // Legacy driver-style context management. Left in for completeness; not used
    // by HipContext (which prefers hipSetDevice).
    [LibraryImport(LibName)]
    internal static partial int hipCtxCreate(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int hipCtxDestroy(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxGetCurrent(out nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxGetDevice(out int device);

    // ── Module (code-object loading) ────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipModuleLoadData(out nint module, nint image);

    [LibraryImport(LibName)]
    internal static partial int hipModuleLoadDataEx(
        out nint module, nint image, uint numOptions,
        nint options, nint optionValues);

    [LibraryImport(LibName)]
    internal static partial int hipModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int hipModuleUnload(nint module);

    // ── Kernel launch ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipModuleLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipMalloc(out nint devicePtr, nuint bytesize);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    internal static partial int hipFree(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyHtoD(
        nint dstDevice, nint srcHost, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoH(
        nint dstHost, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoD(
        nint dstDevice, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyHtoDAsync(
        nint dstDevice, nint srcHost, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoHAsync(
        nint dstHost, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoDAsync(
        nint dstDevice, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemsetD8(nint dstDevice, byte value, nuint n);

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipStreamCreate(out nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipStreamDestroy(nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipStreamSynchronize(nint stream);

    // ── Error ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial nint hipGetErrorName(int error);

    [LibraryImport(LibName)]
    internal static partial nint hipGetErrorString(int error);

    // ── Device attribute constants ──────────────────────────────────
    // HIP attribute enum values; match hip_runtime_api.h (hipDeviceAttribute_t).

    /// <summary>Maximum number of threads per block.</summary>
    internal const int HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;

    /// <summary>Maximum shared memory per block in bytes.</summary>
    internal const int HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;

    /// <summary>Warp size in threads (typically 32 on CDNA, 32 or 64 on RDNA depending on wave mode).</summary>
    internal const int HIP_DEVICE_ATTRIBUTE_WARP_SIZE = 10;

    /// <summary>Number of compute units (analogous to CUDA SM count).</summary>
    internal const int HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;

    /// <summary>Compute capability major (CUDA-like, not meaningful on AMD — reports 0 or PCI ID-ish value).</summary>
    internal const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 23;

    /// <summary>Compute capability minor.</summary>
    internal const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 24;
}
