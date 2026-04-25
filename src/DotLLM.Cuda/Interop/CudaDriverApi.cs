using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal P/Invoke declarations against NVIDIA's CUDA Driver API.
/// libcuda.so (Linux) / nvcuda.dll (Windows) — installed with GPU driver.
/// All functions return CUresult (int): 0 = CUDA_SUCCESS, non-zero = error.
/// </summary>
internal static partial class CudaDriverApi
{
    // .NET resolves "cuda" to libcuda.so (Linux) / nvcuda.dll (Windows)
    // via CudaLibraryResolver registered at startup.
    private const string LibName = "cuda";

    // ── Initialization ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuInit(uint flags);

    // ── Device ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceTotalMem_v2(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int cuMemGetInfo_v2(out nuint free, out nuint total);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuCtxCreate_v2(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int cuCtxDestroy_v2(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetCurrent(out nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetDevice(out int device);

    // ── Module (PTX loading) ────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadData(out nint module, nint ptxImage);

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadDataEx(
        out nint module, nint ptxImage, uint numOptions,
        nint options, nint optionValues);

    [LibraryImport(LibName)]
    internal static partial int cuModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int cuModuleUnload(nint module);

    /// <summary>
    /// Set a CUfunction attribute. Currently used to opt kernels into the
    /// device's full dynamic-shared-memory budget (default cap is 48 KB on
    /// most archs; sm_75+ supports raising it to MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    /// — typically 100 KB+ on Ampere/Ada/Hopper).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuFuncSetAttribute(
        nint function, int attribute, int value);

    /// <summary>CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES — opt in to >48 KB dynamic shmem.</summary>
    internal const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

    // ── Kernel launch ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuMemAlloc_v2(out nint devicePtr, nuint bytesize);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    internal static partial int cuMemFree_v2(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoD_v2(
        nint dstDevice, nint srcHost, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoH_v2(
        nint dstHost, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoD_v2(
        nint dstDevice, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoDAsync_v2(
        nint dstDevice, nint srcHost, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoHAsync_v2(
        nint dstHost, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoDAsync_v2(
        nint dstDevice, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemsetD8_v2(nint dstDevice, byte value, nuint n);

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuStreamDestroy_v2(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    // ── Graphs (capture + replay) ──────────────────────────────────────
    //
    // Capture path: cuStreamBeginCapture_v2 → run normal stream operations →
    // cuStreamEndCapture → cuGraphInstantiateWithFlags → cache the cuGraphExec.
    // Replay path: cuGraphLaunch (single packet submission, ~1 µs vs ~22 µs/launch
    // on WDDM). Suitable for the inner decode loop where launch sequence is
    // topology-invariant; per-step variability handled via device-resident state.

    /// <summary>Thread-local capture mode — only operations on this thread's stream are captured.
    /// Safer than relaxed/global mode when other threads might touch the stream.</summary>
    internal const uint CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 2;

    [LibraryImport(LibName)]
    internal static partial int cuStreamBeginCapture_v2(nint stream, uint mode);

    [LibraryImport(LibName)]
    internal static partial int cuStreamEndCapture(nint stream, out nint graph);

    [LibraryImport(LibName)]
    internal static partial int cuGraphInstantiateWithFlags(out nint graphExec, nint graph, ulong flags);

    [LibraryImport(LibName)]
    internal static partial int cuGraphLaunch(nint graphExec, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuGraphDestroy(nint graph);

    [LibraryImport(LibName)]
    internal static partial int cuGraphExecDestroy(nint graphExec);

    // ── Events (used for GPU-side profiling) ────────────────────────

    /// <summary>Default flags. Use for profiling: blocking-sync isn't needed when we cuEventSynchronize host-side.</summary>
    internal const uint CU_EVENT_DEFAULT = 0;

    [LibraryImport(LibName)]
    internal static partial int cuEventCreate(out nint evt, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuEventDestroy_v2(nint evt);

    [LibraryImport(LibName)]
    internal static partial int cuEventRecord(nint evt, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuEventSynchronize(nint evt);

    [LibraryImport(LibName)]
    internal static partial int cuEventElapsedTime(out float milliseconds, nint start, nint end);

    // ── Error ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorName(int error, out nint str);

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorString(int error, out nint str);

    // ── Device attribute constants ──────────────────────────────────

    internal const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    internal const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    internal const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
    internal const int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
    /// <summary>
    /// MAX_SHARED_MEMORY_PER_BLOCK_OPTIN — the dynamic-shmem cap a kernel can
    /// raise itself to via cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, ...).
    /// Typically 100+ KB on sm_86 (RTX 3060), 164 KB on sm_80/89, etc.
    /// </summary>
    internal const int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
}
