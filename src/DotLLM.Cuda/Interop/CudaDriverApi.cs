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

    /// <summary>
    /// Cooperative-kernel launch — required for CUDA Cooperative Groups <c>grid.sync()</c> (a
    /// grid-wide barrier). Unlike <see cref="cuLaunchKernel"/>, the driver guarantees ALL blocks
    /// in the grid are resident simultaneously (query <see cref="cuOccupancyMaxActiveBlocksPerMultiprocessor"/>
    /// first — exceeding the co-residency ceiling is a hard launch error, not a silent fallback).
    /// No <c>extra</c> parameter (unlike <c>cuLaunchKernel</c>) — cooperative launch does not
    /// support the legacy extra-options array. Used only by
    /// <see cref="DotLLM.Cuda.CudaKernels.LaunchGdnScanStepF32CoopSplit4"/> (issue #180, opt-in,
    /// default-off — see <c>gated_delta_net_scan.cu</c>'s header for why this isn't the default).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuLaunchCooperativeKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams);

    /// <summary>
    /// Queries the maximum number of resident blocks per SM for <paramref name="func"/> at the
    /// given block size / dynamic shared memory — the value cooperative-launch validation uses to
    /// decide whether a candidate grid size can ever be launched via <see cref="cuLaunchCooperativeKernel"/>.
    /// Multiply by <c>CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT</c> for the max co-resident grid.
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuOccupancyMaxActiveBlocksPerMultiprocessor(
        out int numBlocks, nint func, int blockSize, nuint dynamicSMemSize);

    /// <summary>
    /// Diagnostic-only (issue #213): queries a compiled kernel's static attributes (register
    /// count, local-memory spill, max threads/block, etc.) without needing Nsight Compute.
    /// <c>attr</c> is a <c>CUfunction_attribute</c> enum value — see
    /// <see cref="CU_FUNC_ATTRIBUTE_NUM_REGS"/> / <see cref="CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"/>.
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuFuncGetAttribute(out int value, int attr, nint hfunc);

    internal const int CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;
    internal const int CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;
    internal const int CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2;
    internal const int CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
    internal const int CU_FUNC_ATTRIBUTE_NUM_REGS = 4;
    internal const int CU_FUNC_ATTRIBUTE_PTX_VERSION = 5;
    internal const int CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6;

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

    // ── External semaphores (cross-API sync: Vulkan → CUDA) ─────────────
    //
    // The Vulkan iGPU exports a VkSemaphore as a Win32 HANDLE (or D3D12 fence)
    // which CUDA imports via cuImportExternalSemaphore. cuWaitExternalSemaphoresAsync
    // then gates a CUDA stream on the Vulkan signal without a host fence-wait,
    // letting the CUDA H2D + compute overlap the next decode step's Vulkan
    // recording. See HybridVulkanCudaTransformerModel (M3 async pipelining).

    /// <summary>CUexternalSemaphoreHandleType: OPAQUE_WIN32 — a Vulkan VkSemaphore exported as an NT HANDLE. Same-stack handle, may not interop cross-vendor.</summary>
    internal const int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2;

    /// <summary>CUexternalSemaphoreHandleType: D3D12_FENCE — the cross-vendor-portable Win32 handle type; preferred fallback when OPAQUE_WIN32 import fails between Intel Vulkan and NVIDIA CUDA.</summary>
    internal const int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4;

    /// <summary>
    /// Imports an external semaphore (e.g. an exported Vulkan VkSemaphore) into
    /// CUDA. The current context must be set on the calling thread. For
    /// OPAQUE_WIN32 the driver duplicates the supplied HANDLE — the caller
    /// must CloseHandle its own copy afterwards.
    /// </summary>
    /// <param name="extSemOut">Receives the imported CUexternalSemaphore handle.</param>
    /// <param name="semHandleDesc">Pointer to a populated <see cref="CudaExternalSemaphoreHandleDesc"/>.</param>
    /// <returns>CUresult — 0 on success.</returns>
    [LibraryImport(LibName)]
    internal static partial int cuImportExternalSemaphore(out nint extSemOut, in CudaExternalSemaphoreHandleDesc semHandleDesc);

    /// <summary>
    /// Enqueues a wait on one or more external semaphores into a stream. The
    /// stream's subsequent work does not begin until each semaphore is signalled
    /// (by Vulkan, in the M3 handoff). For a binary semaphore the params must be
    /// zeroed; for a timeline/D3D12-fence semaphore the wait value applies.
    /// </summary>
    /// <param name="extSemArray">Pointer to an array of CUexternalSemaphore handles.</param>
    /// <param name="paramsArray">Pointer to a matching array of <see cref="CudaExternalSemaphoreWaitParams"/>.</param>
    /// <param name="numExtSems">Number of semaphores in the arrays.</param>
    /// <param name="stream">Target CUDA stream.</param>
    /// <returns>CUresult — 0 on success.</returns>
    [LibraryImport(LibName)]
    internal static partial int cuWaitExternalSemaphoresAsync(
        in nint extSemArray, in CudaExternalSemaphoreWaitParams paramsArray, uint numExtSems, nint stream);

    /// <summary>
    /// Enqueues a signal on one or more external semaphores into a stream.
    /// Used when CUDA must signal back to Vulkan; not required for the
    /// Vulkan-signals-CUDA-waits M3 first cut, but kept for the timeline /
    /// ping-pong double-buffering path.
    /// </summary>
    /// <param name="extSemArray">Pointer to an array of CUexternalSemaphore handles.</param>
    /// <param name="paramsArray">Pointer to a matching array of <see cref="CudaExternalSemaphoreSignalParams"/>.</param>
    /// <param name="numExtSems">Number of semaphores in the arrays.</param>
    /// <param name="stream">Target CUDA stream.</param>
    /// <returns>CUresult — 0 on success.</returns>
    [LibraryImport(LibName)]
    internal static partial int cuSignalExternalSemaphoresAsync(
        in nint extSemArray, in CudaExternalSemaphoreSignalParams paramsArray, uint numExtSems, nint stream);

    /// <summary>Destroys an imported external semaphore handle. Does not affect the underlying Vulkan semaphore.</summary>
    /// <param name="extSem">The CUexternalSemaphore handle to destroy.</param>
    /// <returns>CUresult — 0 on success.</returns>
    [LibraryImport(LibName)]
    internal static partial int cuDestroyExternalSemaphore(nint extSem);

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
    /// <summary>Device supports launching cooperative kernels via <see cref="cuLaunchCooperativeKernel"/>.</summary>
    internal const int CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95;
}
