using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

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

    /// <summary>PCI bus id ("0000:01:00.0") of a device - the key that maps a CUDA ordinal to its NVML handle (#1023).</summary>
    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetPCIBusId(
        [MarshalAs(UnmanagedType.LPArray)] byte[] pciBusId, int len, int device);

    /// <summary>The driver's own <c>cuMemGetInfo_v2</c>, unclamped. Planners call <see cref="cuMemGetInfo_v2"/>.</summary>
    [LibraryImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
    internal static partial int cuMemGetInfoRaw(out nuint free, out nuint total);

    /// <summary>
    /// <c>cuMemGetInfo_v2</c> for the current context's device, clamped to <c>DOTLLM_VRAM_BUDGET_MB</c> when set (#1295), so
    /// every planner that reads free memory sees the simulated card.
    /// </summary>
    /// <param name="free">Free bytes (simulated when the budget is on).</param>
    /// <param name="total">Total bytes (at most the budget when it is on).</param>
    /// <returns>The CUDA result code.</returns>
    internal static int cuMemGetInfo_v2(out nuint free, out nuint total)
    {
        int rc = cuMemGetInfoRaw(out free, out total);
        if (rc == 0 && DeviceMemoryBudget.SimulatedBudgetBytes() is not null && cuCtxGetDevice(out int device) == 0)
        {
            free = (nuint)DeviceMemoryBudget.ClampSimulatedFree(device, (long)free);
            total = (nuint)DeviceMemoryBudget.ClampSimulatedCapacity((long)total);
        }
        return rc;
    }

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

    /// <summary>
    /// CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT — percentage of the unified L1/shared
    /// block to give shared memory (-1 = let the driver choose). Issue #492: a kernel whose static
    /// shared memory is just over a third of the SM's shared budget only gets two resident blocks
    /// at the *maximum* carveout, and the driver's default heuristic may pick a smaller one.
    /// </summary>
    internal const int CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9;

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

    [LibraryImport(LibName, EntryPoint = "cuMemAlloc_v2")]
    private static partial int cuMemAllocNative(out nint devicePtr, nuint bytesize);

    [LibraryImport(LibName, EntryPoint = "cuMemFree_v2")]
    [SuppressGCTransition]
    private static partial int cuMemFreeNative(nint devicePtr);

    /// <summary>Allocates device memory; recorded by <see cref="CudaMemTrace"/> when <c>DOTLLM_CUDA_MEM_TRACE=1</c> (#1211).</summary>
    /// <param name="devicePtr">The allocated device pointer.</param>
    /// <param name="bytesize">Bytes to allocate.</param>
    /// <returns>The CUresult.</returns>
    internal static int cuMemAlloc_v2(out nint devicePtr, nuint bytesize)
    {
        int r = cuMemAllocNative(out devicePtr, bytesize);
        if (r == 0 && CudaMemTrace.Enabled)
            CudaMemTrace.OnAlloc(devicePtr, (long)bytesize);
        if (r == 0 && TestPoisonAllocations)
            cuMemsetD8_v2(devicePtr, 0xFF, bytesize).ThrowOnError();   // 0xFFFF / 0xFFFFFFFF: NaN in F16 and F32
        return r;
    }

    /// <summary>
    /// Test hook (#1290): fill every new <see cref="cuMemAlloc_v2"/> allocation with NaN bytes, so a kernel that reads
    /// scratch it never wrote (padding rows past the token count, stale capacity) poisons its output instead of passing.
    /// </summary>
    internal static bool TestPoisonAllocations;

    /// <summary>Frees device memory allocated by <see cref="cuMemAlloc_v2"/>.</summary>
    /// <param name="devicePtr">The device pointer.</param>
    /// <returns>The CUresult.</returns>
    internal static int cuMemFree_v2(nint devicePtr)
    {
        if (CudaMemTrace.Enabled)
            CudaMemTrace.OnFree(devicePtr);
        return cuMemFreeNative(devicePtr);
    }

    // ── Virtual memory management (issue #1080) ─────────────────────
    // Reserve a device address range once, then back it with physical allocations as it grows. Growth maps more pages
    // behind the same pointer, so nothing is copied and old and new storage never coexist.

    /// <summary><c>CUmemLocation</c>: where an allocation lives or who may access it.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemLocation
    {
        /// <summary><c>CUmemLocationType</c>; <see cref="CU_MEM_LOCATION_TYPE_DEVICE"/> here.</summary>
        public int Type;

        /// <summary>Device ordinal.</summary>
        public int Id;
    }

    /// <summary><c>CUmemAllocationProp</c> (32 bytes on x64; the trailing <c>allocFlags</c> struct is 8 bytes).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemAllocationProp
    {
        /// <summary><c>CUmemAllocationType</c>; <see cref="CU_MEM_ALLOCATION_TYPE_PINNED"/>.</summary>
        public int Type;

        /// <summary><c>CUmemAllocationHandleType</c>; 0 = not exportable.</summary>
        public int RequestedHandleTypes;

        /// <summary>The device that backs the allocation.</summary>
        public CudaMemLocation Location;

        /// <summary>Windows security metadata for exportable handles; zero.</summary>
        public nint Win32HandleMetaData;

        /// <summary><c>allocFlags</c> (compression type, RDMA, usage, reserved); zero.</summary>
        public ulong AllocFlags;
    }

    /// <summary><c>CUmemAccessDesc</c>: one location's access to a mapped range.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemAccessDesc
    {
        /// <summary>Who gets access.</summary>
        public CudaMemLocation Location;

        /// <summary><c>CUmemAccess_flags</c>; <see cref="CU_MEM_ACCESS_FLAGS_PROT_READWRITE"/>.</summary>
        public int Flags;
    }

    internal const int CU_MEM_ALLOCATION_TYPE_PINNED = 1;
    internal const int CU_MEM_LOCATION_TYPE_DEVICE = 1;
    internal const int CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3;
    internal const int CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0;

    /// <summary><c>CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED</c>: the cuMem* VMM calls below work on this device.</summary>
    internal const int CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102;

    [LibraryImport(LibName)]
    internal static partial int cuMemGetAllocationGranularity(out nuint granularity, in CudaMemAllocationProp prop, int option);

    [LibraryImport(LibName)]
    internal static partial int cuMemAddressReserve(out nint ptr, nuint size, nuint alignment, nint addr, ulong flags);

    [LibraryImport(LibName)]
    internal static partial int cuMemAddressFree(nint ptr, nuint size);

    [LibraryImport(LibName, EntryPoint = "cuMemCreate")]
    private static partial int cuMemCreateNative(out ulong handle, nuint size, in CudaMemAllocationProp prop, ulong flags);

    [LibraryImport(LibName, EntryPoint = "cuMemRelease")]
    private static partial int cuMemReleaseNative(ulong handle);

    /// <summary>Creates a physical allocation; recorded by <see cref="CudaMemTrace"/> when tracing (#1211).</summary>
    /// <param name="handle">The allocation handle.</param>
    /// <param name="size">Bytes.</param>
    /// <param name="prop">Allocation properties.</param>
    /// <param name="flags">Reserved, 0.</param>
    /// <returns>The CUresult.</returns>
    internal static int cuMemCreate(out ulong handle, nuint size, in CudaMemAllocationProp prop, ulong flags)
    {
        int r = cuMemCreateNative(out handle, size, prop, flags);
        if (r == 0 && CudaMemTrace.Enabled)
            CudaMemTrace.OnAlloc((nint)handle, (long)size);
        return r;
    }

    /// <summary>Releases a physical allocation made by <see cref="cuMemCreate"/>.</summary>
    /// <param name="handle">The allocation handle.</param>
    /// <returns>The CUresult.</returns>
    internal static int cuMemRelease(ulong handle)
    {
        if (CudaMemTrace.Enabled)
            CudaMemTrace.OnFree((nint)handle);
        return cuMemReleaseNative(handle);
    }

    [LibraryImport(LibName)]
    internal static partial int cuMemMap(nint ptr, nuint size, nuint offset, ulong handle, ulong flags);

    [LibraryImport(LibName)]
    internal static partial int cuMemUnmap(nint ptr, nuint size);

    [LibraryImport(LibName)]
    internal static partial int cuMemSetAccess(nint ptr, nuint size, in CudaMemAccessDesc desc, nuint count);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoD_v2(
        nint dstDevice, nint srcHost, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoH_v2(
        nint dstHost, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoD_v2(
        nint dstDevice, nint srcDevice, nuint byteCount);

    /// <summary>CU_MEMORYTYPE_DEVICE — the only memory type <see cref="cuMemcpy2DAsync_v2"/> is used with here.</summary>
    internal const uint CU_MEMORYTYPE_DEVICE = 2;

    /// <summary>
    /// <c>CUDA_MEMCPY2D</c> (issue #492). Field order and the implicit padding after each
    /// <c>CUmemorytype</c> match the driver header exactly under x64 sequential layout.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemcpy2D
    {
        public nuint SrcXInBytes;
        public nuint SrcY;
        public uint SrcMemoryType;
        public nint SrcHost;
        public nint SrcDevice;
        public nint SrcArray;
        public nuint SrcPitch;
        public nuint DstXInBytes;
        public nuint DstY;
        public uint DstMemoryType;
        public nint DstHost;
        public nint DstDevice;
        public nint DstArray;
        public nuint DstPitch;
        public nuint WidthInBytes;
        public nuint Height;
    }

    /// <summary>
    /// Strided device-to-device copy on a stream — one launch for a whole row-interleave instead of
    /// one per row (issue #492: the MTP batched absorb's <c>[e_i, h_i]</c> concat).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuMemcpy2DAsync_v2(ref CudaMemcpy2D pCopy, nint hStream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoDAsync_v2(
        nint dstDevice, nint srcHost, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoHAsync_v2(
        nint dstHost, nint srcDevice, nuint byteCount, nint stream);

    /// <summary>
    /// Queues a host function on <paramref name="stream"/>: it runs on a driver thread once the work before it completes, and
    /// the work after it waits for it. Captured into a graph as a host node. The function must not call the CUDA API.
    /// </summary>
    /// <param name="stream">The stream.</param>
    /// <param name="fn">A <c>void(void*)</c> unmanaged function pointer.</param>
    /// <param name="userData">Passed to <paramref name="fn"/>.</param>
    /// <returns>The CUresult.</returns>
    [LibraryImport(LibName)]
    internal static partial int cuLaunchHostFunc(nint stream, nint fn, nint userData);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoDAsync_v2(
        nint dstDevice, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemsetD8_v2(nint dstDevice, byte value, nuint n);

    /// <summary>
    /// Allocates page-locked ("pinned") host memory. Required for a genuinely
    /// asynchronous <see cref="cuMemcpyDtoHAsync_v2"/> — a D2H copy into pageable
    /// host memory blocks the calling thread regardless of the Async entry point
    /// (see issue #251's batched-decode design note in CudaTransformerModel.cs).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuMemHostAlloc(out nint pp, nuint bytesize, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuMemFreeHost(nint p);

    /// <summary>
    /// Page-locks an existing host range (here: the mmap'd GGUF pages of CPU-resident experts, #1202) so the device can
    /// DMA from it directly: no staging copy, and <see cref="cuMemcpyHtoDAsync_v2"/> from it is truly asynchronous.
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuMemHostRegister_v2(nint p, nuint bytesize, uint flags);

    /// <summary>Undoes <see cref="cuMemHostRegister_v2"/>; must run before the range is unmapped.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuMemHostUnregister(nint p);

    /// <summary>CU_MEMHOSTREGISTER_READ_ONLY: the device only reads the registered range (required for a read-only file mapping).</summary>
    internal const uint CU_MEMHOSTREGISTER_READ_ONLY = 0x08;

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuStreamDestroy_v2(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    /// <summary>CU_STREAM_NON_BLOCKING: the stream does not synchronize with the legacy NULL stream.</summary>
    internal const uint CU_STREAM_NON_BLOCKING = 1;

    /// <summary>
    /// Makes all future work submitted to <paramref name="stream"/> wait until <paramref name="evt"/> completes (GPU-side; the
    /// host does not block). Used by the MoE expert copy stream (#1070).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuStreamWaitEvent(nint stream, nint evt, uint flags);

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

    /// <summary>Relaxed capture mode: the thread may make "unsafe" calls such as <c>cuMemAlloc</c> while a capture is open.</summary>
    internal const uint CU_STREAM_CAPTURE_MODE_RELAXED = 2;

    [LibraryImport(LibName)]
    internal static partial int cuStreamBeginCapture_v2(nint stream, uint mode);

    /// <summary>Swaps this thread's stream-capture interaction mode; <paramref name="mode"/> receives the previous one.</summary>
    /// <param name="mode">In: the new mode. Out: the previous mode.</param>
    /// <returns>The CUresult.</returns>
    [LibraryImport(LibName)]
    internal static partial int cuThreadExchangeStreamCaptureMode(ref uint mode);

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

    /// <summary>
    /// Updates an instantiated graph in place from a freshly captured graph of the same topology (kernel
    /// arguments and launch dimensions may differ). <paramref name="resultInfo"/> points at a
    /// <c>CUgraphExecUpdateResultInfo</c> (24 bytes). Non-zero means the topology changed: re-instantiate.
    /// </summary>
    /// <param name="graphExec">The executable graph to update.</param>
    /// <param name="graph">The newly captured graph.</param>
    /// <param name="resultInfo">Receives the update result details.</param>
    /// <returns>The CUresult.</returns>
    [LibraryImport(LibName, EntryPoint = "cuGraphExecUpdate_v2")]
    internal static partial int cuGraphExecUpdate(nint graphExec, nint graph, nint resultInfo);

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

    /// <summary>
    /// Non-blocking completion check of <paramref name="evt"/>: 0 (CUDA_SUCCESS) when the work before its record has finished,
    /// <see cref="CUDA_ERROR_NOT_READY"/> while it is pending (#1071: recycling promotion-copy events).
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cuEventQuery(nint evt);

    /// <summary>CUDA_ERROR_NOT_READY: the queried work has not completed yet.</summary>
    internal const int CUDA_ERROR_NOT_READY = 600;

    /// <summary>CU_EVENT_DISABLE_TIMING: no timestamp (cheaper to record and wait on).</summary>
    internal const uint CU_EVENT_DISABLE_TIMING = 2;

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
