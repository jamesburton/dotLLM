using System.Runtime.InteropServices;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.DirectStorage;

/// <summary>
/// Loads as-is weight tensors from the GGUF straight into device-local memory with Windows DirectStorage (#1304): the bytes
/// go disk -&gt; shared D3D12 heap, and Vulkan imports that heap (<c>VK_KHR_external_memory_win32</c>) as ordinary
/// device-local memory. Each tensor is a sub-buffer of an imported heap, like a <see cref="VulkanWeightArena"/> slab.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why not import the mmap'd file pages</b> (#1250, <c>tools/d3d12-vk-import-spike</c>): file-backed host pages are about
/// 2x slower than device-local memory under weight reuse (Qwen3.8 dense pp512 -52%), while a D3D12 default heap imported
/// into Vulkan runs at device-local speed. DirectStorage fills it from the file without a process-private or page-cache
/// duplicate (CLAUDE.md rule 3). Its own bounded staging ring (<see cref="StagingMiB"/>) is the one host buffer on the path;
/// accepted by the user on 2026-10-26 and freed again at <see cref="Seal"/>.
/// </para>
/// <para>
/// <b>Contract.</b> <see cref="TryLoad"/> only sees tensors whose device bytes equal their file bytes (the callers are the
/// <see cref="VulkanWeightImportPolicy.TryImport"/> sites), and returns false - never throws - for anything it cannot do:
/// not Windows, the DLL or the extension is missing, the adapter cannot be matched, the file is held open for write, a heap
/// cannot be created. The caller then stages as before. Reads are asynchronous; <see cref="WaitAll"/> (called by
/// <see cref="VulkanStagingBuffer"/> at the end of each load) blocks until every read landed, and a failed read throws there.
/// </para>
/// <para>Default on Windows UMA when <c>dstorage.dll</c> is present (#1994); see <see cref="EnabledFor"/>.</para>
/// </remarks>
internal sealed unsafe class DirectWeightStore : IDisposable
{
    /// <summary>Shared slab size for small tensors: 128 MiB. A new one is opened when the current one is full.</summary>
    internal const long SlabBytes = 128L << 20;

    /// <summary>Tensors at least this large get an exactly-sized heap of their own (no slab tail to waste).</summary>
    internal const long DedicatedThresholdBytes = 16L << 20;

    /// <summary>D3D12 heap alignment: 64 KiB, or 4 MiB with <c>DOTLLM_VULKAN_DS_HEAP_ALIGN_MIB=4</c> (the only two values D3D12 accepts).</summary>
    private static readonly long HeapAlignment =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DS_HEAP_ALIGN_MIB") == "4" ? 4L << 20 : 64L << 10;
    /// <summary>Size of one DirectStorage read request (MiB), <c>DOTLLM_VULKAN_DS_REQUEST_MIB</c>, default 16.</summary>
    private static readonly uint RequestBytes =
        (uint.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DS_REQUEST_MIB"), out uint r) && r is >= 1 and <= 1024 ? r : 16u) << 20;
    private const int MaxPendingRequests = 6000;

    /// <summary>DirectStorage staging ring size (MiB), <c>DOTLLM_VULKAN_DS_STAGING_MIB</c>, default 64.</summary>
    internal static int StagingMiB { get; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DS_STAGING_MIB"), out int v) && v is >= 4 and <= 2048 ? v : 64;

    /// <summary>dstorage.dll found (cached: the per-tensor gate must be cheap).</summary>
    private static readonly Lazy<bool> s_dllAvailable = new(DirectStorageInstaller.IsAvailable);

    /// <summary>
    /// Whether the DirectStorage tier runs for <paramref name="device"/> (#1994). <c>DOTLLM_VULKAN_DIRECT_STORAGE</c>: <c>0</c> = never,
    /// <c>1</c> = always (any device, DLL permitting), unset = <b>on for an integrated (UMA) device on Windows when <c>dstorage.dll</c> is
    /// present</b>. Discrete GPUs keep staging by default (their weights must reach VRAM anyway). Read per call (tests toggle it).
    /// </summary>
    internal static bool EnabledFor(VulkanDevice device)
        => Decide(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE"), OperatingSystem.IsWindows(),
            device.PhysicalDeviceTypeValue == VkPhysicalDeviceType.IntegratedGpu, s_dllAvailable.Value);

    /// <summary>The pure tri-state decision behind <see cref="EnabledFor"/> (testable without a device).</summary>
    internal static bool Decide(string? env, bool windows, bool integrated, bool dllPresent)
    {
        if (string.Equals(env, "0", StringComparison.Ordinal)) return false;
        if (string.Equals(env, "1", StringComparison.Ordinal)) return true;
        return windows && integrated && dllPresent;
    }

    /// <summary>True when the default would apply but for a missing DLL: Windows, integrated device, tier not switched off.</summary>
    internal static bool WouldBeDefaultButDllMissing(VulkanDevice device)
    {
        string? v = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE");
        if (v is "0" or "1") return false;
        return OperatingSystem.IsWindows()
            && device.PhysicalDeviceTypeValue == VkPhysicalDeviceType.IntegratedGpu
            && DirectStorageInstaller.IsSupportedPlatform && !s_dllAvailable.Value;
    }

    private static readonly object s_lock = new();
    private static readonly Dictionary<VulkanDevice, DirectWeightStore> s_byDevice = [];

    /// <summary>The store for <paramref name="device"/>, created on first use; null when this platform / device cannot use it.</summary>
    internal static DirectWeightStore? For(VulkanDevice device)
    {
        if (!OperatingSystem.IsWindows() || !device.HasExternalMemoryWin32) return null;
        lock (s_lock)
        {
            if (s_byDevice.TryGetValue(device, out var existing)) return existing.Failed ? null : existing;
            var created = new DirectWeightStore(device);
            s_byDevice[device] = created;
            return created.Failed ? null : created;
        }
    }

    /// <summary>Why <see cref="For"/> returned null for <paramref name="device"/> (diagnostics).</summary>
    internal static string UnavailableReason(VulkanDevice device)
    {
        if (!OperatingSystem.IsWindows()) return "not_windows";
        if (!device.HasExternalMemoryWin32) return "no_VK_KHR_external_memory_win32";
        lock (s_lock)
            return s_byDevice.TryGetValue(device, out var s) ? s.FailureReason : "no_store";
    }

    /// <summary>The existing store for <paramref name="device"/>, or null when none was created or it failed.</summary>
    internal static DirectWeightStore? TryGet(VulkanDevice device)
    {
        lock (s_lock)
            return s_byDevice.TryGetValue(device, out var s) && !s.Failed ? s : null;
    }

    /// <summary>Releases and forgets the store of a device that is going away.</summary>
    internal static void Forget(VulkanDevice device)
    {
        DirectWeightStore? store;
        lock (s_lock)
        {
            if (!s_byDevice.Remove(device, out store)) return;
        }
        store.Dispose();
    }

    private sealed class Heap
    {
        public nint D3dHeap;
        public nint Resource;
        public required VulkanDevice.Buffer Slab;
        public long Used;
        public bool Dedicated;

        /// <summary>Drops the D3D12 heap + resource (idempotent). Vulkan's import holds its own reference to the allocation.</summary>
        public void ReleaseD3d()
        {
            nint r = Resource, h = D3dHeap;
            if (h != 0) Interlocked.Decrement(ref s_liveD3dHeaps);
            Resource = 0;
            D3dHeap = 0;
            ComVtbl.Release(r);
            ComVtbl.Release(h);
        }
    }

    private static int s_liveD3dHeaps;

    /// <summary>D3D12 heap objects currently held by any store (created minus released). Tests and diagnostics.</summary>
    internal static int LiveD3d12Heaps => Volatile.Read(ref s_liveD3dHeaps);

    /// <summary>
    /// D3D12 "local" usage fraction (of the DXGI budget) that DirectStorage heaps may not push the process past,
    /// <c>DOTLLM_VULKAN_DS_BUDGET_PCT</c> (default 75). Measured on Strix Halo (#2044): the device is removed at
    /// ~85 GiB of a 101 GiB budget (84%), so the margin is deliberately generous.
    /// </summary>
    internal static double BudgetFraction { get; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DS_BUDGET_PCT"), out int pct) && pct is >= 10 and <= 100 ? pct / 100.0 : 0.75;

    /// <summary>
    /// Retire window, <c>DOTLLM_VULKAN_DS_RETIRE_MIB</c> (default 2048, range 64..65536): once this many bytes of reads were enqueued since
    /// the last drain, the next heap first drains and releases the D3D12 side of the finished dedicated heaps (#2044).
    /// </summary>
    internal static long RetireEveryBytes { get; } =
        (long.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DS_RETIRE_MIB"), out long rm) && rm is >= 64 and <= 65536 ? rm : 2048) << 20;

    /// <summary>
    /// The admission rule (#2044), pure so it is testable without a device: may a heap of <paramref name="heapBytes"/> be created
    /// given D3D12's <paramref name="currentUsage"/> and <paramref name="budget"/>? While a heap is live on both sides (the D3D12
    /// heap and the Vulkan import of it are each charged) it costs <b>twice</b> its size, so the projection is
    /// <c>usage + 2 x heap</c>. An unknown budget (0) admits.
    /// </summary>
    internal static bool AdmitHeap(long currentUsage, long budget, long heapBytes, double fraction)
        => budget <= 0 || currentUsage + 2 * heapBytes <= (long)(budget * fraction);

    /// <summary>This store's retire window (bytes); starts at <see cref="RetireEveryBytes"/>. Tests shrink it so the mid-load retire runs on small data.</summary>
    internal long RetireWindowBytes { get; set; } = RetireEveryBytes;

    private readonly VulkanDevice _device;
    private readonly Dictionary<string, nint> _files = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<Heap> _heaps = [];
    private nint _d3dDevice;
    private nint _adapter;
    private nint _fence;
    private nint _event;
    private ulong _fenceValue;
    private nint _dsFactory;
    private nint _dsQueue;
    private int _pendingRequests;
    private long _inflightBytes;
    private bool _budgetWarned;
    private bool _disposed;

    /// <summary>Why the store is unusable on this device (empty when usable).</summary>
    internal string FailureReason { get; private set; } = string.Empty;

    private bool Failed => FailureReason.Length > 0;

    /// <summary>Tensors loaded through DirectStorage since the last <see cref="Reset"/>.</summary>
    internal int LoadedTensorCount { get; private set; }

    /// <summary>Bytes read through DirectStorage since the last <see cref="Reset"/>.</summary>
    internal long LoadedBytes { get; private set; }

    /// <summary>Last reason <see cref="TryLoad"/> declined a tensor (empty after a success).</summary>
    internal string LastFallbackReason { get; private set; } = string.Empty;

    private DirectWeightStore(VulkanDevice device)
    {
        _device = device;
        try
        {
            ulong? luid = device.DeviceLuid();
            if (luid is null) { FailureReason = "no_adapter_luid"; return; }

            int hr = D3dNative.CreateDXGIFactory1(D3dNative.IidDxgiFactory4, out nint factory);
            if (hr < 0) { FailureReason = $"CreateDXGIFactory1(0x{hr:X8})"; return; }
            try
            {
                hr = D3dNative.EnumAdapterByLuid(factory, luid.Value, out nint adapter);
                if (hr < 0) { FailureReason = $"EnumAdapterByLuid(0x{hr:X8})"; return; }
                try
                {
                    hr = D3dNative.D3D12CreateDevice(adapter, D3dNative.D3D_FEATURE_LEVEL_11_0, D3dNative.IidD3D12Device, out _d3dDevice);
                    if (hr < 0) { FailureReason = $"D3D12CreateDevice(0x{hr:X8})"; return; }
                    _adapter = adapter; // kept for QueryVideoMemoryInfo; released in ReleaseDevice
                    adapter = 0;
                }
                finally { ComVtbl.Release(adapter); }
            }
            finally { ComVtbl.Release(factory); }

            hr = D3dNative.CreateFence(_d3dDevice, out _fence);
            if (hr < 0) { FailureReason = $"CreateFence(0x{hr:X8})"; return; }
            _event = D3dNative.CreateEvent(0, 0, 0, 0);
            if (_event == 0) { FailureReason = "CreateEvent"; return; }

            if (!TryLoadDStorage(out string why)) { FailureReason = why; return; }
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            FailureReason = $"{e.GetType().Name}: {e.Message}";
        }
        if (Failed) ReleaseDevice();
    }

    private static nint s_dstorageModule;
    private static delegate* unmanaged[Stdcall]<Guid*, nint*, int> s_getFactory;

    private static bool TryLoadDStorage(out string why)
    {
        why = string.Empty;
        if (s_getFactory != null) return true;
        string? found = DirectStorageInstaller.FindDll();
        nint module = 0;
        // Full path, restricted search (own folder + System32): a dstorage.dll in the CWD or on PATH is never loaded.
        // dstorage.dll does NOT import dstoragecore.dll: it LoadLibrary's it by bare name at DStorageGetFactory time, with the default search
        // order. For a pair outside the application directory (the install dir after the NuGet fetch, DOTLLM_DIRECTSTORAGE_DIR) that fails and the
        // factory returns E_NOTIMPL (0x80004001), so every tensor is declined (#1994 regression of the LoadLibraryEx change). Load the core by
        // FULL PATH first, from the same directory: the later bare-name load then resolves to this already-mapped module, and no CWD/PATH search runs.
        if (found is not null && !DirectStorageInstaller.PairVersionsMatch(Path.GetDirectoryName(found)!))
        {
            why = "dstorage_pair_version_mismatch";
            return false;
        }
        if (found is not null)
        {
            string core = Path.Combine(Path.GetDirectoryName(found)!, "dstoragecore.dll");
            if (!DirectStorageInstaller.TryLoad(core, out _))
            {
                why = "dstoragecore.dll_not_loadable";
                return false;
            }
        }
        if (found is null || !DirectStorageInstaller.TryLoad(found, out module))
        {
            why = "dstorage.dll_not_found";
            return false;
        }
        if (!NativeLibrary.TryGetExport(module, "DStorageGetFactory", out nint export))
        {
            why = "DStorageGetFactory_missing";
            return false;
        }
        s_dstorageModule = module;
        s_getFactory = (delegate* unmanaged[Stdcall]<Guid*, nint*, int>)export;
        return true;
    }

    /// <summary>
    /// Tries to load <paramref name="bytes"/> bytes of the GGUF that <paramref name="srcPtr"/> points into, straight into
    /// device-local memory. False (with <see cref="LastFallbackReason"/>) means "stage it as usual".
    /// </summary>
    /// <param name="srcPtr">Address of the tensor inside an open <see cref="GgufFile"/>'s mapping.</param>
    /// <param name="bytes">Tensor size in bytes (as stored in the file).</param>
    /// <param name="buf">The tensor's device buffer; its size is <paramref name="bytes"/> rounded up to a uint.</param>
    internal bool TryLoad(nint srcPtr, long bytes, out VulkanDevice.Buffer? buf)
    {
        buf = null;
        if (_disposed || Failed) return Decline("store_unavailable");
        if (!GgufFile.TryResolveFileRange(srcPtr, bytes, out string path, out long fileOffset))
            return Decline("not_file_backed");

        long padded = (bytes + 3) & ~3L;
        if (padded > (long)_device.MaxStorageBufferRange) return Decline("exceeds_storage_buffer_range");
        nint dsFile;
        try
        {
            if (!EnsureSession(out string sessionWhy)) return Decline(sessionWhy);
            if (!_files.TryGetValue(path, out dsFile))
            {
                int hr = DStorageNative.OpenFile(_dsFactory, path, out dsFile);
                if (hr < 0) return Decline($"OpenFile(0x{hr:X8})");
                _files[path] = dsFile;
            }
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            FailureReason = e.Message;
            return Decline("store_unavailable");
        }

        // Place it: a dedicated, exactly-sized heap for big tensors; otherwise first fit in the open slab.
        var sub = Place(padded, out string placeWhy);
        if (sub is null) return Decline(placeWhy);
        var (heap, subBuffer, offset) = sub.Value;

        // Pending requests are bounded by the queue capacity: drain before it fills.
        long remaining = bytes;
        long fileCursor = fileOffset;
        long destCursor = offset;
        while (remaining > 0)
        {
            if (_pendingRequests >= MaxPendingRequests) DrainQueue();
            uint chunk = (uint)Math.Min(remaining, RequestBytes);
            var req = new DStorageNative.Request
            {
                OptionsWord0 = 0,
                OptionsWord1 = 2, // source = FILE (bit 0 = 0), destination = BUFFER (bits 1..7 = 1)
                SourceFile = dsFile,
                SourceOffset = (ulong)fileCursor,
                SourceSize = chunk,
                DestResource = heap.Resource,
                DestOffset = (ulong)destCursor,
                DestSize = chunk,
                UncompressedSize = chunk,
            };
            DStorageNative.EnqueueRequest(_dsQueue, req);
            _pendingRequests++;
            _inflightBytes += chunk;
            remaining -= chunk;
            fileCursor += chunk;
            destCursor += chunk;
        }
        DStorageNative.Submit(_dsQueue);
        if (s_trace && D3dNative.GetDeviceRemovedReason(_d3dDevice) != 0)
            Console.Error.WriteLine($"[ds-mem] DEVICE REMOVED right after enqueue of {bytes >> 20} MiB: {Diagnose()}");

        LoadedTensorCount++;
        LoadedBytes += bytes;
        LastFallbackReason = string.Empty;
        buf = subBuffer;
        return true;
    }

    private bool Decline(string reason)
    {
        LastFallbackReason = reason;
        return false;
    }

    private bool EnsureSession(out string why)
    {
        why = string.Empty;
        if (_dsQueue != 0) return true;
        if (_dsFactory == 0)
        {
            Guid iid = DStorageNative.IidFactory;
            nint f = 0;
            int hr = s_getFactory(&iid, &f);
            if (hr < 0) { why = $"DStorageGetFactory(0x{hr:X8})"; return false; }
            _dsFactory = f;
            DStorageNative.SetStagingBufferSize(_dsFactory, (uint)StagingMiB << 20);
        }

        var qd = new DStorageNative.QueueDesc
        {
            SourceType = 0, // FILE
            Capacity = DStorageNative.MaxQueueCapacity,
            Priority = 0,
            Device = _d3dDevice,
        };
        int qr = DStorageNative.CreateQueue(_dsFactory, qd, out _dsQueue);
        if (qr < 0) { why = $"CreateQueue(0x{qr:X8})"; _dsQueue = 0; return false; }
        _pendingRequests = 0;
        return true;
    }

    private (Heap Heap, VulkanDevice.Buffer Buffer, long Offset)? Place(long padded, out string why)
    {
        why = string.Empty;
        if (padded >= DedicatedThresholdBytes)
        {
            var heap = OpenHeap(AlignUp(padded, HeapAlignment), dedicated: true, out why);
            if (heap is null) return null;
            return CarveSubBuffer(heap, padded, out why);
        }

        // Small tensor: first fit over every open slab (the tail a big tensor left behind is reused by later small ones, which
        // keeps the unused bytes - allocated device memory - to a fraction of one slab), then a fresh slab.
        foreach (var open in _heaps)
        {
            if (open.Dedicated) continue;
            var carved = CarveSubBuffer(open, padded, out why);
            if (carved is not null) return carved;
            if (why != "slab_full") return null;
        }

        var fresh = OpenHeap(SlabBytes, dedicated: false, out why);
        if (fresh is null) return null;
        return CarveSubBuffer(fresh, padded, out why);
    }

    private (Heap, VulkanDevice.Buffer, long)? CarveSubBuffer(Heap heap, long padded, out string why)
    {
        why = string.Empty;
        var ext = new VkExternalMemoryBufferCreateInfo { sType = 1000072000, handleTypes = VulkanDevice.ExternalMemoryHandleD3D12Heap };
        var bci = new VkBufferCreateInfo
        {
            sType = VkStructureType.BufferCreateInfo,
            pNext = (nint)(&ext),
            size = (ulong)padded,
            usage = VulkanDevice.BufferUsage,
            sharingMode = VkSharingMode.Exclusive,
        };
        int cr = VulkanApi.vkCreateBuffer(_device.Handle, bci, 0, out nint buffer);
        if (cr < 0) { why = $"vkCreateBuffer({cr})"; return null; }

        VulkanApi.vkGetBufferMemoryRequirements(_device.Handle, buffer, out var req);
        long align = Math.Max(256L, (long)req.alignment);
        long offset = AlignUp(heap.Used, align);
        if (offset + (long)req.size > heap.Slab.Size)
        {
            VulkanApi.vkDestroyBuffer(_device.Handle, buffer, 0);
            why = "slab_full";
            return null;
        }
        if (heap.Slab.TypeIndex < 32 && (req.memoryTypeBits & (1u << (int)heap.Slab.TypeIndex)) == 0)
        {
            VulkanApi.vkDestroyBuffer(_device.Handle, buffer, 0);
            why = "memory_type_mismatch";
            return null;
        }
        int br = VulkanApi.vkBindBufferMemory(_device.Handle, buffer, heap.Slab.Memory, (ulong)offset);
        if (br < 0)
        {
            VulkanApi.vkDestroyBuffer(_device.Handle, buffer, 0);
            why = $"vkBindBufferMemory({br})";
            return null;
        }
        heap.Used = offset + (long)req.size;
        return (heap, new VulkanDevice.Buffer(_device, buffer, heap.Slab, offset, padded), offset);
    }

    private Heap? OpenHeap(long bytes, bool dedicated, out string why)
    {
        why = string.Empty;
        if (_inflightBytes >= RetireWindowBytes) DrainAndRetire();
        if (!AdmitAgainstBudget(bytes, out why)) return null;
        var desc = new D3dNative.HeapDesc
        {
            SizeInBytes = (ulong)bytes,
            Type = D3dNative.HeapTypeDefault,
            Alignment = (ulong)HeapAlignment,
            Flags = D3dNative.HeapFlagShared | D3dNative.HeapFlagAllowOnlyBuffers,
        };
        int hr = D3dNative.CreateHeap(_d3dDevice, desc, out nint d3dHeap);
        if (hr < 0) { why = $"CreateHeap(0x{hr:X8}; {Diagnose()})"; return null; }
        if (s_trace && (_heaps.Count & 15) == 0) Console.Error.WriteLine($"[ds-mem] heaps={_heaps.Count} +{bytes >> 20} MiB; {Diagnose()}");

        nint resource = 0, sharedHandle = 0;
        VulkanDevice.Buffer? slab = null;
        try
        {
            if (s_trace && _heaps.Count is 0 or 1 or 2 or 3) Console.Error.WriteLine($"[ds-mem] stage after CreateHeap {bytes >> 20} MiB: {Diagnose()}");
            hr = D3dNative.CreateBufferResource(_d3dDevice, d3dHeap, 0, (ulong)bytes, out resource);
            if (hr < 0) { why = $"CreatePlacedResource(0x{hr:X8}; {Diagnose()})"; return null; }
            hr = D3dNative.CreateSharedHandle(_d3dDevice, d3dHeap, out sharedHandle);
            if (hr < 0) { why = $"CreateSharedHandle(0x{hr:X8})"; return null; }

            if (s_trace && _heaps.Count is 0 or 1 or 2 or 3) Console.Error.WriteLine($"[ds-mem] stage after shared handle: {Diagnose()}");
            slab = _device.TryImportD3D12Heap(sharedHandle, bytes, out why);
            if (s_trace && _heaps.Count is 0 or 1 or 2 or 3) Console.Error.WriteLine($"[ds-mem] stage after vk import: {Diagnose()}");
            if (slab is null) return null;

            Interlocked.Increment(ref s_liveD3dHeaps);
            slab.AddSlabRef(); // this store's reference, released by Seal / Dispose
            var heap = new Heap { D3dHeap = d3dHeap, Resource = resource, Slab = slab, Dedicated = dedicated };
            slab.FreeHook = heap.ReleaseD3d; // no-op when the heap was already retired after its reads landed
            _heaps.Add(heap);
            d3dHeap = 0;
            resource = 0;
            return heap;
        }
        finally
        {
            if (sharedHandle != 0) D3dNative.CloseHandle(sharedHandle);
            if (slab is null || d3dHeap != 0)
            {
                ComVtbl.Release(resource);
                ComVtbl.Release(d3dHeap);
            }
        }
    }

    private static readonly bool s_trace = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MEM_TRACE") == "1";

    /// <summary>D3D12's current "local" usage of this process on the adapter (bytes), 0 when unknown. Tests and diagnostics.</summary>
    internal long D3d12LocalUsageBytes()
        => _adapter != 0 && D3dNative.QueryVideoMemoryInfo(_adapter, false, out var i) >= 0 ? (long)i.CurrentUsage : 0;

    /// <summary>"removed=0x.. local usage/budget MiB, nonlocal usage/budget MiB, ds-live MiB": what D3D12 itself thinks (diagnostics).</summary>
    internal string Diagnose()
    {
        if (_d3dDevice == 0 || _adapter == 0) return "no_device";
        int removed = D3dNative.GetDeviceRemovedReason(_d3dDevice);
        int lhr = D3dNative.QueryVideoMemoryInfo(_adapter, false, out var l);
        string local = lhr >= 0 ? $"local {l.CurrentUsage >> 20}/{l.Budget >> 20} MiB (avail-for-reservation {l.AvailableForReservation >> 20})" : $"local n/a(0x{lhr:X8})";
        int nhr = D3dNative.QueryVideoMemoryInfo(_adapter, true, out var n);
        string non = nhr >= 0 ? $"nonlocal {n.CurrentUsage >> 20}/{n.Budget >> 20} MiB" : $"nonlocal n/a(0x{nhr:X8})";
        D3dNative.GetAdapterMemory(_adapter, out long ded, out long shr);
        return $"removed=0x{removed:X8}, adapter dedicated {ded >> 20} MiB shared {shr >> 20} MiB, {local}, {non}, heaps={_heaps.Count}, ds_loaded={LoadedBytes >> 20} MiB";
    }

    /// <summary>Checks D3D12's own usage/budget (what the OS acts on) before a heap is created; retires finished heaps once if over.</summary>
    private bool AdmitAgainstBudget(long bytes, out string why)
    {
        why = string.Empty;
        if (_adapter == 0 || D3dNative.QueryVideoMemoryInfo(_adapter, false, out var info) < 0) return true;
        if (AdmitHeap((long)info.CurrentUsage, (long)info.Budget, bytes, BudgetFraction)) return true;
        DrainAndRetire(); // finished heaps give back their D3D12-side charge
        if (D3dNative.QueryVideoMemoryInfo(_adapter, false, out info) < 0
            || AdmitHeap((long)info.CurrentUsage, (long)info.Budget, bytes, BudgetFraction)) return true;
        why = $"d3d12_budget(usage {info.CurrentUsage >> 20} MiB + 2 x {bytes >> 20} MiB exceeds {(int)(BudgetFraction * 100)}% of {info.Budget >> 20} MiB)";
        if (!_budgetWarned)
        {
            _budgetWarned = true;
            Console.Error.WriteLine($"[directstorage] declining further DirectStorage heaps: {why}. Remaining weights take the staged path (host copy through the Vulkan staging ring; page-cache read, so no zero-copy).");
        }
        return false;
    }

    private static long AlignUp(long value, long alignment) => (value + alignment - 1) / alignment * alignment;

    /// <summary>
    /// Waits for every enqueued read, then drops the D3D12 side (heap + resource) of every dedicated heap: its reads have landed and
    /// Vulkan's import keeps the allocation alive, so the second charge (#2044: D3D12 and Vulkan each count a shared heap) goes away.
    /// Only called between tensors, never while a tensor's requests are still being enqueued. Open slabs stay until <see cref="Seal"/>.
    /// </summary>
    private void DrainAndRetire()
    {
        WaitAll();
        foreach (var h in _heaps)
            if (h.Dedicated) h.ReleaseD3d();
    }

    private void DrainQueue()
    {
        if (_dsQueue == 0) return;
        WaitAll();
    }

    /// <summary>
    /// Blocks until every enqueued read has landed in its heap. Throws when DirectStorage reports a failed request (the
    /// weights would be incomplete). Cheap when nothing is pending.
    /// </summary>
    internal void WaitAll()
    {
        if (_dsQueue == 0 || _pendingRequests == 0) return;
        ulong value = ++_fenceValue;
        DStorageNative.EnqueueSignal(_dsQueue, _fence, value);
        DStorageNative.Submit(_dsQueue);
        if (D3dNative.FenceGetCompletedValue(_fence) < value)
        {
            int hr = D3dNative.FenceSetEventOnCompletion(_fence, value, _event);
            if (hr < 0) throw new InvalidOperationException($"DirectStorage fence wait failed (0x{hr:X8}).");
            D3dNative.WaitForSingleObject(_event, 0xFFFFFFFF);
        }
        _pendingRequests = 0;
        _inflightBytes = 0;

        byte* record = stackalloc byte[DStorageNative.ErrorRecordSize];
        new Span<byte>(record, DStorageNative.ErrorRecordSize).Clear();
        DStorageNative.RetrieveErrorRecord(_dsQueue, record);
        uint failures = *(uint*)(record + DStorageNative.ErrorRecordFailureCountOffset);
        if (failures != 0)
        {
            int hr = *(int*)(record + DStorageNative.ErrorRecordHResultOffset);
            throw new IOException($"DirectStorage failed {failures} weight read request(s); first failure 0x{hr:X8}. The model weights are incomplete.");
        }
    }

    /// <summary>Clears the per-load counters.</summary>
    internal void Reset()
    {
        LoadedTensorCount = 0;
        LoadedBytes = 0;
        LastFallbackReason = string.Empty;
    }

    /// <summary>
    /// Ends a load: waits for the reads, closes the file handles and the DirectStorage queue and factory (freeing the staging
    /// ring), and drops this store's reference to every heap so each is freed when its last tensor is disposed.
    /// </summary>
    internal void Seal()
    {
        try
        {
            WaitAll();
            foreach (var h in _heaps) h.ReleaseD3d(); // every read landed; Vulkan owns the memory from here
            if (s_trace) Console.Error.WriteLine($"[ds-mem] seal: {Diagnose()}");
        }
        finally
        {
            foreach (var f in _files.Values) DStorageNative.CloseFile(f);
            _files.Clear();
            DStorageNative.CloseQueue(_dsQueue);
            _dsQueue = 0;
            ComVtbl.Release(_dsFactory);
            _dsFactory = 0;
            _pendingRequests = 0;
            foreach (var h in _heaps) h.Slab.ReleaseSlabRef();
            _heaps.Clear();
        }
    }

    private void ReleaseDevice()
    {
        if (_event != 0) { D3dNative.CloseHandle(_event); _event = 0; }
        ComVtbl.Release(_fence); _fence = 0;
        ComVtbl.Release(_d3dDevice); _d3dDevice = 0;
        ComVtbl.Release(_adapter); _adapter = 0;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { Seal(); } catch { /* disposing a store whose reads failed must not mask the original error */ }
        ReleaseDevice();
    }
}
