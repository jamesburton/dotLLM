using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.DirectStorage;

/// <summary>
/// Hand-rolled, allocation-free COM interop for the handful of D3D12 / DXGI / DirectStorage calls #1304 needs. Plain
/// vtable slot calls through <c>delegate* unmanaged</c>: no <c>ComWrappers</c>, no generated COM, so it is trim- and
/// AOT-friendly and adds no managed dependency. Slot numbers were taken from the Windows SDK / DirectStorage 1.3 headers
/// (C <c>offsetof</c> on the <c>*Vtbl</c> structs, and declaration order in <c>dstorage.h</c>) and are exercised end to end by
/// <c>DirectWeightStoreTests</c> (a byte-identity read-back of a DirectStorage-loaded range).
/// </summary>
internal static unsafe class ComVtbl
{
    /// <summary>Returns the function pointer in vtable slot <paramref name="index"/> of COM object <paramref name="obj"/>.</summary>
    public static nint Slot(nint obj, int index) => ((nint*)*(nint*)obj)[index];

    /// <summary><c>IUnknown::Release</c>; no-op for a null pointer.</summary>
    public static void Release(nint obj)
    {
        if (obj != 0) ((delegate* unmanaged[Stdcall]<nint, uint>)Slot(obj, 2))(obj);
    }
}

/// <summary>Raw entry points. Only ever called after an OS check, and any <see cref="DllNotFoundException"/> disables the feature.</summary>
internal static unsafe partial class D3dNative
{
    internal static readonly Guid IidDxgiFactory4 = new("1bc6ea02-ef36-464f-bf0c-21ca39e5168a");
    internal static readonly Guid IidD3D12Device = new("189819f1-1db6-4b57-be54-1821339b85f7");
    internal static readonly Guid IidD3D12Heap = new("6b3b2502-6e51-45b3-90ee-9884265e8df3");
    internal static readonly Guid IidD3D12Fence = new("0a753dcf-c4d8-4b91-adf6-be5a60d95a76");
    internal static readonly Guid IidD3D12Resource = new("696442be-a72e-4059-bc79-5b5c98040fad");

    internal const int D3D_FEATURE_LEVEL_11_0 = 0xb000;

    [LibraryImport("dxgi.dll")]
    internal static partial int CreateDXGIFactory1(in Guid riid, out nint factory);

    [LibraryImport("d3d12.dll")]
    internal static partial int D3D12CreateDevice(nint adapter, int minimumFeatureLevel, in Guid riid, out nint device);

    [LibraryImport("kernel32.dll", EntryPoint = "CreateEventW")]
    internal static partial nint CreateEvent(nint attributes, int manualReset, int initialState, nint name);

    [LibraryImport("kernel32.dll")]
    internal static partial uint WaitForSingleObject(nint handle, uint milliseconds);

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    internal static partial bool CloseHandle(nint handle);

    // ---- IDXGIFactory4 ----
    private const int FactoryEnumAdapterByLuid = 26;

    /// <summary><c>IDXGIFactory4::EnumAdapterByLuid</c>.</summary>
    internal static int EnumAdapterByLuid(nint factory, ulong luid, out nint adapter)
    {
        nint a = 0;
        fixed (Guid* iid = &IidAdapter)
        {
            // LUID is an 8-byte struct: passed by value in one register on x64.
            int hr = ((delegate* unmanaged[Stdcall]<nint, ulong, Guid*, nint*, int>)ComVtbl.Slot(factory, FactoryEnumAdapterByLuid))
                (factory, luid, iid, &a);
            adapter = a;
            return hr;
        }
    }

    internal static readonly Guid IidAdapter = new("2411e7e1-12ac-4ccf-bd14-9798e8534dc0"); // IDXGIAdapter

    // ---- ID3D12Device ----
    private const int DeviceCreateHeap = 28, DeviceCreatePlacedResource = 29, DeviceCreateSharedHandle = 31, DeviceCreateFence = 36;

    /// <summary>D3D12_HEAP_DESC (48 bytes): Size@0, Properties@8 (5 x u32), Alignment@32, Flags@40.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct HeapDesc
    {
        public ulong SizeInBytes;
        public int Type;
        public int CpuPageProperty;
        public int MemoryPoolPreference;
        public uint CreationNodeMask;
        public uint VisibleNodeMask;
        public ulong Alignment;
        public int Flags;
    }

    internal const int HeapTypeDefault = 1;
    internal const int HeapFlagShared = 0x1;
    internal const int HeapFlagAllowOnlyBuffers = 0xC0;

    /// <summary>D3D12_RESOURCE_DESC (56 bytes) for a plain buffer.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ResourceDesc
    {
        public int Dimension;            // 1 = BUFFER
        public ulong Alignment;
        public ulong Width;
        public uint Height;
        public ushort DepthOrArraySize;
        public ushort MipLevels;
        public int Format;
        public uint SampleCount;
        public uint SampleQuality;
        public int Layout;               // 1 = ROW_MAJOR
        public int Flags;
    }

    internal static int CreateHeap(nint device, in HeapDesc desc, out nint heap)
    {
        nint h = 0;
        fixed (HeapDesc* d = &desc)
        fixed (Guid* iid = &IidD3D12Heap)
        {
            int hr = ((delegate* unmanaged[Stdcall]<nint, HeapDesc*, Guid*, nint*, int>)ComVtbl.Slot(device, DeviceCreateHeap))(device, d, iid, &h);
            heap = h;
            return hr;
        }
    }

    internal static int CreateBufferResource(nint device, nint heap, ulong offset, ulong bytes, out nint resource)
    {
        var rd = new ResourceDesc { Dimension = 1, Width = bytes, Height = 1, DepthOrArraySize = 1, MipLevels = 1, SampleCount = 1, Layout = 1 };
        nint r = 0;
        fixed (Guid* iid = &IidD3D12Resource)
        {
            // initial state COMMON (0), no clear value
            int hr = ((delegate* unmanaged[Stdcall]<nint, nint, ulong, ResourceDesc*, int, nint, Guid*, nint*, int>)
                ComVtbl.Slot(device, DeviceCreatePlacedResource))(device, heap, offset, &rd, 0, 0, iid, &r);
            resource = r;
            return hr;
        }
    }

    internal static int CreateSharedHandle(nint device, nint heap, out nint handle)
    {
        nint h = 0;
        // GENERIC_ALL = 0x10000000
        int hr = ((delegate* unmanaged[Stdcall]<nint, nint, nint, uint, nint, nint*, int>)ComVtbl.Slot(device, DeviceCreateSharedHandle))
            (device, heap, 0, 0x10000000, 0, &h);
        handle = h;
        return hr;
    }

    internal static int CreateFence(nint device, out nint fence)
    {
        nint f = 0;
        fixed (Guid* iid = &IidD3D12Fence)
        {
            int hr = ((delegate* unmanaged[Stdcall]<nint, ulong, int, Guid*, nint*, int>)ComVtbl.Slot(device, DeviceCreateFence))(device, 0, 0, iid, &f);
            fence = f;
            return hr;
        }
    }

    // ---- ID3D12Fence ----
    internal static int FenceSetEventOnCompletion(nint fence, ulong value, nint evt)
        => ((delegate* unmanaged[Stdcall]<nint, ulong, nint, int>)ComVtbl.Slot(fence, 9))(fence, value, evt);

    internal static ulong FenceGetCompletedValue(nint fence)
        => ((delegate* unmanaged[Stdcall]<nint, ulong>)ComVtbl.Slot(fence, 8))(fence);
}

/// <summary>DirectStorage 1.3 (<c>dstorage.dll</c>) structs and vtable calls. Layouts verified against <c>dstorage.h</c> with MSVC-compatible <c>sizeof/offsetof</c>.</summary>
internal static unsafe class DStorageNative
{
    internal static readonly Guid IidFactory = new("6924ea0c-c3cd-4826-b10a-f64f4ed927c1");
    internal static readonly Guid IidFile = new("5de95e7b-955a-4868-a73c-243b29f4b8da");
    internal static readonly Guid IidQueue = new("cfdbd83f-9e06-4fda-8ea5-69042137f49b");

    /// <summary>DSTORAGE_QUEUE_DESC, 32 bytes.</summary>
    [StructLayout(LayoutKind.Explicit, Size = 32)]
    internal struct QueueDesc
    {
        [FieldOffset(0)] public ulong SourceType;     // DSTORAGE_REQUEST_SOURCE_FILE = 0
        [FieldOffset(8)] public ushort Capacity;
        [FieldOffset(10)] public sbyte Priority;      // NORMAL = 0
        [FieldOffset(16)] public nint Name;
        [FieldOffset(24)] public nint Device;
    }

    internal const ushort MaxQueueCapacity = 0x2000;

    /// <summary>DSTORAGE_REQUEST, 104 bytes: file source -> buffer destination, uncompressed.</summary>
    [StructLayout(LayoutKind.Explicit, Size = 104)]
    internal struct Request
    {
        // DSTORAGE_REQUEST_OPTIONS (16 bytes): byte 0 = compression format (NONE = 0), then a u64 whose bit 0 is the source
        // type (FILE = 0) and bits 1..7 the destination type (BUFFER = 1) => 0x2.
        [FieldOffset(0)] public ulong OptionsWord0;
        [FieldOffset(8)] public ulong OptionsWord1;
        // DSTORAGE_SOURCE_FILE
        [FieldOffset(16)] public nint SourceFile;
        [FieldOffset(24)] public ulong SourceOffset;
        [FieldOffset(32)] public uint SourceSize;
        // DSTORAGE_DESTINATION_BUFFER
        [FieldOffset(40)] public nint DestResource;
        [FieldOffset(48)] public ulong DestOffset;
        [FieldOffset(56)] public uint DestSize;
        [FieldOffset(80)] public uint UncompressedSize;
        [FieldOffset(88)] public ulong CancellationTag;
        [FieldOffset(96)] public nint Name;
    }

    internal const int ErrorRecordSize = 704;
    internal const int ErrorRecordFailureCountOffset = 0;
    internal const int ErrorRecordHResultOffset = 8;

    // IDStorageFactory: [3] CreateQueue, [4] OpenFile, [5] CreateStatusArray, [6] SetDebugFlags, [7] SetStagingBufferSize
    internal static int CreateQueue(nint factory, in QueueDesc desc, out nint queue)
    {
        nint q = 0;
        fixed (QueueDesc* d = &desc)
        fixed (Guid* iid = &IidQueue)
        {
            int hr = ((delegate* unmanaged[Stdcall]<nint, QueueDesc*, Guid*, nint*, int>)ComVtbl.Slot(factory, 3))(factory, d, iid, &q);
            queue = q;
            return hr;
        }
    }

    internal static int OpenFile(nint factory, string path, out nint file)
    {
        nint f = 0;
        fixed (char* p = path)
        fixed (Guid* iid = &IidFile)
        {
            int hr = ((delegate* unmanaged[Stdcall]<nint, char*, Guid*, nint*, int>)ComVtbl.Slot(factory, 4))(factory, p, iid, &f);
            file = f;
            return hr;
        }
    }

    internal static int SetStagingBufferSize(nint factory, uint bytes)
        => ((delegate* unmanaged[Stdcall]<nint, uint, int>)ComVtbl.Slot(factory, 7))(factory, bytes);

    // IDStorageFile: [3] Close (void)
    internal static void CloseFile(nint file)
    {
        if (file == 0) return;
        ((delegate* unmanaged[Stdcall]<nint, void>)ComVtbl.Slot(file, 3))(file);
        ComVtbl.Release(file);
    }

    // IDStorageQueue: [3] EnqueueRequest, [4] EnqueueStatus, [5] EnqueueSignal, [6] Submit, [7] CancelRequestsWithTag, [8] Close,
    // [9] GetErrorEvent, [10] RetrieveErrorRecord, [11] Query
    internal static void EnqueueRequest(nint queue, in Request request)
    {
        fixed (Request* r = &request)
            ((delegate* unmanaged[Stdcall]<nint, Request*, void>)ComVtbl.Slot(queue, 3))(queue, r);
    }

    internal static void EnqueueSignal(nint queue, nint fence, ulong value)
        => ((delegate* unmanaged[Stdcall]<nint, nint, ulong, void>)ComVtbl.Slot(queue, 5))(queue, fence, value);

    internal static void Submit(nint queue)
        => ((delegate* unmanaged[Stdcall]<nint, void>)ComVtbl.Slot(queue, 6))(queue);

    internal static void CloseQueue(nint queue)
    {
        if (queue == 0) return;
        ((delegate* unmanaged[Stdcall]<nint, void>)ComVtbl.Slot(queue, 8))(queue);
        ComVtbl.Release(queue);
    }

    /// <summary>Fills <paramref name="record"/> (<see cref="ErrorRecordSize"/> bytes) with the queue's first failure, if any.</summary>
    internal static void RetrieveErrorRecord(nint queue, byte* record)
        => ((delegate* unmanaged[Stdcall]<nint, byte*, void>)ComVtbl.Slot(queue, 10))(queue, record);
}
