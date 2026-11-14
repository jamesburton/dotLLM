using System.Runtime.InteropServices;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip.DirectStorage;

/// <summary>
/// Loads file ranges through Windows DirectStorage into D3D12 buffers imported into HIP.
/// No application host staging or host-copy fallback is used. Returned buffers outlive the store.
/// </summary>
public sealed unsafe class HipDirectWeightStore : IDisposable
{
    private const long HeapAlignment = 64 * 1024;
    private const uint RequestBytes = 32 * 1024 * 1024;
    private readonly HipContext _context;
    private readonly object _gate = new();
    private nint _device, _factory, _queue, _fence, _event, _module, _coreModule;
    private ulong _fenceValue;
    private bool _disposed;

    /// <summary>Creates a DirectStorage session on the context's exact Windows x64 adapter.</summary>
    public HipDirectWeightStore(HipContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
            throw new PlatformNotSupportedException("HIP direct weight loading currently requires Windows x64 and DirectStorage.");
        _context = context;
        try
        {
            context.MakeCurrent();
            HipExternalMemoryApi.GetDeviceProperties(out var properties, context.DeviceId).ThrowOnError();
            CheckHr(D3dNative.CreateDXGIFactory1(D3dNative.IidDxgiFactory4, out nint dxgi), "CreateDXGIFactory1");
            try
            {
                CheckHr(D3dNative.EnumAdapterByLuid(dxgi, properties.Luid, out nint adapter), "EnumAdapterByLuid");
                try
                {
                    CheckHr(D3dNative.D3D12CreateDevice(adapter, D3dNative.D3D_FEATURE_LEVEL_11_0,
                        D3dNative.IidD3D12Device, out _device), "D3D12CreateDevice");
                }
                finally { ComVtbl.Release(adapter); }
            }
            finally { ComVtbl.Release(dxgi); }

            LoadDirectStorage();
            var getFactory = (delegate* unmanaged[Stdcall]<Guid*, nint*, int>)NativeLibrary.GetExport(_module, "DStorageGetFactory");
            Guid iid = DStorageNative.IidFactory;
            nint factory = 0;
            CheckHr(getFactory(&iid, &factory), "DStorageGetFactory");
            _factory = factory;
            // The factory is shared with other backends; keep its staging configuration unchanged.
            var queue = new DStorageNative.QueueDesc { Capacity = 128, Device = _device };
            CheckHr(DStorageNative.CreateQueue(_factory, queue, out _queue), "CreateQueue");
            CheckHr(D3dNative.CreateFence(_device, out _fence), "CreateFence");
            _event = D3dNative.CreateEvent(0, 0, 0, 0);
            if (_event == 0) throw new IOException("Creating the HIP DirectStorage completion event failed.");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Loads exactly one file range, retaining its original packed or narrow representation.</summary>
    public HipDirectWeightBuffer Load(string path, long offset, long byteLength)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _context.MakeCurrent();
            string fullPath = Path.GetFullPath(path);
            // Hold the same file against writes/deletion until its queued read completes.
            using var fileGuard = File.OpenHandle(fullPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            ValidateRange(RandomAccess.GetLength(fileGuard), offset, byteLength);
            CheckHr(DStorageNative.OpenFile(_factory, fullPath, out nint file), "OpenFile");
            HipDirectWeightBuffer? buffer = null;
            try
            {
                buffer = Allocate(byteLength);
                long remaining = byteLength, destinationOffset = 0;
                int pending = 0;
                while (remaining != 0)
                {
                    uint count = (uint)Math.Min(remaining, RequestBytes);
                    var request = new DStorageNative.Request
                    {
                        OptionsWord1 = 2,
                        SourceFile = file,
                        SourceOffset = (ulong)(offset + destinationOffset),
                        SourceSize = count,
                        DestResource = buffer.Resource,
                        DestOffset = (ulong)destinationOffset,
                        DestSize = count,
                        UncompressedSize = count,
                    };
                    DStorageNative.EnqueueRequest(_queue, request);
                    remaining -= count;
                    destinationOffset += count;
                    if (++pending == 64) { WaitAll(); pending = 0; }
                }
                if (pending != 0) WaitAll();
                return buffer;
            }
            catch
            {
                buffer?.Dispose();
                throw;
            }
            finally { DStorageNative.CloseFile(file); }
        }
    }

    internal static void ValidateRange(long fileLength, long offset, long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        if (offset > fileLength || byteLength > fileLength - offset)
            throw new ArgumentOutOfRangeException(nameof(byteLength), "The requested tensor range extends past the end of the file.");
    }

    private HipDirectWeightBuffer Allocate(long byteLength)
    {
        long allocated = checked(byteLength + HeapAlignment - 1) & -HeapAlignment;
        HipDriverApi.hipMemGetInfo(out nuint free, out _).ThrowOnError();
        if ((ulong)allocated > (ulong)free)
            throw new OutOfMemoryException($"HIP direct weight loading needs {allocated} device bytes; only {free} are free.");
        var heapDesc = new D3dNative.HeapDesc
        {
            SizeInBytes = (ulong)allocated, Alignment = (ulong)HeapAlignment,
            Type = D3dNative.HeapTypeDefault,
            CreationNodeMask = 1, VisibleNodeMask = 1,
            Flags = D3dNative.HeapFlagShared | D3dNative.HeapFlagAllowOnlyBuffers,
        };
        nint heap = 0, resource = 0, handle = 0, imported = 0, pointer = 0;
        try
        {
            CheckHr(D3dNative.CreateHeap(_device, heapDesc, out heap), "CreateHeap");
            CheckHr(D3dNative.CreateBufferResource(_device, heap, 0, (ulong)allocated, out resource), "CreatePlacedResource");
            CheckHr(D3dNative.CreateSharedHandle(_device, heap, out handle), "CreateSharedHandle");
            var memory = new HipExternalMemoryApi.MemoryHandleDesc
            {
                Type = HipExternalMemoryApi.D3D12Heap, Handle = handle, Size = (ulong)allocated,
            };
            HipExternalMemoryApi.ImportExternalMemory(out imported, memory).ThrowOnError();
            var map = new HipExternalMemoryApi.BufferDesc { Size = (ulong)allocated };
            HipExternalMemoryApi.GetMappedBuffer(out pointer, imported, map).ThrowOnError();
            return new HipDirectWeightBuffer(_context.DeviceId, pointer, imported, resource, heap, byteLength, allocated);
        }
        catch
        {
            if (pointer != 0) HipDriverApi.hipFree(pointer);
            if (imported != 0) HipExternalMemoryApi.DestroyExternalMemory(imported);
            ComVtbl.Release(resource);
            ComVtbl.Release(heap);
            throw;
        }
        finally { if (handle != 0) D3dNative.CloseHandle(handle); }
    }

    private void WaitAll()
    {
        ulong target = ++_fenceValue;
        DStorageNative.EnqueueSignal(_queue, _fence, target);
        DStorageNative.Submit(_queue);
        if (D3dNative.FenceGetCompletedValue(_fence) < target)
        {
            CheckHr(D3dNative.FenceSetEventOnCompletion(_fence, target, _event), "SetEventOnCompletion");
            uint wait = D3dNative.WaitForSingleObject(_event, uint.MaxValue);
            if (wait != 0) throw new IOException($"HIP DirectStorage completion wait failed (0x{wait:X8}).");
        }
        if (D3dNative.FenceGetCompletedValue(_fence) == ulong.MaxValue)
            throw new IOException("The D3D12 device was removed while loading HIP weights.");
        byte* record = stackalloc byte[DStorageNative.ErrorRecordSize];
        new Span<byte>(record, DStorageNative.ErrorRecordSize).Clear();
        DStorageNative.RetrieveErrorRecord(_queue, record);
        uint failures = *(uint*)(record + DStorageNative.ErrorRecordFailureCountOffset);
        if (failures != 0)
        {
            int hr = *(int*)(record + DStorageNative.ErrorRecordHResultOffset);
            throw new IOException($"DirectStorage failed {failures} HIP weight read(s) (0x{hr:X8}); weights are incomplete.");
        }
    }

    private void LoadDirectStorage()
    {
        string installed = Environment.GetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "native", "directstorage", "1.3.0", "x64");
        foreach (string directory in new[] { AppContext.BaseDirectory, installed })
        {
            string path = Path.Combine(directory, "dstorage.dll");
            string core = Path.Combine(directory, "dstoragecore.dll");
            if (!File.Exists(path) || !File.Exists(core)) continue;
            if (!NativeLibrary.TryLoad(core, out _coreModule)) continue;
            if (NativeLibrary.TryLoad(path, out _module)) return;
            NativeLibrary.Free(_coreModule);
            _coreModule = 0;
        }
        if (NativeLibrary.TryLoad("dstorage.dll", out _module)) return;
        throw new DllNotFoundException("HIP direct loading requires Microsoft's dstorage.dll and dstoragecore.dll. Install DirectStorage through dotLLM or place both DLLs next to the executable; host-copy fallback is disabled.");
    }

    private static void CheckHr(int hr, string operation)
    {
        if (hr < 0) throw new IOException($"HIP DirectStorage {operation} failed (0x{hr:X8}).");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            DStorageNative.CloseQueue(_queue);
            ComVtbl.Release(_factory);
            ComVtbl.Release(_fence);
            if (_event != 0) D3dNative.CloseHandle(_event);
            ComVtbl.Release(_device);
            if (_module != 0) NativeLibrary.Free(_module);
            if (_coreModule != 0) NativeLibrary.Free(_coreModule);
        }
    }
}
