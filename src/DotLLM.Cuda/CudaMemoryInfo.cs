using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Device-wide free / total memory of a CUDA device, read through NVML (#1023).
/// </summary>
/// <remarks>
/// <para>
/// <c>cuMemGetInfo</c> needs a current context, and creating one just to ask costs hundreds of MiB of VRAM and a noticeable delay - on
/// the very device the caller is trying to fit a model onto. NVML answers without a context and sees every process's usage. It ships with
/// the driver (<c>nvml.dll</c> / <c>libnvidia-ml.so.1</c>) and is bound late through <see cref="NativeLibrary"/>, so a machine without it
/// simply reports "unknown".
/// </para>
/// <para>
/// CUDA ordinals (fastest first) and NVML indices (PCI order) can differ, so the device is matched by PCI bus id.
/// </para>
/// </remarks>
public static unsafe class CudaMemoryInfo
{
    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemory
    {
        public ulong Total;
        public ulong Free;
        public ulong Used;
    }

    private static readonly object Gate = new();
    private static bool _initTried;
    private static delegate* unmanaged<int> _nvmlInit;
    private static delegate* unmanaged<byte*, nint*, int> _getHandleByPciBusId;
    private static delegate* unmanaged<nint, NvmlMemory*, int> _getMemoryInfo;

    /// <summary>
    /// Reads the free and total memory of CUDA device <paramref name="ordinal"/>. Never throws.
    /// </summary>
    /// <param name="ordinal">CUDA device ordinal.</param>
    /// <param name="freeBytes">Device-wide free memory (all processes), when known.</param>
    /// <param name="totalBytes">Total device memory, when known.</param>
    /// <returns>True when NVML answered; false when it is unavailable or the device could not be matched.</returns>
    public static bool TryGet(int ordinal, out long freeBytes, out long totalBytes)
    {
        freeBytes = 0;
        totalBytes = 0;
        try
        {
            if (!EnsureNvml()) return false;
            if (!CudaDevice.IsAvailable()) return false;
            string? busId = PciBusId(ordinal);
            if (busId is null) return false;

            byte[] id = Encoding.ASCII.GetBytes(busId + "\0");
            nint handle;
            NvmlMemory mem;
            fixed (byte* p = id)
            {
                if (_getHandleByPciBusId(p, &handle) != 0) return false;
            }
            if (_getMemoryInfo(handle, &mem) != 0) return false;
            freeBytes = DotLLM.Core.Configuration.DeviceMemoryBudget.ClampSimulatedFree(ordinal, (long)mem.Free);   // #1295
            totalBytes = DotLLM.Core.Configuration.DeviceMemoryBudget.ClampSimulatedCapacity((long)mem.Total);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string? PciBusId(int ordinal)
    {
        CudaDriverApi.cuDeviceGet(out int device, ordinal).ThrowOnError();
        byte[] buffer = new byte[64];
        if (CudaDriverApi.cuDeviceGetPCIBusId(buffer, buffer.Length, device) != 0) return null;
        int end = Array.IndexOf(buffer, (byte)0);
        return Encoding.ASCII.GetString(buffer, 0, end >= 0 ? end : buffer.Length);
    }

    private static bool EnsureNvml()
    {
        lock (Gate)
        {
            if (_initTried) return _getMemoryInfo != null;
            _initTried = true;

            string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvml.dll" : "libnvidia-ml.so.1";
            if (!NativeLibrary.TryLoad(lib, out nint h)
                && !(RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                     && NativeLibrary.TryLoad(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                         "NVIDIA Corporation", "NVSMI", "nvml.dll"), out h)))
                return false;

            if (!NativeLibrary.TryGetExport(h, "nvmlInit_v2", out nint init)
                || !NativeLibrary.TryGetExport(h, "nvmlDeviceGetHandleByPciBusId_v2", out nint byBus)
                || !NativeLibrary.TryGetExport(h, "nvmlDeviceGetMemoryInfo", out nint memInfo))
                return false;

            _nvmlInit = (delegate* unmanaged<int>)init;
            if (_nvmlInit() != 0) return false;   // NVML_SUCCESS = 0; init is reference-counted and process-wide
            _getHandleByPciBusId = (delegate* unmanaged<byte*, nint*, int>)byBus;
            _getMemoryInfo = (delegate* unmanaged<nint, NvmlMemory*, int>)memInfo;
            return true;
        }
    }
}
