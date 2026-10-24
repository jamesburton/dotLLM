using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Queries AMD GPU device properties via the HIP runtime API.
/// Mirrors <c>DotLLM.Cuda.CudaDevice</c>.
/// </summary>
public sealed class HipDevice
{
    /// <summary>Device ordinal (0-based).</summary>
    public int Ordinal { get; }

    /// <summary>Device name (e.g., "AMD Radeon RX 7900 XTX").</summary>
    public string Name { get; }

    /// <summary>Total device memory in bytes.</summary>
    public long TotalMemoryBytes { get; }

    /// <summary>Number of compute units (analogous to CUDA SM count).</summary>
    public int ComputeUnitCount { get; }

    /// <summary>Warp (wavefront) size reported by the runtime.</summary>
    public int WarpSize { get; }

    private HipDevice(int ordinal, string name, long totalMem, int cuCount, int warpSize)
    {
        Ordinal = ordinal;
        Name = name;
        TotalMemoryBytes = totalMem;
        ComputeUnitCount = cuCount;
        WarpSize = warpSize;
    }

    /// <summary>Total device memory formatted as a human-readable string (e.g., "24.0 GB").</summary>
    public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB";

    /// <summary>
    /// Checks whether HIP / ROCm is available on this system (runtime installed, at least one GPU).
    /// Does not throw; returns false if HIP is unavailable.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            // Probe for the HIP runtime library before any P/Invoke.
            // This must happen BEFORE any reference to HipDriverApi, because
            // the JIT resolves [LibraryImport] P/Invoke stubs when compiling
            // a method — triggering DllNotFoundException on systems without ROCm.
            string hipLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "amdhip64.dll" : "libamdhip64.so";
            if (!NativeLibrary.TryLoad(hipLib, out nint handle))
            {
                // Try Linux versioned fallbacks.
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    foreach (var n in new[] { "libamdhip64.so.6", "libamdhip64.so.5" })
                        if (NativeLibrary.TryLoad(n, out handle))
                            goto loaded;
                }
                return false;
            }
        loaded:
            NativeLibrary.Free(handle);
            return ProbeGpuCount();
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Isolated in <see cref="MethodImplOptions.NoInlining"/> so the JIT only resolves
    /// HipDriverApi P/Invoke stubs when this method is actually called — after the
    /// NativeLibrary.TryLoad probe confirms libamdhip64 is present.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool ProbeGpuCount()
    {
        HipLibraryResolver.Register();
        HipDriverApi.hipInit(0).ThrowOnError();
        HipDriverApi.hipGetDeviceCount(out int count).ThrowOnError();
        return count > 0;
    }

    /// <summary>Returns the number of HIP-capable GPUs.</summary>
    public static int GetDeviceCount()
    {
        HipLibraryResolver.Register();
        HipDriverApi.hipInit(0).ThrowOnError();
        HipDriverApi.hipGetDeviceCount(out int count).ThrowOnError();
        return count;
    }

    /// <summary>Queries device properties for the given ordinal.</summary>
    public static HipDevice GetDevice(int ordinal)
    {
        HipLibraryResolver.Register();
        HipDriverApi.hipInit(0).ThrowOnError();

        HipDriverApi.hipDeviceGet(out int device, ordinal).ThrowOnError();

        // Name
        byte[] nameBuffer = new byte[256];
        HipDriverApi.hipDeviceGetName(nameBuffer, nameBuffer.Length, device).ThrowOnError();
        int nullIdx = Array.IndexOf(nameBuffer, (byte)0);
        string name = Encoding.ASCII.GetString(nameBuffer, 0, nullIdx >= 0 ? nullIdx : nameBuffer.Length).Trim();

        // Total memory
        HipDriverApi.hipDeviceTotalMem(out nuint totalMem, device).ThrowOnError();

        // Compute units
        HipDriverApi.hipDeviceGetAttribute(out int cuCount,
            HipDriverApi.HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).ThrowOnError();

        // Warp size
        HipDriverApi.hipDeviceGetAttribute(out int warpSize,
            HipDriverApi.HIP_DEVICE_ATTRIBUTE_WARP_SIZE, device).ThrowOnError();

        return new HipDevice(ordinal, name, (long)totalMem, cuCount, warpSize);
    }

    /// <summary>Uploads host memory to device (blocking).</summary>
    public static void Upload(nint devicePtr, nint hostPtr, long byteCount)
    {
        HipDriverApi.hipMemcpyHtoD(devicePtr, hostPtr, (nuint)byteCount).ThrowOnError();
    }

    /// <summary>Downloads device memory to host (blocking).</summary>
    public static void Download(nint hostPtr, nint devicePtr, long byteCount)
    {
        HipDriverApi.hipMemcpyDtoH(hostPtr, devicePtr, (nuint)byteCount).ThrowOnError();
    }

    /// <summary>Allocates device memory of the given byte size.</summary>
    public static nint Allocate(long byteCount)
    {
        HipDriverApi.hipMalloc(out nint dptr, (nuint)byteCount).ThrowOnError();
        return dptr;
    }

    /// <summary>Frees device memory.</summary>
    public static void Free(nint devicePtr)
    {
        if (devicePtr != 0)
            HipDriverApi.hipFree(devicePtr).ThrowOnError();
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"GPU {Ordinal}: {Name} ({TotalMemoryFormatted}, {ComputeUnitCount} CUs, wave={WarpSize})";
}
