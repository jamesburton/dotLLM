using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// M3 feasibility probe: checks whether the Vulkan iGPU (Intel Arc) and the CUDA
/// GPU (RTX 3060) both expose the Vulkan timeline + Win32-export + CUDA import
/// APIs required for async cross-device semaphore pipelining.
/// </summary>
/// <remarks>
/// <para>
/// The M3 plan requires an exportable Vulkan timeline semaphore on the Vulkan
/// device, imported into CUDA via <c>cuImportExternalSemaphore</c> with handle
/// type <c>CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32</c>.
/// </para>
/// <para>
/// Required Vulkan extensions on the iGPU:
/// <list type="bullet">
///   <item><c>VK_KHR_timeline_semaphore</c> — timeline counter semantics (core in 1.2).</item>
///   <item><c>VK_KHR_external_semaphore</c> — export-handle infrastructure (core in 1.1).</item>
///   <item><c>VK_KHR_external_semaphore_win32</c> — Win32 HANDLE export for the semaphore.</item>
/// </list>
/// Required CUDA API: <c>cuImportExternalSemaphore</c> (CUDA 10.0+, driver API).
/// </para>
/// <para>
/// This test reports the raw probe findings via xUnit output. It does not attempt
/// to create semaphores or perform interop — that is the M3 implementation task.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class HybridVulkanCudaM3SemaphoreProbeTests
{
    private readonly ITestOutputHelper _out;
    public HybridVulkanCudaM3SemaphoreProbeTests(ITestOutputHelper output) => _out = output;

    // CUDA Driver API ordinals for the external-semaphore functions.
    // cuImportExternalSemaphore is ordinal 349 in the CUDA 10.x+ driver.
    // We resolve it at runtime via NativeLibrary to avoid a hard link.

    private static bool IsBothAvailable()
        => VulkanDevice.IsAvailable() && IsCudaDriverPresent();

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows()
            ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    /// <summary>
    /// Reports which Vulkan extensions required for M3 are present on the iGPU
    /// and whether the CUDA driver exposes <c>cuImportExternalSemaphore</c>.
    /// </summary>
    [SkippableFact]
    public void M3_Probe_VulkanTimelineSemaphoreExportAndCudaImport()
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");

        _out.WriteLine("=== M3 semaphore interop feasibility probe ===");

        // ── Vulkan extension probe ──────────────────────────────────────────

        // Use reflection to call HasDeviceExtension via the VulkanDevice.
        // VulkanDevice doesn't expose a public HasExtension API, so we read
        // the SupportedExtensions list that IS exposed via VulkanDevice.Extensions
        // if it exists, or probe indirectly by attempting to create a semaphore.
        // As a pragmatic alternative, use the public VulkanDevice.Create() and
        // check the device feature set.

        // Probe by attempting to resolve vkGetSemaphoreWin32HandleKHR via
        // vkGetDeviceProcAddr — if the function pointer is non-null the extension
        // is present and the driver implements it.
        bool timelineExtPresent = false;
        bool externalSemExtPresent = false;
        bool win32SemExtPresent = false;

        using (var device = VulkanDevice.Create())
        {
            // We access the Vulkan device handle and instance handle via the
            // known internal fields to call vkGetDeviceProcAddr directly.
            // Use the DeviceHandle property if it is exposed, or reflection.
            nint vkDevice = GetVkDevice(device);
            nint vkInstance = GetVkInstance(device);

            if (vkDevice == 0 || vkInstance == 0)
            {
                _out.WriteLine("  [WARN] Cannot obtain raw Vulkan handles — skipping extension probe.");
            }
            else
            {
                // vkGetDeviceProcAddr: non-null return = extension function present.
                nint fnTimeline = VulkanApi.vkGetDeviceProcAddr(vkDevice,
                    "vkGetSemaphoreCounterValueKHR");
                timelineExtPresent = fnTimeline != 0;

                nint fnExport = VulkanApi.vkGetDeviceProcAddr(vkDevice,
                    "vkGetSemaphoreWin32HandleKHR");
                win32SemExtPresent = fnExport != 0;

                // VK_KHR_external_semaphore is core in 1.1; probe via a core 1.1 fn.
                nint fnExtSem = VulkanApi.vkGetDeviceProcAddr(vkDevice,
                    "vkImportSemaphoreWin32HandleKHR");
                externalSemExtPresent = fnExtSem != 0 || win32SemExtPresent;
            }
        }

        _out.WriteLine($"  VK_KHR_timeline_semaphore (vkGetSemaphoreCounterValueKHR): {timelineExtPresent}");
        _out.WriteLine($"  VK_KHR_external_semaphore present: {externalSemExtPresent}");
        _out.WriteLine($"  VK_KHR_external_semaphore_win32 (vkGetSemaphoreWin32HandleKHR): {win32SemExtPresent}");

        // ── CUDA external semaphore API probe ──────────────────────────────

        bool cudaImportExtSemPresent = false;
        string cudaLib = OperatingSystem.IsWindows()
            ? "nvcuda.dll" : "libcuda.so.1";
        if (NativeLibrary.TryLoad(cudaLib, out nint cudaHandle))
        {
            try
            {
                cudaImportExtSemPresent =
                    NativeLibrary.TryGetExport(cudaHandle, "cuImportExternalSemaphore", out _);
            }
            finally
            {
                NativeLibrary.Free(cudaHandle);
            }
        }
        _out.WriteLine($"  CUDA cuImportExternalSemaphore present: {cudaImportExtSemPresent}");

        // ── Summary ────────────────────────────────────────────────────────

        bool allRequired = timelineExtPresent && win32SemExtPresent && cudaImportExtSemPresent;
        _out.WriteLine($"");
        _out.WriteLine($"  M3 feasibility: {(allRequired ? "ALL PREREQUISITES MET" : "BLOCKED")}");

        if (!timelineExtPresent)
            _out.WriteLine("  BLOCKED: VK_KHR_timeline_semaphore not exposed on this Vulkan device.");
        if (!win32SemExtPresent)
            _out.WriteLine("  BLOCKED: VK_KHR_external_semaphore_win32 not exposed on this Vulkan device.");
        if (!cudaImportExtSemPresent)
            _out.WriteLine("  BLOCKED: cuImportExternalSemaphore not found in nvcuda.dll.");

        _out.WriteLine("=== End M3 probe ===");

        // Always passes — this is a diagnostic probe, not a requirement test.
        // The pass/fail verdict is in the printed output.
        Assert.True(true);
    }

    // ── Vulkan handle accessors ────────────────────────────────────────────

    // VulkanDevice exposes device/instance handles as internal properties.
    // Use reflection to read them non-invasively for this one-time probe test.
    private static nint GetVkDevice(VulkanDevice device)
    {
        try
        {
            var prop = typeof(VulkanDevice).GetProperty("DeviceHandle",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.Instance);
            if (prop is not null) return (nint)(prop.GetValue(device) ?? (nint)0);

            var field = typeof(VulkanDevice).GetField("_device",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field is not null) return (nint)(field.GetValue(device) ?? (nint)0);
        }
        catch { }
        return 0;
    }

    private static nint GetVkInstance(VulkanDevice device)
    {
        try
        {
            var prop = typeof(VulkanDevice).GetProperty("InstanceHandle",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.Instance);
            if (prop is not null) return (nint)(prop.GetValue(device) ?? (nint)0);

            var field = typeof(VulkanDevice).GetField("_instance",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field is not null) return (nint)(field.GetValue(device) ?? (nint)0);
        }
        catch { }
        return 0;
    }
}
