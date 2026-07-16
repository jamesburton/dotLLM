using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Represents a Vulkan logical device bound to a single physical GPU plus a
/// compute queue and command pool. Owns the instance, device, and allocator
/// state; disposal tears everything down in reverse order.
/// </summary>
/// <remarks>
/// Scaffold semantics — proof-of-pipeline only:
/// <list type="bullet">
///   <item>No fence-based pipelining. Submits are synchronous (<c>vkQueueWaitIdle</c>).</item>
///   <item>No staging buffers. Device memory is allocated <c>HostVisible|HostCoherent</c>
///     so uploads/downloads hit the same VRAM region — fine for small tests, not for
///     large model weights. A proper arena + staging ring lands with the first real kernel.</item>
///   <item>Single queue. Multi-queue (transfer/compute separation) is deferred.</item>
/// </list>
/// </remarks>
public sealed class VulkanDevice : IDisposable
{
    private nint _instance;
    private nint _physicalDevice;
    private nint _device;
    private nint _queue;
    private nint _commandPool;
    private bool _disposed;

    /// <summary>Device name (e.g. "AMD Radeon RX 7900 XT", "NVIDIA GeForce RTX 4090").</summary>
    public string DeviceName { get; }

    /// <summary>PCI vendor ID (0x10DE = NVIDIA, 0x1002 = AMD, 0x8086 = Intel).</summary>
    public uint VendorId { get; }

    /// <summary>Vulkan device type (discrete, integrated, virtual, CPU).</summary>
    public int DeviceType { get; }

    /// <summary>Queue family index selected for compute.</summary>
    public uint QueueFamilyIndex { get; }

    /// <summary>
    /// Hardware subgroup width reported by the driver — e.g. 32 on NVIDIA /
    /// Intel, 64 on AMD GCN / RDNA3.5 iGPU. Zero when the probe could not
    /// run (Vulkan 1.0 driver, loader missing <c>vkGetPhysicalDeviceProperties2</c>).
    /// Exposed so kernel code can size cross-subgroup scratch without guessing.
    /// </summary>
    public uint SubgroupSize { get; }

    /// <summary>
    /// True when the physical device advertises
    /// <c>VK_SUBGROUP_FEATURE_ARITHMETIC_BIT</c> AND the compute stage is in
    /// <c>supportedStages</c>. Kernels use this to pick the
    /// <c>subgroupAdd</c> / <c>subgroupMax</c> fast path over shared-memory
    /// tree reductions. Falls back to <c>false</c> on any device that does not
    /// report Vulkan 1.1 core subgroup properties.
    /// </summary>
    public bool HasSubgroupArithmetic { get; }

    /// <summary>
    /// True when the physical device advertises <c>VK_KHR_cooperative_matrix</c>,
    /// the <c>cooperativeMatrix</c> feature bit is set, and the driver reports
    /// at least one F16×F16→F32 (or Sint8×Sint8→Sint32) tile shape ≥ 16×16×16 at
    /// subgroup scope. This is the prerequisite for the coopmat GEMM path
    /// (<c>MatMulQ8_0GemmCoopmatKernel</c>); scalar GEMM remains the safe
    /// fallback when <c>false</c>.
    /// </summary>
    public bool HasCooperativeMatrix { get; }

    /// <summary>
    /// All cooperative-matrix tile shapes reported by the driver, in the
    /// order returned by <c>vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR</c>.
    /// Empty when <see cref="HasCooperativeMatrix"/> is <c>false</c>. Kernels
    /// typically pick a 16×16×16 F16/F16/F32 entry at shader-compile time.
    /// </summary>
    public IReadOnlyList<CooperativeMatrixShape> SupportedCooperativeMatrixProperties { get; }

    /// <summary>
    /// True when the physical device advertises <c>VK_EXT_external_memory_host</c>
    /// AND the device-create call successfully enabled the extension. When set,
    /// callers may import a page-aligned host pointer (e.g. the mmap'd GGUF
    /// tensor data section) directly into a <c>VkDeviceMemory</c> via
    /// <c>HostVisibleBuffer.TryCreate</c>, eliminating the host→device staging
    /// copy on unified-memory APUs (Strix Halo, Intel iGPU, MoltenVK).
    /// </summary>
    public bool HasExternalMemoryHost { get; }

    /// <summary>
    /// Minimum alignment (in bytes) a host pointer must satisfy to be
    /// importable via <c>VK_EXT_external_memory_host</c>. Driver-reported
    /// (typically 4096 on x86-64 — page size). Zero when
    /// <see cref="HasExternalMemoryHost"/> is <c>false</c>. Callers can round
    /// the candidate pointer down to a multiple of this value and use the
    /// resulting offset as the <c>VkBuffer</c> view offset.
    /// </summary>
    public ulong MinImportedHostPointerAlignment { get; }

    /// <summary>
    /// True when the physical device advertises <c>VK_KHR_shader_integer_dot_product</c>
    /// (or Vulkan 1.3 core), the <c>shaderIntegerDotProduct</c> feature bit is
    /// set, AND the device-create call enabled it. Prerequisite for the dp4a
    /// MMVQ decode path (<c>MatMulQ8_0MmvqKernel</c>, which uses
    /// <c>dotPacked4x8AccSatEXT</c> / SPIR-V <c>DotProductInput4x8BitPackedKHR</c>).
    /// gfx1151 (Strix Halo, RDNA 3.5) reports this true. When <c>false</c> the
    /// router falls back to the F32-in Q8_0 GEMV.
    /// </summary>
    public bool HasIntegerDotProduct { get; }

    /// <summary>
    /// True when the physical device advertises <c>VK_EXT_subgroup_size_control</c>
    /// (or Vulkan 1.3 core), reports a valid <c>minSubgroupSize</c>..<c>maxSubgroupSize</c>
    /// range with the compute stage in <c>requiredSubgroupSizeStages</c>, AND the
    /// device-create call enabled the <c>subgroupSizeControl</c> +
    /// <c>computeFullSubgroups</c> features. Prerequisite for pinning a single
    /// compute pipeline to a specific wave width via
    /// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c>. On RDNA3.5
    /// (gfx1151) this lets the K-quant decode GEMV run wave32 while the driver's
    /// global compute default stays wave64 — see <see cref="SupportsRequiredSubgroupSize"/>.
    /// </summary>
    /// <remarks>
    /// Support is keyed off the <c>VkPhysicalDeviceSubgroupSizeControlProperties</c>
    /// query (min/max + requiredSubgroupSizeStages), not the
    /// <c>VkPhysicalDeviceSubgroupSizeControlFeatures</c> bits: when the feature
    /// was promoted to Vulkan 1.3 core, some loader/driver combinations only
    /// populate the promoted feature struct for an instance created at
    /// apiVersion ≥ 1.3 (this backend's instance requests 1.2), whereas the
    /// EXT-keyed properties struct is populated reliably. The features are still
    /// enabled at device-create time; <c>vkCreateDevice</c> would fail if the
    /// driver could not honour them.
    /// </remarks>
    public bool HasSubgroupSizeControl { get; }

    /// <summary>
    /// Minimum subgroup width the device will accept as a required size
    /// (<c>minSubgroupSize</c> from <c>VkPhysicalDeviceSubgroupSizeControlProperties</c>).
    /// Zero when <see cref="HasSubgroupSizeControl"/> is <c>false</c>. gfx1151
    /// reports 32.
    /// </summary>
    public uint MinSubgroupSize { get; }

    /// <summary>
    /// Maximum subgroup width the device will accept as a required size
    /// (<c>maxSubgroupSize</c>). Zero when <see cref="HasSubgroupSizeControl"/>
    /// is <c>false</c>. gfx1151 reports 64.
    /// </summary>
    public uint MaxSubgroupSize { get; }

    // VkShaderStageFlags bitmask of stages that accept a required subgroup size
    // (requiredSubgroupSizeStages). Checked by SupportsRequiredSubgroupSize.
    private readonly uint _requiredSubgroupSizeStages;

    /// <summary>
    /// Returns <c>true</c> when the device can pin a pipeline of the given
    /// <paramref name="stage"/> to the exact wave width <paramref name="size"/>:
    /// the <c>subgroupSizeControl</c> feature is enabled, <paramref name="size"/>
    /// lies within <see cref="MinSubgroupSize"/>..<see cref="MaxSubgroupSize"/>,
    /// and <paramref name="stage"/> is in the driver's
    /// <c>requiredSubgroupSizeStages</c>. Pass <c>VkShaderStageFlags.Compute</c>
    /// (0x20) for the MMVQ decode kernels. Falls back to <c>false</c> (callers
    /// then use the unset/default subgroup size) on any device lacking the feature.
    /// </summary>
    public bool SupportsRequiredSubgroupSize(uint size, uint stage)
    {
        if (!HasSubgroupSizeControl) return false;
        if (size < MinSubgroupSize || size > MaxSubgroupSize) return false;
        return (_requiredSubgroupSizeStages & stage) != 0;
    }

    /// <summary>
    /// True when the physical device advertises both <c>VK_KHR_external_semaphore</c>
    /// and <c>VK_KHR_external_semaphore_win32</c> AND the device-create call enabled
    /// them. This is the prerequisite for exporting a <c>VkSemaphore</c> as a Win32
    /// HANDLE for cross-API synchronisation with CUDA (the M3 async handoff:
    /// <see cref="CreateExportableSemaphore"/> + <see cref="GetSemaphoreWin32Handle"/>).
    /// Falls back to <c>false</c> on non-Windows or on drivers that don't expose
    /// the extension; callers must keep the fence-serialized path available.
    /// </summary>
    public bool HasExternalSemaphoreWin32 { get; }

    internal nint Handle => _device;
    internal nint Queue => _queue;
    internal nint CommandPool => _commandPool;
    internal nint PhysicalDevice => _physicalDevice;

    /// <summary>
    /// Nanoseconds per timestamp tick (<c>VkPhysicalDeviceLimits.timestampPeriod</c>).
    /// Read lazily for the env-gated decode profiler (issue #143). Returns 0 when
    /// the reported value is implausible (&lt;=0 or &gt;10µs) — callers should then
    /// skip GPU timestamping.
    /// </summary>
    internal unsafe float TimestampPeriodNs
    {
        get
        {
            VulkanApi.vkGetPhysicalDeviceProperties(_physicalDevice, out var props);
            // The C# struct's byte tail starts at offset 292 (after the UUID),
            // but VkPhysicalDeviceLimits is 8-byte aligned in the C layout, so
            // it begins at 296 = tail+4. timestampPeriod is the float at limits
            // offset 424 (right after timestampComputeAndGraphics) → tail+428.
            float p = *(float*)(props.tail + 428);
            return p >= 0.01f && p < 10_000f ? p : 0f;
        }
    }

    private VulkanDevice(
        nint instance, nint physical, nint device, nint queue,
        nint commandPool, string name, uint vendor, int type, uint queueFamily,
        uint subgroupSize, bool hasSubgroupArithmetic,
        bool hasCooperativeMatrix, IReadOnlyList<CooperativeMatrixShape> coopmatShapes,
        bool hasExternalMemoryHost, ulong minImportedHostPointerAlignment,
        bool hasIntegerDotProduct,
        bool hasSubgroupSizeControl, uint minSubgroupSize, uint maxSubgroupSize,
        uint requiredSubgroupSizeStages,
        bool hasExternalSemaphoreWin32)
    {
        _instance = instance;
        _physicalDevice = physical;
        _device = device;
        _queue = queue;
        _commandPool = commandPool;
        DeviceName = name;
        VendorId = vendor;
        DeviceType = type;
        QueueFamilyIndex = queueFamily;
        SubgroupSize = subgroupSize;
        HasSubgroupArithmetic = hasSubgroupArithmetic;
        HasCooperativeMatrix = hasCooperativeMatrix;
        SupportedCooperativeMatrixProperties = coopmatShapes;
        HasExternalMemoryHost = hasExternalMemoryHost;
        MinImportedHostPointerAlignment = minImportedHostPointerAlignment;
        HasIntegerDotProduct = hasIntegerDotProduct;
        HasSubgroupSizeControl = hasSubgroupSizeControl;
        MinSubgroupSize = minSubgroupSize;
        MaxSubgroupSize = maxSubgroupSize;
        _requiredSubgroupSizeStages = requiredSubgroupSizeStages;
        HasExternalSemaphoreWin32 = hasExternalSemaphoreWin32;
    }

    /// <summary>
    /// Probes whether a Vulkan loader is present and whether <c>vkCreateInstance</c>
    /// succeeds on this machine. Does not throw.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "vulkan-1.dll"
                : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                    ? "libvulkan.dylib"
                    : "libvulkan.so.1";
            if (!NativeLibrary.TryLoad(lib, out nint handle))
                return false;
            NativeLibrary.Free(handle);

            return ProbeInstance();
        }
        catch
        {
            return false;
        }
    }

    // Isolated so the JIT only resolves VulkanApi P/Invokes when the loader is confirmed present.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool ProbeInstance()
    {
        VulkanLibraryResolver.Register();
        nint inst = CreateInstance();
        if (inst == 0) return false;
        try
        {
            uint count = 0;
            int r = VulkanApi.vkEnumeratePhysicalDevices(inst, ref count, null);
            return r >= 0 && count > 0;
        }
        finally
        {
            VulkanApi.vkDestroyInstance(inst, 0);
        }
    }

    /// <summary>
    /// Returns the number of Vulkan physical devices the loader can enumerate, or 0 when no loader /
    /// driver is present. Used to gate cross-device tests (a two-GPU KV handoff needs ≥ 2 — e.g. the
    /// Framework iGPU + RTX 3060 box). Does not throw.
    /// </summary>
    public static int PhysicalDeviceCount()
    {
        if (!IsAvailable()) return 0;
        try { return ProbePhysicalDeviceCount(); }
        catch { return 0; }
    }

    // Isolated so the JIT only resolves VulkanApi P/Invokes when the loader is confirmed present.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int ProbePhysicalDeviceCount()
    {
        VulkanLibraryResolver.Register();
        nint inst = CreateInstance();
        if (inst == 0) return 0;
        try
        {
            uint count = 0;
            int r = VulkanApi.vkEnumeratePhysicalDevices(inst, ref count, null);
            return r >= 0 ? (int)count : 0;
        }
        finally
        {
            VulkanApi.vkDestroyInstance(inst, 0);
        }
    }

    /// <summary>
    /// Creates a Vulkan device bound to the first suitable GPU.
    /// Selection order: discrete GPU (preferring AMD/NVIDIA over Intel) → integrated → first available.
    /// </summary>
    public static VulkanDevice Create() => CreateCore(forcedIndex: null);

    /// <summary>
    /// Creates a Vulkan device bound to the physical device at the given <c>vkEnumeratePhysicalDevices</c>
    /// enumeration index. Unlike the parameterless overload (which scores devices) and the process-global
    /// <c>DOTLLM_VULKAN_DEVICE_INDEX</c> env override, this lets a single process bind <em>different</em>
    /// GPUs for different replicas — e.g. prefill on device 0 and decode on device 1 for a cross-device
    /// <see cref="T:DotLLM.Engine.Scheduler.DisaggregatedScheduler"/> KV handoff. The explicit index takes
    /// precedence over the env overrides.
    /// </summary>
    /// <param name="deviceIndex">Zero-based physical-device enumeration index.</param>
    public static VulkanDevice Create(int deviceIndex)
    {
        if (deviceIndex < 0) throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        return CreateCore(deviceIndex);
    }

    private static VulkanDevice CreateCore(int? forcedIndex)
    {
        VulkanLibraryResolver.Register();
        nint instance = CreateInstance();
        if (instance == 0)
            throw new VulkanException(-3, "vkCreateInstance failed — no Vulkan loader or driver available.");

        try
        {
            nint physical = SelectPhysicalDevice(instance, forcedIndex, out string name, out uint vendor, out int type, out uint apiVersion);
            uint queueFamily = SelectComputeQueueFamily(physical);

            // Probe Vulkan 1.1 subgroup properties. Skipped gracefully on
            // Vulkan 1.0 drivers — SubgroupSize=0, HasSubgroupArithmetic=false.
            ProbeSubgroup(physical, apiVersion, out uint subgroupSize, out bool hasArithmetic);

            // Probe VK_KHR_cooperative_matrix. Requires the device extension
            // to be enabled at vkCreateDevice time for the shader to use it,
            // so we must decide support *before* creating the logical device.
            // Skipped gracefully on Vulkan 1.0 — returns empty shape list.
            ProbeCooperativeMatrix(
                instance, physical, apiVersion,
                out bool hasCoopmat, out var coopmatShapes);

            // Probe VK_EXT_external_memory_host. Same gating as coopmat — the
            // extension must be enabled at vkCreateDevice time before
            // vkAllocateMemory will accept VkImportMemoryHostPointerInfoEXT.
            // VK_KHR_external_memory is the dependency (core in 1.1) and is
            // always available on a 1.1+ driver. Falls back silently when
            // absent — caller checks HasExternalMemoryHost.
            ProbeExternalMemoryHost(
                physical, apiVersion,
                out bool hasExternalMemoryHost, out ulong minImportedHostPointerAlignment);

            // Probe VK_KHR_shader_integer_dot_product (Vulkan 1.3 core). Like
            // coopmat, the extension + feature must be enabled at
            // vkCreateDevice time before the dp4a MMVQ shader can run, so we
            // decide support before creating the logical device. Skipped
            // gracefully on Vulkan 1.0 — returns false.
            ProbeIntegerDotProduct(
                physical, apiVersion,
                out bool hasIntegerDotProduct);

            // Probe VK_EXT_subgroup_size_control (Vulkan 1.3 core). Like the
            // others, the feature must be enabled at vkCreateDevice time before
            // a pipeline may pin its subgroup size, so we decide support before
            // creating the logical device. Skipped gracefully on < 1.3 / missing
            // extension — returns false + zero sizes.
            ProbeSubgroupSizeControl(
                physical, apiVersion,
                out bool hasSubgroupSizeControl, out uint minSubgroupSize,
                out uint maxSubgroupSize, out uint requiredSubgroupSizeStages);

            // Probe VK_KHR_external_semaphore + VK_KHR_external_semaphore_win32
            // (Win32 only). Required for the M3 cross-API handoff: the Vulkan
            // forward submit signals an exported semaphore that CUDA waits on.
            // Falls back silently when absent — caller checks HasExternalSemaphoreWin32.
            ProbeExternalSemaphoreWin32(physical, apiVersion, out bool hasExternalSemaphoreWin32);

            nint device = CreateLogicalDevice(
                physical, queueFamily, hasCoopmat, hasExternalMemoryHost, hasIntegerDotProduct,
                hasSubgroupSizeControl, hasExternalSemaphoreWin32);

            VulkanApi.vkGetDeviceQueue(device, queueFamily, 0, out nint queue);

            var cpInfo = new VkCommandPoolCreateInfo
            {
                sType = VkStructureType.CommandPoolCreateInfo,
                flags = VkCommandPoolCreateFlags.ResetCommandBuffer,
                queueFamilyIndex = queueFamily,
            };
            VulkanApi.vkCreateCommandPool(device, cpInfo, 0, out nint pool)
                .ThrowOnError("vkCreateCommandPool");

            // Transfer ownership of instance to the device on success.
            var result = new VulkanDevice(
                instance, physical, device, queue, pool, name, vendor, type, queueFamily,
                subgroupSize, hasArithmetic, hasCoopmat, coopmatShapes,
                hasExternalMemoryHost, minImportedHostPointerAlignment,
                hasIntegerDotProduct,
                hasSubgroupSizeControl, minSubgroupSize, maxSubgroupSize,
                requiredSubgroupSizeStages, hasExternalSemaphoreWin32);
            instance = 0;
            return result;
        }
        finally
        {
            if (instance != 0)
                VulkanApi.vkDestroyInstance(instance, 0);
        }
    }

    private static nint CreateInstance()
    {
        // VK_MAKE_API_VERSION(0, 1, 2, 0) = Vulkan 1.2
        const uint apiVersion = (1u << 22) | (2u << 12);

        // Note: pApplicationName / pEngineName left null — we don't need strings.
        var appInfo = new VkApplicationInfo
        {
            sType = VkStructureType.ApplicationInfo,
            apiVersion = apiVersion,
        };

        unsafe
        {
            VkInstanceCreateInfo ci = default;
            ci.sType = VkStructureType.InstanceCreateInfo;
            ci.pApplicationInfo = (nint)(&appInfo);
            int r = VulkanApi.vkCreateInstance(ci, 0, out nint inst);
            return r >= 0 ? inst : 0;
        }
    }

    private static nint SelectPhysicalDevice(
        nint instance, int? forcedIndex, out string name, out uint vendor, out int type, out uint apiVersion)
    {
        uint count = 0;
        VulkanApi.vkEnumeratePhysicalDevices(instance, ref count, null)
            .ThrowOnError("vkEnumeratePhysicalDevices (count)");
        if (count == 0)
            throw new VulkanException(-3, "No Vulkan physical devices found.");

        var devices = new nint[count];
        VulkanApi.vkEnumeratePhysicalDevices(instance, ref count, devices)
            .ThrowOnError("vkEnumeratePhysicalDevices");

        // Explicit per-replica selection (Create(int deviceIndex)) takes precedence over env + scoring, so a
        // single process can bind different GPUs for prefill vs decode in a cross-device handoff.
        if (forcedIndex is int fi)
        {
            if ((uint)fi >= (uint)devices.Length)
                throw new VulkanException(-3,
                    $"Requested Vulkan device index {fi} is out of range (0..{devices.Length - 1}).");
            nint chosen = devices[fi];
            VulkanApi.vkGetPhysicalDeviceProperties(chosen, out var cp);
            name = ReadDeviceName(cp);
            vendor = cp.vendorID;
            type = cp.deviceType;
            apiVersion = cp.apiVersion;
            return chosen;
        }

        // Manual override (testing / iGPU targeting): DOTLLM_VULKAN_DEVICE_INDEX picks a device by
        // enumeration index; DOTLLM_VULKAN_DEVICE_VENDOR (hex, e.g. 0x8086) picks the first device of
        // that PCI vendor. Either lets us force the integrated Intel Arc on a box where the scorer would
        // otherwise pick the discrete NVIDIA. Falls through to scoring if unset/invalid.
        nint forced = ResolveForcedDevice(devices);
        if (forced != 0)
        {
            VulkanApi.vkGetPhysicalDeviceProperties(forced, out var fp);
            name = ReadDeviceName(fp);
            vendor = fp.vendorID;
            type = fp.deviceType;
            apiVersion = fp.apiVersion;
            return forced;
        }

        // Score every device. Prefer: discrete > integrated > other/CPU.
        // Within discrete, prefer AMD/NVIDIA over Intel (Intel rarely has dGPUs,
        // but if one is present it's often weaker than an AMD/NVIDIA dGPU).
        nint bestDev = 0;
        int bestScore = int.MinValue;
        string bestName = "unknown";
        uint bestVendor = 0;
        int bestType = 0;
        uint bestApi = 0;

        foreach (var dev in devices)
        {
            VulkanApi.vkGetPhysicalDeviceProperties(dev, out var props);
            string devName = ReadDeviceName(props);
            int score = ScoreDevice(props.deviceType, props.vendorID);

            if (score > bestScore)
            {
                bestScore = score;
                bestDev = dev;
                bestName = devName;
                bestVendor = props.vendorID;
                bestType = props.deviceType;
                bestApi = props.apiVersion;
            }
        }

        name = bestName;
        vendor = bestVendor;
        type = bestType;
        apiVersion = bestApi;
        return bestDev;
    }

    /// <summary>
    /// Resolves a manually-forced physical device from environment overrides, or 0 if none/invalid.
    /// <c>DOTLLM_VULKAN_DEVICE_INDEX</c> selects by enumeration index; <c>DOTLLM_VULKAN_DEVICE_VENDOR</c>
    /// (hex PCI vendor, e.g. 0x8086 for Intel) selects the first device of that vendor.
    /// </summary>
    private static nint ResolveForcedDevice(nint[] devices)
    {
        string? idxEnv = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_INDEX");
        if (int.TryParse(idxEnv, out int idx) && idx >= 0 && idx < devices.Length)
            return devices[idx];

        string? vendorEnv = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_VENDOR");
        if (!string.IsNullOrEmpty(vendorEnv))
        {
            string hex = vendorEnv.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? vendorEnv[2..] : vendorEnv;
            if (uint.TryParse(hex, System.Globalization.NumberStyles.HexNumber, null, out uint wantVendor))
            {
                foreach (var dev in devices)
                {
                    VulkanApi.vkGetPhysicalDeviceProperties(dev, out var props);
                    if (props.vendorID == wantVendor)
                        return dev;
                }
            }
        }
        return 0;
    }

    // Packed Vulkan API version helpers. Layout: variant(3) | major(7) | minor(10) | patch(12).
    private static uint VkApiMajor(uint packed) => (packed >> 22) & 0x7Fu;
    private static uint VkApiMinor(uint packed) => (packed >> 12) & 0x3FFu;

    /// <summary>
    /// Queries <c>VkPhysicalDeviceSubgroupProperties</c> via the Vulkan 1.1
    /// core entry point <c>vkGetPhysicalDeviceProperties2</c>. Safely degrades
    /// on Vulkan 1.0 devices (where the entry point does not exist) by
    /// returning <c>size=0, hasArithmetic=false</c> — callers then stick to
    /// the shared-memory path without regressing on older hardware.
    /// </summary>
    private static void ProbeSubgroup(nint physical, uint apiVersion, out uint subgroupSize, out bool hasArithmetic)
    {
        subgroupSize = 0;
        hasArithmetic = false;

        // Gate on the driver's reported API version. vkGetPhysicalDeviceProperties2
        // is core in Vulkan 1.1 (May 2018). Prior to that the function symbol is
        // not guaranteed to exist in the loader's dispatch table.
        if (VkApiMajor(apiVersion) < 1u || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) < 1u))
            return;

        try
        {
            VkPhysicalDeviceSubgroupProperties sub = default;
            sub.sType = VkStructureType.PhysicalDeviceSubgroupProperties;

            VkPhysicalDeviceProperties2 props2 = default;
            props2.sType = VkStructureType.PhysicalDeviceProperties2;

            unsafe
            {
                props2.pNext = (nint)(&sub);
                VulkanApi.vkGetPhysicalDeviceProperties2(physical, ref props2);
            }

            subgroupSize = sub.subgroupSize;

            // Require BOTH: arithmetic op support AND compute-stage visibility.
            // The Vulkan spec (§36.2) lists individual stage bits in supportedStages;
            // COMPUTE is 0x20. On a conformant driver both conditions are usually
            // set together for arithmetic, but we're explicit.
            const uint stageCompute = VkShaderStageFlags.Compute;
            bool stageOk = (sub.supportedStages & stageCompute) != 0;
            bool featureOk = ((VkSubgroupFeatureFlags)sub.supportedOperations & VkSubgroupFeatureFlags.Arithmetic) != 0;
            hasArithmetic = stageOk && featureOk && subgroupSize > 0;
        }
        catch
        {
            // Loader or driver returned garbage — disable fast path. The
            // shared-memory shaders run on every Vulkan 1.0+ device.
            subgroupSize = 0;
            hasArithmetic = false;
        }
    }

    /// <summary>
    /// Probes <c>VK_KHR_cooperative_matrix</c> support. Enumerates the device
    /// extensions, chains the <c>VkPhysicalDeviceCooperativeMatrixFeaturesKHR</c>
    /// feature bit through <c>vkGetPhysicalDeviceFeatures2</c>, and
    /// dynamically resolves <c>vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR</c>
    /// via <c>vkGetInstanceProcAddr</c> to list every driver-supported tile
    /// shape. Support is declared when at least one F16×F16→F32 (or
    /// Sint8×Sint8→Sint32) shape of at least 16×16×16 at subgroup scope is
    /// reported. Safe on Vulkan 1.0 / non-coopmat drivers: returns
    /// <c>false</c> + empty list without throwing.
    /// </summary>
    private static unsafe void ProbeCooperativeMatrix(
        nint instance, nint physical, uint apiVersion,
        out bool hasCoopmat, out IReadOnlyList<CooperativeMatrixShape> shapes)
    {
        hasCoopmat = false;
        shapes = Array.Empty<CooperativeMatrixShape>();

        // vkGetPhysicalDeviceFeatures2 is core in Vulkan 1.1. Drivers that
        // only report 1.0 definitely don't support VK_KHR_cooperative_matrix
        // anyway (the extension requires 1.1 + subgroup_basic).
        if (VkApiMajor(apiVersion) < 1u || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) < 1u))
            return;

        try
        {
            // 1. Confirm the device advertises the extension.
            if (!HasDeviceExtension(physical, "VK_KHR_cooperative_matrix"u8))
                return;

            // 2. Confirm the `cooperativeMatrix` feature bit is actually set.
            VkPhysicalDeviceCooperativeMatrixFeaturesKhr coopFeatures = default;
            coopFeatures.sType = VkStructureType.PhysicalDeviceCooperativeMatrixFeaturesKhr;

            VkPhysicalDeviceFeatures2 features2 = default;
            features2.sType = VkStructureType.PhysicalDeviceFeatures2;
            features2.pNext = (nint)(&coopFeatures);

            VulkanApi.vkGetPhysicalDeviceFeatures2(physical, ref features2);
            if (coopFeatures.cooperativeMatrix == 0)
                return;

            // 3. Resolve vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR via
            //    vkGetInstanceProcAddr — it's instance-level but registered by
            //    the driver's extension dispatch table rather than the core
            //    loader, so a static P/Invoke would fail on non-coopmat drivers.
            nint fn = VulkanApi.vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
            if (fn == 0) return;

            var getCoopmatProps = Marshal.GetDelegateForFunctionPointer<VkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(fn);

            // 4. Two-pass enumeration (spec idiom).
            uint count = 0;
            int r = getCoopmatProps(physical, ref count, 0);
            if (r < 0 || count == 0) return;

            var props = new VkCooperativeMatrixPropertiesKhr[count];
            for (int i = 0; i < count; i++)
                props[i].sType = VkStructureType.CooperativeMatrixPropertiesKhr;

            fixed (VkCooperativeMatrixPropertiesKhr* pPtr = props)
            {
                r = getCoopmatProps(physical, ref count, (nint)pPtr);
                if (r < 0) return;
            }

            // 5. Convert to the public shape list and check for a usable entry.
            var shapeList = new List<CooperativeMatrixShape>((int)count);
            bool foundUsable = false;
            for (int i = 0; i < count; i++)
            {
                var p = props[i];
                shapeList.Add(new CooperativeMatrixShape(
                    (int)p.MSize, (int)p.NSize, (int)p.KSize,
                    p.AType, p.BType, p.CType, p.ResultType,
                    p.scope));

                if (p.scope != VkScopeKhr.Subgroup) continue;
                if (p.MSize < 16u || p.NSize < 16u || p.KSize < 16u) continue;

                // F16 × F16 → F32 accumulator (the main dotLLM coopmat path).
                bool f16f32 =
                    p.AType == VkComponentTypeKhr.Float16 &&
                    p.BType == VkComponentTypeKhr.Float16 &&
                    p.CType == VkComponentTypeKhr.Float32 &&
                    p.ResultType == VkComponentTypeKhr.Float32;

                // Sint8 × Sint8 → Sint32 (future Q8_0-direct int tile path).
                bool i8i32 =
                    p.AType == VkComponentTypeKhr.Sint8 &&
                    p.BType == VkComponentTypeKhr.Sint8 &&
                    p.CType == VkComponentTypeKhr.Sint32 &&
                    p.ResultType == VkComponentTypeKhr.Sint32;

                if (f16f32 || i8i32) foundUsable = true;
            }

            shapes = shapeList;
            hasCoopmat = foundUsable;
        }
        catch
        {
            // Loader / driver behaved unexpectedly — disable coopmat and
            // fall back to the scalar GEMM path on every shape.
            hasCoopmat = false;
            shapes = Array.Empty<CooperativeMatrixShape>();
        }
    }

    // Delegate matching vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR for
    // dynamic resolution via vkGetInstanceProcAddr. Standalone function-pointer
    // type keeps the signature colocated with its sole use.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int VkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
        nint physicalDevice, ref uint pPropertyCount, nint pProperties);

    // Delegate matching vkGetMemoryHostPointerPropertiesEXT for dynamic
    // resolution via vkGetDeviceProcAddr. Resolved lazily after device
    // creation (the device must have enabled VK_EXT_external_memory_host)
    // and cached on the VulkanDevice instance.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int VkGetMemoryHostPointerPropertiesEXT(
        nint device, uint handleType, nint pHostPointer,
        ref VkMemoryHostPointerPropertiesExt pMemoryHostPointerProperties);

    // Delegate matching vkGetSemaphoreWin32HandleKHR (VK_KHR_external_semaphore_win32).
    // Resolved lazily via vkGetDeviceProcAddr after device creation (the extension
    // must have been enabled at vkCreateDevice). Returns the exportable
    // semaphore's Win32 HANDLE in pHandle for import into CUDA.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int VkGetSemaphoreWin32HandleKHR(
        nint device, in VkSemaphoreGetWin32HandleInfoKhr pGetWin32HandleInfo, out nint pHandle);

    /// <summary>
    /// Probes <c>VK_EXT_external_memory_host</c> support. The extension lets
    /// callers back a <c>VkDeviceMemory</c> with an existing host mmap'd
    /// pointer via <c>VkImportMemoryHostPointerInfoEXT</c>, which on
    /// unified-memory APUs eliminates the host→device staging copy for
    /// weight uploads. We probe by enumerating device extensions; when
    /// present we chain a <c>VkPhysicalDeviceExternalMemoryHostPropertiesEXT</c>
    /// off <c>vkGetPhysicalDeviceProperties2</c> to fetch the minimum
    /// import alignment.
    /// </summary>
    /// <remarks>
    /// AMD driver coverage (researched 2026-05-14):
    /// <list type="bullet">
    ///   <item>amdvlk (Windows + Linux): supports VK_EXT_external_memory_host.</item>
    ///   <item>Mesa radv (Linux): supports VK_EXT_external_memory_host since
    ///     Mesa 18.3 (2018).</item>
    ///   <item>Windows AMD driver (radv stack via OpenCL or amdvlk): exposes
    ///     the extension on RDNA generations including gfx1151 (Strix Halo).</item>
    /// </list>
    /// Intel ANV exposes the extension as well; NVIDIA exposes it on
    /// post-Pascal hardware. On any driver that does not advertise the
    /// extension, this probe returns <c>false</c> and the caller falls
    /// back to the staging-copy upload path.
    /// </remarks>
    private static unsafe void ProbeExternalMemoryHost(
        nint physical, uint apiVersion,
        out bool hasExternalMemoryHost, out ulong minImportedHostPointerAlignment)
    {
        hasExternalMemoryHost = false;
        minImportedHostPointerAlignment = 0;

        // vkGetPhysicalDeviceProperties2 is core in Vulkan 1.1. Drivers that
        // only report 1.0 don't expose the extension either. Gate accordingly.
        if (VkApiMajor(apiVersion) < 1u || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) < 1u))
            return;

        try
        {
            if (!HasDeviceExtension(physical, "VK_EXT_external_memory_host"u8))
                return;

            VkPhysicalDeviceExternalMemoryHostPropertiesExt extProps = default;
            extProps.sType = VkStructureType.PhysicalDeviceExternalMemoryHostPropertiesExt;

            VkPhysicalDeviceProperties2 props2 = default;
            props2.sType = VkStructureType.PhysicalDeviceProperties2;
            props2.pNext = (nint)(&extProps);

            VulkanApi.vkGetPhysicalDeviceProperties2(physical, ref props2);

            // Drivers must return a non-zero alignment when the extension is
            // present. Defensive guard: a buggy driver reporting 0 would let
            // us divide by zero downstream — treat as "feature unusable".
            if (extProps.minImportedHostPointerAlignment == 0)
                return;

            minImportedHostPointerAlignment = extProps.minImportedHostPointerAlignment;
            hasExternalMemoryHost = true;
        }
        catch
        {
            // Loader/driver returned garbage — disable feature, callers fall
            // back to the staging-copy path. Mirrors the cooperative-matrix
            // probe's defensive posture.
            hasExternalMemoryHost = false;
            minImportedHostPointerAlignment = 0;
        }
    }

    /// <summary>
    /// Probes <c>VK_KHR_shader_integer_dot_product</c> (Vulkan 1.3 core)
    /// support. Chains a <c>VkPhysicalDeviceShaderIntegerDotProductFeatures</c>
    /// off <c>vkGetPhysicalDeviceFeatures2</c> and checks the
    /// <c>shaderIntegerDotProduct</c> bit. On a 1.3 driver the feature is core
    /// (no extension string needed); on a 1.1/1.2 driver the extension must
    /// also be advertised. We treat "extension present OR core 1.3" plus the
    /// feature bit set as supported. The actual hardware acceleration (the
    /// <c>integerDotProduct4x8BitPackedSignedAccelerated</c> property) is not
    /// required for correctness — the shader works on any driver exposing the
    /// feature; acceleration only affects throughput.
    /// </summary>
    private static unsafe void ProbeIntegerDotProduct(
        nint physical, uint apiVersion, out bool hasIntegerDotProduct)
    {
        hasIntegerDotProduct = false;

        // vkGetPhysicalDeviceFeatures2 is core in Vulkan 1.1.
        if (VkApiMajor(apiVersion) < 1u || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) < 1u))
            return;

        try
        {
            // On a Vulkan 1.2 driver the feature lives behind the extension;
            // on 1.3 it is core. Accept either: a 1.3+ device, or a device
            // that advertises the extension string.
            bool core13 = VkApiMajor(apiVersion) > 1u
                || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) >= 3u);
            if (!core13 && !HasDeviceExtension(physical, "VK_KHR_shader_integer_dot_product"u8))
                return;

            VkPhysicalDeviceShaderIntegerDotProductFeatures dotFeatures = default;
            dotFeatures.sType = VkStructureType.PhysicalDeviceShaderIntegerDotProductFeatures;

            VkPhysicalDeviceFeatures2 features2 = default;
            features2.sType = VkStructureType.PhysicalDeviceFeatures2;
            features2.pNext = (nint)(&dotFeatures);

            VulkanApi.vkGetPhysicalDeviceFeatures2(physical, ref features2);

            hasIntegerDotProduct = dotFeatures.shaderIntegerDotProduct != 0;
        }
        catch
        {
            // Loader/driver returned garbage — disable the feature; callers
            // fall back to the F32-in Q8_0 GEMV. Mirrors the coopmat /
            // external-memory probe's defensive posture.
            hasIntegerDotProduct = false;
        }
    }

    /// <summary>
    /// Probes <c>VK_EXT_subgroup_size_control</c> (Vulkan 1.3 core). Reads
    /// <c>VkPhysicalDeviceSubgroupSizeControlProperties</c> (min/max subgroup
    /// size + the stages that accept a required size) via
    /// <c>vkGetPhysicalDeviceProperties2</c> and the
    /// <c>subgroupSizeControl</c>/<c>computeFullSubgroups</c> feature bits via
    /// <c>vkGetPhysicalDeviceFeatures2</c>. Support is declared when the device
    /// reports a valid subgroup-size range AND the compute stage accepts a
    /// required size — that is exactly what a per-kernel wave32 pin (with
    /// REQUIRE_FULL_SUBGROUPS) needs, and what llama.cpp keys off. The feature
    /// bits are read for completeness but are NOT part of the gate (they are
    /// unreliable on a 1.3-core feature under a 1.2 instance — see the remarks on
    /// <see cref="HasSubgroupSizeControl"/>); the features are still enabled at
    /// device create. Like the integer-dot probe, accepts a 1.3+ core device or
    /// one that advertises the extension string. Safe on older drivers: returns
    /// <c>false</c> + zero sizes without throwing.
    /// </summary>
    private static unsafe void ProbeSubgroupSizeControl(
        nint physical, uint apiVersion,
        out bool hasSubgroupSizeControl, out uint minSubgroupSize,
        out uint maxSubgroupSize, out uint requiredSubgroupSizeStages)
    {
        hasSubgroupSizeControl = false;
        minSubgroupSize = 0;
        maxSubgroupSize = 0;
        requiredSubgroupSizeStages = 0;

        // vkGetPhysicalDeviceProperties2 / Features2 are core in Vulkan 1.1.
        if (VkApiMajor(apiVersion) < 1u || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) < 1u))
            return;

        try
        {
            // Core in Vulkan 1.3; on 1.1/1.2 the extension must be advertised.
            bool core13 = VkApiMajor(apiVersion) > 1u
                || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) >= 3u);
            bool hasExt = HasDeviceExtension(physical, "VK_EXT_subgroup_size_control"u8);
            if (!core13 && !hasExt)
                return;

            // Read the feature bits for completeness. NOTE: these are advisory —
            // when the feature was promoted to Vulkan 1.3 core, some loader/driver
            // combinations only populate the promoted *feature* struct when the
            // INSTANCE was created at apiVersion >= 1.3 (our instance requests
            // 1.2). The *properties* struct (below) is keyed off the EXT and is
            // populated reliably. We therefore gate support on the properties +
            // the compute stage being in requiredSubgroupSizeStages, which is
            // exactly what enabling a per-pipeline required size needs, and what
            // llama.cpp keys off. The features are still enabled at device
            // create; vkCreateDevice would fail if the driver could not honour
            // them (it does not on gfx1151).
            VkPhysicalDeviceSubgroupSizeControlFeatures sscFeatures = default;
            sscFeatures.sType = VkStructureType.PhysicalDeviceSubgroupSizeControlFeatures;

            VkPhysicalDeviceFeatures2 features2 = default;
            features2.sType = VkStructureType.PhysicalDeviceFeatures2;
            features2.pNext = (nint)(&sscFeatures);

            VulkanApi.vkGetPhysicalDeviceFeatures2(physical, ref features2);

            // Properties — min/max size and which stages accept a required size.
            VkPhysicalDeviceSubgroupSizeControlProperties sscProps = default;
            sscProps.sType = VkStructureType.PhysicalDeviceSubgroupSizeControlProperties;

            VkPhysicalDeviceProperties2 props2 = default;
            props2.sType = VkStructureType.PhysicalDeviceProperties2;
            props2.pNext = (nint)(&sscProps);

            VulkanApi.vkGetPhysicalDeviceProperties2(physical, ref props2);

            // Defensive: a driver reporting a degenerate range is unusable.
            if (sscProps.minSubgroupSize == 0 || sscProps.maxSubgroupSize == 0
                || sscProps.minSubgroupSize > sscProps.maxSubgroupSize)
                return;

            // Require the COMPUTE stage to accept a required subgroup size — the
            // MMVQ decode pipelines are compute. Without this bit a per-pipeline
            // pin is illegal and vkCreateComputePipelines would fail.
            if ((sscProps.requiredSubgroupSizeStages & VkShaderStageFlags.Compute) == 0)
                return;

            minSubgroupSize = sscProps.minSubgroupSize;
            maxSubgroupSize = sscProps.maxSubgroupSize;
            requiredSubgroupSizeStages = sscProps.requiredSubgroupSizeStages;
            hasSubgroupSizeControl = true;
        }
        catch
        {
            // Loader/driver returned garbage — disable; callers fall back to the
            // default (unset) subgroup size. Mirrors the other probes' posture.
            hasSubgroupSizeControl = false;
            minSubgroupSize = 0;
            maxSubgroupSize = 0;
            requiredSubgroupSizeStages = 0;
        }
    }

    /// <summary>
    /// Probes <c>VK_KHR_external_semaphore</c> + <c>VK_KHR_external_semaphore_win32</c>
    /// support. Both must be advertised for the Vulkan→CUDA Win32-handle handoff;
    /// the extensions carry no feature bit, so presence in the device extension
    /// list is sufficient. Windows-only (the win32 extension does not exist on
    /// other platforms). Safe on Vulkan 1.0 / non-Windows: returns <c>false</c>
    /// and callers fall back to the fence-serialized handoff.
    /// </summary>
    private static unsafe void ProbeExternalSemaphoreWin32(
        nint physical, uint apiVersion, out bool hasExternalSemaphoreWin32)
    {
        hasExternalSemaphoreWin32 = false;

        // The win32 export handle type only exists on Windows.
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return;

        // VK_KHR_external_semaphore is core in Vulkan 1.1; the win32 companion
        // requires it. Gate on the reported API version.
        if (VkApiMajor(apiVersion) < 1u || (VkApiMajor(apiVersion) == 1u && VkApiMinor(apiVersion) < 1u))
            return;

        try
        {
            hasExternalSemaphoreWin32 =
                HasDeviceExtension(physical, "VK_KHR_external_semaphore"u8) &&
                HasDeviceExtension(physical, "VK_KHR_external_semaphore_win32"u8);
        }
        catch
        {
            hasExternalSemaphoreWin32 = false;
        }
    }

    /// <summary>
    /// Returns <c>true</c> when <paramref name="physical"/> advertises the
    /// given device extension. <paramref name="name"/> must be the
    /// NUL-terminated UTF-8 extension name (e.g. <c>"VK_KHR_cooperative_matrix\0"u8</c>).
    /// </summary>
    private static unsafe bool HasDeviceExtension(nint physical, ReadOnlySpan<byte> name)
    {
        uint count = 0;
        VulkanApi.vkEnumerateDeviceExtensionProperties(physical, 0, ref count, 0);
        if (count == 0) return false;

        // VkExtensionProperties = char extensionName[256] + uint specVersion.
        // Total stride 260 bytes per entry. Allocate from unmanaged memory —
        // safer than a large stackalloc when count is driver-dependent.
        const int entrySize = 256 + 4;
        nint ptr = (nint)NativeMemory.Alloc((nuint)entrySize * count);
        try
        {
            int r = VulkanApi.vkEnumerateDeviceExtensionProperties(physical, 0, ref count, ptr);
            if (r < 0) return false;

            byte* p = (byte*)ptr;
            for (uint i = 0; i < count; i++)
            {
                byte* entry = p + (nint)i * entrySize;
                if (EqualsUtf8CString(entry, name))
                    return true;
            }
            return false;
        }
        finally
        {
            NativeMemory.Free((void*)ptr);
        }
    }

    /// <summary>
    /// Compares a NUL-terminated C string at <paramref name="p"/> against a
    /// NUL-terminated <paramref name="needle"/> span. Returns true on exact
    /// match (including matching NULs).
    /// </summary>
    private static unsafe bool EqualsUtf8CString(byte* p, ReadOnlySpan<byte> needle)
    {
        for (int i = 0; i < needle.Length; i++)
        {
            if (p[i] != needle[i]) return false;
            if (needle[i] == 0) return true;
        }
        return p[needle.Length] == 0;
    }

    // Vendor IDs are PCI SIG assignments. 0x10DE=NVIDIA, 0x1002=AMD, 0x8086=Intel, 0x13B5=ARM, 0x5143=Qualcomm.
    private static int ScoreDevice(int deviceType, uint vendorId)
    {
        int typeScore = deviceType switch
        {
            VkPhysicalDeviceType.DiscreteGpu => 1000,
            VkPhysicalDeviceType.IntegratedGpu => 500,
            VkPhysicalDeviceType.VirtualGpu => 100,
            _ => 0,
        };
        int vendorScore = vendorId switch
        {
            0x10DE => 20, // NVIDIA
            0x1002 => 20, // AMD
            0x8086 => 10, // Intel — lower preference when a dGPU is also present
            _ => 5,
        };
        return typeScore + vendorScore;
    }

    private static unsafe string ReadDeviceName(VkPhysicalDeviceProperties props)
    {
        byte* p = props.deviceName;
        int len = 0;
        while (len < 256 && p[len] != 0) len++;
        return Encoding.UTF8.GetString(p, len);
    }

    private static uint SelectComputeQueueFamily(nint physical)
    {
        uint count = 0;
        VulkanApi.vkGetPhysicalDeviceQueueFamilyProperties(physical, ref count, null);
        if (count == 0)
            throw new VulkanException(-3, "Physical device reports zero queue families.");

        var families = new VkQueueFamilyProperties[count];
        VulkanApi.vkGetPhysicalDeviceQueueFamilyProperties(physical, ref count, families);

        // Pick the first family that supports COMPUTE. A dedicated compute-only
        // queue (compute without graphics) is nice-to-have but not required for
        // this scaffold.
        for (uint i = 0; i < count; i++)
        {
            if ((families[i].queueFlags & VkQueueFlags.Compute) != 0)
                return i;
        }
        throw new VulkanException(-3, "No queue family with COMPUTE capability.");
    }

    private static unsafe nint CreateLogicalDevice(
        nint physical, uint queueFamily,
        bool enableCoopmat, bool enableExternalMemoryHost, bool enableIntegerDotProduct,
        bool enableSubgroupSizeControl, bool enableExternalSemaphoreWin32)
    {
        float priority = 1.0f;

        var qci = new VkDeviceQueueCreateInfo
        {
            sType = VkStructureType.DeviceQueueCreateInfo,
            queueFamilyIndex = queueFamily,
            queueCount = 1,
            pQueuePriorities = (nint)(&priority),
        };

        VkDeviceCreateInfo ci = default;
        ci.sType = VkStructureType.DeviceCreateInfo;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = (nint)(&qci);

        // Stack-buffer for the (small) extension-name table. VK_KHR_external_memory
        // is a hard dependency of VK_EXT_external_memory_host and is core in 1.1,
        // but conservatively we enable both extension names explicitly — drivers
        // that promoted the symbol to core ignore the duplicate.
        ReadOnlySpan<byte> coopmatName = "VK_KHR_cooperative_matrix\0"u8;
        ReadOnlySpan<byte> extMemHostName = "VK_EXT_external_memory_host\0"u8;
        ReadOnlySpan<byte> extMemName = "VK_KHR_external_memory\0"u8;
        ReadOnlySpan<byte> intDotName = "VK_KHR_shader_integer_dot_product\0"u8;
        ReadOnlySpan<byte> sscName = "VK_EXT_subgroup_size_control\0"u8;
        ReadOnlySpan<byte> extSemName = "VK_KHR_external_semaphore\0"u8;
        ReadOnlySpan<byte> extSemWin32Name = "VK_KHR_external_semaphore_win32\0"u8;

        // Pack name bytes + pointer array onto the stack. Worst case all seven
        // extension names are enabled at once (coopmat, external-memory ×2,
        // integer-dot-product, subgroup-size-control, external-semaphore ×2). The
        // integer-dot-product and subgroup-size-control strings are harmless to
        // name even on a 1.3 driver where they are core — drivers ignore the
        // duplicate, same as the external-memory/semaphore names below.
        byte* nameBytes = stackalloc byte[
            coopmatName.Length + extMemHostName.Length + extMemName.Length
            + intDotName.Length + sscName.Length
            + extSemName.Length + extSemWin32Name.Length];
        nint* namePtrs = stackalloc nint[7];
        int nameOffset = 0;
        uint extCount = 0;

        if (enableCoopmat)
        {
            for (int i = 0; i < coopmatName.Length; i++) nameBytes[nameOffset + i] = coopmatName[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += coopmatName.Length;
        }

        if (enableExternalSemaphoreWin32)
        {
            // VK_KHR_external_semaphore is core in 1.1 but, like external_memory,
            // some drivers (amdvlk) require it named explicitly when the
            // VkExportSemaphoreCreateInfo pNext struct is used. Enable both names.
            for (int i = 0; i < extSemName.Length; i++) nameBytes[nameOffset + i] = extSemName[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += extSemName.Length;

            for (int i = 0; i < extSemWin32Name.Length; i++) nameBytes[nameOffset + i] = extSemWin32Name[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += extSemWin32Name.Length;
        }

        if (enableExternalMemoryHost)
        {
            // VK_KHR_external_memory is core in Vulkan 1.1 so the driver
            // typically does not require us to name it here, but spec is
            // explicit that the bit type's home extension must be in the
            // enabled-extension list when external-memory pNext structs are
            // used at allocation time. Mesa radv tolerates omitting it;
            // amdvlk is stricter. Enable both — costs one extra string pointer.
            for (int i = 0; i < extMemName.Length; i++) nameBytes[nameOffset + i] = extMemName[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += extMemName.Length;

            for (int i = 0; i < extMemHostName.Length; i++) nameBytes[nameOffset + i] = extMemHostName[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += extMemHostName.Length;
        }

        if (enableIntegerDotProduct)
        {
            for (int i = 0; i < intDotName.Length; i++) nameBytes[nameOffset + i] = intDotName[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += intDotName.Length;
        }

        if (enableSubgroupSizeControl)
        {
            for (int i = 0; i < sscName.Length; i++) nameBytes[nameOffset + i] = sscName[i];
            namePtrs[extCount++] = (nint)(nameBytes + nameOffset);
            nameOffset += sscName.Length;
        }

        // Feature structs chained through pNext on top of the extension enables:
        //  - VK_KHR_cooperative_matrix requires `cooperativeMatrix=VK_TRUE`.
        //  - VK_KHR_shader_integer_dot_product requires
        //    `shaderIntegerDotProduct=VK_TRUE`.
        // Both are chained when enabled (coopmat first, dot-product second).
        // No feature bits are needed for VK_EXT_external_memory_host.
        VkPhysicalDeviceCooperativeMatrixFeaturesKhr coopmatFeatures = default;
        VkPhysicalDeviceShaderIntegerDotProductFeatures dotFeatures = default;
        VkPhysicalDeviceSubgroupSizeControlFeatures sscFeatures = default;
        nint featureChain = 0;
        if (enableCoopmat)
        {
            coopmatFeatures.sType = VkStructureType.PhysicalDeviceCooperativeMatrixFeaturesKhr;
            coopmatFeatures.cooperativeMatrix = 1; // VK_TRUE
            coopmatFeatures.cooperativeMatrixRobustBufferAccess = 0;
            coopmatFeatures.pNext = featureChain;
            featureChain = (nint)(&coopmatFeatures);
        }
        if (enableIntegerDotProduct)
        {
            dotFeatures.sType = VkStructureType.PhysicalDeviceShaderIntegerDotProductFeatures;
            dotFeatures.shaderIntegerDotProduct = 1; // VK_TRUE
            dotFeatures.pNext = featureChain;
            featureChain = (nint)(&dotFeatures);
        }
        if (enableSubgroupSizeControl)
        {
            // Enable both bits: subgroupSizeControl lets a pipeline pin its
            // subgroup size; computeFullSubgroups lets us set the
            // REQUIRE_FULL_SUBGROUPS stage flag that pairs with the pin.
            sscFeatures.sType = VkStructureType.PhysicalDeviceSubgroupSizeControlFeatures;
            sscFeatures.subgroupSizeControl = 1;  // VK_TRUE
            sscFeatures.computeFullSubgroups = 1; // VK_TRUE
            sscFeatures.pNext = featureChain;
            featureChain = (nint)(&sscFeatures);
        }

        // Timeline semaphores (core 1.2) require `timelineSemaphore=VK_TRUE` enabled
        // at device create. We need them for the D3D12_FENCE export type CUDA imports
        // cross-vendor (Intel Vulkan → NVIDIA CUDA) — the OPAQUE_WIN32 binary form
        // fails import on that pairing. Enable alongside the external-semaphore-win32
        // path so the M3 handoff can create exportable timeline semaphores.
        VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = default;
        if (enableExternalSemaphoreWin32)
        {
            timelineFeatures.sType = VkStructureType.PhysicalDeviceTimelineSemaphoreFeatures;
            timelineFeatures.timelineSemaphore = 1; // VK_TRUE
            timelineFeatures.pNext = featureChain;
            featureChain = (nint)(&timelineFeatures);
        }

        ci.pNext = featureChain;

        // NOTE: with the validation layers enabled you may see a benign
        // VUID-VkDeviceCreateInfo-pNext-pNext "unexpected VkStructureType" naming a
        // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo in this chain. It is a
        // validation false-positive: the chained struct is VkPhysicalDeviceSubgroupSizeControl-
        // Features and its runtime sType is correct (1000225001, verified). The layer
        // mis-attributes it because we deliberately request a 1.2 instance while using the
        // (1.3-core) subgroup-size-control feature via its EXT alias. The driver accepts it
        // and the wave32 required-subgroup-size pins work. Not a real bug; do not "fix".

        if (extCount > 0)
        {
            ci.enabledExtensionCount = extCount;
            ci.ppEnabledExtensionNames = (nint)namePtrs;
        }

        VulkanApi.vkCreateDevice(physical, ci, 0, out nint dev)
            .ThrowOnError("vkCreateDevice");
        return dev;
    }

    // ────────────────────────────────────────────────────────────────
    // Buffer & memory helpers
    // ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Device-owned buffer + backing memory. Caller owns the <see cref="IDisposable"/>.
    /// </summary>
    /// <remarks>
    /// A <see cref="Buffer"/> may wrap either a driver-allocated
    /// <c>VkDeviceMemory</c> (the default path produced by
    /// <see cref="VulkanDevice.Allocate(long)"/> and
    /// <see cref="VulkanDevice.AllocateDeviceLocal"/>) <i>or</i> a host-imported
    /// allocation backed by an mmap'd file (produced by
    /// <see cref="HostVisibleBuffer.TryCreate"/> and exposed through
    /// <see cref="VulkanDevice.TryWrapHostVisible"/>). Downstream kernel
    /// code reads only <see cref="Handle"/>; the distinction matters
    /// for lifetime — in the host-imported case <see cref="Dispose"/>
    /// frees the import but does NOT touch the underlying mmap (the
    /// <c>GgufFile</c> owns that).
    /// </remarks>
    public sealed class Buffer : IDisposable
    {
        private readonly VulkanDevice _device;
        private nint _buffer;
        private nint _memory;
        private readonly HostVisibleBuffer? _hostImport;

        /// <summary>Buffer size in bytes.</summary>
        public long Size { get; }

        /// <summary>Underlying <c>VkBuffer</c> handle.</summary>
        public nint Handle => _buffer;

        /// <summary>
        /// True when this buffer wraps an mmap'd host pointer via
        /// <c>VK_EXT_external_memory_host</c> — disposal frees the Vulkan
        /// import but the underlying pages outlive it.
        /// </summary>
        public bool IsHostImported => _hostImport is not null;

        /// <summary>
        /// True when this buffer's memory is <c>HOST_VISIBLE</c> and can be mapped with
        /// <c>vkMapMemory</c> directly. Host-visible allocations and UMA device-local types are
        /// mappable; a <em>strictly</em> <c>DEVICE_LOCAL</c> type on a discrete GPU is NOT — host
        /// readback/upload to such a buffer must stage through a host-visible buffer +
        /// <c>vkCmdCopyBuffer</c> (see <see cref="VulkanDevice.Download"/> /
        /// <see cref="VulkanDevice.UploadToDeviceLocal"/>). On UMA parts (Strix Halo iGPU, Intel Arc)
        /// device-local is typically host-visible too, so the direct map fast-path applies.
        /// </summary>
        public bool IsHostVisible { get; }

        internal Buffer(VulkanDevice device, nint buffer, nint memory, long size, bool hostVisible)
        {
            _device = device;
            _buffer = buffer;
            _memory = memory;
            Size = size;
            _hostImport = null;
            IsHostVisible = hostVisible;
        }

        internal Buffer(VulkanDevice device, HostVisibleBuffer hostImport)
        {
            _device = device;
            _buffer = hostImport.Handle;
            _memory = hostImport.Memory;
            Size = hostImport.Size;
            _hostImport = hostImport;
            IsHostVisible = true; // wraps host pages — mappable by construction
        }

        /// <summary>Underlying <c>VkDeviceMemory</c> handle.</summary>
        public nint Memory => _memory;

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_hostImport is not null)
            {
                // The import wrapper owns lifetime — destroying the buffer +
                // freeing the memory go through its Dispose. Clear our local
                // copies so we don't double-free.
                _hostImport.Dispose();
                _buffer = 0;
                _memory = 0;
                return;
            }

            if (_buffer != 0)
            {
                VulkanApi.vkDestroyBuffer(_device._device, _buffer, 0);
                _buffer = 0;
            }
            if (_memory != 0)
            {
                VulkanApi.vkFreeMemory(_device._device, _memory, 0);
                _memory = 0;
            }
        }
    }

    /// <summary>
    /// Allocates a storage buffer of <paramref name="bytes"/> bytes backed by
    /// host-visible, host-coherent device memory. The returned buffer can be
    /// mapped directly from the host — use for activations / scratch the
    /// forward pass reads/writes from the host between kernel launches.
    /// </summary>
    public Buffer Allocate(long bytes) => AllocateInternal(bytes, deviceLocal: false);

    /// <summary>
    /// Allocates a host-visible storage buffer optimised for per-token CPU
    /// readback (e.g. the logits buffer): prefers a memory type that is also
    /// <c>HOST_CACHED</c> so host reads go through the CPU cache hierarchy.
    /// The default host-visible type on AMD/Windows is write-combined
    /// (uncached) — CPU reads from it run at &lt;1 GB/s, which cost ~0.4 ms
    /// per decoded token on a 49k-vocab logits row (issue #143). Falls back
    /// to the plain host-visible type when no cached type exists.
    /// </summary>
    public Buffer AllocateHostReadback(long bytes)
        => AllocateInternal(bytes, deviceLocal: false, preferHostCached: true);

    /// <summary>
    /// Allocates a storage buffer of <paramref name="bytes"/> bytes backed by
    /// device-local memory. The buffer is <b>not</b> host-mappable; use this
    /// for immutable weights and the KV cache, populating the contents via
    /// <see cref="UploadToDeviceLocal"/> (weights) or <c>vkCmdCopyBuffer</c>
    /// between a host-visible source and this device-local destination
    /// (KV cache update path).
    /// </summary>
    /// <remarks>
    /// On discrete GPUs this puts the data in VRAM — reads from a compute
    /// shader hit the driver's native tiled layout rather than going over
    /// PCIe / DF at host-memory bandwidth. On UMA parts (iGPU, APU) the
    /// bytes still physically sit in shared DDR, but the driver picks a
    /// swizzled storage layout that reads significantly faster from a
    /// compute shader than host-coherent linear memory. Always measure.
    /// </remarks>
    public Buffer AllocateDeviceLocal(long bytes) => AllocateInternal(bytes, deviceLocal: true);

    /// <summary>
    /// Attempts to wrap an mmap'd host pointer (e.g. a GGUF tensor's offset
    /// inside <c>GgufFile.DataBasePointer</c>) in a <see cref="Buffer"/>
    /// backed by imported host memory via <c>VK_EXT_external_memory_host</c>.
    /// Returns <c>null</c> when the device does not advertise the extension,
    /// when the alignment math can't be satisfied, or when the driver rejects
    /// the import — callers must always have a staging-copy fallback ready.
    /// </summary>
    /// <param name="hostPointer">Host pointer to the buffer's logical first byte.</param>
    /// <param name="size">Logical buffer size in bytes.</param>
    /// <returns>The wrapped buffer, or <c>null</c> when zero-copy import is not possible.</returns>
    /// <remarks>
    /// On a unified-memory APU (Strix Halo, Apple Silicon, Intel iGPU) the
    /// returned <see cref="Buffer.Handle"/> reads the same DDR pages the host
    /// mmap exposes — no staging copy, no double-counting of physical RAM.
    /// On a discrete GPU the extension is still useful (e.g. NVIDIA exposes
    /// it via PCIe BAR) but the win is smaller. <see cref="Buffer.Dispose"/>
    /// destroys the import; the underlying <c>MemoryMappedFile</c> is the
    /// caller's responsibility.
    /// </remarks>
    public Buffer? TryWrapHostVisible(nint hostPointer, long size)
    {
        var import = HostVisibleBuffer.TryCreate(this, hostPointer, size);
        return import is null ? null : new Buffer(this, import);
    }

    /// <summary>VkResult VK_ERROR_OUT_OF_DEVICE_MEMORY.</summary>
    private const int VkErrorOutOfDeviceMemory = -2;

    /// <summary>
    /// <c>DOTLLM_VULKAN_STRICT_DEVICE_LOCAL=1</c> disables the host-visible fallback
    /// on device-local allocation failure (an exhausted strict heap then throws, the
    /// pre-fallback behaviour).
    /// </summary>
    private static readonly bool s_strictDeviceLocal =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_STRICT_DEVICE_LOCAL") == "1";

    private long _deviceLocalFallbacks;

    /// <summary>
    /// Number of device-local allocations that fell back to a host-visible memory
    /// type because the strict DEVICE_LOCAL heap was exhausted. Non-zero means part
    /// of the working set lives in the slower (on discrete GPUs) or GTT (on UMA)
    /// heap — perf harnesses should report it alongside any measurement.
    /// </summary>
    public long DeviceLocalFallbackCount => Interlocked.Read(ref _deviceLocalFallbacks);

    private Buffer AllocateInternal(long bytes, bool deviceLocal, bool preferHostCached = false)
    {
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

        var bci = new VkBufferCreateInfo
        {
            sType = VkStructureType.BufferCreateInfo,
            size = (ulong)bytes,
            usage = VkBufferUsageFlags.StorageBuffer
                  | VkBufferUsageFlags.TransferSrc
                  | VkBufferUsageFlags.TransferDst,
            sharingMode = VkSharingMode.Exclusive,
        };
        VulkanApi.vkCreateBuffer(_device, bci, 0, out nint buffer)
            .ThrowOnError("vkCreateBuffer");

        VulkanApi.vkGetBufferMemoryRequirements(_device, buffer, out var req);

        VkMemoryPropertyFlags required = deviceLocal
            ? VkMemoryPropertyFlags.DeviceLocal
            : VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent;

        // On UMA drivers (AMD integrated, Intel) every memory type may expose
        // DEVICE_LOCAL + HOST_VISIBLE simultaneously. For weights we prefer a
        // strictly device-local-only type (driver is free to use a tiled /
        // swizzled layout — see AllocateDeviceLocal remarks). Fall back to a
        // DEVICE_LOCAL-that-is-also-host-visible type when the GPU only
        // exposes the combined pool (older Intel, some mobile).
        uint typeIndex;
        if (deviceLocal)
        {
            if (!TryFindMemoryType(req.memoryTypeBits,
                    required: VkMemoryPropertyFlags.DeviceLocal,
                    excluded: VkMemoryPropertyFlags.HostVisible,
                    out typeIndex))
            {
                typeIndex = FindMemoryType(req.memoryTypeBits, VkMemoryPropertyFlags.DeviceLocal);
            }
        }
        else if (preferHostCached
            && TryFindMemoryType(req.memoryTypeBits,
                required: required | VkMemoryPropertyFlags.HostCached,
                excluded: default,
                out typeIndex))
        {
            // Cached host-visible type found — CPU readback at full speed.
        }
        else
        {
            typeIndex = FindMemoryType(req.memoryTypeBits, required);
        }

        var mai = new VkMemoryAllocateInfo
        {
            sType = VkStructureType.MemoryAllocateInfo,
            allocationSize = req.size,
            memoryTypeIndex = typeIndex,
        };
        int allocResult = VulkanApi.vkAllocateMemory(_device, mai, 0, out nint memory);

        // The strict device-local heap (discrete VRAM, or the UMA carve-out — e.g. a
        // 16 GB heap[0] on Strix Halo while heap[1] exposes 96 GB of DEVICE_LOCAL +
        // HOST_VISIBLE GTT) can be far smaller than what the device can actually
        // address. When it is exhausted, retry on the combined DEVICE_LOCAL +
        // HOST_VISIBLE type, then plain host-visible — the llama.cpp
        // GGML_VK_ALLOW_SYSMEM_FALLBACK equivalent. On UMA parts the fallback reads
        // the same DRAM; on discrete GPUs it is slower than VRAM but beats an OOM
        // crash. Opt out with DOTLLM_VULKAN_STRICT_DEVICE_LOCAL=1; occurrences are
        // counted in DeviceLocalFallbackCount for harness reporting.
        if (allocResult == VkErrorOutOfDeviceMemory && deviceLocal && !s_strictDeviceLocal)
        {
            // Heap-aware: on AMD APUs the FIRST type matching a fallback flag combo can
            // sit on the same exhausted carve-out heap as the failed type, so ranking by
            // flags alone re-fails. Try every eligible type, other heaps before the
            // failed heap, larger heaps first, host-visible rungs after combined ones.
            foreach (uint fbIndex in EnumerateDeviceLocalFallbackTypes(req.memoryTypeBits, typeIndex))
            {
                mai.memoryTypeIndex = fbIndex;
                allocResult = VulkanApi.vkAllocateMemory(_device, mai, 0, out memory);
                if (allocResult >= 0)
                {
                    typeIndex = fbIndex;
                    Interlocked.Increment(ref _deviceLocalFallbacks);
                    break;
                }
            }
        }

        if (allocResult < 0)
        {
            VulkanApi.vkDestroyBuffer(_device, buffer, 0);
            allocResult.ThrowOnError("vkAllocateMemory");
        }

        int bindResult = VulkanApi.vkBindBufferMemory(_device, buffer, memory, 0);
        if (bindResult < 0)
        {
            VulkanApi.vkFreeMemory(_device, memory, 0);
            VulkanApi.vkDestroyBuffer(_device, buffer, 0);
            bindResult.ThrowOnError("vkBindBufferMemory");
        }

        // Host-visible allocations are mappable; a device-local allocation is mappable only when the
        // chosen type also carries HOST_VISIBLE (the UMA case). On a discrete GPU the strict
        // device-local type is NOT mappable, so Download/UploadToDeviceLocal must stage.
        bool hostVisible = !deviceLocal || MemoryTypeIsHostVisible(typeIndex);
        return new Buffer(this, buffer, memory, bytes, hostVisible);
    }

    private unsafe bool MemoryTypeIsHostVisible(uint typeIndex)
    {
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, out var mem);
        uint* types = (uint*)mem.memoryTypes; // 8-byte entries: u32 propertyFlags, u32 heapIndex
        var flags = (VkMemoryPropertyFlags)types[typeIndex * 2];
        return (flags & VkMemoryPropertyFlags.HostVisible) != 0;
    }

    /// <summary>
    /// Candidate memory types for the device-local OOM fallback, best first: for each
    /// rung (DEVICE_LOCAL+HOST_VISIBLE, then HOST_VISIBLE+HOST_COHERENT) every eligible
    /// type is yielded — types on a different heap than the exhausted one before types
    /// sharing it, larger heaps before smaller. On a UMA APU this walks the allocation
    /// off the small strict carve-out (e.g. 15.8 GiB on Strix Halo) onto the large
    /// GTT heap that maps the same DRAM.
    /// </summary>
    private unsafe List<uint> EnumerateDeviceLocalFallbackTypes(uint typeBits, uint failedTypeIndex)
    {
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, out var mem);
        uint* types = (uint*)mem.memoryTypes;   // 8-byte entries: u32 propertyFlags, u32 heapIndex
        byte* heaps = (byte*)mem.memoryHeaps;   // 16-byte entries: u64 size, u32 flags, padding
        uint failedHeap = types[failedTypeIndex * 2 + 1];

        var ordered = new List<uint>(8);
        Span<VkMemoryPropertyFlags> rungs =
        [
            VkMemoryPropertyFlags.DeviceLocal | VkMemoryPropertyFlags.HostVisible,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent,
        ];
        foreach (var required in rungs)
        {
            // Two passes per rung: other-heap types first, failed-heap types last.
            for (int pass = 0; pass < 2; pass++)
            {
                var passList = new List<(uint Index, ulong HeapSize)>(4);
                for (uint i = 0; i < mem.memoryTypeCount; i++)
                {
                    if ((typeBits & (1u << (int)i)) == 0 || i == failedTypeIndex) continue;
                    var flags = (VkMemoryPropertyFlags)types[i * 2];
                    if ((flags & required) != required) continue;
                    uint heapIdx = types[i * 2 + 1];
                    bool otherHeap = heapIdx != failedHeap;
                    if (otherHeap != (pass == 0)) continue;
                    passList.Add((i, *(ulong*)(heaps + heapIdx * 16)));
                }
                passList.Sort(static (a, b) => b.HeapSize.CompareTo(a.HeapSize));
                foreach (var (idx, _) in passList)
                    if (!ordered.Contains(idx))
                        ordered.Add(idx);
            }
        }
        return ordered;
    }

    /// <summary>
    /// Size in bytes of the largest <c>DEVICE_LOCAL</c> memory heap — the effective VRAM budget for
    /// weight/KV/compute allocations. On a discrete GPU this is the dedicated VRAM; on a UMA part
    /// (Strix Halo iGPU, Intel Arc integrated) it is the driver-reported device-local carve-out of
    /// system RAM, which is typically far smaller than total system memory and is the real ceiling
    /// for how many model layers that device can hold when spanning.
    /// </summary>
    public unsafe long DeviceLocalHeapBytes()
    {
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, out var mem);
        byte* heaps = (byte*)mem.memoryHeaps; // 16-byte entries: u64 size, u32 flags, padding
        long max = 0;
        for (uint i = 0; i < mem.memoryHeapCount; i++)
        {
            ulong size = *(ulong*)(heaps + i * 16);
            uint flags = *(uint*)(heaps + i * 16 + 8);
            if ((flags & (uint)VkMemoryHeapFlags.DeviceLocal) != 0)
                max = Math.Max(max, (long)size);
        }
        return max;
    }

    /// <summary>
    /// Copies <paramref name="source"/> bytes from host memory into
    /// <paramref name="dst"/> (which may be device-local, i.e. not
    /// host-mappable) via an intermediate <paramref name="staging"/> buffer.
    /// Records a <c>vkCmdCopyBuffer</c> on a transient command buffer and
    /// waits on a fence. <paramref name="staging"/> must be host-visible
    /// host-coherent and at least <paramref name="source"/>.Length bytes.
    /// </summary>
    /// <remarks>
    /// This is the weight-upload path. Callers pre-allocate one staging
    /// buffer sized for the largest single weight row/matrix and reuse it
    /// across all <c>vkCmdCopyBuffer</c> uploads — saves the per-upload
    /// <c>vkAllocateMemory</c>/<c>vkCreateBuffer</c> cost that would dominate
    /// at 30 layers × 7 matrices.
    /// </remarks>
    public unsafe void UploadToDeviceLocal(ReadOnlySpan<byte> source, Buffer staging, Buffer dst)
    {
        if (source.Length > staging.Size)
            throw new ArgumentException("Staging buffer too small.", nameof(staging));
        if (source.Length > dst.Size)
            throw new ArgumentException("Destination buffer too small.", nameof(dst));

        // 1. Copy host → staging.
        VulkanApi.vkMapMemory(_device, staging.Memory, 0, (ulong)source.Length, 0, out nint mapped)
            .ThrowOnError("vkMapMemory staging");
        try
        {
            source.CopyTo(new Span<byte>((void*)mapped, source.Length));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, staging.Memory);
        }

        // 2. Record + submit staging → dst copy, wait on fence.
        CopyBufferSynchronous(staging, dst, (ulong)source.Length);
    }

    /// <summary>
    /// Records a one-shot <c>vkCmdCopyBuffer</c> from offset 0 of
    /// <paramref name="src"/> to offset 0 of <paramref name="dst"/> and waits
    /// for it on a fence. Used by the device-local weight-upload path.
    /// </summary>
    public void CopyBufferSynchronous(Buffer src, Buffer dst, ulong size)
        => CopyBufferRangeSynchronous(src, dst, srcOffset: 0, dstOffset: 0, size: size);

    /// <summary>
    /// Records a one-shot <c>vkCmdCopyBuffer</c> between arbitrary offsets
    /// and waits for it on a fence. Used by the synchronous KV-cache update
    /// path (the fence-pipelined path uses <c>vkCmdCopyBuffer</c> directly
    /// against the forward pass's shared command buffer).
    /// </summary>
    public unsafe void CopyBufferRangeSynchronous(Buffer src, Buffer dst, ulong srcOffset, ulong dstOffset, ulong size)
    {
        var cbai = new VkCommandBufferAllocateInfo
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _commandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VulkanApi.vkAllocateCommandBuffers(_device, cbai, out nint cmdBuf)
            .ThrowOnError("vkAllocateCommandBuffers CopyBufferRangeSynchronous");

        var fenceCi = new VkFenceCreateInfo { sType = VkStructureType.FenceCreateInfo };
        VulkanApi.vkCreateFence(_device, fenceCi, 0, out nint fence)
            .ThrowOnError("vkCreateFence CopyBufferRangeSynchronous");

        try
        {
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer CopyBufferRangeSynchronous");

            var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);

            VulkanApi.vkEndCommandBuffer(cmdBuf).ThrowOnError("vkEndCommandBuffer CopyBufferRangeSynchronous");

            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBuf),
            };
            VulkanApi.vkQueueSubmit(_queue, 1, submit, fence).ThrowOnError("vkQueueSubmit CopyBufferRangeSynchronous");

            nint fenceLocal = fence;
            VulkanApi.vkWaitForFences(_device, 1, fenceLocal, waitAll: 1, ulong.MaxValue)
                .ThrowOnError("vkWaitForFences CopyBufferRangeSynchronous");
        }
        finally
        {
            VulkanApi.vkDestroyFence(_device, fence, 0);
            VulkanApi.vkFreeCommandBuffers(_device, _commandPool, 1, cmdBuf);
        }
    }

    /// <summary>
    /// Picks a memory type index for an imported host pointer. The candidate
    /// <paramref name="typeBits"/> is the intersection of
    /// <c>vkGetBufferMemoryRequirements.memoryTypeBits</c> and
    /// <c>VkMemoryHostPointerPropertiesEXT.memoryTypeBits</c>; both filters
    /// have already been applied by the caller. We additionally prefer a
    /// type that is HOST_VISIBLE (so the host mmap can still be read/written
    /// after import) over one that isn't, but accept either since the driver
    /// is the authority on what's compatible.
    /// </summary>
    internal unsafe bool TryFindHostImportMemoryType(uint typeBits, out uint memoryTypeIndex)
    {
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, out var mem);
        uint* types = (uint*)mem.memoryTypes;

        uint fallbackIdx = uint.MaxValue;
        for (uint i = 0; i < mem.memoryTypeCount; i++)
        {
            if ((typeBits & (1u << (int)i)) == 0) continue;
            var flags = (VkMemoryPropertyFlags)types[i * 2];
            if ((flags & VkMemoryPropertyFlags.HostVisible) != 0)
            {
                memoryTypeIndex = i;
                return true;
            }
            if (fallbackIdx == uint.MaxValue) fallbackIdx = i;
        }

        if (fallbackIdx != uint.MaxValue)
        {
            memoryTypeIndex = fallbackIdx;
            return true;
        }

        memoryTypeIndex = 0;
        return false;
    }

    private unsafe uint FindMemoryType(uint typeBits, VkMemoryPropertyFlags required)
    {
        if (TryFindMemoryType(typeBits, required, excluded: default, out uint idx))
            return idx;
        throw new VulkanException(-3,
            $"No memory type satisfies typeBits=0x{typeBits:X8} and flags={required}.");
    }

    private unsafe bool TryFindMemoryType(
        uint typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags excluded,
        out uint memoryTypeIndex)
    {
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, out var mem);
        // memoryTypes is an array of 8-byte entries: u32 propertyFlags, u32 heapIndex.
        uint* types = (uint*)mem.memoryTypes;
        for (uint i = 0; i < mem.memoryTypeCount; i++)
        {
            if ((typeBits & (1u << (int)i)) == 0) continue;
            var flags = (VkMemoryPropertyFlags)types[i * 2];
            if ((flags & required) != required) continue;
            if (excluded != default && (flags & excluded) != 0) continue;
            memoryTypeIndex = i;
            return true;
        }
        memoryTypeIndex = 0;
        return false;
    }

    /// <summary>Copies <paramref name="source"/> from host memory into the start of <paramref name="dst"/>.</summary>
    public unsafe void Upload(ReadOnlySpan<float> source, Buffer dst)
    {
        long bytes = (long)source.Length * sizeof(float);
        if (bytes > dst.Size)
            throw new ArgumentException("Source larger than destination buffer.", nameof(source));

        VulkanApi.vkMapMemory(_device, dst.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory");
        try
        {
            var destSpan = new Span<float>((void*)mapped, source.Length);
            source.CopyTo(destSpan);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, dst.Memory);
        }
    }

    /// <summary>
    /// Copies raw <paramref name="source"/> bytes from host memory into the start of <paramref name="dst"/>.
    /// Used for quantized weight blobs (Q8_0, Q4_K, etc.) where the GPU sees the
    /// data as <c>uint[]</c> and the shader extracts bytes.
    /// </summary>
    public unsafe void Upload(ReadOnlySpan<byte> source, Buffer dst)
    {
        if (source.Length > dst.Size)
            throw new ArgumentException("Source larger than destination buffer.", nameof(source));

        VulkanApi.vkMapMemory(_device, dst.Memory, 0, (ulong)source.Length, 0, out nint mapped)
            .ThrowOnError("vkMapMemory");
        try
        {
            var destSpan = new Span<byte>((void*)mapped, source.Length);
            source.CopyTo(destSpan);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, dst.Memory);
        }
    }

    /// <summary>Copies from the start of <paramref name="src"/> into <paramref name="destination"/> host memory.</summary>
    /// <remarks>
    /// When <paramref name="src"/> is device-local-only (a discrete GPU's VRAM — see
    /// <see cref="Buffer.IsHostVisible"/>) the host cannot map it, so the bytes are first copied into a
    /// transient host-visible staging buffer via <c>vkCmdCopyBuffer</c> and that is mapped instead. On
    /// UMA parts the device-local memory is host-visible, so it is mapped directly (no copy). This is the
    /// readback mirror of <see cref="UploadToDeviceLocal"/>; the bug it fixes only surfaces on a discrete
    /// GPU (e.g. the cross-device KV handoff reading a prefill cache out of an RTX 3060's VRAM).
    /// </remarks>
    public unsafe void Download(Buffer src, Span<float> destination)
    {
        if (destination.IsEmpty) return;
        long bytes = (long)destination.Length * sizeof(float);
        if (bytes > src.Size)
            throw new ArgumentException("Destination larger than source buffer.", nameof(destination));

        if (!src.IsHostVisible)
        {
            // Device-local-only source: stage device → host-visible buffer, then map the staging copy.
            using Buffer staging = Allocate(bytes);
            CopyBufferRangeSynchronous(src, staging, srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
            DownloadHostVisible(staging, destination, bytes);
            return;
        }

        DownloadHostVisible(src, destination, bytes);
    }

    private unsafe void DownloadHostVisible(Buffer src, Span<float> destination, long bytes)
    {
        VulkanApi.vkMapMemory(_device, src.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory");
        try
        {
            var srcSpan = new ReadOnlySpan<float>((void*)mapped, destination.Length);
            srcSpan.CopyTo(destination);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, src.Memory);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_device != 0)
        {
            VulkanApi.vkDeviceWaitIdle(_device);
        }
        if (_commandPool != 0)
        {
            VulkanApi.vkDestroyCommandPool(_device, _commandPool, 0);
            _commandPool = 0;
        }
        if (_device != 0)
        {
            VulkanApi.vkDestroyDevice(_device, 0);
            _device = 0;
        }
        if (_instance != 0)
        {
            VulkanApi.vkDestroyInstance(_instance, 0);
            _instance = 0;
        }
    }

    // ────────────────────────────────────────────────────────────────
    // External semaphores (cross-API sync with CUDA — M3 async handoff)
    // ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a binary <c>VkSemaphore</c> that may be exported as a Win32 HANDLE
    /// of <paramref name="handleType"/> (OPAQUE_WIN32 or D3D12_FENCE) for import
    /// into CUDA. Requires <see cref="HasExternalSemaphoreWin32"/>; the caller
    /// owns the returned handle and must release it via <see cref="DestroySemaphore"/>.
    /// </summary>
    /// <param name="handleType">The external handle type the semaphore will be exported as.</param>
    /// <returns>The created <c>VkSemaphore</c> handle.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the device did not enable the external-semaphore-win32 extension.</exception>
    public unsafe nint CreateExportableSemaphore(
        ExternalSemaphoreHandleType handleType = ExternalSemaphoreHandleType.OpaqueWin32)
    {
        if (!HasExternalSemaphoreWin32)
            throw new InvalidOperationException(
                "Device does not support VK_KHR_external_semaphore_win32 — cannot export a semaphore to CUDA.");

        var exportInfo = new VkExportSemaphoreCreateInfo
        {
            sType = VkStructureType.ExportSemaphoreCreateInfo,
            handleTypes = (uint)ToVkHandleType(handleType),
        };

        var ci = new VkSemaphoreCreateInfo
        {
            sType = VkStructureType.SemaphoreCreateInfo,
            pNext = (nint)(&exportInfo),
        };

        VulkanApi.vkCreateSemaphore(_device, ci, 0, out nint semaphore)
            .ThrowOnError("vkCreateSemaphore (exportable)");
        return semaphore;
    }

    /// <summary>
    /// Exports the Win32 HANDLE for an exportable semaphore created by
    /// <see cref="CreateExportableSemaphore"/>. The returned HANDLE is owned by
    /// the caller; once CUDA has imported it (CUDA duplicates the handle on
    /// import for OPAQUE_WIN32), the caller must <c>CloseHandle</c> this copy.
    /// </summary>
    /// <param name="semaphore">The exportable semaphore handle.</param>
    /// <param name="handleType">Must match the type the semaphore was created with.</param>
    /// <returns>The Win32 NT HANDLE referencing the semaphore.</returns>
    public unsafe nint GetSemaphoreWin32Handle(
        nint semaphore,
        ExternalSemaphoreHandleType handleType = ExternalSemaphoreHandleType.OpaqueWin32)
    {
        nint fn = VulkanApi.vkGetDeviceProcAddr(_device, "vkGetSemaphoreWin32HandleKHR");
        if (fn == 0)
            throw new VulkanException(-3,
                "vkGetSemaphoreWin32HandleKHR not resolvable — extension not enabled at device create.");

        var getInfo = new VkSemaphoreGetWin32HandleInfoKhr
        {
            sType = VkStructureType.SemaphoreGetWin32HandleInfoKhr,
            semaphore = semaphore,
            handleType = (uint)ToVkHandleType(handleType),
        };

        var getHandle = Marshal.GetDelegateForFunctionPointer<VkGetSemaphoreWin32HandleKHR>(fn);
        getHandle(_device, getInfo, out nint handle).ThrowOnError("vkGetSemaphoreWin32HandleKHR");
        return handle;
    }

    /// <summary>
    /// Creates an exportable <b>timeline</b> <c>VkSemaphore</c> for the
    /// cross-vendor M3 handoff. A timeline semaphore is required for the
    /// <see cref="ExternalSemaphoreHandleType.D3D12Fence"/> handle type, which is
    /// the form CUDA can import from an Intel-Arc Vulkan device (the
    /// <see cref="ExternalSemaphoreHandleType.OpaqueWin32"/> binary form fails
    /// <c>cuImportExternalSemaphore</c> on the Arc→NVIDIA pairing). The semaphore
    /// starts at counter value <paramref name="initialValue"/> and is advanced by
    /// each <see cref="SubmitContext.SubmitAndSignalTimeline"/> call.
    /// </summary>
    /// <param name="handleType">External handle type — D3D12_FENCE for the working cross-vendor path.</param>
    /// <param name="initialValue">Initial timeline counter value (typically 0).</param>
    /// <returns>The created timeline <c>VkSemaphore</c> handle.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the device did not enable the external-semaphore-win32 extension.</exception>
    public unsafe nint CreateExportableTimelineSemaphore(
        ExternalSemaphoreHandleType handleType = ExternalSemaphoreHandleType.D3D12Fence,
        ulong initialValue = 0)
    {
        if (!HasExternalSemaphoreWin32)
            throw new InvalidOperationException(
                "Device does not support VK_KHR_external_semaphore_win32 — cannot export a semaphore to CUDA.");

        // pNext chain: VkSemaphoreCreateInfo -> VkExportSemaphoreCreateInfo -> VkSemaphoreTypeCreateInfo.
        var typeInfo = new VkSemaphoreTypeCreateInfo
        {
            sType = VkStructureType.SemaphoreTypeCreateInfo,
            semaphoreType = VkSemaphoreType.Timeline,
            initialValue = initialValue,
        };

        var exportInfo = new VkExportSemaphoreCreateInfo
        {
            sType = VkStructureType.ExportSemaphoreCreateInfo,
            pNext = (nint)(&typeInfo),
            handleTypes = (uint)ToVkHandleType(handleType),
        };

        var ci = new VkSemaphoreCreateInfo
        {
            sType = VkStructureType.SemaphoreCreateInfo,
            pNext = (nint)(&exportInfo),
        };

        VulkanApi.vkCreateSemaphore(_device, ci, 0, out nint semaphore)
            .ThrowOnError("vkCreateSemaphore (exportable timeline)");
        return semaphore;
    }

    /// <summary>Destroys a semaphore previously created by <see cref="CreateExportableSemaphore"/>.</summary>
    /// <param name="semaphore">The semaphore handle; no-op when zero.</param>
    public void DestroySemaphore(nint semaphore)
    {
        if (semaphore != 0)
            VulkanApi.vkDestroySemaphore(_device, semaphore, 0);
    }

    // Maps the public handle-type enum to the internal interop flag bits.
    private static VkExternalSemaphoreHandleTypeFlags ToVkHandleType(ExternalSemaphoreHandleType t) => t switch
    {
        ExternalSemaphoreHandleType.OpaqueWin32 => VkExternalSemaphoreHandleTypeFlags.OpaqueWin32,
        ExternalSemaphoreHandleType.D3D12Fence => VkExternalSemaphoreHandleTypeFlags.D3D12Fence,
        _ => throw new ArgumentOutOfRangeException(nameof(t), t, "Unsupported external semaphore handle type."),
    };

    // ────────────────────────────────────────────────────────────────
    // Forward-pass command submission
    // ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Hazard-scoped barrier tracker armed for the recording currently in
    /// progress on this device (issue #144), or <c>null</c> when the legacy
    /// blanket-barrier scheme is in effect. Armed by
    /// <see cref="VulkanTransformerModel"/> right after
    /// <see cref="SubmitContext.Begin"/> on its tracked forward path;
    /// disarmed automatically by every <see cref="SubmitContext"/>
    /// begin/submit so an aborted recording can never leak tracking into an
    /// unrelated model's command buffer. Recording is single-threaded per
    /// device, so a plain field suffices.
    /// </summary>
    internal VulkanHazardTracker? ActiveHazards;

    /// <summary>
    /// Reusable command-buffer + fence pair used by the fence-pipelined
    /// forward pass. One instance per <see cref="VulkanTransformerModel"/>;
    /// <see cref="Begin"/> resets and opens the buffer, <see cref="SubmitAndWait"/>
    /// submits and waits on the fence, leaving both ready for the next forward.
    /// </summary>
    public sealed class SubmitContext : IDisposable
    {
        private readonly VulkanDevice _device;
        private nint _cmdBuf;
        private nint _fence;
        private bool _disposed;

        /// <summary>Underlying command buffer. Valid between <see cref="Begin"/> and <see cref="SubmitAndWait"/>.</summary>
        public nint CommandBuffer => _cmdBuf;

        internal SubmitContext(VulkanDevice device, nint cmdBuf, nint fence)
        {
            _device = device;
            _cmdBuf = cmdBuf;
            _fence = fence;
        }

        /// <summary>
        /// Resets the command buffer (and the fence) and opens the buffer for
        /// recording. Call once at the start of each forward pass.
        /// </summary>
        public void Begin()
        {
            // A fresh recording always starts untracked; the model re-arms
            // the hazard tracker for its tracked path after Begin returns.
            _device.ActiveHazards = null;
            _splitThisForward = false;
            VulkanApi.vkResetCommandBuffer(_cmdBuf, 0).ThrowOnError("vkResetCommandBuffer");
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(_cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer");
        }

        /// <summary>
        /// Ends the command buffer, submits on the queue, waits on the fence,
        /// resets the fence for reuse. Call once at the end of each forward
        /// pass.
        /// </summary>
        public unsafe void SubmitAndWait()
        {
            _device.ActiveHazards = null;
            VulkanApi.vkEndCommandBuffer(_cmdBuf).ThrowOnError("vkEndCommandBuffer");

            nint cmdBufLocal = _cmdBuf;
            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBufLocal),
            };
            VulkanApi.vkQueueSubmit(_device._queue, 1, submit, _fence).ThrowOnError("vkQueueSubmit SubmitContext");

            nint fenceLocal = _fence;
            VulkanApi.vkWaitForFences(_device._device, 1, fenceLocal, waitAll: 1, ulong.MaxValue)
                .ThrowOnError("vkWaitForFences SubmitContext");
            VulkanApi.vkResetFences(_device._device, 1, fenceLocal).ThrowOnError("vkResetFences SubmitContext");
        }

        // Lazily-allocated second command buffer for SplitSubmit. At most ONE
        // split per Begin/SubmitAndWait cycle: with two buffers, a second split
        // would reset a buffer submitted earlier in the SAME forward (no fence
        // yet) — guarded below.
        private nint _cmdBufAlt;
        private bool _splitThisForward;

        /// <summary>
        /// Mid-forward split: ends and submits everything recorded so far
        /// WITHOUT a fence or host wait, then re-opens recording on a second
        /// command buffer. The GPU starts executing the first chunk while the
        /// host keeps recording — llama.cpp's chunked-submit overlap (issue
        /// #143: hides most of the ~0.2-0.3 ms/token host recording cost on
        /// small models). Queue-timeline pipeline barriers already recorded
        /// (and recorded later) synchronize across the submit boundary, so the
        /// dependency chain is unchanged — results are bit-identical.
        /// At most one split per forward; the final <see cref="SubmitAndWait"/>
        /// fence covers both submissions (same-queue ordering), so both
        /// buffers are idle by the next <see cref="Begin"/>.
        /// </summary>
        public unsafe void SplitSubmit()
        {
            if (_splitThisForward)
                throw new InvalidOperationException(
                    "SplitSubmit may be called at most once per forward (two-buffer ring).");
            _splitThisForward = true;

            VulkanApi.vkEndCommandBuffer(_cmdBuf).ThrowOnError("vkEndCommandBuffer SplitSubmit");
            nint cmdBufLocal = _cmdBuf;
            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBufLocal),
            };
            VulkanApi.vkQueueSubmit(_device._queue, 1, submit, fence: 0)
                .ThrowOnError("vkQueueSubmit SplitSubmit");

            if (_cmdBufAlt == 0)
            {
                var cbai = new VkCommandBufferAllocateInfo
                {
                    sType = VkStructureType.CommandBufferAllocateInfo,
                    commandPool = _device._commandPool,
                    level = VkCommandBufferLevel.Primary,
                    commandBufferCount = 1,
                };
                VulkanApi.vkAllocateCommandBuffers(_device._device, cbai, out _cmdBufAlt)
                    .ThrowOnError("vkAllocateCommandBuffers SplitSubmit");
            }

            (_cmdBuf, _cmdBufAlt) = (_cmdBufAlt, _cmdBuf);
            VulkanApi.vkResetCommandBuffer(_cmdBuf, 0).ThrowOnError("vkResetCommandBuffer SplitSubmit");
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(_cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer SplitSubmit");

            // Keep the hazard tracker (issue #144) pointed at the live buffer.
            // Epoch state carries across the chunk boundary: pipeline barriers
            // synchronize prior command buffers in same-queue submission order.
            _device.ActiveHazards?.OnCommandBufferSwitch(_cmdBuf);
        }

        /// <summary>
        /// Ends the command buffer and submits it on the queue with
        /// <paramref name="signalSemaphore"/> in <c>pSignalSemaphores</c>, so the
        /// semaphore signals once the GPU finishes this submission. Does NOT wait
        /// on the host fence — the consumer (CUDA, via
        /// <c>cuWaitExternalSemaphoresAsync</c>) gates on the semaphore instead.
        /// This is the M3 async-handoff submit: it removes the host fence-wait
        /// stall that serialized the M2 path.
        /// </summary>
        /// <param name="signalSemaphore">An exportable semaphore the GPU signals on completion.</param>
        /// <remarks>
        /// The fence is still passed to <c>vkQueueSubmit</c> so the host can
        /// reclaim the command buffer on the next <see cref="Begin"/> (which
        /// resets it). Callers that reuse the buffer across overlapping steps
        /// must double-buffer the <see cref="SubmitContext"/> — a single binary
        /// semaphore + single command buffer cannot overlap two in-flight
        /// submissions.
        /// </remarks>
        public unsafe void SubmitAndSignal(nint signalSemaphore)
        {
            _device.ActiveHazards = null;
            VulkanApi.vkEndCommandBuffer(_cmdBuf).ThrowOnError("vkEndCommandBuffer SubmitAndSignal");

            nint cmdBufLocal = _cmdBuf;
            nint semLocal = signalSemaphore;
            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBufLocal),
                signalSemaphoreCount = 1,
                pSignalSemaphores = (nint)(&semLocal),
            };
            VulkanApi.vkQueueSubmit(_device._queue, 1, submit, _fence)
                .ThrowOnError("vkQueueSubmit SubmitAndSignal");
        }

        /// <summary>
        /// Ends the command buffer and submits it with a <b>timeline</b> signal:
        /// <paramref name="signalSemaphore"/> is advanced to
        /// <paramref name="signalValue"/> once the GPU finishes this submission.
        /// CUDA gates on the same (semaphore, value) pair via
        /// <c>cuWaitExternalSemaphoresAsync</c>. Does NOT host-wait on the fence —
        /// this is the M3 async-handoff submit for the cross-vendor D3D12_FENCE path.
        /// </summary>
        /// <param name="signalSemaphore">An exportable timeline semaphore.</param>
        /// <param name="signalValue">The counter value to signal on GPU completion (must exceed the prior signalled value).</param>
        /// <remarks>
        /// The fence is still passed so the host can reclaim the command buffer on
        /// the next <see cref="Begin"/>. Overlapping in-flight submissions require a
        /// double-buffered <see cref="SubmitContext"/> ring — one command buffer +
        /// fence cannot host two simultaneous submissions even with a timeline
        /// semaphore (the fence and command buffer are still single-use-at-a-time).
        /// </remarks>
        public unsafe void SubmitAndSignalTimeline(nint signalSemaphore, ulong signalValue)
        {
            _device.ActiveHazards = null;
            VulkanApi.vkEndCommandBuffer(_cmdBuf).ThrowOnError("vkEndCommandBuffer SubmitAndSignalTimeline");

            nint cmdBufLocal = _cmdBuf;
            nint semLocal = signalSemaphore;
            ulong signalValueLocal = signalValue;

            var timelineInfo = new VkTimelineSemaphoreSubmitInfo
            {
                sType = VkStructureType.TimelineSemaphoreSubmitInfo,
                signalSemaphoreValueCount = 1,
                pSignalSemaphoreValues = (nint)(&signalValueLocal),
            };

            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                pNext = (nint)(&timelineInfo),
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBufLocal),
                signalSemaphoreCount = 1,
                pSignalSemaphores = (nint)(&semLocal),
            };
            VulkanApi.vkQueueSubmit(_device._queue, 1, submit, _fence)
                .ThrowOnError("vkQueueSubmit SubmitAndSignalTimeline");
        }

        /// <summary>
        /// Host-waits on the fence from a prior <see cref="SubmitAndSignal"/> and
        /// resets it for reuse. Call before the next <see cref="Begin"/> when the
        /// previous submission did not host-wait, to guarantee the command buffer
        /// is no longer in flight before it is reset.
        /// </summary>
        public unsafe void WaitFence()
        {
            nint fenceLocal = _fence;
            VulkanApi.vkWaitForFences(_device._device, 1, fenceLocal, waitAll: 1, ulong.MaxValue)
                .ThrowOnError("vkWaitForFences WaitFence");
            VulkanApi.vkResetFences(_device._device, 1, fenceLocal).ThrowOnError("vkResetFences WaitFence");
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_fence != 0)
            {
                VulkanApi.vkDestroyFence(_device._device, _fence, 0);
                _fence = 0;
            }
            if (_cmdBuf != 0)
            {
                nint local = _cmdBuf;
                VulkanApi.vkFreeCommandBuffers(_device._device, _device._commandPool, 1, local);
                _cmdBuf = 0;
            }
            if (_cmdBufAlt != 0)
            {
                nint local = _cmdBufAlt;
                VulkanApi.vkFreeCommandBuffers(_device._device, _device._commandPool, 1, local);
                _cmdBufAlt = 0;
            }
        }
    }

    /// <summary>
    /// Allocates one command buffer and one fence bound to the compute
    /// queue. The returned <see cref="SubmitContext"/> is intended to live
    /// for the lifetime of the caller (e.g. <see cref="VulkanTransformerModel"/>)
    /// and be reused <see cref="SubmitContext.Begin"/>-&gt;record-&gt;
    /// <see cref="SubmitContext.SubmitAndWait"/> once per forward pass.
    /// </summary>
    public SubmitContext CreateSubmitContext()
    {
        var cbai = new VkCommandBufferAllocateInfo
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _commandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VulkanApi.vkAllocateCommandBuffers(_device, cbai, out nint cmdBuf)
            .ThrowOnError("vkAllocateCommandBuffers CreateSubmitContext");

        var fenceCi = new VkFenceCreateInfo { sType = VkStructureType.FenceCreateInfo };
        int r = VulkanApi.vkCreateFence(_device, fenceCi, 0, out nint fence);
        if (r < 0)
        {
            nint local = cmdBuf;
            VulkanApi.vkFreeCommandBuffers(_device, _commandPool, 1, local);
            r.ThrowOnError("vkCreateFence CreateSubmitContext");
        }

        return new SubmitContext(this, cmdBuf, fence);
    }
}

/// <summary>
/// External-semaphore handle type for the Vulkan→CUDA M3 async handoff. Selects
/// which Win32 NT-handle flavour the exportable semaphore is created and exported
/// as; the same type must be passed to CUDA's <c>cuImportExternalSemaphore</c>.
/// </summary>
public enum ExternalSemaphoreHandleType
{
    /// <summary>
    /// Opaque Win32 NT handle (<c>VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT</c>).
    /// Works when the same handle semantics are understood by both APIs on Windows.
    /// </summary>
    OpaqueWin32,

    /// <summary>
    /// Direct3D 12 fence handle (<c>VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT</c>).
    /// The cross-vendor-portable Win32 fence both Intel Vulkan and NVIDIA CUDA accept.
    /// </summary>
    D3D12Fence,
}

/// <summary>
/// One cooperative-matrix tile shape reported by the Vulkan driver, mirroring
/// <c>VkCooperativeMatrixPropertiesKHR</c>. Component types are
/// <c>VkComponentTypeKHR</c> values (0=Float16, 1=Float32, 3=Sint8, 5=Sint32,
/// etc.); scope is a <c>VkScopeKHR</c> value (3=Subgroup).
/// </summary>
/// <param name="MSize">First tile dimension (rows of the A and C/Result tiles).</param>
/// <param name="NSize">Second tile dimension (cols of the B and C/Result tiles).</param>
/// <param name="KSize">Inner tile dimension (cols of A / rows of B).</param>
/// <param name="AType">Component type of the A (MatrixUseA) operand.</param>
/// <param name="BType">Component type of the B (MatrixUseB) operand.</param>
/// <param name="CType">Component type of the C (MatrixUseAccumulator input) operand.</param>
/// <param name="ResultType">Component type of the accumulator output.</param>
/// <param name="Scope">VkScopeKHR value — on KHR (non-NV) always Subgroup.</param>
public readonly record struct CooperativeMatrixShape(
    int MSize, int NSize, int KSize,
    int AType, int BType, int CType, int ResultType,
    int Scope);
