using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.Interop;

// Vulkan uses int32 "structure type" tags (sType) on every struct to permit
// forward extension. Only the tags we actually use are listed here.
internal static class VkStructureType
{
    internal const int ApplicationInfo = 0;
    internal const int InstanceCreateInfo = 1;
    internal const int DeviceQueueCreateInfo = 2;
    internal const int DeviceCreateInfo = 3;
    internal const int SubmitInfo = 4;
    internal const int MemoryAllocateInfo = 5;
    internal const int MappedMemoryRange = 6;
    internal const int BindSparseInfo = 7;
    internal const int FenceCreateInfo = 8;
    internal const int BufferCreateInfo = 12;
    internal const int ShaderModuleCreateInfo = 16;
    internal const int PipelineLayoutCreateInfo = 30;
    internal const int ComputePipelineCreateInfo = 29;
    internal const int PipelineShaderStageCreateInfo = 18;
    internal const int DescriptorSetLayoutCreateInfo = 32;
    internal const int DescriptorPoolCreateInfo = 33;
    internal const int DescriptorSetAllocateInfo = 34;
    internal const int WriteDescriptorSet = 35;
    internal const int CommandPoolCreateInfo = 39;
    internal const int CommandBufferAllocateInfo = 40;
    internal const int CommandBufferBeginInfo = 42;
    internal const int MemoryBarrier = 46;
    // Vulkan 1.1 core structure types (used by vkGetPhysicalDeviceProperties2).
    internal const int PhysicalDeviceProperties2 = 1000059001;
    internal const int PhysicalDeviceFeatures2 = 1000059000;
    internal const int PhysicalDeviceSubgroupProperties = 1000094000;
    // VK_KHR_cooperative_matrix extension structures.
    internal const int PhysicalDeviceCooperativeMatrixFeaturesKhr = 1000506000;
    internal const int CooperativeMatrixPropertiesKhr = 1000506001;
    internal const int PhysicalDeviceCooperativeMatrixPropertiesKhr = 1000506002;
}

// VkComponentTypeKHR — component type of an element in a cooperative matrix.
// Values from the VK_KHR_cooperative_matrix specification.
internal static class VkComponentTypeKhr
{
    internal const int Float16 = 0;
    internal const int Float32 = 1;
    internal const int Float64 = 2;
    internal const int Sint8   = 3;
    internal const int Sint16  = 4;
    internal const int Sint32  = 5;
    internal const int Sint64  = 6;
    internal const int Uint8   = 7;
    internal const int Uint16  = 8;
    internal const int Uint32  = 9;
    internal const int Uint64  = 10;
}

// VkScopeKHR — scope at which a cooperative matrix is allocated. For
// VK_KHR_cooperative_matrix (KHR, not NV) only Subgroup scope is standardised.
internal static class VkScopeKhr
{
    internal const int Device     = 1;
    internal const int Workgroup  = 2;
    internal const int Subgroup   = 3;
    internal const int QueueFamily = 5;
}

// VkSubgroupFeatureFlagBits — capabilities advertised by the driver for a given
// subgroup size. ARITHMETIC is the one we care about (subgroupAdd, subgroupMax,
// subgroupMin, etc.). Others listed for reference; matching the spec bit values.
[Flags]
internal enum VkSubgroupFeatureFlags : uint
{
    Basic        = 0x00000001,
    Vote         = 0x00000002,
    Arithmetic   = 0x00000004,
    Ballot       = 0x00000008,
    Shuffle      = 0x00000010,
    ShuffleRelative = 0x00000020,
    Clustered    = 0x00000040,
    Quad         = 0x00000080,
}

// VkPhysicalDeviceType (chosen enum values)
internal static class VkPhysicalDeviceType
{
    internal const int Other = 0;
    internal const int IntegratedGpu = 1;
    internal const int DiscreteGpu = 2;
    internal const int VirtualGpu = 3;
    internal const int Cpu = 4;
}

// VkBufferUsageFlagBits (bitflags)
[Flags]
internal enum VkBufferUsageFlags : uint
{
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    StorageBuffer = 0x00000020,
}

// VkMemoryPropertyFlagBits (bitflags)
[Flags]
internal enum VkMemoryPropertyFlags : uint
{
    DeviceLocal = 0x00000001,
    HostVisible = 0x00000002,
    HostCoherent = 0x00000004,
    HostCached = 0x00000008,
}

[Flags]
internal enum VkMemoryHeapFlags : uint
{
    DeviceLocal = 0x00000001,
}

[Flags]
internal enum VkQueueFlags : uint
{
    Graphics = 0x00000001,
    Compute = 0x00000002,
    Transfer = 0x00000004,
    SparseBinding = 0x00000008,
}

internal static class VkDescriptorType
{
    internal const int StorageBuffer = 7;
}

internal static class VkShaderStageFlags
{
    internal const uint Compute = 0x00000020;
}

internal static class VkCommandPoolCreateFlags
{
    internal const uint ResetCommandBuffer = 0x00000002;
}

internal static class VkCommandBufferLevel
{
    internal const int Primary = 0;
}

internal static class VkCommandBufferUsageFlags
{
    internal const uint OneTimeSubmit = 0x00000001;
}

internal static class VkSharingMode
{
    internal const int Exclusive = 0;
}

internal static class VkPipelineBindPoint
{
    internal const int Compute = 1;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkApplicationInfo
{
    internal int sType;
    internal nint pNext;
    internal nint pApplicationName;
    internal uint applicationVersion;
    internal nint pEngineName;
    internal uint engineVersion;
    internal uint apiVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkInstanceCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal nint pApplicationInfo;
    internal uint enabledLayerCount;
    internal nint ppEnabledLayerNames;
    internal uint enabledExtensionCount;
    internal nint ppEnabledExtensionNames;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceQueueCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint queueFamilyIndex;
    internal uint queueCount;
    internal nint pQueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint queueCreateInfoCount;
    internal nint pQueueCreateInfos;
    internal uint enabledLayerCount;
    internal nint ppEnabledLayerNames;
    internal uint enabledExtensionCount;
    internal nint ppEnabledExtensionNames;
    internal nint pEnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkQueueFamilyProperties
{
    internal VkQueueFlags queueFlags;
    internal uint queueCount;
    internal uint timestampValidBits;
    // VkExtent3D minImageTransferGranularity
    internal uint minTransferWidth;
    internal uint minTransferHeight;
    internal uint minTransferDepth;
}

// VkPhysicalDeviceProperties is a large struct with VkPhysicalDeviceLimits
// and VkPhysicalDeviceSparseProperties tails. We only need the header fields
// (apiVersion..deviceName). The tail is reserved as an oversized byte buffer
// to ensure the native callee has enough space to write without blowing the
// stack — we never read those bytes.
//
// Upper-bound size: Vulkan 1.3 reports the total is 824 bytes; rounding up
// to 2048 gives plenty of headroom across any future extension and avoids
// maintenance when minor versions add fields at the tail.
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceProperties
{
    internal uint apiVersion;
    internal uint driverVersion;
    internal uint vendorID;
    internal uint deviceID;
    internal int deviceType;
    internal fixed byte deviceName[256]; // VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
    internal fixed byte pipelineCacheUUID[16];
    // Limits + SparseProperties tail — intentionally oversized.
    internal fixed byte tail[2048];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMemoryProperties
{
    internal uint memoryTypeCount;
    // 32 * VkMemoryType (each 8 bytes: propertyFlags + heapIndex)
    internal fixed byte memoryTypes[32 * 8];
    internal uint memoryHeapCount;
    // 16 * VkMemoryHeap (each 16 bytes: size(u64) + flags(u32) + padding)
    internal fixed byte memoryHeaps[16 * 16];
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryRequirements
{
    internal ulong size;
    internal ulong alignment;
    internal uint memoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryAllocateInfo
{
    internal int sType;
    internal nint pNext;
    internal ulong allocationSize;
    internal uint memoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal ulong size;
    internal VkBufferUsageFlags usage;
    internal int sharingMode;
    internal uint queueFamilyIndexCount;
    internal nint pQueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkShaderModuleCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal nuint codeSize;
    internal nint pCode; // uint32_t array
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutBinding
{
    internal uint binding;
    internal int descriptorType;
    internal uint descriptorCount;
    internal uint stageFlags;
    internal nint pImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint bindingCount;
    internal nint pBindings;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPushConstantRange
{
    internal uint stageFlags;
    internal uint offset;
    internal uint size;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineLayoutCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint setLayoutCount;
    internal nint pSetLayouts;
    internal uint pushConstantRangeCount;
    internal nint pPushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineShaderStageCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint stage;
    internal nint module;
    internal nint pName; // entry-point name, null-terminated UTF-8
    internal nint pSpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkComputePipelineCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal VkPipelineShaderStageCreateInfo stage;
    internal nint layout;
    internal nint basePipelineHandle;
    internal int basePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolSize
{
    internal int type;
    internal uint descriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint maxSets;
    internal uint poolSizeCount;
    internal nint pPoolSizes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetAllocateInfo
{
    internal int sType;
    internal nint pNext;
    internal nint descriptorPool;
    internal uint descriptorSetCount;
    internal nint pSetLayouts;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorBufferInfo
{
    internal nint buffer;
    internal ulong offset;
    internal ulong range;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkWriteDescriptorSet
{
    internal int sType;
    internal nint pNext;
    internal nint dstSet;
    internal uint dstBinding;
    internal uint dstArrayElement;
    internal uint descriptorCount;
    internal int descriptorType;
    internal nint pImageInfo;
    internal nint pBufferInfo;
    internal nint pTexelBufferView;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandPoolCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint queueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferAllocateInfo
{
    internal int sType;
    internal nint pNext;
    internal nint commandPool;
    internal int level;
    internal uint commandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferBeginInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal nint pInheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkSubmitInfo
{
    internal int sType;
    internal nint pNext;
    internal uint waitSemaphoreCount;
    internal nint pWaitSemaphores;
    internal nint pWaitDstStageMask;
    internal uint commandBufferCount;
    internal nint pCommandBuffers;
    internal uint signalSemaphoreCount;
    internal nint pSignalSemaphores;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCopy
{
    internal ulong srcOffset;
    internal ulong dstOffset;
    internal ulong size;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryBarrier
{
    internal int sType;
    internal nint pNext;
    internal uint srcAccessMask;
    internal uint dstAccessMask;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkFenceCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
}

// VkPipelineStageFlagBits — stage masks for vkCmdPipelineBarrier. Only the few
// we need in the compute-only hot loop are listed.
internal static class VkPipelineStageFlags
{
    internal const uint TopOfPipe = 0x00000001;
    internal const uint Transfer = 0x00001000;
    internal const uint ComputeShader = 0x00000800;
    internal const uint BottomOfPipe = 0x00002000;
    internal const uint Host = 0x00004000;
}

// VkAccessFlagBits — memory access masks for vkCmdPipelineBarrier.
internal static class VkAccessFlags
{
    internal const uint ShaderRead = 0x00000020;
    internal const uint ShaderWrite = 0x00000040;
    internal const uint TransferRead = 0x00000800;
    internal const uint TransferWrite = 0x00001000;
    internal const uint HostRead = 0x00002000;
    internal const uint HostWrite = 0x00004000;
    internal const uint MemoryRead = 0x00008000;
    internal const uint MemoryWrite = 0x00010000;
}

// VkPhysicalDeviceSubgroupProperties — returned by vkGetPhysicalDeviceProperties2
// on Vulkan 1.1+ when chained via pNext. `subgroupSize` is the hardware-fixed
// wave/warp width (32 on NVIDIA/Intel, 64 on AMD GCN/RDNA pre-3, 32-or-64 on
// RDNA3+). `supportedStages` tells us which shader stages may use subgroup ops,
// and `supportedOperations` is the VkSubgroupFeatureFlags bitmask.
[StructLayout(LayoutKind.Sequential)]
internal struct VkPhysicalDeviceSubgroupProperties
{
    internal int sType;
    internal nint pNext;
    internal uint subgroupSize;
    internal uint supportedStages;       // VkShaderStageFlags bitmask
    internal uint supportedOperations;   // VkSubgroupFeatureFlags bitmask
    internal uint quadOperationsInAllStages; // VkBool32
}

// VkPhysicalDeviceProperties2 — Vulkan 1.1 core. We only read the `sType`
// (driver ignores) and `pNext` (chain). `properties` is a VkPhysicalDeviceProperties
// which is large; we reserve the full upper-bound byte tail exactly as in the
// 1.0 struct to guarantee the driver has room to write without stack issues.
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceProperties2
{
    internal int sType;
    internal nint pNext;
    // Inline VkPhysicalDeviceProperties body — same layout as the 1.0 struct.
    internal uint apiVersion;
    internal uint driverVersion;
    internal uint vendorID;
    internal uint deviceID;
    internal int deviceType;
    internal fixed byte deviceName[256];
    internal fixed byte pipelineCacheUUID[16];
    internal fixed byte tail[2048];
}

// VkPhysicalDeviceFeatures2 — Vulkan 1.1 core feature-query header. `features`
// is a VkPhysicalDeviceFeatures (55 VkBool32 fields = 220 bytes). We reserve a
// generously-oversized byte tail rather than spelling out every feature bit;
// no kernel actually reads from this struct after the driver writes it — we
// only set the sType/pNext chain for feature-extension queries (e.g. the
// cooperative-matrix feature struct chained off pNext).
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceFeatures2
{
    internal int sType;
    internal nint pNext;
    // 55 VkBool32 fields. Oversize to 512 bytes for forward-compat safety;
    // any driver that writes fewer bytes is still covered.
    internal fixed byte features[512];
}

// VkPhysicalDeviceCooperativeMatrixFeaturesKHR — feature bits from the
// VK_KHR_cooperative_matrix extension. Chained off VkPhysicalDeviceFeatures2
// via pNext for feature-enable at device creation.
[StructLayout(LayoutKind.Sequential)]
internal struct VkPhysicalDeviceCooperativeMatrixFeaturesKhr
{
    internal int sType;
    internal nint pNext;
    internal uint cooperativeMatrix;                // VkBool32
    internal uint cooperativeMatrixRobustBufferAccess; // VkBool32
}

// VkCooperativeMatrixPropertiesKHR — one entry per driver-supported tile shape
// returned by vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR. Values are
// used to pick the (MSize, NSize, KSize) baked into the compiled coopmat shader.
[StructLayout(LayoutKind.Sequential)]
internal struct VkCooperativeMatrixPropertiesKhr
{
    internal int sType;
    internal nint pNext;
    internal uint MSize;
    internal uint NSize;
    internal uint KSize;
    internal int AType;                    // VkComponentTypeKHR
    internal int BType;                    // VkComponentTypeKHR
    internal int CType;                    // VkComponentTypeKHR
    internal int ResultType;               // VkComponentTypeKHR
    internal uint saturatingAccumulation;  // VkBool32
    internal int scope;                    // VkScopeKHR
}
