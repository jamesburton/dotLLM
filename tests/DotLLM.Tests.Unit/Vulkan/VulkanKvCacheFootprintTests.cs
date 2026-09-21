using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Measures the device-local footprint of <see cref="VulkanNemotronHKvCache"/> at Bonsai 2 27B
/// geometry, eager versus lazily grown (issue #437).
/// </summary>
/// <remarks>
/// <para>Bonsai 2 (<c>qwen35</c>, PQ2_0): <c>block_count = 64</c> with
/// <c>full_attention_interval = 4</c> gives 16 full-attention layers; <c>head_count_kv = 4</c> and
/// <c>key_length = value_length = 256</c> give a 1024-element K (and V) row, i.e. 4 KiB per layer
/// per token for K plus 4 KiB for V — <b>128 KiB per token</b> across the 16 slots.</para>
///
/// <para><b>Instrument.</b> <see cref="VulkanDevice.MemorySnapshot"/>'s <c>ours=</c> figure, which is
/// the allocator's own per-heap live-byte accounting. The driver's <c>usage=</c>/<c>budget=</c>
/// fields are per-process and describe the querying process, and a host working-set reading does
/// not show device-local memory at all — neither is usable here.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvCacheFootprintTests
{
    private readonly ITestOutputHelper _out;

    public VulkanKvCacheFootprintTests(ITestOutputHelper output) => _out = output;

    private const int AttentionLayers = 16;   // 64 blocks / full_attention_interval 4
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;

    [SkippableTheory]
    [InlineData(4096)]
    [InlineData(32768)]
    public void Bonsai2Geometry_LazyGrowth_CutsDeviceLocalFootprint(int maxSeqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();

        var slotMap = new int[AttentionLayers];
        for (int i = 0; i < AttentionLayers; i++) slotMap[i] = i;

        _out.WriteLine($"=== ctx {maxSeqLen} — 16 attn layers, kvHeads=4, headDim=256 (128 KiB/token) ===");
        _out.WriteLine($"baseline (no cache):  {device.MemorySnapshot()}");
        long fallbacksBefore = device.DeviceLocalFallbackCount;

        long eagerBytes;
        using (var eager = new VulkanNemotronHKvCache(device, slotMap, AttentionLayers,
                   NumKvHeads, HeadDim, maxSeqLen, initialCapacity: maxSeqLen))
        {
            eagerBytes = eager.AllocatedBytes;
            _out.WriteLine($"EAGER   (pre-#437):   {eager.AllocatedBytes / (1024.0 * 1024.0):F1} MiB  "
                         + $"capacity={eager.Capacity}  | {device.MemorySnapshot()}");
            _out.WriteLine($"  device-local fallbacks during eager alloc: {device.DeviceLocalFallbackCount - fallbacksBefore}");
        }

        long lazyBytes;
        using (var lazy = new VulkanNemotronHKvCache(device, slotMap, AttentionLayers,
                   NumKvHeads, HeadDim, maxSeqLen))
        {
            lazyBytes = lazy.AllocatedBytes;
            _out.WriteLine($"LAZY    (default):    {lazy.AllocatedBytes / (1024.0 * 1024.0):F1} MiB  "
                         + $"capacity={lazy.Capacity}  | {device.MemorySnapshot()}");

            // A realistic 2k-token session never touches the declared context.
            lazy.EnsureCapacity(2048);
            _out.WriteLine($"LAZY    after 2k ctx: {lazy.AllocatedBytes / (1024.0 * 1024.0):F1} MiB  "
                         + $"capacity={lazy.Capacity}  | {device.MemorySnapshot()}");
            Assert.Equal(2048, lazy.Capacity);
            Assert.True(lazy.AllocatedBytes < eagerBytes);
        }

        // F16 store: the same rows at 2 bytes per element instead of 4.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using (var f16Kernels = VulkanKvF16Kernels.Create(device, spvDir))
        {
            using var eagerF16 = new VulkanNemotronHKvCache(device, slotMap, AttentionLayers,
                NumKvHeads, HeadDim, maxSeqLen, maxSeqLen, VulkanKvDataType.F16, f16Kernels, ownsKernels: false);
            _out.WriteLine($"EAGER F16 (opt-in):   {eagerF16.AllocatedBytes / (1024.0 * 1024.0):F1} MiB  "
                         + $"capacity={eagerF16.Capacity}  | {device.MemorySnapshot()}");
            Assert.Equal(eagerBytes / 2, eagerF16.AllocatedBytes);

            using var lazyF16 = new VulkanNemotronHKvCache(device, slotMap, AttentionLayers,
                NumKvHeads, HeadDim, maxSeqLen, Math.Min(VulkanNemotronHKvCache.DefaultInitialCapacity, maxSeqLen),
                VulkanKvDataType.F16, f16Kernels, ownsKernels: false);
            lazyF16.EnsureCapacity(Math.Min(2048, maxSeqLen));
            _out.WriteLine($"LAZY  F16 after 2k:   {lazyF16.AllocatedBytes / (1024.0 * 1024.0):F1} MiB  "
                         + $"capacity={lazyF16.Capacity}  | {device.MemorySnapshot()}");
        }

        // The arithmetic in the issue: 128 KiB per token across the 16 attention slots.
        Assert.Equal((long)maxSeqLen * 128 * 1024, eagerBytes);
        Assert.Equal((long)Math.Min(VulkanNemotronHKvCache.DefaultInitialCapacity, maxSeqLen) * 128 * 1024, lazyBytes);
    }
}
