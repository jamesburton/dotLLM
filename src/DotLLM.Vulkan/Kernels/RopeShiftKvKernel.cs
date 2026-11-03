using System.Buffers.Binary;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Rotates a row range of a <see cref="VulkanKvCache"/> K buffer by a RoPE position delta, in place on the device
/// (issue #1531 step 2). K is stored post-RoPE, so moving cached rows from position <c>p</c> to <c>p + delta</c> (chunk-level
/// KV reuse) is the pair rotation by <c>delta × freq_i</c>; V is position-independent and untouched.
/// </summary>
/// <remarks>
/// Plain RoPE only (frequencies <c>1 / theta^(2i/freqDim)</c>, as <see cref="RopeF32Kernel"/>): scaled variants (YaRN, Llama-3
/// frequency factors, MLA) are not representable and must be refused by the caller. Reads and writes the cache's own element
/// format (F32 or F16); the rotation itself is F32 in registers with one rounding at the store.
/// </remarks>
public sealed class RopeShiftKvKernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 10 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule[] _modules = new VulkanModule[2];
    private readonly ComputePipeline[] _pipelines = new ComputePipeline[2];
    private readonly DescriptorSetCache[] _caches = new DescriptorSetCache[2];
    private readonly nint _pool;
    private bool _disposed;

    private RopeShiftKvKernel(VulkanDevice device, string spvDir)
    {
        _device = device;
        _pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 1);
        string[] files = { "rope_shift_kv_f32.spv", "rope_shift_kv_f16.spv" };
        for (int i = 0; i < 2; i++)
        {
            string path = Path.Combine(spvDir, files[i]);
            if (!File.Exists(path)) throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");
            _modules[i] = VulkanModule.LoadFromFile(device, path);
            var bindings = new VkDescriptorBinding[] { new VkDescriptorBinding(0) };
            _pipelines[i] = _modules[i].CreateComputePipeline("main", bindings, PushConstantBytes);
            _caches[i] = new DescriptorSetCache(device, _pool, _pipelines[i], buffersPerSet: 1);
        }
    }

    /// <summary>Loads the F32 and F16 shift pipelines from <paramref name="spvDir"/>.</summary>
    public static RopeShiftKvKernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);

    /// <summary>Rotates K rows <c>[startRow, startRow + rowCount)</c> of <paramref name="layerIndex"/> by <paramref name="delta"/> positions and submits (synchronous).</summary>
    public void Shift(VulkanKvCache cache, int layerIndex, int startRow, int rowCount, int delta,
        int numKvHeads, int headDim, int ropeDim, float theta, RopeF32Kernel.Variant variant, int freqDim = 0, int? neoxPairOffset = null)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, cache, layerIndex, startRow, rowCount, delta, numKvHeads, headDim, ropeDim, theta, variant, freqDim, neoxPairOffset);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the shift into <paramref name="cmdBuf"/>. The caller orders it against prior writes of the same K buffer.</summary>
    public unsafe void Record(nint cmdBuf, VulkanKvCache cache, int layerIndex, int startRow, int rowCount, int delta,
        int numKvHeads, int headDim, int ropeDim, float theta, RopeF32Kernel.Variant variant, int freqDim = 0, int? neoxPairOffset = null)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if ((uint)layerIndex >= (uint)cache.LayerCount) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        if (startRow < 0 || rowCount <= 0 || (long)startRow + rowCount > cache.MaxLength) throw new ArgumentOutOfRangeException(nameof(rowCount));
        if (startRow + rowCount > cache.CurrentLength) throw new ArgumentOutOfRangeException(nameof(rowCount), "rows past CurrentLength were never written.");
        if (numKvHeads <= 0 || headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (cache.KvStrideOf(layerIndex) != numKvHeads * headDim) throw new ArgumentException("numKvHeads × headDim must equal the layer's KV stride.");
        if (ropeDim <= 0 || (ropeDim & 1) != 0 || ropeDim > headDim) throw new ArgumentException($"ropeDim must be positive, even and <= headDim; got {ropeDim}.", nameof(ropeDim));
        if (freqDim <= 0) freqDim = ropeDim;
        int pairOffset = neoxPairOffset ?? (ropeDim / 2);
        if (variant == RopeF32Kernel.Variant.NeoX && (pairOffset <= 0 || pairOffset + ropeDim / 2 > headDim))
            throw new ArgumentException($"neoxPairOffset {pairOffset} + ropeDim/2 must be <= headDim.", nameof(neoxPairOffset));
        if (delta == 0) return;

        int which = cache.Storage == VulkanKvStorage.F16 ? 1 : 0;
        Span<nint> buffers = stackalloc nint[1] { cache.GetKeysBuffer(layerIndex).Handle };
        nint set = _caches[which].GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipelines[which].Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipelines[which].Layout, 0, 1, set, 0, 0);

        Span<byte> pc = stackalloc byte[PushConstantBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(pc[0..], (uint)startRow);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[4..], (uint)rowCount);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[8..], (uint)numKvHeads);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[12..], (uint)headDim);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[16..], (uint)ropeDim);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[20..], (uint)variant);
        BinaryPrimitives.WriteSingleLittleEndian(pc[24..], theta);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[28..], (uint)freqDim);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[32..], (uint)pairOffset);
        BinaryPrimitives.WriteInt32LittleEndian(pc[36..], delta);
        fixed (byte* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipelines[which].Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);

        long pairs = (long)rowCount * numKvHeads * (ropeDim / 2);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((pairs + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        for (int i = 0; i < 2; i++) { _pipelines[i]?.Dispose(); _modules[i]?.Dispose(); }
    }
}
