using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// One-dispatch replacements for the per-token <c>vkCmdCopyBuffer</c> fan-out loops of the hybrid (GDN + attention) prefill: the GDN
/// <c>[Q | K | V]</c> row split (<see cref="Kind.GdnQkvSplit"/>) and the fused attention Q+gate per-head de-interleave
/// (<see cref="Kind.QGateDeinterleave"/>). Pure F32 moves, bit-exact with the loops they replace; each saves ~seqLen x 3 (or x 2 x numHeads)
/// commands per layer and the transfer/compute barrier pair around them.
/// </summary>
public sealed class DeinterleaveF32Kernel : IDisposable
{
    /// <summary>Which de-interleave this instance performs.</summary>
    public enum Kind
    {
        /// <summary><c>gdn_split_qkv_f32</c>: rows <c>[Q(kDim) | K(kDim) | V(vDim)]</c> to three dense buffers.</summary>
        GdnQkvSplit,

        /// <summary><c>qgate_deinterleave_f32</c>: per-head interleaved <c>[Q_h | Gate_h]</c> rows to dense Q and Gate.</summary>
        QGateDeinterleave,
    }

    private const int PushConstantBytes = 3 * sizeof(uint);
    private const int GroupSize = 256;
    private const int MaxGroupsX = 65535;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _buffers;
    private bool _disposed;

    private DeinterleaveF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int buffers)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _buffers = buffers;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: buffers);
    }

    /// <summary>Loads the SPIR-V for <paramref name="kind"/>; <see langword="null"/> when it is missing (older builds).</summary>
    public static DeinterleaveF32Kernel? TryCreate(VulkanDevice device, string spvDir, Kind kind)
    {
        (string name, int buffers) = kind == Kind.GdnQkvSplit ? ("gdn_split_qkv_f32.spv", 4) : ("qgate_deinterleave_f32.spv", 3);
        string path = Path.Combine(spvDir, name);
        if (!File.Exists(path)) return null;

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffers];
            for (int i = 0; i < buffers; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
        return new DeinterleaveF32Kernel(device, module, pipeline, pool, buffers);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the GDN split: <paramref name="qkv"/> <c>[seqLen, 2*kDim + vDim]</c> into <paramref name="q"/> / <paramref name="k"/>
    /// (<c>[seqLen, kDim]</c>) and <paramref name="v"/> (<c>[seqLen, vDim]</c>).
    /// </summary>
    public void RecordGdnQkvSplit(
        nint cmdBuf, VulkanDevice.Buffer qkv, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        int seqLen, int kDim, int vDim)
    {
        if (_buffers != 4) throw new InvalidOperationException("Not a GDN QKV split kernel.");
        RecordCore(cmdBuf, stackalloc nint[4] { qkv.Handle, q.Handle, k.Handle, v.Handle }, seqLen, kDim, vDim, (long)seqLen * (2 * kDim + vDim));
    }

    /// <summary>
    /// Records the Q+gate de-interleave: <paramref name="qg"/> <c>[seqLen, numHeads * 2 * headDim]</c> into dense
    /// <paramref name="q"/> and <paramref name="gate"/> (<c>[seqLen, numHeads * headDim]</c>).
    /// </summary>
    public void RecordQGateDeinterleave(
        nint cmdBuf, VulkanDevice.Buffer qg, VulkanDevice.Buffer q, VulkanDevice.Buffer gate,
        int seqLen, int numHeads, int headDim)
    {
        if (_buffers != 3) throw new InvalidOperationException("Not a Q+gate de-interleave kernel.");
        RecordCore(cmdBuf, stackalloc nint[3] { qg.Handle, q.Handle, gate.Handle }, seqLen, numHeads, headDim, (long)seqLen * numHeads * headDim);
    }

    private unsafe void RecordCore(nint cmdBuf, Span<nint> buffers, int a, int b, int c, long elements)
    {
        if (a <= 0 || b <= 0 || c <= 0) throw new ArgumentOutOfRangeException(nameof(a));
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3] { (uint)a, (uint)b, (uint)c };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        long groups = (elements + GroupSize - 1) / GroupSize;
        uint gx = (uint)Math.Min(groups, MaxGroupsX);
        uint gy = (uint)((groups + MaxGroupsX - 1) / MaxGroupsX);
        VulkanApi.vkCmdDispatch(cmdBuf, gx, gy, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
