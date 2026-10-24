using System.Buffers.Binary;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Element-wise ops of the Qwen4-Exp (Qwen3.8-Flash-Next) gated residual ("hyper-connection"): embedding broadcast into the
/// <c>S</c> residual streams, the <c>silu(v / S)</c> low-rank activation, the stream mix-and-mean that produces the block input,
/// the <c>2 * sigmoid(g / S)</c> write gains and the gain-scaled write back into every stream. One shader
/// (<c>qwen4exp_gated_residual.comp</c>) with the op selected by a push constant; each method mirrors the matching
/// <c>DotLLM.Cpu.Kernels.Qwen4ExpGatedResidual</c> function. The per-stream group RMSNorm and the three GEMVs (down, up, inject)
/// reuse <see cref="GroupRmsNormF32Kernel"/> and the model's quant-aware matmul dispatcher.
/// </summary>
/// <remarks>Residual layout is <c>[seqLen, S, H]</c> row-major (HF <c>unflatten(-1, (S, H))</c>).</remarks>
public sealed class Qwen4ExpGatedResidualKernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 5 * sizeof(uint);   // op, n, S, H, invS

    private enum Op : uint { Broadcast = 0, Activate = 1, MixMean = 2, InjectGains = 3, Write = 4 }

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Qwen4ExpGatedResidualKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>qwen4exp_gated_residual.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Qwen4ExpGatedResidualKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "qwen4exp_gated_residual.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            for (int i = 0; i < 3; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new Qwen4ExpGatedResidualKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Embedding broadcast: <c>R[t, s, :] = emb[t, :]</c> for every stream.</summary>
    public void RecordBroadcast(nint cmdBuf, VulkanDevice.Buffer residual, VulkanDevice.Buffer embedding, int seqLen, int streams, int hidden)
        => Record(cmdBuf, Op.Broadcast, residual, embedding, residual, (long)seqLen * streams * hidden, streams, hidden,
                  (long)seqLen * streams * hidden, (long)seqLen * hidden);

    /// <summary>In place <c>v = silu(v / S)</c> over <paramref name="count"/> low-rank elements.</summary>
    public void RecordActivateLowRank(nint cmdBuf, VulkanDevice.Buffer v, int count, int streams)
        => Record(cmdBuf, Op.Activate, v, v, v, count, streams, 1, count, count);

    /// <summary>Block input <c>h[t,:] = mean_s(sigmoid(mix[t,s,:]) * xn[t,s,:])</c>.</summary>
    public void RecordMixMean(nint cmdBuf, VulkanDevice.Buffer blockInput, VulkanDevice.Buffer mix, VulkanDevice.Buffer normed,
                              int seqLen, int streams, int hidden)
        => Record(cmdBuf, Op.MixMean, blockInput, mix, normed, (long)seqLen * hidden, streams, hidden,
                  (long)seqLen * hidden, (long)seqLen * streams * hidden);

    /// <summary>In place write gains <c>g = 2 * sigmoid(g / S)</c> over <c>seqLen * S</c> inject-projection outputs.</summary>
    public void RecordInjectGains(nint cmdBuf, VulkanDevice.Buffer gains, int seqLen, int streams)
        => Record(cmdBuf, Op.InjectGains, gains, gains, gains, (long)seqLen * streams, streams, 1,
                  (long)seqLen * streams, (long)seqLen * streams);

    /// <summary>Write: <c>R[t,s,:] += gains[t,s] * y[t,:]</c> against the raw residual.</summary>
    public void RecordWrite(nint cmdBuf, VulkanDevice.Buffer residual, VulkanDevice.Buffer blockOutput, VulkanDevice.Buffer gains,
                            int seqLen, int streams, int hidden)
        => Record(cmdBuf, Op.Write, residual, blockOutput, gains, (long)seqLen * streams * hidden, streams, hidden,
                  (long)seqLen * streams * hidden, (long)seqLen * streams);

    private unsafe void Record(nint cmdBuf, Op op, VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
                               long n, int streams, int hidden, long aElems, long bcElems)
    {
        if (n <= 0 || n > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(n));
        if (streams <= 0 || hidden <= 0) throw new ArgumentOutOfRangeException(nameof(streams));
        if (a.Size < aElems * sizeof(float)) throw new ArgumentException("buffer A too small.", nameof(a));
        if (b.Size < bcElems * sizeof(float) && op is Op.Broadcast or Op.MixMean)
            throw new ArgumentException("buffer B too small.", nameof(b));

        Span<nint> buffers = stackalloc nint[3] { a.Handle, b.Handle, c.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);

        Span<byte> pc = stackalloc byte[PushConstantBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(pc, (uint)op);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[4..], (uint)n);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[8..], (uint)streams);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[12..], (uint)hidden);
        BinaryPrimitives.WriteSingleLittleEndian(pc[16..], 1.0f / streams);
        fixed (byte* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((n + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
    }

    /// <summary>Synchronous test launch of one op via a throwaway submit context.</summary>
    internal void LaunchWrite(VulkanDevice.Buffer residual, VulkanDevice.Buffer y, VulkanDevice.Buffer gains, int seqLen, int streams, int hidden)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordWrite(ctx.CommandBuffer, residual, y, gains, seqLen, streams, hidden);
        ctx.SubmitAndWait();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
