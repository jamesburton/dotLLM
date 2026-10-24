using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// LFM2 / LFM2.5 gated short-convolution mixer core (<c>lfm2_shortconv_f32.comp</c>, issue #1015). Given the
/// <c>in_proj</c> output <c>[seqLen, 3*channels]</c> laid out <c>[B | C | X]</c> per token it computes
/// <c>y = C * causal_depthwise_conv(B * X)</c> against the per-sequence rolling state and then rolls the state
/// (<see cref="RecordStateUpdate"/>, in place, after a compute barrier). No bias, no activation.
/// </summary>
public sealed class Lfm2ShortConvF32Kernel : IDisposable
{
    private const int PushConstantBytes = 4 * sizeof(uint);
    private const int TimeTile = 16;
    private const int WorkgroupSize = 256;

    /// <summary>Largest supported kernel width (taps).</summary>
    public const int MaxConvWidth = 8;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Lfm2ShortConvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>True when the compiled SPIR-V blob is present in <paramref name="spvDir"/>.</summary>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, "lfm2_shortconv_f32.spv"));

    /// <summary>Loads <c>lfm2_shortconv_f32.spv</c> and creates the pipeline.</summary>
    public static Lfm2ShortConvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "lfm2_shortconv_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < 4; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new Lfm2ShortConvF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous convenience wrapper (tests): conv then state roll in one submission.</summary>
    public void Launch(VulkanDevice.Buffer state, VulkanDevice.Buffer proj, VulkanDevice.Buffer weight, VulkanDevice.Buffer output,
        int dConv, int channels, int seqLen)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, proj, weight, output, dConv, channels, seqLen);
        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
        RecordStateUpdate(ctx.CommandBuffer, state, proj, weight, output, dConv, channels, seqLen);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the conv + C-gate pass. Reads the OLD <paramref name="state"/>; call <see cref="RecordStateUpdate"/> afterwards.</summary>
    public void Record(nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer proj, VulkanDevice.Buffer weight,
        VulkanDevice.Buffer output, int dConv, int channels, int seqLen)
        => RecordCore(cmdBuf, state, proj, weight, output, dConv, channels, seqLen, mode: 0);

    /// <summary>Records the in-place roll of <paramref name="state"/> (must follow <see cref="Record"/> behind a compute barrier).</summary>
    public void RecordStateUpdate(nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer proj, VulkanDevice.Buffer weight,
        VulkanDevice.Buffer output, int dConv, int channels, int seqLen)
        => RecordCore(cmdBuf, state, proj, weight, output, dConv, channels, seqLen, mode: 1);

    private unsafe void RecordCore(nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer proj, VulkanDevice.Buffer weight,
        VulkanDevice.Buffer output, int dConv, int channels, int seqLen, uint mode)
    {
        if (dConv < 2 || dConv > MaxConvWidth) throw new ArgumentOutOfRangeException(nameof(dConv));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (state.Size < (long)(dConv - 1) * channels * sizeof(float)) throw new ArgumentException("state buffer too small.", nameof(state));
        if (proj.Size < (long)seqLen * 3 * channels * sizeof(float)) throw new ArgumentException("proj buffer too small.", nameof(proj));
        if (weight.Size < (long)dConv * channels * sizeof(float)) throw new ArgumentException("weight buffer too small.", nameof(weight));
        if (output.Size < (long)seqLen * channels * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));
        if (ReferenceEquals(proj, output)) throw new ArgumentException("output must not alias proj.", nameof(output));

        Span<nint> buffers = stackalloc nint[4] { state.Handle, proj.Handle, weight.Handle, output.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)dConv, (uint)channels, (uint)seqLen, mode };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        uint groupsX = (uint)((channels + WorkgroupSize - 1) / WorkgroupSize);
        uint groupsY = mode == 0 ? (uint)((seqLen + TimeTile - 1) / TimeTile) : 1u;
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
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
