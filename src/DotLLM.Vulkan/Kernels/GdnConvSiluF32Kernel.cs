using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused causal depthwise conv + SiLU for the Gated-DeltaNet prefill (issue #695): reads the cached conv state and the qkv rows directly
/// (no [state | qkv] concatenation copy), applies SiLU and writes the result in one pass. Bit-identical to
/// <see cref="Conv1dCausalF32Kernel"/> followed by <see cref="SiluInplaceF32Kernel"/>. The caller must update the conv state AFTER it.
/// </summary>
public sealed class GdnConvSiluF32Kernel : IDisposable
{
    private const int PushConstantBytes = 3 * sizeof(uint);
    private const int TimeTile = 16;
    private const int WorkgroupSize = 256;
    /// <summary>Largest supported convolution width.</summary>
    public const int MaxConvWidth = 8;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnConvSiluF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>True when the SPIR-V is present.</summary>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, "gdn_conv_silu_f32.spv"));

    /// <summary>Loads <c>gdn_conv_silu_f32.spv</c>.</summary>
    public static GdnConvSiluF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_conv_silu_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            for (int i = 0; i < 5; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new GdnConvSiluF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer state, VulkanDevice.Buffer qkv, VulkanDevice.Buffer weight, VulkanDevice.Buffer bias,
        VulkanDevice.Buffer output, int dConv, int channels, int seqLen)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, qkv, weight, bias, output, dConv, channels, seqLen);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records <c>output[t, c] = silu(bias[c] + sum_k x[t + k - (dConv-1), c] * weight[c * dConv + k])</c> where <c>x</c> is the conv
    /// <paramref name="state"/> (<c>dConv - 1</c> rows) followed by <paramref name="qkv"/> (<c>seqLen</c> rows). <paramref name="output"/>
    /// must be a different buffer from <paramref name="qkv"/>.
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer qkv, VulkanDevice.Buffer weight,
        VulkanDevice.Buffer bias, VulkanDevice.Buffer output, int dConv, int channels, int seqLen)
    {
        if (dConv < 2 || dConv > MaxConvWidth) throw new ArgumentOutOfRangeException(nameof(dConv));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (state.Size < (long)(dConv - 1) * channels * sizeof(float)) throw new ArgumentException("state buffer too small.", nameof(state));
        if (qkv.Size < (long)seqLen * channels * sizeof(float)) throw new ArgumentException("qkv buffer too small.", nameof(qkv));
        if (output.Size < (long)seqLen * channels * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));
        if (ReferenceEquals(qkv, output)) throw new ArgumentException("output must not alias qkv.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { state.Handle, qkv.Handle, weight.Handle, bias.Handle, output.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[3] { (uint)dConv, (uint)channels, (uint)seqLen };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((channels + WorkgroupSize - 1) / WorkgroupSize), (uint)((seqLen + TimeTile - 1) / TimeTile), 1);
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
