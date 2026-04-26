using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Depthwise causal 1-D convolution kernel for Mamba2 SSM layers
/// (NVIDIA Nemotron-H, Mamba-3). Each channel is convolved
/// independently along the time axis with a small kernel of width
/// <c>dConv</c> (typically 4) followed by a per-channel bias:
/// <code>
///     y[t, c] = bias[c] + sum_{k=0..dConv-1} input[t+k, c] * weight[c * dConv + k]
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// Memory layout mirrors the CPU reference
/// (<see cref="DotLLM.Cpu.Kernels.Conv1dCausal"/>) and llama.cpp /
/// GGUF exactly:
/// </para>
/// <list type="bullet">
///   <item><description>
///     <b>input</b> is the concatenated <c>[conv_state | xBC]</c> buffer
///     of shape <c>[dConv-1+seqLen, channels]</c>, row-major. Element
///     <c>(t, c)</c> at flat index <c>t * channels + c</c>.
///   </description></item>
///   <item><description>
///     <b>weight</b> follows GGUF's channel-major layout — element
///     <c>(k, c)</c> at flat index <c>c * dConv + k</c>, so each
///     channel's <c>dConv</c> taps are contiguous and the inner loop
///     does <c>dConv</c> sequential loads per channel.
///   </description></item>
///   <item><description>
///     <b>bias</b> has length <c>channels</c>;
///     <b>output</b> shape <c>[seqLen, channels]</c> row-major.
///   </description></item>
/// </list>
/// <para>
/// Dispatch is 2D, one thread per <c>(t, c)</c> output cell, with a
/// (16, 16) workgroup.
/// </para>
/// </remarks>
public sealed class Conv1dCausalF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // dConv, channels, seqLen (all u32)
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Conv1dCausalF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>Loads <c>conv1d_causal_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Conv1dCausalF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "conv1d_causal_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new Conv1dCausalF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer bias, VulkanDevice.Buffer output,
        int dConv, int channels, int seqLen)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, input, weight, bias, output, dConv, channels, seqLen);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the depthwise causal conv dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="input">
    /// F32 concatenated <c>[conv_state | xBC]</c> buffer with shape
    /// <c>[dConv-1+seqLen, channels]</c>, row-major.
    /// </param>
    /// <param name="weight">
    /// F32 conv kernel weights, GGUF channel-major shape
    /// <c>[dConv, channels]</c> — element <c>(k, c)</c> at
    /// <c>c * dConv + k</c>.
    /// </param>
    /// <param name="bias">F32 per-channel bias, length <paramref name="channels"/>.</param>
    /// <param name="output">F32 output buffer, shape <c>[seqLen, channels]</c> row-major.</param>
    /// <param name="dConv">Convolution kernel width (typically 4).</param>
    /// <param name="channels">Number of channels (depthwise width).</param>
    /// <param name="seqLen">Number of output time steps.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer bias, VulkanDevice.Buffer output,
        int dConv, int channels, int seqLen)
    {
        if (dConv <= 0) throw new ArgumentOutOfRangeException(nameof(dConv));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        long inputRows = (long)(dConv - 1) + seqLen;
        long inputBytes = inputRows * channels * sizeof(float);
        long weightBytes = (long)dConv * channels * sizeof(float);
        long biasBytes = (long)channels * sizeof(float);
        long outputBytes = (long)seqLen * channels * sizeof(float);
        if (input.Size < inputBytes) throw new ArgumentException("input buffer too small.", nameof(input));
        if (weight.Size < weightBytes) throw new ArgumentException("weight buffer too small.", nameof(weight));
        if (bias.Size < biasBytes) throw new ArgumentException("bias buffer too small.", nameof(bias));
        if (output.Size < outputBytes) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[4]
        {
            input.Handle, weight.Handle, bias.Handle, output.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)dConv);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)channels);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)seqLen);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((channels + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((seqLen + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
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
