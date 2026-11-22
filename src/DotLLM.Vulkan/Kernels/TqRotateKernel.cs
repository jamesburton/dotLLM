using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// TurboQuant Q/O rotations (#1892 step 2): attention runs in code space, so the query is rotated INTO K code space
/// (<c>tq_rotate_q.spv</c>: <c>q~ = H(S_k·q)/√d</c>) and the accumulated output back OUT of V code space
/// (<c>tq_rotate_o.spv</c>: <c>o = S_v·H(a~)/√d</c>). Both are in place on an F32 <c>[rows, numHeads, headDim]</c> buffer, one
/// workgroup per (row, head) with the butterfly in shared memory (the <c>turboquant_dequant_f32.comp</c> pattern). The sign
/// buffer is the codec's F32 ±1 vector (K seed for the query, <c>seed ^ TurboQuantCodec.VSeedMask</c> for the output), uploaded once with
/// the cache. <b>headDim a power of two ≤ 256.</b>
/// </summary>
public sealed class TqRotateKernel : IDisposable
{
    /// <summary>Maximum supported per-head dimension (workgroup size / shared-array bound).</summary>
    public const int MaxHeadDim = 256;

    private const int BindingCount = 2;
    private const int PushConstantBytes = 3 * sizeof(uint); // headDim, numHeads, invSqrtD

    private readonly VulkanDevice _device;
    private readonly Variant _query;
    private readonly Variant _output;
    private bool _disposed;

    private sealed class Variant : IDisposable
    {
        public required VulkanModule Module;
        public required ComputePipeline Pipeline;
        public required nint Pool;
        public required DescriptorSetCache Cache;
        private readonly VulkanDevice _device;
        public Variant(VulkanDevice device) => _device = device;

        public void Dispose()
        {
            if (Pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, Pool, 0);
            Pipeline.Dispose();
            Module.Dispose();
        }
    }

    private TqRotateKernel(VulkanDevice device, Variant query, Variant output)
    {
        _device = device;
        _query = query;
        _output = output;
    }

    /// <summary>Loads <c>tq_rotate_q.spv</c> and <c>tq_rotate_o.spv</c> from the given directory and creates both pipelines.</summary>
    public static TqRotateKernel Create(VulkanDevice device, string spvDir)
    {
        var q = Load(device, Path.Combine(spvDir, "tq_rotate_q.spv"));
        try
        {
            var o = Load(device, Path.Combine(spvDir, "tq_rotate_o.spv"));
            return new TqRotateKernel(device, q, o);
        }
        catch
        {
            q.Dispose();
            throw;
        }
    }

    private static Variant Load(VulkanDevice device, string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BindingCount];
            for (int i = 0; i < BindingCount; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: BindingCount);
        return new Variant(device)
        {
            Module = module,
            Pipeline = pipeline,
            Pool = pool,
            Cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BindingCount),
        };
    }

    internal void InvalidateDescriptorCache()
    {
        _query.Cache.Reset();
        _output.Cache.Reset();
    }

    /// <summary>Rotates the query in place into K code space and waits (test / one-shot use).</summary>
    public void LaunchQuery(VulkanDevice.Buffer q, int rows, int numHeads, int headDim, VulkanDevice.Buffer signsK)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordQuery(ctx.CommandBuffer, q, rows, numHeads, headDim, signsK);
        ctx.SubmitAndWait();
    }

    /// <summary>Rotates the output in place back out of V code space and waits (test / one-shot use).</summary>
    public void LaunchOutput(VulkanDevice.Buffer o, int rows, int numHeads, int headDim, VulkanDevice.Buffer signsV)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordOutput(ctx.CommandBuffer, o, rows, numHeads, headDim, signsV);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the in-place query rotation <c>q~ = H(S_k·q)/√d</c> into <paramref name="cmd"/> without submitting.</summary>
    public void RecordQuery(nint cmd, VulkanDevice.Buffer q, int rows, int numHeads, int headDim, VulkanDevice.Buffer signsK)
        => Record(_query, cmd, q, rows, numHeads, headDim, signsK);

    /// <summary>Records the in-place output rotation <c>o = S_v·H(a~)/√d</c> into <paramref name="cmd"/> without submitting.</summary>
    public void RecordOutput(nint cmd, VulkanDevice.Buffer o, int rows, int numHeads, int headDim, VulkanDevice.Buffer signsV)
        => Record(_output, cmd, o, rows, numHeads, headDim, signsV);

    private static unsafe void Record(
        Variant v, nint cmd, VulkanDevice.Buffer data, int rows, int numHeads, int headDim, VulkanDevice.Buffer signs)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (rows > 65535) throw new ArgumentOutOfRangeException(nameof(rows), "one workgroup row per dispatch Y; split the call.");
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > MaxHeadDim)
            throw new ArgumentException($"headDim must be a power of two in [1,{MaxHeadDim}]; got {headDim}.", nameof(headDim));
        if (data.Size < (long)rows * numHeads * headDim * sizeof(float))
            throw new ArgumentException("data buffer too small.", nameof(data));
        if (signs.Size < (long)headDim * sizeof(float))
            throw new ArgumentException("signs buffer too small.", nameof(signs));

        Span<nint> buffers = stackalloc nint[BindingCount] { data.Handle, signs.Handle };
        nint descriptorSet = v.Cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, v.Pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, v.Pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3]
        {
            (uint)headDim, (uint)numHeads, BitConverter.SingleToUInt32Bits(1.0f / MathF.Sqrt(headDim)),
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmd, v.Pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmd, (uint)numHeads, (uint)rows, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _query.Dispose();
        _output.Dispose();
    }
}
