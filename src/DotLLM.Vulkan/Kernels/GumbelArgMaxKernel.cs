using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Position-seeded Gumbel-max sampling over F32 logits rows on the device (#1382): per row,
/// <c>argmax_i (logits[i] * invTemp + G(seed, firstTokenIndex + row, i))</c> over the finite entries, reading the rows in place and
/// writing only one token id per row. Bit-for-bit the same draw as <c>DotLLM.Engine.Samplers.GumbelNoise.Sample</c> (see
/// <c>gumbel_argmax_f32.comp</c>), so a device verify row and a CPU plain-decode step with the same logits pick the same token.
/// </summary>
/// <remarks>
/// The shader takes <b>raw</b> logits and applies only the temperature scale. Top-k / top-p / min-p / penalties / constraint masks
/// are not applied here: they need either the host (history, constraint state, which depends on tokens a verify batch has not
/// accepted yet) or a per-row selection (k-th largest, sort) whose exact tie semantics are fragile. The caller gates on that
/// (<c>SamplerPipeline.DeviceGumbelInvTemperature</c>) and keeps the host path for every other shape.
/// </remarks>
public sealed class GumbelArgMaxKernel : IDisposable
{
    private const int PushConstantBytes = 6 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GumbelArgMaxKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>The shader file name.</summary>
    public const string SpvFileName = "gumbel_argmax_f32.spv";

    /// <summary>True when the shader is present in <paramref name="spvDir"/>.</summary>
    public static bool IsAvailable(string? spvDir) => spvDir is not null && File.Exists(Path.Combine(spvDir, SpvFileName));

    /// <summary>Loads the shader from <paramref name="spvDir"/>.</summary>
    public static GumbelArgMaxKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, SpvFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Build the Vulkan project (glslc) first.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        return new GumbelArgMaxKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when the logits buffer has been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch of <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer logits, VulkanDevice.Buffer ids, int rows, int vocab, int rowStride,
        ulong seed, int firstTokenIndex, float invTemperature)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
        Record(ctx.CommandBuffer, logits, ids, rows, vocab, rowStride, seed, firstTokenIndex, invTemperature);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the draw of <paramref name="rows"/> rows into <paramref name="ids"/> (one uint per row).</summary>
    /// <param name="cmdBuf">Command buffer being recorded.</param>
    /// <param name="logits">F32 logits, row r at element <c>r * rowStride</c>.</param>
    /// <param name="ids">Receives <paramref name="rows"/> token ids.</param>
    /// <param name="rows">Rows to draw (one workgroup each).</param>
    /// <param name="vocab">Vocabulary size (entries per row).</param>
    /// <param name="rowStride">Elements between rows.</param>
    /// <param name="seed">Request noise seed (<c>SamplerPipeline</c> Gumbel seed).</param>
    /// <param name="firstTokenIndex">Generation index of row 0; row r uses <c>firstTokenIndex + r</c>.</param>
    /// <param name="invTemperature">The F32 <c>1f / temperature</c> the CPU temperature step multiplies by (1 for T = 1).</param>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer logits, VulkanDevice.Buffer ids, int rows, int vocab, int rowStride,
        ulong seed, int firstTokenIndex, float invTemperature)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (vocab <= 0) throw new ArgumentOutOfRangeException(nameof(vocab));
        if (rowStride < vocab) throw new ArgumentOutOfRangeException(nameof(rowStride));
        if (firstTokenIndex < 0) throw new ArgumentOutOfRangeException(nameof(firstTokenIndex));
        if (logits.Size < ((long)(rows - 1) * rowStride + vocab) * sizeof(float))
            throw new ArgumentException("Logits buffer too small.", nameof(logits));
        if (ids.Size < (long)rows * sizeof(uint))
            throw new ArgumentException("Id buffer too small.", nameof(ids));

        Span<nint> buffers = stackalloc nint[2] { logits.Handle, ids.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[6]
        {
            (uint)vocab, (uint)rowStride, (uint)seed, (uint)(seed >> 32), (uint)firstTokenIndex,
            BitConverter.SingleToUInt32Bits(invTemperature),
        };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)rows, 1, 1);
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

/// <summary>
/// Diagnostic twin of <see cref="GumbelArgMaxKernel"/>: writes the raw Gumbel noise <c>G(seed, tokenIndex, firstIndex + i)</c> so tests can
/// compare it to <c>GumbelNoise.Noise</c> bit for bit over the whole 2^24 uniform domain. Not on any inference path.
/// </summary>
internal sealed class GumbelNoiseDumpKernel : IDisposable
{
    private const int PushConstantBytes = 6 * sizeof(uint);
    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;

    public GumbelNoiseDumpKernel(VulkanDevice device, string spvDir)
    {
        _device = device;
        _module = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, "gumbel_noise_dump_f32.spv"));
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[1];
        bindings[0] = new VkDescriptorBinding(0);
        _pipeline = _module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        _descriptorPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 1);
        _descriptorCache = new DescriptorSetCache(device, _descriptorPool, _pipeline, buffersPerSet: 1);
    }

    /// <summary>Synchronously writes <paramref name="count"/> noise values into <paramref name="dst"/>.</summary>
    public unsafe void Launch(VulkanDevice.Buffer dst, int count, int firstIndex, int tokenIndex, ulong seed, bool uniformSweep = false)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        nint cmd = ctx.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);
        Span<nint> buffers = stackalloc nint[1] { dst.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[6] { (uint)count, (uint)firstIndex, (uint)tokenIndex, (uint)seed, (uint)(seed >> 32), uniformSweep ? 1u : 0u };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmd, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmd, (uint)((count + 255) / 256), 1, 1);
        KernelSupport.ComputeToHostBarrier(cmd);
        ctx.SubmitAndWait();
    }

    public void Dispose()
    {
        VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
