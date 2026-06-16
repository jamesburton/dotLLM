using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 decode-path GEMV using DP4a with a <b>shared activation
/// pre-quantization</b> pass: <c>y[M] = W_q8[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Runs two dispatches behind one submit: (1) <c>quantize_q8_act</c> quantizes
/// the FP32 activation <c>x</c> to INT8 once (packed <c>xq</c> + per-block scale
/// <c>dx</c>), then (2) <c>matmul_q8_0_dp4a_pq</c> performs the GEMV reading the
/// shared <c>xq</c>/<c>dx</c> via <c>dotPacked4x8AccSatEXT</c>. This removes the
/// per-workgroup activation re-quantization of <see cref="MatMulQ8_0Dp4aKernel"/>
/// (which re-quantizes <c>x</c> M times and so regresses on high-M shapes like
/// lm_head). See <c>.docs/local-optimization-campaign.md</c> V2.
/// </para>
/// <para>
/// Requires <see cref="VulkanDevice.HasIntegerDotProduct"/>. The caller supplies
/// scratch buffers for <c>xq</c> (<c>K/4</c> uints) and <c>dx</c> (<c>K/32</c>
/// floats) so they can be reused across calls without per-call allocation.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0Dp4aPqKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int QuantPushConstantBytes = 2 * sizeof(uint); // K, blocksPerRow
    private const int GemvPushConstantBytes = 4 * sizeof(uint);  // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _quantModule;
    private readonly VulkanModule _gemvModule;
    private readonly ComputePipeline _quantPipeline;
    private readonly ComputePipeline _gemvPipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _quantCache;
    private readonly DescriptorSetCache _gemvCache;
    private bool _disposed;

    private MatMulQ8_0Dp4aPqKernel(
        VulkanDevice device,
        VulkanModule quantModule, ComputePipeline quantPipeline,
        VulkanModule gemvModule, ComputePipeline gemvPipeline,
        nint pool)
    {
        _device = device;
        _quantModule = quantModule;
        _gemvModule = gemvModule;
        _quantPipeline = quantPipeline;
        _gemvPipeline = gemvPipeline;
        _descriptorPool = pool;
        _quantCache = new DescriptorSetCache(device, pool, quantPipeline.DescriptorSetLayout, buffersPerSet: 3);
        _gemvCache = new DescriptorSetCache(device, pool, gemvPipeline.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads the prequant + DP4a GEMV SPIR-V and creates both pipelines.
    /// </summary>
    /// <param name="device">Device; must report <see cref="VulkanDevice.HasIntegerDotProduct"/>.</param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V blobs.</param>
    /// <returns>An initialized kernel.</returns>
    /// <exception cref="NotSupportedException">If the device lacks integer dot product support.</exception>
    /// <exception cref="FileNotFoundException">If a SPIR-V blob is missing.</exception>
    public static MatMulQ8_0Dp4aPqKernel Create(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            throw new NotSupportedException(
                "Device does not support VK_KHR_shader_integer_dot_product; use MatMulQ8_0Kernel instead.");

        string quantPath = Path.Combine(spvDir, "quantize_q8_act.spv");
        string gemvPath = Path.Combine(spvDir, "matmul_q8_0_dp4a_pq.spv");
        foreach (var p in new[] { quantPath, gemvPath })
            if (!File.Exists(p))
                throw new FileNotFoundException(
                    $"Vulkan SPIR-V not found: {p}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");

        var quantModule = VulkanModule.LoadFromFile(device, quantPath);
        VulkanModule? gemvModule = null;
        ComputePipeline? quantPipeline = null;
        ComputePipeline? gemvPipeline = null;
        try
        {
            Span<VkDescriptorBinding> quantBindings = stackalloc VkDescriptorBinding[3];
            quantBindings[0] = new VkDescriptorBinding(0);
            quantBindings[1] = new VkDescriptorBinding(1);
            quantBindings[2] = new VkDescriptorBinding(2);
            quantPipeline = quantModule.CreateComputePipeline("main", quantBindings, QuantPushConstantBytes);

            gemvModule = VulkanModule.LoadFromFile(device, gemvPath);
            Span<VkDescriptorBinding> gemvBindings = stackalloc VkDescriptorBinding[4];
            gemvBindings[0] = new VkDescriptorBinding(0);
            gemvBindings[1] = new VkDescriptorBinding(1);
            gemvBindings[2] = new VkDescriptorBinding(2);
            gemvBindings[3] = new VkDescriptorBinding(3);
            gemvPipeline = gemvModule.CreateComputePipeline("main", gemvBindings, GemvPushConstantBytes);
        }
        catch
        {
            gemvPipeline?.Dispose();
            quantPipeline?.Dispose();
            gemvModule?.Dispose();
            quantModule.Dispose();
            throw;
        }

        // One pool sized for the wider (4-buffer) set covers both caches.
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulQ8_0Dp4aPqKernel(device, quantModule, quantPipeline, gemvModule, gemvPipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _quantCache.Reset();
        _gemvCache.Reset();
    }

    /// <summary>
    /// Required <c>xq</c> scratch size in bytes for a given <paramref name="k"/>
    /// (packed INT8, 4 lanes per uint).
    /// </summary>
    /// <param name="k">Input dimension.</param>
    /// <returns>Byte length the <c>xq</c> scratch buffer must have.</returns>
    public static long XqScratchBytes(int k) => (long)(k / 4) * sizeof(uint);

    /// <summary>Required <c>dx</c> scratch size in bytes for a given <paramref name="k"/>.</summary>
    /// <param name="k">Input dimension.</param>
    /// <returns>Byte length the <c>dx</c> scratch buffer must have.</returns>
    public static long DxScratchBytes(int k) => (long)(k / Q8_0GroupSize) * sizeof(float);

    /// <summary>
    /// Dispatches prequant + GEMV synchronously (<c>vkQueueWaitIdle</c>).
    /// </summary>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="xqScratch">Scratch for packed INT8 activations; >= <see cref="XqScratchBytes"/>.</param>
    /// <param name="dxScratch">Scratch for per-block FP32 scales; >= <see cref="DxScratchBytes"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x,
        VulkanDevice.Buffer xqScratch, VulkanDevice.Buffer dxScratch, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, x, xqScratch, dxScratch, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the prequant dispatch, a compute→compute barrier, and the GEMV
    /// dispatch into <paramref name="cmdBuf"/> without submitting.
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="xqScratch">Scratch for packed INT8 activations; >= <see cref="XqScratchBytes"/>.</param>
    /// <param name="dxScratch">Scratch for per-block FP32 scales; >= <see cref="DxScratchBytes"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x,
        VulkanDevice.Buffer xqScratch, VulkanDevice.Buffer dxScratch, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        RecordQuantizeActivation(cmdBuf, x, xqScratch, dxScratch, k);

        // xq/dx written by the quantize shader must be visible to the GEMV shader.
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordGemvPrequant(cmdBuf, weightsQ8, xqScratch, dxScratch, y, m, k);
    }

    /// <summary>
    /// Records only the activation pre-quantization pass (<c>quantize_q8_act</c>):
    /// <c>x[K]</c> (FP32) → <c>xqScratch</c> (packed INT8) + <c>dxScratch</c>
    /// (per-32-block FP32 scale). No barrier is recorded — the caller must insert
    /// a compute→compute barrier before any <see cref="RecordGemvPrequant"/> that
    /// reads the scratch. Split out so one activation quant can be shared across
    /// several same-input projections (e.g. Q/K/V, gate/up) instead of being
    /// re-run per projection.
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="xqScratch">Scratch for packed INT8 activations; >= <see cref="XqScratchBytes"/>.</param>
    /// <param name="dxScratch">Scratch for per-block FP32 scales; >= <see cref="DxScratchBytes"/>.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void RecordQuantizeActivation(
        nint cmdBuf, VulkanDevice.Buffer x,
        VulkanDevice.Buffer xqScratch, VulkanDevice.Buffer dxScratch, int k)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (xqScratch.Size < XqScratchBytes(k))
            throw new ArgumentException($"xq scratch too small: need >= {XqScratchBytes(k)} bytes.", nameof(xqScratch));
        if (dxScratch.Size < DxScratchBytes(k))
            throw new ArgumentException($"dx scratch too small: need >= {DxScratchBytes(k)} bytes.", nameof(dxScratch));

        int blocksPerRow = k / Q8_0GroupSize;
        Span<nint> quantBuffers = stackalloc nint[3] { x.Handle, xqScratch.Handle, dxScratch.Handle };
        nint quantSet = _quantCache.GetOrCreate(quantBuffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _quantPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _quantPipeline.Layout, 0, 1, quantSet, 0, 0);
        Span<uint> quantPc = stackalloc uint[2] { (uint)k, (uint)blocksPerRow };
        fixed (uint* pcPtr = quantPc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _quantPipeline.Layout, VkShaderStageFlags.Compute, 0, QuantPushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)blocksPerRow, 1, 1);
    }

    /// <summary>
    /// Records only the DP4a GEMV pass (<c>matmul_q8_0_dp4a_pq</c>) reading a
    /// pre-quantized activation: <c>y[M] = W_q8[M,K] @ (xqScratch, dxScratch)</c>.
    /// The caller is responsible for having recorded a
    /// <see cref="RecordQuantizeActivation"/> for the same <c>k</c> plus a
    /// compute→compute barrier beforehand. Multiple calls with the same scratch
    /// and different (<paramref name="weightsQ8"/>, <paramref name="y"/>) share
    /// one activation quant — they need no barrier between them (distinct outputs,
    /// read-only scratch).
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes.</param>
    /// <param name="xqScratch">Packed INT8 activations from <see cref="RecordQuantizeActivation"/>.</param>
    /// <param name="dxScratch">Per-block FP32 scales from <see cref="RecordQuantizeActivation"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void RecordGemvPrequant(
        nint cmdBuf, VulkanDevice.Buffer weightsQ8,
        VulkanDevice.Buffer xqScratch, VulkanDevice.Buffer dxScratch, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        if (weightsQ8.Size < (long)m * rowBytes)
            throw new ArgumentException("Weights buffer too small.", nameof(weightsQ8));
        if (xqScratch.Size < XqScratchBytes(k))
            throw new ArgumentException($"xq scratch too small: need >= {XqScratchBytes(k)} bytes.", nameof(xqScratch));
        if (dxScratch.Size < DxScratchBytes(k))
            throw new ArgumentException($"dx scratch too small: need >= {DxScratchBytes(k)} bytes.", nameof(dxScratch));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> gemvBuffers = stackalloc nint[4] { weightsQ8.Handle, xqScratch.Handle, dxScratch.Handle, y.Handle };
        nint gemvSet = _gemvCache.GetOrCreate(gemvBuffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _gemvPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _gemvPipeline.Layout, 0, 1, gemvSet, 0, 0);
        Span<uint> gemvPc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, (uint)rowUints };
        fixed (uint* pcPtr = gemvPc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _gemvPipeline.Layout, VkShaderStageFlags.Compute, 0, GemvPushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _gemvPipeline.Dispose();
        _quantPipeline.Dispose();
        _gemvModule.Dispose();
        _quantModule.Dispose();
    }
}
