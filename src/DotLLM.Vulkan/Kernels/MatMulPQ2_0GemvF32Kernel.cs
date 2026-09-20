using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// PQ2_0 (PrismML Bonsai ternary) decode-path GEMV: <c>y[m] = Σ_g scale(m,g) · Σ_{c∈g} (code−1)·x[c]</c>.
/// </summary>
/// <remarks>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.MatMul.UnpackPQ2_0Row</c> /
/// <c>Dequantize.DequantizePQ2_0</c>: each 128 contiguous columns of a row form one 34-byte
/// group — 2 bytes little-endian fp16 group scale followed by 32 bytes of packed 2-bit ternary
/// codes (byte <c>b</c>, 0..31, packs 4 CONSECUTIVE codes for group-relative positions
/// <c>{4b, 4b+1, 4b+2, 4b+3}</c> at ASCENDING bit offsets <c>{0,2,4,6}</c>; value = code − 1 —
/// PrismML's real format, verified against their reference dequantize_row_q2_0, see #271).
/// Unlike I2_S (a single per-tensor tail scale applied once after the full row-dot), PQ2_0
/// applies each group's own scale to that group's partial dot product — the group scale is
/// read in-shader per 128-element span, not once at the end.
/// Row stride is <c>(K/128)·34</c> bytes; there is no per-tensor tail scale (contrast I2_S).
/// Activation <c>x</c> and output <c>y</c> are FP32. One workgroup per output row, 128 threads,
/// shared-memory tree reduce. Correctness-first baseline (no coopmat / dp4a MMVQ path yet) —
/// GEMV/decode only; GEMM/prefill is explicit follow-on scope (#205).
/// </remarks>
public sealed class MatMulPQ2_0GemvF32Kernel : IDisposable
{
    /// <summary>PQ2_0 group: 2 bytes fp16 scale + 32 bytes packed ternary codes.</summary>
    public const int PQ2_0GroupBytes = QuantFormat.PQ2_0BlockBytes;

    /// <summary>Elements per PQ2_0 group.</summary>
    public const int PQ2_0GroupSize = QuantFormat.TernaryGroupSize;

    private const int PushConstantBytes = 6 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulPQ2_0GemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_pq2_0_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulPQ2_0GemvF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, "matmul_pq2_0_f32_gemv.spv");

    /// <summary>
    /// Loads the named SPIR-V from <paramref name="spvDir"/> and creates the pipeline. The
    /// <paramref name="spvFileName"/> overload exists so a benchmark can A/B successive kernel
    /// variants side by side in one process.
    /// </summary>
    public static MatMulPQ2_0GemvF32Kernel Create(VulkanDevice device, string spvDir, string spvFileName)
    {
        string path = Path.Combine(spvDir, spvFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MatMulPQ2_0GemvF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsPQ2_0, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the PQ2_0 GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
        => Record(cmdBuf, weightsPQ2_0, x, y, m, k, xOffsetElements: 0, yOffsetElements: 0);

    /// <summary>
    /// Records the PQ2_0 GEMV for one row of a batched activation, reading
    /// <c>x[xOffsetElements .. +k]</c> and writing <c>y[yOffsetElements .. +m]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Issue #446. <see cref="VulkanDevice.Buffer"/> has no offset view, so a looped GEMV over an
    /// <c>[n, K]</c> activation batch expresses the per-token stride in push constants instead of
    /// by binding sub-ranges. Two consequences, both load-bearing for the loop being cheap: the
    /// three bound handles are identical across tokens so the handle-keyed
    /// <c>DescriptorSetCache</c> hits every time, and successive tokens touch disjoint
    /// <c>y</c> ranges so the dispatches need <b>no barriers</b> between them and are free to
    /// overlap.
    /// </para>
    /// <para>
    /// <b>Why a caller would want this.</b> The 128x128 coopmat GEMM does a 128-wide N-tile's
    /// worth of PQ2_0 unpack however few tokens it is given, so it is near-flat in <c>n</c> while
    /// the GEMV loop is linear. Measured on gfx1151 against the shipping
    /// <c>ladder_128x128x4</c> tile, the GEMM only overtakes the loop at <b>n ~ 4.4</b>
    /// (<c>lm_head</c>) / <b>n ~ 6.5</b> (<c>ffn_gate/up</c>) — so at the 2-8 token verify batches
    /// MTP speculative decoding produces, the single GEMM dispatch was the wrong call.
    /// </para>
    /// </remarks>
    /// <param name="cmdBuf">Command buffer to record into.</param>
    /// <param name="weightsPQ2_0">Packed PQ2_0 weights, <c>m</c> rows of <c>(k/128)*34</c> bytes.</param>
    /// <param name="x">Activation buffer; this dispatch reads <c>k</c> floats from <paramref name="xOffsetElements"/>.</param>
    /// <param name="y">Output buffer; this dispatch writes <c>m</c> floats at <paramref name="yOffsetElements"/>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Inner dimension; must be a multiple of 128.</param>
    /// <param name="xOffsetElements">First float of this token's activation row.</param>
    /// <param name="yOffsetElements">First float of this token's output row.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k, int xOffsetElements, int yOffsetElements)
    {
        if (xOffsetElements < 0) throw new ArgumentOutOfRangeException(nameof(xOffsetElements));
        if (yOffsetElements < 0) throw new ArgumentOutOfRangeException(nameof(yOffsetElements));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % PQ2_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {PQ2_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / PQ2_0GroupSize;
        long rowBytes = (long)blocksPerRow * PQ2_0GroupBytes;
        // 34 is not divisible by 4 — every group past the first straddles uint
        // boundaries. rowUints rounds up to cover safe straddle reads.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsPQ2_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes (m·(k/128)·34), got {weightsPQ2_0.Size}.",
                nameof(weightsPQ2_0));
        if (x.Size < ((long)xOffsetElements + k) * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < ((long)yOffsetElements + m) * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsPQ2_0.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[6]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)rowUints,
            (uint)xOffsetElements,
            (uint)yOffsetElements,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
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
        _pipeline.Dispose();
        _module.Dispose();
    }
}
