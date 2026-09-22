using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Issue #496 — int8-activation PQ2_0 decode GEMV via <c>VK_KHR_shader_integer_dot_product</c>
/// (<c>dotPacked4x8AccSatEXT</c>). The Vulkan twin of CUDA #485
/// (<c>native/kernels/pq2_0_gemv_dp4a.cu</c>), which measured +10% plain decode and +25% MTP on
/// Bonsai 2 27B with byte-identical greedy output.
/// </summary>
/// <remarks>
/// <para>
/// The activations are quantized once per projection input to int8 with a per-32 group scale (the
/// CPU W2A8 tier — <see cref="QuantizePQ2_0Int8Kernel"/>), and each 4 ternary weights × 4
/// activations become one packed integer dot instead of four float FMAs plus four int→float
/// converts. Per 32-element block per column the whole cost is 8 dot instructions, one I2F and one
/// FMA; the weight decode is shared by every column.
/// </para>
/// <para>
/// <b>Not bit-exact</b> with the float GEMV — the activation is quantized first. That is also the
/// cheap way to prove the kernel actually ran: an identical result means the old shader was
/// dispatched (or a stale <c>.spv</c> was copied into <c>bin/</c>).
/// </para>
/// <para>
/// <b>Opt-in.</b> <see cref="TryCreate"/> returns <c>null</c> unless the device advertises
/// <see cref="VulkanDevice.HasIntegerDotProduct"/>, the SPIR-V is present, and
/// <see cref="EnvVar"/> (<c>DOTLLM_VK_PQ2_0_INT8</c>) is <c>1</c> — or
/// <see cref="EnabledGlobalOverride"/> is set, which exists so a benchmark can A/B both paths in
/// one session.
/// </para>
/// <para>
/// One compiled variant per column count 1..8 (<see cref="MatMulPQ2_0GemvF32Kernel.MaxColumns"/>),
/// each 4 output rows per 64-lane workgroup: #470 measured 2/4/8-only variants costing ~40% at
/// S = 5 from the dead columns.
/// </para>
/// </remarks>
public sealed class MatMulPQ2_0Int8GemvKernel : IDisposable
{
    /// <summary>Environment variable enabling the int8 path; <c>1</c> turns it on.</summary>
    public const string EnvVar = "DOTLLM_VK_PQ2_0_INT8";

    /// <summary>Output rows per workgroup.</summary>
    public const int Rows = 4;

    /// <summary>Workgroup size the shaders are compiled with.</summary>
    public const int WorkgroupSize = 64;

    private const int PushConstantBytes = 6 * sizeof(uint); // M, K, blocksPerRow, qBlocksPerCol, yOff, ncols

    private static readonly bool EnabledFromEnv =
        Environment.GetEnvironmentVariable(EnvVar) is "1";

    /// <summary>
    /// In-process override of the env gate, read by <see cref="TryCreate"/>. A benchmark sets it
    /// before constructing a model so both arms of an A/B live in one session.
    /// </summary>
    public static bool? EnabledGlobalOverride { get; set; }

    /// <summary>Whether the int8 path is switched on for newly created kernels.</summary>
    public static bool Enabled => EnabledGlobalOverride ?? EnabledFromEnv;

    private readonly VulkanDevice _device;
    private readonly Variant[] _variants;   // index = columns - 1
    private bool _disposed;

    private MatMulPQ2_0Int8GemvKernel(VulkanDevice device, Variant[] variants)
    {
        _device = device;
        _variants = variants;
    }

    /// <summary>One compiled column-count variant and its descriptor state.</summary>
    private sealed class Variant : IDisposable
    {
        public required VulkanDevice Device { get; init; }
        public required VulkanModule Module { get; init; }
        public required ComputePipeline Pipeline { get; init; }
        public required nint Pool { get; init; }
        public required DescriptorSetCache Cache { get; init; }

        public void Dispose()
        {
            if (Pool != 0)
                VulkanApi.vkDestroyDescriptorPool(Device.Handle, Pool, 0);
            Pipeline.Dispose();
            Module.Dispose();
        }
    }

    /// <summary>
    /// Loads <c>matmul_pq2_0_int8_gemv_r4_c{1..8}_w64.spv</c> from <paramref name="spvDir"/>.
    /// Returns <c>null</c> when the int8 path is switched off, the device has no integer dot
    /// product, or any SPIR-V is missing — the caller then uses the float GEMV.
    /// </summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V modules.</param>
    public static MatMulPQ2_0Int8GemvKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (!Enabled) return null;
        if (!device.HasIntegerDotProduct) return null;

        var variants = new List<Variant>(MatMulPQ2_0GemvF32Kernel.MaxColumns);
        // One stackalloc for the whole loop (CA2014); the four bindings are identical per variant.
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
        bindings[0] = new VkDescriptorBinding(0);
        bindings[1] = new VkDescriptorBinding(1);
        bindings[2] = new VkDescriptorBinding(2);
        bindings[3] = new VkDescriptorBinding(3);
        try
        {
            for (int c = 1; c <= MatMulPQ2_0GemvF32Kernel.MaxColumns; c++)
            {
                string path = Path.Combine(spvDir, $"matmul_pq2_0_int8_gemv_r4_c{c}_w64.spv");
                if (!File.Exists(path))
                {
                    foreach (var v in variants) v.Dispose();
                    return null;
                }

                var module = VulkanModule.LoadFromFile(device, path);
                ComputePipeline pipeline;
                try
                {
                    pipeline = module.CreateComputePipeline(
                        entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
                }
                catch
                {
                    module.Dispose();
                    throw;
                }

                nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
                variants.Add(new Variant
                {
                    Device = device,
                    Module = module,
                    Pipeline = pipeline,
                    Pool = pool,
                    Cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4),
                });
            }
        }
        catch
        {
            foreach (var v in variants) v.Dispose();
            throw;
        }

        return new MatMulPQ2_0Int8GemvKernel(device, variants.ToArray());
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated (#467).</summary>
    internal void InvalidateDescriptorCache()
    {
        foreach (var v in _variants) v.Cache.Reset();
    }

    /// <summary>
    /// Records <c>y[s, :] = W @ x_int8[s, :]</c> for <paramref name="columns"/> columns of an
    /// already-quantized activation batch.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="weightsPQ2_0">Packed PQ2_0 weights, <paramref name="m"/> rows of <c>(k/128)*34</c> bytes.</param>
    /// <param name="xq">Permuted packed-int8 activations from <see cref="QuantizePQ2_0Int8Kernel"/>, compacted.</param>
    /// <param name="xmeta">Per-32-block <c>(scale bits, sum q)</c> from the same quantizer.</param>
    /// <param name="y">Output, <c>[columns, m]</c> row-major from <paramref name="yOffsetElements"/>.</param>
    /// <param name="m">Output rows per column.</param>
    /// <param name="k">Inner dimension; must be a multiple of 128.</param>
    /// <param name="columns">Token count, 1..<see cref="MatMulPQ2_0GemvF32Kernel.MaxColumns"/>.</param>
    /// <param name="yOffsetElements">First float of column 0's output row.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer xq, VulkanDevice.Buffer xmeta,
        VulkanDevice.Buffer y,
        int m, int k, int columns, int yOffsetElements = 0)
    {
        ArgumentNullException.ThrowIfNull(weightsPQ2_0);
        ArgumentNullException.ThrowIfNull(xq);
        ArgumentNullException.ThrowIfNull(xmeta);
        ArgumentNullException.ThrowIfNull(y);
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (columns < 1 || columns > _variants.Length) throw new ArgumentOutOfRangeException(nameof(columns));
        if (yOffsetElements < 0) throw new ArgumentOutOfRangeException(nameof(yOffsetElements));
        if ((k % MatMulPQ2_0GemvF32Kernel.PQ2_0GroupSize) != 0)
            throw new ArgumentException(
                $"k must be a multiple of {MatMulPQ2_0GemvF32Kernel.PQ2_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / MatMulPQ2_0GemvF32Kernel.PQ2_0GroupSize;
        long rowBytes = (long)blocksPerRow * MatMulPQ2_0GemvF32Kernel.PQ2_0GroupBytes;
        long weightsMin = (long)m * rowBytes;
        if (weightsPQ2_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes (m·(k/128)·34), got {weightsPQ2_0.Size}.",
                nameof(weightsPQ2_0));
        if (xq.Size < QuantizePQ2_0Int8Kernel.PackedBytes(k, columns))
            throw new ArgumentException("Quantized activation buffer too small.", nameof(xq));
        if (xmeta.Size < QuantizePQ2_0Int8Kernel.MetaBytes(k, columns))
            throw new ArgumentException("Activation metadata buffer too small.", nameof(xmeta));
        if (y.Size < ((long)yOffsetElements + (long)columns * m) * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Variant variant = _variants[columns - 1];

        Span<nint> buffers = stackalloc nint[4]
            { weightsPQ2_0.Handle, xq.Handle, xmeta.Handle, y.Handle };
        nint descriptorSet = variant.Cache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, variant.Pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, variant.Pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[6]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)(k / QuantizePQ2_0Int8Kernel.GroupSize),
            (uint)yOffsetElements,
            (uint)columns,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, variant.Pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Rows - 1) / Rows), 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var v in _variants) v.Dispose();
        _ = _device;
    }
}
