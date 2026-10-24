using System;
using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_0 MMVQ decode-path GEMV via integer dp4a (issue #1018) against the coalesced
/// lane=K-position layout (issue #339). Q4_0 reconstructs each weight as d * (nibble - 8),
/// an int8 times an fp16 scale — ideal for dp4a. The
/// activation must already be Q8_1 (<see cref="QuantizeQ8_1Kernel"/>). Created only
/// when the device advertises <see cref="VulkanDevice.HasIntegerDotProduct"/>; the
/// router falls back to <see cref="MatMulF32Kernel"/> otherwise. NOT
/// bit-exact vs the F32-in GEMV — validated argmax + tolerance.
/// </summary>
public sealed class MatMulQ4_0MmvqKernel : IDisposable
{
    /// <summary>Q4_0 block bytes.</summary>
    public const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    /// <summary>Elements per Q4_0 block.</summary>
    public const int Q4_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, pad

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    /// <summary>
    /// Default shader generation: V2 (a lane owns a whole 18-byte block; ~3x fewer load instructions per weight byte, issue #1018) unless
    /// <c>DOTLLM_VK_Q4_0_V2=0</c> selects the V1 4-lanes-per-block shader (A/B).
    /// </summary>
    internal static bool V2Default { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4_0_V2") != "0";

    /// <summary>True when this instance runs the V2 shader.</summary>
    public bool IsV2 { get; private set; }

    /// <summary>The legacy 4-bit format this instance reads: <see cref="QuantizationType.Q4_0"/> (18 B blocks) or
    /// <see cref="QuantizationType.Q4_1"/> (20 B blocks, issue #1171: same kernel shape, value <c>d * nibble + m</c>).</summary>
    public QuantizationType Quant { get; private init; } = QuantizationType.Q4_0;

    private int BlockBytes => Quant == QuantizationType.Q4_1 ? QuantFormat.Q4_1BlockBytes : QuantFormat.Q4_0BlockBytes;

    private static string ShaderFor(string q4_0Name, QuantizationType quant) => quant switch
    {
        QuantizationType.Q4_0 => q4_0Name,
        QuantizationType.Q4_1 => q4_0Name.Replace("q4_0", "q4_1", StringComparison.Ordinal),
        _ => throw new ArgumentOutOfRangeException(nameof(quant), quant, "Only Q4_0 and Q4_1 are served by this kernel."),
    };

    private MatMulQ4_0MmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>matmul_q4_0_mmvq[_v2].spv</c> (Q4_1: <c>matmul_q4_1_mmvq_v2.spv</c>, V2 only); null when the SPV is missing or the device
    /// lacks integer-dot-product support.</summary>
    /// <param name="device">Vulkan device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="v2">Shader generation override (null = <see cref="V2Default"/>; ignored for Q4_1, which has only V2).</param>
    /// <param name="quant"><see cref="QuantizationType.Q4_0"/> (default) or <see cref="QuantizationType.Q4_1"/>.</param>
    /// <returns>The kernel, or null.</returns>
    public static MatMulQ4_0MmvqKernel? TryCreate(VulkanDevice device, string spvDir, bool? v2 = null, QuantizationType quant = QuantizationType.Q4_0)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        bool wantV2 = quant == QuantizationType.Q4_1 || (v2 ?? V2Default);
        string path = Path.Combine(spvDir, ShaderFor(wantV2 ? "matmul_q4_0_mmvq_v2.spv" : "matmul_q4_0_mmvq.spv", quant));
        if (!File.Exists(path))
            return null;

        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
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
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: requiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulQ4_0MmvqKernel(device, module, pipeline, pool) { IsV2 = wantV2, Quant = quant };
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the Q4_0 MMVQ GEMV into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q4_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4_0GroupSize;
        long rowBytes = (long)blocksPerRow * BlockBytes;

        if (weights.Size < (long)m * rowBytes)
            throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4] { weights.Handle, xq.Handle, xds.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, 0u };
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
