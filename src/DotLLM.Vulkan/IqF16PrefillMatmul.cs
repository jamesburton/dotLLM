using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Prefill matmul for the IQ1/IQ2/IQ3 formats: dequantise the weight matrix once into an F16 scratch buffer, then run the F16
/// cooperative-matrix GEMM (issue #621).
/// </summary>
/// <remarks>
/// <para><b>Why.</b> Only IQ4_XS has a blocked coopmat GEMM (#601); the other IQ formats went through a scalar tiled GEMM that
/// dequantises inside its inner loop: ~70 tok/s prefill on Llama-3.2-3B (pp128), slower than their own decode (82-116 tok/s) and
/// ~25x below the K-quant coopmat path. The traffic of this two-pass scheme (F16 write + read per weight element) is ~2 matmuls'
/// worth of memory time at n = 128, which is small against the 25x it replaces, and the compute runs at coopmat speed.</para>
/// <para><b>Hazards.</b> The scratch is a single fixed-capacity buffer reused by every matmul, so each call starts with a
/// compute-to-compute barrier (previous GEMM's read of the scratch vs this dequant's write) and puts one between dequant and GEMM.
/// Capacity is fixed up front because a buffer freed mid-recording would still be referenced by already-recorded dispatches.
/// A matrix larger than the capacity (e.g. a vocabulary-sized all-row logits projection) declines, and the caller falls back to
/// the original GEMM.</para>
/// <para>Opt-out: <c>DOTLLM_VULKAN_DISABLE_IQ_F16_PREFILL=1</c>. Minimum batch: <c>DOTLLM_VULKAN_IQ_F16_PREFILL_MIN_TOKENS</c>
/// (default <see cref="DefaultMinTokens"/>).</para>
/// </remarks>
internal sealed class IqF16PrefillMatmul : IDisposable
{
    public const string DisableEnvVar = "DOTLLM_VULKAN_DISABLE_IQ_F16_PREFILL";
    public const string MinTokensEnvVar = "DOTLLM_VULKAN_IQ_F16_PREFILL_MIN_TOKENS";

    /// <summary>Below this many tokens the dequant pass costs more than the old GEMM saves.</summary>
    public const int DefaultMinTokens = 32;

    private readonly VulkanDevice _device;
    private readonly string _spvDir;
    private readonly MatMulF16GemmCoopmatKernel _gemm;
    private readonly long _capacityElements;
    private VulkanDevice.Buffer? _scratch;

    private Iq1SDequantF16Kernel? _iq1S;
    private Iq2XxsDequantF16Kernel? _iq2Xxs;
    private Iq2XsDequantF16Kernel? _iq2Xs;
    private Iq2SDequantF16Kernel? _iq2S;
    private Iq3XxsDequantF16Kernel? _iq3Xxs;
    private Iq3SDequantF16Kernel? _iq3S;

    private IqF16PrefillMatmul(VulkanDevice device, string spvDir, MatMulF16GemmCoopmatKernel gemm,
                               long capacityElements, int minTokens)
    {
        _device = device;
        _spvDir = spvDir;
        _gemm = gemm;
        _capacityElements = capacityElements;
        MinTokens = minTokens;
    }

    /// <summary>Smallest token count routed here.</summary>
    public int MinTokens { get; }

    /// <summary>True for the IQ formats this path serves.</summary>
    public static bool Handles(QuantizationType qt) =>
        qt is QuantizationType.IQ1_S or QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS
            or QuantizationType.IQ2_S or QuantizationType.IQ3_XXS or QuantizationType.IQ3_S;

    /// <summary>
    /// Creates the helper, or null when disabled, when the device has no cooperative-matrix F16 GEMM, or when the shaders are absent.
    /// </summary>
    /// <param name="device">Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <param name="gemm">The model's F16 cooperative-matrix GEMM (null when the device lacks one).</param>
    /// <param name="maxLayerElements">Largest <c>m * k</c> of any per-layer matmul (the scratch capacity in F16 elements).</param>
    public static IqF16PrefillMatmul? TryCreate(VulkanDevice device, string spvDir, MatMulF16GemmCoopmatKernel? gemm,
                                                long maxLayerElements)
    {
        if (gemm is null || Environment.GetEnvironmentVariable(DisableEnvVar) == "1")
            return null;
        if (!File.Exists(Path.Combine(spvDir, "iq2_xxs_dequant_f16.spv")))
            return null;

        int minTokens = int.TryParse(Environment.GetEnvironmentVariable(MinTokensEnvVar), out int v) && v > 0
            ? v : DefaultMinTokens;
        return new IqF16PrefillMatmul(device, spvDir, gemm, maxLayerElements, minTokens);
    }

    /// <summary>
    /// Records <c>output[n,m] = input[n,k] x dequant(weights[m,k])^T</c>. Returns false (recording nothing) when this path does not
    /// apply, so the caller can use its own GEMM.
    /// </summary>
    public bool TryRecord(nint cmdBuf, QuantizationType qt, VulkanDevice.Buffer weights, VulkanDevice.Buffer input,
                          VulkanDevice.Buffer output, int m, int k, int n)
    {
        if (n < MinTokens || !Handles(qt) || (k % 256) != 0)
            return false;
        long elements = (long)m * k;
        if (elements > _capacityElements)
            return false;

        _scratch ??= _device.AllocateDeviceLocal(_capacityElements * sizeof(ushort));

        KernelSupport.ComputeToComputeBarrier(cmdBuf);     // previous GEMM's read of the scratch vs this write
        int superBlocks = checked((int)(elements / 256));
        switch (qt)
        {
            case QuantizationType.IQ1_S: (_iq1S ??= Iq1SDequantF16Kernel.Create(_device, _spvDir)).Record(cmdBuf, weights, _scratch, superBlocks); break;
            case QuantizationType.IQ2_XXS: (_iq2Xxs ??= Iq2XxsDequantF16Kernel.Create(_device, _spvDir)).Record(cmdBuf, weights, _scratch, superBlocks); break;
            case QuantizationType.IQ2_XS: (_iq2Xs ??= Iq2XsDequantF16Kernel.Create(_device, _spvDir)).Record(cmdBuf, weights, _scratch, superBlocks); break;
            case QuantizationType.IQ2_S: (_iq2S ??= Iq2SDequantF16Kernel.Create(_device, _spvDir)).Record(cmdBuf, weights, _scratch, superBlocks); break;
            case QuantizationType.IQ3_XXS: (_iq3Xxs ??= Iq3XxsDequantF16Kernel.Create(_device, _spvDir)).Record(cmdBuf, weights, _scratch, superBlocks); break;
            case QuantizationType.IQ3_S: (_iq3S ??= Iq3SDequantF16Kernel.Create(_device, _spvDir)).Record(cmdBuf, weights, _scratch, superBlocks); break;
            default: return false;
        }

        KernelSupport.ComputeToComputeBarrier(cmdBuf);     // dequant write vs GEMM read
        _gemm.Record(cmdBuf, _scratch, input, output, m, k, n);
        return true;
    }

    /// <summary>Drops cached descriptor sets of the dequant kernels (scratch buffers were re-allocated by the owner).</summary>
    public void InvalidateDescriptorCache()
    {
        _iq1S?.InvalidateDescriptorCache();
        _iq2Xxs?.InvalidateDescriptorCache();
        _iq2Xs?.InvalidateDescriptorCache();
        _iq2S?.InvalidateDescriptorCache();
        _iq3Xxs?.InvalidateDescriptorCache();
        _iq3S?.InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _iq1S?.Dispose();
        _iq2Xxs?.Dispose();
        _iq2Xs?.Dispose();
        _iq2S?.Dispose();
        _iq3Xxs?.Dispose();
        _iq3S?.Dispose();
        _scratch?.Dispose();
    }
}
