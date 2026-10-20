using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Prefill matmul for the IQ1/IQ2/IQ3 formats and NVFP4 (#1060): dequantise the weight matrix once into an F16 scratch buffer, then
/// run the F16 cooperative-matrix GEMM (issue #621). NVFP4's per-tensor <c>X.scale</c> multiplies the F32 output afterwards, because
/// the unscaled NVFP4 decode is exact in F16 and the scaled one is not.
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
    private Nvfp4DequantF16Kernel? _nvfp4;
    private ScaleInplaceF32Kernel? _scale;

    /// <summary>Calls served per format family, read by tests to prove the F16 path ran (#1060).</summary>
    internal static class Counters
    {
        /// <summary>IQ1/IQ2/IQ3 matmuls routed through the F16 GEMM.</summary>
        public static long Iq;
        /// <summary>NVFP4 matmuls routed through the F16 GEMM.</summary>
        public static long Nvfp4;
        /// <summary>Zeroes both counters.</summary>
        public static void Reset()
        {
            Interlocked.Exchange(ref Iq, 0);
            Interlocked.Exchange(ref Nvfp4, 0);
        }
    }

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

    /// <summary>Bytes of the F16 scratch allocated so far (0 until the first served matmul); read by tests to prove the native NVFP4
    /// GEMM (#1064) leaves it unallocated.</summary>
    internal long ScratchBytes => _scratch?.Size ?? 0;

    /// <summary>True for the formats this path serves: the IQ1/IQ2/IQ3 family and NVFP4 (#1060).</summary>
    public static bool Handles(QuantizationType qt) =>
        qt is QuantizationType.IQ1_S or QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS
            or QuantizationType.IQ2_S or QuantizationType.IQ3_XXS or QuantizationType.IQ3_S or QuantizationType.NVFP4;

    /// <summary>The K granule of <paramref name="qt"/>: 64 for NVFP4 blocks, 256 for the IQ super-blocks.</summary>
    private static int KGranule(QuantizationType qt) => qt == QuantizationType.NVFP4 ? QuantFormat.Nvfp4GroupSize : 256;

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
    /// Records <c>output[n,m] = scale * (input[n,k] x dequant(weights[m,k])^T)</c>. Returns false (recording nothing) when this path
    /// does not apply, so the caller can use its own GEMM.
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="qt">Device storage type of <paramref name="weights"/>.</param>
    /// <param name="weights">Packed weight matrix, <c>[m, k]</c>.</param>
    /// <param name="input">F32 activations, <c>[n, k]</c>.</param>
    /// <param name="output">F32 output, <c>[n, m]</c>.</param>
    /// <param name="m">Output rows of the weight.</param>
    /// <param name="k">Input width.</param>
    /// <param name="n">Tokens.</param>
    /// <param name="scale">Per-tensor output scale (NVFP4 <c>X.scale</c>); 1 for none.</param>
    /// <returns>Whether the matmul was recorded.</returns>
    public bool TryRecord(nint cmdBuf, QuantizationType qt, VulkanDevice.Buffer weights, VulkanDevice.Buffer input,
                          VulkanDevice.Buffer output, int m, int k, int n, float scale = 1.0f)
    {
        if (n < MinTokens || !Handles(qt) || (k % KGranule(qt)) != 0)
            return false;
        long elements = (long)m * k;
        if (elements > _capacityElements)
            return false;

        _scratch ??= _device.AllocateDeviceLocal(_capacityElements * sizeof(ushort));

        KernelSupport.ComputeToComputeBarrier(cmdBuf);     // previous GEMM's read of the scratch vs this write
        int superBlocks = checked((int)(elements / 256));
        switch (qt)
        {
            case QuantizationType.NVFP4:
                (_nvfp4 ??= Nvfp4DequantF16Kernel.Create(_device, _spvDir))
                    .Record(cmdBuf, weights, _scratch, checked((int)(elements / QuantFormat.Nvfp4GroupSize)));
                break;
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
        if (scale != 1.0f)
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf); // GEMM write vs in-place scale
            (_scale ??= ScaleInplaceF32Kernel.Create(_device, _spvDir)).Record(cmdBuf, output, checked(n * m), scale);
        }
        Interlocked.Increment(ref qt == QuantizationType.NVFP4 ? ref Counters.Nvfp4 : ref Counters.Iq);
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
        _nvfp4?.InvalidateDescriptorCache();
        _scale?.InvalidateDescriptorCache();
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
        _nvfp4?.Dispose();
        _scale?.Dispose();
        _scratch?.Dispose();
    }
}
