using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Vulkan;

/// <summary>
/// Pre-load residency verdict for a Qwen4-Exp GGUF on a Vulkan device (#818): does everything the device will hold fit in
/// <see cref="VulkanDevice.ResidentCapacityBytes"/> minus a headroom, with the n-gram table excluded (it stays on the host)?
/// Pure arithmetic, no GPU calls, so it is unit-testable.
/// </summary>
/// <remarks>
/// <para>
/// Why a hard gate: on a UMA iGPU the "device" memory is system RAM. The 122B-class failure (0.01 tok/s) was a model whose weights
/// plus working set exceeded what WDDM could keep resident - it did not fail to load, it paged. Refusing at load time with the
/// numbers is strictly better than a silent thrash. <c>DOTLLM_VK_ALLOW_OVERCOMMIT=1</c> turns the refusal into a warning.
/// </para>
/// <para>
/// The estimate mirrors the upload policy rather than the file size: expert banks whose quant has no resident indexed kernel
/// (anything but Q4_K/Q5_K/Q6_K today - Q5_1 / Q8_0 / IQ*) are WIDENED to F32 on upload, which is exactly what makes the real
/// UD-Q4_K_XL file (Q5_1 down experts) not fit until those banks get resident kernels; the estimate must say so up front.
/// </para>
/// </remarks>
internal readonly record struct Qwen4ExpResidencyPlan(
    long DeviceWeightBytes, long KvAndScratchBytes, long HostOnlyBytes,
    long CapacityBytes, long HeadroomBytes, long PhysicalRamBytes)
{
    /// <summary>Bytes the device must hold: weights + KV + scratch.</summary>
    public long RequiredBytes => DeviceWeightBytes + KvAndScratchBytes;

    /// <summary>Usable resident budget: capacity minus headroom.</summary>
    public long BudgetBytes => Math.Max(0, CapacityBytes - HeadroomBytes);

    /// <summary>True when <see cref="RequiredBytes"/> fits the budget.</summary>
    public bool Fits => RequiredBytes <= BudgetBytes;

    /// <summary>True when device weights plus the (possibly fully page-cache-warm) host-only table exceed physical RAM less headroom (UMA only matters).</summary>
    public bool HostPressure => PhysicalRamBytes > 0 && RequiredBytes + HostOnlyBytes > Math.Max(0, PhysicalRamBytes - HeadroomBytes);

    /// <summary>Default headroom: 6 GiB or 5% of the capacity, whichever is larger.</summary>
    public static long DefaultHeadroom(long capacityBytes) => Math.Max(6L << 30, capacityBytes / 20);

    /// <summary>Human-readable verdict used in the refusal / warning text.</summary>
    public string Describe()
    {
        static string G(long b) => $"{b / (double)(1L << 30):F1} GiB";
        return $"device needs {G(RequiredBytes)} (weights {G(DeviceWeightBytes)} + KV/scratch {G(KvAndScratchBytes)}) against a resident " +
               $"budget of {G(BudgetBytes)} (capacity {G(CapacityBytes)} - headroom {G(HeadroomBytes)}); host-only n-gram table {G(HostOnlyBytes)} " +
               $"(never uploaded); physical RAM {G(PhysicalRamBytes)}";
    }

    /// <summary>Quant types whose routed expert banks stay packed on the device (mirrors <c>VulkanQwen3MoeMoeUpload</c>).</summary>
    private static bool BankStaysPacked(QuantizationType qt)
        => qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;

    /// <summary>Estimates device bytes for the tensors Qwen4Exp uploads, from the GGUF tensor table. Host-only tensors are excluded.</summary>
    public static (long DeviceBytes, long HostOnlyBytes) EstimateWeights(
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ModelConfig config)
    {
        long device = 0, hostOnly = 0;
        foreach (var (name, d) in tensors)
        {
            long elems = 1;
            for (int i = 0; i < d.Shape.Rank; i++) elems *= d.Shape[i];
            long packed = d.QuantizationType == QuantizationType.F32 ? elems * 4
                : Dequantize.RowByteSize(d.Shape[0], d.QuantizationType) * (elems / d.Shape[0]);

            if (name == Qwen4ExpTensors.PerLayerTokenEmbd) { hostOnly += packed; continue; }
            // PLE projections / norms / conv run on the host branch; the dense indexer is not used by the V1 dense QSA fallback.
            if (name.Contains(".ple_", StringComparison.Ordinal) || name.Contains(".indexer.", StringComparison.Ordinal))
            { hostOnly += packed; continue; }
            if (name.Contains(".nextn.", StringComparison.Ordinal)) continue;   // MTP block: not loaded by V1

            if (name.Contains("_exps.weight", StringComparison.Ordinal))
                device += BankStaysPacked(d.QuantizationType) ? packed : elems * 4;
            else if (name == Qwen4ExpTensors.TokenEmbd)
                device += elems * 4;                                        // the embedding gather table is always widened to F32
            else if (name.Contains("_shexp.weight", StringComparison.Ordinal))
                device += elems * 2 + (d.QuantizationType == QuantizationType.Q8_0 && !name.Contains("_down_", StringComparison.Ordinal) ? packed : 0);   // F16 copy + decode-only Q8_0 gate/up alias
            else if (d.Shape.Rank >= 2 && name.EndsWith(".weight", StringComparison.Ordinal))
                device += VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(d.Shape[1], d.Shape[0], d.QuantizationType);
            else
                device += elems * 4;
        }
        return (device, hostOnly);
    }

    /// <summary>Dense-attention KV cache bytes for <paramref name="positions"/> positions across the QSA layers (F32 K and V).</summary>
    public static long KvBytes(ModelConfig config, int positions)
    {
        var layout = config.HybridLayout!;
        long layers = layout.LayerKind.Count(k => k == HybridLayerKind.Attention);
        return layers * 2L * positions * config.NumKvHeads * config.HeadDim * 4;
    }

    /// <summary>
    /// Builds the plan for <paramref name="tensors"/> on a device with <paramref name="capacityBytes"/> resident capacity.
    /// </summary>
    public static Qwen4ExpResidencyPlan Create(
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ModelConfig config,
        long capacityBytes, int kvPositions, long physicalRamBytes, long? headroomBytes = null)
    {
        var (device, hostOnly) = EstimateWeights(tensors, config);
        // Scratch: the prefill chunk working set (routed-expert intermediates dominate) - a flat allowance plus per-position KV.
        long scratch = 1L << 30;
        return new Qwen4ExpResidencyPlan(device, KvBytes(config, kvPositions) + scratch, hostOnly,
            capacityBytes, headroomBytes ?? DefaultHeadroom(capacityBytes), physicalRamBytes);
    }
}
