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
/// (anything but Q4_K/Q5_K/Q6_K/Q5_1/Q8_0 today - e.g. IQ*) are WIDENED to F32 on upload, which is what made the real
/// UD-Q4_K_XL file (Q5_1 down experts) not fit before #849; the estimate must say so up front.
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

    /// <summary>GPU memory other processes hold on this adapter right now (#880); 0 when unknown.</summary>
    public long OtherProcessBytes { get; init; }

    /// <summary>"pid 1234 (llama-server) 31.2 GiB, ..." naming the top holders; null when unknown.</summary>
    public string? OtherProcessDetail { get; init; }

    /// <summary>True when <see cref="RequiredBytes"/> plus what other processes hold fits the budget.</summary>
    public bool Fits => RequiredBytes + OtherProcessBytes <= BudgetBytes;

    /// <summary>Bytes by which <see cref="RequiredBytes"/> + other processes exceed the budget (0 when it fits).</summary>
    public long ShortfallBytes => Math.Max(0, RequiredBytes + OtherProcessBytes - BudgetBytes);

    /// <summary>
    /// Post-upload check (#880): given what this process actually holds after the weights are resident, does it still fit
    /// beside the other processes? Returns the shortfall in bytes (0 = fine).
    /// </summary>
    public static long PostUploadShortfall(long ourBytes, long otherBytes, long capacityBytes, long headroomBytes)
        => Math.Max(0, ourBytes + otherBytes - Math.Max(0, capacityBytes - headroomBytes));

    /// <summary>
    /// Rows the per-forward scratch is pre-sized to at load (<c>DOTLLM_VK_PLANNED_ROWS</c>, default 1024: ~0.55 GiB, because on a 127 GiB box the real file leaves only ~1.3 GiB under the OS limit), clamped to
    /// <paramref name="kvCapacity"/>. Scratch otherwise grows lazily on the first larger forward - AFTER the weights already fill the
    /// device-local heap - and a 512-row then 1024-row call sequence was seen to end in VK_ERROR_DEVICE_LOST (#880).
    /// </summary>
    public static int PlannedRows(int kvCapacity)
        => Math.Clamp(int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_PLANNED_ROWS"), out int v) && v > 0 ? v : 1024, 1, Math.Max(1, kvCapacity));

    /// <summary>Error text for a scratch (re)allocation that failed after the weights were resident.</summary>
    public static string ScratchGrowthMessage(int rows, string inner)
        => $"growing the per-forward scratch to {rows} rows failed after the weights were loaded ({inner}). The device-local heap is nearly full of " +
           $"weights, so lazily growing scratch can run out of memory (or end in VK_ERROR_DEVICE_LOST). Lower the prompt/chunk size, close other GPU " +
           $"consumers, or set DOTLLM_VK_PLANNED_ROWS>={rows} so the scratch is allocated (and counted by the residency check) at load time.";

    /// <summary>The refusal / warning text naming the shortfall and the likely culprits.</summary>
    public string DescribeShortfall()
    {
        static string G(long b) => $"{b / (double)(1L << 30):F1} GiB";
        string others = OtherProcessBytes > 0
            ? $"Other processes already hold {G(OtherProcessBytes)} of GPU memory ({OtherProcessDetail ?? "unnamed"}). "
            : "";
        return $"short by {G(ShortfallBytes)} once other GPU users are counted. {others}" +
               "Close the other GPU consumers (a second dotllm, llama.cpp, Lemonade, Docker, ollama, a browser) or lower the context; " +
               "oversubscribing GPU memory can end in VK_ERROR_DEVICE_LOST.";
    }

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
               $"(never uploaded); physical RAM {G(PhysicalRamBytes)}" +
               (OtherProcessBytes > 0 ? $"; other processes' GPU memory {G(OtherProcessBytes)}" : "");
    }

    /// <summary>Quant types whose routed expert banks stay packed on the device (mirrors <c>VulkanQwen3MoeMoeUpload</c>).</summary>
    private static bool BankStaysPacked(QuantizationType qt, int kDim)
        => VulkanQwen3MoeMoeUpload.BankStaysPacked(qt, kDim);

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
            // PLE projections / norms / conv run on the host branch. The QSA indexer projections are device weights (#819).
            if (name.Contains(".ple_", StringComparison.Ordinal))
            { hostOnly += packed; continue; }
            if (name.Contains(".nextn.", StringComparison.Ordinal)) continue;   // MTP block: not loaded by V1

            if (name.Contains("_exps.weight", StringComparison.Ordinal))
                device += BankStaysPacked(d.QuantizationType, d.Shape[0]) ? packed : elems * 4;
            else if (name == Qwen4ExpTensors.TokenEmbd)
            {
                if (VulkanQwen4ExpTransformerModel.HostEmbedding) hostOnly += packed;   // rows are gathered on the host from the mmap'd table (#819)
                else device += elems * 4;                                                // the device gather table is widened to F32
            }
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
    /// Device bytes of everything that grows with context for a sequence: the F32 K/V rows of the QSA layers plus the indexer's raw and
    /// pooled keys (#819). Raw keys are one row per position, pooled keys one per block.
    /// </summary>
    public static long ContextBytes(ModelConfig config, int positions)
    {
        var layout = config.HybridLayout!;
        long layers = layout.LayerKind.Count(k => k == HybridLayerKind.Attention);
        long idx = config.Qwen4Exp is { } q4 && q4.IndexerBlockSize > 0
            ? VulkanQwen4ExpIndexerState.BytesFor((int)layers, positions, q4.IndexerKeyLength, q4.IndexerBlockSize) : 0;
        return KvBytes(config, positions) + idx;
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
        return new Qwen4ExpResidencyPlan(device, ContextBytes(config, kvPositions) + scratch, hostOnly,
            capacityBytes, headroomBytes ?? DefaultHeadroom(capacityBytes), physicalRamBytes);
    }
}
