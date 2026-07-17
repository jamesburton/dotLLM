using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer streaming uploader for Qwen3MoeHybrid routed MoE banks. At
/// Qwen3.6-35B-A3B scale (256 experts × 40 layers × 3 matrices × ~1M elems)
/// a fully-resident F32 routed-expert layout would consume ~120 GB of device
/// memory; we instead keep weights HOST-side as raw quant views inside
/// <see cref="MoeLayerWeights"/> and dequantise+upload one layer's banks
/// right before each MoE dispatch.
/// </summary>
/// <remarks>
/// <para>
/// The returned <see cref="LayerBundle"/> is <see cref="IDisposable"/>. The
/// model's default policy is to dispose after each forward (streaming mode);
/// opt into per-layer resident caching with
/// <c>DOTLLM_VK_MOE_RESIDENT=1</c>, in which case the bundle survives for
/// the lifetime of the model.
/// </para>
/// <para>
/// <b>Resident-quant overlay (Q6_K, Q4_K).</b> When the caller asks for a
/// resident-quant upload (<c>residentQuant: true</c>) AND the source MoE
/// banks are uniformly one of the supported quant types, the routed banks
/// are uploaded as raw quant bytes (block-sized per <c>Dequantize.RowByteSize</c>,
/// layout matching the corresponding <c>DotLLM.Cpu.Kernels.Dequantize*</c>
/// scalar reference) so a matching device-side indexed matmul kernel
/// (<see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ6_KF32Kernel"/> or
/// <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ4_KF32Kernel"/>) can
/// dequantise per row. This is what lets multi-GB routed banks (e.g.
/// Qwen3.6-A3B-Q6_K_XL: ~25 GB Q6_K vs ~120 GB F32; UD-Q4_K_XL: ~15 GB Q4_K)
/// fit on Strix Halo's 128 GB unified memory while still resident across
/// forwards. Falls back to F32 dequant when the source isn't uniformly one
/// of the supported quant types (or is a quant type without a resident
/// kernel wired up) — the F32 indexed kernel is the historical
/// correctness-first path and remains the streaming default.
/// </para>
/// </remarks>
internal static class VulkanQwen3MoeMoeUpload
{
    /// <summary>
    /// Quant types with a resident (raw-block, on-device-dequant) indexed
    /// matmul kernel wired into <see cref="VulkanQwen3MoeHybridTransformerModel.RecordIndexedMoeMatmul"/>.
    /// Adding a new entry here requires: (1) a kernel following the
    /// <c>MoeIndexedMatmulQ6_KF32Kernel</c> pattern, (2) wiring it into
    /// <see cref="VulkanQwen3MoeHybridKernels"/>, and (3) a dispatch arm in
    /// <c>RecordIndexedMoeMatmul</c>.
    /// </summary>
    private static readonly HashSet<QuantizationType> s_ResidentQuantTypes = new()
    {
        QuantizationType.Q6_K,
        QuantizationType.Q4_K,
    };

    /// <summary>
    /// One layer's worth of device-resident MoE weights: router gate, three
    /// routed-expert banks (F32, Q6_K-resident, or Q4_K-resident), optional
    /// shared-expert projections, optional Qwen1.5-MoE sigmoid gate.
    /// </summary>
    public sealed class LayerBundle : IDisposable
    {
        public VulkanDevice.Buffer Gate { get; }
        /// <summary>
        /// Routed-expert W1 (gate) bank. Layout depends on <see cref="BankQuantType"/>:
        /// <list type="bullet">
        ///   <item><c>F32</c>: <c>[numExperts, intermediate, hidden]</c> contiguous floats.</item>
        ///   <item><c>Q6_K</c>: <c>[numExperts, intermediate, (hidden/256)*210]</c> raw block bytes.</item>
        ///   <item><c>Q4_K</c>: <c>[numExperts, intermediate, (hidden/256)*144]</c> raw block bytes.</item>
        /// </list>
        /// </summary>
        public VulkanDevice.Buffer W1Bank { get; }
        /// <summary>Routed-expert W2 (down) bank. See <see cref="W1Bank"/>; the M/K dims swap (M=hidden, K=intermediate).</summary>
        public VulkanDevice.Buffer W2Bank { get; }
        /// <summary>Routed-expert W3 (up) bank. See <see cref="W1Bank"/>.</summary>
        public VulkanDevice.Buffer W3Bank { get; }

        /// <summary>
        /// Storage type of the three routed banks (<see cref="W1Bank"/>,
        /// <see cref="W2Bank"/>, <see cref="W3Bank"/>) — one of
        /// <see cref="QuantizationType.F32"/> (default; bank holds dequantised
        /// floats) or <see cref="QuantizationType.Q6_K"/> (resident raw Q6_K
        /// blocks). The Vulkan model's <c>RecordMoeLayer</c> dispatches the
        /// matching indexed kernel based on this field.
        /// </summary>
        public QuantizationType BankQuantType { get; }

        public int NumExperts { get; }
        public int NumExpertsPerTok { get; }
        public int IntermediateSize { get; }
        public bool NormTopKProb { get; }

        public bool HasSharedExpert { get; }
        public int NumSharedExperts { get; }
        public int SharedIntermediateSize { get; }
        // Shared expert convention: Qwen3MoeHybrid carries one shared expert per
        // layer, so we fuse the per-shared-expert weights into single banks here.
        // (DeepSeek-V2/V3 can have N>1 shared experts; the qwen35moe production
        // path is N==1 with the sigmoid-gated convention.)
        public VulkanDevice.Buffer? SharedGate { get; }
        public VulkanDevice.Buffer? SharedUp { get; }
        public VulkanDevice.Buffer? SharedDown { get; }
        public VulkanDevice.Buffer? SharedExpertGate { get; }   // Qwen1.5-MoE per-token sigmoid gate

        public LayerBundle(
            VulkanDevice.Buffer gate, VulkanDevice.Buffer w1Bank, VulkanDevice.Buffer w2Bank, VulkanDevice.Buffer w3Bank,
            QuantizationType bankQuantType,
            int numExperts, int numExpertsPerTok, int intermediate, bool normTopKProb,
            bool hasShared, int numSharedExperts, int sharedIntermediate,
            VulkanDevice.Buffer? sharedGate, VulkanDevice.Buffer? sharedUp, VulkanDevice.Buffer? sharedDown,
            VulkanDevice.Buffer? sharedExpertGate)
        {
            Gate = gate; W1Bank = w1Bank; W2Bank = w2Bank; W3Bank = w3Bank;
            BankQuantType = bankQuantType;
            NumExperts = numExperts; NumExpertsPerTok = numExpertsPerTok;
            IntermediateSize = intermediate; NormTopKProb = normTopKProb;
            HasSharedExpert = hasShared; NumSharedExperts = numSharedExperts;
            SharedIntermediateSize = sharedIntermediate;
            SharedGate = sharedGate; SharedUp = sharedUp; SharedDown = sharedDown;
            SharedExpertGate = sharedExpertGate;
        }

        public void Dispose()
        {
            Gate.Dispose();
            W1Bank.Dispose(); W2Bank.Dispose(); W3Bank.Dispose();
            SharedGate?.Dispose(); SharedUp?.Dispose(); SharedDown?.Dispose();
            SharedExpertGate?.Dispose();
        }
    }

    /// <summary>
    /// Uploads one MoE layer's routed banks plus shared-expert weights.
    /// </summary>
    /// <param name="device">Vulkan device.</param>
    /// <param name="moe">CPU-side weight bundle; raw quant view is consulted when present.</param>
    /// <param name="hiddenSize">Hidden dim, needed to size W1/W3 banks.</param>
    /// <param name="residentQuant">
    /// When <c>true</c>, opt into resident-quant upload of the routed banks
    /// when the source is uniformly one of <see cref="s_ResidentQuantTypes"/>
    /// (currently Q6_K and Q4_K — see
    /// <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ6_KF32Kernel"/> and
    /// <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ4_KF32Kernel"/>).
    /// Source bytes are copied without dequantisation, so device memory is
    /// ~25 GB instead of ~120 GB at qwen35moe-A3B-Q6_K_XL scale (Q4_K is
    /// smaller still). When the source isn't uniformly a supported quant
    /// type (or this flag is <c>false</c>), the routed banks dequant to F32
    /// — the historical correctness-first path. Always <c>true</c> in
    /// resident-MoE mode (the bundle survives across forwards), <c>false</c>
    /// in streaming mode (per-forward F32 upload is the existing default).
    /// </param>
    public static unsafe LayerBundle UploadLayer(
        VulkanDevice device, MoeLayerWeights moe, int hiddenSize,
        bool residentQuant = false)
    {
        int numE = moe.NumExperts;
        int interm = moe.IntermediateSize;
        long w1Elems = (long)interm * hiddenSize;
        long w2Elems = (long)hiddenSize * interm;

        // Decide bank storage type once per layer. We require ALL three
        // routed projections to share one quant type — Qwen3.6-A3B GGUFs
        // produced by llama.cpp's quantizer always do. A mixed quant layer
        // (e.g. Q6_K W1/W3 + Q5_K W2 from a hand-edited UD-style GGUF) falls
        // back to F32 since the dispatcher only carries one bank kernel per
        // bundle. Only quant types with a resident indexed-matmul kernel
        // wired up (see s_ResidentQuantTypes) are eligible.
        QuantizationType bankQt = QuantizationType.F32;
        if (residentQuant && moe.HasRawQuantView
            && moe.GateExpsRawQt == moe.UpExpsRawQt
            && moe.GateExpsRawQt == moe.DownExpsRawQt
            && s_ResidentQuantTypes.Contains(moe.GateExpsRawQt))
        {
            bankQt = moe.GateExpsRawQt;
        }

        // Bounded persistently-mapped staging (issue #147): banks stream through
        // it one expert slab (or chunk) at a time — the previous whole-bank
        // staging write (≈8 GB at qwen35moe-A3B scale at fp32) is gone.
        long maxSlabBytesF32 = Math.Max(w1Elems, w2Elems) * sizeof(float);
        long gateBytes = (long)numE * hiddenSize * sizeof(float);
        long stageBytes = Math.Max(maxSlabBytesF32, gateBytes);
        using var staging = VulkanStagingBuffer.Create(device, stageBytes);

        // ── Router gate ──────────────────────────────────────────────────────
        var gate = device.AllocateDeviceLocal(gateBytes);
        staging.UploadFloats(moe.Gate, gate);

        // ── Routed expert banks. Two cases:
        //   (1) F32 (default): dequant each per-expert slice into staging at
        //       the expert's contiguous slot, then copy the whole bank to
        //       device memory. 3 device buffers per layer, each ~F32 sized.
        //   (2) Q6_K resident: copy each per-expert raw block slab into
        //       staging at the expert's contiguous slot, then copy. 3 device
        //       buffers per layer, each at ~25% of the F32 size.
        VulkanDevice.Buffer w1Bank, w2Bank, w3Bank;
        if (bankQt != QuantizationType.F32)
        {
            long w1RowBytes = Dequantize.RowByteSize(hiddenSize, bankQt);
            long w2RowBytes = Dequantize.RowByteSize(interm, bankQt);
            long w1PerExpertBytes = w1RowBytes * interm;
            long w2PerExpertBytes = w2RowBytes * hiddenSize;
            long w1BankBytes = (long)numE * w1PerExpertBytes;
            long w2BankBytes = (long)numE * w2PerExpertBytes;

            w1Bank = device.AllocateDeviceLocal(w1BankBytes);
            w2Bank = device.AllocateDeviceLocal(w2BankBytes);
            w3Bank = device.AllocateDeviceLocal(w1BankBytes);

            UploadRoutedBankResidentQuant(device, staging, moe, bankQt, kind: 'G', bank: w1Bank,
                numE: numE, perExpertBytes: w1PerExpertBytes);
            UploadRoutedBankResidentQuant(device, staging, moe, bankQt, kind: 'D', bank: w2Bank,
                numE: numE, perExpertBytes: w2PerExpertBytes);
            UploadRoutedBankResidentQuant(device, staging, moe, bankQt, kind: 'U', bank: w3Bank,
                numE: numE, perExpertBytes: w1PerExpertBytes);
        }
        else
        {
            long w1BankBytes = (long)numE * w1Elems * sizeof(float);
            long w2BankBytes = (long)numE * w2Elems * sizeof(float);
            long w3BankBytes = w1BankBytes;
            w1Bank = device.AllocateDeviceLocal(w1BankBytes);
            w2Bank = device.AllocateDeviceLocal(w2BankBytes);
            w3Bank = device.AllocateDeviceLocal(w3BankBytes);

            UploadRoutedBank(device, staging, moe, kind: 'G', bank: w1Bank, numE: numE, perExpertElems: w1Elems);
            UploadRoutedBank(device, staging, moe, kind: 'D', bank: w2Bank, numE: numE, perExpertElems: w2Elems);
            UploadRoutedBank(device, staging, moe, kind: 'U', bank: w3Bank, numE: numE, perExpertElems: w1Elems);
        }

        // ── Shared expert (optional) ─────────────────────────────────────────
        VulkanDevice.Buffer? sharedGate = null, sharedUp = null, sharedDown = null;
        VulkanDevice.Buffer? sharedExpertGate = null;
        int numShared = moe.NumSharedExperts;
        int sharedI = moe.SharedIntermediateSize;
        bool hasShared = moe.HasSharedExpert;
        if (hasShared && numShared > 0)
        {
            // qwen35moe convention is one shared expert per layer — upload its
            // gate/up/down directly. Larger numShared (DeepSeek family) would
            // need an outer-loop sum; deliberately not implemented here.
            long sharedGateUpElems = (long)sharedI * hiddenSize;
            long sharedDownElems = (long)hiddenSize * sharedI;
            sharedGate = UploadF32FromPointer(device, staging, moe.SharedGateProj[0], sharedGateUpElems);
            sharedUp = UploadF32FromPointer(device, staging, moe.SharedUpProj[0], sharedGateUpElems);
            sharedDown = UploadF32FromPointer(device, staging, moe.SharedDownProj[0], sharedDownElems);
        }
        if (moe.SharedExpertGate is not null)
            sharedExpertGate = UploadFloatArray(device, staging, moe.SharedExpertGate);

        // Post-attn norm weight is owned by VulkanQwen3MoeHybridWeights (uploaded
        // once at load time). The shared-expert branch needs it to re-derive
        // the RMSNormed hidden state, so the model passes it directly into
        // RecordSharedExpert from the per-layer LayerBuffers — keeping this
        // bundle self-contained and free of borrowed references.
        return new LayerBundle(
            gate, w1Bank, w2Bank, w3Bank,
            bankQuantType: bankQt,
            numExperts: numE, numExpertsPerTok: moe.NumExpertsPerTok,
            intermediate: interm, normTopKProb: moe.NormTopKProb,
            hasShared: hasShared, numSharedExperts: numShared, sharedIntermediate: sharedI,
            sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown,
            sharedExpertGate: sharedExpertGate);
    }

    /// <summary>
    /// Walks the experts of one routed projection (W1=Gate, W2=Down, W3=Up),
    /// dequantising each per-expert slice into the staging buffer at the
    /// expert's contiguous slot, then issues a single device copy. Mirrors
    /// the CPU <c>DequantRoutedExpertsIntoScratch</c> pattern.
    /// </summary>
    private static void UploadRoutedBank(
        VulkanDevice device, VulkanStagingBuffer staging, MoeLayerWeights moe,
        char kind, VulkanDevice.Buffer bank, int numE, long perExpertElems)
    {
        long perExpertBytes = perExpertElems * sizeof(float);

        {
            for (int e = 0; e < numE; e++)
            {
                long slotOffset = (long)e * perExpertBytes;

                if (moe.HasRawQuantView)
                {
                    nint srcPtr;
                    QuantizationType qt;
                    int mDim, kDim;
                    switch (kind)
                    {
                        case 'G':
                            srcPtr = moe.GateExpsRaw; qt = moe.GateExpsRawQt;
                            mDim = moe.GateExpsMDim; kDim = moe.GateExpsKDim;
                            break;
                        case 'D':
                            srcPtr = moe.DownExpsRaw; qt = moe.DownExpsRawQt;
                            mDim = moe.DownExpsMDim; kDim = moe.DownExpsKDim;
                            break;
                        case 'U':
                            srcPtr = moe.UpExpsRaw; qt = moe.UpExpsRawQt;
                            mDim = moe.UpExpsMDim; kDim = moe.UpExpsKDim;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(kind));
                    }

                    // Per-expert row byte size — matches GGUF fused-expert layout
                    // [K, M, num_experts] with K innermost: each expert is M rows of
                    // K elements.
                    long rowBytes = Dequantize.RowByteSize(kDim, qt);
                    long perExpertSrcBytes = rowBytes * mDim;
                    nint expertSrc = srcPtr + (nint)(e * perExpertSrcBytes);
                    VulkanWeights.DequantAndUploadSlot(device, staging, expertSrc,
                        perExpertElems, qt, scale: 1.0f, bank, slotOffset);
                }
                else
                {
                    // Already-F32 fallback — used by the synthetic-fixture tests
                    // that populate MoeLayerWeights directly without a raw view.
                    // W1/W2/W3 are nint[] pointers to F32 row-major matrices.
                    nint src = kind switch
                    {
                        'G' => moe.W1[e],
                        'D' => moe.W2[e],
                        'U' => moe.W3[e],
                        _ => 0,
                    };
                    staging.UploadBytes(src, perExpertBytes, bank, slotOffset);
                }
            }
        }
    }

    /// <summary>
    /// Resident-quant variant of <see cref="UploadRoutedBank"/>. Walks the
    /// experts of one routed projection and copies the raw quant block bytes
    /// (Q6_K, Q4_K, ...) into the staging buffer at the expert's contiguous
    /// slot, then issues a single device copy. The destination layout is
    /// <c>[numExperts, M_kind, RowByteSize(K_kind, quantType)]</c> — what the
    /// matching indexed matmul shader (e.g. <c>moe_indexed_matmul_q6_k_f32.comp</c>,
    /// <c>moe_indexed_matmul_q4_k_f32.comp</c>) expects: each per-expert slab
    /// is <c>M</c> rows of <c>RowByteSize(K, quantType)</c> bytes, expert
    /// slabs contiguous. Source byte layout in <c>moe.GateExpsRaw</c> et al.
    /// matches because GGUF's fused-expert tensors store experts contiguously
    /// in the same per-row stride.
    /// </summary>
    private static void UploadRoutedBankResidentQuant(
        VulkanDevice device, VulkanStagingBuffer staging, MoeLayerWeights moe,
        QuantizationType quantType, char kind, VulkanDevice.Buffer bank, int numE, long perExpertBytes)
    {
        long totalBytes = (long)numE * perExpertBytes;
        if (!moe.HasRawQuantView)
            throw new InvalidOperationException(
                $"{quantType}-resident MoE upload requires moe.HasRawQuantView — synthetic fixtures must populate the raw view.");

        nint srcPtr;
        switch (kind)
        {
            case 'G':
                if (moe.GateExpsRawQt != quantType)
                    throw new InvalidOperationException(
                        $"{quantType}-resident upload (kind=G) requires {quantType} source, got {moe.GateExpsRawQt}.");
                srcPtr = moe.GateExpsRaw;
                break;
            case 'D':
                if (moe.DownExpsRawQt != quantType)
                    throw new InvalidOperationException(
                        $"{quantType}-resident upload (kind=D) requires {quantType} source, got {moe.DownExpsRawQt}.");
                srcPtr = moe.DownExpsRaw;
                break;
            case 'U':
                if (moe.UpExpsRawQt != quantType)
                    throw new InvalidOperationException(
                        $"{quantType}-resident upload (kind=U) requires {quantType} source, got {moe.UpExpsRawQt}.");
                srcPtr = moe.UpExpsRaw;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }

        // GGUF fused-expert layout [E, M, K] (K innermost) matches the bank
        // layout byte-for-byte, so the whole bank is one contiguous streamed
        // copy through the bounded staging buffer.
        staging.UploadBytes(srcPtr, totalBytes, bank);
    }

    /// <summary>
    /// Uploads an F32 matrix from an unmanaged pointer (the shared-expert
    /// projections live as <c>nint</c> in <see cref="MoeLayerWeights"/>).
    /// </summary>
    private static VulkanDevice.Buffer UploadF32FromPointer(
        VulkanDevice device, VulkanStagingBuffer staging, nint src, long elems)
    {
        long bytes = elems * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);
        staging.UploadBytes(src, bytes, buf);
        return buf;
    }

    private static VulkanDevice.Buffer UploadFloatArray(
        VulkanDevice device, VulkanStagingBuffer staging, float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);
        staging.UploadFloats(data, buf);
        return buf;
    }
}
