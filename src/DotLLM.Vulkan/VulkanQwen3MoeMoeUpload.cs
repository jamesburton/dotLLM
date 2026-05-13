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
/// The MoE indexed-matmul kernel (<see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulF32Kernel"/>)
/// is F32-only today — there is no Q8_0 / K-quant variant in tree yet — so the
/// per-layer banks always upload as F32 regardless of the source quant. This
/// is what bounds the resident-cache policy: opting in requires the F32-
/// dequantised banks to fit in unified memory. A Q6_K MoE matmul shader is a
/// known follow-up to lift that bound for Qwen3.6-A3B-Q6_K_XL.
/// </para>
/// </remarks>
internal static class VulkanQwen3MoeMoeUpload
{
    /// <summary>
    /// One layer's worth of device-resident MoE weights: router gate, three
    /// routed-expert F32 banks, optional shared-expert projections, optional
    /// Qwen1.5-MoE sigmoid gate, plus the post-attn norm weight needed by the
    /// shared-expert branch (re-derives the RMSNormed hidden state).
    /// </summary>
    public sealed class LayerBundle : IDisposable
    {
        public VulkanDevice.Buffer Gate { get; }
        public VulkanDevice.Buffer W1Bank { get; }    // [numExperts, intermediate, hidden] F32
        public VulkanDevice.Buffer W2Bank { get; }    // [numExperts, hidden, intermediate] F32
        public VulkanDevice.Buffer W3Bank { get; }    // [numExperts, intermediate, hidden] F32

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
            int numExperts, int numExpertsPerTok, int intermediate, bool normTopKProb,
            bool hasShared, int numSharedExperts, int sharedIntermediate,
            VulkanDevice.Buffer? sharedGate, VulkanDevice.Buffer? sharedUp, VulkanDevice.Buffer? sharedDown,
            VulkanDevice.Buffer? sharedExpertGate)
        {
            Gate = gate; W1Bank = w1Bank; W2Bank = w2Bank; W3Bank = w3Bank;
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
    public static unsafe LayerBundle UploadLayer(
        VulkanDevice device, MoeLayerWeights moe, int hiddenSize)
    {
        int numE = moe.NumExperts;
        int interm = moe.IntermediateSize;
        long w1Elems = (long)interm * hiddenSize;
        long w2Elems = (long)hiddenSize * interm;

        // Sized to the largest per-expert F32 matrix; reused across all bank
        // uploads. Staging is host-visible so we can write the dequant output
        // straight into it.
        long maxPerExpertBytes = Math.Max(w1Elems, w2Elems) * sizeof(float);
        long gateBytes = (long)numE * hiddenSize * sizeof(float);
        long stageBytes = Math.Max(maxPerExpertBytes, gateBytes);
        using var staging = device.Allocate(stageBytes);

        // ── Router gate ──────────────────────────────────────────────────────
        var gate = device.AllocateDeviceLocal(gateBytes);
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)gateBytes, 0, out nint mappedGate)
            .ThrowOnError("vkMapMemory VulkanQwen3MoeMoeUpload router gate");
        try
        {
            moe.Gate.AsSpan().CopyTo(new Span<float>((void*)mappedGate, moe.Gate.Length));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, gate, (ulong)gateBytes);

        // ── Routed expert banks. For each of W1/W2/W3 we walk the experts and
        //    dequantise (or copy F32) each slice into the staging buffer at the
        //    expert's slot, then copy the whole bank to a device-local buffer.
        //    Allocates 3 device buffers per layer per forward — see class
        //    remarks for the optimisation roadmap.
        long w1BankBytes = (long)numE * w1Elems * sizeof(float);
        long w2BankBytes = (long)numE * w2Elems * sizeof(float);
        long w3BankBytes = w1BankBytes;
        var w1Bank = device.AllocateDeviceLocal(w1BankBytes);
        var w2Bank = device.AllocateDeviceLocal(w2BankBytes);
        var w3Bank = device.AllocateDeviceLocal(w3BankBytes);

        UploadRoutedBank(device, staging, moe, kind: 'G', bank: w1Bank, numE: numE, perExpertElems: w1Elems);
        UploadRoutedBank(device, staging, moe, kind: 'D', bank: w2Bank, numE: numE, perExpertElems: w2Elems);
        UploadRoutedBank(device, staging, moe, kind: 'U', bank: w3Bank, numE: numE, perExpertElems: w1Elems);

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
    private static unsafe void UploadRoutedBank(
        VulkanDevice device, VulkanDevice.Buffer staging, MoeLayerWeights moe,
        char kind, VulkanDevice.Buffer bank, int numE, long perExpertElems)
    {
        long perExpertBytes = perExpertElems * sizeof(float);
        long totalBytes = (long)numE * perExpertBytes;

        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)totalBytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanQwen3MoeMoeUpload routed bank");
        try
        {
            float* dst = (float*)mapped;
            for (int e = 0; e < numE; e++)
            {
                Span<float> slot = new(dst + (long)e * perExpertElems, checked((int)perExpertElems));

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
                    Dequantize.ToFloat32(expertSrc, (int)perExpertElems, qt, slot);
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
                    new ReadOnlySpan<float>((void*)src, checked((int)perExpertElems))
                        .CopyTo(slot);
                }
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, bank, (ulong)totalBytes);
    }

    /// <summary>
    /// Uploads an F32 matrix from an unmanaged pointer (the shared-expert
    /// projections live as <c>nint</c> in <see cref="MoeLayerWeights"/>).
    /// </summary>
    private static unsafe VulkanDevice.Buffer UploadF32FromPointer(
        VulkanDevice device, VulkanDevice.Buffer staging, nint src, long elems)
    {
        long bytes = elems * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanQwen3MoeMoeUpload F32-from-pointer");
        try
        {
            new ReadOnlySpan<float>((void*)src, checked((int)elems))
                .CopyTo(new Span<float>((void*)mapped, checked((int)elems)));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        return buf;
    }

    private static unsafe VulkanDevice.Buffer UploadFloatArray(
        VulkanDevice device, VulkanDevice.Buffer staging, float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);
        VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanQwen3MoeMoeUpload float-array");
        try
        {
            data.AsSpan().CopyTo(new Span<float>((void*)mapped, data.Length));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }
        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        return buf;
    }
}
