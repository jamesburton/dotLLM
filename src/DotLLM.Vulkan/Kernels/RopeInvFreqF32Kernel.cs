using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// RoPE kernel driven by an explicit per-pair inverse-frequency device buffer (issue #743), with an
/// optional Q-only Mistral-3 attention-temperature multiplier. Used for dense models carrying
/// llama.cpp <c>rope_freqs.weight</c> (Llama-3.1/3.2/3.3) and/or <c>attention.temperature_scale</c>
/// (Ministral-3); every other model stays on <see cref="RopeF32Kernel"/>. Mirrors CPU
/// <c>RoPE.PrecomputeFrequencyTableWithFactors</c> + <c>TransformerModel.ApplyAttnTemperature</c>.
/// </summary>
public sealed class RopeInvFreqF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 9 * sizeof(uint) + 2 * sizeof(float); // seqLen, numHeads, numKvHeads, headDim, ropeDim, ropeType, theta, freqDim, neoxPairOffset

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private RopeInvFreqF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>rope_invfreq_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static RopeInvFreqF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rope_invfreq_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new RopeInvFreqF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Applies RoPE to Q and K in place. Synchronous — returns after
    /// <c>vkQueueWaitIdle</c>. Legacy wrapper around <see cref="Record"/>.
    /// </summary>
    /// <param name="q">Query buffer (FP32), layout <c>[seqLen, numHeads * headDim]</c>.</param>
    /// <param name="k">Key buffer (FP32), layout <c>[seqLen, numKvHeads * headDim]</c>.</param>
    /// <param name="positions">Position indices buffer (int32), length <paramref name="seqLen"/>.</param>
    /// <param name="invFreq">Per-pair inverse frequencies (FP32), length <c>ropeDim/2</c> (factors already folded in).</param>
    /// <param name="tempScale">Attention-temperature scale (Q only); 0 disables.</param>
    /// <param name="tempFloor">Attention-temperature position bucket width; 0 disables.</param>
    /// <param name="seqLen">Number of query/key positions.</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="ropeDim">Number of dims to rotate per head (even, &lt;= headDim).</param>
    /// <param name="theta">RoPE base (typical 10000 for Llama-2, 500000 for Llama-3).</param>
    /// <param name="variant">Pair-layout variant.</param>
    /// <param name="freqDim">
    /// Frequency-denominator dim for the exponent <c>2*pair/freqDim</c>. Pass 0 (default)
    /// to use <paramref name="ropeDim"/> — correct for full rotation AND for standard
    /// partial-rotary NeoX (Qwen3 / NemotronH / Llama). For Gemma-4 partial global layers
    /// pass the FULL head dim so the exponent matches the CPU oracle's partial freq table
    /// (<c>RoPE.PrecomputeFrequencyTablePartial</c>, denom = fullHeadDim).
    /// </param>
    /// <param name="neoxPairOffset">
    /// NeoX rotate-half pairing offset (<c>i1 = i0 + offset</c>). Pass <c>null</c> (default)
    /// for the standard partial-rotary convention <c>ropeDim/2</c> (Qwen3 / NemotronH /
    /// Llama-family — matches CPU <c>RoPE.Execute</c> → <c>ApplyRotationNeoX</c>); pass
    /// <c>headDim/2</c> for Gemma-4 partial global layers (matches CPU
    /// <c>RoPE.ApplyRotationNeoXPartial</c>). Ignored for <see cref="RopeF32Kernel.Variant.Norm"/>.
    /// </param>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer positions, VulkanDevice.Buffer invFreq,
        int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, float theta,
        RopeF32Kernel.Variant variant = RopeF32Kernel.Variant.Norm, int freqDim = 0, int? neoxPairOffset = null,
        float tempScale = 0f, int tempFloor = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, positions, invFreq, seqLen, numHeads, numKvHeads, headDim, ropeDim, theta, variant, freqDim, neoxPairOffset, tempScale, tempFloor);
        ctx.SubmitAndWait();
    }

    /// <summary>Records RoPE into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer positions, VulkanDevice.Buffer invFreq,
        int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, float theta,
        RopeF32Kernel.Variant variant = RopeF32Kernel.Variant.Norm, int freqDim = 0, int? neoxPairOffset = null,
        float tempScale = 0f, int tempFloor = 0)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (ropeDim <= 0 || (ropeDim & 1) != 0) throw new ArgumentException($"ropeDim must be a positive even integer, got {ropeDim}", nameof(ropeDim));
        if (ropeDim > headDim) throw new ArgumentException($"ropeDim ({ropeDim}) must be <= headDim ({headDim})", nameof(ropeDim));
        // Default the frequency denominator to ropeDim (full-rotation + standard partial).
        // Gemma-4 partial global passes headDim so the exponent matches the CPU oracle.
        if (freqDim <= 0) freqDim = ropeDim;
        // Default NeoX pairing offset is ropeDim/2 (standard partial-rotary; equals
        // headDim/2 for full rope). Gemma-4 global partial rope overrides with headDim/2.
        int pairOffset = neoxPairOffset ?? (ropeDim / 2);
        if (variant == RopeF32Kernel.Variant.NeoX)
        {
            if (pairOffset <= 0) throw new ArgumentException($"neoxPairOffset must be positive, got {pairOffset}", nameof(neoxPairOffset));
            if (pairOffset + ropeDim / 2 > headDim)
                throw new ArgumentException(
                    $"neoxPairOffset ({pairOffset}) + ropeDim/2 ({ropeDim / 2}) must be <= headDim ({headDim}) "
                    + "or the high pair index runs past the head.", nameof(neoxPairOffset));
        }

        long qBytes = (long)seqLen * numHeads * headDim * sizeof(float);
        long kBytes = (long)seqLen * numKvHeads * headDim * sizeof(float);
        long posBytes = (long)seqLen * sizeof(int);
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (k.Size < kBytes) throw new ArgumentException("K buffer too small.", nameof(k));
        if (positions.Size < posBytes) throw new ArgumentException("Positions buffer too small.", nameof(positions));
        if (invFreq.Size < (long)(ropeDim / 2) * sizeof(float)) throw new ArgumentException("Inverse-frequency buffer too small.", nameof(invFreq));
        if ((tempScale != 0f) != (tempFloor > 0)) throw new ArgumentException("tempScale and tempFloor must be set together.");

        int halfRope = ropeDim / 2;
        long totalQ = (long)seqLen * numHeads * halfRope;
        long totalK = (long)seqLen * numKvHeads * halfRope;
        long maxPairs = Math.Max(totalQ, totalK);

        Span<nint> buffers = stackalloc nint[4] { q.Handle, k.Handle, positions.Handle, invFreq.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        // Push constants: 9 uint-sized + 2 float = 44 bytes.
        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[0..],  (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],  (uint)numHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],  (uint)numKvHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)headDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)ropeDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[20..], (uint)variant);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[24..], theta);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[28..], (uint)freqDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[32..], (uint)pairOffset);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[36..], tempScale);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[40..], (uint)tempFloor);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groups = (uint)((maxPairs + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
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
