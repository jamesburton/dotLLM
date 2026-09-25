using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused RoPE + KV-cache-write kernel with FP32 Q/K/V data — Vulkan port of
/// the CUDA <c>fused_rope_kv_write_f16</c> kernel
/// (<c>native/kernels/fused_rope_kv_write.cu</c>).
/// </summary>
/// <remarks>
/// <para>
/// Collapses the current two-dispatch sequence — a <see cref="RopeF32Kernel"/>
/// COMPUTE dispatch, a COMPUTE→COMPUTE barrier, then
/// <see cref="VulkanKvCache.RecordUpdate"/>'s two <c>vkCmdCopyBuffer</c>
/// TRANSFER copies (K and V), then a TRANSFER→COMPUTE barrier — into a single
/// COMPUTE dispatch. Q is rotated in place in the Q scratch buffer (unchanged
/// from <see cref="RopeF32Kernel"/> — attention always reads Q from scratch).
/// K is rotated and the ROTATED result is written directly into the K-cache
/// row for its token; V is copied (no rotation) directly into the V-cache
/// row. Math for the rotation itself is bit-identical to
/// <c>rope_f32.comp</c> — only the K/V destination changed from a scratch
/// buffer to the cache buffer at the correct row offset.
/// </para>
/// <para>
/// <b>Contiguous positions only.</b> The shader computes each source token
/// <c>t</c>'s cache row as <c>(startPos + t) * kvStride</c> — a single base
/// push-constant plus a per-thread stride multiply, mirroring
/// <see cref="VulkanKvCache.RecordUpdate"/>'s contiguous fast path (one
/// <c>vkCmdCopyBuffer</c> region covering the whole <c>seqLen</c>).
/// Non-contiguous / gapped positions (that method's per-row-copy fallback)
/// are NOT supported — callers must detect that case and fall back to the
/// unfused <see cref="RopeF32Kernel"/> + <see cref="VulkanKvCache.RecordUpdate"/>
/// path, exactly as <c>VulkanTransformerModel</c> does.
/// </para>
/// <para>
/// Gate this kernel to a plain <see cref="VulkanKvCache"/> destination only —
/// TurboQuant-quantized KV caches and MLA's split KV layout are out of scope
/// (matching the CUDA precedent's <c>kvCache is CudaKvCache</c> gate,
/// <c>CudaTransformerModel.cs:977</c>).
/// </para>
/// </remarks>
public sealed class RopeKvWriteF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    // seqLen, numHeads, numKvHeads, headDim, ropeDim, ropeType, freqDim, neoxPairOffset,
    // startPos, kvStride (10 uint) + theta (1 float) = 44 bytes.
    private const int PushConstantBytes = 10 * sizeof(uint) + sizeof(float);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private RopeKvWriteF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 6);
    }

    /// <summary>Loads <c>rope_kv_write_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static RopeKvWriteF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rope_kv_write_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[6];
            for (int i = 0; i < 6; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 6);
        return new RopeKvWriteF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Returns true if <c>rope_kv_write_f32.spv</c> exists under <paramref name="spvDir"/>.</summary>
    public static bool IsAvailable(string spvDir) => File.Exists(Path.Combine(spvDir, "rope_kv_write_f32.spv"));

    /// <summary>
    /// Creates the kernel if <c>rope_kv_write_f32.spv</c> is present, otherwise returns
    /// <c>null</c> — lets older builds (or SPV directories predating this kernel) fall
    /// back gracefully to the unfused <see cref="RopeF32Kernel"/> + <see cref="VulkanKvCache.RecordUpdate"/>
    /// path, mirroring <c>RmsNormMatmulQ8_0FusedKernel.TryCreate</c>'s pattern for optional fusions.
    /// </summary>
    public static RopeKvWriteF32Kernel? TryCreate(VulkanDevice device, string spvDir)
        => IsAvailable(spvDir) ? Create(device, spvDir) : null;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the fused RoPE + KV-cache-write dispatch into <paramref name="cmdBuf"/>
    /// without submitting. Caller is responsible for barriers before (ensuring
    /// q/k/v scratch writes from the preceding projection are visible) and after
    /// (ensuring this dispatch's cache writes are visible to attention) —
    /// exactly the same barrier shape as the RoPE dispatch it replaces plus the
    /// KV-cache-write's TRANSFER→COMPUTE barrier collapsed into one
    /// COMPUTE→COMPUTE barrier, since this is now a single compute dispatch.
    /// </summary>
    /// <param name="cmdBuf">Vulkan command buffer to record into.</param>
    /// <param name="q">Query scratch buffer (FP32), layout <c>[seqLen, numHeads * headDim]</c> — rotated in place.</param>
    /// <param name="k">Key scratch buffer (FP32), layout <c>[seqLen, numKvHeads * headDim]</c> — read-only source.</param>
    /// <param name="v">Value scratch buffer (FP32), layout <c>[seqLen, numKvHeads * headDim]</c> — read-only source.</param>
    /// <param name="positions">Position indices buffer (int32), length <paramref name="seqLen"/> — RoPE angle source.</param>
    /// <param name="kCache">Destination K-cache buffer for this layer, layout <c>[maxSeqLen, kvStride]</c>.</param>
    /// <param name="vCache">Destination V-cache buffer for this layer, layout <c>[maxSeqLen, kvStride]</c>.</param>
    /// <param name="startPos">Cache row index of source token 0 (contiguous positions only — caller must verify).</param>
    /// <param name="kvStride">Elements per cache row (<c>numKvHeads * headDim</c> for this layer).</param>
    /// <param name="seqLen">Number of query/key/value positions in this dispatch.</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="ropeDim">Number of dims to rotate per head (even, &lt;= headDim).</param>
    /// <param name="theta">RoPE base.</param>
    /// <param name="variant">Pair-layout variant — see <see cref="RopeF32Kernel.Variant"/>.</param>
    /// <param name="freqDim">Frequency-denominator dim; 0 (default) uses <paramref name="ropeDim"/> — see <see cref="RopeF32Kernel.Record"/>.</param>
    /// <param name="neoxPairOffset">NeoX rotate-half pairing offset; null (default) uses <c>ropeDim/2</c> — see <see cref="RopeF32Kernel.Record"/>.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer positions,
        VulkanDevice.Buffer kCache, VulkanDevice.Buffer vCache,
        int startPos, int kvStride,
        int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, float theta,
        RopeF32Kernel.Variant variant = RopeF32Kernel.Variant.Norm, int freqDim = 0, int? neoxPairOffset = null)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (ropeDim <= 0 || (ropeDim & 1) != 0) throw new ArgumentException($"ropeDim must be a positive even integer, got {ropeDim}", nameof(ropeDim));
        if (ropeDim > headDim) throw new ArgumentException($"ropeDim ({ropeDim}) must be <= headDim ({headDim})", nameof(ropeDim));
        if (startPos < 0) throw new ArgumentOutOfRangeException(nameof(startPos));
        if (kvStride < numKvHeads * headDim) throw new ArgumentException($"kvStride ({kvStride}) must be >= numKvHeads * headDim ({numKvHeads * headDim})", nameof(kvStride));

        if (freqDim <= 0) freqDim = ropeDim;
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
        if (v.Size < kBytes) throw new ArgumentException("V buffer too small.", nameof(v));
        if (positions.Size < posBytes) throw new ArgumentException("Positions buffer too small.", nameof(positions));
        long cacheRowsNeededBytes = (long)(startPos + seqLen) * kvStride * sizeof(float);
        if (kCache.Size < cacheRowsNeededBytes) throw new ArgumentException("K-cache buffer too small for startPos+seqLen rows.", nameof(kCache));
        if (vCache.Size < cacheRowsNeededBytes) throw new ArgumentException("V-cache buffer too small for startPos+seqLen rows.", nameof(vCache));

        int halfRope = ropeDim / 2;
        int tail = headDim - ropeDim;
        long r0 = (long)seqLen * numHeads * halfRope;
        long r1 = r0 + (long)seqLen * numKvHeads * halfRope;
        long r2 = r1 + (long)seqLen * numKvHeads * tail;
        long r3 = r2 + (long)seqLen * numKvHeads * headDim;

        Span<nint> buffers = stackalloc nint[6] { q.Handle, k.Handle, v.Handle, positions.Handle, kCache.Handle, vCache.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

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
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[36..], (uint)startPos);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[40..], (uint)kvStride);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        long totalThreads = r3;
        uint groups = (uint)((totalThreads + WorkgroupSize - 1) / WorkgroupSize);
        if (groups == 0) groups = 1; // headDim==ropeDim, numKvHeads>0 always yields >0 in practice; guard degenerate configs
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
