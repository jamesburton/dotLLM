using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Mamba-3 streaming-chunk boundary state adjustment — applied at the START of
/// a chunk that carries a non-empty <c>(k_state, v_state)</c> pair from the
/// previous chunk. Mirrors
/// <c>DotLLM.Models.Architectures.Mamba3Block.ApplyChunkBoundaryAdjustment</c>
/// (SISO) and the rank-summed boundary in
/// <c>DotLLM.Cpu.Kernels.Mamba3CanonicalSsd.ExecuteMimoStreaming</c> (MIMO):
/// <c>ssm_state[h, p, n] += v_state[h, p] · (Σ_r k_state[r, h, n]) · coef[h]</c>
/// where <c>coef[h] = dt[0, h] · (1 - trap[0, h])</c>.
/// </summary>
/// <remarks>
/// <para>
/// Both SISO and MIMO use the same kernel — SISO is the <c>nRank == 1</c>
/// special case where the rank loop runs exactly once per <c>(h, p, n)</c>.
/// The caller is responsible for skipping dispatch (or zeroing the
/// <c>coef</c> buffer) on the first chunk of a sequence where
/// <c>k_state</c> / <c>v_state</c> are still all-zero.
/// </para>
/// <para>
/// <b>Numerical equivalence.</b> The kernel sums kSum across the rank axis as
/// a sequential float accumulator — bit-equal to the CPU oracle's
/// <c>kSum += kState[...]</c> loop in
/// <c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>. The final FMA
/// <c>state += v · kSum · coef</c> matches the CPU's hot-path tighter to
/// <c>O(ulp)</c> per element.
/// </para>
/// </remarks>
public sealed class Mamba3ChunkBoundaryF32Kernel : IDisposable
{
    // nHead, headDim, dState, nRank — all u32.
    private const int PushConstantBytes = 4 * sizeof(uint);
    private const int BufferCount = 4;

    // Must mirror the layout-qualifier sizes in mamba3_chunk_boundary_f32.comp.
    private const int WgX = 16;
    private const int WgY = 16;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Mamba3ChunkBoundaryF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: BufferCount);
    }

    /// <summary>Loads <c>mamba3_chunk_boundary_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Mamba3ChunkBoundaryF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "mamba3_chunk_boundary_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BufferCount];
            for (int i = 0; i < BufferCount; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: BufferCount);
        return new Mamba3ChunkBoundaryF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer state,
        VulkanDevice.Buffer vState,
        VulkanDevice.Buffer kState,
        VulkanDevice.Buffer coef,
        int nHead, int headDim, int dState, int nRank)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, vState, kState, coef, nHead, headDim, dState, nRank);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the boundary-adjustment dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are FP32 row-major.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="state">SSM hidden state [<paramref name="nHead"/>, <paramref name="headDim"/>, <paramref name="dState"/>] — read-modify-write.</param>
    /// <param name="vState">Previous chunk's last-token V [<paramref name="nHead"/>, <paramref name="headDim"/>].</param>
    /// <param name="kState">
    /// Previous chunk's last-token post-RoPE K, layout
    /// [<paramref name="nRank"/>, <paramref name="nHead"/>, <paramref name="dState"/>]. For SISO
    /// pass <paramref name="nRank"/>=1 — the kernel collapses cleanly to the [H, N] shape.
    /// </param>
    /// <param name="coef">
    /// Per-head coefficient <c>dt[0, h] · (1 - trap[0, h])</c>, length
    /// <paramref name="nHead"/>. Caller computes host-side from the per-token
    /// preprocessing tables.
    /// </param>
    /// <param name="nHead">Head count H.</param>
    /// <param name="headDim">Channels per head P.</param>
    /// <param name="dState">State width N.</param>
    /// <param name="nRank">MIMO rank R (1 for SISO, ≥ 2 for MIMO).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state,
        VulkanDevice.Buffer vState,
        VulkanDevice.Buffer kState,
        VulkanDevice.Buffer coef,
        int nHead, int headDim, int dState, int nRank)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (nRank <= 0) throw new ArgumentOutOfRangeException(nameof(nRank));

        long stateBytes = (long)nHead * headDim * dState * sizeof(float);
        long vBytes     = (long)nHead * headDim          * sizeof(float);
        long kBytes     = (long)nRank * nHead * dState   * sizeof(float);
        long coefBytes  = (long)nHead                    * sizeof(float);
        if (state.Size  < stateBytes) throw new ArgumentException("state buffer too small.",  nameof(state));
        if (vState.Size < vBytes)     throw new ArgumentException("vState buffer too small.", nameof(vState));
        if (kState.Size < kBytes)     throw new ArgumentException("kState buffer too small.", nameof(kState));
        if (coef.Size   < coefBytes)  throw new ArgumentException("coef buffer too small.",   nameof(coef));

        Span<nint> buffers = stackalloc nint[BufferCount]
        {
            state.Handle, vState.Handle, kState.Handle, coef.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes,       (uint)nHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],  (uint)headDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],  (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)nRank);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // 2D grid (n, p) per workgroup; nHead workgroups in z. Ceil-div to cover
        // any (dState, headDim) that aren't multiples of (WgX, WgY).
        uint groupsX = (uint)((dState  + WgX - 1) / WgX);
        uint groupsY = (uint)((headDim + WgY - 1) / WgY);
        uint groupsZ = (uint)nHead;
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, groupsZ);
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
