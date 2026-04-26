using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Mamba-3 canonical SSD scan (SISO variant) — recurrent attention-style SSM kernel.
/// Mirrors <c>DotLLM.Cpu.Kernels.Mamba3CanonicalSsd.ExecuteSiso</c>: per-token
/// sequential time, per-head + per-(p, n) parallel space.
/// </summary>
/// <remarks>
/// <para>
/// Dispatch model: one workgroup per head (heads are independent), 256
/// threads per workgroup striding over <c>headDim</c> in <c>p</c>. Each
/// thread owns the <c>dState</c>-wide state row for its assigned
/// <c>(h, p)</c> pair across every token in the call. Per token <c>t</c>
/// thread 0 broadcasts the per-head <c>decay = exp(adt[t, h])</c>, the
/// scale, and the precomputed skip term <c>d[h] + gamma[t, h] * qkPreDot[t, h]</c>
/// through shared memory (one workgroup barrier), then every thread runs
/// its inner <c>n</c> loop fused — updating its state row and accumulating
/// the state read-out through Q in a single sweep.
/// </para>
/// <para>
/// State persistence across calls is automatic: <c>state</c> is the only
/// read-write buffer. Two consecutive <c>seqLen=4</c> calls on the same
/// state buffer produce the bit-equivalent result of one <c>seqLen=8</c>
/// call (modulo F32 reduction-order noise, which is zero here because the
/// per-thread inner loop is sequential by definition).
/// </para>
/// <para>
/// The recurrence is sequential over <c>t</c> by spec — parallelising
/// across time requires a chunked prefix-scan and is left as a perf
/// follow-up.
/// </para>
/// <para>
/// <b>hasZ flag.</b> When <c>hasZ != 0</c> the shader applies the
/// <c>silu(z)</c> output gate. When <c>hasZ == 0</c> the <c>z</c> buffer
/// is unused but must still be bound (Vulkan requires every binding to
/// resolve to a valid buffer); callers pass any small valid buffer in
/// that case.
/// </para>
/// </remarks>
public sealed class Mamba3CanonicalSsdSisoF32Kernel : IDisposable
{
    // seqLen, nHead, headDim, dState, hasZ (all u32)
    private const int PushConstantBytes = 5 * sizeof(uint);
    private const int BufferCount = 11;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Mamba3CanonicalSsdSisoF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: BufferCount);
    }

    /// <summary>Loads <c>mamba3_canonical_ssd_siso_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Mamba3CanonicalSsdSisoF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "mamba3_canonical_ssd_siso_f32.spv");
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
        return new Mamba3CanonicalSsdSisoF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer state,
        VulkanDevice.Buffer v,
        VulkanDevice.Buffer qRoped,
        VulkanDevice.Buffer kRoped,
        VulkanDevice.Buffer qkPreDot,
        VulkanDevice.Buffer scale,
        VulkanDevice.Buffer gamma,
        VulkanDevice.Buffer adt,
        VulkanDevice.Buffer d,
        VulkanDevice.Buffer z,
        VulkanDevice.Buffer y,
        int seqLen, int nHead, int headDim, int dState, bool hasZ)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, v, qRoped, kRoped, qkPreDot,
               scale, gamma, adt, d, z, y,
               seqLen, nHead, headDim, dState, hasZ);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the canonical-SSD-SISO scan dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are FP32 row-major.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="state">SSM hidden state [<paramref name="nHead"/>, <paramref name="headDim"/>, <paramref name="dState"/>] — read-modify-write.</param>
    /// <param name="v">V per head [<paramref name="seqLen"/>, <paramref name="nHead"/>, <paramref name="headDim"/>].</param>
    /// <param name="qRoped">Post-RoPE Q [<paramref name="seqLen"/>, <paramref name="nHead"/>, <paramref name="dState"/>].</param>
    /// <param name="kRoped">Post-RoPE K [<paramref name="seqLen"/>, <paramref name="nHead"/>, <paramref name="dState"/>].</param>
    /// <param name="qkPreDot">Pre-RoPE QK dot [<paramref name="seqLen"/>, <paramref name="nHead"/>].</param>
    /// <param name="scale">Per-token per-head scale [<paramref name="seqLen"/>, <paramref name="nHead"/>].</param>
    /// <param name="gamma">DT·trap per-token per-head [<paramref name="seqLen"/>, <paramref name="nHead"/>].</param>
    /// <param name="adt">_A·DT per-token per-head (already negative) [<paramref name="seqLen"/>, <paramref name="nHead"/>].</param>
    /// <param name="d">Per-head skip coefficient [<paramref name="nHead"/>].</param>
    /// <param name="z">
    /// Output gate Z [<paramref name="seqLen"/>, <paramref name="nHead"/>, <paramref name="headDim"/>].
    /// When <paramref name="hasZ"/> is false the buffer is bound but ignored — callers may pass any
    /// valid buffer.
    /// </param>
    /// <param name="y">Output [<paramref name="seqLen"/>, <paramref name="nHead"/>, <paramref name="headDim"/>] — written.</param>
    /// <param name="seqLen">Number of tokens to scan (≥ 0; 0 is a no-op).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head P.</param>
    /// <param name="dState">SSM state width N.</param>
    /// <param name="hasZ">Whether to apply the <c>silu(z)</c> output gate.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state,
        VulkanDevice.Buffer v,
        VulkanDevice.Buffer qRoped,
        VulkanDevice.Buffer kRoped,
        VulkanDevice.Buffer qkPreDot,
        VulkanDevice.Buffer scale,
        VulkanDevice.Buffer gamma,
        VulkanDevice.Buffer adt,
        VulkanDevice.Buffer d,
        VulkanDevice.Buffer z,
        VulkanDevice.Buffer y,
        int seqLen, int nHead, int headDim, int dState, bool hasZ)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (seqLen == 0) return; // no-op

        long stateBytes  = (long)nHead * headDim * dState * sizeof(float);
        long vBytes      = (long)seqLen * nHead * headDim * sizeof(float);
        long bcBytes     = (long)seqLen * nHead * dState  * sizeof(float);
        long hdrBytes    = (long)seqLen * nHead           * sizeof(float);
        long dBytes      = (long)nHead                    * sizeof(float);
        long yBytes      = vBytes;
        if (state.Size    < stateBytes) throw new ArgumentException("state buffer too small.",    nameof(state));
        if (v.Size        < vBytes)     throw new ArgumentException("v buffer too small.",        nameof(v));
        if (qRoped.Size   < bcBytes)    throw new ArgumentException("qRoped buffer too small.",   nameof(qRoped));
        if (kRoped.Size   < bcBytes)    throw new ArgumentException("kRoped buffer too small.",   nameof(kRoped));
        if (qkPreDot.Size < hdrBytes)   throw new ArgumentException("qkPreDot buffer too small.", nameof(qkPreDot));
        if (scale.Size    < hdrBytes)   throw new ArgumentException("scale buffer too small.",    nameof(scale));
        if (gamma.Size    < hdrBytes)   throw new ArgumentException("gamma buffer too small.",    nameof(gamma));
        if (adt.Size      < hdrBytes)   throw new ArgumentException("adt buffer too small.",      nameof(adt));
        if (d.Size        < dBytes)     throw new ArgumentException("d buffer too small.",        nameof(d));
        // z: when hasZ we need vBytes; when !hasZ the shader does not read it but a
        // valid buffer must still be bound — so just require Size > 0.
        if (hasZ && z.Size < vBytes)    throw new ArgumentException("z buffer too small.",        nameof(z));
        if (z.Size <= 0)                throw new ArgumentException("z buffer must be a valid (non-empty) buffer even when hasZ is false.", nameof(z));
        if (y.Size        < yBytes)     throw new ArgumentException("y buffer too small.",        nameof(y));

        Span<nint> buffers = stackalloc nint[BufferCount]
        {
            state.Handle, v.Handle, qRoped.Handle, kRoped.Handle, qkPreDot.Handle,
            scale.Handle, gamma.Handle, adt.Handle, d.Handle, z.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes,        (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],   (uint)nHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],   (uint)headDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..],  (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..],  hasZ ? 1u : 0u);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per head — heads are independent. Each workgroup loops
        // over t internally (the recurrence is sequential by spec).
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nHead, 1, 1);
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
