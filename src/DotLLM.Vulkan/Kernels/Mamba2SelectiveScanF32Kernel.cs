using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Mamba2 selective state-space scan — Nemotron-H SSM recurrent kernel.
/// Mirrors <c>DotLLM.Cpu.Kernels.Mamba2SelectiveScan.Execute</c>: per-token
/// sequential time, per-head and per-(head_dim, dstate) parallel space.
/// </summary>
/// <remarks>
/// <para>
/// Dispatch model: one workgroup per head (heads are independent), 64
/// threads per workgroup striding over <c>head_dim</c>. Each thread owns
/// the <c>dState</c>-wide state row for its assigned <c>(h, i)</c> pair
/// across every token in the call. Per token <c>t</c> thread 0 broadcasts
/// the per-head <c>dt_sp = softplus(dt[t, h])</c> and decay
/// <c>dA = exp(dt_sp * a[h])</c> through shared memory (one workgroup
/// barrier), then every thread runs its inner <c>k</c> loop fused —
/// updating its state row and accumulating the output projection through
/// <c>C</c> in a single sweep.
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
/// across time requires a chunked prefix-scan (Mamba-2's selective-scan-2D
/// algorithm) and is left as a perf follow-up.
/// </para>
/// </remarks>
public sealed class Mamba2SelectiveScanF32Kernel : IDisposable
{
    private const int WorkgroupSize = 64;
    // nHead, headDim, dState, nGroup, seqLen (all u32)
    private const int PushConstantBytes = 5 * sizeof(uint);
    private const int BufferCount = 7;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Mamba2SelectiveScanF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: BufferCount);
    }

    /// <summary>Loads <c>mamba2_selective_scan_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Mamba2SelectiveScanF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "mamba2_selective_scan_f32.spv");
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
        return new Mamba2SelectiveScanF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer state, VulkanDevice.Buffer x, VulkanDevice.Buffer dt,
        VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        VulkanDevice.Buffer y,
        int nHead, int headDim, int dState, int nGroup, int seqLen)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, x, dt, a, b, c, y, nHead, headDim, dState, nGroup, seqLen);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the selective-scan dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are FP32 row-major.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="state">SSM hidden state [<paramref name="nHead"/>, <paramref name="headDim"/>, <paramref name="dState"/>] — read-modify-write.</param>
    /// <param name="x">Post-conv activations [<paramref name="seqLen"/>, <paramref name="nHead"/>*<paramref name="headDim"/>].</param>
    /// <param name="dt">Time-step parameter [<paramref name="seqLen"/>, <paramref name="nHead"/>] — already bias-added.</param>
    /// <param name="a">A parameter [<paramref name="nHead"/>] — scalar per head (Mamba-2).</param>
    /// <param name="b">B coefficient [<paramref name="seqLen"/>, <paramref name="nGroup"/>, <paramref name="dState"/>].</param>
    /// <param name="c">C coefficient [<paramref name="seqLen"/>, <paramref name="nGroup"/>, <paramref name="dState"/>].</param>
    /// <param name="y">Output [<paramref name="seqLen"/>, <paramref name="nHead"/>*<paramref name="headDim"/>] — written.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head (<c>d_inner / n_head</c>).</param>
    /// <param name="dState">SSM state width (typically 128).</param>
    /// <param name="nGroup">Number of B/C groups (<paramref name="nHead"/> must be divisible by it).</param>
    /// <param name="seqLen">Number of tokens to scan (≥ 0; 0 is a no-op).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state, VulkanDevice.Buffer x, VulkanDevice.Buffer dt,
        VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        VulkanDevice.Buffer y,
        int nHead, int headDim, int dState, int nGroup, int seqLen)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead % nGroup != 0)
            throw new ArgumentException($"nHead ({nHead}) must be divisible by nGroup ({nGroup}).", nameof(nGroup));
        if (seqLen == 0) return; // no-op

        int dInner = nHead * headDim;
        long stateBytes = (long)nHead * headDim * dState * sizeof(float);
        long xBytes = (long)seqLen * dInner * sizeof(float);
        long dtBytes = (long)seqLen * nHead * sizeof(float);
        long aBytes = (long)nHead * sizeof(float);
        long bBytes = (long)seqLen * nGroup * dState * sizeof(float);
        long cBytes = (long)seqLen * nGroup * dState * sizeof(float);
        long yBytes = xBytes;
        if (state.Size < stateBytes) throw new ArgumentException("state buffer too small.", nameof(state));
        if (x.Size < xBytes)         throw new ArgumentException("x buffer too small.", nameof(x));
        if (dt.Size < dtBytes)       throw new ArgumentException("dt buffer too small.", nameof(dt));
        if (a.Size < aBytes)         throw new ArgumentException("a buffer too small.", nameof(a));
        if (b.Size < bBytes)         throw new ArgumentException("b buffer too small.", nameof(b));
        if (c.Size < cBytes)         throw new ArgumentException("c buffer too small.", nameof(c));
        if (y.Size < yBytes)         throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[BufferCount]
        {
            state.Handle, x.Handle, dt.Handle, a.Handle, b.Handle, c.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes,        (uint)nHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],   (uint)headDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],   (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..],  (uint)nGroup);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..],  (uint)seqLen);
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
