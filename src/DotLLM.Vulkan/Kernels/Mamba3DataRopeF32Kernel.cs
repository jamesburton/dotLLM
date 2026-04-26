using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Rotation convention selector for <see cref="Mamba3DataRopeF32Kernel"/>.
/// Mirrors <c>DotLLM.Cpu.Kernels.Mamba3RoPEMode</c> — the C# values are
/// kept bit-identical so callers can cast between them when both backends
/// agree on the convention.
/// </summary>
public enum Mamba3RopeMode
{
    /// <summary>Adjacent-pair rotation <c>(v[2k], v[2k+1])</c> over the first <c>2*numRopeAngles</c> channels (SISO canonical).</summary>
    Pairwise = 0,

    /// <summary>Halved-split rotation <c>(v[k], v[k + dState/2])</c> over the first <c>numRopeAngles</c> lanes of each half (MIMO canonical).</summary>
    Halved = 1,
}

/// <summary>
/// Mamba-3 data-dependent RoPE — canonical (state-spaces/mamba) entry point.
/// Mirrors <c>DotLLM.Cpu.Kernels.Mamba3DataRoPE.ExecuteCanonical</c>: per-token
/// sequential time, per-head and per-lane parallel space.
/// </summary>
/// <remarks>
/// <para>
/// Dispatch model: one workgroup per head (heads are independent), 64
/// threads per workgroup striding over <c>numRopeAngles</c>. Per token <c>t</c>
/// every thread <c>k</c> reads <c>anglesRaw[t, k]</c>, advances its lane of
/// the per-head running cum_angle (<c>cum[h, k] += dt[t, h] * tanh(raw)*π</c>),
/// wraps mod 2π, computes cos/sin, and publishes them through shared memory.
/// After a barrier every thread <c>k &lt; numRopeAngles</c> rotates its
/// lane-pair (Pairwise: <c>(2k, 2k+1)</c>; Halved: <c>(k, k+dState/2)</c>) of
/// b and c across every rank slice.
/// </para>
/// <para>
/// Cum-angle continuity for autoregressive decode is exposed via two flags
/// (<c>hasCumPrev</c>, <c>writeCumOut</c>): pass real buffer handles for both
/// bindings (Vulkan needs them) but flip the flag to seed-from-zero or skip
/// the final write. Two consecutive <c>seqLen=4</c> calls with the previous
/// call's <c>cumOut</c> threaded into the next call's <c>cumPrev</c> produce
/// the bit-equivalent result of one <c>seqLen=8</c> call.
/// </para>
/// <para>
/// The recurrence over <c>t</c> is sequential by spec — parallelising across
/// time would require a chunked prefix-scan with a wrap-aware combiner and is
/// left as a perf follow-up.
/// </para>
/// </remarks>
public sealed class Mamba3DataRopeF32Kernel : IDisposable
{
    private const int WorkgroupSize = 64;
    // seqLen, nRank, nHead, dState, numRopeAngles, mode, hasCumPrev, writeCumOut (all u32)
    private const int PushConstantBytes = 8 * sizeof(uint);
    private const int BufferCount = 6;
    /// <summary>Shader-side upper bound on numRopeAngles (sized for shared mem).</summary>
    private const int MaxRopeAngles = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Mamba3DataRopeF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: BufferCount);
    }

    /// <summary>Loads <c>mamba3_data_rope_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Mamba3DataRopeF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "mamba3_data_rope_f32.spv");
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
        return new Mamba3DataRopeF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        VulkanDevice.Buffer anglesRaw, VulkanDevice.Buffer dt,
        VulkanDevice.Buffer cumPrev, VulkanDevice.Buffer cumOut,
        int seqLen, int nRank, int nHead, int dState, int numRopeAngles,
        Mamba3RopeMode mode,
        bool hasCumPrev, bool writeCumOut)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, b, c, anglesRaw, dt, cumPrev, cumOut,
               seqLen, nRank, nHead, dState, numRopeAngles, mode, hasCumPrev, writeCumOut);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the data-dependent RoPE dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are FP32 row-major.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="b">B coefficient [<paramref name="seqLen"/>, <paramref name="nRank"/>, <paramref name="nHead"/>, <paramref name="dState"/>] — read-modify-write.</param>
    /// <param name="c">C coefficient [<paramref name="seqLen"/>, <paramref name="nRank"/>, <paramref name="nHead"/>, <paramref name="dState"/>] — read-modify-write.</param>
    /// <param name="anglesRaw">Per-token angle projection [<paramref name="seqLen"/>, <paramref name="numRopeAngles"/>] (shared across rank &amp; head).</param>
    /// <param name="dt">Post-softplus timestep [<paramref name="seqLen"/>, <paramref name="nHead"/>].</param>
    /// <param name="cumPrev">Seed cum_angle [<paramref name="nHead"/>, <paramref name="numRopeAngles"/>]. Read iff <paramref name="hasCumPrev"/> is true; pass any valid buffer handle when false.</param>
    /// <param name="cumOut">Final cum_angle [<paramref name="nHead"/>, <paramref name="numRopeAngles"/>]. Written iff <paramref name="writeCumOut"/> is true; pass any valid buffer handle when false.</param>
    /// <param name="seqLen">Number of tokens T (≥ 0; 0 is a no-op).</param>
    /// <param name="nRank">MIMO rank R (SISO = 1).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="dState">State width.</param>
    /// <param name="numRopeAngles">Rotated-pair count — rotates first <c>2 * numRopeAngles</c> channels.</param>
    /// <param name="mode">Pair ordering (Pairwise for SISO, Halved for MIMO).</param>
    /// <param name="hasCumPrev">When true, seed cum from <paramref name="cumPrev"/>; when false, seed to 0.</param>
    /// <param name="writeCumOut">When true, write final cum to <paramref name="cumOut"/>; when false, skip the write.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        VulkanDevice.Buffer anglesRaw, VulkanDevice.Buffer dt,
        VulkanDevice.Buffer cumPrev, VulkanDevice.Buffer cumOut,
        int seqLen, int nRank, int nHead, int dState, int numRopeAngles,
        Mamba3RopeMode mode,
        bool hasCumPrev, bool writeCumOut)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nRank <= 0) throw new ArgumentOutOfRangeException(nameof(nRank));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (numRopeAngles <= 0) throw new ArgumentOutOfRangeException(nameof(numRopeAngles));
        if (numRopeAngles > MaxRopeAngles)
            throw new ArgumentException(
                $"numRopeAngles ({numRopeAngles}) exceeds shader-side upper bound {MaxRopeAngles}.",
                nameof(numRopeAngles));
        int rotaryDim = 2 * numRopeAngles;
        if (rotaryDim > dState)
            throw new ArgumentException(
                $"2*numRopeAngles ({rotaryDim}) > dState ({dState}).", nameof(numRopeAngles));
        if (mode == Mamba3RopeMode.Halved && (dState & 1) != 0)
            throw new ArgumentException(
                $"Halved mode requires even dState, got {dState}.", nameof(dState));
        if (seqLen == 0) return; // no-op

        long bcBytes  = (long)seqLen * nRank * nHead * dState * sizeof(float);
        long angBytes = (long)seqLen * numRopeAngles * sizeof(float);
        long dtBytes  = (long)seqLen * nHead * sizeof(float);
        long cumBytes = (long)nHead * numRopeAngles * sizeof(float);
        if (b.Size < bcBytes)         throw new ArgumentException("b buffer too small.",         nameof(b));
        if (c.Size < bcBytes)         throw new ArgumentException("c buffer too small.",         nameof(c));
        if (anglesRaw.Size < angBytes) throw new ArgumentException("anglesRaw buffer too small.", nameof(anglesRaw));
        if (dt.Size < dtBytes)        throw new ArgumentException("dt buffer too small.",        nameof(dt));
        // cumPrev/cumOut sizes only matter when their flag is set — the shader
        // only reads/writes them under the corresponding flag. The buffer must
        // still be a valid Vulkan handle so the descriptor binding works.
        if (hasCumPrev && cumPrev.Size < cumBytes)
            throw new ArgumentException("cumPrev buffer too small.", nameof(cumPrev));
        if (writeCumOut && cumOut.Size < cumBytes)
            throw new ArgumentException("cumOut buffer too small.", nameof(cumOut));

        Span<nint> buffers = stackalloc nint[BufferCount]
        {
            b.Handle, c.Handle, anglesRaw.Handle, dt.Handle, cumPrev.Handle, cumOut.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes,        (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],   (uint)nRank);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],   (uint)nHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..],  (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..],  (uint)numRopeAngles);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[20..],  (uint)mode);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[24..],  hasCumPrev  ? 1u : 0u);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[28..],  writeCumOut ? 1u : 0u);
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
