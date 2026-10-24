using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// GQA-grouped tiled (flash) prefill attention for head_dim up to 256 (issue #441): one workgroup serves a block of query rows
/// drawn from all <c>numHeads / numKvHeads</c> query heads of one KV head, so each K / V tile is read once per group instead of once
/// per query head, and the KV cache is read in its stored format (F32 or F16) in place.
/// </summary>
/// <remarks>
/// <para>
/// Two implementations behind one API. On a device with a subgroup-scope 16x16x16 f16 cooperative-matrix tile and a pinnable
/// 32- or 64-lane subgroup (Strix's RDNA 3.5, NVIDIA), <c>attention_flash_gqa_cm</c> runs Q.K^T and P.V on the matrix cores with
/// Q and O in registers (64 query rows per workgroup, one 16-row strip per subgroup). Elsewhere (the Meteor Lake Arc), the scalar
/// <c>attention_flash_gqa</c> keeps each row's Q and O in the registers of 8 lanes (16 rows per workgroup). Both accumulate in F32
/// and keep the online-softmax state in F32 (the precision rules); see the shader headers for the numerics and the LDS budgets
/// (both fit 32 KiB).
/// </para>
/// <para>
/// It replaces, for head_dim above 128, <see cref="VulkanFlashAttentionCoopmatKernel"/>'s hd256 pipeline (one query head x 16 rows
/// per workgroup, ~19 workgroup barriers per 32-column KV tile; measured ~1 TFLOP/s at depth on Strix) and the scalar
/// <see cref="VulkanFlashAttentionF32Kernel"/> wide variants. <c>DOTLLM_VULKAN_FA_GQA=0</c> turns it off for A/B runs.
/// </para>
/// <para>
/// Scope: head_dim a multiple of 16 (cooperative-matrix) or 32 (scalar) up to 256, no ALiBi; every mask mode, sliding window,
/// soft-cap and scale override is supported. <see cref="CanDispatch"/> says whether a shape qualifies.
/// </para>
/// <para>
/// An instance created with <c>sinks: true</c> runs the <c>_sinks</c> twins (issue #1200): a fifth binding carries one learned logit per
/// query head (gpt-oss, Flint) that joins the softmax denominator only, as in <see cref="VulkanGptOssKernels.RecordAttentionSinks"/>.
/// </para>
/// </remarks>
public sealed class VulkanFlashAttentionGqaKernel : IDisposable
{
    /// <summary>Largest head_dim either implementation dispatches.</summary>
    public const int MaxHeadDim = 256;

    /// <summary>Environment variable: <c>0</c> disables the kernel (hosts fall back to the per-head flash kernels).</summary>
    public const string DisableEnvVar = "DOTLLM_VULKAN_FA_GQA";

    /// <summary>Environment variable: <c>scalar</c> forces the scalar implementation even on a cooperative-matrix device (A/B).</summary>
    public const string ForceScalarEnvVar = "DOTLLM_VULKAN_FA_GQA_SCALAR";

    /// <summary>
    /// Environment variable: <c>1</c> restores the cooperative-matrix QK^T's Q lo-half term (#1148). Off by default: Q is rounded to
    /// F16, attention is 1.67x faster at depth on Strix, and the #1109 study measured KL at the F32 noise floor to 128k on Bonsai-2.
    /// That rests on RMS-normed q/k (max |s| ≈ 33 over a 128k prefill); re-run <c>DOTLLM_VK_EXP_FA_ACC=stats</c> before relying on
    /// it for a model without q/k norm.
    /// </summary>
    public const string SplitQEnvVar = "DOTLLM_VULKAN_FA_GQA_QLO";

    private const string CoopmatShader = "attention_flash_gqa_cm";
    private const string ScalarShader = "attention_flash_gqa";
    private const int PushConstantBytes = 12 * sizeof(uint);
    private const int CoopmatRowsPerWorkgroup = 64;
    private const int ScalarRowsPerWorkgroup = 16;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private VulkanFlashAttentionGqaKernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, bool isCoopmat, VulkanKvStorage kvStorage,
        bool hasSinks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BindingCount(hasSinks));
        IsCoopmat = isCoopmat;
        KvStorage = kvStorage;
        HasSinks = hasSinks;
    }

    /// <summary>Gets a value indicating whether this instance runs the cooperative-matrix implementation.</summary>
    public bool IsCoopmat { get; }

    /// <summary>Gets the KV element format this instance reads.</summary>
    public VulkanKvStorage KvStorage { get; }

    /// <summary>Gets a value indicating whether this instance runs the attention-sink twins (a sinks buffer is required).</summary>
    public bool HasSinks { get; }

    /// <summary>Gets the number of dispatches recorded (test and profiling hook).</summary>
    public long Dispatches { get; private set; }

    /// <summary>Gets a value indicating whether <see cref="DisableEnvVar"/> turns the kernel off.</summary>
    public static bool IsDisabled() => Environment.GetEnvironmentVariable(DisableEnvVar) == "0";

    /// <summary>
    /// Whether this instance can serve a shape: head_dim within bounds and a multiple of the implementation's granule, a whole
    /// number of query heads per KV head, and no ALiBi.
    /// </summary>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="useAlibi">Whether ALiBi biases are requested.</param>
    /// <returns>True when <see cref="Record"/> accepts the shape.</returns>
    public bool CanDispatch(int headDim, int numHeads, int numKvHeads, bool useAlibi = false)
        => !useAlibi && headDim > 0 && headDim <= MaxHeadDim && headDim % (IsCoopmat ? 16 : 32) == 0
           && numKvHeads > 0 && numHeads % numKvHeads == 0;

    /// <summary>
    /// Creates the kernel for <paramref name="kvStorage"/>, picking the cooperative-matrix implementation when the device supports
    /// it and the scalar one otherwise; null when neither can be built (missing SPIR-V, unsuitable subgroups, a driver failure).
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="kvStorage">The KV element format to read.</param>
    /// <param name="allowCoopmat">False forces the scalar implementation.</param>
    /// <param name="sinks">True builds the attention-sink twins.</param>
    /// <param name="splitQ">Cooperative-matrix only: whether QK^T adds the Q lo-half term; null reads <see cref="SplitQEnvVar"/>.</param>
    /// <returns>The kernel, or null.</returns>
    public static VulkanFlashAttentionGqaKernel? TryCreate(
        VulkanDevice device, string spvDir, VulkanKvStorage kvStorage = VulkanKvStorage.F32, bool allowCoopmat = true,
        bool sinks = false, bool? splitQ = null)
    {
        if (IsDisabled()) return null;
        bool forceScalar = string.Equals(Environment.GetEnvironmentVariable(ForceScalarEnvVar), "1", StringComparison.Ordinal)
            || string.Equals(Environment.GetEnvironmentVariable(ForceScalarEnvVar), "scalar", StringComparison.OrdinalIgnoreCase);
        if (allowCoopmat && !forceScalar)
        {
            var cm = TryCreateCoopmat(device, spvDir, kvStorage, splitQ, sinks);
            if (cm is not null) return cm;
        }

        return TryCreateScalar(device, spvDir, kvStorage, sinks);
    }

    /// <summary>Creates the cooperative-matrix implementation, or null when the device or build cannot run it.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="kvStorage">The KV element format to read.</param>
    /// <param name="splitQ">Whether QK^T adds the Q lo-half term; null reads <see cref="SplitQEnvVar"/> (off unless <c>1</c>).</param>
    /// <param name="sinks">True builds the attention-sink twin.</param>
    /// <returns>The kernel, or null.</returns>
    internal static VulkanFlashAttentionGqaKernel? TryCreateCoopmat(
        VulkanDevice device, string spvDir, VulkanKvStorage kvStorage, bool? splitQ = null, bool sinks = false)
    {
        if (!VulkanFlashAttentionCoopmatKernel.SupportsDevice(device)) return null;

        // The shader assumes exactly 4 subgroups of 32 or 64 lanes per workgroup: pin the wave width unless the device has one.
        uint sg = device.SubgroupSize;
        if (sg != 32 && sg != 64) return null;
        bool fixedWidth = device.MinSubgroupSize == device.MaxSubgroupSize && device.MinSubgroupSize == sg;
        uint pin = 0;
        if (!fixedWidth)
        {
            if (!device.SupportsRequiredSubgroupSize(sg, VkShaderStageFlags.Compute)) return null;
            pin = sg;
        }

        uint gate = VulkanFlashAttentionCoopmatKernel.RequiresInvariantPv(device.VendorId);
        bool split = splitQ ?? Environment.GetEnvironmentVariable(SplitQEnvVar) == "1";
        ReadOnlySpan<uint> spec = [gate, 4 * sg, split ? 1u : 0u];
        return TryBuild(device, spvDir, CoopmatShader, pin, spec, isCoopmat: true, kvStorage, sinks);
    }

    /// <summary>Creates the scalar implementation, or null when the build or device cannot run it.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="kvStorage">The KV element format to read.</param>
    /// <param name="sinks">True builds the attention-sink twin.</param>
    /// <returns>The kernel, or null.</returns>
    internal static VulkanFlashAttentionGqaKernel? TryCreateScalar(
        VulkanDevice device, string spvDir, VulkanKvStorage kvStorage, bool sinks = false)
    {
        // A row's 8 lanes must share a subgroup: every subgroup the driver may pick must have at least 8 lanes.
        uint minWidth = device.MinSubgroupSize != 0 ? device.MinSubgroupSize : device.SubgroupSize;
        uint pin = 0;
        if (minWidth < 8)
        {
            uint want = Math.Max(device.SubgroupSize, 8u);
            if (!device.SupportsRequiredSubgroupSize(want, VkShaderStageFlags.Compute)) return null;
            pin = want;
        }

        return TryBuild(device, spvDir, ScalarShader, pin, default, isCoopmat: false, kvStorage, sinks);
    }

    // The sink twins add binding 4 (the per-head sink logits).
    private static int BindingCount(bool sinks) => sinks ? 5 : 4;

    private static VulkanFlashAttentionGqaKernel? TryBuild(
        VulkanDevice device, string spvDir, string baseShader, uint requiredSubgroupSize, ReadOnlySpan<uint> spec,
        bool isCoopmat, VulkanKvStorage kvStorage, bool sinks)
    {
        string shaderName = kvStorage.ShaderName(sinks ? baseShader + "_sinks" : baseShader);
        int bindingCount = BindingCount(sinks);
        string path = Path.Combine(spvDir, shaderName + ".spv");
        if (!File.Exists(path)) return null;
        VulkanModule? module = null;
        ComputePipeline? pipeline = null;
        try
        {
            module = VulkanModule.LoadFromFile(device, path);
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[bindingCount];
            for (int i = 0; i < bindingCount; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: requiredSubgroupSize,
                specConstants: spec);
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)bindingCount);
            return new VulkanFlashAttentionGqaKernel(device, module, pipeline, pool, isCoopmat, kvStorage, sinks);
        }
        catch (VulkanException)
        {
            pipeline?.Dispose();
            module?.Dispose();
            return null;
        }
    }

    /// <summary>Drops cached descriptor sets (a buffer they name was freed or re-created).</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records, submits and waits for one dispatch (tests and benches).</summary>
    /// <param name="q">Queries, F32 <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Keys, <see cref="KvStorage"/> <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Values, same layout as <paramref name="k"/>.</param>
    /// <param name="output">Output, F32 <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">KV positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="positionOffset">Absolute position of the first query row.</param>
    /// <param name="slidingWindow">Sliding window (0 = none).</param>
    /// <param name="softCap">Score soft-cap (0 = none).</param>
    /// <param name="scaleOverride">Score scale (0 = 1/sqrt(headDim)).</param>
    /// <param name="maskMode">Mask mode.</param>
    /// <param name="prefixLen">Hybrid-mask causal prefix length.</param>
    /// <param name="sinks">Per-head sink logits, F32 <c>[numHeads]</c>; required exactly when <see cref="HasSinks"/>.</param>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0, VulkanDevice.Buffer? sinks = null)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
            positionOffset, slidingWindow, softCap, scaleOverride, maskMode, prefixLen, sinks);
        ctx.SubmitAndWait();
    }

    /// <summary>Records one dispatch into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">The command buffer.</param>
    /// <param name="q">Queries, F32 <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Keys, <see cref="KvStorage"/> <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Values, same layout as <paramref name="k"/>.</param>
    /// <param name="output">Output, F32 <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">KV positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="positionOffset">Absolute position of the first query row.</param>
    /// <param name="slidingWindow">Sliding window (0 = none).</param>
    /// <param name="softCap">Score soft-cap (0 = none).</param>
    /// <param name="scaleOverride">Score scale (0 = 1/sqrt(headDim)).</param>
    /// <param name="maskMode">Mask mode.</param>
    /// <param name="prefixLen">Hybrid-mask causal prefix length.</param>
    /// <param name="sinks">Per-head sink logits, F32 <c>[numHeads]</c>; required exactly when <see cref="HasSinks"/>.</param>
    /// <exception cref="ArgumentException">The shape is outside <see cref="CanDispatch"/>, a buffer is too small, or
    /// <paramref name="sinks"/> does not match <see cref="HasSinks"/>.</exception>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0, VulkanDevice.Buffer? sinks = null)
    {
        if (HasSinks != sinks is not null)
            throw new ArgumentException(HasSinks ? "This sink kernel needs a sinks buffer." : "This kernel has no sink binding.", nameof(sinks));
        if (sinks is not null && sinks.Size < (long)numHeads * sizeof(float))
            throw new ArgumentException("Sinks buffer too small.", nameof(sinks));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqQ);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqKv);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(slidingWindow);
        ArgumentOutOfRangeException.ThrowIfNegative(prefixLen);
        if (!CanDispatch(headDim, numHeads, numKvHeads))
            throw new ArgumentException(
                $"Shape headDim={headDim} heads={numHeads}/{numKvHeads} is outside this kernel ({(IsCoopmat ? "coopmat" : "scalar")}).",
                nameof(headDim));

        long qBytes = (long)seqQ * numHeads * headDim * sizeof(float);
        long kvBytes = (long)seqKv * numKvHeads * headDim * KvStorage.ElementBytes();
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (k.Size < kvBytes) throw new ArgumentException("K buffer too small.", nameof(k));
        if (v.Size < kvBytes) throw new ArgumentException("V buffer too small.", nameof(v));
        if (output.Size < qBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { q.Handle, k.Handle, v.Handle, output.Handle, sinks?.Handle ?? 0 };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers[..BindingCount(HasSinks)]);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[12];
        pc[0] = (uint)seqQ;
        pc[1] = (uint)seqKv;
        pc[2] = (uint)numHeads;
        pc[3] = (uint)numKvHeads;
        pc[4] = (uint)headDim;
        pc[5] = (uint)positionOffset;
        pc[6] = (uint)slidingWindow;
        pc[7] = 0u;
        pc[8] = BitConverter.SingleToUInt32Bits(softCap);
        pc[9] = BitConverter.SingleToUInt32Bits(scaleOverride);
        pc[10] = (uint)maskMode;
        pc[11] = (uint)prefixLen;
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        long rows = (long)seqQ * (numHeads / numKvHeads);
        int rowsPerGroup = IsCoopmat ? CoopmatRowsPerWorkgroup : ScalarRowsPerWorkgroup;
        uint groups = (uint)((rows + rowsPerGroup - 1) / rowsPerGroup);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, (uint)numKvHeads, 1);
        Dispatches++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
