using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// KHR cooperative-matrix Flash-Attention kernel for the GQA prefill path
/// (issue #149). Same tiling, dispatch geometry, bindings and push constants
/// as <see cref="VulkanFlashAttentionF32Kernel"/>, but QK^T and P·V run on
/// the matrix cores via 16x16x16 f16xf16-&gt;f32 cooperative-matrix tiles —
/// llama.cpp's FA_COOPMAT1 shape (Br=16, Bc=4x16 subgroup slices).
/// </summary>
/// <remarks>
/// <para>
/// Parity target: <c>DotLLM.Cpu.Kernels.Attention.ExecuteScalar</c> (same as
/// the scalar FA kernel) — but at f16-input tolerance, NOT the all-F32
/// tolerance: Q/K/V are rounded to f16 for the matrix multiplies, exactly the
/// rounding class llama.cpp ships (its KV cache is f16). Softmax state and
/// both matmul accumulators stay F32. See
/// <c>attention_flash_f32_coopmat.comp</c> for the full numerics contract.
/// </para>
/// <para>
/// Availability: requires <see cref="VulkanDevice.HasCooperativeMatrix"/>
/// with a subgroup-scope 16x16x16 F16xF16-&gt;F32 tile (checked via
/// <see cref="SupportsDevice"/>). Callers route to the scalar FA kernel when
/// unavailable, when the env kill-switch
/// <c>DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1</c> is set, or when
/// <c>headDim &gt; MaxHeadDim</c>. Decode (seqQ == 1) must never route here —
/// llama.cpp itself prefers FA_SCALAR at n_rows==1, matching the 2026-04
/// finding that coopmat attention loses at decode shapes on gfx1151.
/// </para>
/// </remarks>
public sealed class VulkanFlashAttentionCoopmatKernel : IDisposable
{
    /// <summary>Compile-time upper bound on head_dim baked into the shader.</summary>
    public const int MaxHeadDim = 128;

    /// <summary>Query rows per workgroup (Br).</summary>
    public const int QueryTileRows = 16;

    /// <summary>KV tile columns per workgroup iteration (Bc).</summary>
    public const int KvTileCols = 64;

    private const int PushConstantBytes = 12 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private VulkanFlashAttentionCoopmatKernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Returns <c>true</c> when the device advertises a subgroup-scope
    /// 16x16x16 F16 x F16 -&gt; F32 cooperative-matrix tile — the exact shape
    /// the shader's <c>coopMatMulAdd</c> calls require.
    /// </summary>
    public static bool SupportsDevice(VulkanDevice device)
    {
        if (!device.HasCooperativeMatrix) return false;
        foreach (var s in device.SupportedCooperativeMatrixProperties)
        {
            if (s.Scope == VkScopeKhr.Subgroup
                && s.MSize == 16 && s.NSize == 16 && s.KSize == 16
                && s.AType == VkComponentTypeKhr.Float16 && s.BType == VkComponentTypeKhr.Float16
                && s.CType == VkComponentTypeKhr.Float32 && s.ResultType == VkComponentTypeKhr.Float32)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Loads <c>attention_flash_f32_coopmat.spv</c> and creates the compute
    /// pipeline. Throws when the device lacks the required coopmat tile or
    /// the SPV is missing — use <see cref="TryCreate"/> for graceful fallback.
    /// </summary>
    public static VulkanFlashAttentionCoopmatKernel Create(VulkanDevice device, string spvDir)
    {
        if (!SupportsDevice(device))
            throw new InvalidOperationException(
                "VulkanFlashAttentionCoopmatKernel requires a subgroup-scope 16x16x16 " +
                "F16xF16->F32 VK_KHR_cooperative_matrix tile. Check SupportsDevice() first " +
                "and fall back to VulkanFlashAttentionF32Kernel.");

        string path = Path.Combine(spvDir, "attention_flash_f32_coopmat.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
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
        return new VulkanFlashAttentionCoopmatKernel(device, module, pipeline, pool);
    }

    /// <summary>
    /// Returns <c>null</c> when the device lacks coopmat support, the SPV is
    /// missing, or pipeline creation fails — callers fall back to the scalar
    /// FA kernel.
    /// </summary>
    public static VulkanFlashAttentionCoopmatKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!SupportsDevice(device)) return null;
        try
        {
            return Create(device, spvDir);
        }
        catch (FileNotFoundException)
        {
            return null;
        }
        catch (VulkanException)
        {
            return null;
        }
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers were re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Synchronous one-shot launch (parity tests). Production callers use
    /// <see cref="Record"/> inside a batched command buffer.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, bool useAlibi = false,
        float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
               positionOffset, slidingWindow, useAlibi, softCap, scaleOverride, maskMode, prefixLen);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the coopmat FA dispatch. Contract is identical to
    /// <see cref="VulkanFlashAttentionF32Kernel.Record"/> — same buffer
    /// shapes, same parameters, same dispatch geometry.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, bool useAlibi = false,
        float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (headDim > MaxHeadDim)
            throw new ArgumentException(
                $"headDim ({headDim}) exceeds shader MAX_HEAD_DIM ({MaxHeadDim}). " +
                $"Route to {nameof(VulkanFlashAttentionF32Kernel)} / {nameof(AttentionF32Kernel)}.",
                nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (slidingWindow < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));
        if (softCap < 0.0f) throw new ArgumentOutOfRangeException(nameof(softCap),
            "softCap must be non-negative (use 0 to disable).");
        if (scaleOverride < 0.0f) throw new ArgumentOutOfRangeException(nameof(scaleOverride),
            "scaleOverride must be non-negative (use 0 for the default 1/sqrt(headDim)).");
        if (prefixLen < 0) throw new ArgumentOutOfRangeException(nameof(prefixLen));

        long qBytes   = (long)seqQ  * numHeads   * headDim * sizeof(float);
        long kvBytes  = (long)seqKv * numKvHeads * headDim * sizeof(float);
        long outBytes = qBytes;
        if (q.Size      < qBytes)   throw new ArgumentException("Q buffer too small.",      nameof(q));
        if (k.Size      < kvBytes)  throw new ArgumentException("K buffer too small.",      nameof(k));
        if (v.Size      < kvBytes)  throw new ArgumentException("V buffer too small.",      nameof(v));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[4] { q.Handle, k.Handle, v.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[12];
        pc[0]  = (uint)seqQ;
        pc[1]  = (uint)seqKv;
        pc[2]  = (uint)numHeads;
        pc[3]  = (uint)numKvHeads;
        pc[4]  = (uint)headDim;
        pc[5]  = (uint)positionOffset;
        pc[6]  = (uint)slidingWindow;
        pc[7]  = useAlibi ? 1u : 0u;
        pc[8]  = BitConverter.SingleToUInt32Bits(softCap);
        pc[9]  = BitConverter.SingleToUInt32Bits(scaleOverride);
        pc[10] = (uint)maskMode;
        pc[11] = (uint)prefixLen;
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint qTiles = ((uint)seqQ + (uint)QueryTileRows - 1u) / (uint)QueryTileRows;
        uint groups = qTiles * (uint)numHeads;
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
