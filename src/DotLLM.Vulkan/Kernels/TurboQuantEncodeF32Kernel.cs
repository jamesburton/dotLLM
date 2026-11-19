using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// TurboQuant (MSE stage) KV encode from FP32 — the GPU port of
/// <c>DotLLM.Engine.KvCache.Codecs.TurboQuantCodec.Encode</c> (without the QJL residual).
/// Reads contiguous fresh K/V activations <c>[seqLen, numKvHeads*headDim]</c> and writes packed
/// per-head-vector Lloyd–Max codes (uint-aligned per vector) + a norm in the cache's <see cref="TqNormFormat"/>.
/// </summary>
/// <remarks>
/// Per head-vector: norm → unit direction → forward RHT (sign-flip, Walsh–Hadamard, ×invSqrtD) →
/// nearest centroid → bit-pack. <c>centroids</c>/<c>signs</c>/<c>invSqrtD</c> are codec constants
/// supplied as buffers / push constants (backend-pure). One workgroup per head-vector, 256 threads;
/// each workgroup owns its (disjoint, uint-aligned) code uints, so output is single-writer — no
/// atomics or pre-zeroing (the 16-bit norm halves are the one exception: two vectors share a uint, written with
/// atomicAnd/atomicOr). Encode is byte-identical to the CPU codec and the CUDA kernel (#1892): fixed pairwise-tree
/// norm, IEEE sqrt/divide, NoContraction arithmetic, lower-index tie policy. Supports
/// <c>headDim</c> a power of two ≤ 256. Destination head-vector index is
/// <c>(startPos + srcRow)*numKvHeads + h</c> (contiguous append).
/// </remarks>
public sealed class TurboQuantEncodeF32Kernel : IDisposable
{
    /// <summary>Maximum supported per-head dimension (workgroup size / shared-array bound).</summary>
    public const int MaxHeadDim = 256;

    private const int BindingCount = 5;
    private const int PushConstantBytes = 8 * sizeof(uint); // 6 uints + 1 float + normFormat

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private TurboQuantEncodeF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BindingCount);
    }

    /// <summary>Loads <c>turboquant_encode_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static TurboQuantEncodeF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "turboquant_encode_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BindingCount];
            for (int i = 0; i < BindingCount; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: BindingCount);
        return new TurboQuantEncodeF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Bytes a norm buffer must hold for <paramref name="vectors"/> head-vectors (rounded up to whole uints).</summary>
    public static long NormBufferBytes(long vectors, TqNormFormat f)
        => (vectors * new TurboQuantLayout(1, 1, 1, f).NormBytes + 3) / 4 * 4;

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the encode synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer src, VulkanDevice.Buffer centroids, VulkanDevice.Buffer signs,
        VulkanDevice.Buffer codes, VulkanDevice.Buffer norms,
        int seqLen, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, int startPos, float invSqrtD,
        TqNormFormat normFormat = TqNormFormat.F32Reference)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, centroids, signs, codes, norms,
               seqLen, headDim, numKvHeads, mseBits, codeUintsPerVec, startPos, invSqrtD, normFormat);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the encode into <paramref name="cmdBuf"/> without submitting. One workgroup per
    /// head-vector (<paramref name="seqLen"/> × numKvHeads).</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer src, VulkanDevice.Buffer centroids, VulkanDevice.Buffer signs,
        VulkanDevice.Buffer codes, VulkanDevice.Buffer norms,
        int seqLen, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, int startPos, float invSqrtD,
        TqNormFormat normFormat = TqNormFormat.F32Reference)
    {
        if (normFormat is not (TqNormFormat.F32Reference or TqNormFormat.F16 or TqNormFormat.Bf16))
            throw new NotSupportedException($"Vulkan TurboQuant encode supports F32Reference, F16 and Bf16 norms; got {normFormat}.");
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > MaxHeadDim)
            throw new ArgumentException($"headDim must be a power of two in [1,{MaxHeadDim}]; got {headDim}.", nameof(headDim));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (mseBits is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(mseBits));
        if (startPos < 0) throw new ArgumentOutOfRangeException(nameof(startPos));

        int numVectors = seqLen * numKvHeads;
        int levelCount = 1 << mseBits;

        long srcMin = (long)seqLen * numKvHeads * headDim * sizeof(float);
        if (src.Size < srcMin) throw new ArgumentException("src buffer too small.", nameof(src));
        if (centroids.Size < (long)levelCount * sizeof(float)) throw new ArgumentException("centroids buffer too small.", nameof(centroids));
        if (signs.Size < (long)headDim * sizeof(float)) throw new ArgumentException("signs buffer too small.", nameof(signs));
        // Destination spans positions [startPos, startPos+seqLen); the last vector's codes occupy
        // codeUintsPerVec uints.
        long codeUintsMin = (long)(startPos + seqLen) * numKvHeads * codeUintsPerVec;
        if (codes.Size < codeUintsMin * sizeof(uint)) throw new ArgumentException("codes buffer too small.", nameof(codes));
        if (norms.Size < NormBufferBytes((long)(startPos + seqLen) * numKvHeads, normFormat)) throw new ArgumentException("norms buffer too small.", nameof(norms));

        Span<nint> buffers = stackalloc nint[BindingCount]
            { src.Handle, centroids.Handle, signs.Handle, codes.Handle, norms.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[8]
        {
            (uint)headDim, (uint)numKvHeads, (uint)mseBits, (uint)codeUintsPerVec,
            (uint)levelCount, (uint)startPos, BitConverter.SingleToUInt32Bits(invSqrtD), (uint)normFormat,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)numVectors, 1, 1);
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
