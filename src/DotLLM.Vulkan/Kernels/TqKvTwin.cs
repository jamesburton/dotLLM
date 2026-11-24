using DotLLM.Core.Attention;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The buffers a TurboQuant attention twin reads besides the K and V code buffers (issue #1892): the per-head-vector norms of each
/// side and the shared centroid table. Passed to the twin's <c>Record</c>; null for every other KV storage.
/// </summary>
/// <param name="KeyNorms">K norms of the layer, in the layout's norm format.</param>
/// <param name="ValueNorms">V norms of the layer.</param>
/// <param name="Centroids">The <c>2^mseBits</c> F32 centroids (already scaled by 1/sqrt(d)).</param>
public readonly record struct TqKvOperands(VulkanDevice.Buffer KeyNorms, VulkanDevice.Buffer ValueNorms, VulkanDevice.Buffer Centroids);

/// <summary>
/// Shared plumbing of the <c>_kvtq</c> attention twins (issue #1892, <c>kv_load.glsl</c> <c>KV_TQ</c>). A twin is its F32 body with the K / V
/// loads replaced by "centroid[code] x norm", so besides the body's own bindings it binds the K norms, V norms and centroids at the
/// FIXED bindings 12, 13 and 14 (no body uses them), and takes the layout as specialization constants at fixed high ids: the loader
/// folds at pipeline compile time and costs the body no push-constant register (the hd64 / VGPR cliffs).
/// </summary>
internal static class TqKvTwin
{
    /// <summary>Binding of the K norms; the V norms and the centroids follow at +1 and +2.</summary>
    public const uint FirstBinding = 12;

    /// <summary>Buffers a twin binds after its body's own.</summary>
    public const int ExtraBuffers = 3;

    /// <summary>First <c>constant_id</c> of the loader: mseBits, codeUintsPerVec, normFormat, log2(headDim).</summary>
    public const uint FirstSpecId = 20;

    /// <summary>Extra buffers of a kernel reading <paramref name="tq"/> (0 for a non-TurboQuant kernel).</summary>
    public static int Extra(TurboQuantLayout? tq) => tq is null ? 0 : ExtraBuffers;

    /// <summary>The layout a kernel created for <paramref name="storage"/> needs: required for TurboQuant, refused otherwise.</summary>
    public static TurboQuantLayout? Require(VulkanKvStorage storage, TurboQuantLayout? tq)
    {
        if (storage == VulkanKvStorage.Tq)
        {
            if (tq is not { } l) throw new ArgumentException("A TurboQuant attention twin needs its layout.", nameof(tq));
            if (l.HeadDim <= 0 || (l.HeadDim & (l.HeadDim - 1)) != 0 || l.HeadDim > TqRotateKernel.MaxHeadDim)
                throw new NotSupportedException($"TurboQuant KV needs a power-of-two head dim <= {TqRotateKernel.MaxHeadDim}; got {l.HeadDim} (#1897).");
            if (l.MseBits is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(tq), "mseBits must be 1..8.");
            if (l.Norm is not (TqNormFormat.F16 or TqNormFormat.Bf16 or TqNormFormat.F32Reference))
                throw new NotSupportedException($"The Vulkan TurboQuant loader reads F16, BF16 or (reference) F32 norms; got {l.Norm}.");
            return l;
        }
        if (tq is not null) throw new ArgumentException($"A {storage} kernel takes no TurboQuant layout.", nameof(tq));
        return null;
    }

    /// <summary>
    /// Creates the pipeline of a body with <paramref name="bodyBindings"/> contiguous bindings, adding the TurboQuant bindings and
    /// constants when <paramref name="tq"/> is set. <paramref name="spec"/> are the body's own constants (ids 0..n-1).
    /// </summary>
    public static ComputePipeline CreatePipeline(VulkanModule module, int bodyBindings, uint pushConstantBytes, TurboQuantLayout? tq,
        uint requiredSubgroupSize = 0, ReadOnlySpan<uint> spec = default)
    {
        int extra = Extra(tq);
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[bodyBindings + extra];
        for (int i = 0; i < bodyBindings; i++) bindings[i] = new VkDescriptorBinding((uint)i);
        for (int i = 0; i < extra; i++) bindings[bodyBindings + i] = new VkDescriptorBinding(FirstBinding + (uint)i);
        if (tq is not { } l)
            return module.CreateComputePipeline("main", bindings, pushConstantBytes, requiredSubgroupSize, spec);

        Span<uint> values = stackalloc uint[spec.Length + 4];
        Span<uint> ids = stackalloc uint[spec.Length + 4];
        for (int i = 0; i < spec.Length; i++) { values[i] = spec[i]; ids[i] = (uint)i; }
        int n = spec.Length;
        values[n] = (uint)l.MseBits; ids[n] = FirstSpecId;
        values[n + 1] = (uint)l.CodeUintsPerVec; ids[n + 1] = FirstSpecId + 1;
        values[n + 2] = (uint)l.Norm; ids[n + 2] = FirstSpecId + 2;   // TqNormFormat: 0 F32 (reference), 1 F16, 2 BF16, as the encode shader
        values[n + 3] = (uint)System.Numerics.BitOperations.Log2((uint)l.HeadDim); ids[n + 3] = FirstSpecId + 3;
        return module.CreateComputePipeline("main", bindings, pushConstantBytes, requiredSubgroupSize, values, ids);
    }

    /// <summary>The descriptor cache of a body with <paramref name="bodyBuffers"/> buffers (plus the TurboQuant tail when set).</summary>
    public static DescriptorSetCache CreateCache(VulkanDevice device, nint pool, ComputePipeline pipeline, int bodyBuffers, TurboQuantLayout? tq)
        => tq is null
            ? new DescriptorSetCache(device, pool, pipeline, bodyBuffers)
            : new DescriptorSetCache(device, pool, pipeline, bodyBuffers + ExtraBuffers, tailCount: ExtraBuffers, tailFirstBinding: FirstBinding);

    /// <summary>The descriptor set for the body's <paramref name="bodyBuffers"/> followed by the TurboQuant operands when the kernel reads them.</summary>
    public static nint GetSet(DescriptorSetCache cache, ReadOnlySpan<nint> bodyBuffers, TurboQuantLayout? tq, TqKvOperands? operands)
    {
        if (tq is null)
        {
            if (operands is not null) throw new ArgumentException("This kernel does not read TurboQuant KV.", nameof(operands));
            return cache.GetOrCreate(bodyBuffers);
        }
        if (operands is not { } o) throw new ArgumentException("A TurboQuant attention twin needs the norm and centroid buffers.", nameof(operands));
        Span<nint> all = stackalloc nint[bodyBuffers.Length + ExtraBuffers];
        bodyBuffers.CopyTo(all);
        all[bodyBuffers.Length] = o.KeyNorms.Handle;
        all[bodyBuffers.Length + 1] = o.ValueNorms.Handle;
        all[bodyBuffers.Length + 2] = o.Centroids.Handle;
        return cache.GetOrCreate(all);
    }

    /// <summary>Bytes of the norm buffer of <paramref name="vectors"/> head vectors (F16 / BF16 share a 32-bit word per two vectors).</summary>
    public static long NormBufferBytes(long vectors, TqNormFormat norm) => TurboQuantEncodeF32Kernel.NormBufferBytes(vectors, norm);

    /// <summary>
    /// Validates the K / V buffers of one dispatch: element rows for the element and block formats, code words (plus the straddle
    /// guard word) and norms for TurboQuant. Also refuses a head dim the twin was not specialised for.
    /// </summary>
    public static void CheckKv(VulkanKvStorage storage, TurboQuantLayout? tq, TqKvOperands? operands,
        VulkanDevice.Buffer k, VulkanDevice.Buffer v, int seqKv, int numKvHeads, int headDim)
    {
        if (tq is not { } l)
        {
            if (k.Size < storage.KeyRowBytes((long)numKvHeads * headDim) * seqKv) throw new ArgumentException("K buffer too small.", nameof(k));
            if (v.Size < storage.ValueRowBytes((long)numKvHeads * headDim) * seqKv) throw new ArgumentException("V buffer too small.", nameof(v));
            return;
        }
        if (headDim != l.HeadDim)
            throw new ArgumentException($"This TurboQuant twin is specialised for head dim {l.HeadDim}; got {headDim}.", nameof(headDim));
        if (operands is not { } o) throw new ArgumentException("A TurboQuant attention twin needs the norm and centroid buffers.", nameof(operands));
        long vectors = (long)seqKv * numKvHeads;
        long codeBytes = (vectors * l.CodeUintsPerVec + 1) * sizeof(uint), normBytes = NormBufferBytes(vectors, l.Norm);
        if (k.Size < codeBytes) throw new ArgumentException("K code buffer too small.", nameof(k));
        if (v.Size < codeBytes) throw new ArgumentException("V code buffer too small.", nameof(v));
        if (o.KeyNorms.Size < normBytes || o.ValueNorms.Size < normBytes) throw new ArgumentException("Norm buffer too small.", nameof(operands));
        if (o.Centroids.Size < (1L << l.MseBits) * sizeof(float)) throw new ArgumentException("Centroid buffer too small.", nameof(operands));
    }
}
