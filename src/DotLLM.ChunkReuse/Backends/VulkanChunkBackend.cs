using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.ChunkReuse.Backends;

/// <summary>
/// Model features that change how cached keys must be re-rotated but are not part of <see cref="ModelConfig"/>: a GGUF
/// <c>rope_freqs.weight</c> tensor (Llama-3.x per-pair frequency factors) and attention sinks (<c>blk.N.attn_sinks.weight</c>).
/// </summary>
/// <param name="HasRopeFreqFactors">The weights carry <c>rope_freqs.weight</c>.</param>
/// <param name="HasAttentionSinks">Some layer carries <c>attn_sinks.weight</c>.</param>
public readonly record struct ChunkModelFeatures(bool HasRopeFreqFactors, bool HasAttentionSinks)
{
    /// <summary>Reads the features from the tensor names of <paramref name="gguf"/>.</summary>
    /// <param name="gguf">The model file.</param>
    /// <returns>The features.</returns>
    public static ChunkModelFeatures FromGguf(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        bool factors = gguf.TensorsByName.ContainsKey("rope_freqs.weight");
        bool sinks = false;
        foreach (string name in gguf.TensorsByName.Keys)
        {
            if (name.EndsWith(".attn_sinks.weight", StringComparison.Ordinal))
            {
                sinks = true;
                break;
            }
        }
        return new ChunkModelFeatures(factors, sinks);
    }
}

/// <summary>
/// The Vulkan backend: rows move through <see cref="IPositionedHostKvCache"/> on a <see cref="VulkanKvCache"/> with F32 storage, and
/// keys are re-rotated on the device with <see cref="RopeShiftKvKernel"/>. Only plain RoPE is accepted
/// (<see cref="CheckModelSupported"/>); a model with YaRN, frequency factors, MLA or attention sinks is refused at construction.
/// </summary>
/// <remarks>
/// The backend does not own <see cref="Device"/>. Caches it creates live on that device, so the model that runs on them must be
/// loaded on the same device (<c>VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir)</c>).
/// </remarks>
public sealed class VulkanChunkBackend : IChunkBackend, IDisposable
{
    private readonly ModelConfig _config;
    private readonly RopeShiftKvKernel _shift;
    private readonly RopeParameters _rope;
    private readonly RopeF32Kernel.Variant _variant;
    private bool _disposed;

    /// <summary>Creates the backend after checking that the model's keys can be re-rotated by a plain RoPE delta.</summary>
    /// <param name="device">The device caches are allocated on (not owned).</param>
    /// <param name="spvDir">Directory holding <c>rope_shift_kv_f32.spv</c>.</param>
    /// <param name="config">The model configuration.</param>
    /// <param name="features">Features read from the weights (<see cref="ChunkModelFeatures.FromGguf"/>).</param>
    /// <exception cref="NotSupportedException">The model is not plain RoPE (see <see cref="CheckModelSupported"/>).</exception>
    public VulkanChunkBackend(VulkanDevice device, string spvDir, ModelConfig config, ChunkModelFeatures features)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentException.ThrowIfNullOrEmpty(spvDir);
        RopeParameters rope = CheckModelSupported(config, features);
        Device = device;
        _config = config;
        _rope = rope;
        _variant = ToVariant(rope.Type);
        _shift = RopeShiftKvKernel.Create(device, spvDir);
    }

    /// <summary>The device caches are allocated on.</summary>
    public VulkanDevice Device { get; }

    /// <inheritdoc/>
    public string Tag => ChunkBackendTags.Vulkan;

    /// <summary>The model's checked plain-RoPE parameters; <see cref="ShiftKeys"/> rotates with exactly these.</summary>
    public RopeParameters Rope => _rope;

    /// <summary>
    /// Throws <see cref="NotSupportedException"/> naming the feature unless a plain RoPE delta rotation re-rotates the model's stored
    /// keys exactly: refuses everything <see cref="RopeParameters.FromConfig"/> refuses (YaRN, MLA, per-attention-type RoPE, scaled
    /// RoPE), Llama-3-style frequency factors and attention sinks (the per-query-position attention and the probe do not support sinks).
    /// </summary>
    /// <param name="config">The model configuration.</param>
    /// <param name="features">Features read from the weights.</param>
    /// <returns>The plain-RoPE parameters.</returns>
    public static RopeParameters CheckModelSupported(ModelConfig config, ChunkModelFeatures features)
    {
        ArgumentNullException.ThrowIfNull(config);
        RopeParameters rope = RopeParameters.FromConfig(config);
        if (features.HasRopeFreqFactors)
            throw new NotSupportedException(
                "RoPE frequency factors (GGUF rope_freqs.weight, Llama-3.x) are not supported by the Vulkan chunk backend: its device shift rotates with plain 1/theta^(2i/d) frequencies.");
        if (features.HasAttentionSinks)
            throw new NotSupportedException(
                "Attention sinks are not supported by the Vulkan chunk backend (the per-query-position attention and the relevance probe have no sink path).");
        return rope;
    }

    /// <summary>
    /// Throws <see cref="NotSupportedException"/> unless <paramref name="cache"/> has positioned row IO and stores exact F32 rows: a
    /// <see cref="VulkanKvCache"/> with F32 storage and no compact sliding windows.
    /// </summary>
    /// <param name="cache">The destination.</param>
    public static void CheckCacheSupported(IKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (cache is not IPositionedHostKvCache)
            throw new NotSupportedException(
                $"The Vulkan chunk backend needs a cache with {nameof(IPositionedHostKvCache)} positioned row IO; {cache.GetType().Name} has none.");
        if (cache is not VulkanKvCache vk)
            throw new NotSupportedException($"The Vulkan chunk backend shifts keys on the device and needs a {nameof(VulkanKvCache)}; got {cache.GetType().Name}.");
        if (vk.Storage != VulkanKvStorage.F32)
            throw new NotSupportedException(
                $"The Vulkan chunk backend writes exact F32 rows; a {vk.Storage} cache would round them (F16 KV is not supported for chunk reuse).");
        if (vk.HasCompactWindows)
            throw new NotSupportedException("The Vulkan chunk backend addresses rows by absolute position; a compact sliding-window cache is not supported.");
    }

    /// <inheritdoc/>
    public IKvCache CreateCache(int maxSeqLen)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new VulkanKvCache(Device, KvGeometry.FromConfig(_config, compactSlidingWindows: false), maxSeqLen, VulkanKvStorage.F32);
    }

    /// <inheritdoc/>
    public void RequireCache(IKvCache cache) => CheckCacheSupported(cache);

    /// <inheritdoc/>
    public void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
    {
        VulkanKvCache c = Checked(cache);
        int n = CpuChunkBackend.RowCount(k, v, c.KvStrideOf(layer));
        c.WriteRows(layer, startPos, n, k, v);
    }

    /// <inheritdoc/>
    public void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v)
        => Checked(cache).ReadRows(layer, startPos, count, k, v);

    /// <inheritdoc/>
    public void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters rope)
    {
        VulkanKvCache c = Checked(cache);
        ObjectDisposedException.ThrowIf(_disposed, this);
        // One source of truth: the parameters checked at construction. A caller passing anything else is a bug, not a request.
        if (rope != _rope)
            throw new ArgumentException($"RoPE parameters {rope} differ from the model's checked {_rope}.", nameof(rope));
        int headDim = _config.HeadDim;
        for (int l = 0; l < c.LayerCount; l++)
        {
            if (_config.IsNoRopeLayer(l)) continue;
            int stride = c.KvStrideOf(l);
            _shift.Shift(c, l, startPos, count, delta, stride / headDim, headDim, _rope.RopeDim, _rope.Theta, _variant);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _shift.Dispose();
    }

    // Same mapping as VulkanTransformerModel's own Q/K RoPE (NeoX -> NeoX, otherwise Norm).
    private static RopeF32Kernel.Variant ToVariant(RoPEType type) => type switch
    {
        RoPEType.NeoX => RopeF32Kernel.Variant.NeoX,
        RoPEType.Norm => RopeF32Kernel.Variant.Norm,
        _ => throw new NotSupportedException($"RoPE pairing {type} is not supported by the Vulkan chunk backend."),
    };

    private VulkanKvCache Checked(IKvCache cache)
    {
        CheckCacheSupported(cache);
        return (VulkanKvCache)cache;
    }
}
