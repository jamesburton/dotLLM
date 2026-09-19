using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Sparse KV cache for the Vulkan NemotronH path. Mirrors
/// <see cref="VulkanKvCache"/> but only allocates per-layer K/V buffers for layers whose
/// <c>HybridLayerKind</c> is <c>Attention</c> — SSM and FFN layers don't need a KV cache.
/// </summary>
/// <remarks>
/// <para>The internal physical layout is a dense array indexed by KV-slot in
/// <c>[0, attentionLayerCount)</c>. <c>kvSlotForLayer[absoluteLayerIndex]</c> maps
/// model-level layer indices into slot indices; a value of <c>-1</c> indicates a
/// non-attention layer and the cache rejects requests against that index.</para>
///
/// <para><b>Lazy growth (issue #437).</b> The per-layer buffers are sized to
/// <see cref="Capacity"/> rows, not <see cref="MaxLength"/>. Capacity starts at
/// <see cref="DefaultInitialCapacity"/> (or <c>DOTLLM_VK_KV_INITIAL_CAPACITY</c>) and doubles
/// via <see cref="EnsureCapacity"/>, which the owning model MUST call before it opens its
/// command buffer — growth reallocates every per-layer buffer, so it cannot happen while
/// commands referencing the old handles are being recorded. <see cref="RecordUpdate"/>
/// throws rather than silently growing mid-recording.</para>
///
/// <para>Why the caller has to react: freed Vulkan buffer handles are recycled by the driver
/// and the kernels' <c>DescriptorSetCache</c> is keyed on the handle, so a grown cache whose
/// descriptor sets were not invalidated produces output that is correct and then silently
/// zeros. <see cref="EnsureCapacity"/> returns <c>true</c> exactly when the caller must
/// invalidate its kernel descriptor caches — the same contract the forward-state scratch
/// buffers already use (<c>_state.EnsureCapacity</c> → <c>InvalidateKernelCaches</c>).</para>
/// </remarks>
public sealed class VulkanNemotronHKvCache : IKvCache
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _keys;
    private readonly VulkanDevice.Buffer[] _values;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _kvStride;
    private readonly int _rowBytes;
    private int _capacity;
    private int _currentLength;
    private bool _disposed;

    /// <summary>
    /// Rows allocated on first construction when growth is enabled. Chosen so a short prompt
    /// never pays for the declared context: Bonsai 2 (<c>head_count_kv=4</c>,
    /// <c>key_length=256</c>, 16 attention layers) costs 128 KiB per token, so 512 rows is
    /// 64 MiB against the 4 GiB an eager ctx-32k allocation would take.
    /// </summary>
    public const int DefaultInitialCapacity = 512;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of attention layers (KV slots).</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>
    /// Rows currently allocated per layer. Always <c>&lt;=</c> <see cref="MaxLength"/>; grown by
    /// <see cref="EnsureCapacity"/>. <see cref="MaxLength"/> remains the logical bound that
    /// position validation uses, so growth is invisible to callers that only append.
    /// </summary>
    public int Capacity => _capacity;

    /// <summary>Device-local bytes currently held by this cache (K + V across all slots).</summary>
    public long AllocatedBytes => (long)_attentionLayerCount * 2 * _capacity * _rowBytes;

    /// <summary>
    /// Creates the per-attention-layer K/V buffers. <paramref name="kvSlotForLayer"/> is the
    /// physical-layer-index -> slot mapping (<c>-1</c> for non-attention layers); the
    /// <c>attentionLayerCount</c> is derived as the maximum slot + 1. Buffers start at
    /// <see cref="DefaultInitialCapacity"/> rows and grow on demand — see the type remarks.
    /// </summary>
    public VulkanNemotronHKvCache(VulkanDevice device, int[] kvSlotForLayer, int attentionLayerCount,
        int numKvHeads, int headDim, int maxSeqLen)
        : this(device, kvSlotForLayer, attentionLayerCount, numKvHeads, headDim, maxSeqLen,
               initialCapacity: ResolveInitialCapacity(maxSeqLen))
    {
    }

    /// <summary>
    /// Creates the cache with an explicit initial row capacity. Passing
    /// <paramref name="initialCapacity"/> equal to <paramref name="maxSeqLen"/> reproduces the
    /// pre-#437 eager allocation exactly (no growth can then occur). Tests pass a small value to
    /// force growth deterministically.
    /// </summary>
    public VulkanNemotronHKvCache(VulkanDevice device, int[] kvSlotForLayer, int attentionLayerCount,
        int numKvHeads, int headDim, int maxSeqLen, int initialCapacity)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(kvSlotForLayer);
        if (attentionLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(attentionLayerCount));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (initialCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(initialCapacity));
        if (initialCapacity > maxSeqLen) initialCapacity = maxSeqLen;

        _device = device;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _kvStride = numKvHeads * headDim;
        _rowBytes = _kvStride * sizeof(float);
        _capacity = initialCapacity;

        _keys = new VulkanDevice.Buffer[attentionLayerCount];
        _values = new VulkanDevice.Buffer[attentionLayerCount];

        if (attentionLayerCount == 0) return;

        long bytesPerLayer = (long)_capacity * _rowBytes;
        for (int i = 0; i < attentionLayerCount; i++)
        {
            _keys[i] = device.AllocateDeviceLocal(bytesPerLayer);
            _values[i] = device.AllocateDeviceLocal(bytesPerLayer);
        }
    }

    /// <summary>
    /// Initial capacity for the growth-enabled constructor. <c>DOTLLM_VK_KV_INITIAL_CAPACITY</c>
    /// overrides <see cref="DefaultInitialCapacity"/>; the value <c>0</c> restores the pre-#437
    /// eager <c>maxSeqLen</c> allocation, which is how the growth path is A/B'd out.
    /// </summary>
    private static int ResolveInitialCapacity(int maxSeqLen)
    {
        string? raw = Environment.GetEnvironmentVariable("DOTLLM_VK_KV_INITIAL_CAPACITY");
        if (raw is not null && int.TryParse(raw, out int parsed) && parsed >= 0)
            return parsed == 0 ? maxSeqLen : Math.Min(parsed, maxSeqLen);
        return Math.Min(DefaultInitialCapacity, maxSeqLen);
    }

    /// <summary>
    /// Ensures at least <paramref name="requiredLength"/> rows are allocated per layer,
    /// reallocating (and copying the live <see cref="CurrentLength"/> rows) if not.
    /// Returns <c>true</c> when buffers were reallocated, in which case the caller MUST
    /// invalidate its kernel descriptor caches before recording — see the type remarks.
    /// </summary>
    /// <remarks>
    /// Must be called BEFORE the caller opens its command buffer. The copy runs on its own
    /// one-shot submit (<see cref="VulkanDevice.CreateSubmitContext"/>) which waits on a fence
    /// before the old buffers are destroyed; because <see cref="VulkanDevice"/> submits
    /// synchronously, no earlier submission can still be reading them. What it cannot protect
    /// against is an <i>un-submitted</i> command buffer that already references the old handles —
    /// hence the ordering requirement and the throw in <see cref="RecordUpdate"/>.
    /// </remarks>
    public bool EnsureCapacity(int requiredLength)
    {
        if (requiredLength <= _capacity) return false;
        if (requiredLength > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(requiredLength),
                $"Required length {requiredLength} exceeds max cache length {_maxSeqLen}.");
        if (_attentionLayerCount == 0) { _capacity = requiredLength; return false; }

        // Doubling keeps the number of reallocations logarithmic in the context length.
        int newCapacity = _capacity;
        while (newCapacity < requiredLength) newCapacity = Math.Min(_maxSeqLen, newCapacity * 2);

        long bytesPerLayer = (long)newCapacity * _rowBytes;
        var newKeys = new VulkanDevice.Buffer[_attentionLayerCount];
        var newValues = new VulkanDevice.Buffer[_attentionLayerCount];
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            newKeys[i] = _device.AllocateDeviceLocal(bytesPerLayer);
            newValues[i] = _device.AllocateDeviceLocal(bytesPerLayer);
        }

        long liveBytes = (long)_currentLength * _rowBytes;
        if (liveBytes > 0)
        {
            using var ctx = _device.CreateSubmitContext();
            ctx.Begin();
            var region = new VkBufferCopy { srcOffset = 0, dstOffset = 0, size = (ulong)liveBytes };
            for (int i = 0; i < _attentionLayerCount; i++)
            {
                VulkanApi.vkCmdCopyBuffer(ctx.CommandBuffer, _keys[i].Handle, newKeys[i].Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(ctx.CommandBuffer, _values[i].Handle, newValues[i].Handle, 1, region);
            }
            ctx.SubmitAndWait();
        }

        for (int i = 0; i < _attentionLayerCount; i++)
        {
            _keys[i].Dispose();
            _values[i].Dispose();
            _keys[i] = newKeys[i];
            _values[i] = newValues[i];
        }
        _capacity = newCapacity;
        return true;
    }

    /// <summary>
    /// Largest <c>position + 1</c> in <paramref name="positions"/> — the length a cache must have
    /// capacity for to accept this step. Callers feed this to <see cref="EnsureCapacity"/>.
    /// </summary>
    public static int RequiredLengthFor(ReadOnlySpan<int> positions)
    {
        int max = -1;
        for (int i = 0; i < positions.Length; i++)
            if (positions[i] > max) max = positions[i];
        return max + 1;
    }

    /// <summary>Raw Vulkan buffer handle of a layer's key store. Test/diagnostic use: the
    /// handle-keyed descriptor caches make handle identity across growths observable.</summary>
    public nint KeysHandleForTest(int absoluteLayerIndex) => _keys[RequireSlot(absoluteLayerIndex)].Handle;

    /// <summary>Raw Vulkan buffer handle of a layer's value store. See <see cref="KeysHandleForTest"/>.</summary>
    public nint ValuesHandleForTest(int absoluteLayerIndex) => _values[RequireSlot(absoluteLayerIndex)].Handle;

    /// <summary>Downloads the live key rows of a layer to host. Test helper.</summary>
    public void DownloadKeysForTest(int absoluteLayerIndex, Span<float> destination)
        => _device.Download(_keys[RequireSlot(absoluteLayerIndex)], destination);

    /// <summary>Downloads the live value rows of a layer to host. Test helper.</summary>
    public void DownloadValuesForTest(int absoluteLayerIndex, Span<float> destination)
        => _device.Download(_values[RequireSlot(absoluteLayerIndex)], destination);

    private int RequireSlot(int absoluteLayerIndex)
    {
        if ((uint)absoluteLayerIndex >= (uint)_kvSlotForLayer.Length)
            throw new ArgumentOutOfRangeException(nameof(absoluteLayerIndex));
        int slot = _kvSlotForLayer[absoluteLayerIndex];
        if (slot < 0)
            throw new InvalidOperationException(
                $"Layer {absoluteLayerIndex} is not an attention layer; no KV-cache slot exists.");
        return slot;
    }

    /// <summary>Returns the device buffer holding cached keys for the given (absolute) layer index.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer(int absoluteLayerIndex) => _keys[RequireSlot(absoluteLayerIndex)];

    /// <summary>Returns the device buffer holding cached values for the given (absolute) layer index.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer(int absoluteLayerIndex) => _values[RequireSlot(absoluteLayerIndex)];

    /// <summary>
    /// Appends K/V copy commands onto the supplied <paramref name="cmdBuf"/>. Caller is
    /// responsible for the <c>TRANSFER → COMPUTE_SHADER</c> barrier before the next attention
    /// dispatch reads from the cache, and for having called <see cref="EnsureCapacity"/> before
    /// the command buffer was opened.
    /// </summary>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int absoluteLayerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int slot = RequireSlot(absoluteLayerIndex);
        int rowBytes = _rowBytes;
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        if (maxPos >= _capacity)
            throw new InvalidOperationException(
                $"KV cache capacity is {_capacity} rows but position {maxPos} was requested. " +
                $"Call {nameof(EnsureCapacity)}({maxPos + 1}) BEFORE opening the command buffer — " +
                "growing mid-recording would free buffer handles the recording still references.");
        bool contiguous = IsContiguousAscending(positions);

        if (contiguous)
        {
            int startPos = positions[0];
            var region = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = (ulong)((long)startPos * rowBytes),
                size = (ulong)rowBytes * (ulong)seqLen,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[slot].Handle, 1, region);
            VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[slot].Handle, 1, region);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                var region = new VkBufferCopy
                {
                    srcOffset = (ulong)((long)i * rowBytes),
                    dstOffset = (ulong)((long)pos * rowBytes),
                    size = (ulong)rowBytes,
                };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[slot].Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[slot].Handle, 1, region);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    private int ValidateAndFindMaxPos(ReadOnlySpan<int> positions, int seqLen)
    {
        int maxPos = -1;
        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        return maxPos;
    }

    private static bool IsContiguousAscending(ReadOnlySpan<int> positions)
    {
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] != positions[i - 1] + 1)
                return false;
        }
        return true;
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanNemotronHKvCache is updated via RecordUpdate from the Vulkan forward pass.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanNemotronHKvCache is updated via RecordUpdate from the Vulkan forward pass.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache exposes device buffers, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache exposes device buffers, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanNemotronHKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => _currentLength = 0;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
        }
    }
}
