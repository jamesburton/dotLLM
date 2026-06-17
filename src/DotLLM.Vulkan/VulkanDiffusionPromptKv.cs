namespace DotLLM.Vulkan;

/// <summary>
/// Device-resident per-layer prompt key/value store for the DiffusionGemma masked-diffusion
/// <b>prompt-KV (PKV) prefill/decode</b> optimisation on the Vulkan backend — the device
/// analogue of <see cref="DotLLM.Core.Models.DiffusionPromptKvStore"/>.
/// </summary>
/// <remarks>
/// <para>During a canvas's PKV <i>prefill</i> each transformer layer's final K (post per-head
/// norm + RoPE) and weight-less-normed V are captured into device-local buffers; the per-step
/// PKV <i>decode</i> then prepends these cached prompt K/V ahead of the freshly-computed canvas
/// K/V (a single device copy) instead of recomputing the whole prompt prefix.</para>
/// <para>Each layer's K/V are stored row-major <c>[promptLen, nKvHead*headDim]</c> in F32 — the
/// exact layout the attention kernel consumes. Per-layer KV widths differ (Gemma 4 sliding vs
/// global layers), so each layer's buffer is sized independently. Buffers grow monotonically
/// across canvas blocks; a block's prefill overwrites the previous block's captured K/V. Owned
/// by the model (single in-flight generation), freed on model disposal.</para>
/// </remarks>
internal sealed class VulkanDiffusionPromptKv : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _numLayers;
    private readonly VulkanDevice.Buffer?[] _keys;
    private readonly VulkanDevice.Buffer?[] _values;
    private readonly int[] _kvBlockElems;
    private int _capacityTokens;

    /// <summary>Number of prompt tokens currently captured (the prefill length). 0 until a prefill runs.</summary>
    public int PromptLen { get; private set; }

    public VulkanDiffusionPromptKv(VulkanDevice device, int numLayers)
    {
        _device = device;
        _numLayers = numLayers;
        _keys = new VulkanDevice.Buffer?[numLayers];
        _values = new VulkanDevice.Buffer?[numLayers];
        _kvBlockElems = new int[numLayers];
    }

    /// <summary>True when any per-layer buffer was (re)allocated by the most recent <see cref="BeginPrefill"/>.</summary>
    public bool LastBeginReallocated { get; private set; }

    /// <summary>
    /// Ensures every per-layer buffer can hold <paramref name="promptLen"/> tokens at the given
    /// per-layer KV widths, then records <see cref="PromptLen"/>. Reuses existing allocations when
    /// large enough. Sets <see cref="LastBeginReallocated"/> when any buffer grew/changed.
    /// </summary>
    public void BeginPrefill(int promptLen, ReadOnlySpan<int> kvBlockElemsPerLayer)
    {
        LastBeginReallocated = false;
        bool grow = promptLen > _capacityTokens;
        for (int l = 0; l < _numLayers; l++)
        {
            int want = kvBlockElemsPerLayer[l];
            if (grow || _keys[l] is null || _kvBlockElems[l] != want)
            {
                _keys[l]?.Dispose();
                _values[l]?.Dispose();
                long bytes = (long)promptLen * want * sizeof(float);
                _keys[l] = _device.AllocateDeviceLocal(bytes);
                _values[l] = _device.AllocateDeviceLocal(bytes);
                _kvBlockElems[l] = want;
                LastBeginReallocated = true;
            }
        }
        if (grow) _capacityTokens = promptLen;
        PromptLen = promptLen;
    }

    public int KvBlockElems(int layer) => _kvBlockElems[layer];
    public VulkanDevice.Buffer Keys(int layer) => _keys[layer]!;
    public VulkanDevice.Buffer Values(int layer) => _values[layer]!;

    public void Dispose()
    {
        for (int l = 0; l < _numLayers; l++)
        {
            _keys[l]?.Dispose();
            _values[l]?.Dispose();
        }
    }
}
