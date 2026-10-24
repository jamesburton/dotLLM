using System.Runtime.InteropServices;

namespace DotLLM.Core.Models;

/// <summary>
/// Per-layer prompt key/value store for the DiffusionGemma masked-diffusion
/// <b>prompt-KV (PKV) prefill/decode</b> optimisation. During a canvas's PKV
/// <i>prefill</i> each transformer layer's post-norm/post-rope prompt <c>K</c> and
/// weight-less-normed prompt <c>V</c> (for V-less global layers <c>V</c> is the raw
/// <c>K</c> projection, exactly as the unified forward) are captured here once; the
/// per-step PKV <i>decode</i> then concatenates these cached prompt K/V ahead of the
/// freshly-computed canvas K/V instead of recomputing the whole prompt prefix.
/// </summary>
/// <remarks>
/// <para>The prompt embedding is FIXED across denoise steps (only the canvas embedding
/// changes, via self-conditioning), so the prompt's per-layer K/V are CONSTANT across a
/// canvas block and safe to cache. This is a pure throughput optimisation: PKV decode
/// produces byte-equivalent canvas logits to the cacheless unified <c>[prompt | canvas]</c>
/// forward.</para>
/// <para>Each layer's <c>K</c> and <c>V</c> are stored row-major
/// <c>[promptLen, nKvHead*headDim]</c> in F32 — the SAME memory layout the attention kernel
/// consumes for a key/value tensor — so decode can read them with a single contiguous copy
/// ahead of the canvas K/V. Per-layer head_dim / n_kv_head differ (Gemma 4 sliding vs global
/// layers), so each layer's buffer is sized independently from its own
/// <c>kvBlockElems = nKvHead*headDim</c>.</para>
/// <para>Buffers are unmanaged (64-byte aligned) and reused across canvas blocks: a block's
/// prefill overwrites the previous block's captured K/V. <see cref="Dispose"/> frees them.
/// Single-threaded per model instance, matching the model's forward-state contract.</para>
/// </remarks>
public sealed unsafe class DiffusionPromptKvStore : IDisposable
{
    private readonly int _numLayers;
    // Per-layer K/V buffers. Each holds [_capacityTokens × kvBlockElems[layer]] F32.
    private readonly nint[] _keys;
    private readonly nint[] _values;
    private readonly int[] _kvBlockElems;
    private int _capacityTokens;

    /// <summary>Number of prompt tokens currently captured (the prefill length). 0 until a prefill runs.</summary>
    public int PromptLen { get; private set; }

    /// <summary>Creates a store for <paramref name="numLayers"/> layers. Buffers allocate lazily on first capture.</summary>
    /// <param name="numLayers">Transformer layer count.</param>
    public DiffusionPromptKvStore(int numLayers)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
        _numLayers = numLayers;
        _keys = new nint[numLayers];
        _values = new nint[numLayers];
        _kvBlockElems = new int[numLayers];
        _capacityTokens = 0;
        PromptLen = 0;
    }

    /// <summary>
    /// Ensures every per-layer buffer can hold <paramref name="promptLen"/> tokens and records the
    /// per-layer KV block width, then sets <see cref="PromptLen"/>. Call once at the start of a
    /// prefill before capturing layers. Reuses existing allocations when they are large enough.
    /// </summary>
    /// <param name="promptLen">Number of prompt tokens this prefill will capture.</param>
    /// <param name="kvBlockElemsPerLayer">Per-layer <c>nKvHead*headDim</c> (length == layer count).</param>
    public void BeginPrefill(int promptLen, ReadOnlySpan<int> kvBlockElemsPerLayer)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(promptLen);
        if (kvBlockElemsPerLayer.Length != _numLayers)
            throw new ArgumentException(
                $"kvBlockElemsPerLayer length {kvBlockElemsPerLayer.Length} != layer count {_numLayers}.",
                nameof(kvBlockElemsPerLayer));

        bool grow = promptLen > _capacityTokens;
        for (int l = 0; l < _numLayers; l++)
        {
            int wantElems = kvBlockElemsPerLayer[l];
            if (grow || _keys[l] == 0 || _kvBlockElems[l] != wantElems)
            {
                FreeLayer(l);
                long n = (long)promptLen * wantElems;
                _keys[l] = (nint)NativeMemory.AlignedAlloc((nuint)(n * sizeof(float)), 64);
                _values[l] = (nint)NativeMemory.AlignedAlloc((nuint)(n * sizeof(float)), 64);
                _kvBlockElems[l] = wantElems;
            }
        }
        if (grow) _capacityTokens = promptLen;
        PromptLen = promptLen;
    }

    /// <summary>Per-layer KV block width (<c>nKvHead*headDim</c>) recorded at the last prefill.</summary>
    public int KvBlockElems(int layer) => _kvBlockElems[layer];

    /// <summary>Pointer to layer <paramref name="layer"/>'s prompt keys, row-major <c>[PromptLen × KvBlockElems]</c>.</summary>
    public float* Keys(int layer) => (float*)_keys[layer];

    /// <summary>Pointer to layer <paramref name="layer"/>'s prompt values, row-major <c>[PromptLen × KvBlockElems]</c>.</summary>
    public float* Values(int layer) => (float*)_values[layer];

    private void FreeLayer(int l)
    {
        if (_keys[l] != 0) { NativeMemory.AlignedFree((void*)_keys[l]); _keys[l] = 0; }
        if (_values[l] != 0) { NativeMemory.AlignedFree((void*)_values[l]); _values[l] = 0; }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int l = 0; l < _numLayers; l++)
            FreeLayer(l);
        _capacityTokens = 0;
        PromptLen = 0;
    }
}
