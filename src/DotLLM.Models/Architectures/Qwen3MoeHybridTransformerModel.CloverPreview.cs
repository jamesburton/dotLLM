using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Architectures;

/// <summary>Clover's bidirectional block preview on the CPU host (#1815); see <see cref="ICloverBlockPreview"/>.</summary>
/// <remarks>
/// A separate entry point, not a flag through <c>ForwardCore</c>: the causal forward is untouched. Per layer it is the causal trunk with
/// two substitutions: GDN layers run <see cref="CloverPreviewGdnMixer"/> against the committed (read-only) recurrent state, and attention
/// layers attend the whole prefix plus the whole block (<see cref="AttentionMaskMode.Bidirectional"/> over <c>start + T</c> keys, the
/// block's K/V parked in the cache's rows at and above <c>start</c> and rolled back afterwards). Embedding, norms, MoE routing and the LM
/// head are the causal path's.
/// </remarks>
public sealed unsafe partial class Qwen3MoeHybridTransformerModel : ICloverBlockPreview
{
    /// <inheritdoc/>
    public bool SupportsBlockPreview
    {
        get
        {
            if (Config.Clover is null) return false;
            foreach (var lw in _layers)
                if (lw.Gdn is { Preview: null }) return false;
            return true;
        }
    }

    /// <inheritdoc/>
    public ITensor ForwardBlockPreview(ReadOnlySpan<int> blockTokenIds, int startPosition, IKvCache? kvCache, IGdnState? gdnState, int deviceId)
    {
        var clover = Config.Clover ?? throw new NotSupportedException("The block preview is a Clover (clover_moe) forward; this model is not Clover.");
        if (!SupportsBlockPreview)
            throw new NotSupportedException("This Clover file carries no bidirectional-preview tensors (llama.cpp's converter drops them); "
                                            + "use the official DiffuSpace GGUF.");
        int T = blockTokenIds.Length;
        if (T == 0 || T > Math.Min(clover.BlockSize, CloverPreviewGdnMixer.MaxBlock))
            throw new ArgumentException($"Preview block length {T} must be 1..{clover.BlockSize}.", nameof(blockTokenIds));
        ArgumentOutOfRangeException.ThrowIfNegative(startPosition);
        if (kvCache is null && startPosition != 0)
            throw new ArgumentException("A preview after a non-empty prefix needs the prefix's KV cache.", nameof(kvCache));
        if (kvCache is not null && kvCache.CurrentLength != startPosition)
            throw new ArgumentException($"startPosition {startPosition} must equal the committed KV length {kvCache.CurrentLength}.", nameof(startPosition));
        if (startPosition + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(startPosition), "The block runs past the model's maximum sequence length.");
        if (kvCache is not null && !kvCache.TryGrowTo(startPosition + T))
            throw new InvalidOperationException($"The KV cache cannot hold {startPosition + T} positions.");

        GdnStateCache gdnCache = gdnState switch
        {
            null => _gdnCache,
            GdnStateCache typed when typed.NumGdnLayers == _gdnCache.NumGdnLayers => typed,
            GdnStateCache typed => throw new ArgumentException(
                $"GdnState NumGdnLayers ({typed.NumGdnLayers}) does not match model GDN-layer count ({_gdnCache.NumGdnLayers}).", nameof(gdnState)),
            _ => throw new ArgumentException($"Qwen3MoeHybridTransformerModel requires a CPU GdnStateCache; got {gdnState.GetType().Name}.", nameof(gdnState)),
        };
        // Same new-sequence rule as the causal forward (#1786): nothing is committed at position 0, so a stale model-owned state is dropped.
        if (kvCache is null && gdnState is null && startPosition == 0)
            ResetSequenceState();

        Span<int> positions = T <= 64 ? stackalloc int[T] : new int[T];
        for (int i = 0; i < T; i++) positions[i] = startPosition + i;

        int hiddenSize = Config.HiddenSize;
        float eps = Config.NormEpsilon;
        _state.EnsureCapacity(T);
        _threadPool?.SetDispatchMode(DispatchMode.EventBased);
        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;

        float coresetTopP = clover.PreviewCoresetTopP;
        EmbedTokens(blockTokenIds, hidden, hiddenSize);
        try
        {
            var kinds = _layout.LayerKind;
            for (int layer = 0; layer < _layers.Length; layer++)
            {
                var lw = _layers[layer];
                new Span<float>(hidden, T * hiddenSize).CopyTo(new Span<float>(residual, T * hiddenSize));
                for (int t = 0; t < T; t++)
                    RmsNorm.Execute(new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize), lw.AttnNormWeight, eps,
                                    new Span<float>(normOut + t * hiddenSize, hiddenSize));

                if (kinds[layer] == HybridLayerKind.GatedDeltaNet)
                    ForwardGdnPreviewBody(lw.Gdn!, layer, T, hiddenSize, normOut, eps, gdnCache);
                else
                    ForwardFullAttnBody(lw.FullAttn!, layer, T, positions, normOut,
                        (float*)_state.QScratch, (float*)_state.KScratch, (float*)_state.VScratch, (float*)_state.AttnOutput,
                        Config.NumAttentionHeads, Config.NumKvHeads, Config.HeadDim, kvCache, bidirectionalBlock: true);

                for (int t = 0; t < T; t++)
                    Add.Execute(new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize), new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                                new Span<float>(hidden + t * hiddenSize, hiddenSize));
                new Span<float>(hidden, T * hiddenSize).CopyTo(new Span<float>(residual, T * hiddenSize));
                for (int t = 0; t < T; t++)
                    RmsNorm.Execute(new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize), lw.PostAttnNormWeight, eps,
                                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
                ForwardMoeBody(lw.Moe, layer, T, hiddenSize, normOut, coresetTopP);
                for (int t = 0; t < T; t++)
                    Add.Execute(new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize), new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                                new Span<float>(hidden + t * hiddenSize, hiddenSize));
                if (ResidualModifier is { } modifier)
                    ApplyResidualModifier(modifier, layer, hidden, T);
            }
        }
        finally
        {
            // The block's K/V were parked at [start, start + T): drop them so the committed length is unchanged.
            if (kvCache is not null && kvCache.CurrentLength != startPosition)
                kvCache.Rollback(startPosition);
        }

        var logits = RunOutputHead(T, deviceId)!;
        MaskOutMaskToken(logits, T, clover.MaskTokenId, Config.VocabSize);
        return logits;
    }

    /// <summary>Sets the <c>&lt;|mask|&gt;</c> column of every row to <c>-inf</c> (HF <c>generate</c>: the mask token is never predicted).</summary>
    internal static void MaskOutMaskToken(ITensor logits, int rows, int maskTokenId, int vocab)
    {
        if ((uint)maskTokenId >= (uint)vocab) return;
        float* p = (float*)logits.DataPointer;
        for (int r = 0; r < rows; r++)
            p[(long)r * vocab + maskTokenId] = float.NegativeInfinity;
    }

    private void ForwardGdnPreviewBody(GdnTokenMixingWeights gdnW, int absoluteLayerIdx, int seqLen, int hiddenSize, float* normOut, float eps,
                                       GdnStateCache gdnCache)
    {
        int nV = _gdn.NVHead, nK = _gdn.NKHead, dS = _gdn.DState, dC = _gdn.DConv;
        int convDim = (2 * nK + nV) * dS, vDim = nV * dS, kDim = nK * dS, T = seqLen;
        var scratch = new GdnMixerScratch(
            new Span<float>((float*)_state.GdnQkvBuf, T * convDim), new Span<float>((float*)_state.GdnZBuf, T * vDim),
            new Span<float>((float*)_state.GdnAlphaBuf, T * nV), new Span<float>((float*)_state.GdnBetaBuf, T * nV),
            new Span<float>((float*)_state.GdnConvInput, (dC - 1 + T) * convDim), new Span<float>((float*)_state.GdnQBuf, T * kDim),
            new Span<float>((float*)_state.GdnKBuf, T * kDim), new Span<float>((float*)_state.GdnVBuf, T * vDim),
            new Span<float>((float*)_state.GdnOut, T * vDim));
        CloverPreviewGdnMixer.Forward(gdnW, _gdn, _gdnLayerOrdinal[absoluteLayerIdx], T, eps,
            new ReadOnlySpan<float>(normOut, T * hiddenSize), new Span<float>(normOut, T * hiddenSize),
            _gdnGemm ??= GdnGemmAdapter, scratch, gdnCache, _threadPool);
    }
}
