using System.Buffers;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// The <c>qwen35moe</c> MTP head (issue #1188) and the recurrent checkpoint pair speculative decoding needs.
/// </summary>
/// <remarks>
/// <para>
/// The head is the dense <c>qwen35</c> head (<see cref="Qwen3HybridDenseTransformerModel"/>, #253 / #469) with the trunk's MoE FFN in place of the
/// dense SwiGLU — llama.cpp <c>src/models/qwen35moe.cpp</c> <c>graph_mtp</c>: <c>eh_proj([enorm(embed(x_p)) ; hnorm(h_{p-1})])</c>, gated full
/// attention over the head's own position-indexed KV, <c>post_attention_norm</c>, routed top-k + sigmoid-gated shared expert, then
/// <c>nextn.shared_head_norm</c> (or the trunk's <c>output_norm</c>) and the trunk's LM head. Every trunk forward that carries the
/// <see cref="CpuMtpState"/> captures its post-final-norm rows and absorbs the batch into the head's KV (K/V only, so the experts run only
/// on draft steps).
/// </para>
/// <para>
/// This is the CPU oracle. It reuses <see cref="CpuMtpState"/> and therefore its F32 head KV — an existing store tracked for removal in #1107,
/// not a new one; the GPU heads keep their KV in the trunk type.
/// </para>
/// </remarks>
public sealed unsafe partial class Qwen3MoeHybridTransformerModel
{
    /// <summary>The MTP block's weights; the embedding and LM head default to the trunk's.</summary>
    private sealed class MoeMtpHead
    {
        public required float[] AttnNorm { get; init; }
        public required float[] PostAttnNorm { get; init; }
        public required Qwen3FullAttnWeights Attn { get; init; }
        public required MoeLayerWeights Moe { get; init; }
        public required nint EhProj { get; init; }
        public required QuantizationType EhProjQt { get; init; }
        public required int EhProjIn { get; init; }
        public required int EhProjOut { get; init; }
        public required float[] Enorm { get; init; }
        public required float[] Hnorm { get; init; }
        public float[]? SharedHeadNorm { get; init; }
    }

    private MoeMtpHead? _mtpHead;

    /// <inheritdoc/>
    public bool SupportsMtp => _mtpHead is not null;

    /// <summary>
    /// Loads <c>blk.{NumLayers}</c> as the MTP block when the GGUF declares <c>nextn_predict_layers == 1</c> and carries the tensors;
    /// null otherwise. Head-local <c>nextn.embed_tokens</c> / <c>nextn.shared_head_head</c> are not supported (no released qwen35moe file
    /// ships them) and are refused rather than ignored, since ignoring them would draft from the wrong matrices.
    /// </summary>
    private static MoeMtpHead? LoadMtpHeadIfPresent(GgufDataBase dataBase, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
                                                    ModelConfig config, List<nint> owned)
    {
        if (config.NextnPredictLayers <= 0)
            return null;
        if (config.NextnPredictLayers != 1)
            throw new NotSupportedException(
                $"qwen35moe MTP supports a single MTP block (nextn_predict_layers=1), got {config.NextnPredictLayers} — llama.cpp asserts the same.");
        int il = config.NumLayers;
        string p = $"blk.{il}";
        if (!tensors.ContainsKey($"{p}.nextn.eh_proj.weight"))
            return null;
        foreach (string local in new[] { "nextn.embed_tokens.weight", "nextn.shared_head_head.weight" })
        {
            if (tensors.ContainsKey($"{p}.{local}"))
                throw new NotSupportedException($"qwen35moe MTP head carries a head-local {local}, which this loader does not implement.");
        }

        int h = config.HiddenSize;
        var eh = tensors[$"{p}.nextn.eh_proj.weight"];
        if (eh.Shape[0] != 2 * h || eh.Shape[1] != h)
            throw new InvalidDataException($"{p}.nextn.eh_proj is [{eh.Shape[0]}, {eh.Shape[1]}]; expected [{2 * h}, {h}].");
        return new MoeMtpHead
        {
            AttnNorm = DequantizeF32(dataBase, tensors[$"{p}.attn_norm.weight"], h),
            PostAttnNorm = DequantizeF32(dataBase, tensors[$"{p}.post_attention_norm.weight"], h),
            Attn = LoadFullAttnLayer(p, dataBase, tensors, config, config.NumKvHeads),
            Moe = TransformerWeights.LoadDeepSeekMoeLayer(il, dataBase, tensors, config, owned, skipRoutedF32Only: true),
            EhProj = dataBase.Of(eh),
            EhProjQt = eh.QuantizationType,
            EhProjIn = eh.Shape[0],
            EhProjOut = eh.Shape[1],
            Enorm = DequantizeF32(dataBase, tensors[$"{p}.nextn.enorm.weight"], h),
            Hnorm = DequantizeF32(dataBase, tensors[$"{p}.nextn.hnorm.weight"], h),
            SharedHeadNorm = tensors.TryGetValue($"{p}.nextn.shared_head_norm.weight", out var shn) ? DequantizeF32(dataBase, shn, h) : null,
        };
    }

    // ── Recurrent checkpoint (issue #287 contract): a rejected draft must not leave its GDN contribution behind ──

    private GdnStateCache? _spareGdnCheckpoint;

    /// <inheritdoc/>
    public bool SupportsRecurrentStateCheckpoint => true;

    /// <inheritdoc/>
    /// <remarks>Pooled (one spare): the speculative decoder takes one per round. Disposing the checkpoint returns its buffers.</remarks>
    public object? CheckpointRecurrentState()
    {
        GdnStateCache snapshot = Interlocked.Exchange(ref _spareGdnCheckpoint, null) ?? new GdnStateCache(_gdn, _gdnCache.NumGdnLayers);
        _gdnCache.CopyTo(snapshot);
        return new PooledGdnCheckpoint(this, snapshot);
    }

    /// <inheritdoc/>
    public void RestoreRecurrentState(object? checkpoint)
    {
        GdnStateCache? snapshot = checkpoint switch
        {
            null => null,
            PooledGdnCheckpoint pooled => pooled.Snapshot,
            GdnStateCache raw => raw,
            _ => throw new ArgumentException(
                $"{GetType().Name}.RestoreRecurrentState expects a checkpoint from CheckpointRecurrentState; got {checkpoint.GetType().Name}.",
                nameof(checkpoint)),
        };
        snapshot?.CopyTo(_gdnCache);
    }

    private sealed class PooledGdnCheckpoint(Qwen3MoeHybridTransformerModel owner, GdnStateCache snapshot) : IDisposable
    {
        private GdnStateCache? _snapshot = snapshot;

        public GdnStateCache Snapshot => _snapshot ?? throw new ObjectDisposedException(nameof(PooledGdnCheckpoint));

        public void Dispose()
        {
            var s = Interlocked.Exchange(ref _snapshot, null);
            if (s is null) return;
            if (Interlocked.CompareExchange(ref owner._spareGdnCheckpoint, s, null) is not null)
                s.Dispose();
        }
    }

    // ── MTP state / draft surface ──

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => CreateMtpState(Config.MaxSequenceLength);

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength)
        => _mtpHead is null ? null : new CpuMtpState(Config.HiddenSize, _mtpHead.Attn.NumKvHeads, Config.HeadDim, maxSequenceLength);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           Core.Lora.ILoraAdapter? adapter, IMtpState? mtpState)
    {
        if (mtpState is null)
            return Forward(tokenIds, positions, deviceId, kvCache, adapter);
        if (adapter is not null)
            throw new NotSupportedException("MTP with a LoRA adapter is not supported on qwen35moe (the head would draft without the adapter).");
        return Forward(tokenIds, positions, deviceId, kvCache, gdnState: null, mtpState);
    }

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        var head = _mtpHead ?? throw new NotSupportedException($"{nameof(Qwen3MoeHybridTransformerModel)} has no MTP head (SupportsMtp=false).");
        if (state is not CpuMtpState mtp)
            throw new ArgumentException($"{nameof(Qwen3MoeHybridTransformerModel)} requires a CpuMtpState; got {state.GetType().Name}.", nameof(state));
        if ((uint)tokenId >= (uint)Config.VocabSize)
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        return DraftStep(head, mtp, tokenId, position);
    }

    /// <summary>Called by the trunk forward with its post-final-norm rows: capture them, then absorb the batch into the head KV.</summary>
    private void CaptureAndAbsorbMtp(CpuMtpState state, float* postNormHidden, ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions)
    {
        var head = _mtpHead!;
        int s = tokenIds.Length, h = Config.HiddenSize;
        state.SetCapturedRows(new ReadOnlySpan<float>(postNormHidden, s * h), s);
        for (int i = 1; i < s; i++)
        {
            if (positions[i] != positions[0] + i)
                throw new NotSupportedException("The qwen35moe MTP absorb needs contiguous positions (prefill and verify batches always are).");
        }

        int kvHeads = head.Attn.NumKvHeads, hd = Config.HeadDim, kvStride = kvHeads * hd;
        float eps = Config.NormEpsilon;
        state.BeginAbsorb(positions[0], s);
        var pool = ArrayPool<float>.Shared;
        float[] embedArr = pool.Rent(s * h), pairArr = pool.Rent(s * h), concatArr = pool.Rent(s * 2 * h), curArr = pool.Rent(s * h);
        try
        {
            var embed = embedArr.AsSpan(0, s * h);
            var pair = pairArr.AsSpan(0, s * h);
            var concat = concatArr.AsSpan(0, s * 2 * h);
            var cur = curArr.AsSpan(0, s * h);
            fixed (float* embedPtr = embed)
                EmbedTokens(tokenIds, embedPtr, h);
            state.CopyAbsorbPairingRows(s, pair);   // #469: token i pairs with h_{p-1}
            for (int i = 0; i < s; i++)
            {
                var row = concat.Slice(i * 2 * h, 2 * h);
                RmsNorm.Execute(embed.Slice(i * h, h), head.Enorm, eps, row.Slice(0, h));
                RmsNorm.Execute(pair.Slice(i * h, h), head.Hnorm, eps, row.Slice(h, h));
            }
            fixed (float* concatPtr = concat, curPtr = cur)
                Gemm(head.EhProj, head.EhProjQt, concatPtr, curPtr, head.EhProjOut, head.EhProjIn, s, preQuantizedInput: null);
            for (int i = 0; i < s; i++)
                RmsNorm.Execute(cur.Slice(i * h, h), head.AttnNorm, eps, cur.Slice(i * h, h));

            // K/V straight into the head's slab [positions[0], +s); the attention, O-projection and MoE of an absorbed cell are dead compute.
            float* kDst = state.KeyCachePtr + (long)positions[0] * kvStride;
            float* vDst = state.ValueCachePtr + (long)positions[0] * kvStride;
            fixed (float* curPtr = cur)
            {
                Gemm(head.Attn.KWeight, head.Attn.KQuantType, curPtr, kDst, head.Attn.KOutputDim, head.Attn.KInputDim, s, preQuantizedInput: null);
                Gemm(head.Attn.VWeight, head.Attn.VQuantType, curPtr, vDst, head.Attn.VOutputDim, head.Attn.VInputDim, s, preQuantizedInput: null);
            }
            var kSlab = new Span<float>(kDst, s * kvStride);
            Mamba3QkNorm.Execute(kSlab, head.Attn.KNormWeight, eps, s, kvHeads, hd);
            RoPE.Execute(Span<float>.Empty, kSlab, positions, 0, kvHeads, hd, _ropeDim, _ropeCosTable, _ropeSinTable, RoPEType.NeoX);
            state.EndAbsorb(positions[0] + s);
        }
        finally
        {
            pool.Return(embedArr);
            pool.Return(pairArr);
            pool.Return(concatArr);
            pool.Return(curArr);
        }
        state.SeedFromCapturedRow(s - 1);
    }

    /// <summary>One autoregressive head step at <paramref name="position"/>, paired with the state's pending hidden; returns [1, vocab] logits.</summary>
    private ITensor DraftStep(MoeMtpHead head, CpuMtpState state, int tokenId, int position)
    {
        int h = Config.HiddenSize, vocab = Config.VocabSize;
        int heads = Config.NumAttentionHeads, kvHeads = head.Attn.NumKvHeads, hd = Config.HeadDim;
        int qElems = heads * hd, kvStride = kvHeads * hd;
        float eps = Config.NormEpsilon;

        // Slot p holds the pair (h_{p-1}, x_p): writing at p discards anything speculative beyond it; a gap is a protocol error.
        if (state.CurrentLength > position)
            state.Rollback(position);
        else if (state.CurrentLength < position)
            throw new InvalidOperationException(
                $"MTP step at position {position} but the head KV covers {state.CurrentLength} positions: every trunk Forward of the sequence must pass the MTP state.");
        if (position >= state.MaxSteps || position >= Config.MaxSequenceLength)
            throw new InvalidOperationException($"MTP head KV exhausted at position {position} (MaxSteps={state.MaxSteps}).");
        state.EnsureCapacity(position + 1);
        _state.EnsureCapacity(1);

        var pool = ArrayPool<float>.Shared;
        float[] embArr = pool.Rent(h), concatArr = pool.Rent(2 * h), curArr = pool.Rent(h), resArr = pool.Rent(h), normArr = pool.Rent(h);
        float[] qgArr = pool.Rent(2 * qElems), qArr = pool.Rent(qElems), gateArr = pool.Rent(qElems), attnArr = pool.Rent(qElems);
        float[] logitsArr = pool.Rent(vocab);
        try
        {
            var emb = embArr.AsSpan(0, h);
            var concat = concatArr.AsSpan(0, 2 * h);
            var cur = curArr.AsSpan(0, h);
            var res = resArr.AsSpan(0, h);
            var norm = normArr.AsSpan(0, h);
            var qg = qgArr.AsSpan(0, 2 * qElems);
            var q = qArr.AsSpan(0, qElems);
            var gate = gateArr.AsSpan(0, qElems);
            var attnOut = attnArr.AsSpan(0, qElems);
            var k = state.GetKeyRow(position);
            var v = state.GetValueRow(position);

            fixed (float* embPtr = emb)
                EmbedTokens([tokenId], embPtr, h);
            RmsNorm.Execute(emb, head.Enorm, eps, concat.Slice(0, h));
            RmsNorm.Execute(state.PendingHidden, head.Hnorm, eps, concat.Slice(h, h));
            fixed (float* concatPtr = concat, curPtr = cur)
                Gemm(head.EhProj, head.EhProjQt, concatPtr, curPtr, head.EhProjOut, head.EhProjIn, 1, preQuantizedInput: null);
            cur.CopyTo(res);   // inpSA is the eh_proj output

            // Gated full attention over the head's own KV (ForwardFullAttnBody's math, seqQ = 1).
            RmsNorm.Execute(cur, head.AttnNorm, eps, norm);
            fixed (float* normPtr = norm, qgPtr = qg, kPtr = k, vPtr = v)
            {
                Gemm(head.Attn.QWeight, head.Attn.QQuantType, normPtr, qgPtr, head.Attn.QOutputDim, head.Attn.QInputDim, 1, preQuantizedInput: null);
                Gemm(head.Attn.KWeight, head.Attn.KQuantType, normPtr, kPtr, head.Attn.KOutputDim, head.Attn.KInputDim, 1, preQuantizedInput: null);
                Gemm(head.Attn.VWeight, head.Attn.VQuantType, normPtr, vPtr, head.Attn.VOutputDim, head.Attn.VInputDim, 1, preQuantizedInput: null);
            }
            for (int i = 0; i < heads; i++)
            {
                qg.Slice(i * 2 * hd, hd).CopyTo(q.Slice(i * hd));
                qg.Slice(i * 2 * hd + hd, hd).CopyTo(gate.Slice(i * hd));
            }
            Mamba3QkNorm.Execute(q, head.Attn.QNormWeight, eps, 1, heads, hd);
            Mamba3QkNorm.Execute(k, head.Attn.KNormWeight, eps, 1, kvHeads, hd);
            RoPE.Execute(q, k, [position], heads, kvHeads, hd, _ropeDim, _ropeCosTable, _ropeSinTable, RoPEType.NeoX);
            int seqKv = position + 1;
            Attention.Execute(q, new ReadOnlySpan<float>(state.KeyCachePtr, seqKv * kvStride),
                new ReadOnlySpan<float>(state.ValueCachePtr, seqKv * kvStride), attnOut,
                1, seqKv, heads, kvHeads, hd, position, (int?)null);
            for (int i = 0; i < qElems; i++)
                attnOut[i] *= 1f / (1f + MathF.Exp(-gate[i]));
            fixed (float* aPtr = attnOut, curPtr = cur)
                Gemm(head.Attn.OWeight, head.Attn.OQuantType, aPtr, curPtr, head.Attn.OOutputDim, head.Attn.OInputDim, 1, preQuantizedInput: null);
            Add.Execute(res, cur, cur);

            // MoE FFN (routed + gated shared expert), residual-added.
            cur.CopyTo(res);
            RmsNorm.Execute(cur, head.PostAttnNorm, eps, norm);
            fixed (float* normPtr = norm)
                ForwardMoeBody(head.Moe, Config.NumLayers, 1, h, normPtr);
            Add.Execute(res, norm, cur);
            state.Advance();

            // Shared head norm, then the trunk LM head. The next step chains the POST-norm hidden (llama.cpp's h_nextn, #469).
            RmsNorm.Execute(cur, head.SharedHeadNorm ?? _outputNormWeight, eps, norm);
            norm.CopyTo(state.PendingHiddenMutable);
            fixed (float* normPtr = norm, logitsPtr = logitsArr)
                Gemm(_outputWeight, _outputQuantType, normPtr, logitsPtr, _outputOutputDim, _outputInputDim, 1, preQuantizedInput: null);

            var result = UnmanagedTensor.Allocate(new TensorShape(1, vocab), DType.Float32, deviceId: -1);
            logitsArr.AsSpan(0, vocab).CopyTo(new Span<float>((void*)result.DataPointer, vocab));
            return result;
        }
        finally
        {
            pool.Return(embArr);
            pool.Return(concatArr);
            pool.Return(curArr);
            pool.Return(resArr);
            pool.Return(normArr);
            pool.Return(qgArr);
            pool.Return(qArr);
            pool.Return(gateArr);
            pool.Return(attnArr);
            pool.Return(logitsArr);
        }
    }
}
