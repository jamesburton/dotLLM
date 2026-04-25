using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU forward pass for one MLA (Multi-head Latent Attention) layer — Phase A
/// naive form. Runs the equivalent of <c>DotLLM.Cpu.Kernels.MlaAttention.Execute</c>
/// in F32 on CUDA: pre-attention RMSNorm → Q path (LoRA-factored or monolithic)
/// → KV path (LoRA + MQA-shared rope-K) → decoupled RoPE on rope sub-dim only →
/// Phase A expanded-cache write → causal-masked softmax over per-head scores
/// (Q_nope·K_nope_per_head + Q_pe·K_pe_shared) → weighted sum over per-head V →
/// o_proj.
/// </summary>
/// <remarks>
/// <para>
/// <b>F32 throughout.</b> Inputs / outputs / cache / weights are all F32.
/// Phase 1 keeps the entire MLA pipeline in F32 to match the CPU oracle
/// byte-for-byte algorithmically (numerical drift comes only from the order
/// of GPU floating-point reductions). Quantized / FP16 paths are an explicit
/// follow-up.
/// </para>
/// <para>
/// <b>Caller scratch contract.</b> The forward needs scratch buffers for
/// intermediate tensors (Q, compressed-KV, KV-B expansion, attention output).
/// The caller passes these via <see cref="CudaMlaScratch"/> — sized once for
/// the maximum <c>seqLen</c> the model handles, reused across layers and
/// across calls.
/// </para>
/// <para>
/// <b>Cache lifecycle.</b> The kernel writes the new <c>seqLen</c> rows of
/// K_nope / V / K_pe into the layer's slot in <see cref="CudaMlaKvCache"/>
/// at offset <c>cachedLength</c>, then attends over all
/// <c>cachedLength + seqLen</c> cached positions. The caller is responsible
/// for advancing <see cref="CudaMlaKvCache.Advance"/> after a successful
/// forward.
/// </para>
/// </remarks>
public static unsafe partial class CudaMlaAttention
{
    /// <summary>
    /// Runs one MLA layer's forward pass on the GPU.
    /// </summary>
    /// <param name="hiddenF32">Device pointer to F32 hidden state input <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="outputF32">Device pointer to F32 output <c>[seqLen, hiddenSize]</c>. Receives the post-o_proj attention output (does NOT include the residual add — caller is expected to add hidden+output as a separate step).</param>
    /// <param name="seqLen">Number of tokens this call processes (prefill = prompt length, decode = 1).</param>
    /// <param name="positionOffset">Absolute position of token 0 in the full causal window (= cachedLength in the typical autoregressive case).</param>
    /// <param name="layer">Per-layer MLA weights bundle.</param>
    /// <param name="kvCache">Phase A expanded MLA KV cache.</param>
    /// <param name="layerIndex">Which layer in <paramref name="kvCache"/> to write into / read from.</param>
    /// <param name="ropeCosF32">Device pointer to F32 RoPE cos table <c>[maxSeq, qkRope/2]</c>.</param>
    /// <param name="ropeSinF32">Device pointer to F32 RoPE sin table <c>[maxSeq, qkRope/2]</c>.</param>
    /// <param name="rmsNormEps">RMSNorm epsilon shared by all three RMSNorm sites in the layer.</param>
    /// <param name="softmaxScale">Combined softmax scale: <c>(1 / sqrt(qk_head_dim)) * yarn_mscale²</c>.</param>
    /// <param name="scratch">Caller-owned scratch buffers (see <see cref="CudaMlaScratch"/>).</param>
    /// <param name="cublasHandle">cuBLAS handle for F32 GEMM/GEMV.</param>
    /// <param name="kernels">Loaded PTX kernel module.</param>
    /// <param name="stream">CUDA stream.</param>
    public static void Forward(
        nint hiddenF32, nint outputF32,
        int seqLen, int positionOffset,
        in CudaMlaLayerWeights layer,
        CudaMlaKvCache kvCache, int layerIndex,
        nint ropeCosF32, nint ropeSinF32,
        float rmsNormEps, float softmaxScale,
        CudaMlaScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream)
    {
        if (!kernels.HasMlaAttentionKernel || !kernels.HasMlaHelpers)
            throw new InvalidOperationException(
                "MLA kernels not available. Compile native/kernels/attention_mla.cu and mla_helpers.cu.");

        scratch.EnsureCapacity(seqLen, layer);

        int hiddenSize = layer.HiddenSize;
        int qkNope = layer.QkNopeHeadDim;
        int qkRope = layer.QkRopeHeadDim;
        int qkHead = qkNope + qkRope;
        int vHead = layer.VHeadDim;
        int numHeads = layer.NumHeads;
        int qLora = layer.QLoraRank;
        int kvLora = layer.KvLoraRank;
        int qTotal = layer.QTotalElems;
        int kvAOut = layer.KvAOutElems;
        int kvBOut = layer.KvBOutElems;
        int oInput = layer.OInputDim;

        // ── Pre-attention RMSNorm (input → normHidden) ─────────────────
        kernels.LaunchMlaRmsNormF32(
            hiddenF32, layer.AttnNormWeight, scratch.NormHidden,
            seqLen, hiddenSize, rmsNormEps, stream);

        // ── Q path ──
        // LoRA-factored: q_latent = q_a @ normHidden;  q_latent_norm = RMSNorm(q_latent)
        //                q = q_b @ q_latent_norm.
        // Monolithic: q = q_proj @ normHidden.
        if (qLora > 0)
        {
            CudaGemm.LinearF32(
                cublasHandle, scratch.NormHidden, layer.QAProj, scratch.QLatent,
                seqLen, hiddenSize, qLora, stream);
            kernels.LaunchMlaRmsNormF32(
                scratch.QLatent, layer.QALayernormWeight, scratch.QLatentNorm,
                seqLen, qLora, rmsNormEps, stream);
            CudaGemm.LinearF32(
                cublasHandle, scratch.QLatentNorm, layer.QBProj, scratch.Q,
                seqLen, qLora, qTotal, stream);
        }
        else
        {
            CudaGemm.LinearF32(
                cublasHandle, scratch.NormHidden, layer.QProj, scratch.Q,
                seqLen, hiddenSize, qTotal, stream);
        }

        // ── KV down-projection + split ──
        // compressed = kv_a @ normHidden  → [seqLen, kvLora + qkRope]
        // first kvLora floats per row → c_kv (latent), next qkRope → k_pe (shared, pre-RoPE)
        CudaGemm.LinearF32(
            cublasHandle, scratch.NormHidden, layer.KvAProjWithMqa, scratch.CompressedKv,
            seqLen, hiddenSize, kvAOut, stream);

        // RMSNorm the latent half row-by-row (dim=kvLora, stride=kvAOut).
        // The MLA helper RMSNorm operates on contiguous rows; here we pass the
        // strided latent slice as input and a contiguous output. The kernel
        // doesn't support stride yet — we extract the latent first into a
        // contiguous scratch.
        ExtractLatentSlice(
            scratch.CompressedKv, scratch.KvLatentNorm, seqLen, kvAOut, kvLora, stream);
        kernels.LaunchMlaRmsNormF32(
            scratch.KvLatentNorm, layer.KvALayernormWeight, scratch.KvLatentNormOut,
            seqLen, kvLora, rmsNormEps, stream);

        // kvB expansion: kvBExpanded = kv_b @ kvLatentNormOut → [seqLen, numHeads * (qkNope + vHead)]
        CudaGemm.LinearF32(
            cublasHandle, scratch.KvLatentNormOut, layer.KvBProj, scratch.KvBExpanded,
            seqLen, kvLora, kvBOut, stream);

        // Per-head split into K_nope and V scratch buffers (still local —
        // we'll memcpy into the persistent cache after RoPE on K_pe).
        kernels.LaunchMlaSplitKvB(
            scratch.KvBExpanded, scratch.KNope, scratch.V,
            seqLen, numHeads, qkNope, vHead, stream);

        // Extract the shared k_pe (last qkRope floats per row of compressedKv) into
        // its own contiguous buffer (one per token, pre-RoPE).
        ExtractKpeSlice(
            scratch.CompressedKv, scratch.KPe, seqLen, kvAOut, kvLora, qkRope, stream);

        // ── RoPE on Q.rope (per head) and shared K_pe ──
        kernels.LaunchMlaRopeQpe(
            scratch.Q, ropeCosF32, ropeSinF32,
            seqLen, numHeads, qkNope, qkRope, positionOffset, stream);
        kernels.LaunchMlaRopeKpe(
            scratch.KPe, ropeCosF32, ropeSinF32,
            seqLen, qkRope, positionOffset, stream);

        // ── Append new K_nope / V / K_pe rows into the persistent cache at
        //    offset cachedLength = positionOffset (autoregressive convention). ──
        int cachedLength = kvCache.GetCurrentLength(layerIndex);
        long kNopeRowBytes = kvCache.KNopeRowBytes;
        long vRowBytes = kvCache.VRowBytes;
        long kPeRowBytes = kvCache.KPeRowBytes;

        nint dstKNope = kvCache.GetKNopePtr(layerIndex) + (nint)((long)cachedLength * kNopeRowBytes);
        nint dstV = kvCache.GetVPtr(layerIndex) + (nint)((long)cachedLength * vRowBytes);
        nint dstKPe = kvCache.GetKPePtr(layerIndex) + (nint)((long)cachedLength * kPeRowBytes);

        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstKNope, scratch.KNope,
            (nuint)((long)seqLen * kNopeRowBytes), stream).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstV, scratch.V,
            (nuint)((long)seqLen * vRowBytes), stream).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstKPe, scratch.KPe,
            (nuint)((long)seqLen * kPeRowBytes), stream).ThrowOnError();

        // ── Attention over [0, cachedLength + seqLen) ──
        int seqKv = cachedLength + seqLen;
        kernels.LaunchAttentionMla(
            scratch.Q, kvCache.GetKNopePtr(layerIndex), kvCache.GetKPePtr(layerIndex),
            kvCache.GetVPtr(layerIndex), scratch.AttnOut,
            seqLen, seqKv, numHeads, qkNope, qkRope, vHead,
            positionOffset, softmaxScale, stream);

        // ── O projection: output = o_proj @ attn_out ──
        CudaGemm.LinearF32(
            cublasHandle, scratch.AttnOut, layer.OProj, outputF32,
            seqLen, oInput, hiddenSize, stream);
    }

    /// <summary>
    /// Extracts the first <paramref name="dstWidth"/> floats of each row of a
    /// strided <paramref name="src"/> buffer into a contiguous destination.
    /// Used to feed the RMSNorm kernel which currently expects contiguous input.
    /// </summary>
    /// <remarks>
    /// Two cuMemcpyDtoDAsync would suffice here but we keep it as a tight
    /// strided D2D loop to make the per-row stride explicit; the launch
    /// stays asynchronous on the same stream.
    /// </remarks>
    private static void ExtractLatentSlice(
        nint src, nint dst, int seqLen, int srcStride, int dstWidth, nint stream)
    {
        long dstRowBytes = (long)dstWidth * sizeof(float);
        long srcRowBytes = (long)srcStride * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                dst + (nint)((long)t * dstRowBytes),
                src + (nint)((long)t * srcRowBytes),
                (nuint)dstRowBytes, stream).ThrowOnError();
        }
    }

    /// <summary>
    /// Extracts the last <paramref name="kPeWidth"/> floats of each row of a
    /// strided <paramref name="src"/> buffer (offset by the <c>kvLora</c> head
    /// dimension) into a contiguous destination.
    /// </summary>
    private static void ExtractKpeSlice(
        nint src, nint dst, int seqLen, int srcStride, int kvLora, int kPeWidth, nint stream)
    {
        long srcRowBytes = (long)srcStride * sizeof(float);
        long dstRowBytes = (long)kPeWidth * sizeof(float);
        long offsetWithinRow = (long)kvLora * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                dst + (nint)((long)t * dstRowBytes),
                src + (nint)((long)t * srcRowBytes + (nint)offsetWithinRow),
                (nuint)dstRowBytes, stream).ThrowOnError();
        }
    }
}

/// <summary>
/// Caller-owned per-call scratch buffers for <see cref="CudaMlaAttention.Forward"/>.
/// Sized to the maximum <c>seqLen</c> the caller will ever pass; the helper
/// re-allocates with power-of-2 growth on demand. Reused across layers and
/// across forward calls.
/// </summary>
public sealed unsafe class CudaMlaScratch : IDisposable
{
    private nint _normHidden;       // [seqLen, hidden]
    private nint _qLatent;          // [seqLen, qLora] (only when LoRA path)
    private nint _qLatentNorm;
    private nint _q;                // [seqLen, numHeads * qkHead]
    private nint _compressedKv;     // [seqLen, kvLora + qkRope]
    private nint _kvLatentNorm;     // [seqLen, kvLora] (extracted slice — pre-RMSNorm)
    private nint _kvLatentNormOut;  // [seqLen, kvLora]
    private nint _kvBExpanded;      // [seqLen, numHeads * (qkNope + vHead)]
    private nint _kNope;            // [seqLen, numHeads * qkNope]
    private nint _v;                // [seqLen, numHeads * vHead]
    private nint _kPe;              // [seqLen, qkRope]
    private nint _attnOut;          // [seqLen, numHeads * vHead]

    private int _capacitySeqLen;
    private int _hidden, _qLora, _qkHead, _qkRope, _qkNope, _vHead, _numHeads, _kvLora;

    /// <summary>Total allocated bytes across all scratch buffers.</summary>
    public long AllocatedBytes { get; private set; }

    internal nint NormHidden => _normHidden;
    internal nint QLatent => _qLatent;
    internal nint QLatentNorm => _qLatentNorm;
    internal nint Q => _q;
    internal nint CompressedKv => _compressedKv;
    internal nint KvLatentNorm => _kvLatentNorm;
    internal nint KvLatentNormOut => _kvLatentNormOut;
    internal nint KvBExpanded => _kvBExpanded;
    internal nint KNope => _kNope;
    internal nint V => _v;
    internal nint KPe => _kPe;
    internal nint AttnOut => _attnOut;

    /// <summary>
    /// Ensures all scratch buffers can hold <paramref name="seqLen"/> tokens
    /// for this layer's MLA shapes. Reallocates with power-of-2 growth when
    /// the requested capacity exceeds the current allocation.
    /// </summary>
    public void EnsureCapacity(int seqLen, in CudaMlaLayerWeights layer)
    {
        if (seqLen <= _capacitySeqLen
            && _hidden == layer.HiddenSize
            && _qLora == layer.QLoraRank
            && _qkNope == layer.QkNopeHeadDim
            && _qkRope == layer.QkRopeHeadDim
            && _vHead == layer.VHeadDim
            && _numHeads == layer.NumHeads
            && _kvLora == layer.KvLoraRank)
            return;

        int newCap = Math.Max(seqLen, 1);
        // Power-of-2 growth keeps reallocs amortised when the caller ramps
        // through small seqLens before stabilising on the prompt length.
        if (newCap > _capacitySeqLen)
        {
            newCap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)newCap);
        }

        Free();

        _hidden = layer.HiddenSize;
        _qLora = layer.QLoraRank;
        _qkNope = layer.QkNopeHeadDim;
        _qkRope = layer.QkRopeHeadDim;
        _qkHead = _qkNope + _qkRope;
        _vHead = layer.VHeadDim;
        _numHeads = layer.NumHeads;
        _kvLora = layer.KvLoraRank;
        _capacitySeqLen = newCap;

        int qTotal = _numHeads * _qkHead;
        int kvAOut = _kvLora + _qkRope;
        int kvBOut = _numHeads * (_qkNope + _vHead);

        _normHidden = Alloc((long)newCap * _hidden);
        if (_qLora > 0)
        {
            _qLatent = Alloc((long)newCap * _qLora);
            _qLatentNorm = Alloc((long)newCap * _qLora);
        }
        _q = Alloc((long)newCap * qTotal);
        _compressedKv = Alloc((long)newCap * kvAOut);
        _kvLatentNorm = Alloc((long)newCap * _kvLora);
        _kvLatentNormOut = Alloc((long)newCap * _kvLora);
        _kvBExpanded = Alloc((long)newCap * kvBOut);
        _kNope = Alloc((long)newCap * _numHeads * _qkNope);
        _v = Alloc((long)newCap * _numHeads * _vHead);
        _kPe = Alloc((long)newCap * _qkRope);
        _attnOut = Alloc((long)newCap * _numHeads * _vHead);
    }

    private nint Alloc(long elemCount)
    {
        long bytes = elemCount * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private void Free()
    {
        FreeIfNonZero(ref _normHidden);
        FreeIfNonZero(ref _qLatent);
        FreeIfNonZero(ref _qLatentNorm);
        FreeIfNonZero(ref _q);
        FreeIfNonZero(ref _compressedKv);
        FreeIfNonZero(ref _kvLatentNorm);
        FreeIfNonZero(ref _kvLatentNormOut);
        FreeIfNonZero(ref _kvBExpanded);
        FreeIfNonZero(ref _kNope);
        FreeIfNonZero(ref _v);
        FreeIfNonZero(ref _kPe);
        FreeIfNonZero(ref _attnOut);
        AllocatedBytes = 0;
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    /// <summary>Frees every device buffer.</summary>
    public void Dispose()
    {
        Free();
        _capacitySeqLen = 0;
    }
}

#region MLA Phase B (latent KV cache + W_UK absorbed attention)

/// <summary>
/// MLA Phase B forward helpers: latent-cache + absorbed attention. The
/// production decode efficiency path for DeepSeek-V2/V3 — stores only the
/// shared <c>c_kv</c> + <c>k_pe</c> latents (~7-14× smaller than Phase A's
/// expanded cache) and recovers per-head K/V on the fly via the W_UK / W_UV
/// absorption identities derived in <see cref="DotLLM.Cpu.Kernels.MlaAttention.ExecuteLatent"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Coexists with Phase A.</b> <see cref="CudaMlaAttention.Forward"/> stays
/// unchanged and continues to drive the expanded <see cref="CudaMlaKvCache"/>.
/// <see cref="CudaMlaAttention.ForwardLatent"/> is a parallel path the wiring
/// agent can pick per layer or per call. The on-disk caches are different
/// types (<see cref="CudaMlaKvCache"/> vs <see cref="CudaMlaLatentKvCache"/>),
/// so the two paths are not interchangeable mid-sequence.
/// </para>
/// <para>
/// <b>F32 throughout.</b> Matches Phase A's correctness-first contract. FP16
/// is the next follow-up agent.
/// </para>
/// </remarks>
public static unsafe partial class CudaMlaAttention
{
    /// <summary>
    /// MLA Phase B forward: latent KV cache + W_UK absorbed attention. Runs the
    /// equivalent of <c>DotLLM.Cpu.Kernels.MlaAttention.ExecuteLatent</c> on GPU.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Math: per head <c>h</c>, per query token <c>t</c>:
    /// </para>
    /// <list type="number">
    ///   <item>Project + RoPE + cache-write (identical to Phase A but writes only
    ///     <c>c_kv</c> + <c>k_pe</c>).</item>
    ///   <item><c>Q_absorbed[h,t] = W_UK[h]^T @ Q_nope[h,t]</c> (per (h,t),
    ///     yields <c>kvLoraRank</c> floats).</item>
    ///   <item>Score = <c>Q_absorbed[h,t] · c_kv[s] + Q_pe[h,t] · k_pe[s]</c>,
    ///     causal-masked, softmaxed.</item>
    ///   <item><c>c_v_out[h,t] = Σ_s softmax · c_kv[s]</c> (latent V output,
    ///     <c>kvLoraRank</c> floats).</item>
    ///   <item><c>attn_out[h,t] = W_UV[h] @ c_v_out[h,t]</c> (vHead floats).</item>
    ///   <item><c>output = o_proj @ attn_out</c> (hidden floats).</item>
    /// </list>
    /// </remarks>
    public static void ForwardLatent(
        nint hiddenF32, nint outputF32,
        int seqLen, int positionOffset,
        in CudaMlaLayerWeights layer,
        CudaMlaLatentKvCache kvCache, int layerIndex,
        nint ropeCosF32, nint ropeSinF32,
        float rmsNormEps, float softmaxScale,
        CudaMlaLatentScratch scratch, nint cublasHandle, CudaKernels kernels, nint stream)
    {
        if (!kernels.HasMlaPhaseB || !kernels.HasMlaHelpers)
            throw new InvalidOperationException(
                "MLA Phase B kernels not available. Compile native/kernels/attention_mla_latent.cu and mla_helpers.cu.");

        scratch.EnsureCapacity(seqLen, layer);

        int hiddenSize = layer.HiddenSize;
        int qkNope = layer.QkNopeHeadDim;
        int qkRope = layer.QkRopeHeadDim;
        int qkHead = qkNope + qkRope;
        int vHead = layer.VHeadDim;
        int numHeads = layer.NumHeads;
        int qLora = layer.QLoraRank;
        int kvLora = layer.KvLoraRank;
        int qTotal = layer.QTotalElems;
        int kvAOut = layer.KvAOutElems;
        int oInput = layer.OInputDim;

        // ── Pre-attention RMSNorm ──
        kernels.LaunchMlaRmsNormF32(
            hiddenF32, layer.AttnNormWeight, scratch.NormHidden,
            seqLen, hiddenSize, rmsNormEps, stream);

        // ── Q path (LoRA-factored or monolithic) ──
        if (qLora > 0)
        {
            CudaGemm.LinearF32(
                cublasHandle, scratch.NormHidden, layer.QAProj, scratch.QLatent,
                seqLen, hiddenSize, qLora, stream);
            kernels.LaunchMlaRmsNormF32(
                scratch.QLatent, layer.QALayernormWeight, scratch.QLatentNorm,
                seqLen, qLora, rmsNormEps, stream);
            CudaGemm.LinearF32(
                cublasHandle, scratch.QLatentNorm, layer.QBProj, scratch.Q,
                seqLen, qLora, qTotal, stream);
        }
        else
        {
            CudaGemm.LinearF32(
                cublasHandle, scratch.NormHidden, layer.QProj, scratch.Q,
                seqLen, hiddenSize, qTotal, stream);
        }

        // ── KV down-projection + split ──
        // compressed = kv_a @ normHidden  → [seqLen, kvLora + qkRope]
        CudaGemm.LinearF32(
            cublasHandle, scratch.NormHidden, layer.KvAProjWithMqa, scratch.CompressedKv,
            seqLen, hiddenSize, kvAOut, stream);

        // Extract latent slice → RMSNorm.
        ExtractLatentSliceLatent(
            scratch.CompressedKv, scratch.KvLatentNorm, seqLen, kvAOut, kvLora, stream);
        kernels.LaunchMlaRmsNormF32(
            scratch.KvLatentNorm, layer.KvALayernormWeight, scratch.KvLatentNormOut,
            seqLen, kvLora, rmsNormEps, stream);

        // Extract k_pe slice (last qkRope of compressed row, pre-RoPE).
        ExtractKpeSliceLatent(
            scratch.CompressedKv, scratch.KPe, seqLen, kvAOut, kvLora, qkRope, stream);

        // ── RoPE on Q.rope (per head) and shared K_pe ──
        kernels.LaunchMlaRopeQpe(
            scratch.Q, ropeCosF32, ropeSinF32,
            seqLen, numHeads, qkNope, qkRope, positionOffset, stream);
        kernels.LaunchMlaRopeKpe(
            scratch.KPe, ropeCosF32, ropeSinF32,
            seqLen, qkRope, positionOffset, stream);

        // ── Cache write: append latentNorm + k_pe at offset cachedLength ──
        int cachedLength = kvCache.GetCurrentLength(layerIndex);
        long cKvRowBytes = kvCache.CKvRowBytes;
        long kPeRowBytes = kvCache.KPeRowBytes;
        nint dstCKv = kvCache.GetCKvPtr(layerIndex) + (nint)((long)cachedLength * cKvRowBytes);
        nint dstKPe = kvCache.GetKPePtr(layerIndex) + (nint)((long)cachedLength * kPeRowBytes);

        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstCKv, scratch.KvLatentNormOut,
            (nuint)((long)seqLen * cKvRowBytes), stream).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstKPe, scratch.KPe,
            (nuint)((long)seqLen * kPeRowBytes), stream).ThrowOnError();

        // ── Q absorption: Q_absorbed = W_UK^T @ Q_nope ──
        kernels.LaunchMlaQAbsorbUk(
            scratch.Q, layer.KvBProj, scratch.QAbsorbed,
            seqLen, numHeads, qkNope, qkRope, vHead, kvLora, stream);

        // ── Absorbed attention over the latent cache ──
        // Note: the kernel expects q_pe as a separate buffer, but Phase A's
        // packed Q layout has q_pe interleaved at offset qkNope inside each
        // head. We pass a base pointer + the kernel walks by num_heads *
        // qkRope stride per token; the per-(t,h) offset is done internally.
        // Same trick: pass scratch.Q + qkNope so the kernel reads from the
        // right starting offset. The kernel uses q_pe_stride = num_heads *
        // qkRope, but our q layout uses stride num_heads * (qkNope + qkRope).
        // → we must materialise q_pe contiguously OR change the kernel.
        //
        // Materialise q_pe contiguously into scratch.QPe so the kernel
        // contract holds without surgical kernel changes.
        ExtractQPeSliceLatent(
            scratch.Q, scratch.QPe, seqLen, numHeads, qkNope, qkRope, stream);

        int seqKv = cachedLength + seqLen;
        kernels.LaunchAttentionMlaLatent(
            scratch.QAbsorbed, scratch.QPe,
            kvCache.GetCKvPtr(layerIndex), kvCache.GetKPePtr(layerIndex),
            scratch.CVOut,
            seqLen, seqKv,
            numHeads, kvLora, qkRope,
            positionOffset, softmaxScale, stream);

        // ── V expansion: attn_out = W_UV @ c_v_out (per head, per token) ──
        kernels.LaunchMlaVExpandUv(
            scratch.CVOut, layer.KvBProj, scratch.AttnOut,
            seqLen, numHeads, qkNope, vHead, kvLora, stream);

        // ── O projection ──
        CudaGemm.LinearF32(
            cublasHandle, scratch.AttnOut, layer.OProj, outputF32,
            seqLen, oInput, hiddenSize, stream);
    }

    /// <summary>Strided D2D extract of the first dstWidth floats per row (Phase B copy of ExtractLatentSlice).</summary>
    private static void ExtractLatentSliceLatent(
        nint src, nint dst, int seqLen, int srcStride, int dstWidth, nint stream)
    {
        long dstRowBytes = (long)dstWidth * sizeof(float);
        long srcRowBytes = (long)srcStride * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                dst + (nint)((long)t * dstRowBytes),
                src + (nint)((long)t * srcRowBytes),
                (nuint)dstRowBytes, stream).ThrowOnError();
        }
    }

    /// <summary>Strided D2D extract of the last kPeWidth floats per row (Phase B copy of ExtractKpeSlice).</summary>
    private static void ExtractKpeSliceLatent(
        nint src, nint dst, int seqLen, int srcStride, int kvLora, int kPeWidth, nint stream)
    {
        long srcRowBytes = (long)srcStride * sizeof(float);
        long dstRowBytes = (long)kPeWidth * sizeof(float);
        long offsetWithinRow = (long)kvLora * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                dst + (nint)((long)t * dstRowBytes),
                src + (nint)((long)t * srcRowBytes + (nint)offsetWithinRow),
                (nuint)dstRowBytes, stream).ThrowOnError();
        }
    }

    /// <summary>
    /// Materialises the per-head q_pe slices contiguously so the absorbed
    /// attention kernel can use a [seqLen, numHeads, qkRope] stride. Source
    /// layout interleaves q_nope and q_pe per head, so each (t, h) row needs
    /// its own short DtoD copy. seqLen * numHeads launches per call — small
    /// enough for decode (numHeads is 16 on V2-Lite, 128 on V2-full); a fused
    /// kernel is a follow-up.
    /// </summary>
    private static void ExtractQPeSliceLatent(
        nint qSrc, nint qPeDst,
        int seqLen, int numHeads, int qkNope, int qkRope, nint stream)
    {
        int qkHead = qkNope + qkRope;
        long qStrideBytes = (long)numHeads * qkHead * sizeof(float);
        long qPeStrideBytes = (long)numHeads * qkRope * sizeof(float);
        long perHeadBytes = (long)qkRope * sizeof(float);
        long qkNopeBytes = (long)qkNope * sizeof(float);
        long qkHeadBytes = (long)qkHead * sizeof(float);

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                nint srcPtr = qSrc + (nint)((long)t * qStrideBytes + (long)h * qkHeadBytes + qkNopeBytes);
                nint dstPtr = qPeDst + (nint)((long)t * qPeStrideBytes + (long)h * perHeadBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(dstPtr, srcPtr, (nuint)perHeadBytes, stream)
                             .ThrowOnError();
            }
        }
    }
}

/// <summary>
/// Caller-owned scratch for <see cref="CudaMlaAttention.ForwardLatent"/>.
/// Sized once for the maximum <c>seqLen</c> the caller will pass (with
/// power-of-2 growth), reused across layers and across forward calls.
/// Distinct from <see cref="CudaMlaScratch"/> because Phase B's intermediate
/// shapes differ — no kvBExpanded / kNope / v (Phase A's per-head expansion
/// is gone), but adds qAbsorbed / cVOut / qPe (the absorbed-path scratch).
/// </summary>
public sealed unsafe class CudaMlaLatentScratch : IDisposable
{
    private nint _normHidden;       // [seqLen, hidden]
    private nint _qLatent;          // [seqLen, qLora]
    private nint _qLatentNorm;
    private nint _q;                // [seqLen, numHeads * qkHead]
    private nint _qPe;              // [seqLen, numHeads * qkRope] — contiguous q_pe slice
    private nint _qAbsorbed;        // [seqLen, numHeads * kvLora]
    private nint _compressedKv;     // [seqLen, kvLora + qkRope]
    private nint _kvLatentNorm;     // [seqLen, kvLora] (extracted slice, pre-RMSNorm)
    private nint _kvLatentNormOut;  // [seqLen, kvLora]
    private nint _kPe;              // [seqLen, qkRope]
    private nint _cVOut;            // [seqLen, numHeads * kvLora] — latent attention output
    private nint _attnOut;          // [seqLen, numHeads * vHead]

    private int _capacitySeqLen;
    private int _hidden, _qLora, _qkHead, _qkRope, _qkNope, _vHead, _numHeads, _kvLora;

    /// <summary>Total allocated bytes across all scratch buffers.</summary>
    public long AllocatedBytes { get; private set; }

    internal nint NormHidden => _normHidden;
    internal nint QLatent => _qLatent;
    internal nint QLatentNorm => _qLatentNorm;
    internal nint Q => _q;
    internal nint QPe => _qPe;
    internal nint QAbsorbed => _qAbsorbed;
    internal nint CompressedKv => _compressedKv;
    internal nint KvLatentNorm => _kvLatentNorm;
    internal nint KvLatentNormOut => _kvLatentNormOut;
    internal nint KPe => _kPe;
    internal nint CVOut => _cVOut;
    internal nint AttnOut => _attnOut;

    /// <summary>
    /// Ensures all scratch buffers can hold <paramref name="seqLen"/> tokens
    /// for this layer's MLA shapes. Reallocates with power-of-2 growth on demand.
    /// </summary>
    public void EnsureCapacity(int seqLen, in CudaMlaLayerWeights layer)
    {
        if (seqLen <= _capacitySeqLen
            && _hidden == layer.HiddenSize
            && _qLora == layer.QLoraRank
            && _qkNope == layer.QkNopeHeadDim
            && _qkRope == layer.QkRopeHeadDim
            && _vHead == layer.VHeadDim
            && _numHeads == layer.NumHeads
            && _kvLora == layer.KvLoraRank)
            return;

        int newCap = Math.Max(seqLen, 1);
        if (newCap > _capacitySeqLen)
            newCap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)newCap);

        Free();

        _hidden = layer.HiddenSize;
        _qLora = layer.QLoraRank;
        _qkNope = layer.QkNopeHeadDim;
        _qkRope = layer.QkRopeHeadDim;
        _qkHead = _qkNope + _qkRope;
        _vHead = layer.VHeadDim;
        _numHeads = layer.NumHeads;
        _kvLora = layer.KvLoraRank;
        _capacitySeqLen = newCap;

        int qTotal = _numHeads * _qkHead;
        int kvAOut = _kvLora + _qkRope;

        _normHidden = Alloc((long)newCap * _hidden);
        if (_qLora > 0)
        {
            _qLatent = Alloc((long)newCap * _qLora);
            _qLatentNorm = Alloc((long)newCap * _qLora);
        }
        _q = Alloc((long)newCap * qTotal);
        _qPe = Alloc((long)newCap * _numHeads * _qkRope);
        _qAbsorbed = Alloc((long)newCap * _numHeads * _kvLora);
        _compressedKv = Alloc((long)newCap * kvAOut);
        _kvLatentNorm = Alloc((long)newCap * _kvLora);
        _kvLatentNormOut = Alloc((long)newCap * _kvLora);
        _kPe = Alloc((long)newCap * _qkRope);
        _cVOut = Alloc((long)newCap * _numHeads * _kvLora);
        _attnOut = Alloc((long)newCap * _numHeads * _vHead);
    }

    private nint Alloc(long elemCount)
    {
        long bytes = elemCount * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private void Free()
    {
        FreeIfNonZero(ref _normHidden);
        FreeIfNonZero(ref _qLatent);
        FreeIfNonZero(ref _qLatentNorm);
        FreeIfNonZero(ref _q);
        FreeIfNonZero(ref _qPe);
        FreeIfNonZero(ref _qAbsorbed);
        FreeIfNonZero(ref _compressedKv);
        FreeIfNonZero(ref _kvLatentNorm);
        FreeIfNonZero(ref _kvLatentNormOut);
        FreeIfNonZero(ref _kPe);
        FreeIfNonZero(ref _cVOut);
        FreeIfNonZero(ref _attnOut);
        AllocatedBytes = 0;
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    /// <summary>Frees every device buffer.</summary>
    public void Dispose()
    {
        Free();
        _capacitySeqLen = 0;
    }
}

#endregion

public static unsafe partial class CudaMlaAttention
{

    #region MLA FP16
    /// <summary>
    /// FP16 sibling of <see cref="Forward"/>. Runs the full MLA layer in FP16
    /// — weights, activations, and cache are all FP16 — with FP32 reductions
    /// in RMSNorm and softmax. Matches the GQA precision pattern on this
    /// branch; ~2× memory reduction vs F32 plus better cuBLAS HGEMM throughput.
    /// </summary>
    /// <param name="hiddenF16">Device pointer to F16 hidden state input <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="outputF16">Device pointer to F16 output <c>[seqLen, hiddenSize]</c>. Receives the post-o_proj attention output (caller adds residual separately).</param>
    /// <param name="seqLen">Number of tokens this call processes.</param>
    /// <param name="positionOffset">Absolute position of token 0 in the full causal window.</param>
    /// <param name="layer">Per-layer MLA weights bundle. <c>layer.Precision</c> must be <see cref="MlaPrecision.F16"/>.</param>
    /// <param name="kvCache">Phase A expanded MLA KV cache — must be F16 (<see cref="CudaMlaKvCache.Precision"/> == F16).</param>
    /// <param name="layerIndex">Which layer in <paramref name="kvCache"/> to write into / read from.</param>
    /// <param name="ropeCosF32">Device pointer to F32 RoPE cos table <c>[maxSeq, qkRope/2]</c>.</param>
    /// <param name="ropeSinF32">Device pointer to F32 RoPE sin table <c>[maxSeq, qkRope/2]</c>.</param>
    /// <param name="rmsNormEps">RMSNorm epsilon shared by all three RMSNorm sites.</param>
    /// <param name="softmaxScale">Combined softmax scale: <c>(1 / sqrt(qk_head_dim)) * yarn_mscale²</c>.</param>
    /// <param name="scratch">Caller-owned FP16 scratch buffers.</param>
    /// <param name="cublasHandle">cuBLAS handle for FP16 HGEMM.</param>
    /// <param name="kernels">Loaded PTX kernel module.</param>
    /// <param name="stream">CUDA stream.</param>
    public static void ForwardF16(
        nint hiddenF16, nint outputF16,
        int seqLen, int positionOffset,
        in CudaMlaLayerWeights layer,
        CudaMlaKvCache kvCache, int layerIndex,
        nint ropeCosF32, nint ropeSinF32,
        float rmsNormEps, float softmaxScale,
        CudaMlaScratchF16 scratch, nint cublasHandle, CudaKernels kernels, nint stream)
    {
        if (layer.Precision != MlaPrecision.F16)
            throw new InvalidOperationException(
                $"ForwardF16 requires FP16 weights but layer.Precision is {layer.Precision}.");
        if (kvCache.Precision != MlaPrecision.F16)
            throw new InvalidOperationException(
                $"ForwardF16 requires an FP16 KV cache but cache.Precision is {kvCache.Precision}.");
        if (!kernels.HasMlaAttentionKernelF16 || !kernels.HasMlaHelpersF16)
            throw new InvalidOperationException(
                "MLA FP16 kernels not available. Rebuild PTX from native/kernels/attention_mla.cu and mla_helpers.cu.");

        scratch.EnsureCapacity(seqLen, layer);

        int hiddenSize = layer.HiddenSize;
        int qkNope = layer.QkNopeHeadDim;
        int qkRope = layer.QkRopeHeadDim;
        int qkHead = qkNope + qkRope;
        int vHead = layer.VHeadDim;
        int numHeads = layer.NumHeads;
        int qLora = layer.QLoraRank;
        int kvLora = layer.KvLoraRank;
        int qTotal = layer.QTotalElems;
        int kvAOut = layer.KvAOutElems;
        int kvBOut = layer.KvBOutElems;
        int oInput = layer.OInputDim;

        // ── Pre-attention RMSNorm (FP16 in / FP16 out, F32 weight) ──
        kernels.LaunchMlaRmsNormF16(
            hiddenF16, layer.AttnNormWeight, scratch.NormHidden,
            seqLen, hiddenSize, rmsNormEps, stream);

        // ── Q path (cuBLAS HGEMM) ──
        if (qLora > 0)
        {
            CudaGemm.LinearF16(
                cublasHandle, scratch.NormHidden, layer.QAProj, scratch.QLatent,
                seqLen, hiddenSize, qLora, stream);
            kernels.LaunchMlaRmsNormF16(
                scratch.QLatent, layer.QALayernormWeight, scratch.QLatentNorm,
                seqLen, qLora, rmsNormEps, stream);
            CudaGemm.LinearF16(
                cublasHandle, scratch.QLatentNorm, layer.QBProj, scratch.Q,
                seqLen, qLora, qTotal, stream);
        }
        else
        {
            CudaGemm.LinearF16(
                cublasHandle, scratch.NormHidden, layer.QProj, scratch.Q,
                seqLen, hiddenSize, qTotal, stream);
        }

        // ── KV down-projection + split (FP16) ──
        CudaGemm.LinearF16(
            cublasHandle, scratch.NormHidden, layer.KvAProjWithMqa, scratch.CompressedKv,
            seqLen, hiddenSize, kvAOut, stream);

        // Extract latent half (first kvLora F16s of each row), RMSNorm.
        ExtractLatentSliceF16(
            scratch.CompressedKv, scratch.KvLatentNorm, seqLen, kvAOut, kvLora, stream);
        kernels.LaunchMlaRmsNormF16(
            scratch.KvLatentNorm, layer.KvALayernormWeight, scratch.KvLatentNormOut,
            seqLen, kvLora, rmsNormEps, stream);

        // KV-B expansion (FP16 HGEMM).
        CudaGemm.LinearF16(
            cublasHandle, scratch.KvLatentNormOut, layer.KvBProj, scratch.KvBExpanded,
            seqLen, kvLora, kvBOut, stream);

        // Per-head split into K_nope and V scratch buffers (F16 in / F16 out).
        kernels.LaunchMlaSplitKvBF16(
            scratch.KvBExpanded, scratch.KNope, scratch.V,
            seqLen, numHeads, qkNope, vHead, stream);

        // Extract K_pe (last qkRope F16s of each row), pre-RoPE.
        ExtractKpeSliceF16(
            scratch.CompressedKv, scratch.KPe, seqLen, kvAOut, kvLora, qkRope, stream);

        // ── RoPE on Q.rope (per head) and shared K_pe (FP16 in-place) ──
        kernels.LaunchMlaRopeQpeF16(
            scratch.Q, ropeCosF32, ropeSinF32,
            seqLen, numHeads, qkNope, qkRope, positionOffset, stream);
        kernels.LaunchMlaRopeKpeF16(
            scratch.KPe, ropeCosF32, ropeSinF32,
            seqLen, qkRope, positionOffset, stream);

        // ── Append new K_nope / V / K_pe rows into the FP16 cache. ──
        int cachedLength = kvCache.GetCurrentLength(layerIndex);
        long kNopeRowBytes = kvCache.KNopeRowBytes;
        long vRowBytes = kvCache.VRowBytes;
        long kPeRowBytes = kvCache.KPeRowBytes;

        nint dstKNope = kvCache.GetKNopePtr(layerIndex) + (nint)((long)cachedLength * kNopeRowBytes);
        nint dstV = kvCache.GetVPtr(layerIndex) + (nint)((long)cachedLength * vRowBytes);
        nint dstKPe = kvCache.GetKPePtr(layerIndex) + (nint)((long)cachedLength * kPeRowBytes);

        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstKNope, scratch.KNope,
            (nuint)((long)seqLen * kNopeRowBytes), stream).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstV, scratch.V,
            (nuint)((long)seqLen * vRowBytes), stream).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoDAsync_v2(dstKPe, scratch.KPe,
            (nuint)((long)seqLen * kPeRowBytes), stream).ThrowOnError();

        // ── Attention over [0, cachedLength + seqLen) ──
        int seqKv = cachedLength + seqLen;
        kernels.LaunchAttentionMlaF16(
            scratch.Q, kvCache.GetKNopePtr(layerIndex), kvCache.GetKPePtr(layerIndex),
            kvCache.GetVPtr(layerIndex), scratch.AttnOut,
            seqLen, seqKv, numHeads, qkNope, qkRope, vHead,
            positionOffset, softmaxScale, stream);

        // ── O projection (FP16 HGEMM) ──
        CudaGemm.LinearF16(
            cublasHandle, scratch.AttnOut, layer.OProj, outputF16,
            seqLen, oInput, hiddenSize, stream);
    }

    /// <summary>
    /// FP16 strided D2D extract — first <paramref name="dstWidth"/> halfs of
    /// each row of a strided source into a contiguous destination.
    /// </summary>
    private static void ExtractLatentSliceF16(
        nint src, nint dst, int seqLen, int srcStride, int dstWidth, nint stream)
    {
        long dstRowBytes = (long)dstWidth * sizeof(ushort);
        long srcRowBytes = (long)srcStride * sizeof(ushort);
        for (int t = 0; t < seqLen; t++)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                dst + (nint)((long)t * dstRowBytes),
                src + (nint)((long)t * srcRowBytes),
                (nuint)dstRowBytes, stream).ThrowOnError();
        }
    }

    /// <summary>
    /// FP16 strided D2D extract — last <paramref name="kPeWidth"/> halfs of each
    /// row of a strided source (offset by <paramref name="kvLora"/> halfs) into
    /// a contiguous destination.
    /// </summary>
    private static void ExtractKpeSliceF16(
        nint src, nint dst, int seqLen, int srcStride, int kvLora, int kPeWidth, nint stream)
    {
        long srcRowBytes = (long)srcStride * sizeof(ushort);
        long dstRowBytes = (long)kPeWidth * sizeof(ushort);
        long offsetWithinRow = (long)kvLora * sizeof(ushort);
        for (int t = 0; t < seqLen; t++)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                dst + (nint)((long)t * dstRowBytes),
                src + (nint)((long)t * srcRowBytes + (nint)offsetWithinRow),
                (nuint)dstRowBytes, stream).ThrowOnError();
        }
    }
    #endregion
}

#region MLA FP16
/// <summary>
/// Caller-owned per-call scratch buffers for <see cref="CudaMlaAttention.ForwardF16"/>.
/// Mirrors <see cref="CudaMlaScratch"/> exactly but every buffer is FP16
/// (sizeof(ushort) per element). Sized to the maximum <c>seqLen</c> the
/// caller will ever pass; reused across layers and calls.
/// </summary>
public sealed unsafe class CudaMlaScratchF16 : IDisposable
{
    private nint _normHidden;       // [seqLen, hidden]
    private nint _qLatent;          // [seqLen, qLora] (only when LoRA path)
    private nint _qLatentNorm;
    private nint _q;                // [seqLen, numHeads * qkHead]
    private nint _compressedKv;     // [seqLen, kvLora + qkRope]
    private nint _kvLatentNorm;     // [seqLen, kvLora] (extracted slice — pre-RMSNorm)
    private nint _kvLatentNormOut;  // [seqLen, kvLora]
    private nint _kvBExpanded;      // [seqLen, numHeads * (qkNope + vHead)]
    private nint _kNope;            // [seqLen, numHeads * qkNope]
    private nint _v;                // [seqLen, numHeads * vHead]
    private nint _kPe;              // [seqLen, qkRope]
    private nint _attnOut;          // [seqLen, numHeads * vHead]

    private int _capacitySeqLen;
    private int _hidden, _qLora, _qkHead, _qkRope, _qkNope, _vHead, _numHeads, _kvLora;

    /// <summary>Total allocated bytes across all FP16 scratch buffers.</summary>
    public long AllocatedBytes { get; private set; }

    internal nint NormHidden => _normHidden;
    internal nint QLatent => _qLatent;
    internal nint QLatentNorm => _qLatentNorm;
    internal nint Q => _q;
    internal nint CompressedKv => _compressedKv;
    internal nint KvLatentNorm => _kvLatentNorm;
    internal nint KvLatentNormOut => _kvLatentNormOut;
    internal nint KvBExpanded => _kvBExpanded;
    internal nint KNope => _kNope;
    internal nint V => _v;
    internal nint KPe => _kPe;
    internal nint AttnOut => _attnOut;

    /// <summary>
    /// Ensures all FP16 scratch buffers can hold <paramref name="seqLen"/> tokens
    /// for this layer's MLA shapes. Reallocates with power-of-2 growth.
    /// </summary>
    public void EnsureCapacity(int seqLen, in CudaMlaLayerWeights layer)
    {
        if (seqLen <= _capacitySeqLen
            && _hidden == layer.HiddenSize
            && _qLora == layer.QLoraRank
            && _qkNope == layer.QkNopeHeadDim
            && _qkRope == layer.QkRopeHeadDim
            && _vHead == layer.VHeadDim
            && _numHeads == layer.NumHeads
            && _kvLora == layer.KvLoraRank)
            return;

        int newCap = Math.Max(seqLen, 1);
        if (newCap > _capacitySeqLen)
        {
            newCap = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)newCap);
        }

        Free();

        _hidden = layer.HiddenSize;
        _qLora = layer.QLoraRank;
        _qkNope = layer.QkNopeHeadDim;
        _qkRope = layer.QkRopeHeadDim;
        _qkHead = _qkNope + _qkRope;
        _vHead = layer.VHeadDim;
        _numHeads = layer.NumHeads;
        _kvLora = layer.KvLoraRank;
        _capacitySeqLen = newCap;

        int qTotal = _numHeads * _qkHead;
        int kvAOut = _kvLora + _qkRope;
        int kvBOut = _numHeads * (_qkNope + _vHead);

        _normHidden = Alloc((long)newCap * _hidden);
        if (_qLora > 0)
        {
            _qLatent = Alloc((long)newCap * _qLora);
            _qLatentNorm = Alloc((long)newCap * _qLora);
        }
        _q = Alloc((long)newCap * qTotal);
        _compressedKv = Alloc((long)newCap * kvAOut);
        _kvLatentNorm = Alloc((long)newCap * _kvLora);
        _kvLatentNormOut = Alloc((long)newCap * _kvLora);
        _kvBExpanded = Alloc((long)newCap * kvBOut);
        _kNope = Alloc((long)newCap * _numHeads * _qkNope);
        _v = Alloc((long)newCap * _numHeads * _vHead);
        _kPe = Alloc((long)newCap * _qkRope);
        _attnOut = Alloc((long)newCap * _numHeads * _vHead);
    }

    private nint Alloc(long elemCount)
    {
        long bytes = elemCount * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private void Free()
    {
        FreeIfNonZero(ref _normHidden);
        FreeIfNonZero(ref _qLatent);
        FreeIfNonZero(ref _qLatentNorm);
        FreeIfNonZero(ref _q);
        FreeIfNonZero(ref _compressedKv);
        FreeIfNonZero(ref _kvLatentNorm);
        FreeIfNonZero(ref _kvLatentNormOut);
        FreeIfNonZero(ref _kvBExpanded);
        FreeIfNonZero(ref _kNope);
        FreeIfNonZero(ref _v);
        FreeIfNonZero(ref _kPe);
        FreeIfNonZero(ref _attnOut);
        AllocatedBytes = 0;
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    /// <summary>Frees every FP16 device buffer.</summary>
    public void Dispose()
    {
        Free();
        _capacitySeqLen = 0;
    }
}
#endregion
