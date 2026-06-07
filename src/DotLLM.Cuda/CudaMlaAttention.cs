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
public static unsafe class CudaMlaAttention
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
