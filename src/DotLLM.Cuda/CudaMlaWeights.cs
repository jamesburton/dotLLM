using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda;

#region MLA FP16
/// <summary>
/// Numeric precision used by an MLA layer's weights and activations. F32 is
/// the Phase A reference (matches the CPU oracle byte-for-byte algorithmically);
/// F16 is the production-target sibling that mirrors the rest of the CUDA
/// backend's GQA path (FP16 weights / activations / cache, FP32 softmax /
/// RMSNorm reduction). <c>Quantized</c> keeps raw GGUF Q4_K/Q8_0 bytes on
/// device and dequantizes per call into reused F16 scratch — the only viable
/// path for full DeepSeek-V2-Lite (the F32 / F16 fully-resident variants
/// don't fit a 12 GB GPU).
/// </summary>
public enum MlaPrecision
{
    /// <summary>F32 weights + activations + cache. Matches the CPU oracle; debug / fallback only.</summary>
    F32 = 0,
    /// <summary>F16 weights + activations + cache, FP32 reductions. Default production path.</summary>
    F16 = 1,
    /// <summary>
    /// Raw GGUF quantized weights on device; dequantize per call into reused
    /// F16 scratch. Activations + cache + reductions same as <see cref="F16"/>.
    /// </summary>
    Quantized = 2,
}
#endregion

/// <summary>
/// Per-layer GPU weight pointers for Multi-head Latent Attention (MLA, DeepSeek-V2/V3).
/// Phase 1: every projection is uploaded as F32 row-major to GPU memory and the kernel
/// path operates fully in F32 (matches the CPU oracle byte-for-byte algorithmically).
/// Quantized / FP16 paths are deferred to a follow-up agent.
/// </summary>
/// <remarks>
/// <para>
/// <b>Q topology.</b> Exactly one of the Q paths is populated per layer:
/// </para>
/// <list type="bullet">
///   <item>LoRA-factored Q (<see cref="QLoraRank"/> &gt; 0): <see cref="QAProj"/>,
///     <see cref="QALayernormWeight"/>, <see cref="QBProj"/> are non-zero;
///     <see cref="QProj"/> is zero. Used by DeepSeek-V2 full and V3.</item>
///   <item>Monolithic Q (<see cref="QLoraRank"/> == 0): <see cref="QProj"/> is non-zero;
///     <see cref="QAProj"/> / <see cref="QBProj"/> are zero, <see cref="QALayernormWeight"/>
///     is zero. Used by DeepSeek-V2-Lite.</item>
/// </list>
/// <para>
/// <b>KV topology.</b> Always LoRA-factored via <see cref="KvAProjWithMqa"/> →
/// split → RMSNorm with <see cref="KvALayernormWeight"/> on the latent half →
/// expansion via <see cref="KvBProj"/>. The MQA-shared rope-K rides along on the
/// last <c>qk_rope_head_dim</c> rows of <c>kv_a_proj_with_mqa</c>.
/// </para>
/// <para>
/// <b>O projection</b> is the existing <see cref="CudaLayerWeights.O"/> slot — the
/// loader uploads it the same way (F32 here, matching the rest of MLA Phase 1).
/// </para>
/// </remarks>
public readonly struct CudaMlaLayerWeights
{
    /// <summary>Q down-projection F32 [qLoraRank, hidden]. 0 when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QAProj;
    /// <summary>Q LoRA RMSNorm weight F32 [qLoraRank]. 0 when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QALayernormWeight;
    /// <summary>Q up-projection F32 [numHeads * qkHeadDim, qLoraRank]. 0 when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QBProj;
    /// <summary>Monolithic Q projection F32 [numHeads * qkHeadDim, hidden]. 0 when <see cref="QLoraRank"/>&gt;0.</summary>
    public readonly nint QProj;

    /// <summary>KV down-projection (MQA-shared) F32 [kvLoraRank + qkRopeHeadDim, hidden].</summary>
    public readonly nint KvAProjWithMqa;
    /// <summary>KV LoRA RMSNorm weight F32 [kvLoraRank].</summary>
    public readonly nint KvALayernormWeight;
    /// <summary>KV up-projection F32 [numHeads * (qkNopeHeadDim + vHeadDim), kvLoraRank].</summary>
    public readonly nint KvBProj;

    /// <summary>O projection F32 [hidden, numHeads * vHeadDim].</summary>
    public readonly nint OProj;

    /// <summary>Pre-attention RMSNorm weight F32 [hidden] (Llama-style input_layernorm).</summary>
    public readonly nint AttnNormWeight;

    /// <summary>Post-attention RMSNorm weight F32 [hidden] (post_attention_layernorm).</summary>
    public readonly nint FfnNormWeight;

    /// <summary>Optional O bias F32 [hidden]. 0 when absent (DeepSeek default).</summary>
    public readonly nint OBias;

    /// <summary>Number of attention heads.</summary>
    public readonly int NumHeads;
    /// <summary>Per-head non-rope Q·K dimension.</summary>
    public readonly int QkNopeHeadDim;
    /// <summary>Per-head rope Q·K dimension (must be even).</summary>
    public readonly int QkRopeHeadDim;
    /// <summary>Per-head V dimension (may differ from <see cref="QkNopeHeadDim"/>+<see cref="QkRopeHeadDim"/>).</summary>
    public readonly int VHeadDim;
    /// <summary>Q LoRA bottleneck rank. 0 selects the monolithic <see cref="QProj"/> path.</summary>
    public readonly int QLoraRank;
    /// <summary>KV LoRA bottleneck rank.</summary>
    public readonly int KvLoraRank;
    /// <summary>Model hidden size.</summary>
    public readonly int HiddenSize;

    /// <summary>
    /// Precision of the projection weights and the activations the kernel
    /// path expects. F32 weights mean the F32 attention/helpers run; F16
    /// weights mean the FP16 sibling kernels run. Norm weights stay F32
    /// in both cases (the F16 RMSNorm helper takes a FP32 weight buffer).
    /// </summary>
    public readonly MlaPrecision Precision;

    /// <summary>
    /// When <see cref="Precision"/> is <see cref="MlaPrecision.Quantized"/>:
    /// raw on-device GGUF quant bytes per projection + their quant type. The
    /// forward path dequantizes per call into reused F16 scratch (or, for
    /// decode batch=1, dispatches a quantized-GEMV kernel directly when the
    /// quant type has one). When <see cref="Precision"/> is F16 / F32, these
    /// are zero / F32 and the corresponding F16 / F32 fields above are the
    /// only views.
    /// </summary>
    /// <summary>Raw on-device GGUF quant bytes for q_a_proj. 0 unless <see cref="Precision"/>==<see cref="MlaPrecision.Quantized"/>.</summary>
    public readonly nint QAProjQuant;
    /// <summary>Quant type for <see cref="QAProjQuant"/>.</summary>
    public readonly QuantizationType QAProjQuantType;
    /// <summary>Raw on-device GGUF quant bytes for q_b_proj. 0 unless <see cref="Precision"/>==<see cref="MlaPrecision.Quantized"/>.</summary>
    public readonly nint QBProjQuant;
    /// <summary>Quant type for <see cref="QBProjQuant"/>.</summary>
    public readonly QuantizationType QBProjQuantType;
    /// <summary>Raw on-device GGUF quant bytes for monolithic q_proj. 0 unless <see cref="Precision"/>==<see cref="MlaPrecision.Quantized"/>.</summary>
    public readonly nint QProjQuant;
    /// <summary>Quant type for <see cref="QProjQuant"/>.</summary>
    public readonly QuantizationType QProjQuantType;
    /// <summary>Raw on-device GGUF quant bytes for kv_a_proj_with_mqa. 0 unless <see cref="Precision"/>==<see cref="MlaPrecision.Quantized"/>.</summary>
    public readonly nint KvAProjWithMqaQuant;
    /// <summary>Quant type for <see cref="KvAProjWithMqaQuant"/>.</summary>
    public readonly QuantizationType KvAProjWithMqaQuantType;
    /// <summary>Raw on-device GGUF quant bytes for kv_b_proj. 0 unless <see cref="Precision"/>==<see cref="MlaPrecision.Quantized"/>.</summary>
    public readonly nint KvBProjQuant;
    /// <summary>Quant type for <see cref="KvBProjQuant"/>.</summary>
    public readonly QuantizationType KvBProjQuantType;
    /// <summary>Raw on-device GGUF quant bytes for o_proj. 0 unless <see cref="Precision"/>==<see cref="MlaPrecision.Quantized"/>.</summary>
    public readonly nint OProjQuant;
    /// <summary>Quant type for <see cref="OProjQuant"/>.</summary>
    public readonly QuantizationType OProjQuantType;

    /// <summary>
    /// Constructs a fully-populated F32 MLA layer weight bundle (back-compat
    /// constructor — matches the original Phase A signature exactly).
    /// </summary>
    public CudaMlaLayerWeights(
        nint qAProj, nint qALayernormWeight, nint qBProj, nint qProj,
        nint kvAProjWithMqa, nint kvALayernormWeight, nint kvBProj,
        nint oProj, nint attnNormWeight, nint ffnNormWeight, nint oBias,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank, int hiddenSize)
        : this(qAProj, qALayernormWeight, qBProj, qProj,
               kvAProjWithMqa, kvALayernormWeight, kvBProj,
               oProj, attnNormWeight, ffnNormWeight, oBias,
               numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
               qLoraRank, kvLoraRank, hiddenSize, MlaPrecision.F32)
    {
    }

    /// <summary>
    /// Constructs a fully-populated MLA layer weight bundle with explicit
    /// <paramref name="precision"/>. F32 selects the original Phase A kernels;
    /// F16 selects the FP16 sibling kernels (default production path).
    /// </summary>
    public CudaMlaLayerWeights(
        nint qAProj, nint qALayernormWeight, nint qBProj, nint qProj,
        nint kvAProjWithMqa, nint kvALayernormWeight, nint kvBProj,
        nint oProj, nint attnNormWeight, nint ffnNormWeight, nint oBias,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank, int hiddenSize,
        MlaPrecision precision)
        : this(qAProj, qALayernormWeight, qBProj, qProj,
               kvAProjWithMqa, kvALayernormWeight, kvBProj,
               oProj, attnNormWeight, ffnNormWeight, oBias,
               numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
               qLoraRank, kvLoraRank, hiddenSize, precision,
               qAProjQuant: 0, qAProjQuantType: QuantizationType.F32,
               qBProjQuant: 0, qBProjQuantType: QuantizationType.F32,
               qProjQuant: 0, qProjQuantType: QuantizationType.F32,
               kvAProjWithMqaQuant: 0, kvAProjWithMqaQuantType: QuantizationType.F32,
               kvBProjQuant: 0, kvBProjQuantType: QuantizationType.F32,
               oProjQuant: 0, oProjQuantType: QuantizationType.F32)
    {
    }

    /// <summary>
    /// Full ctor — used by the quantized loader path. Norm weights + bias stay
    /// F32 device buffers (the FP16 RMSNorm helper takes a FP32 weight). The
    /// non-quant projection pointers should be 0 when <paramref name="precision"/>
    /// is <see cref="MlaPrecision.Quantized"/> — the forward path reads from
    /// <c>*Quant</c> + <c>*QuantType</c> instead.
    /// </summary>
    public CudaMlaLayerWeights(
        nint qAProj, nint qALayernormWeight, nint qBProj, nint qProj,
        nint kvAProjWithMqa, nint kvALayernormWeight, nint kvBProj,
        nint oProj, nint attnNormWeight, nint ffnNormWeight, nint oBias,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank, int hiddenSize,
        MlaPrecision precision,
        nint qAProjQuant, QuantizationType qAProjQuantType,
        nint qBProjQuant, QuantizationType qBProjQuantType,
        nint qProjQuant, QuantizationType qProjQuantType,
        nint kvAProjWithMqaQuant, QuantizationType kvAProjWithMqaQuantType,
        nint kvBProjQuant, QuantizationType kvBProjQuantType,
        nint oProjQuant, QuantizationType oProjQuantType)
    {
        QAProj = qAProj;
        QALayernormWeight = qALayernormWeight;
        QBProj = qBProj;
        QProj = qProj;
        KvAProjWithMqa = kvAProjWithMqa;
        KvALayernormWeight = kvALayernormWeight;
        KvBProj = kvBProj;
        OProj = oProj;
        AttnNormWeight = attnNormWeight;
        FfnNormWeight = ffnNormWeight;
        OBias = oBias;
        NumHeads = numHeads;
        QkNopeHeadDim = qkNopeHeadDim;
        QkRopeHeadDim = qkRopeHeadDim;
        VHeadDim = vHeadDim;
        QLoraRank = qLoraRank;
        KvLoraRank = kvLoraRank;
        HiddenSize = hiddenSize;
        Precision = precision;
        QAProjQuant = qAProjQuant; QAProjQuantType = qAProjQuantType;
        QBProjQuant = qBProjQuant; QBProjQuantType = qBProjQuantType;
        QProjQuant = qProjQuant; QProjQuantType = qProjQuantType;
        KvAProjWithMqaQuant = kvAProjWithMqaQuant; KvAProjWithMqaQuantType = kvAProjWithMqaQuantType;
        KvBProjQuant = kvBProjQuant; KvBProjQuantType = kvBProjQuantType;
        OProjQuant = oProjQuant; OProjQuantType = oProjQuantType;
    }

    /// <summary>Total Q vector elements per token: <c>numHeads * (qkNope + qkRope)</c>.</summary>
    public int QTotalElems => NumHeads * (QkNopeHeadDim + QkRopeHeadDim);
    /// <summary>Compressed KV vector elements per token: <c>kvLoraRank + qkRope</c>.</summary>
    public int KvAOutElems => KvLoraRank + QkRopeHeadDim;
    /// <summary>Expanded per-head KV elements per token: <c>numHeads * (qkNope + vHead)</c>.</summary>
    public int KvBOutElems => NumHeads * (QkNopeHeadDim + VHeadDim);
    /// <summary>O projection input dim: <c>numHeads * vHead</c>.</summary>
    public int OInputDim => NumHeads * VHeadDim;
}

/// <summary>
/// Loader that uploads a per-layer <see cref="MlaLayerWeights"/> bundle to GPU as F32.
/// CPU-side weights are already F32 (the safetensors loader upcasts BF16/F16 via
/// <c>ResolveLinearAsF32</c>). We allocate device buffers and synchronously copy.
/// </summary>
/// <remarks>
/// <para>
/// <b>Phase 1 simplification.</b> No FP16 conversion, no quantized GEMV, no packed
/// fusion. The MLA forward path runs entirely in F32 to match the CPU oracle
/// numerically; downstream agents will widen this with quantized weight support
/// once correctness is locked in.
/// </para>
/// <para>
/// <b>Lifetime.</b> All allocations are added to the supplied <c>allocs</c> list,
/// which the parent <see cref="CudaWeights"/> owns and frees on dispose.
/// </para>
/// </remarks>
internal static unsafe class CudaMlaWeightsLoader
{
    /// <summary>
    /// Uploads a single MLA layer's projections to F32 device buffers.
    /// </summary>
    /// <param name="cpuLayer">Per-layer weight bundle from the safetensors loader.</param>
    /// <param name="hiddenSize">Model hidden dimension.</param>
    /// <param name="allocs">Allocation list to extend (caller owns + frees on dispose).</param>
    /// <returns>Populated <see cref="CudaMlaLayerWeights"/> with device pointers.</returns>
    public static CudaMlaLayerWeights LoadLayer(
        in TransformerLayerWeights cpuLayer, int hiddenSize, List<nint> allocs)
    {
        var mla = cpuLayer.Mla
            ?? throw new InvalidOperationException(
                "CudaMlaWeightsLoader.LoadLayer called with non-MLA layer.");

        int qTotal = mla.NumHeads * (mla.QkNopeHeadDim + mla.QkRopeHeadDim);
        int kvAOut = mla.KvLoraRank + mla.QkRopeHeadDim;
        int kvBOut = mla.NumHeads * (mla.QkNopeHeadDim + mla.VHeadDim);
        int oInput = mla.NumHeads * mla.VHeadDim;

        // Q path (one of two — the unused side stays at 0).
        nint qAProj = 0, qBProj = 0, qProj = 0, qANorm = 0;
        if (mla.QLoraRank > 0)
        {
            qAProj = UploadF32(mla.QAProj, (long)mla.QLoraRank * hiddenSize, allocs);
            qANorm = UploadF32Array(mla.QALayernormWeight!, allocs);
            qBProj = UploadF32(mla.QBProj, (long)qTotal * mla.QLoraRank, allocs);
        }
        else
        {
            qProj = UploadF32(mla.QProj, (long)qTotal * hiddenSize, allocs);
        }

        // KV path (always LoRA-factored).
        nint kvAProj = UploadF32(mla.KvAProjWithMqa, (long)kvAOut * hiddenSize, allocs);
        nint kvANorm = UploadF32Array(mla.KvALayernormWeight, allocs);
        nint kvBProj = UploadF32(mla.KvBProj, (long)kvBOut * mla.KvLoraRank, allocs);

        // O projection — lives in the existing OWeight slot on the CPU layer
        // and was loaded as F32 by the AttentionTensorLoader MLA path.
        nint oProj = UploadF32(cpuLayer.OWeight, (long)hiddenSize * oInput, allocs);

        // Norm weights (F32 throughout).
        nint attnNorm = UploadF32Array(cpuLayer.AttnNormWeight, allocs);
        nint ffnNorm = UploadF32Array(cpuLayer.FfnNormWeight, allocs);

        nint oBias = cpuLayer.OBias is float[] arr ? UploadF32Array(arr, allocs) : (nint)0;

        return new CudaMlaLayerWeights(
            qAProj, qANorm, qBProj, qProj,
            kvAProj, kvANorm, kvBProj,
            oProj, attnNorm, ffnNorm, oBias,
            mla.NumHeads, mla.QkNopeHeadDim, mla.QkRopeHeadDim, mla.VHeadDim,
            mla.QLoraRank, mla.KvLoraRank, hiddenSize);
    }

    /// <summary>
    /// Allocates a device buffer of <c>elementCount * sizeof(float)</c> bytes and
    /// copies host F32 data into it. Returns the device pointer; appends to
    /// <paramref name="allocs"/>.
    /// </summary>
    private static nint UploadF32(nint hostPtr, long elementCount, List<nint> allocs)
    {
        if (hostPtr == 0)
            throw new InvalidOperationException("UploadF32 called with null host pointer.");
        long bytes = elementCount * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        CudaDriverApi.cuMemcpyHtoD_v2(devPtr, hostPtr, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    /// <summary>
    /// Uploads a managed <c>float[]</c> to GPU as F32.
    /// </summary>
    private static nint UploadF32Array(float[] data, List<nint> allocs)
    {
        long bytes = (long)data.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        fixed (float* p = data)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    #region MLA FP16 — uploader path
    /// <summary>
    /// Uploads a single MLA layer's projections to FP16 device buffers (norm
    /// weights stay F32). Mirrors <see cref="LoadLayer"/> but down-casts each
    /// projection tensor to FP16 on the host before HtoD copy. Memory is
    /// roughly half the F32 path; throughput is materially better since
    /// cuBLAS HGEMM and the FP16 attention kernel are bandwidth-bound at
    /// decode batch=1.
    /// </summary>
    /// <param name="cpuLayer">Per-layer weight bundle from the safetensors loader.</param>
    /// <param name="hiddenSize">Model hidden dimension.</param>
    /// <param name="allocs">Allocation list to extend (caller owns + frees on dispose).</param>
    /// <returns>Populated <see cref="CudaMlaLayerWeights"/> with FP16 device pointers and <c>Precision = F16</c>.</returns>
    public static CudaMlaLayerWeights LoadLayerF16(
        in TransformerLayerWeights cpuLayer, int hiddenSize, List<nint> allocs)
    {
        var mla = cpuLayer.Mla
            ?? throw new InvalidOperationException(
                "CudaMlaWeightsLoader.LoadLayerF16 called with non-MLA layer.");

        int qTotal = mla.NumHeads * (mla.QkNopeHeadDim + mla.QkRopeHeadDim);
        int kvAOut = mla.KvLoraRank + mla.QkRopeHeadDim;
        int kvBOut = mla.NumHeads * (mla.QkNopeHeadDim + mla.VHeadDim);
        int oInput = mla.NumHeads * mla.VHeadDim;

        // Q path
        nint qAProj = 0, qBProj = 0, qProj = 0, qANorm = 0;
        if (mla.QLoraRank > 0)
        {
            qAProj = UploadF32AsF16(mla.QAProj, (long)mla.QLoraRank * hiddenSize, allocs);
            qANorm = UploadF32Array(mla.QALayernormWeight!, allocs);
            qBProj = UploadF32AsF16(mla.QBProj, (long)qTotal * mla.QLoraRank, allocs);
        }
        else
        {
            qProj = UploadF32AsF16(mla.QProj, (long)qTotal * hiddenSize, allocs);
        }

        // KV path (always LoRA-factored).
        nint kvAProj = UploadF32AsF16(mla.KvAProjWithMqa, (long)kvAOut * hiddenSize, allocs);
        nint kvANorm = UploadF32Array(mla.KvALayernormWeight, allocs);
        nint kvBProj = UploadF32AsF16(mla.KvBProj, (long)kvBOut * mla.KvLoraRank, allocs);

        // O projection — FP16.
        nint oProj = UploadF32AsF16(cpuLayer.OWeight, (long)hiddenSize * oInput, allocs);

        // Norm weights stay F32 (the FP16 RMSNorm kernel takes a FP32 weight buffer
        // — saves the FP16 round-trip and keeps the multiplicative scale precise).
        nint attnNorm = UploadF32Array(cpuLayer.AttnNormWeight, allocs);
        nint ffnNorm = UploadF32Array(cpuLayer.FfnNormWeight, allocs);

        nint oBias = cpuLayer.OBias is float[] arr ? UploadF32Array(arr, allocs) : (nint)0;

        return new CudaMlaLayerWeights(
            qAProj, qANorm, qBProj, qProj,
            kvAProj, kvANorm, kvBProj,
            oProj, attnNorm, ffnNorm, oBias,
            mla.NumHeads, mla.QkNopeHeadDim, mla.QkRopeHeadDim, mla.VHeadDim,
            mla.QLoraRank, mla.KvLoraRank, hiddenSize,
            MlaPrecision.F16);
    }

    /// <summary>
    /// Loads a single MLA layer's projections to GPU as raw GGUF quantized
    /// bytes (zero-copy upload from the CPU mmap). Used when the source layer
    /// has populated raw quant views (<see cref="MlaLayerWeights.HasRawQuantView"/>)
    /// — i.e. GGUF-loaded DeepSeek-V2 weights. Norm weights stay F32 (same
    /// as the FP16 path). The forward path (CudaMlaAttention.ForwardF16
    /// quantized branch) dequantizes per call into reused F16 scratch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Memory budget</b> at DeepSeek-V2-Lite scale (Q4_K_M):
    /// q_proj=2 MB, kv_a=0.4 MB, kv_b=0.6 MB, o_proj=2 MB per layer ⇒
    /// ~5 MB raw + 64 KB norm weights ⇒ ~140 MB total across 27 layers.
    /// Compared to ~1.4 GB for F16 fully-resident or ~2.8 GB F32, this is
    /// the only path that fits a 12 GB consumer GPU after the MoE expert
    /// block (separate path, see <see cref="CudaMoeWeightsLoader"/> +
    /// task #10) lands too.
    /// </para>
    /// </remarks>
    /// <param name="cpuLayer">Per-layer weight bundle (must have raw quant views populated by GGUF loader).</param>
    /// <param name="hiddenSize">Model hidden dimension.</param>
    /// <param name="oQuantType">O-projection quant type (lives in TransformerLayerWeights.OQuantType, not on Mla).</param>
    /// <param name="allocs">Allocation list (caller owns + frees on dispose).</param>
    public static CudaMlaLayerWeights LoadLayerQuant(
        in TransformerLayerWeights cpuLayer, int hiddenSize,
        QuantizationType oQuantType, List<nint> allocs)
    {
        var mla = cpuLayer.Mla
            ?? throw new InvalidOperationException(
                "CudaMlaWeightsLoader.LoadLayerQuant called with non-MLA layer.");
        if (!mla.HasRawQuantView)
            throw new InvalidOperationException(
                "CudaMlaWeightsLoader.LoadLayerQuant requires the source MlaLayerWeights to carry " +
                "raw GGUF quant views (the GGUF loader populates these alongside the F32 dequants).");

        int qTotal = mla.NumHeads * (mla.QkNopeHeadDim + mla.QkRopeHeadDim);
        int kvAOut = mla.KvLoraRank + mla.QkRopeHeadDim;
        int kvBOut = mla.NumHeads * (mla.QkNopeHeadDim + mla.VHeadDim);
        int oInput = mla.NumHeads * mla.VHeadDim;

        // Q path — upload raw GGUF quant bytes for whichever Q variant the model uses.
        nint qAQuant = 0, qBQuant = 0, qQuant = 0, qANorm = 0;
        QuantizationType qAQt = QuantizationType.F32, qBQt = QuantizationType.F32, qQt = QuantizationType.F32;
        if (mla.QLoraRank > 0)
        {
            qAQuant = UploadQuantBytes(mla.QAProjRaw, mla.QLoraRank, hiddenSize, mla.QAProjRawQt, allocs);
            qAQt = mla.QAProjRawQt;
            qANorm = UploadF32Array(mla.QALayernormWeight!, allocs);
            qBQuant = UploadQuantBytes(mla.QBProjRaw, qTotal, mla.QLoraRank, mla.QBProjRawQt, allocs);
            qBQt = mla.QBProjRawQt;
        }
        else
        {
            qQuant = UploadQuantBytes(mla.QProjRaw, qTotal, hiddenSize, mla.QProjRawQt, allocs);
            qQt = mla.QProjRawQt;
        }

        // KV path
        nint kvAQuant = UploadQuantBytes(mla.KvAProjWithMqaRaw, kvAOut, hiddenSize, mla.KvAProjWithMqaRawQt, allocs);
        QuantizationType kvAQt = mla.KvAProjWithMqaRawQt;
        nint kvANorm = UploadF32Array(mla.KvALayernormWeight, allocs);
        nint kvBQuant = UploadQuantBytes(mla.KvBProjRaw, kvBOut, mla.KvLoraRank, mla.KvBProjRawQt, allocs);
        QuantizationType kvBQt = mla.KvBProjRawQt;

        // O projection — raw quant from cpuLayer.OWeight + cpuLayer.OQuantType.
        // Output dim is hidden, input dim is numHeads * vHeadDim.
        nint oQuant = UploadQuantBytes(cpuLayer.OWeight, hiddenSize, oInput, oQuantType, allocs);

        // Norms + bias stay F32 (same as the F16 path).
        nint attnNorm = UploadF32Array(cpuLayer.AttnNormWeight, allocs);
        nint ffnNorm = UploadF32Array(cpuLayer.FfnNormWeight, allocs);
        nint oBias = cpuLayer.OBias is float[] biasArr ? UploadF32Array(biasArr, allocs) : (nint)0;

        // The F16 / F32 projection slots are zero — the forward path reads
        // *Quant + *QuantType when Precision == Quantized.
        return new CudaMlaLayerWeights(
            qAProj: 0, qALayernormWeight: qANorm,
            qBProj: 0, qProj: 0,
            kvAProjWithMqa: 0, kvALayernormWeight: kvANorm,
            kvBProj: 0,
            oProj: 0, attnNormWeight: attnNorm, ffnNormWeight: ffnNorm, oBias: oBias,
            numHeads: mla.NumHeads,
            qkNopeHeadDim: mla.QkNopeHeadDim,
            qkRopeHeadDim: mla.QkRopeHeadDim,
            vHeadDim: mla.VHeadDim,
            qLoraRank: mla.QLoraRank,
            kvLoraRank: mla.KvLoraRank,
            hiddenSize: hiddenSize,
            precision: MlaPrecision.Quantized,
            qAProjQuant: qAQuant, qAProjQuantType: qAQt,
            qBProjQuant: qBQuant, qBProjQuantType: qBQt,
            qProjQuant: qQuant, qProjQuantType: qQt,
            kvAProjWithMqaQuant: kvAQuant, kvAProjWithMqaQuantType: kvAQt,
            kvBProjQuant: kvBQuant, kvBProjQuantType: kvBQt,
            oProjQuant: oQuant, oProjQuantType: oQuantType);
    }

    /// <summary>
    /// Uploads raw GGUF quantized bytes to GPU (zero-copy from CPU mmap into
    /// a fresh device buffer). Byte size is computed from the GGUF
    /// row-byte-stride convention (<see cref="DotLLM.Cpu.Kernels.Dequantize.RowByteSize"/>).
    /// </summary>
    private static unsafe nint UploadQuantBytes(nint hostPtr, int outputDim, int inputDim,
                                                 QuantizationType qt, List<nint> allocs)
    {
        if (hostPtr == 0)
            throw new InvalidOperationException(
                $"UploadQuantBytes called with null host pointer (qt={qt}, dims={outputDim}x{inputDim}).");
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(inputDim, qt);
        long bytes = rowBytes * outputDim;
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        CudaDriverApi.cuMemcpyHtoD_v2(devPtr, hostPtr, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    /// <summary>
    /// Down-casts an F32 host tensor to FP16 (via Half) and uploads it to a new
    /// device buffer. Returns the device pointer; appends to
    /// <paramref name="allocs"/>. Memory cost: half of the F32 equivalent.
    /// </summary>
    private static nint UploadF32AsF16(nint hostF32Ptr, long elementCount, List<nint> allocs)
    {
        if (hostF32Ptr == 0)
            throw new InvalidOperationException("UploadF32AsF16 called with null host pointer.");
        long bytes = elementCount * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);

        // Stage in a Half[] on the heap; cast each element. For weight uploads
        // (one-shot at model load), the GC pressure is acceptable — we don't
        // hot-path this. Caller already keeps the F32 tensor mapped for the
        // duration of the load, so the source is stable.
        var staging = new Half[elementCount];
        unsafe
        {
            float* src = (float*)hostF32Ptr;
            for (long i = 0; i < elementCount; i++)
                staging[i] = (Half)src[i];
            fixed (Half* dst = staging)
                CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)dst, (nuint)bytes).ThrowOnError();
        }
        return devPtr;
    }
    #endregion
}
