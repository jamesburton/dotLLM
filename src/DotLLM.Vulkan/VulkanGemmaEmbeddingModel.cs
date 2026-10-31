using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// EmbeddingGemma-300M (<c>gemma-embedding</c>: a bidirectional Gemma 3 tower with two bias-free dense heads after mean pooling) on the Vulkan
/// backend (issue #1472).
/// </summary>
/// <remarks>
/// <para>The graph is the CPU <c>TransformerModel</c> bidirectional path step for step: <c>x = embed * sqrt(h)</c>; per layer
/// <c>x += post_attn_norm(o_proj(attn(rope(qk_norm(q, k)), v)))</c> over <c>attn_norm(x)</c>, then
/// <c>x += post_ffw_norm(down(gelu_tanh(gate(n)) * up(n)))</c> over <c>ffn_norm(x)</c>; final <c>output_norm</c>; mean pooling; <c>dense_2</c> (768 to 3072)
/// then <c>dense_3</c> (3072 to 768). Attention is one <see cref="AttentionMaskMode.SymmetricWindow"/> coopmat flash dispatch per document
/// (5 of 6 layers window 512 = keys <c>q-256 .. q+256</c>, every sixth layer full bidirectional), GQA 3:1, head_dim 256, dual RoPE theta.</para>
/// <para><b>Activations.</b> Weights (Q8_0 / F16 / F32) are resident and read in place by the same blocked coopmat GEMMs as the BERT encoder (no expansion).
/// Every stored activation is F16 (QKV, attention output, FFN, GEMM outputs, normalised rows, pooled rows) with F32 accumulators, <b>except the residual
/// stream, which is BF16</b>: it reaches |x| ~ 1.5e5 in the last layers (measured, docs/EMBEDDINGS.md), which F16 (65504) cannot hold, and Google documents
/// that EmbeddingGemma does not support F16 activations. The FFN down-projection output peaks at ~6.2e4, just under the F16 ceiling; an overflow sentinel
/// in the post-norm kernel makes the forward fail loudly (<see cref="InvalidOperationException"/>) instead of returning garbage.</para>
/// <para>Pooling: only mean pooling (the model's declared pooling) is supported, because the dense heads were trained after it; the heads run on the
/// device over the pooled rows.</para>
/// </remarks>
public sealed unsafe class VulkanGemmaEmbeddingModel : IModel, IEmbeddingModel
{
    private sealed class Layer
    {
        public required Weight Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
        public required VulkanDevice.Buffer AttnNorm, QNorm, KNorm, PostAttnNorm, FfnNorm, PostFfnNorm;
        public int Window;          // 0 = full bidirectional layer
        public float RopeTheta;
    }

    private readonly record struct Weight(VulkanDevice.Buffer Buf, EncoderWeightFormat Format);

    /// <summary>Documents per packed forward (the flash kernel's sub-range descriptor cache holds 256 distinct tuples).</summary>
    internal const int MaxDocsPerPack = 128;

    /// <summary>Token budget of one packed forward. Env <c>DOTLLM_EMBED_PACK_TOKENS</c> overrides (default 1024).</summary>
    internal static int PackTokenBudget =>
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_EMBED_PACK_TOKENS"), out int v) && v > 0 ? v : 1024;

    /// <summary>Test seam: attend across document boundaries (one dispatch over the concatenation).</summary>
    internal bool PackLeakMutant { get; set; }

    /// <summary>Test seam: causal instead of bidirectional attention.</summary>
    internal bool CausalMutant { get; set; }

    /// <summary>Test seam: ignore the sliding window (every layer full bidirectional); must be detected past ~257 tokens.</summary>
    internal bool WindowIgnoredMutant { get; set; }

    /// <summary>Test seam: skip the QK-norm.</summary>
    internal bool NoQkNormMutant { get; set; }

    /// <summary>Test seam: return the final-norm hidden state of every token (pre dense heads) from <see cref="ForwardHidden"/>.</summary>
    internal bool RawHiddenForTests { get; set; }

    private readonly VulkanDevice _device;
    private readonly bool _ownsDevice;
    private readonly GgufFile _gguf;
    private readonly Layer[] _layers;
    private readonly nint _tokenEmbd;
    private readonly QuantizationType _tokenEmbdQt;
    private readonly float _embScale, _eps;
    private readonly int _h, _ff, _nh, _nkv, _hd, _qDim, _kvDim, _ropeDim, _cap;
    private readonly bool _ropeNeox;
    private readonly float _attnScale;
    private readonly Weight _d2, _d3;
    private readonly bool _hasHeads;
    private readonly int _d2Out;
    private readonly VulkanDevice.Buffer _outNorm;

    private readonly MatMulF16GemmCoopmatKernel _gemm;
    private readonly Dictionary<EncoderWeightFormat, MatMulEncoderGemmKernel> _encGemm = new();
    private readonly VulkanFlashAttentionCoopmatKernel _flash;
    private readonly SimpleComputeKernel _preNorm, _postNorm, _qkNorm, _rope, _geglu, _poolMean;

    // Activation scratch. _res is the BF16 residual stream; everything else F16.
    private readonly VulkanDevice.Buffer _res, _hid2, _q, _k, _v, _attn, _tmp, _gate, _up, _pos, _seg, _flag, _outBuf;
    private readonly ushort* _hostEmb;   // [_cap, h] BF16 words
    private readonly VulkanDevice.SubmitContext _submit;
    private readonly List<IDisposable> _owned;
    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes { get; }

    /// <inheritdoc/>
    public PoolingType? DeclaredPoolingType => Config.PoolingType;

    /// <inheritdoc/>
    public bool AppendsEosForLastPooling => false;

    private VulkanGemmaEmbeddingModel(VulkanDevice device, bool ownsDevice, GgufFile gguf, ModelConfig config, string spvDir)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        _gguf = gguf;
        Config = config;
        if (config.Architecture != DotLLM.Core.Configuration.Architecture.GemmaEmbedding)
            throw new NotSupportedException($"VulkanGemmaEmbeddingModel serves Architecture.GemmaEmbedding (got {config.Architecture}).");
        _h = config.HiddenSize; _ff = config.IntermediateSize; _nh = config.NumAttentionHeads; _nkv = config.NumKvHeads; _hd = config.HeadDim;
        _qDim = _nh * _hd; _kvDim = _nkv * _hd;
        _eps = config.NormEpsilon;
        _embScale = config.EmbeddingScale ?? 1.0f;
        _attnScale = config.AttentionScoreScale(_hd);
        if (_h % 64 != 0 || _ff % 64 != 0 || _qDim % 64 != 0)
            throw new NotSupportedException($"Vulkan gemma-embedding needs hidden, ffn and q width divisible by 64 (got {_h}, {_ff}, {_qDim}).");
        if (_nh % _nkv != 0 || _hd > 256 || _hd <= 128 || _hd % 16 != 0)
            throw new NotSupportedException($"Vulkan gemma-embedding supports the head_dim-256 flash pipeline (got {_nh} heads, {_nkv} kv heads, head_dim {_hd}).");
        if ((long)_qDim * sizeof(ushort) % 256 != 0 || (long)_kvDim * sizeof(ushort) % 256 != 0)
            throw new NotSupportedException("Vulkan gemma-embedding: Q/K row widths must be multiples of 128 elements (sub-range descriptor alignment).");
        if (!device.HasCooperativeMatrix)
            throw new NotSupportedException("The Vulkan gemma-embedding encoder needs VK_KHR_cooperative_matrix (F16 coopmat GEMM + flash attention).");
        _flash = VulkanFlashAttentionCoopmatKernel.TryCreateF16Activations(device, spvDir)
            ?? throw new NotSupportedException("The Vulkan gemma-embedding encoder needs the F16-activation coopmat flash-attention kernel (not available on this device / build).");
        if (_flash.SupportedMaxHeadDim < _hd)
            throw new NotSupportedException($"The flash kernel's head_dim-256 pipeline is unavailable (supports up to {_flash.SupportedMaxHeadDim}).");

        var rope = config.RoPEConfig ?? throw new InvalidDataException("gemma-embedding config has no RoPE configuration.");
        var grope = config.GlobalRoPEConfig ?? rope;
        if (grope.ScalingType != RoPEScalingType.None && grope.ScalingFactor != 1.0f)
            throw new NotSupportedException($"Vulkan gemma-embedding: global RoPE scaling ({grope.ScalingType} x{grope.ScalingFactor}) is not implemented; use --device cpu.");
        _ropeDim = rope.DimensionCount > 0 ? rope.DimensionCount : _hd;
        if (_ropeDim != _hd) throw new NotSupportedException("Vulkan gemma-embedding: partial rotary dimensions are not implemented.");
        _ropeNeox = rope.Type == RoPEType.NeoX;

        _cap = Math.Max(PackTokenBudget, config.MaxSequenceLength);
        var dataBase = new GgufDataBase(gguf);
        var t = gguf.TensorsByName;
        Nvfp4Support.RejectUnappliedScales(t, consumedScales: null, "The Vulkan gemma-embedding loader");
        if (t.ContainsKey("rope_freqs.weight"))
            throw new NotSupportedException("Vulkan gemma-embedding: rope_freqs.weight is not implemented.");

        GgufTensorDescriptor Req(string name)
            => t.TryGetValue(name, out var d) ? d : throw new InvalidDataException($"gemma-embedding GGUF is missing required tensor '{name}'.");

        var toDispose = new List<IDisposable>();
        VulkanDevice.Buffer Norm(string name, int n)
        {
            var d = Req(name);
            var r = new float[n];
            Dequantize.ToFloat32(dataBase.Of(d), n, d.QuantizationType, r);
            var b = device.AllocateDeviceLocal(Math.Max(16, (long)n * sizeof(float)));
            toDispose.Add(b);
            device.Upload(r, b);
            return b;
        }
        VulkanDevice.Buffer Scratch(long halves)
        {
            var b = device.AllocateDeviceLocal(Math.Max(16, halves * sizeof(ushort)));
            toDispose.Add(b);
            return b;
        }
        Weight WeightIn(string name, int expectK, int expectM)
        {
            var d = Req(name);
            var fmt = d.QuantizationType switch
            {
                QuantizationType.F16 => EncoderWeightFormat.F16,
                QuantizationType.Q8_0 => EncoderWeightFormat.Q8_0,
                QuantizationType.F32 => EncoderWeightFormat.F32,
                _ => throw new NotSupportedException(
                    $"Vulkan gemma-embedding: tensor '{name}' is {d.QuantizationType}; F16, Q8_0 and F32 weights are read in place, other formats run with --device cpu."),
            };
            if (d.Shape[0] != expectK || d.Shape[1] != expectM)
                throw new InvalidDataException($"Tensor '{name}' has shape [{d.Shape[0]}, {d.Shape[1]}], expected [{expectK}, {expectM}].");
            if (!MatMulEncoderGemmKernel.SupportsK(fmt, expectK))
                throw new NotSupportedException($"Vulkan gemma-embedding: '{name}' ({fmt}) has K={expectK}, which the in-place kernel cannot read (Q8_0 needs K % 64 == 0).");
            if (fmt != EncoderWeightFormat.F16 && !MatMulEncoderGemmKernel.IsSupportedOn(device, spvDir, fmt))
                throw new NotSupportedException(
                    $"Vulkan gemma-embedding: {fmt} weights are read in place by the blocked 128x128 coopmat tile, which needs a wave64 cooperative-matrix device (subgroup size {device.SubgroupSize}). Use an F16 GGUF or --device cpu.");
            long bytes = MatMulEncoderGemmKernel.WeightBytes(fmt, expectM, expectK);
            var b = device.AllocateDeviceLocal((bytes + 3) & ~3L);
            toDispose.Add(b);
            device.Upload(new ReadOnlySpan<byte>((byte*)dataBase.Of(d), checked((int)bytes)), b);
            return new Weight(b, fmt);
        }

        try
        {
            var tokDesc = Req("token_embd.weight");
            if (tokDesc.Shape[0] != _h) throw new InvalidDataException($"token_embd.weight row width {tokDesc.Shape[0]} != hidden size {_h}.");
            _tokenEmbd = dataBase.Of(tokDesc); _tokenEmbdQt = tokDesc.QuantizationType;
            _outNorm = Norm("output_norm.weight", _h);

            _layers = new Layer[config.NumLayers];
            for (int i = 0; i < _layers.Length; i++)
            {
                string p = $"blk.{i}.";
                foreach (var bias in new[] { "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias", "ffn_up.bias", "ffn_down.bias", "ffn_gate.bias" })
                    if (t.ContainsKey(p + bias)) throw new NotSupportedException($"Vulkan gemma-embedding: layer {i} has a bias ({bias}), which the Gemma tower does not use.");
                int window = config.SlidingWindowOf(i) ?? 0;
                _layers[i] = new Layer
                {
                    Wq = WeightIn(p + "attn_q.weight", _h, _qDim), Wk = WeightIn(p + "attn_k.weight", _h, _kvDim), Wv = WeightIn(p + "attn_v.weight", _h, _kvDim),
                    Wo = WeightIn(p + "attn_output.weight", _qDim, _h),
                    Wgate = WeightIn(p + "ffn_gate.weight", _h, _ff), Wup = WeightIn(p + "ffn_up.weight", _h, _ff), Wdown = WeightIn(p + "ffn_down.weight", _ff, _h),
                    AttnNorm = Norm(p + "attn_norm.weight", _h), QNorm = Norm(p + "attn_q_norm.weight", _hd), KNorm = Norm(p + "attn_k_norm.weight", _hd),
                    PostAttnNorm = Norm(p + "post_attention_norm.weight", _h), FfnNorm = Norm(p + "ffn_norm.weight", _h), PostFfnNorm = Norm(p + "post_ffw_norm.weight", _h),
                    Window = window,
                    RopeTheta = config.IsFullAttentionLayer(i) ? grope.Theta : rope.Theta,
                };
            }
            if (t.ContainsKey("dense_2.weight") && t.ContainsKey("dense_3.weight"))
            {
                _d2Out = Req("dense_2.weight").Shape[1];
                _d2 = WeightIn("dense_2.weight", _h, _d2Out);
                _d3 = WeightIn("dense_3.weight", _d2Out, _h);
                _hasHeads = true;
                if ((long)MaxDocsPerPack * _d2Out > (long)_cap * _ff)
                    throw new NotSupportedException($"Vulkan gemma-embedding: dense head width {_d2Out} exceeds the FFN scratch.");
            }

            _gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir, F16GemmCoopmatVariant.SelectF16ActFor(device, spvDir));
            var formats = new HashSet<EncoderWeightFormat>();
            foreach (var l in _layers) foreach (var w in new[] { l.Wq, l.Wk, l.Wv, l.Wo, l.Wgate, l.Wup, l.Wdown }) formats.Add(w.Format);
            if (_hasHeads) { formats.Add(_d2.Format); formats.Add(_d3.Format); }
            foreach (var f in formats)
                if (f != EncoderWeightFormat.F16) _encGemm[f] = MatMulEncoderGemmKernel.Create(device, spvDir, f);
            _preNorm = SimpleComputeKernel.Create(device, spvDir, "gemma_rmsnorm_bf16_f16", buffers: 3, pushWords: 2);
            _postNorm = SimpleComputeKernel.Create(device, spvDir, "gemma_postnorm_residual_bf16", buffers: 4, pushWords: 2);
            _qkNorm = SimpleComputeKernel.Create(device, spvDir, "gemma_qknorm_f16", buffers: 2, pushWords: 2);
            _rope = SimpleComputeKernel.Create(device, spvDir, "rope_f16", buffers: 3, pushWords: 9);
            _geglu = SimpleComputeKernel.Create(device, spvDir, "gemma_geglu_f16", buffers: 2, pushWords: 1);
            _poolMean = SimpleComputeKernel.Create(device, spvDir, "gemma_pool_mean_f16", buffers: 3, pushWords: 1);

            long rowsH = (long)_cap * _h;
            _res = Scratch(rowsH); _hid2 = Scratch(rowsH);
            _q = Scratch((long)_cap * _qDim); _k = Scratch((long)_cap * _kvDim); _v = Scratch((long)_cap * _kvDim);
            _attn = Scratch((long)_cap * _qDim); _tmp = Scratch(rowsH);
            _gate = Scratch((long)_cap * _ff); _up = Scratch((long)_cap * _ff);
            _pos = device.Allocate((long)_cap * sizeof(int)); toDispose.Add(_pos);
            _seg = device.Allocate(2L * MaxDocsPerPack * sizeof(uint)); toDispose.Add(_seg);
            _flag = device.AllocateHostReadback(16); toDispose.Add(_flag);
            _outBuf = device.AllocateHostReadback((long)Math.Max(MaxDocsPerPack, 1) * _h * sizeof(ushort)); toDispose.Add(_outBuf);
            _hostEmb = (ushort*)NativeMemory.AlignedAlloc((nuint)(rowsH * sizeof(ushort)), 64);
            _submit = device.CreateSubmitContext();
            ComputeMemoryBytes = (3 * rowsH + 2L * _cap * _qDim + 2L * _cap * _kvDim + 2L * _cap * _ff + rowsH) * sizeof(ushort) + (long)_cap * sizeof(int);
        }
        catch
        {
            foreach (var d in toDispose) d.Dispose();
            if (_hostEmb != null) NativeMemory.AlignedFree(_hostEmb);
            throw;
        }
        _owned = toDispose;
    }

    /// <summary>Loads a <c>gemma-embedding</c> GGUF onto a new Vulkan device (owned by the model).</summary>
    public static VulkanGemmaEmbeddingModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        var device = VulkanDevice.Create();
        try { return new VulkanGemmaEmbeddingModel(device, ownsDevice: true, gguf, config, spvDir ?? VulkanModelLoader.ResolveSpvDir()); }
        catch { device.Dispose(); throw; }
    }

    /// <summary>Loads a <c>gemma-embedding</c> GGUF onto a caller-owned Vulkan device (not disposed with the model).</summary>
    public static VulkanGemmaEmbeddingModel LoadFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        return new VulkanGemmaEmbeddingModel(device, ownsDevice: false, gguf, config, spvDir ?? VulkanModelLoader.ResolveSpvDir());
    }

    // ───────────────────────────── IModel: embedding-only ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw NoLmHead();

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw NoLmHead();

    private NotSupportedException NoLmHead() => new(
        $"{Config.Architecture} is an embedding-only encoder: it has no LM head, KV cache or sampler. Use POST /v1/embeddings or `dotllm embed`.");

    private NotSupportedException MeanOnly() => new(
        "The Vulkan gemma-embedding encoder serves mean pooling only (the dense heads are trained after it); per-token hidden states are not available. Use --pooling mean or --device cpu.");

    // ───────────────────────────── IEmbeddingModel ─────────────────────────────

    /// <inheritdoc/>
    public ITensor ForwardHidden(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
    {
        if (!RawHiddenForTests) throw MeanOnly();
        for (int i = 0; i < positions.Length; i++)
            if (positions[i] != i) throw new NotSupportedException("Vulkan ForwardHidden supports only positions 0..n-1 (cacheless embedding forward).");
        if (tokenIds.Length == 0) throw new ArgumentException("Empty token sequence.", nameof(tokenIds));
        var result = UnmanagedTensor.Allocate(new TensorShape(tokenIds.Length, _h), DType.Float32, -1);
        RunPack([tokenIds.ToArray()], 0, 1, new Span<float>((void*)result.DataPointer, tokenIds.Length * _h), raw: true);
        return result;
    }

    /// <inheritdoc/>
    public void ForwardHiddenBatch(IReadOnlyList<int[]> sequences, IEmbeddingModel.HiddenStateConsumer consume, int deviceId)
        => throw MeanOnly();

    /// <inheritdoc/>
    public void ForwardHiddenForPooling(IReadOnlyList<int[]> sequences, PoolingType pooling, IEmbeddingModel.HiddenStateConsumer consume, int deviceId)
    {
        if (pooling != PoolingType.Mean) throw MeanOnly();
        ArgumentNullException.ThrowIfNull(sequences);
        ArgumentNullException.ThrowIfNull(consume);
        ObjectDisposedException.ThrowIf(_disposed, this);
        int i = 0;
        while (i < sequences.Count)
        {
            int n = 0, tokens = 0;
            while (i + n < sequences.Count && n < MaxDocsPerPack)
            {
                int len = sequences[i + n].Length;
                if (len == 0) throw new ArgumentException("Empty token sequence.", nameof(sequences));
                if (n > 0 && tokens + len > Math.Min(PackTokenBudget, _cap)) break;
                tokens += len; n++;
            }
            var buf = System.Buffers.ArrayPool<float>.Shared.Rent(n * _h);
            try
            {
                RunPack(sequences, i, n, buf.AsSpan(0, n * _h), raw: false);
                for (int s = 0; s < n; s++) consume(i + s, buf.AsSpan(s * _h, _h));   // one pooled (+ projected) row per document: Pool(1 row) is the identity
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(buf); }
            i += n;
        }
    }

    // ───────────────────────────── forward ─────────────────────────────

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static ushort ToBf16(float f)
    {
        uint b = BitConverter.SingleToUInt32Bits(f);
        b += 0x7FFFu + ((b >> 16) & 1u);   // round to nearest even (finite inputs)
        return (ushort)(b >> 16);
    }

    private void EmbedHost(int[] tokens, ushort* dst)
    {
        Span<float> row = _h <= 4096 ? stackalloc float[_h] : new float[_h];
        for (int t = 0; t < tokens.Length; t++)
        {
            int id = tokens[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token ID {id} at index {t} is out of range [0, {Config.VocabSize}).");
            Dequantize.ToFloat32(_tokenEmbd + (nint)((long)id * Dequantize.RowByteSize(_h, _tokenEmbdQt)), _h, _tokenEmbdQt, row);
            ushort* o = dst + (long)t * _h;
            for (int c = 0; c < _h; c++) o[c] = ToBf16(row[c] * _embScale);   // same F32 product as the CPU model; the one BF16 rounding of the embedding
        }
    }

    private void Gemm(nint cmd, Weight w, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n)
    {
        if (w.Format == EncoderWeightFormat.F16) _gemm.Record(cmd, w.Buf, input, output, m, k, n);
        else _encGemm[w.Format].Record(cmd, w.Buf, input, output, m, k, n);
    }

    private static void Barrier(nint cmd) => KernelSupport.ComputeToComputeBarrier(cmd);

    private void RunPack(IReadOnlyList<int[]> docs, int from, int count, Span<float> output, bool raw)
    {
        int h = _h, nh = _nh, nkv = _nkv, hd = _hd;
        int total = 0;
        for (int i = 0; i < count; i++) total += docs[from + i].Length;
        if (total > _cap) throw new ArgumentException($"Packed forward of {total} tokens exceeds the scratch capacity {_cap}.");
        for (int i = 0; i < count; i++)
            if (docs[from + i].Length > Config.MaxSequenceLength)
                throw new ArgumentException($"Sequence of {docs[from + i].Length} tokens exceeds the model's context length {Config.MaxSequenceLength}.");

        // Host: embedding sums (x sqrt(h), BF16), per-document positions, segment table.
        Span<uint> seg = stackalloc uint[2 * count];
        var pos = System.Buffers.ArrayPool<int>.Shared.Rent(total);
        try
        {
            int off = 0;
            for (int i = 0; i < count; i++)
            {
                var d = docs[from + i];
                EmbedHost(d, _hostEmb + (long)off * h);
                seg[2 * i] = (uint)off; seg[2 * i + 1] = (uint)d.Length;
                for (int tI = 0; tI < d.Length; tI++) pos[off + tI] = tI;
                off += d.Length;
            }
            _device.Upload(new ReadOnlySpan<byte>(_hostEmb, checked(total * h * sizeof(ushort))), _res);
            _device.Upload(MemoryMarshal.AsBytes(seg), _seg);
            _device.Upload(MemoryMarshal.AsBytes(pos.AsSpan(0, total)), _pos);
        }
        finally { System.Buffers.ArrayPool<int>.Shared.Return(pos); }
        _device.Upload(new byte[16], _flag);
        _flash.InvalidateDescriptorCache();

        uint epsBits = BitConverter.SingleToUInt32Bits(_eps);
        long qRowBytes = (long)_qDim * sizeof(ushort), kvRowBytes = (long)_kvDim * sizeof(ushort);
        var ctx = _submit;
        ctx.Begin();
        nint cmd = ctx.CommandBuffer;
        int halfRope = _ropeDim / 2;
        foreach (var l in _layers)
        {
            _preNorm.Record(cmd, [_res.Handle, l.AttnNorm.Handle, _hid2.Handle], [(uint)h, epsBits], (uint)total);
            Barrier(cmd);
            Gemm(cmd, l.Wq, _hid2, _q, _qDim, h, total);
            Gemm(cmd, l.Wk, _hid2, _k, _kvDim, h, total);
            Gemm(cmd, l.Wv, _hid2, _v, _kvDim, h, total);
            Barrier(cmd);
            if (!NoQkNormMutant)
            {
                _qkNorm.Record(cmd, [_q.Handle, l.QNorm.Handle], [(uint)hd, epsBits], (uint)(total * nh));
                _qkNorm.Record(cmd, [_k.Handle, l.KNorm.Handle], [(uint)hd, epsBits], (uint)(total * nkv));
                Barrier(cmd);
            }
            _rope.Record(cmd, [_q.Handle, _k.Handle, _pos.Handle],
                [(uint)total, (uint)nh, (uint)nkv, (uint)hd, (uint)_ropeDim, _ropeNeox ? 1u : 0u, BitConverter.SingleToUInt32Bits(l.RopeTheta), (uint)_ropeDim, (uint)halfRope],
                (uint)(((long)total * nh * halfRope + 255) / 256));
            Barrier(cmd);
            int window = WindowIgnoredMutant ? 0 : l.Window;
            var mode = CausalMutant ? AttentionMaskMode.Causal : AttentionMaskMode.SymmetricWindow;
            if (CausalMutant) window = 0;
            if (PackLeakMutant)
                _flash.RecordSlice(cmd, _q, 0, _k, _v, 0, _attn, 0, total, nh, nkv, hd, _attnScale, mode, window);
            else
            {
                long start = 0;
                for (int s = 0; s < count; s++)
                {
                    int len = docs[from + s].Length;
                    _flash.RecordSlice(cmd, _q, start * qRowBytes, _k, _v, start * kvRowBytes, _attn, start * qRowBytes, len, nh, nkv, hd, _attnScale, mode, window);
                    start += len;
                }
            }
            Barrier(cmd);
            Gemm(cmd, l.Wo, _attn, _tmp, h, _qDim, total);
            Barrier(cmd);
            _postNorm.Record(cmd, [_tmp.Handle, l.PostAttnNorm.Handle, _res.Handle, _flag.Handle], [(uint)h, epsBits], (uint)total);
            Barrier(cmd);
            _preNorm.Record(cmd, [_res.Handle, l.FfnNorm.Handle, _hid2.Handle], [(uint)h, epsBits], (uint)total);
            Barrier(cmd);
            Gemm(cmd, l.Wgate, _hid2, _gate, _ff, h, total);
            Gemm(cmd, l.Wup, _hid2, _up, _ff, h, total);
            Barrier(cmd);
            _geglu.Record(cmd, [_gate.Handle, _up.Handle], [(uint)(total * _ff)], (uint)(((long)total * _ff + 255) / 256));
            Barrier(cmd);
            Gemm(cmd, l.Wdown, _up, _tmp, h, _ff, total);
            Barrier(cmd);
            _postNorm.Record(cmd, [_tmp.Handle, l.PostFfnNorm.Handle, _res.Handle, _flag.Handle], [(uint)h, epsBits], (uint)total);
            Barrier(cmd);
        }

        _preNorm.Record(cmd, [_res.Handle, _outNorm.Handle, _hid2.Handle], [(uint)h, epsBits], (uint)total);
        Barrier(cmd);
        VulkanDevice.Buffer result;
        int resultRows;
        if (raw)
        {
            result = _hid2; resultRows = total;
        }
        else
        {
            _poolMean.Record(cmd, [_hid2.Handle, _seg.Handle, _tmp.Handle], [(uint)h], (uint)count, (uint)((h + 255) / 256));
            Barrier(cmd);
            result = _tmp; resultRows = count;
            if (_hasHeads)
            {
                Gemm(cmd, _d2, _tmp, _up, _d2Out, h, count);
                Barrier(cmd);
                Gemm(cmd, _d3, _up, _attn, h, _d2Out, count);
                Barrier(cmd);
                result = _attn;
            }
        }
        KernelSupport.ComputeToHostBarrier(cmd);
        ctx.SubmitAndWait();

        var flag = new float[1];
        _device.Download(_flag, flag);
        if (BitConverter.SingleToUInt32Bits(flag[0]) != 0)
            throw new InvalidOperationException(
                "Vulkan gemma-embedding: an F16 activation overflowed (a GEMM output exceeded 65504; EmbeddingGemma does not support F16 activations on such inputs). Re-run this input with --device cpu.");

        int halves = resultRows * h, words = (halves + 1) / 2;
        var tmp = System.Buffers.ArrayPool<float>.Shared.Rent(words);
        try
        {
            _device.Download(result, tmp.AsSpan(0, words));   // raw F16 words
            TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<float, Half>(tmp.AsSpan(0, words)).Slice(0, halves), output[..halves]);
        }
        finally { System.Buffers.ArrayPool<float>.Shared.Return(tmp); }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _submit.Dispose();
        _gemm.Dispose(); foreach (var g in _encGemm.Values) g.Dispose(); _flash.Dispose();
        _preNorm.Dispose(); _postNorm.Dispose(); _qkNorm.Dispose(); _rope.Dispose(); _geglu.Dispose(); _poolMean.Dispose();
        foreach (var d in _owned) d.Dispose();
        NativeMemory.AlignedFree(_hostEmb);
        if (_ownsDevice) _device.Dispose();
    }
}
