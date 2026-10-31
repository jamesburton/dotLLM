using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// BERT-class bidirectional encoder (<c>bert</c>: bge, e5, gte, MiniLM, mxbai, multilingual-e5) on the Vulkan backend (issue #741).
/// </summary>
/// <remarks>
/// <para>The graph is the CPU <c>BertEncoderModel</c> step for step (post-LN, learned absolute positions, token-type row 0, tanh GELU),
/// recorded into ONE command buffer per forward:</para>
/// <list type="bullet">
///   <item><description><b>Embedding</b>: the token / type / position row sum runs on the host (a few hundred KiB, same arithmetic as
///     the CPU model), is uploaded once, and the embedding LayerNorm runs on the device.</description></item>
///   <item><description><b>Packing, no padding</b>: documents are concatenated into one <c>[tokens, hidden]</c> matrix (the #1437
///     scheme). Every GEMM / bias / LayerNorm / GELU runs once over all rows with the weights read once; attention is one
///     <see cref="AttentionMaskMode.Bidirectional"/> coopmat flash dispatch per document over its slice of Q/K/V (sub-range
///     descriptors), so a query never sees another document's keys.</description></item>
///   <item><description><b>Weights stay F16 and resident</b>, read in place by the coopmat GEMM (no expansion). Activations are F32
///     scratch sized once at load for the pack budget (the same convention as the Vulkan decoder embedder, #1406; F16 activation
///     kernels are a follow-up).</description></item>
///   <item><description><b>Pooling</b> (CLS or mean) runs on the device and reads back <c>[docs, hidden]</c> only.</description></item>
/// </list>
/// <para>Scope of this first slice: <c>Architecture.Bert</c> with F16 weights. nomic-bert (RoPE + SwiGLU), nomic-bert-moe,
/// gemma-embedding and Q8_0 / F32 weight files are refused loudly, never run on a different path.</para>
/// </remarks>
public sealed unsafe class VulkanBertEncoderModel : IModel, IEmbeddingModel
{
    private sealed class Layer
    {
        public required VulkanDevice.Buffer Wq, Wk, Wv, Wo, Wup, Wdown;
        public required VulkanDevice.Buffer Bq, Bk, Bv, Bo, Bup, Bdown;
        public required VulkanDevice.Buffer Ln1W, Ln1B, Ln2W, Ln2B;
    }

    private enum PoolMode { All, Cls, Mean }

    /// <summary>Documents per packed forward (the flash kernel's sub-range descriptor cache holds 256 distinct tuples).</summary>
    internal const int MaxDocsPerPack = 128;

    /// <summary>Token budget of one packed forward. Env <c>DOTLLM_EMBED_PACK_TOKENS</c> overrides (default 1024, as the decoder embedder).</summary>
    internal static int PackTokenBudget =>
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_EMBED_PACK_TOKENS"), out int v) && v > 0 ? v : 1024;

    /// <summary>Test seam: attend across document boundaries (one dispatch over the concatenation) to prove the tests can detect a wrong mask.</summary>
    internal bool PackLeakMutant { get; set; }

    /// <summary>Test seam: causal instead of bidirectional attention, to prove the parity tests can detect a wrong mask mode.</summary>
    internal bool CausalMutant { get; set; }

    private readonly VulkanDevice _device;
    private readonly bool _ownsDevice;
    private readonly GgufFile _gguf;
    private readonly Layer[] _layers;
    private readonly nint _tokenEmbd, _posEmbd;
    private readonly QuantizationType _tokenEmbdQt, _posEmbdQt;
    private readonly int _posRows;
    private readonly float[] _typeRow;
    private readonly VulkanDevice.Buffer _embNormW, _embNormB, _zeros;
    private readonly float _eps;
    private readonly int _cap;   // max tokens per forward

    private readonly MatMulF16GemmCoopmatKernel _gemm;
    private readonly VulkanFlashAttentionCoopmatKernel _flash;
    private readonly BiasAddF32Kernel _biasAdd;
    private readonly SimpleComputeKernel _layerNorm, _biasGelu, _pool;

    // Activation scratch (F32), sized once for _cap tokens.
    private readonly VulkanDevice.Buffer _embIn, _hid, _hid2, _q, _k, _v, _attn, _tmp, _up, _seg, _poolOut;
    private readonly float* _hostEmb;   // [_cap, hidden] host staging of the embedding sum
    private readonly VulkanDevice.SubmitContext _submit;
    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes { get; }

    /// <inheritdoc/>
    public PoolingType? DeclaredPoolingType => Config.PoolingType;

    /// <inheritdoc/>
    public bool AppendsEosForLastPooling => false;

    private VulkanBertEncoderModel(VulkanDevice device, bool ownsDevice, GgufFile gguf, ModelConfig config, string spvDir)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        _gguf = gguf;
        Config = config;
        _eps = config.NormEpsilon;
        int h = config.HiddenSize, ff = config.IntermediateSize;

        if (config.Architecture != DotLLM.Core.Configuration.Architecture.Bert)
            throw new NotSupportedException(
                $"The Vulkan encoder supports Architecture.Bert only (got {config.Architecture}); nomic-bert / nomic-bert-moe / gemma-embedding run with --device cpu for now (#741 follow-ups).");
        if (h % 64 != 0 || ff % 32 != 0)
            throw new NotSupportedException($"Vulkan encoder needs hidden % 64 == 0 and ffn % 32 == 0 (got {h}, {ff}).");
        if (!device.HasCooperativeMatrix)
            throw new NotSupportedException("The Vulkan encoder needs VK_KHR_cooperative_matrix (F16 coopmat GEMM + flash attention).");
        _flash = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir)
            ?? throw new NotSupportedException("The Vulkan encoder needs the coopmat flash-attention kernel (not available on this device / build).");
        if (config.HeadDim > _flash.SupportedMaxHeadDim)
            throw new NotSupportedException($"Head dim {config.HeadDim} exceeds the flash kernel's {_flash.SupportedMaxHeadDim}.");

        _cap = Math.Max(PackTokenBudget, config.MaxSequenceLength);
        var dataBase = new GgufDataBase(gguf);
        var t = gguf.TensorsByName;
        Nvfp4Support.RejectUnappliedScales(t, consumedScales: null, "The Vulkan BERT encoder loader");

        GgufTensorDescriptor Req(string name)
            => t.TryGetValue(name, out var d) ? d : throw new InvalidDataException($"BERT GGUF is missing required tensor '{name}'.");

        float[] Vec(string name, int n)
        {
            var d = Req(name);
            var r = new float[n];
            Dequantize.ToFloat32(dataBase.Of(d), n, d.QuantizationType, r);
            return r;
        }

        var toDispose = new List<IDisposable>();
        VulkanDevice.Buffer F32(float[] data)
        {
            var b = device.AllocateDeviceLocal(Math.Max(16, (long)data.Length * sizeof(float)));
            toDispose.Add(b);
            device.Upload(data, b);
            return b;
        }
        VulkanDevice.Buffer Scratch(long floats)
        {
            var b = device.AllocateDeviceLocal(Math.Max(16, floats * sizeof(float)));
            toDispose.Add(b);
            return b;
        }
        VulkanDevice.Buffer WeightF16(string name, int expectK, int expectM)
        {
            var d = Req(name);
            if (d.QuantizationType != QuantizationType.F16)
                throw new NotSupportedException(
                    $"Vulkan encoder: tensor '{name}' is {d.QuantizationType}; only F16 weights are supported (Q8_0 / F32 encoder weights are a follow-up, #741). Use an F16 GGUF or --device cpu.");
            if (d.Shape[0] != expectK || d.Shape[1] != expectM)
                throw new InvalidDataException($"Tensor '{name}' has shape [{d.Shape[0]}, {d.Shape[1]}], expected [{expectK}, {expectM}].");
            long bytes = (long)expectK * expectM * 2;
            var b = device.AllocateDeviceLocal(bytes);
            toDispose.Add(b);
            device.Upload(new ReadOnlySpan<byte>((void*)dataBase.Of(d), checked((int)bytes)), b);
            return b;
        }
        float[] Zero(int n) => new float[n];
        float[] BiasOrZero(string name, int n) => t.ContainsKey(name) ? Vec(name, n) : Zero(n);

        try
        {
            var tokDesc = Req("token_embd.weight");
            if (tokDesc.Shape[0] != h) throw new InvalidDataException($"token_embd.weight row width {tokDesc.Shape[0]} != hidden size {h}.");
            _tokenEmbd = dataBase.Of(tokDesc); _tokenEmbdQt = tokDesc.QuantizationType;
            var posDesc = Req("position_embd.weight");
            _posEmbd = dataBase.Of(posDesc); _posEmbdQt = posDesc.QuantizationType; _posRows = posDesc.Shape[1];
            _typeRow = new float[h];
            if (t.TryGetValue("token_types.weight", out var typeDesc))
                Dequantize.ToFloat32(dataBase.Of(typeDesc), h, typeDesc.QuantizationType, _typeRow);

            _embNormW = F32(Vec("token_embd_norm.weight", h));
            _embNormB = F32(Vec("token_embd_norm.bias", h));
            _zeros = F32(Zero(Math.Max(h, ff)));

            _layers = new Layer[config.NumLayers];
            for (int i = 0; i < _layers.Length; i++)
            {
                string p = $"blk.{i}.";
                if (t.ContainsKey(p + "attn_qkv.weight") || t.ContainsKey(p + "ffn_up_exps.weight") || t.ContainsKey(p + "ffn_gate.weight"))
                    throw new NotSupportedException($"Vulkan encoder: layer {i} uses a fused-QKV / gated / MoE layout (nomic-bert family); not supported yet.");
                _layers[i] = new Layer
                {
                    Wq = WeightF16(p + "attn_q.weight", h, h), Wk = WeightF16(p + "attn_k.weight", h, h), Wv = WeightF16(p + "attn_v.weight", h, h),
                    Wo = WeightF16(p + "attn_output.weight", h, h),
                    Wup = WeightF16(p + "ffn_up.weight", h, ff), Wdown = WeightF16(p + "ffn_down.weight", ff, h),
                    Bq = F32(BiasOrZero(p + "attn_q.bias", h)), Bk = F32(BiasOrZero(p + "attn_k.bias", h)), Bv = F32(BiasOrZero(p + "attn_v.bias", h)),
                    Bo = F32(BiasOrZero(p + "attn_output.bias", h)), Bup = F32(BiasOrZero(p + "ffn_up.bias", ff)), Bdown = F32(BiasOrZero(p + "ffn_down.bias", h)),
                    Ln1W = F32(Vec(p + "attn_output_norm.weight", h)), Ln1B = F32(Vec(p + "attn_output_norm.bias", h)),
                    Ln2W = F32(Vec(p + "layer_output_norm.weight", h)), Ln2B = F32(Vec(p + "layer_output_norm.bias", h)),
                };
            }

            _gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
            _biasAdd = BiasAddF32Kernel.Create(device, spvDir);
            _layerNorm = SimpleComputeKernel.Create(device, spvDir, "bert_layernorm_f32", buffers: 6, pushWords: 3);
            _biasGelu = SimpleComputeKernel.Create(device, spvDir, "bert_bias_gelu_f32", buffers: 2, pushWords: 2);
            _pool = SimpleComputeKernel.Create(device, spvDir, "bert_pool_f32", buffers: 3, pushWords: 2);

            long rowsH = (long)_cap * h;
            _embIn = Scratch(rowsH); _hid = Scratch(rowsH); _hid2 = Scratch(rowsH);
            _q = Scratch(rowsH); _k = Scratch(rowsH); _v = Scratch(rowsH); _attn = Scratch(rowsH); _tmp = Scratch(rowsH);
            _up = Scratch((long)_cap * ff);
            _seg = device.Allocate(2L * MaxDocsPerPack * sizeof(uint)); toDispose.Add(_seg);
            _poolOut = device.AllocateHostReadback((long)MaxDocsPerPack * h * sizeof(float)); toDispose.Add(_poolOut);
            _hostEmb = (float*)NativeMemory.AlignedAlloc((nuint)(rowsH * sizeof(float)), 64);
            _submit = device.CreateSubmitContext();
            ComputeMemoryBytes = 14 * rowsH + (long)_cap * ff * 4;   // F32 scratch: 8 [cap,h] + the FFN [cap,ff] + staging
        }
        catch
        {
            foreach (var d in toDispose) d.Dispose();
            if (_hostEmb != null) NativeMemory.AlignedFree(_hostEmb);
            throw;
        }
        _owned = toDispose;
    }

    private readonly List<IDisposable> _owned;

    /// <summary>Loads a <c>bert</c> GGUF onto a new Vulkan device (owned by the model).</summary>
    public static VulkanBertEncoderModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        var device = VulkanDevice.Create();
        try { return new VulkanBertEncoderModel(device, ownsDevice: true, gguf, config, spvDir ?? VulkanModelLoader.ResolveSpvDir()); }
        catch { device.Dispose(); throw; }
    }

    /// <summary>Loads a <c>bert</c> GGUF onto a caller-owned Vulkan device (not disposed with the model).</summary>
    public static VulkanBertEncoderModel LoadFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        return new VulkanBertEncoderModel(device, ownsDevice: false, gguf, config, spvDir ?? VulkanModelLoader.ResolveSpvDir());
    }

    // ───────────────────────────── IModel: embedding-only ─────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw NoLmHead();

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw NoLmHead();

    private NotSupportedException NoLmHead() => new(
        $"{Config.Architecture} is an embedding-only encoder: it has no LM head, KV cache or sampler. Use POST /v1/embeddings or `dotllm embed`.");

    // ───────────────────────────── IEmbeddingModel ─────────────────────────────

    /// <inheritdoc/>
    public ITensor ForwardHidden(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
    {
        for (int i = 0; i < positions.Length; i++)
            if (positions[i] != i)
                throw new NotSupportedException("Vulkan ForwardHidden supports only positions 0..n-1 (cacheless embedding forward).");
        if (tokenIds.Length == 0) throw new ArgumentException("Empty token sequence.", nameof(tokenIds));
        int h = Config.HiddenSize;
        var result = UnmanagedTensor.Allocate(new TensorShape(tokenIds.Length, h), DType.Float32, -1);
        RunPack([tokenIds.ToArray()], 0, 1, PoolMode.All, new Span<float>((void*)result.DataPointer, tokenIds.Length * h));
        return result;
    }

    /// <inheritdoc/>
    public void ForwardHiddenBatch(IReadOnlyList<int[]> sequences, IEmbeddingModel.HiddenStateConsumer consume, int deviceId)
        => RunAll(sequences, PoolMode.All, consume);

    /// <inheritdoc/>
    public void ForwardHiddenForPooling(IReadOnlyList<int[]> sequences, PoolingType pooling, IEmbeddingModel.HiddenStateConsumer consume, int deviceId)
    {
        switch (pooling)
        {
            case PoolingType.Cls: RunAll(sequences, PoolMode.Cls, consume); break;
            case PoolingType.Mean: RunAll(sequences, PoolMode.Mean, consume); break;   // one pre-averaged row per document: Pool(1 row) is the identity
            default:
                RunAll(sequences, PoolMode.All, (idx, hidden) =>
                {
                    int n = sequences[idx].Length, h = hidden.Length / n;
                    consume(idx, pooling == PoolingType.Last ? hidden.Slice((n - 1) * h, h) : hidden);
                });
                break;
        }
    }

    private void RunAll(IReadOnlyList<int[]> sequences, PoolMode mode, IEmbeddingModel.HiddenStateConsumer consume)
    {
        ArgumentNullException.ThrowIfNull(sequences);
        ArgumentNullException.ThrowIfNull(consume);
        ObjectDisposedException.ThrowIf(_disposed, this);
        int h = Config.HiddenSize;
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
            int rows = mode == PoolMode.All ? tokens : n;
            var buf = System.Buffers.ArrayPool<float>.Shared.Rent(rows * h);
            try
            {
                RunPack(sequences, i, n, mode, buf.AsSpan(0, rows * h));
                if (mode == PoolMode.All)
                {
                    int start = 0;
                    for (int s = 0; s < n; s++)
                    {
                        int len = sequences[i + s].Length;
                        consume(i + s, buf.AsSpan(start * h, len * h));
                        start += len;
                    }
                }
                else
                    for (int s = 0; s < n; s++) consume(i + s, buf.AsSpan(s * h, h));
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(buf); }
            i += n;
        }
    }

    // ───────────────────────────── forward ─────────────────────────────

    private void EmbedHost(int[] tokens, float* dst, int h)
    {
        Span<float> pos = h <= 4096 ? stackalloc float[h] : new float[h];
        for (int t = 0; t < tokens.Length; t++)
        {
            int id = tokens[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token ID {id} at index {t} is out of range [0, {Config.VocabSize}).");
            if (t >= _posRows)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Position {t} exceeds the position-embedding table ({_posRows} rows).");
            var row = new Span<float>(dst + (long)t * h, h);
            Dequantize.ToFloat32(_tokenEmbd + (nint)(id * Dequantize.RowByteSize(h, _tokenEmbdQt)), h, _tokenEmbdQt, row);
            TensorPrimitives.Add(row, _typeRow, row);   // same order as the CPU model: (token + type) + position
            Dequantize.ToFloat32(_posEmbd + (nint)(t * Dequantize.RowByteSize(h, _posEmbdQt)), h, _posEmbdQt, pos);
            TensorPrimitives.Add(row, pos, row);
        }
    }

    private static void Barrier(nint cmd) => KernelSupport.ComputeToComputeBarrier(cmd);

    private void RunPack(IReadOnlyList<int[]> docs, int from, int count, PoolMode mode, Span<float> output)
    {
        int h = Config.HiddenSize, ff = Config.IntermediateSize, nh = Config.NumAttentionHeads, hd = Config.HeadDim;
        int total = 0;
        for (int i = 0; i < count; i++) total += docs[from + i].Length;
        if (total > _cap) throw new ArgumentException($"Packed forward of {total} tokens exceeds the scratch capacity {_cap}.");
        for (int i = 0; i < count; i++)
            if (docs[from + i].Length > Config.MaxSequenceLength)
                throw new ArgumentException($"Sequence of {docs[from + i].Length} tokens exceeds the model's context length {Config.MaxSequenceLength}.");

        // Host: embedding sums + segment table.
        Span<uint> seg = stackalloc uint[2 * count];
        int off = 0;
        for (int i = 0; i < count; i++)
        {
            var d = docs[from + i];
            EmbedHost(d, _hostEmb + (long)off * h, h);
            seg[2 * i] = (uint)off; seg[2 * i + 1] = (uint)d.Length;
            off += d.Length;
        }
        _device.Upload(new ReadOnlySpan<float>(_hostEmb, total * h), _embIn);
        if (mode != PoolMode.All) _device.Upload(MemoryMarshal.AsBytes(seg), _seg);

        // The flash kernel's sub-range descriptor sets are keyed by per-document offsets; the previous submit has completed, so reset them.
        _flash.InvalidateDescriptorCache();

        var ctx = _submit;
        ctx.Begin();
        nint cmd = ctx.CommandBuffer;
        _layerNorm.Record(cmd, [_embIn.Handle, _zeros.Handle, _embIn.Handle, _embNormW.Handle, _embNormB.Handle, _hid.Handle],
            [(uint)h, 0u, BitConverter.SingleToUInt32Bits(_eps)], (uint)total);
        Barrier(cmd);

        long rowBytes = (long)h * sizeof(float);
        var maskMode = CausalMutant ? AttentionMaskMode.Causal : AttentionMaskMode.Bidirectional;
        foreach (var l in _layers)
        {
            _gemm.Record(cmd, l.Wq, _hid, _q, h, h, total);
            _gemm.Record(cmd, l.Wk, _hid, _k, h, h, total);
            _gemm.Record(cmd, l.Wv, _hid, _v, h, h, total);
            Barrier(cmd);
            _biasAdd.Record(cmd, _q, l.Bq, total, h);
            _biasAdd.Record(cmd, _k, l.Bk, total, h);
            _biasAdd.Record(cmd, _v, l.Bv, total, h);
            Barrier(cmd);
            if (PackLeakMutant)
                _flash.RecordSlice(cmd, _q, 0, _k, _v, 0, _attn, 0, total, nh, nh, hd, 0f, maskMode);
            else
            {
                long start = 0;
                for (int s = 0; s < count; s++)
                {
                    int len = docs[from + s].Length;
                    _flash.RecordSlice(cmd, _q, start * rowBytes, _k, _v, start * rowBytes, _attn, start * rowBytes, len, nh, nh, hd, 0f, maskMode);
                    start += len;
                }
            }
            Barrier(cmd);
            _gemm.Record(cmd, l.Wo, _attn, _tmp, h, h, total);
            Barrier(cmd);
            _layerNorm.Record(cmd, [_tmp.Handle, l.Bo.Handle, _hid.Handle, l.Ln1W.Handle, l.Ln1B.Handle, _hid2.Handle],
                [(uint)h, 3u, BitConverter.SingleToUInt32Bits(_eps)], (uint)total);
            Barrier(cmd);
            _gemm.Record(cmd, l.Wup, _hid2, _up, ff, h, total);
            Barrier(cmd);
            _biasGelu.Record(cmd, [_up.Handle, l.Bup.Handle], [(uint)(total * ff), (uint)ff], (uint)(((long)total * ff + 255) / 256));
            Barrier(cmd);
            _gemm.Record(cmd, l.Wdown, _up, _tmp, h, ff, total);
            Barrier(cmd);
            _layerNorm.Record(cmd, [_tmp.Handle, l.Bdown.Handle, _hid2.Handle, l.Ln2W.Handle, l.Ln2B.Handle, _hid.Handle],
                [(uint)h, 3u, BitConverter.SingleToUInt32Bits(_eps)], (uint)total);
            Barrier(cmd);
        }

        if (mode != PoolMode.All)
        {
            _pool.Record(cmd, [_hid.Handle, _seg.Handle, _poolOut.Handle], [(uint)h, mode == PoolMode.Cls ? 0u : 1u],
                (uint)count, (uint)((h + 255) / 256));
        }
        KernelSupport.ComputeToHostBarrier(cmd);
        ctx.SubmitAndWait();

        if (mode == PoolMode.All) _device.Download(_hid, output[..(total * h)]);
        else _device.Download(_poolOut, output[..(count * h)]);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _submit.Dispose();
        _gemm.Dispose(); _flash.Dispose(); _biasAdd.Dispose(); _layerNorm.Dispose(); _biasGelu.Dispose(); _pool.Dispose();
        foreach (var d in _owned) d.Dispose();
        NativeMemory.AlignedFree(_hostEmb);
        if (_ownsDevice) _device.Dispose();
    }
}
