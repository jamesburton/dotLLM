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
///   <item><description><b>Weights stay F16 and resident</b>, read in place by the coopmat GEMM (no expansion). <b>Activations are F16</b>
///     (#1474, docs/PRECISION.md rule 1): every stored tensor (hidden, Q/K/V, attention output, FFN) is half-width scratch sized once at
///     load for the pack budget; the GEMM / flash accumulators, LayerNorm / softmax / pooling sums and the GELU arithmetic are F32 in
///     registers, and each stored tensor is rounded to F16 exactly once.</description></item>
///   <item><description><b>Pooling</b> (CLS or mean) runs on the device and reads back <c>[docs, hidden]</c> only.</description></item>
/// </list>
/// <para>Scope: <c>Architecture.Bert</c> with F16, Q8_0 or F32 weights (#1473), each read in place by its own GEMM (Q8_0 blocks dequantised and
/// F32 narrowed inside the LDS tile load; no expanded copy). nomic-bert (RoPE + SwiGLU), nomic-bert-moe and gemma-embedding are refused
/// loudly, never run on a different path.</para>
/// </remarks>
public sealed unsafe class VulkanBertEncoderModel : IModel, IEmbeddingModel
{
    private sealed class Layer
    {
        public required Weight Wq, Wk, Wv, Wo, Wup, Wdown;
        public Weight Wgate;   // nomic-bert SwiGLU only
        public required VulkanDevice.Buffer Bq, Bk, Bv, Bo, Bup, Bdown;
        public required VulkanDevice.Buffer Ln1W, Ln1B, Ln2W, Ln2B;
        public MoeLayer? Moe;   // nomic-bert-moe sparse layer: replaces Wup/Wdown/Wgate
    }

    /// <summary>A nomic-bert-moe sparse FFN layer: F32 router and the two 3-D Q8_0 expert banks, resident and read in place (#1471).</summary>
    private sealed class MoeLayer
    {
        public required VulkanDevice.Buffer Router;      // [E, hidden] F32
        public required VulkanDevice.Buffer UpBank;      // [E, ff, hidden] Q8_0 blocks
        public required VulkanDevice.Buffer DownBank;    // [E, hidden, ff] Q8_0 blocks
    }

    /// <summary>Test seam for the MoE router (#1471): wrong expert ids / renormalised top-k weights, to prove the parity tests can detect them.</summary>
    internal enum MoeRoutingMutant { None = 0, WrongExperts = 1, RenormalisedWeights = 2 }

    /// <summary>Test seam: corrupt the MoE routing (nomic-bert-moe only).</summary>
    internal MoeRoutingMutant RoutingMutant { get; set; }

    /// <summary>A resident weight matrix in its stored format (read in place, never expanded).</summary>
    private readonly record struct Weight(VulkanDevice.Buffer Buf, EncoderWeightFormat Format);

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

    /// <summary>Test seam (nomic-bert): every token gets RoPE position 0 (no relative position signal), to prove the tests can detect wrong positions. (A constant per-document offset would NOT change the output: RoPE is relative.)</summary>
    internal bool PositionMutant { get; set; }

    private readonly VulkanDevice _device;
    private readonly bool _ownsDevice;
    private readonly GgufFile _gguf;
    private readonly Layer[] _layers;
    private readonly nint _tokenEmbd, _posEmbd;
    private readonly QuantizationType _tokenEmbdQt, _posEmbdQt;
    private readonly int _posRows;
    private readonly bool _swiglu;    // nomic-bert (dense) SwiGLU FFN; nomic-bert-moe keeps the biased GELU FFN on its dense layers
    private readonly bool _nomic;     // RoPE NeoX instead of absolute positions; fused attn_qkv (split at load); SwiGLU FFN
    private readonly float[] _typeRow;
    private readonly VulkanDevice.Buffer _embNormW, _embNormB, _zeros;
    private readonly float _eps;
    private readonly int _cap;   // max tokens per forward

    private readonly MatMulF16GemmCoopmatKernel _gemm;
    private readonly Dictionary<EncoderWeightFormat, MatMulEncoderGemmKernel> _encGemm = new();
    private readonly VulkanFlashAttentionCoopmatKernel _flash;
    private readonly SimpleComputeKernel _layerNorm, _biasGelu, _biasAdd, _pool;
    private readonly SimpleComputeKernel? _rope, _swigluK;
    // nomic-bert-moe (#1471)
    private readonly SimpleComputeKernel? _moeRouter, _moeGroup, _moeCombine;
    private readonly MoeExpertOffsetsKernel? _moeOffsets;
    private readonly MoeBuildTileListKernel? _moeTiles;
    private readonly MoeGroupedMatmulQ8_0F16ActKernel? _moeGemm;
    private readonly VulkanDevice.Buffer? _mPacked, _mMid, _mOut, _mIdx, _mWts, _mInv, _mCounts, _mCounters, _mOffsets, _mArgs;
    private readonly int _moeE, _moeTopK, _moeFf;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;

    // Activation scratch (F16), sized once for _cap tokens. _tmp doubles as the embedding-sum upload target (free until the first Wo GEMM).
    private readonly VulkanDevice.Buffer? _gate, _pos;
    private readonly VulkanDevice.Buffer _hid, _hid2, _q, _k, _v, _attn, _tmp, _up, _seg, _poolOut;
    private readonly Half* _hostEmb;   // [_cap, hidden] host staging of the (F16-rounded) embedding sum
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

        if (config.Architecture is not (DotLLM.Core.Configuration.Architecture.Bert or DotLLM.Core.Configuration.Architecture.NomicBert or DotLLM.Core.Configuration.Architecture.NomicBertMoe))
            throw new NotSupportedException(
                $"The Vulkan encoder supports Architecture.Bert, NomicBert and NomicBertMoe (got {config.Architecture}); gemma-embedding runs with --device cpu for now (#1472).");
        _nomic = config.Architecture is DotLLM.Core.Configuration.Architecture.NomicBert or DotLLM.Core.Configuration.Architecture.NomicBertMoe;
        _swiglu = config.Architecture == DotLLM.Core.Configuration.Architecture.NomicBert;
        if (_nomic)
        {
            var rope = config.RoPEConfig ?? throw new InvalidDataException("nomic-bert config has no RoPE configuration.");
            _ropeTheta = rope.Theta; _ropeDim = rope.DimensionCount;
        }
        if (h % 64 != 0 || ff % 32 != 0)
            throw new NotSupportedException($"Vulkan encoder needs hidden % 64 == 0 and ffn % 32 == 0 (got {h}, {ff}).");
        if (!device.HasCooperativeMatrix)
            throw new NotSupportedException("The Vulkan encoder needs VK_KHR_cooperative_matrix (F16 coopmat GEMM + flash attention).");
        _flash = VulkanFlashAttentionCoopmatKernel.TryCreateF16Activations(device, spvDir)
            ?? throw new NotSupportedException("The Vulkan encoder needs the F16-activation coopmat flash-attention kernel (not available on this device / build).");
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
        VulkanDevice.Buffer Scratch(long halves)
        {
            var b = device.AllocateDeviceLocal(Math.Max(16, halves * sizeof(ushort)));
            toDispose.Add(b);
            return b;
        }
        Weight WeightIn(string name, int expectK, int expectM, int rowStart = 0, int totalRows = -1)
        {
            var d = Req(name);
            if (totalRows < 0) totalRows = expectM;
            var fmt = d.QuantizationType switch
            {
                QuantizationType.F16 => EncoderWeightFormat.F16,
                QuantizationType.Q8_0 => EncoderWeightFormat.Q8_0,
                QuantizationType.F32 => EncoderWeightFormat.F32,
                _ => throw new NotSupportedException(
                    $"Vulkan encoder: tensor '{name}' is {d.QuantizationType}; F16, Q8_0 and F32 weights are read in place, other formats run with --device cpu."),
            };
            if (d.Shape[0] != expectK || d.Shape[1] != totalRows)
                throw new InvalidDataException($"Tensor '{name}' has shape [{d.Shape[0]}, {d.Shape[1]}], expected [{expectK}, {totalRows}].");
            if (!MatMulEncoderGemmKernel.SupportsK(fmt, expectK))
                throw new NotSupportedException($"Vulkan encoder: '{name}' ({fmt}) has K={expectK}, which the in-place kernel cannot read (Q8_0 needs K % 64 == 0).");
            if (fmt != EncoderWeightFormat.F16 && !MatMulEncoderGemmKernel.IsSupportedOn(device, spvDir, fmt))
                throw new NotSupportedException(
                    $"Vulkan encoder: {fmt} weights are read in place by the blocked 128x128 coopmat tile, which needs a wave64 cooperative-matrix device (this one has subgroup size {device.SubgroupSize}). Use an F16 GGUF or --device cpu.");
            long bytes = MatMulEncoderGemmKernel.WeightBytes(fmt, expectM, expectK);
            var b = device.AllocateDeviceLocal((bytes + 3) & ~3L);
            toDispose.Add(b);
            long rowBytes = MatMulEncoderGemmKernel.WeightBytes(fmt, 1, expectK);   // a row slice of a row-major [rows, K] tensor is contiguous
            device.Upload(new ReadOnlySpan<byte>((byte*)dataBase.Of(d) + rowStart * rowBytes, checked((int)bytes)), b);
            return new Weight(b, fmt);
        }
        float[] VecSlice(string name, int total, int start, int n)
        {
            var all = Vec(name, total);
            return all.AsSpan(start, n).ToArray();
        }
        float[] Zero(int n) => new float[n];
        float[] BiasOrZero(string name, int n) => t.ContainsKey(name) ? Vec(name, n) : Zero(n);

        try
        {
            var tokDesc = Req("token_embd.weight");
            if (tokDesc.Shape[0] != h) throw new InvalidDataException($"token_embd.weight row width {tokDesc.Shape[0]} != hidden size {h}.");
            _tokenEmbd = dataBase.Of(tokDesc); _tokenEmbdQt = tokDesc.QuantizationType;
            if (!_nomic)
            {
                var posDesc = Req("position_embd.weight");
                _posEmbd = dataBase.Of(posDesc); _posEmbdQt = posDesc.QuantizationType; _posRows = posDesc.Shape[1];
            }
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
                bool isMoe = t.ContainsKey(p + "ffn_up_exps.weight");
                if (isMoe && config.Architecture != DotLLM.Core.Configuration.Architecture.NomicBertMoe)
                    throw new NotSupportedException($"Vulkan encoder: layer {i} is a mixture-of-experts layer in a {config.Architecture} model.");
                if (_swiglu && (t.ContainsKey(p + "ffn_up.bias") || t.ContainsKey(p + "ffn_gate.bias")))
                    throw new NotSupportedException($"Vulkan encoder: layer {i} has SwiGLU FFN biases, which the encoder does not apply.");
                bool fused = t.ContainsKey(p + "attn_qkv.weight");
                if (fused && !_nomic) throw new NotSupportedException($"Vulkan encoder: layer {i} has a fused attn_qkv in a non-nomic architecture.");
                bool qkvBias = t.ContainsKey(p + "attn_qkv.bias");
                MoeLayer? moeLayer = null;
                float[]? expertsBias = null;
                if (isMoe)
                {
                    var moeCfg = config.Moe ?? throw new InvalidDataException("nomic-bert-moe config has no MoE configuration.");
                    var upD = Req(p + "ffn_up_exps.weight"); var downD = Req(p + "ffn_down_exps.weight"); var routD = Req(p + "ffn_gate_inp.weight");
                    int e = upD.Shape[2], eff = upD.Shape[1];
                    if (e != moeCfg.NumExperts || downD.Shape[2] != e || upD.Shape[0] != h || downD.Shape[0] != eff || downD.Shape[1] != h)
                        throw new InvalidDataException($"nomic-bert-moe layer {i}: unexpected expert bank shapes up [{upD.Shape[0]},{upD.Shape[1]},{upD.Shape[2]}] down [{downD.Shape[0]},{downD.Shape[1]},{downD.Shape[2]}].");
                    if (routD.Shape[0] != h || routD.Shape[1] != e) throw new InvalidDataException($"nomic-bert-moe layer {i}: router shape [{routD.Shape[0]},{routD.Shape[1]}] != [{h},{e}].");
                    if (e > 16 || moeCfg.NumExpertsPerTok > 4) throw new NotSupportedException($"Vulkan encoder router supports up to 16 experts and top-4 (got {e}, top-{moeCfg.NumExpertsPerTok}).");
                    if (_moeE != 0 && (_moeE != e || _moeFf != eff)) throw new InvalidDataException("nomic-bert-moe layers disagree on the expert count / width.");
                    _moeE = e; _moeTopK = moeCfg.NumExpertsPerTok; _moeFf = eff;
                    VulkanDevice.Buffer Bank(GgufTensorDescriptor d, int k, int m)
                    {
                        if (d.QuantizationType != QuantizationType.Q8_0)
                            throw new NotSupportedException($"Vulkan encoder: expert bank '{d.Name}' is {d.QuantizationType}; only Q8_0 banks are read in place (F16/F32 banks: use --device cpu, #1471).");
                        if (k % MoeGroupedMatmulQ8_0F16ActKernel.KGroup != 0) throw new NotSupportedException($"Vulkan encoder: expert bank '{d.Name}' has K={k}, which the Q8_0 grouped kernel cannot read (K % 64).");
                        long bytes = (long)e * m * MoeGroupedMatmulQ8_0F16ActKernel.RowBytes(k);
                        var b = device.AllocateDeviceLocal((bytes + 3) & ~3L);
                        toDispose.Add(b);
                        device.Upload(new ReadOnlySpan<byte>((byte*)dataBase.Of(d), checked((int)bytes)), b);
                        return b;
                    }
                    moeLayer = new MoeLayer { Router = F32(Vec(p + "ffn_gate_inp.weight", e * h)), UpBank = Bank(upD, h, eff), DownBank = Bank(downD, eff, h) };
                    expertsBias = t.ContainsKey(p + "ffn_down_exps.bias") ? Vec(p + "ffn_down_exps.bias", h) : null;
                }
                _layers[i] = new Layer
                {
                    Moe = moeLayer,
                    // Fused attn_qkv [h, 3h] is split into three row-range buffers at upload (the same bytes, no copy into a wider format).
                    Wq = fused ? WeightIn(p + "attn_qkv.weight", h, h, 0, 3 * h) : WeightIn(p + "attn_q.weight", h, h),
                    Wk = fused ? WeightIn(p + "attn_qkv.weight", h, h, h, 3 * h) : WeightIn(p + "attn_k.weight", h, h),
                    Wv = fused ? WeightIn(p + "attn_qkv.weight", h, h, 2 * h, 3 * h) : WeightIn(p + "attn_v.weight", h, h),
                    Wo = WeightIn(p + "attn_output.weight", h, h),
                    Wup = isMoe ? default : WeightIn(p + "ffn_up.weight", h, ff), Wdown = isMoe ? default : WeightIn(p + "ffn_down.weight", ff, h),
                    Wgate = _swiglu ? WeightIn(p + "ffn_gate.weight", h, ff) : default,
                    Bq = F32(fused ? (qkvBias ? VecSlice(p + "attn_qkv.bias", 3 * h, 0, h) : Zero(h)) : BiasOrZero(p + "attn_q.bias", h)),
                    Bk = F32(fused ? (qkvBias ? VecSlice(p + "attn_qkv.bias", 3 * h, h, h) : Zero(h)) : BiasOrZero(p + "attn_k.bias", h)),
                    Bv = F32(fused ? (qkvBias ? VecSlice(p + "attn_qkv.bias", 3 * h, 2 * h, h) : Zero(h)) : BiasOrZero(p + "attn_v.bias", h)),
                    Bo = F32(BiasOrZero(p + "attn_output.bias", h)), Bup = isMoe ? _zeros : F32(BiasOrZero(p + "ffn_up.bias", ff)),
                    Bdown = isMoe ? (expertsBias is null ? _zeros : F32(expertsBias)) : F32(BiasOrZero(p + "ffn_down.bias", h)),   // the learned experts.bias is applied by the bias + residual + LayerNorm after the combine (CPU order)
                    Ln1W = F32(Vec(p + "attn_output_norm.weight", h)), Ln1B = F32(Vec(p + "attn_output_norm.bias", h)),
                    Ln2W = F32(Vec(p + "layer_output_norm.weight", h)), Ln2B = F32(Vec(p + "layer_output_norm.bias", h)),
                };
            }

            _gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir, F16GemmCoopmatVariant.SelectF16ActFor(device, spvDir));
            foreach (var l in _layers)
                foreach (var w in l.Moe is not null ? new[] { l.Wq, l.Wk, l.Wv, l.Wo } : _swiglu ? new[] { l.Wq, l.Wk, l.Wv, l.Wo, l.Wup, l.Wdown, l.Wgate } : new[] { l.Wq, l.Wk, l.Wv, l.Wo, l.Wup, l.Wdown })
                    if (w.Format != EncoderWeightFormat.F16 && !_encGemm.ContainsKey(w.Format))
                        _encGemm[w.Format] = MatMulEncoderGemmKernel.Create(device, spvDir, w.Format);
            _biasAdd = SimpleComputeKernel.Create(device, spvDir, "bert_bias_add_f16", buffers: 2, pushWords: 2);
            _layerNorm = SimpleComputeKernel.Create(device, spvDir, "bert_layernorm_f16", buffers: 6, pushWords: 3);
            _biasGelu = SimpleComputeKernel.Create(device, spvDir, "bert_bias_gelu_f16", buffers: 2, pushWords: 2);
            _pool = SimpleComputeKernel.Create(device, spvDir, "bert_pool_f16", buffers: 3, pushWords: 2);
            if (_nomic)
            {
                _rope = SimpleComputeKernel.Create(device, spvDir, "rope_f16", buffers: 3, pushWords: 9);
            }
            if (_swiglu) _swigluK = SimpleComputeKernel.Create(device, spvDir, "bert_swiglu_f16", buffers: 2, pushWords: 1);
            if (_moeE != 0)
            {
                if (!MoeGroupedMatmulQ8_0F16ActKernel.IsSupportedOn(device, spvDir))
                    throw new NotSupportedException($"Vulkan encoder: the nomic-bert-moe grouped Q8_0 expert kernel needs a wave64 cooperative-matrix device (this one has subgroup size {device.SubgroupSize}). Use --device cpu.");
                _moeRouter = SimpleComputeKernel.Create(device, spvDir, "bert_moe_router_f16", buffers: 4, pushWords: 4);
                _moeGroup = SimpleComputeKernel.Create(device, spvDir, "bert_moe_group_f16", buffers: 6, pushWords: 4);
                _moeCombine = SimpleComputeKernel.Create(device, spvDir, "bert_moe_combine_f16", buffers: 4, pushWords: 3);
                _moeOffsets = MoeExpertOffsetsKernel.Create(device, spvDir);
                _moeTiles = MoeBuildTileListKernel.Create(device, spvDir);
                _moeGemm = MoeGroupedMatmulQ8_0F16ActKernel.Create(device, spvDir);
            }

            long rowsH = (long)_cap * h;
            _hid = Scratch(rowsH); _hid2 = Scratch(rowsH);
            _q = Scratch(rowsH); _k = Scratch(rowsH); _v = Scratch(rowsH); _attn = Scratch(rowsH); _tmp = Scratch(rowsH);
            _up = Scratch((long)_cap * ff);
            if (_swiglu) _gate = Scratch((long)_cap * ff);
            if (_nomic) { _pos = device.Allocate((long)_cap * sizeof(int)); toDispose.Add(_pos); }
            _seg = device.Allocate(2L * MaxDocsPerPack * sizeof(uint)); toDispose.Add(_seg);
            _poolOut = device.AllocateHostReadback((long)MaxDocsPerPack * h * sizeof(float)); toDispose.Add(_poolOut);
            if (_moeE != 0)
            {
                long routed = (long)_moeTopK * _cap;
                _mPacked = Scratch(routed * h); _mMid = Scratch(routed * _moeFf); _mOut = Scratch(routed * h);
                _mIdx = device.AllocateDeviceLocal(routed * sizeof(int)); toDispose.Add(_mIdx);
                _mWts = device.AllocateDeviceLocal(routed * sizeof(float)); toDispose.Add(_mWts);
                _mInv = device.AllocateDeviceLocal(routed * sizeof(uint)); toDispose.Add(_mInv);
                _mCounts = device.AllocateDeviceLocal(_moeE * sizeof(uint)); toDispose.Add(_mCounts);
                _mCounters = device.AllocateDeviceLocal(_moeE * sizeof(uint)); toDispose.Add(_mCounters);
                _mOffsets = device.AllocateDeviceLocal(MoeBuildTileListKernel.OffsetsBufferUints(_moeE, (int)routed) * sizeof(uint)); toDispose.Add(_mOffsets);
                _mArgs = device.AllocateDeviceLocal(2 * MoeBuildTileListKernel.ArgsStrideBytes); toDispose.Add(_mArgs);
            }
            _hostEmb = (Half*)NativeMemory.AlignedAlloc((nuint)(rowsH * sizeof(ushort)), 64);
            _submit = device.CreateSubmitContext();
            ComputeMemoryBytes = (7 * rowsH + (_swiglu ? 2L : 1L) * _cap * ff + rowsH) * sizeof(ushort) + (_nomic ? (long)_cap * sizeof(int) : 0)
                + (_moeE == 0 ? 0 : (long)_moeTopK * _cap * (2L * h + _moeFf) * sizeof(ushort) + (long)_moeTopK * _cap * 3 * sizeof(uint));   // F16 scratch: 7 [cap,h] + the FFN [cap,ff], plus the host staging
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

    private void EmbedHost(int[] tokens, Half* dst, int h)
    {
        Span<float> pos = h <= 4096 ? stackalloc float[h] : new float[h];
        Span<float> row = h <= 4096 ? stackalloc float[h] : new float[h];
        for (int t = 0; t < tokens.Length; t++)
        {
            int id = tokens[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token ID {id} at index {t} is out of range [0, {Config.VocabSize}).");
            if (!_nomic && t >= _posRows)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Position {t} exceeds the position-embedding table ({_posRows} rows).");
            Dequantize.ToFloat32(_tokenEmbd + (nint)(id * Dequantize.RowByteSize(h, _tokenEmbdQt)), h, _tokenEmbdQt, row);
            TensorPrimitives.Add(row, _typeRow, row);   // same order as the CPU model: (token + type) + position
            if (!_nomic)   // nomic-bert has no position table: positions enter through RoPE
            {
                Dequantize.ToFloat32(_posEmbd + (nint)(t * Dequantize.RowByteSize(h, _posEmbdQt)), h, _posEmbdQt, pos);
                TensorPrimitives.Add(row, pos, row);
            }
            TensorPrimitives.ConvertToHalf(row, new Span<Half>(dst + (long)t * h, h));   // the one F16 rounding of the embedding sum
        }
    }

    private void Gemm(nint cmd, Weight w, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n)
    {
        if (w.Format == EncoderWeightFormat.F16) _gemm.Record(cmd, w.Buf, input, output, m, k, n);
        else _encGemm[w.Format].Record(cmd, w.Buf, input, output, m, k, n);
    }

    private static void Barrier(nint cmd) => KernelSupport.ComputeToComputeBarrier(cmd);

    /// <summary>
    /// Records the sparse MoE FFN of one nomic-bert-moe layer over 	otal packed tokens (#1471), <c>_hid2 -> _tmp</c>, all on F16 activations:
    /// router (softmax over all experts, top-k, weights NOT renormalised) -> expert offsets -> tile list -> group the routed rows by expert ->
    /// grouped Q8_0 up GEMM with fused tanh GELU -> grouped Q8_0 down GEMM -> weighted gather-combine. The expert banks are read in place.
    /// </summary>
    private void RecordMoeFfn(nint cmd, MoeLayer m, int total)
    {
        int h = Config.HiddenSize, ff = _moeFf, e = _moeE, k = _moeTopK, rows = total * k;
        var idx = _mIdx!; var wts = _mWts!; var inv = _mInv!; var offs = _mOffsets!; var args = _mArgs!; var packed = _mPacked!; var mid = _mMid!; var outB = _mOut!; var gemm = _moeGemm!;
        _moeRouter!.Record(cmd, [_hid2.Handle, m.Router.Handle, idx.Handle, wts.Handle], [(uint)h, (uint)e, (uint)k, (uint)RoutingMutant], (uint)total);
        Barrier(cmd);
        _moeOffsets!.Record(cmd, idx, _mCounts!, offs, _mCounters!, rows, e);
        Barrier(cmd);
        _moeTiles!.Record(cmd, offs, args, e, MoeGroupedMatmulQ8_0F16ActKernel.MTiles(ff), MoeGroupedMatmulQ8_0F16ActKernel.MTiles(h));
        KernelSupport.ComputeToIndirectAndComputeBarrier(cmd);
        _moeGroup!.Record(cmd, [_hid2.Handle, idx.Handle, offs.Handle, _mCounters!.Handle, packed.Handle, inv.Handle], [(uint)rows, (uint)h, (uint)e, (uint)k], (uint)rows);
        Barrier(cmd);
        gemm.RecordIndirect(cmd, m.UpBank, packed, offs, mid, args, 0, ff, h, rows, e, gelu: true);
        Barrier(cmd);
        gemm.RecordIndirect(cmd, m.DownBank, mid, offs, outB, args, MoeBuildTileListKernel.ArgsStrideBytes, h, ff, rows, e, gelu: false);
        Barrier(cmd);
        _moeCombine!.Record(cmd, [outB.Handle, inv.Handle, wts.Handle, _tmp.Handle], [(uint)(total * h), (uint)h, (uint)k], (uint)(((long)total * h + 255) / 256));
    }

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
        _device.Upload(new ReadOnlySpan<byte>(_hostEmb, checked(total * h * sizeof(ushort))), _tmp);
        if (mode != PoolMode.All) _device.Upload(MemoryMarshal.AsBytes(seg), _seg);
        if (_nomic)
        {
            // Per-document positions restart at 0 (RoPE is the only positional signal).
            var pos = System.Buffers.ArrayPool<int>.Shared.Rent(total);
            try
            {
                int o = 0;
                for (int i = 0; i < count; i++)
                    for (int tI = 0; tI < docs[from + i].Length; tI++, o++) pos[o] = PositionMutant ? 0 : tI;
                _device.Upload(MemoryMarshal.AsBytes(pos.AsSpan(0, total)), _pos!);
            }
            finally { System.Buffers.ArrayPool<int>.Shared.Return(pos); }
        }

        // The flash kernel's sub-range descriptor sets are keyed by per-document offsets; the previous submit has completed, so reset them.
        _flash.InvalidateDescriptorCache();

        var ctx = _submit;
        ctx.Begin();
        nint cmd = ctx.CommandBuffer;
        _layerNorm.Record(cmd, [_tmp.Handle, _zeros.Handle, _tmp.Handle, _embNormW.Handle, _embNormB.Handle, _hid.Handle],
            [(uint)h, 0u, BitConverter.SingleToUInt32Bits(_eps)], (uint)total);
        Barrier(cmd);

        long rowBytes = (long)h * sizeof(ushort);   // F16 rows
        var maskMode = CausalMutant ? AttentionMaskMode.Causal : AttentionMaskMode.Bidirectional;
        foreach (var l in _layers)
        {
            Gemm(cmd, l.Wq, _hid, _q, h, h, total);
            Gemm(cmd, l.Wk, _hid, _k, h, h, total);
            Gemm(cmd, l.Wv, _hid, _v, h, h, total);
            Barrier(cmd);
            _biasAdd.Record(cmd, [_q.Handle, l.Bq.Handle], [(uint)(total * h), (uint)h], (uint)(((long)total * h + 255) / 256));
            _biasAdd.Record(cmd, [_k.Handle, l.Bk.Handle], [(uint)(total * h), (uint)h], (uint)(((long)total * h + 255) / 256));
            _biasAdd.Record(cmd, [_v.Handle, l.Bv.Handle], [(uint)(total * h), (uint)h], (uint)(((long)total * h + 255) / 256));
            Barrier(cmd);
            if (_nomic)
            {
                int halfRope = _ropeDim / 2;
                _rope!.Record(cmd, [_q.Handle, _k.Handle, _pos!.Handle],
                    [(uint)total, (uint)nh, (uint)nh, (uint)hd, (uint)_ropeDim, 1u, BitConverter.SingleToUInt32Bits(_ropeTheta), (uint)_ropeDim, (uint)halfRope],
                    (uint)(((long)total * nh * halfRope + 255) / 256));
                Barrier(cmd);
            }
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
            Gemm(cmd, l.Wo, _attn, _tmp, h, h, total);
            Barrier(cmd);
            _layerNorm.Record(cmd, [_tmp.Handle, l.Bo.Handle, _hid.Handle, l.Ln1W.Handle, l.Ln1B.Handle, _hid2.Handle],
                [(uint)h, 3u, BitConverter.SingleToUInt32Bits(_eps)], (uint)total);
            Barrier(cmd);
            if (l.Moe is not null)
            {
                RecordMoeFfn(cmd, l.Moe, total);   // _hid2 -> _tmp (weighted expert sum)
            }
            else
            {
                if (_swiglu)
                {
                    Gemm(cmd, l.Wgate, _hid2, _gate!, ff, h, total);
                    Gemm(cmd, l.Wup, _hid2, _up, ff, h, total);
                    Barrier(cmd);
                    _swigluK!.Record(cmd, [_gate!.Handle, _up.Handle], [(uint)(total * ff)], (uint)(((long)total * ff + 255) / 256));
                }
                else
                {
                    Gemm(cmd, l.Wup, _hid2, _up, ff, h, total);
                    Barrier(cmd);
                    _biasGelu.Record(cmd, [_up.Handle, l.Bup.Handle], [(uint)(total * ff), (uint)ff], (uint)(((long)total * ff + 255) / 256));
                }
                Barrier(cmd);
                Gemm(cmd, l.Wdown, _up, _tmp, h, ff, total);
            }
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

        if (mode == PoolMode.All)
        {
            int halves = total * h, words = (halves + 1) / 2;
            var tmp = System.Buffers.ArrayPool<float>.Shared.Rent(words);
            try
            {
                _device.Download(_hid, tmp.AsSpan(0, words));   // raw F16 words
                TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<float, Half>(tmp.AsSpan(0, words)).Slice(0, halves), output[..halves]);
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(tmp); }
        }
        else _device.Download(_poolOut, output[..(count * h)]);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _submit.Dispose();
        _rope?.Dispose(); _swigluK?.Dispose(); _moeRouter?.Dispose(); _moeGroup?.Dispose(); _moeCombine?.Dispose(); _moeOffsets?.Dispose(); _moeTiles?.Dispose(); _moeGemm?.Dispose(); _gemm.Dispose(); foreach (var g in _encGemm.Values) g.Dispose(); _flash.Dispose(); _biasAdd.Dispose(); _layerNorm.Dispose(); _biasGelu.Dispose(); _pool.Dispose();
        foreach (var d in _owned) d.Dispose();
        NativeMemory.AlignedFree(_hostEmb);
        if (_ownsDevice) _device.Dispose();
    }
}
