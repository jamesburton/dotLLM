using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Nemotron-H (<c>nemotron_h</c>) hybrid SSM + Transformer model.
///
/// Each layer is exactly one of:
///   - Mamba2 selective state-space (SSM),
///   - GQA multi-head attention, or
///   - squared-ReLU MLP (no gate).
/// with a single RMSNorm before and one residual add after.
/// </summary>
public sealed unsafe class NemotronHTransformerModel : IModel
{
    private const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;
    private const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;
    private const int Q8_1GroupSize = QuantFormat.LegacyGroupSize;

    private readonly GgufFile? _gguf; // keep alive (null when constructed from prebuilt weights)
    private readonly NemotronHLayerWeights[] _layers;

    /// <summary>Per-layer weights as loaded (read-only view for tests and diagnostics).</summary>
    internal IReadOnlyList<NemotronHLayerWeights> LayerWeights => _layers;
    private readonly float[] _outputNormWeight;

    private readonly nint _tokenEmbedWeight;
    private readonly QuantizationType _tokenEmbedQuantType;
    private readonly nint _outputWeight;
    private readonly QuantizationType _outputQuantType;
    private readonly int _outputOutputDim;
    private readonly int _outputInputDim;

    private readonly HybridLayerLayout _layout;
    private readonly MambaSsmConfig _ssm;

    // Sparse KV-cache mapping: physical layer index -> slot in a cache sized to
    // the attention-layer count. -1 for non-attention layers.
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;

    // Ordinal mapping: physical layer index -> index into _ssmCache for SSM layers; -1 otherwise.
    private readonly int[] _ssmLayerOrdinal;
    private readonly int _numSsmLayers;

    // #386: compute pool for the Gemm hot path. Null = single-threaded (tests,
    // prebuilt-weights path). Owned iff created by LoadFromGguf.
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    private readonly NemotronHForwardState _state;
    private readonly SsmStateCache _ssmCache;

    // Caller-supplied per-sequence SSM state for the in-flight Forward, set by the
    // Forward(...,ISsmState?) overload and consumed by ForwardSsmBody (instance-scoped, single-threaded
    // per generation — same pattern as the model's other forward-time state). Null ⇒ use _ssmCache.
    private SsmStateCache? _activeSsm;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _ssmCache.AllocatedBytes;

    /// <summary>Number of attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    private NemotronHTransformerModel(
        ModelConfig config,
        GgufFile? gguf,
        NemotronHLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        int[] kvSlotForLayer, int attentionLayerCount,
        ComputeThreadPool? threadPool = null, bool ownsThreadPool = false)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _outputNormWeight = outputNormWeight;
        _tokenEmbedWeight = tokenEmbedWeight;
        _tokenEmbedQuantType = tokenEmbedQuantType;
        _outputWeight = outputWeight;
        _outputQuantType = outputQuantType;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _layout = config.HybridLayout!;
        _ssm = config.SsmConfig!.Value;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _threadPool = threadPool;
        _ownsThreadPool = ownsThreadPool;

        _ssmLayerOrdinal = new int[config.NumLayers];
        int ssmOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _ssmLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.Ssm
                ? ssmOrdinal++
                : -1;
        }
        _numSsmLayers = ssmOrdinal;

        int maxIntermediate = 0;
        for (int i = 0; i < _layers.Length; i++)
        {
            var ffn = _layers[i].Ffn;
            if (ffn is not null && ffn.UpOutputDim > maxIntermediate)
                maxIntermediate = ffn.UpOutputDim;
        }
        if (maxIntermediate == 0) maxIntermediate = config.HiddenSize;

        _state = new NemotronHForwardState(
            hiddenSize: config.HiddenSize,
            maxIntermediateSize: maxIntermediate,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            inputProjectionDim: _ssm.InputProjectionDim,
            convDim: _ssm.ConvDim,
            dConv: _ssm.DConv,
            dInner: _ssm.DInner,
            nHead: _ssm.NHead,
            nGroup: _ssm.NGroup,
            dState: _ssm.DState);

        _ssmCache = new SsmStateCache(_ssm, _numSsmLayers);
    }

    /// <summary>
    /// Loads a Nemotron-H model from an opened GGUF file. The <paramref name="gguf"/> must
    /// remain alive for the lifetime of the returned model.
    /// </summary>
    public static NemotronHTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a Nemotron-H / Nemotron-H-MoE model with threading configuration (#386).
    /// Mirrors the Qwen hybrid loaders: a parallel config creates a ComputeThreadPool
    /// that every projection GEMM dispatches through (row-partitioned, so results are
    /// bit-identical to the single-threaded path).
    /// </summary>
    public static NemotronHTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        if (config.Architecture is not (Architecture.NemotronH or Architecture.NemotronHMoe))
            throw new ArgumentException(
                $"NemotronHTransformerModel requires Architecture.NemotronH/NemotronHMoe, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));

        var dataBase = new GgufDataBase(gguf);
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout;
        var ssm = config.SsmConfig.Value;

        // #1038: NVFP4 weights carry a per-tensor X.scale. This model applies the SSM projections' scales
        // (ForwardSsmBody); a scaled NVFP4 weight anywhere else would run unscaled, so refuse it here.
        Nvfp4Support.RejectUnappliedScales(tensors, SsmProjectionScaleNames(tensors), "The Nemotron-H CPU model");

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase.Of(embDesc);

        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeF32(dataBase, outNormDesc, config.HiddenSize);

        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outputPtr = dataBase.Of(outDesc);
            outputQt = outDesc.QuantizationType;
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        // NO RoPE for nemotron_h (issue #372): llama.cpp's nemotron-h.cpp and HF's
        // NemotronHAttention apply no position encoding on the attention layers —
        // position information comes entirely from the Mamba2 layers. The GGUF
        // rope.* keys (rope.dimension_count = head_dim) are converter artifacts
        // llama.cpp never consumes, so they are neither required nor validated here.

        var layers = new NemotronHLayerWeights[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config, layout, ssm);
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        ComputeThreadPool? pool = CreatePool(threading);

        return new NemotronHTransformerModel(
            config, gguf, layers, outputNormWeight,
            embPtr, embDesc.QuantizationType,
            outputPtr, outputQt, outputM, outputK,
            kvSlotForLayer, attentionLayerCount,
            pool, ownsThreadPool: pool is not null);
    }

    private static ComputeThreadPool? CreatePool(ThreadingConfig threading)
    {
        if (!threading.IsParallel)
            return null;

        int effectiveThreads = threading.EffectiveThreadCount;
        if (threading.EnableNumaPinning || threading.EnablePCorePinning)
        {
            var topology = NumaTopology.Detect();
            if (threading.EnablePCorePinning && topology.IsHybrid)
                effectiveThreads = Math.Min(effectiveThreads, topology.PerformanceCoreIds.Count);
            return new ComputeThreadPool(effectiveThreads, topology, threading);
        }

        return new ComputeThreadPool(effectiveThreads, topology: null, threading);
    }

    /// <summary>
    /// Builds a NemotronH model from caller-owned, pre-dequantised weight pointers — used by the
    /// Vulkan parity tests that construct synthetic weight banks in unmanaged memory directly,
    /// bypassing the GGUF/safetensors loaders. Caller retains ownership of every <see cref="nint"/>
    /// pointer (token embed, output, plus every projection inside <paramref name="layers"/>).
    /// </summary>
    /// <remarks>
    /// Unlike <c>LoadFromGguf</c> there is no <see cref="GgufFile"/> to keep alive — the
    /// model holds <c>null</c> for the gguf reference. Disposing this model frees only the forward
    /// scratch and the SSM cache; weight memory belongs to the caller.
    /// </remarks>
    internal static NemotronHTransformerModel BuildFromPrebuiltWeights(
        ModelConfig config,
        NemotronHLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim)
    {
        if (config.Architecture is not (Architecture.NemotronH or Architecture.NemotronHMoe))
            throw new ArgumentException(
                $"NemotronHTransformerModel requires Architecture.NemotronH/NemotronHMoe, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));
        if (layers.Length != config.NumLayers)
            throw new ArgumentException(
                $"layers length {layers.Length} != config.NumLayers {config.NumLayers}.", nameof(layers));

        var layout = config.HybridLayout!;

        // Compute per-layer KV slot map (sparse over attention layers).
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        // Any RoPEConfig on the ModelConfig is ignored — nemotron_h applies no
        // position encoding on attention (issue #372; see ForwardAttentionBody).
        return new NemotronHTransformerModel(
            config, gguf: null, layers, outputNormWeight,
            tokenEmbedWeight, tokenEmbedQuantType,
            outputWeight, outputQuantType, outputOutputDim, outputInputDim,
            kvSlotForLayer, attentionLayerCount);
    }

    private static NemotronHLayerWeights LoadLayer(
        int layerIdx,
        GgufDataBase dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        HybridLayerLayout layout,
        MambaSsmConfig ssm)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Pre-sublayer RMSNorm is always named attn_norm in GGUF, even on FFN and SSM layers.
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNormWeight = DequantizeF32(dataBase, attnNormDesc, hiddenSize);

        return layout.LayerKind[layerIdx] switch
        {
            HybridLayerKind.Ssm => new NemotronHLayerWeights
            {
                AttnNormWeight = attnNormWeight,
                Ssm = LoadSsmLayer(prefix, dataBase, tensors, ssm),
            },
            HybridLayerKind.Attention => new NemotronHLayerWeights
            {
                AttnNormWeight = attnNormWeight,
                Attention = LoadAttentionLayer(prefix, dataBase, tensors, config, layout.HeadCountKv[layerIdx]),
            },
            // An FFN-classified layer is MoE iff the router tensor exists (nemotron_h_moe).
            HybridLayerKind.Ffn when tensors.ContainsKey($"{prefix}.ffn_gate_inp.weight") =>
                new NemotronHLayerWeights
                {
                    AttnNormWeight = attnNormWeight,
                    Moe = LoadMoeLayer(prefix, layerIdx, dataBase, tensors, config),
                },
            HybridLayerKind.Ffn => new NemotronHLayerWeights
            {
                AttnNormWeight = attnNormWeight,
                Ffn = LoadFfnLayer(prefix, dataBase, tensors, layout.FeedForwardLength[layerIdx]),
            },
            _ => throw new InvalidOperationException(
                $"Unknown HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx}."),
        };
    }

    private static NemotronHSsmWeights LoadSsmLayer(
        string prefix, GgufDataBase dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        MambaSsmConfig ssm)
    {
        var inDesc = tensors[$"{prefix}.ssm_in.weight"];
        var conv1dWDesc = tensors[$"{prefix}.ssm_conv1d.weight"];
        var conv1dBDesc = tensors[$"{prefix}.ssm_conv1d.bias"];
        var aDesc = tensors[$"{prefix}.ssm_a"];
        var dDesc = tensors[$"{prefix}.ssm_d"];
        var dtBDesc = tensors[$"{prefix}.ssm_dt.bias"];
        var normDesc = tensors[$"{prefix}.ssm_norm.weight"];
        var outDesc = tensors[$"{prefix}.ssm_out.weight"];

        if (inDesc.Shape[1] != ssm.InputProjectionDim)
            throw new InvalidDataException(
                $"{prefix}.ssm_in.weight shape[1]={inDesc.Shape[1]}, expected {ssm.InputProjectionDim}.");
        if (conv1dWDesc.Shape[0] != ssm.DConv || conv1dWDesc.Shape[1] != ssm.ConvDim)
            throw new InvalidDataException(
                $"{prefix}.ssm_conv1d.weight shape [{conv1dWDesc.Shape[0]},{conv1dWDesc.Shape[1]}], expected [{ssm.DConv},{ssm.ConvDim}].");
        if (conv1dBDesc.Shape.ElementCount != ssm.ConvDim)
            throw new InvalidDataException(
                $"{prefix}.ssm_conv1d.bias length {conv1dBDesc.Shape.ElementCount}, expected {ssm.ConvDim}.");
        if (aDesc.Shape.ElementCount != ssm.NHead)
            throw new InvalidDataException(
                $"{prefix}.ssm_a length {aDesc.Shape.ElementCount}, expected {ssm.NHead}.");
        if (dDesc.Shape.ElementCount != ssm.NHead)
            throw new InvalidDataException(
                $"{prefix}.ssm_d length {dDesc.Shape.ElementCount}, expected {ssm.NHead}.");
        if (dtBDesc.Shape.ElementCount != ssm.NHead)
            throw new InvalidDataException(
                $"{prefix}.ssm_dt.bias length {dtBDesc.Shape.ElementCount}, expected {ssm.NHead}.");

        return new NemotronHSsmWeights
        {
            InWeight = dataBase.Of(inDesc),
            InQuantType = inDesc.QuantizationType,
            InInputDim = inDesc.Shape[0],
            InOutputDim = inDesc.Shape[1],
            Conv1dWeight = DequantizeF32(dataBase, conv1dWDesc, ssm.DConv * ssm.ConvDim),
            Conv1dBias = DequantizeF32(dataBase, conv1dBDesc, ssm.ConvDim),
            A = DequantizeF32(dataBase, aDesc, ssm.NHead),
            D = DequantizeF32(dataBase, dDesc, ssm.NHead),
            DtBias = DequantizeF32(dataBase, dtBDesc, ssm.NHead),
            NormWeight = DequantizeF32(dataBase, normDesc, ssm.DInner),
            OutWeight = dataBase.Of(outDesc),
            OutQuantType = outDesc.QuantizationType,
            OutInputDim = outDesc.Shape[0],
            OutOutputDim = outDesc.Shape[1],
            InScale = ReadOptionalScale(dataBase, tensors, $"{prefix}.ssm_in.weight"),
            OutScale = ReadOptionalScale(dataBase, tensors, $"{prefix}.ssm_out.weight"),
        };
    }

    /// <summary>
    /// The per-tensor <c>X.scale</c> of <paramref name="weightName"/> (llama.cpp loads these optionally for
    /// every projection and multiplies them into the matmul output), or 1 when the GGUF has none.
    /// </summary>
    private static float ReadOptionalScale(GgufDataBase dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string weightName)
    {
        if (!tensors.TryGetValue(Nvfp4Support.ScaleNameFor(weightName), out var desc))
            return 1.0f;
        if (desc.Shape.ElementCount != 1)
            throw new InvalidDataException(
                $"{desc.Name} has {desc.Shape.ElementCount} elements; a per-tensor weight scale has exactly 1.");
        return DequantizeF32(dataBase, desc, 1)[0];
    }

    /// <summary>The <c>ssm_in.scale</c> / <c>ssm_out.scale</c> names present in the file — the scales this model applies.</summary>
    private static HashSet<string> SsmProjectionScaleNames(IReadOnlyDictionary<string, GgufTensorDescriptor> tensors)
        => tensors.Keys
            .Where(n => n.StartsWith("blk.", StringComparison.Ordinal)
                && (n.EndsWith(".ssm_in.scale", StringComparison.Ordinal) || n.EndsWith(".ssm_out.scale", StringComparison.Ordinal)))
            .ToHashSet(StringComparer.Ordinal);

    private static NemotronHAttentionWeights LoadAttentionLayer(
        string prefix, GgufDataBase dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, int numKvHeads)
    {
        var q = tensors[$"{prefix}.attn_q.weight"];
        var k = tensors[$"{prefix}.attn_k.weight"];
        var v = tensors[$"{prefix}.attn_v.weight"];
        var o = tensors[$"{prefix}.attn_output.weight"];

        return new NemotronHAttentionWeights
        {
            QWeight = dataBase.Of(q),
            QQuantType = q.QuantizationType,
            QInputDim = q.Shape[0],
            QOutputDim = q.Shape[1],

            KWeight = dataBase.Of(k),
            KQuantType = k.QuantizationType,
            KInputDim = k.Shape[0],
            KOutputDim = k.Shape[1],

            VWeight = dataBase.Of(v),
            VQuantType = v.QuantizationType,
            VInputDim = v.Shape[0],
            VOutputDim = v.Shape[1],

            OWeight = dataBase.Of(o),
            OQuantType = o.QuantizationType,
            OInputDim = o.Shape[0],
            OOutputDim = o.Shape[1],

            NumKvHeads = numKvHeads,
        };
    }

    private static NemotronHFfnWeights LoadFfnLayer(
        string prefix, GgufDataBase dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        int intermediateSize)
    {
        if (tensors.ContainsKey($"{prefix}.ffn_gate.weight"))
            throw new InvalidDataException(
                $"{prefix}.ffn_gate.weight present — Nemotron-H FFN is non-gated (squared-ReLU); " +
                "this GGUF does not match the architecture's tensor convention.");

        var up = tensors[$"{prefix}.ffn_up.weight"];
        var down = tensors[$"{prefix}.ffn_down.weight"];

        return new NemotronHFfnWeights
        {
            UpWeight = dataBase.Of(up),
            UpQuantType = up.QuantizationType,
            UpInputDim = up.Shape[0],
            UpOutputDim = up.Shape[1],

            DownWeight = dataBase.Of(down),
            DownQuantType = down.QuantizationType,
            DownInputDim = down.Shape[0],
            DownOutputDim = down.Shape[1],

            IntermediateSize = intermediateSize,
        };
    }

    /// <summary>
    /// Loads a nemotron_h_moe routed-MoE FFN layer (issue #375). Requires
    /// <see cref="ModelConfig.Moe"/> (populated by the GGUF extractor) and the six
    /// MoE tensors llama.cpp's nemotron-h graph consumes: router, selection bias, fused
    /// up/down expert banks, and the shared expert's up/down.
    /// </summary>
    private static NemotronHMoeWeights LoadMoeLayer(
        string prefix, int layerIdx, GgufDataBase dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        var moeCfg = config.Moe ?? throw new InvalidDataException(
            $"{prefix}: ffn_gate_inp.weight present but ModelConfig.MoeConfig is null — " +
            "the GGUF is missing the expert_* metadata keys.");

        var gateInp = tensors[$"{prefix}.ffn_gate_inp.weight"];
        var probsB = tensors[$"{prefix}.exp_probs_b.bias"];
        var upExps = tensors[$"{prefix}.ffn_up_exps.weight"];
        var downExps = tensors[$"{prefix}.ffn_down_exps.weight"];
        var upShexp = tensors[$"{prefix}.ffn_up_shexp.weight"];
        var downShexp = tensors[$"{prefix}.ffn_down_shexp.weight"];

        int hidden = config.HiddenSize;
        int moeInter = moeCfg.IntermediateSizeForLayer(layerIdx);
        int topK = moeCfg.ExpertsPerTokForLayer(layerIdx);
        int latent = moeCfg.LatentSize;

        // With LatentMoE (#1037) the routed experts read and write the latent, not the hidden state.
        int expertDim = latent > 0 ? latent : hidden;
        bool hasDown = tensors.TryGetValue($"{prefix}.ffn_latent_down.weight", out var latentDownDesc);
        bool hasUp = tensors.TryGetValue($"{prefix}.ffn_latent_up.weight", out var latentUpDesc);
        if (latent == 0 && (hasDown || hasUp))
        {
            // Latent tensors without moe_latent_size: running the experts on the full hidden
            // state would read the banks with the wrong row width. Refuse rather than guess.
            throw new InvalidDataException(
                $"{prefix}: ffn_latent_down/ffn_latent_up present but moe_latent_size is 0 or missing.");
        }

        // The expert count is per layer on Puzzle derivatives (Nemotron 3.5 Bolt: 240..496,
        // with expert_count = the maximum) — take it from the bank, as llama.cpp does
        // (ffn_up_exps->ne[2]), and bound it by the metadata maximum.
        int nExpert = upExps.Shape.Rank == 3 ? upExps.Shape[2] : -1;
        if (nExpert <= 0 || nExpert > moeCfg.NumExperts)
            throw new InvalidDataException(
                $"{prefix}.ffn_up_exps.weight expert dimension {nExpert} is outside 1..expert_count ({moeCfg.NumExperts}).");
        if (topK <= 0 || topK > nExpert)
            throw new InvalidDataException(
                $"{prefix}: expert_used_count {topK} is out of range for this layer's {nExpert} experts.");

        if (upExps.Shape[0] != expertDim || upExps.Shape[1] != moeInter)
            throw new InvalidDataException(
                $"{prefix}.ffn_up_exps.weight has unexpected shape; expected [{expertDim},{moeInter},{nExpert}].");
        if (downExps.Shape.Rank != 3 || downExps.Shape[0] != moeInter || downExps.Shape[1] != expertDim || downExps.Shape[2] != nExpert)
            throw new InvalidDataException(
                $"{prefix}.ffn_down_exps.weight has unexpected shape; expected [{moeInter},{expertDim},{nExpert}].");
        if (gateInp.Shape.Rank != 2 || gateInp.Shape[0] != hidden || gateInp.Shape[1] != nExpert)
            throw new InvalidDataException(
                $"{prefix}.ffn_gate_inp.weight has unexpected shape; expected [{hidden},{nExpert}].");
        if (probsB.Shape.ElementCount != nExpert)
            throw new InvalidDataException(
                $"{prefix}.exp_probs_b.bias length {probsB.Shape.ElementCount}, expected {nExpert}.");

        int sharedInter = upShexp.Shape[1];

        nint latentDown = 0, latentUp = 0;
        QuantizationType latentDownQt = default, latentUpQt = default;
        if (latent > 0)
        {
            if (!hasDown || !hasUp)
                throw new InvalidDataException(
                    $"{prefix}: moe_latent_size={latent} but ffn_latent_down/ffn_latent_up are missing.");
            if (latent > hidden)
            {
                // The per-token MoE body sizes its expert-output scratch by the hidden size; a
                // latent wider than hidden is not a bottleneck and would overrun that scratch.
                throw new InvalidDataException(
                    $"{prefix}: moe_latent_size={latent} exceeds the hidden size {hidden}.");
            }
            if (latentDownDesc!.Shape.Rank != 2 || latentDownDesc.Shape[0] != hidden || latentDownDesc.Shape[1] != latent)
                throw new InvalidDataException(
                    $"{prefix}.ffn_latent_down.weight has unexpected shape; expected [{hidden},{latent}].");
            if (latentUpDesc!.Shape.Rank != 2 || latentUpDesc.Shape[0] != latent || latentUpDesc.Shape[1] != hidden)
                throw new InvalidDataException(
                    $"{prefix}.ffn_latent_up.weight has unexpected shape; expected [{latent},{hidden}].");
            latentDown = dataBase.Of(latentDownDesc);
            latentDownQt = latentDownDesc.QuantizationType;
            latentUp = dataBase.Of(latentUpDesc);
            latentUpQt = latentUpDesc.QuantizationType;
        }

        return new NemotronHMoeWeights
        {
            GateInpWeight = dataBase.Of(gateInp),
            GateInpQuantType = gateInp.QuantizationType,
            SelectionBias = DequantizeF32(dataBase, probsB, nExpert),
            UpExpsWeight = dataBase.Of(upExps),
            UpExpsQuantType = upExps.QuantizationType,
            UpPerExpertBytes = Dequantize.RowByteSize(expertDim, upExps.QuantizationType) * moeInter,
            DownExpsWeight = dataBase.Of(downExps),
            DownExpsQuantType = downExps.QuantizationType,
            DownPerExpertBytes = Dequantize.RowByteSize(moeInter, downExps.QuantizationType) * expertDim,
            UpShexpWeight = dataBase.Of(upShexp),
            UpShexpQuantType = upShexp.QuantizationType,
            DownShexpWeight = dataBase.Of(downShexp),
            DownShexpQuantType = downShexp.QuantizationType,
            NumExperts = nExpert,
            NumExpertsPerTok = topK,
            MoeIntermediateSize = moeInter,
            SharedIntermediateSize = sharedInter,
            NormalizeWeights = moeCfg.NormalizeExpertWeights,
            WeightsScale = moeCfg.ExpertWeightsScale,
            LatentSize = latent,
            LatentDownWeight = latentDown,
            LatentDownQuantType = latentDownQt,
            LatentUpWeight = latentUp,
            LatentUpQuantType = latentUpQt,
        };
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;

        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        _state.EnsureCapacity(seqLen);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* ffnMid = (float*)_state.FfnIntermediate;
        float* logits = (float*)_state.Logits;
        byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;
        float* q = (float*)_state.QScratch;
        float* k = (float*)_state.KScratch;
        float* v = (float*)_state.VScratch;
        float* attnOut = (float*)_state.AttnOutput;

        EmbedTokens(tokenIds, hidden, hiddenSize);

        if (NemotronHDiagnostics.TraceLayers)
            NemotronHDiagnostics.DumpStats("embed[last]", hidden + (seqLen - 1) * hiddenSize, hiddenSize);

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];

            // Save residual snapshot and pre-norm into normOut (shared by all three sub-layers).
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            char kindTag;
            switch (kinds[layer])
            {
                case HybridLayerKind.Ffn when lw.Moe is not null:
                    if (seqLen > 1 && GroupedMoePrefill)
                        ForwardMoeFfnBodyGrouped(lw.Moe, seqLen, hiddenSize, normOut);
                    else
                        ForwardMoeFfnBody(lw.Moe, seqLen, hiddenSize, normOut);
                    kindTag = 'M';
                    break;
                case HybridLayerKind.Ffn:
                    ForwardFfnBody(lw.Ffn!, seqLen, hiddenSize, normOut, ffnMid, inputQ8Scratch);
                    kindTag = 'F';
                    break;
                case HybridLayerKind.Attention:
                    ForwardAttentionBody(lw.Attention!, layer, seqLen, positions,
                        normOut, q, k, v, attnOut,
                        numHeads, numKvHeads, headDim, kvCache);
                    kindTag = 'A';
                    break;
                case HybridLayerKind.Ssm:
                    ForwardSsmBody(lw.Ssm!, layer, seqLen, hiddenSize, normOut, eps);
                    kindTag = 'S';
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {kinds[layer]} at layer {layer}.");
            }

            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            if (NemotronHDiagnostics.TraceLayers)
                NemotronHDiagnostics.DumpStats($"L{layer:D2}{kindTag}[last]", hidden + (seqLen - 1) * hiddenSize, hiddenSize);
        }

        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                _outputNormWeight, eps,
                new Span<float>(hidden + t * hiddenSize, hiddenSize));
        }

        Gemm(_outputWeight, _outputQuantType, hidden, logits,
             _outputOutputDim, _outputInputDim, seqLen, preQuantizedInput: null);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));

        return result;
    }

    /// <summary>
    /// Forward with a caller-supplied per-sequence SSM state (the per-token recurrent state that the
    /// continuous-batch scheduler threads so concurrent sequences don't share the model-owned default).
    /// <paramref name="ssmState"/> null falls back to the model-owned <c>_ssmCache</c> (single-sequence
    /// behaviour). Attention layers use <paramref name="kvCache"/> as usual.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ISsmState? ssmState)
    {
        SsmStateCache? prev = _activeSsm;
        _activeSsm = ResolveSsm(ssmState);
        try { return Forward(tokenIds, positions, deviceId, kvCache); }
        finally { _activeSsm = prev; }
    }

    private SsmStateCache? ResolveSsm(ISsmState? ssmState)
    {
        if (ssmState is null) return null; // use _ssmCache
        if (ssmState is SsmStateCache cache)
        {
            if (cache.NumSsmLayers != _numSsmLayers)
                throw new ArgumentException(
                    $"SsmState covers {cache.NumSsmLayers} SSM layers but this model has {_numSsmLayers}.",
                    nameof(ssmState));
            return cache;
        }
        throw new ArgumentException(
            $"NemotronHTransformerModel requires an {nameof(SsmStateCache)} for its SSM state; got {ssmState.GetType().Name}.",
            nameof(ssmState));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Re-zeroes the model-owned SSM cache (conv history + hidden state) used by every forward that
    /// does not carry a caller-supplied <see cref="ISsmState"/>. Callers that treat each forward as
    /// an independent sequence (perplexity windows) must call this between sequences — issue #261.
    /// </remarks>
    public void ResetSequenceState() => _ssmCache.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => new SsmStateCache(_ssm, _numSsmLayers);

    /// <summary>
    /// Batched forward across sequences. Nemotron-H SSM state is per-token recurrent, so this threads
    /// each request's per-seq <see cref="SequenceForwardRequest.SsmState"/> through a per-sequence
    /// <c>Forward</c> (no cross-sequence fusion — same as Mamba-3). For 2+ requests every entry must
    /// supply a per-seq <c>SsmState</c> (a null would silently share the model-owned default and corrupt
    /// concurrent decode). LoRA adapters are not supported.
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "NemotronHTransformerModel.ForwardBatch does not support LoRA adapters.");
        }

        if (requests.Count >= 2)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                if (requests[i].SsmState is null)
                    throw new ArgumentException(
                        $"NemotronHTransformerModel.ForwardBatch with {requests.Count} requests requires " +
                        $"every request to supply a per-seq SsmState; request[{i}] has none.",
                        nameof(requests));
            }
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.SsmState);
        }
        return results;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ForwardFfnBody(NemotronHFfnWeights ffn, int seqLen, int hiddenSize,
                                       float* normOut, float* ffnMid, byte* inputQ8Scratch)
    {
        int intermediateSize = ffn.UpOutputDim;

        byte* preQuantUp = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, ffn.UpQuantType);
        Gemm(ffn.UpWeight, ffn.UpQuantType, normOut, ffnMid,
             ffn.UpOutputDim, ffn.UpInputDim, seqLen, preQuantUp);

        for (int t = 0; t < seqLen; t++)
        {
            ReluSquared.Execute(
                new ReadOnlySpan<float>(ffnMid + t * intermediateSize, intermediateSize),
                new Span<float>(ffnMid + t * intermediateSize, intermediateSize));
        }

        byte* preQuantDown = QuantizeInput(ffnMid, inputQ8Scratch, intermediateSize, seqLen, ffn.DownQuantType);
        Gemm(ffn.DownWeight, ffn.DownQuantType, ffnMid, normOut,
             ffn.DownOutputDim, ffn.DownInputDim, seqLen, preQuantDown);
    }

    /// <summary>
    /// nemotron_h_moe routed-MoE FFN body (issue #375), anchored to llama.cpp's
    /// <c>build_moe_ffn</c> as called from <c>nemotron-h.cpp</c>'s <c>build_ffn_layer</c>:
    /// router on the FULL hidden input, sigmoid, plus selection bias (SELECTION ONLY),
    /// top-k, weights = UNBIASED probabilities of the selected experts, optional
    /// renorm by their sum (denominator clamped like llama.cpp's F16-min clamp),
    /// times weights scale, weighted sum of ungated relu-squared experts, plus shared
    /// relu-squared expert. Reads pre-normed activations from <paramref name="normOut"/>
    /// and writes the result back to the same buffer (same contract as the other
    /// sub-layer bodies). Correctness-first: per-token GEMVs into pooled scratch.
    /// </summary>
    private void ForwardMoeFfnBody(NemotronHMoeWeights moe, int seqLen, int hiddenSize, float* normOut)
    {
        int nExpert = moe.NumExperts;
        int topK = moe.NumExpertsPerTok;
        int moeInter = moe.MoeIntermediateSize;
        int sharedInter = moe.SharedIntermediateSize;

        // LatentMoE (#1037): the routed experts read latent_down(h) and their weighted sum is
        // projected back by latent_up; the router and shared expert keep the full hidden h.
        int latent = moe.LatentSize;
        int expertDim = latent > 0 ? latent : hiddenSize;

        var pool = System.Buffers.ArrayPool<float>.Shared;
        float[] hBuf = pool.Rent(hiddenSize);
        float[] latentBuf = pool.Rent(Math.Max(latent, 1));
        float[] logitsBuf = pool.Rent(nExpert);
        float[] selBuf = pool.Rent(nExpert);
        float[] midBuf = pool.Rent(Math.Max(moeInter, sharedInter));
        float[] outBuf = pool.Rent(hiddenSize);
        float[] accBuf = pool.Rent(hiddenSize);
        int[] topIdx = new int[topK];
        try
        {
            fixed (float* h = hBuf, logits = logitsBuf, mid = midBuf, expOut = outBuf, xLatent = latentBuf, acc = accBuf)
            {
                // Routed experts read this: the latent projection when present, else h itself.
                float* expertIn = latent > 0 ? xLatent : h;
                Span<float> w = stackalloc float[topK];
                for (int t = 0; t < seqLen; t++)
                {
                    // normOut doubles as the output buffer, so snapshot the token's input.
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize)
                        .CopyTo(hBuf.AsSpan(0, hiddenSize));

                    // Router logits on the FULL hidden input, then routing.
                    Gemm(moe.GateInpWeight, moe.GateInpQuantType, h, logits, nExpert, hiddenSize, 1, null);
                    RouteToken(moe, logitsBuf.AsSpan(0, nExpert), selBuf.AsSpan(0, nExpert), topIdx, w);

                    if (latent > 0)
                        Gemm(moe.LatentDownWeight, moe.LatentDownQuantType, h, xLatent, latent, hiddenSize, 1, null);

                    accBuf.AsSpan(0, expertDim).Clear();

                    // Routed experts: ungated relu-squared MLP per selected expert, weighted sum
                    // (accumulated in the latent space when the layer has one).
                    for (int k2 = 0; k2 < topK; k2++)
                    {
                        int e = topIdx[k2];
                        NemotronHMoeExpertCompute.Compute(moe, e, expertDim, expertIn, 1, mid, expOut, _threadPool);

                        float wk = w[k2];
                        for (int i = 0; i < expertDim; i++) accBuf[i] += wk * expOut[i];
                    }

                    if (latent > 0)
                    {
                        // Back to the hidden size: acc[0..latent) -> expOut[0..hidden) -> acc.
                        Gemm(moe.LatentUpWeight, moe.LatentUpQuantType, acc, expOut, hiddenSize, latent, 1, null);
                        new ReadOnlySpan<float>(expOut, hiddenSize).CopyTo(accBuf.AsSpan(0, hiddenSize));
                    }

                    // Shared expert (same ungated relu-squared, on the full input), added unweighted.
                    Gemm(moe.UpShexpWeight, moe.UpShexpQuantType, h, mid, sharedInter, hiddenSize, 1, null);
                    ReluSquared.Execute(midBuf.AsSpan(0, sharedInter), midBuf.AsSpan(0, sharedInter));
                    Gemm(moe.DownShexpWeight, moe.DownShexpQuantType, mid, expOut, hiddenSize, sharedInter, 1, null);
                    for (int i = 0; i < hiddenSize; i++) accBuf[i] += expOut[i];

                    accBuf.AsSpan(0, hiddenSize).CopyTo(new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }
            }
        }
        finally
        {
            pool.Return(hBuf); pool.Return(latentBuf); pool.Return(logitsBuf); pool.Return(selBuf);
            pool.Return(midBuf); pool.Return(outBuf); pool.Return(accBuf);
        }
    }

    /// <summary>
    /// nemotron_h_moe routing for one token (llama.cpp <c>build_moe_ffn</c>, sigmoid gating):
    /// sigmoid over the router logits, top-k on probabilities + selection bias (the bias
    /// biases SELECTION ONLY), gating weights = the UNBIASED probabilities of the chosen
    /// experts, optional renorm by their sum (denominator clamped to the smallest normal F16,
    /// as llama.cpp does), then the weights scale. Shared by the per-token and the grouped
    /// (#1039) MoE bodies so both route identically.
    /// </summary>
    /// <param name="moe">The layer's MoE weights (bias, top-k, renorm and scale).</param>
    /// <param name="logits">Router logits for the token; overwritten with sigmoid probabilities.</param>
    /// <param name="selScratch">Scratch of length <c>NumExperts</c> for the selection scores.</param>
    /// <param name="topIdx">Receives the selected expert ids, in selection order.</param>
    /// <param name="weights">Receives the gating weight of each selected expert.</param>
    /// <remarks>Internal so the CUDA model routes on the host with this exact code (#403).</remarks>
    internal static void RouteToken(
        NemotronHMoeWeights moe, Span<float> logits, Span<float> selScratch, Span<int> topIdx, Span<float> weights)
    {
        int nExpert = moe.NumExperts;
        int topK = moe.NumExpertsPerTok;
        System.Numerics.Tensors.TensorPrimitives.Sigmoid(logits, logits);

        for (int e = 0; e < nExpert; e++)
            selScratch[e] = logits[e] + moe.SelectionBias[e];

        // Top-k by repeated max-scan; ties go to the lower expert id.
        for (int k2 = 0; k2 < topK; k2++)
        {
            int best = -1;
            float bestV = float.NegativeInfinity;
            for (int e = 0; e < nExpert; e++)
            {
                if (selScratch[e] > bestV) { bestV = selScratch[e]; best = e; }
            }
            topIdx[k2] = best;
            selScratch[best] = float.NegativeInfinity;
        }

        RouteTrace.Record(moe, topIdx[..topK]);

        float sum = 0f;
        for (int k2 = 0; k2 < topK; k2++) { weights[k2] = logits[topIdx[k2]]; sum += weights[k2]; }
        if (moe.NormalizeWeights)
        {
            float denom = MathF.Max(sum, 6.103515625e-5f);
            for (int k2 = 0; k2 < topK; k2++) weights[k2] /= denom;
        }
        if (moe.WeightsScale != 0f && moe.WeightsScale != 1f)
        {
            for (int k2 = 0; k2 < topK; k2++) weights[k2] *= moe.WeightsScale;
        }
    }

    /// <summary>
    /// <c>DOTLLM_NEMOTRON_MOE_GROUPED=0</c> forces the per-token MoE body for every batch size
    /// (the reference arm for #1039). Settable from tests.
    /// </summary>
    internal static bool GroupedMoePrefill { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_NEMOTRON_MOE_GROUPED") != "0";

    /// <summary>Token block size for the grouped MoE body; bounds its scratch to O(block x topK x expertDim).</summary>
    internal static int GroupedMoeBlockTokens { get; set; } = 512;

    /// <summary>
    /// Grouped-by-expert MoE FFN body for prefill (issue #1039): same maths as
    /// <see cref="ForwardMoeFfnBody"/>, but the router, shared expert and latent projections run
    /// as one GEMM over the whole token block, and each USED routed expert runs one up/down GEMM
    /// over all tokens routed to it (the grouping llama.cpp's <c>ggml_mul_mat_id</c> does) instead
    /// of one GEMV per (token, slot). Slot outputs are summed per token in slot order, the same
    /// order as the per-token body.
    /// </summary>
    private void ForwardMoeFfnBodyGrouped(NemotronHMoeWeights moe, int seqLen, int hiddenSize, float* normOut)
    {
        for (int start = 0; start < seqLen; start += GroupedMoeBlockTokens)
        {
            int len = Math.Min(GroupedMoeBlockTokens, seqLen - start);
            ForwardMoeFfnBlockGrouped(moe, len, hiddenSize, normOut + (long)start * hiddenSize);
        }
    }

    [SkipLocalsInit]
    private void ForwardMoeFfnBlockGrouped(NemotronHMoeWeights moe, int n, int hiddenSize, float* normOut)
    {
        int nExpert = moe.NumExperts;
        int topK = moe.NumExpertsPerTok;
        int moeInter = moe.MoeIntermediateSize;
        int sharedInter = moe.SharedIntermediateSize;
        int latent = moe.LatentSize;
        int expertDim = latent > 0 ? latent : hiddenSize;
        int slots = n * topK;

        // One native scratch block, carved into the buffers below (all float-sized).
        long inF = (long)n * hiddenSize, logitsF = (long)n * nExpert, latF = latent > 0 ? (long)n * latent : 0;
        long shMidF = (long)n * Math.Max(sharedInter, 1), shOutF = (long)n * hiddenSize;
        long wF = slots, slotOutF = (long)slots * expertDim, gInF = (long)n * expertDim;
        long gMidF = (long)n * moeInter, gOutF = (long)n * expertDim, accF = (long)n * expertDim;
        long selF = nExpert;
        long total = inF + logitsF + latF + shMidF + shOutF + wF + slotOutF + gInF + gMidF + gOutF + accF + selF;
        float* block = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
        int[] topIdx = System.Buffers.ArrayPool<int>.Shared.Rent(slots);
        int[] counts = System.Buffers.ArrayPool<int>.Shared.Rent(nExpert + 1);
        int[] fill = System.Buffers.ArrayPool<int>.Shared.Rent(nExpert);
        int[] order = System.Buffers.ArrayPool<int>.Shared.Rent(slots);
        try
        {
            float* x = block;                    // [n, hidden]  copy of the input (normOut is the output)
            float* logits = x + inF;             // [n, nExpert]
            float* xl = logits + logitsF;        // [n, latent]  latent_down(x), when present
            float* shMid = xl + latF;            // [n, sharedInter]
            float* shOut = shMid + shMidF;       // [n, hidden]
            float* w = shOut + shOutF;           // [n, topK]
            float* slotOut = w + wF;             // [n*topK, expertDim]
            float* gIn = slotOut + slotOutF;     // [<=n, expertDim] gathered expert inputs
            float* gMid = gIn + gInF;            // [<=n, moeInter]
            float* gOut = gMid + gMidF;          // [<=n, expertDim]
            float* acc = gOut + gOutF;           // [n, expertDim]
            float* sel = acc + accF;             // [nExpert]

            new ReadOnlySpan<float>(normOut, (int)inF).CopyTo(new Span<float>(x, (int)inF));

            // Batched dense parts: router, latent_down, shared expert.
            Gemm(moe.GateInpWeight, moe.GateInpQuantType, x, logits, nExpert, hiddenSize, n, null);
            if (latent > 0)
                Gemm(moe.LatentDownWeight, moe.LatentDownQuantType, x, xl, latent, hiddenSize, n, null);
            Gemm(moe.UpShexpWeight, moe.UpShexpQuantType, x, shMid, sharedInter, hiddenSize, n, null);
            ReluSquared.Execute(new Span<float>(shMid, n * sharedInter), new Span<float>(shMid, n * sharedInter));
            Gemm(moe.DownShexpWeight, moe.DownShexpQuantType, shMid, shOut, hiddenSize, sharedInter, n, null);
            float* expertIn = latent > 0 ? xl : x;

            // Per-token routing, then a counting sort of the (token, slot) pairs by expert.
            var selSpan = new Span<float>(sel, nExpert);
            Array.Clear(counts, 0, nExpert + 1);
            for (int t = 0; t < n; t++)
            {
                RouteToken(moe, new Span<float>(logits + (long)t * nExpert, nExpert), selSpan,
                    topIdx.AsSpan(t * topK, topK), new Span<float>(w + (long)t * topK, topK));
                for (int k2 = 0; k2 < topK; k2++) counts[topIdx[t * topK + k2] + 1]++;
            }
            for (int e = 0; e < nExpert; e++) counts[e + 1] += counts[e];   // counts[e] = first slot of expert e
            Array.Copy(counts, fill, nExpert);
            for (int s = 0; s < slots; s++) order[fill[topIdx[s]]++] = s;   // slot ids grouped by expert, ascending

            // One up -> relu2 -> down GEMM pair per used expert, over all of its tokens.
            for (int e = 0; e < nExpert; e++)
            {
                int begin = counts[e], cnt = counts[e + 1] - begin;
                if (cnt == 0) continue;
                for (int j = 0; j < cnt; j++)
                {
                    int t = order[begin + j] / topK;
                    new ReadOnlySpan<float>(expertIn + (long)t * expertDim, expertDim)
                        .CopyTo(new Span<float>(gIn + (long)j * expertDim, expertDim));
                }
                NemotronHMoeExpertCompute.Compute(moe, e, expertDim, gIn, cnt, gMid, gOut, _threadPool);
                for (int j = 0; j < cnt; j++)
                {
                    int s = order[begin + j];
                    new ReadOnlySpan<float>(gOut + (long)j * expertDim, expertDim)
                        .CopyTo(new Span<float>(slotOut + (long)s * expertDim, expertDim));
                }
            }

            // Weighted sum per token in slot order (same order as the per-token body).
            for (int t = 0; t < n; t++)
            {
                float* a = acc + (long)t * expertDim;
                new Span<float>(a, expertDim).Clear();
                for (int k2 = 0; k2 < topK; k2++)
                {
                    float wk = w[t * topK + k2];
                    float* o = slotOut + (long)(t * topK + k2) * expertDim;
                    for (int i = 0; i < expertDim; i++) a[i] += wk * o[i];
                }
            }

            // Back to the hidden size (latent_up) straight into the output, then add the shared expert.
            if (latent > 0)
                Gemm(moe.LatentUpWeight, moe.LatentUpQuantType, acc, normOut, hiddenSize, latent, n, null);
            else
                new ReadOnlySpan<float>(acc, (int)inF).CopyTo(new Span<float>(normOut, (int)inF));
            System.Numerics.Tensors.TensorPrimitives.Add(
                new ReadOnlySpan<float>(normOut, (int)inF), new ReadOnlySpan<float>(shOut, (int)inF),
                new Span<float>(normOut, (int)inF));
        }
        finally
        {
            System.Runtime.InteropServices.NativeMemory.AlignedFree(block);
            System.Buffers.ArrayPool<int>.Shared.Return(topIdx);
            System.Buffers.ArrayPool<int>.Shared.Return(counts);
            System.Buffers.ArrayPool<int>.Shared.Return(fill);
            System.Buffers.ArrayPool<int>.Shared.Return(order);
        }
    }


    private void ForwardAttentionBody(
        NemotronHAttentionWeights attn, int layer, int seqLen, ReadOnlySpan<int> positions,
        float* normOut, float* q, float* k, float* v, float* attnOut,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int kvStride = numKvHeads * headDim;

        Gemm(attn.QWeight, attn.QQuantType, normOut, q, attn.QOutputDim, attn.QInputDim, seqLen, preQuantizedInput: null);
        Gemm(attn.KWeight, attn.KQuantType, normOut, k, attn.KOutputDim, attn.KInputDim, seqLen, preQuantizedInput: null);
        Gemm(attn.VWeight, attn.VQuantType, normOut, v, attn.VOutputDim, attn.VInputDim, seqLen, preQuantizedInput: null);

        // NO position encoding (issue #372). Both references apply nothing here:
        // llama.cpp src/models/nemotron-h.cpp goes straight from build_qkv to
        // build_attn (no rope token in the file), and HF transformers'
        // NemotronHAttention.forward never calls apply_rotary_pos_emb. Position
        // information comes entirely from the Mamba2 layers. The GGUF
        // rope.dimension_count key is a converter artifact and is ignored.

        if (kvCache is not null)
        {
            int kvSlot = _kvSlotForLayer[layer];
            if (kvSlot < 0)
                throw new InvalidOperationException(
                    $"Layer {layer} has no KV-cache slot (not an attention layer).");

            var kRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)k);
            var vRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)v);
            kvCache.Update(kRef, vRef, positions, kvSlot);

            int seqKv = kvCache.CurrentLength;
            var cachedK = kvCache.GetKeysRef(kvSlot);
            var cachedV = kvCache.GetValuesRef(kvSlot);

            Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], pool: null,
                slidingWindowSize: null);
        }
        else
        {
            Attention.Execute(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeads, headDim, 0, pool: null,
                slidingWindowSize: null);
        }

        Gemm(attn.OWeight, attn.OQuantType, attnOut, normOut, attn.OOutputDim, attn.OInputDim, seqLen, preQuantizedInput: null);
    }

    /// <summary>
    /// Mamba2 SSM sub-layer forward (DESIGN.md §4). Reads pre-normed activations from
    /// <paramref name="normOut"/> and writes the ssm_out projection back to the same buffer.
    /// Advances the per-layer conv/SSM state in place.
    /// </summary>
    [SkipLocalsInit]
    private void ForwardSsmBody(NemotronHSsmWeights ssmW, int absoluteLayerIndex, int seqLen,
                                int hiddenSize, float* normOut, float eps)
    {
        bool traceSsm = NemotronHDiagnostics.IsSsmTraced(absoluteLayerIndex);
        int ssmOrdinal = _ssmLayerOrdinal[absoluteLayerIndex];

        int dInner = _ssm.DInner;
        int dConv = _ssm.DConv;
        int nHead = _ssm.NHead;
        int headDim = _ssm.HeadDim;
        int dState = _ssm.DState;
        int nGroup = _ssm.NGroup;
        int convDim = _ssm.ConvDim;
        int groupDim = dInner / nGroup;
        int inProjDim = _ssm.InputProjectionDim;

        float* zxbcdt = (float*)_state.Zxbcdt;
        float* convInput = (float*)_state.ConvInput;
        float* xbc = (float*)_state.XBC;
        float* dtBuf = (float*)_state.DtBuffer;
        float* xBuf = (float*)_state.SsmX;
        float* bBuf = (float*)_state.SsmB;
        float* cBuf = (float*)_state.SsmC;
        float* yBuf = (float*)_state.SsmY;
        int bcDim = nGroup * dState;

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.normOut[last]",
                normOut + (seqLen - 1) * hiddenSize, hiddenSize);

        // 1. ssm_in GEMM
        Gemm(ssmW.InWeight, ssmW.InQuantType, normOut, zxbcdt,
             inProjDim, hiddenSize, seqLen, preQuantizedInput: null);
        ScaleInPlace(zxbcdt, (long)seqLen * inProjDim, ssmW.InScale);

        if (traceSsm)
        {
            int off = (seqLen - 1) * inProjDim;
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.z[last]",
                zxbcdt + off, dInner);
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.xBC[last]",
                zxbcdt + off + dInner, convDim);
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.dt[last]",
                zxbcdt + off + 2 * dInner + 2 * nGroup * dState, nHead);
        }

        // 2. conv_input = concat(conv_state, xBC rows from zxbcdt)
        var convState = (_activeSsm ?? _ssmCache).GetConvState(ssmOrdinal);
        convState.CopyTo(new Span<float>(convInput, (dConv - 1) * convDim));
        for (int t = 0; t < seqLen; t++)
        {
            var src = new ReadOnlySpan<float>(zxbcdt + t * inProjDim + dInner, convDim);
            var dst = new Span<float>(convInput + (dConv - 1 + t) * convDim, convDim);
            src.CopyTo(dst);
        }

        // 3. Conv1d + SiLU
        int convInputElems = (dConv - 1 + seqLen) * convDim;
        int xbcElems = seqLen * convDim;
        Conv1dCausal.Execute(
            input: new ReadOnlySpan<float>(convInput, convInputElems),
            weight: ssmW.Conv1dWeight,
            bias: ssmW.Conv1dBias,
            output: new Span<float>(xbc, xbcElems),
            dConv: dConv,
            channels: convDim,
            seqLen: seqLen);

        SiLu.Execute(
            new ReadOnlySpan<float>(xbc, xbcElems),
            new Span<float>(xbc, xbcElems));

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.xbc_post_silu[last]",
                xbc + (seqLen - 1) * convDim, convDim);

        // 4. Save last (d_conv-1) rows of conv_input (pre-SiLU) back into conv_state.
        for (int r = 0; r < dConv - 1; r++)
        {
            var src = new ReadOnlySpan<float>(
                convInput + (seqLen + r) * convDim, convDim);
            var dst = convState.Slice(r * convDim, convDim);
            src.CopyTo(dst);
        }

        // 5. dt = zxbcdt[:, dtOffset..] + ssm_dt.bias
        int dtOffset = 2 * dInner + 2 * nGroup * dState;
        for (int t = 0; t < seqLen; t++)
        {
            var src = new ReadOnlySpan<float>(zxbcdt + t * inProjDim + dtOffset, nHead);
            var dst = new Span<float>(dtBuf + t * nHead, nHead);
            TensorPrimitives.Add(src, ssmW.DtBias, dst);
        }

        // 6. Selective scan — pack contiguous x/B/C into long-lived state buffers.
        // xbc row layout is [x (dInner) | B (bcDim) | C (bcDim)]; the scan kernel wants each
        // of those as its own contiguous [T, ...] buffer, so split them per token.
        int xElems = seqLen * dInner;
        int bcElems = seqLen * bcDim;

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> row = new(xbc + t * convDim, convDim);
            row.Slice(0, dInner).CopyTo(new Span<float>(xBuf + t * dInner, dInner));
            row.Slice(dInner, bcDim).CopyTo(new Span<float>(bBuf + t * bcDim, bcDim));
            row.Slice(dInner + bcDim, bcDim).CopyTo(new Span<float>(cBuf + t * bcDim, bcDim));
        }

        var ssmState = (_activeSsm ?? _ssmCache).GetSsmState(ssmOrdinal);

        Mamba2SelectiveScan.Execute(
            state: ssmState,
            x: new ReadOnlySpan<float>(xBuf, xElems),
            dt: new ReadOnlySpan<float>(dtBuf, seqLen * nHead),
            a: ssmW.A,
            b: new ReadOnlySpan<float>(bBuf, bcElems),
            c: new ReadOnlySpan<float>(cBuf, bcElems),
            y: new Span<float>(yBuf, xElems),
            nHead: nHead,
            headDim: headDim,
            dState: dState,
            nGroup: nGroup,
            seqLen: seqLen,
            pool: _threadPool); // #1042: heads split across the pool, bit-identical to serial

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_scan[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 7. y += x * D[h] (broadcast D per head across head_dim)
        for (int t = 0; t < seqLen; t++)
        {
            int tBase = t * dInner;
            for (int h = 0; h < nHead; h++)
            {
                float dh = ssmW.D[h];
                int rowBase = tBase + h * headDim;
                for (int iHead = 0; iHead < headDim; iHead++)
                {
                    yBuf[rowBase + iHead] += xBuf[rowBase + iHead] * dh;
                }
            }
        }

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_D[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 8. SwiGLU gating: y = SiLU(z) * y, z = zxbcdt[:, 0..dInner)
        for (int t = 0; t < seqLen; t++)
        {
            var z = new ReadOnlySpan<float>(zxbcdt + t * inProjDim, dInner);
            var yRow = new Span<float>(yBuf + t * dInner, dInner);
            FusedOps.SwiGLU(z, yRow, yRow);
        }

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_swiglu[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 9. Group RMSNorm
        for (int t = 0; t < seqLen; t++)
        {
            for (int g = 0; g < nGroup; g++)
            {
                int off = t * dInner + g * groupDim;
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(yBuf + off, groupDim),
                    ssmW.NormWeight.AsSpan(g * groupDim, groupDim),
                    eps,
                    new Span<float>(yBuf + off, groupDim));
            }
        }

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_groupnorm[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 10. ssm_out projection into normOut
        Gemm(ssmW.OutWeight, ssmW.OutQuantType, yBuf, normOut,
             hiddenSize, dInner, seqLen, preQuantizedInput: null);
        ScaleInPlace(normOut, (long)seqLen * hiddenSize, ssmW.OutScale);

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.out[last]",
                normOut + (seqLen - 1) * hiddenSize, hiddenSize);
    }

    private void EmbedTokens(ReadOnlySpan<int> tokenIds, float* hidden, int hiddenSize)
    {
        nint embPtr = _tokenEmbedWeight;
        var qt = _tokenEmbedQuantType;

        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {Config.VocabSize}).");

            float* dest = hidden + t * hiddenSize;
            var destSpan = new Span<float>(dest, hiddenSize);

            if (qt == QuantizationType.F32)
            {
                float* src = (float*)embPtr + (long)tokenId * hiddenSize;
                new ReadOnlySpan<float>(src, hiddenSize).CopyTo(destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                Half* src = (Half*)embPtr + (long)tokenId * hiddenSize;
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, hiddenSize), destSpan);
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                nint rowPtr = embPtr + (nint)((long)tokenId * rowBytes);
                Dequantize.ToFloat32(rowPtr, hiddenSize, qt, destSpan);
            }
        }
    }

    /// <summary>
    /// Multiplies a GEMM output by its weight's per-tensor scale (#1038), as llama.cpp's
    /// <c>build_lora_mm</c> does after <c>mul_mat</c>. A scale of exactly 1 (no <c>X.scale</c> tensor) is a no-op.
    /// </summary>
    private static void ScaleInPlace(float* data, long count, float scale)
    {
        if (scale == 1.0f) return;
        var span = new Span<float>(data, checked((int)count));
        TensorPrimitives.Multiply(span, scale, span);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c,
                      int m, int k, int n, byte* preQuantizedInput)
        => GemmDispatch(weights, qt, b, c, m, k, n, _threadPool, preQuantizedInput);

    /// <summary>
    /// The Nemotron-H GEMM dispatcher: <c>c[n, m] = b[n, k] · W[m, k]ᵀ</c> for every weight type the model reads. Shared by the
    /// model (<paramref name="pool"/> = its compute pool) and the per-expert CPU compute of the CUDA offload's cold leg (#1071,
    /// <paramref name="pool"/> = null: one expert per worker, single-threaded), so both run the same kernels. Pool-parallel
    /// kernels are row-partitioned, so a null pool gives bit-identical results.
    /// </summary>
    /// <param name="weights">Weight matrix (quantized rows of length <paramref name="k"/>).</param>
    /// <param name="qt">Weight type.</param>
    /// <param name="b">Input rows <c>[n, k]</c>.</param>
    /// <param name="c">Output rows <c>[n, m]</c>.</param>
    /// <param name="m">Output width (weight rows).</param>
    /// <param name="k">Input width.</param>
    /// <param name="n">Rows (tokens).</param>
    /// <param name="pool">Compute pool, or null for single-threaded.</param>
    /// <param name="preQuantizedInput">Input already quantized for <paramref name="qt"/>'s dot, or null.</param>
    internal static void GemmDispatch(nint weights, QuantizationType qt, float* b, float* c,
                      int m, int k, int n, ComputeThreadPool? pool, byte* preQuantizedInput)
    {
        // #386: pool-aware GEMM dispatch — mirrors Qwen3HybridDenseTransformerModel.Gemm.
        // Row-partitioned across the pool, so results are bit-identical to serial.
        switch (qt)
        {
            case QuantizationType.Q8_0:
                MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.Q5_0:
                MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.Q2_K:
                MatMul.GemmQ2_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.Q3_K:
                MatMul.GemmQ3_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.IQ4_XS:
                MatMul.GemmIQ4_XS((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.IQ2_XXS:
            case QuantizationType.IQ2_XS:
            case QuantizationType.IQ2_S:
            case QuantizationType.IQ3_XXS:
            case QuantizationType.IQ3_S:
            case QuantizationType.IQ1_M:
            case QuantizationType.IQ1_S:
                MatMul.GemmIQCodebook(qt, (byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.Q4_K:
                MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.Q5_K:
                MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.Q6_K:
                MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, pool, preQuantizedInput);
                return;
            case QuantizationType.F32:
                MatMul.GemmF32((float*)weights, b, c, m, k, n, pool);
                return;
            case QuantizationType.NVFP4:
                // #1038: packed NVFP4 x Q8_0 (llama.cpp's CPU vec_dot); the caller applies the X.scale.
                MatMul.GemmNvfp4((byte*)weights, b, c, m, k, n, pool);
                return;
            case QuantizationType.F16:
                MatMul.GemmF16(weights, b, c, m, k, n, pool);
                return;
            case QuantizationType.Q4_0:
            case QuantizationType.Q4_1:
            case QuantizationType.Q5_1:
            case QuantizationType.IQ4_NL:
                // Packed x Q8_1 dot instead of dequantize-to-F32 (#489).
                MatMul.GemmLegacyQuantOrDequant((byte*)weights, qt, b, c, m, k, n, pool, preQuantizedInput);
                return;
            default:
                // Shared dequantize-and-dot fallback (#263): decodes each weight row once and
                // reuses it across all n columns instead of re-decoding the matrix per token.
                // It runs serially whatever the pool — bit-identical to the private copy it replaces.
                MatMul.GemmDequantRows((byte*)weights, qt, b, c, m, k, n, pool: null);
                return;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* QuantizeInput(float* input, byte* scratch, int dim, int seqLen, QuantizationType qt)
    {
        if (MatMul.UsesQ8KDot(qt))
        {
            int blockCount = dim / 256;
            int q8kRowBytes = blockCount * MatMul.Q8_K_BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_K(input + t * dim, scratch + t * q8kRowBytes, dim);
            return scratch;
        }

        // Q5_0 and the packed legacy quants (Q4_0/Q4_1/Q5_1/IQ4_NL, #489) all dot against Q8_1.
        if (qt == QuantizationType.Q5_0 || MatMul.UsesPackedLegacyDot(qt, seqLen))
        {
            int blockCount = dim / Q8_1GroupSize;
            int q8_1RowBytes = blockCount * MatMul.Q8_1BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_1(input + t * dim, scratch + t * q8_1RowBytes, dim);
            return scratch;
        }

        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = dim / Q8_0GroupSize;
            int q8RowBytes = blockCount * Q8_0BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_0(input + t * dim, scratch + t * q8RowBytes, dim);
            return scratch;
        }

        return null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsThreadPool)
            _threadPool?.Dispose();
        _state.Dispose();
        _ssmCache.Dispose();
        GC.SuppressFinalize(this);
    }

    private static float[] DequantizeF32(GgufDataBase dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase.Of(desc);
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }
}
