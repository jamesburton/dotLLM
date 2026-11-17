using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// <c>--gpu-layers auto</c> on CUDA (issue #1851): measures free VRAM with the CUDA runtime already up, sizes each layer as the
/// dGPU upload will store it, asks <see cref="LayerSplitPlanner"/> for the largest prefix that fits with the KV for the served
/// context and the forward scratch, and loads every layer (<see cref="CudaTransformerModel"/>), the split
/// (<see cref="HybridTransformerModel"/>, CUDA prefix + CPU tail), or refuses when not one layer fits.
/// </summary>
public static class CudaAutoSplit
{
    /// <summary>Follow-up for splitting Granite (TwIL-8B), Gemma and the rest of the FP32-dense-forward family.</summary>
    public const string CompositeIssueUrl = "https://github.com/jamesburton/dotLLM/issues/1864";

    /// <summary>Context the split is planned for when <c>--ctx-size</c> is not given (as <c>PlanCpuMoeLayers</c>).</summary>
    public const int DefaultPlanContextTokens = 4096;

    /// <summary>What the loader measured and estimated for the planner.</summary>
    /// <param name="FreeBytes">Free device bytes with the runtime and fixed forward buffers allocated, or null without a reading.</param>
    /// <param name="ActivationBytesPerToken">Forward scratch per token.</param>
    /// <param name="PendingLazyBytes">Lazy dequant scratch not yet allocated.</param>
    /// <param name="LayerBytes">Device bytes per layer.</param>
    /// <param name="EmbeddingBytes">Device bytes of the token embedding.</param>
    /// <param name="HeadBytes">Device bytes of the final norm + LM head.</param>
    internal readonly record struct Measurement(long? FreeBytes, long ActivationBytesPerToken, long PendingLazyBytes,
        long[] LayerBytes, long EmbeddingBytes, long HeadBytes);

    /// <summary>
    /// Why the split host cannot run this model, or <see langword="null"/> (#1851): the architecture and config checks of
    /// <see cref="GpuOffloadPlanner.AutoSplitBlocker"/>, the composite-host rejections, and attention sinks.
    /// </summary>
    /// <param name="gguf">The GGUF.</param>
    /// <param name="config">Model configuration.</param>
    /// <returns>The reason, or null.</returns>
    public static string? Blocker(GgufFile gguf, ModelConfig config)
    {
        bool sinks = gguf.TensorsByName.Keys.Any(k => k.EndsWith(".attn_sinks.weight", StringComparison.Ordinal));
        if (GpuOffloadPlanner.AutoSplitBlocker(config, sinks) is { } reason) return reason;
        try
        {
            CudaTransformerModel.RejectGemmaForCompositeHost(config, nameof(HybridTransformerModel), allowFoldedGraniteScalars: true);
        }
        catch (NotSupportedException)
        {
            return $"{config.Architecture} runs only on CUDA's FP32 dense forward, which the GPU/CPU split host does not implement yet " +
                   $"({CompositeIssueUrl})";
        }
        if (config.IsGemma4DensePle) return "the Gemma-4 dense variant cannot be split";
        return null;
    }

    /// <summary>
    /// Loads <paramref name="gguf"/> on CUDA device <paramref name="deviceId"/> with an automatically chosen layer count.
    /// </summary>
    /// <param name="gguf">Opened GGUF; must outlive the model.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="deviceId">CUDA ordinal.</param>
    /// <param name="threading">CPU threading for the tail.</param>
    /// <param name="moeOffload">Offload options; <see cref="MoeOffloadOptions.ContextTokens"/> (<c>--ctx-size</c>) is the planned context.</param>
    /// <param name="igpuProbe">Finds an iGPU for the tail, called only when the model does not fit the dGPU; null for none.</param>
    /// <param name="log">Receives the placement line and its reason.</param>
    /// <returns>The model, its KV factory, and the plan (null when no split was considered).</returns>
    /// <exception cref="DeviceMemoryAdmissionException">Not even one layer fits on the dGPU.</exception>
    public static (IModel Model, Func<int, IKvCache> KvCacheFactory, LayerSplitPlan? Plan) Load(
        GgufFile gguf, ModelConfig config, int deviceId, ThreadingConfig threading, MoeOffloadOptions? moeOffload,
        Func<IgpuTier?>? igpuProbe, Action<string>? log)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        log ??= Console.Error.WriteLine;

        if (Blocker(gguf, config) is { } blocker)
        {
            log($"--gpu-layers auto: no GPU/CPU split for this model ({blocker}); loading every layer on the GPU.");
            var (full, fullKv) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId, moeOffload: moeOffload);
            return (full, fullKv, null);
        }

        int ctx = PlanContext(moeOffload, config.MaxSequenceLength);
        var m = Measure(gguf, config, deviceId);
        if (m.FreeBytes is not long free)
        {
            log("--gpu-layers auto: no free-memory reading from the driver; loading every layer on the GPU.");
            var (full, fullKv) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId, moeOffload: moeOffload);
            return (full, fullKv, null);
        }

        var input = BuildInput(config, m, free, CudaAdmission.ReserveBytes, ctx, igpu: null);
        var plan = LayerSplitPlanner.Plan(input);
        if (!plan.AllOnDgpu && plan.DgpuLayers > 0 && igpuProbe?.Invoke() is { } igpu)
            plan = LayerSplitPlanner.Plan(input with { Igpu = igpu });
        log($"--gpu-layers auto: {plan.Describe()} for a {ctx}-token context ({plan.Reason}).");

        if (plan.AllOnDgpu)
        {
            var (full, fullKv) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId, moeOffload: moeOffload);
            return (full, fullKv, plan);
        }
        if (plan.DgpuLayers == 0)
            throw new DeviceMemoryAdmissionException(
                $"--gpu-layers auto: {plan.Reason}. Nothing was loaded on the GPU. Use --device cpu, a smaller quantization or " +
                "context (--ctx-size), or free GPU memory.");

        // The split was sized for ctx tokens of KV, so it serves no more than that (--ctx-size raises it, at fewer GPU layers).
        var splitConfig = config.MaxSequenceLength > ctx ? config with { MaxSequenceLength = ctx } : config;
        if (splitConfig.MaxSequenceLength < config.MaxSequenceLength)
            log($"--gpu-layers auto: context window {ctx} of {config.MaxSequenceLength} tokens (the split was sized for it; set --ctx-size to change).");
        Nvfp4Support.RejectAny(gguf.TensorsByName, "CUDA"); // #1038, as the explicit --gpu-layers N split does
        QuantSupport.RejectAny(gguf.TensorsByName, QuantizationType.PTQ1_0, "CUDA", 1280);
        var h = HybridTransformerModel.LoadFromGguf(gguf, splitConfig, plan.DgpuLayers, deviceId, threading);
        return (h, size => h.CreateKvCache(size), plan);
    }

    /// <summary>The planned context: <c>--ctx-size</c>, else <see cref="DefaultPlanContextTokens"/>, never above the model's window.</summary>
    /// <param name="moeOffload">Offload options carrying <c>--ctx-size</c>.</param>
    /// <param name="native">The model's window.</param>
    /// <returns>Tokens.</returns>
    internal static int PlanContext(MoeOffloadOptions? moeOffload, int native)
    {
        int? requested = MoeOffloadOptions.WithEnvironmentFallback(moeOffload).ContextTokens;
        return Math.Max(1, Math.Min(requested is > 0 and int r ? r : DefaultPlanContextTokens, native));
    }

    /// <summary>The planner input for a measurement (pure).</summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="m">The measurement.</param>
    /// <param name="freeBytes">Free device bytes.</param>
    /// <param name="reserveBytes">Admission reserve.</param>
    /// <param name="ctx">Planned context.</param>
    /// <param name="igpu">iGPU tier or null.</param>
    /// <returns>The input.</returns>
    internal static LayerSplitInput BuildInput(ModelConfig config, Measurement m, long freeBytes, long reserveBytes, int ctx, IgpuTier? igpu)
        => new(freeBytes, reserveBytes,
            DgpuFixedBytes: m.EmbeddingBytes + m.PendingLazyBytes,
            LayerWeightBytes: m.LayerBytes,
            KvBytesPerTokenPerLayer: CudaKvCache.BytesFor(KvGeometry.FromConfig(config), 1) / Math.Max(1, config.NumLayers),
            ContextTokens: ctx,
            ActivationBytesPerToken: m.ActivationBytesPerToken,
            // Unchunked prefill of a full-context prompt: the scratch the hybrid's forward state grows to.
            ActivationTokens: ctx,
            HeadBytes: m.HeadBytes,
            Igpu: igpu);

    /// <summary>
    /// Brings up the CUDA runtime and the hybrid's fixed forward buffers exactly as <see cref="HybridTransformerModel.LoadFromGguf"/>
    /// does, reads free memory, sizes the weights, and tears it all down.
    /// </summary>
    private static Measurement Measure(GgufFile gguf, ModelConfig config, int deviceId)
    {
        string ptxDir = CudaKernels.ResolveAndValidatePtxDirectory(null);
        using var context = CudaContext.Create(deviceId);
        using var stream = CudaStream.Create();
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        using var kernels = new CudaKernels(ptxDir);
        using var state = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            useFp32Residual: config.Architecture == Architecture.BitNet);
        long? free = CudaAdmission.QueryFreeBytes();
        var (layers, embedding, head) = EstimateWeightBytes(gguf, config, kernels);
        return new Measurement(free, state.ActivationBytesPerToken, state.PendingLazyScratchBytes, layers, embedding, head);
    }

    private static readonly string[] Projections =
        ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];

    /// <summary>
    /// Device bytes of each layer, the token embedding and the head, as <see cref="CudaWeights.LoadFromGguf"/> stores them: the raw
    /// quantized blocks, plus a persistent FP16 copy of each projection whose type has no loaded quantized GEMV, and the embedding
    /// dequantized to FP16 when no per-row lookup kernel exists for its type.
    /// </summary>
    internal static (long[] Layers, long Embedding, long Head) EstimateWeightBytes(GgufFile gguf, ModelConfig config, CudaKernels kernels)
        => EstimateWeightBytes(gguf.TensorsByName.Select(kv => (kv.Key, kv.Value.Shape[0], (long)kv.Value.Shape.ElementCount, kv.Value.QuantizationType)),
            config, qt => kernels.HasLoadedQuantizedGemv(qt) || qt == QuantizationType.I2_S,
            qt => kernels.HasEmbeddingLookup(qt, config.HiddenSize));

    /// <summary>Pure form of <see cref="EstimateWeightBytes(GgufFile, ModelConfig, CudaKernels)"/> (unit-testable).</summary>
    /// <param name="tensors">Name, inner dimension, element count and type of every tensor.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="runsQuantized">Whether a projection of this type runs in place (no FP16 copy).</param>
    /// <param name="hasEmbeddingLookup">Whether the embedding of this type is gathered in place (no FP16 table).</param>
    /// <returns>Per-layer, embedding and head bytes.</returns>
    internal static (long[] Layers, long Embedding, long Head) EstimateWeightBytes(
        IEnumerable<(string Name, int Inner, long Elements, QuantizationType Type)> tensors, ModelConfig config,
        Func<QuantizationType, bool> runsQuantized, Func<QuantizationType, bool> hasEmbeddingLookup)
    {
        var layers = new long[config.NumLayers];
        long embedding = 0, head = 0, embeddingRaw = 0;
        bool hasOutput = false;
        foreach (var (name, inner, elements, type) in tensors)
        {
            long raw = Dequantize.RowByteSize(inner, type) * (elements / Math.Max(1, inner));
            if (name.StartsWith("blk.", StringComparison.Ordinal))
            {
                int dot = name.IndexOf('.', 4);
                if (dot < 0 || !int.TryParse(name.AsSpan(4, dot - 4), out int l) || (uint)l >= (uint)layers.Length) continue;
                layers[l] += raw;
                string rest = name[(dot + 1)..];
                if (Array.Exists(Projections, p => rest == p + ".weight") && !runsQuantized(type))
                    layers[l] += elements * sizeof(ushort);
            }
            else if (name == "token_embd.weight")
            {
                embeddingRaw = raw;
                embedding = hasEmbeddingLookup(type) ? raw : elements * sizeof(ushort);
            }
            else if (name == "output.weight")
            {
                hasOutput = true;
                head += raw + (runsQuantized(type) ? 0 : elements * sizeof(ushort));
            }
            else if (name == "output_norm.weight")
            {
                head += raw;
            }
        }
        if (!hasOutput) head += embeddingRaw;   // tied LM head: the embedding table is uploaded again as the head
        return (layers, embedding, head);
    }
}
