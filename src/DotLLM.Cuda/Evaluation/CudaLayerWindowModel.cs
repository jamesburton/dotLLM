using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Evaluation;

/// <summary>
/// CUDA-backed <see cref="ILayerWindowModel"/>: each layer window is a freshly-built
/// <see cref="CudaPipelineStage"/> whose device weights are uploaded on
/// <see cref="CreateWindow"/> and released again on the executor's
/// <see cref="IDisposable.Dispose"/>.
/// </summary>
/// <remarks>
/// <para><b>Why per-window upload/free (issue #395).</b> Layer cycling exists precisely because the
/// whole model does not fit on the device. A window is therefore uploaded, replays the entire corpus
/// hidden-in → hidden-out, and is then torn down before the next window is uploaded — so peak VRAM is
/// one window's weights plus activations, never the model. Holding two windows of this factory alive
/// at once defeats the purpose and will very likely exhaust the device.</para>
/// <para><b>Why the output head runs on the host.</b> <see cref="CudaPipelineStage.FinishLogits"/>
/// applies the final norm + LM head to the <em>last row only</em> and returns a
/// <c>[1, vocab]</c> tensor — that is the right shape for token generation, but sliding-window
/// perplexity has to score <em>every</em> row of the corpus window. The head is a single GEMM whose
/// weight is negligible next to a layer window's worth of attention + FFN matrices, so delegating it
/// to a host model (<see cref="ApplyOutputHead"/>) costs almost nothing and buys all-rows scoring.
/// The consequence is worth stating plainly: every transformer <em>layer</em> still executes on the
/// GPU — only the final norm and the LM head are host-side. Correspondingly, no window this factory
/// creates is ever built with <c>isFinalStage: true</c>, not even the window containing the last
/// layer.</para>
/// <para><b>Ownership.</b> The host weights are borrowed on the
/// <see cref="BuildFromPrebuiltWeights"/> path and owned on the <see cref="LoadFromGguf"/> path; the
/// <see cref="GgufFile"/> and the head model are <em>never</em> owned (see each factory).</para>
/// <para><b>Scope.</b> <see cref="CudaPipelineStage"/>'s layer loop is standard dense / GQA causal
/// only — the same M-scope as <see cref="CudaPipelineTransformerModel"/> and
/// <see cref="HybridVulkanCudaTransformerModel"/>: no MLA, no MoE, no Gemma graph, no recurrent
/// (Mamba / SSM / Gated-DeltaNet) layers, no per-layer RoPE or sliding-window patterns. The
/// constructor rejects those configurations up front, because a silently-wrong perplexity number is
/// far worse than a load-time failure.</para>
/// </remarks>
public sealed class CudaLayerWindowModel : ILayerWindowModel
{
    private readonly ModelConfig _config;
    private readonly TransformerWeights _cpuWeights;
    private readonly bool _ownsCpuWeights;
    private readonly ILayerWindowModel _hostModel;
    private readonly int _deviceId;
    private readonly string _ptxDir;

    private CudaLayerWindowModel(
        ModelConfig config, TransformerWeights cpuWeights, bool ownsCpuWeights,
        ILayerWindowModel hostModel, int deviceId, string ptxDir)
    {
        _config = config;
        _cpuWeights = cpuWeights;
        _ownsCpuWeights = ownsCpuWeights;
        _hostModel = hostModel;
        _deviceId = deviceId;
        _ptxDir = ptxDir;
    }

    /// <summary>
    /// Loads the host-side weights from an opened GGUF and prepares CUDA device
    /// <paramref name="deviceId"/> for per-window uploads.
    /// </summary>
    /// <param name="gguf">
    /// Opened GGUF file. <b>Not owned</b> — the caller must keep it alive for as long as this model
    /// exists (the host weights are memory-mapped views into it) and dispose it afterwards.
    /// </param>
    /// <param name="config">Model configuration extracted from the GGUF metadata.</param>
    /// <param name="deviceId">CUDA device ordinal (0-based) that every layer window is uploaded to.</param>
    /// <param name="hostHeadModel">
    /// Model supplying <see cref="ApplyOutputHead"/> (normally a CPU model over the same checkpoint).
    /// <b>Not owned</b> — the caller disposes it.
    /// </param>
    /// <param name="ptxDir">PTX kernel directory. Null auto-detects <c>AppContext.BaseDirectory/ptx/</c>.</param>
    /// <returns>A model whose windows upload and free their own device weights.</returns>
    /// <exception cref="NotSupportedException">
    /// The configuration uses an architecture feature the CUDA window layer loop does not implement.
    /// </exception>
    public static CudaLayerWindowModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId,
        ILayerWindowModel hostHeadModel, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(hostHeadModel);
        ValidateSupported(config);

        // GPU-only load: skip the F32 host dequant of per-expert MoE tensors, exactly as
        // CudaTransformerModel/CudaPipelineTransformerModel do. (MoE is rejected above, but keeping
        // the call identical avoids a gratuitous divergence from the sibling loaders.)
        TransformerWeights? cpuWeights = null;
        try
        {
            cpuWeights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);
            cpuWeights.RepackWeights(); // idempotent; matches the CUDA upload contract
            return new CudaLayerWindowModel(
                config, cpuWeights, ownsCpuWeights: true, hostHeadModel, deviceId,
                ptxDir ?? Path.Combine(AppContext.BaseDirectory, "ptx"));
        }
        catch
        {
            cpuWeights?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Wires a CUDA layer-window model around already-built host
    /// <see cref="TransformerWeights"/> (synthetic test fixtures, or weights an outer loader already
    /// owns). Mirrors <c>CudaPipelineTransformerModel.BuildFromPrebuiltWeights</c>.
    /// </summary>
    /// <param name="cpuWeights">
    /// Host weights. <b>Borrowed, not owned</b> — the caller retains every host pointer and disposes
    /// them; <see cref="Dispose"/> releases only what this model allocated.
    /// </param>
    /// <param name="config">Model configuration.</param>
    /// <param name="deviceId">CUDA device ordinal (0-based).</param>
    /// <param name="hostHeadModel">Model supplying <see cref="ApplyOutputHead"/>. Not owned.</param>
    /// <param name="ptxDir">PTX kernel directory. Null auto-detects.</param>
    /// <returns>A model whose windows upload and free their own device weights.</returns>
    /// <exception cref="NotSupportedException">
    /// The configuration uses an architecture feature the CUDA window layer loop does not implement.
    /// </exception>
    internal static CudaLayerWindowModel BuildFromPrebuiltWeights(
        TransformerWeights cpuWeights, ModelConfig config, int deviceId,
        ILayerWindowModel hostHeadModel, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(cpuWeights);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(hostHeadModel);
        ValidateSupported(config);

        cpuWeights.RepackWeights(); // idempotent
        return new CudaLayerWindowModel(
            config, cpuWeights, ownsCpuWeights: false, hostHeadModel, deviceId,
            ptxDir ?? Path.Combine(AppContext.BaseDirectory, "ptx"));
    }

    /// <inheritdoc/>
    public int NumLayers => _config.NumLayers;

    /// <inheritdoc/>
    public int HiddenSize => _config.HiddenSize;

    /// <inheritdoc/>
    public int VocabSize => _config.VocabSize;

    /// <inheritdoc/>
    public int MaxContextLength => _config.MaxSequenceLength;

    /// <inheritdoc/>
    /// <remarks>
    /// Uploads this window's layer weights to the device. The window is built with
    /// <c>isFinalStage: false</c> unconditionally — the final norm + LM head live on the host head
    /// model — and with <c>skipTokenEmbed</c> for every window that does not start at layer 0, so a
    /// mid-trunk window does not pay <c>vocab × hidden</c> of VRAM for a table it can never gather
    /// from.
    /// </remarks>
    public ILayerWindowExecutor CreateWindow(int firstLayer, int layerCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstLayer);
        ArgumentOutOfRangeException.ThrowIfLessThan(layerCount, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(firstLayer + layerCount, _config.NumLayers);

        var stage = CudaPipelineStage.Build(
            _config, _cpuWeights, _deviceId, _ptxDir,
            firstLayer: firstLayer, layerCount: layerCount,
            isFinalStage: false, skipTokenEmbed: firstLayer != 0);
        return new CudaLayerWindow(stage, firstLayer, layerCount, _config.HiddenSize);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Delegated verbatim to the host model. See the type-level remarks for why the head is not on
    /// the device: the CUDA stage's head path produces the last row only, and perplexity needs every
    /// row.
    /// </remarks>
    public ITensor ApplyOutputHead(ReadOnlySpan<float> hidden, int seqLen)
        => _hostModel.ApplyOutputHead(hidden, seqLen);

    /// <inheritdoc/>
    /// <remarks>
    /// Releases the host weights only when this instance loaded them. The GGUF handle and the host
    /// head model are never owned and are left to the caller.
    /// </remarks>
    public void Dispose()
    {
        if (_ownsCpuWeights) _cpuWeights.Dispose();
    }

    /// <summary>
    /// Rejects every configuration whose forward pass <see cref="CudaPipelineStage"/>'s layer loop
    /// does not actually implement, naming the architecture and the offending feature.
    /// </summary>
    /// <remarks>
    /// The loop is: QKV projection → optional bias / QK-norm → RoPE (every layer, one configuration)
    /// → KV-cache update → causal attention with one uniform sliding-window size → O-proj →
    /// fused-add-RMSNorm → SwiGLU FFN. Anything outside that would run to completion and produce a
    /// plausible-looking but wrong perplexity, which is the one failure mode this whole feature must
    /// not have — hence a hard throw at load time rather than a warning.
    /// <para><b>Internal rather than private so it can be tested without a GPU or a GGUF.</b> The
    /// guard's whole value is that it fires; a rejection that silently stopped firing would restore
    /// exactly the silent-wrong-number failure it exists to prevent, and would look identical to a
    /// passing build. Reaching it through the public factories would need a real unsupported
    /// checkpoint on disk for every rejected feature, which is not something the test suite can
    /// carry.</para>
    /// </remarks>
    /// <param name="config">Configuration to check.</param>
    /// <exception cref="NotSupportedException">The configuration is outside the supported scope.</exception>
    internal static void ValidateSupported(ModelConfig config)
    {
        Reject(config.MlaConfig is not null, config, "multi-head latent attention (MLA)");
        Reject(config.Moe is not null, config, "mixture-of-experts FFN routing");
        Reject(config.HybridLayout is not null, config, "hybrid SSM+Transformer layer layouts");
        Reject(config.SsmConfig is not null, config, "Mamba-2 SSM layers");
        Reject(config.Mamba3Config is not null, config, "Mamba-3 layers");
        Reject(config.GdnConfig is not null, config, "Gated-DeltaNet layers");
        Reject(config.DiffusionConfig is not null, config, "masked-canvas diffusion decoding");
        Reject(config.IsGemmaArchitecture, config,
            "the Gemma residual layout (four RMSNorms per layer, (1+w) norm weights, GeGLU)");
        Reject(config.Gemma4DualFfn, config, "the Gemma-4 dual dense+MoE FFN graph");
        Reject(config.Gemma3n is not null, config, "Gemma-3n AltUp / Laurel blocks");
        Reject(config.PerLayerEmbedding is not null, config, "per-layer embeddings (PLE)");
        Reject(config.NoRopeLayers is { Count: > 0 }, config,
            "per-layer NoPE patterns (the window loop applies RoPE to every layer)");
        Reject(config.SlidingWindowPattern != 0, config,
            "interleaved sliding-window patterns (the window loop uses one uniform window size)");
        Reject(config.GlobalRoPEConfig is not null, config,
            "a separate RoPE configuration for full-attention layers");

        // The window loop hands LaunchRoPE nothing but theta / dim / type (CudaPipelineStage's
        // RunLayers), and no CUDA LaunchRoPE overload accepts YaRN's scaling factor, original
        // context length, attn factor or beta fast/slow. The CPU reference DOES apply them
        // (TransformerModel's PrecomputeFrequencyTableYarn), so a scaled-RoPE GGUF would clear every
        // other rejection here and then be scored with the UNSCALED frequency table — a plausible,
        // authoritative, wrong perplexity, which is the single failure mode this guard exists to
        // prevent. The gap is pre-existing and backend-wide (whole-device `--device cuda` is equally
        // unscaled); this rejection does not fix that, it only keeps the new guard's promise honest.
        // Note Llama-3's `llama3` scaling type maps to RoPEScalingType.None in
        // GgufModelConfigExtractor, so the Llama-3.x family is unaffected.
        Reject(config.RoPEConfig is { ScalingType: not RoPEScalingType.None }, config,
            "scaled RoPE (YaRN / linear / NTK / LongRoPE) — the window loop applies the unscaled "
            + "frequency table");
        Reject(config.PartialRotaryFactor is not null, config, "partial rotary embeddings");
        Reject(config.NumGlobalKvHeads is not null, config,
            "a distinct KV-head count for full-attention layers");
        Reject(config.GlobalHeadDim is not null, config,
            "a distinct head dimension for full-attention layers");
    }

    private static void Reject(bool unsupported, ModelConfig config, string feature)
    {
        if (!unsupported) return;
        throw new NotSupportedException(
            $"{nameof(CudaLayerWindowModel)} does not support {config.Architecture}: {feature} is not " +
            "implemented by the CUDA layer-window forward pass, which covers standard dense / GQA " +
            "causal attention with a SwiGLU FFN only. Run this model's perplexity on a backend that " +
            "implements the full architecture instead of getting a plausible but wrong number.");
    }

    /// <summary>
    /// One resident CUDA layer window. Owns the <see cref="CudaPipelineStage"/> and, with it, this
    /// window's device weights and activation scratch — all freed on <see cref="Dispose"/>.
    /// </summary>
    private sealed unsafe class CudaLayerWindow : ILayerWindowExecutor
    {
        private readonly CudaPipelineStage _stage;
        private readonly int _hiddenSize;

        public CudaLayerWindow(CudaPipelineStage stage, int firstLayer, int layerCount, int hiddenSize)
        {
            _stage = stage;
            FirstLayer = firstLayer;
            LayerCount = layerCount;
            _hiddenSize = hiddenSize;
        }

        public int FirstLayer { get; }

        public int LayerCount { get; }

        /// <inheritdoc/>
        /// <remarks>
        /// <c>kvCache: null</c> throughout: perplexity scores each corpus window as an independent
        /// cacheless prefill over the whole window, matching <c>BackendPerplexityModel</c>. There is
        /// no cross-call attention history to preserve.
        /// </remarks>
        public void Run(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> hiddenIn,
                        ReadOnlySpan<int> positions, Span<float> hiddenOut)
        {
            int seqLen = positions.Length;
            ArgumentOutOfRangeException.ThrowIfLessThan(seqLen, 1);
            long expected = (long)seqLen * _hiddenSize;
            if (hiddenOut.Length < expected)
                throw new ArgumentException(
                    $"hiddenOut holds {hiddenOut.Length} floats but the window produces {expected} " +
                    $"({seqLen} rows x {_hiddenSize}).", nameof(hiddenOut));

            if (FirstLayer == 0)
            {
                if (tokenIds.Length != seqLen)
                    throw new ArgumentException(
                        $"The first layer window gathers embeddings, so tokenIds ({tokenIds.Length}) " +
                        $"must match positions ({seqLen}).", nameof(tokenIds));
                _stage.EnqueueFromEmbedding(tokenIds, positions, seqLen, kvCache: null);
            }
            else
            {
                if (hiddenIn.Length < expected)
                    throw new ArgumentException(
                        $"A window starting at layer {FirstLayer} resumes from a boundary hidden state " +
                        $"of {expected} floats but got {hiddenIn.Length}.", nameof(hiddenIn));

                // EnqueueFromHidden's H2D is synchronous, so the pin only has to survive the call.
                fixed (float* hiddenPtr = hiddenIn)
                    _stage.EnqueueFromHidden((nint)hiddenPtr, positions, seqLen, kvCache: null);
            }

            using ITensor result = _stage.DownloadHiddenStateF32(seqLen);
            new ReadOnlySpan<float>((void*)result.DataPointer, (int)expected).CopyTo(hiddenOut);
        }

        /// <inheritdoc/>
        public void ResetState()
        {
            // Deliberately empty, and deliberately not silent about it: CudaPipelineStage's layer
            // loop is dense / GQA only (no Mamba/SSM or Gated-DeltaNet), so no layer in this window
            // carries recurrent state across a Run. The state that issue #261 found leaking between
            // corpus windows simply does not exist here — every Run is a self-contained cacheless
            // prefill. If a recurrent layer type is ever admitted to this window loop, its conv +
            // SSM state must be re-zeroed here.
        }

        public void Dispose() => _stage.Dispose();
    }
}
