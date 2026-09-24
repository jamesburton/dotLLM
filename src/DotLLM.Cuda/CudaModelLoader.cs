using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Cuda;

/// <summary>
/// Convenience helper for loading a model onto a GPU from a GGUF or HF
/// safetensors checkpoint.
/// </summary>
public static class CudaModelLoader
{
    /// <summary>
    /// Loads a transformer model from a GGUF file onto the specified GPU.
    /// </summary>
    /// <param name="path">Path to the GGUF model file.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    /// <returns>The loaded model, GGUF file handle, and model configuration.</returns>
    public static (CudaTransformerModel Model, GgufFile Gguf, ModelConfig Config) LoadFromGguf(
        string path, int deviceId = 0, string? ptxDir = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId, ptxDir);
        return (model, gguf, config);
    }

    /// <summary>
    /// Creates the architecture-appropriate CUDA <see cref="IModel"/> for an already-opened
    /// GGUF file. This is THE per-architecture CUDA dispatch point — the mirror of
    /// <see cref="ModelLoader.CreateCpuModelFromGguf"/>, and CLI commands plus the server call
    /// it so hybrid architectures (Qwen3MoeHybrid / Qwen3HybridDense Gated-DeltaNet layers)
    /// route to their dedicated loaders instead of the plain
    /// <see cref="CudaTransformerModel"/>, whose tensor naming they do not follow (e.g. a GDN
    /// layer has no <c>attn_output.weight</c>, so the plain loader fails with
    /// "blk.0.attn_output.weight not present" — issue #259).
    /// </summary>
    /// <param name="gguf">An opened GGUF file. Must remain alive for the lifetime of the model.</param>
    /// <param name="config">Model configuration extracted from <paramref name="gguf"/>.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.
    /// Honoured only by the plain <see cref="CudaTransformerModel"/> path; the hybrid loaders
    /// auto-detect.</param>
    /// <returns>
    /// The loaded CUDA model together with a factory for the KV-cache it expects. The factory is
    /// returned rather than left to the caller because each architecture needs its own concrete
    /// cache type and there is no common <c>CreateKvCache</c> interface — capturing it here keeps
    /// the pairing in one place instead of forcing every call site to re-switch on the type.
    /// </returns>
    public static (IModel Model, Func<int, IKvCache> KvCacheFactory) CreateFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        switch (config.Architecture)
        {
            case Architecture.Qwen3HybridDense:
            {
                var dense = Architectures.CudaQwen3HybridDenseTransformerModel
                    .LoadFromGguf(gguf, config, deviceId);
                return (dense, size => dense.CreateKvCache(size));
            }

            case Architecture.Qwen3MoeHybrid:
            {
                var moe = Architectures.CudaQwen3MoeHybridTransformerModel
                    .LoadFromGguf(gguf, config, deviceId);
                return (moe, size => moe.CreateKvCache(size));
            }

            // Recurrent-SSM hybrid architectures whose tensor naming the plain
            // CudaTransformerModel does not follow — same category of gap as
            // #259 (Qwen3HybridDense's GDN layers have no attn_output.weight).
            // Neither has a dedicated CUDA loader yet, so fail loudly and
            // specifically instead of either a confusing tensor-not-found
            // error deep in the generic loader, or worse, a silent
            // partial/wrong load on whichever tensors happen to coincide.
            case Architecture.Mamba3:
                throw new NotSupportedException(
                    "Mamba3 has no GGUF tensor-naming convention on ANY dotLLM backend (CPU, Vulkan, "
                    + "or CUDA) — see docs/SUPPORTED_MODELS.md's 'No upstream GGUF mapping' note. "
                    + "Load Mamba3 checkpoints via CudaModelLoader.LoadMamba3FromSafetensors (CUDA), "
                    + "ModelLoader.LoadFromSafetensors (CPU), or VulkanMamba3TransformerModel.LoadFromSafetensors "
                    + "(Vulkan) instead.");

            case Architecture.NemotronHMoe:
                throw new NotSupportedException(
                    "nemotron_h_moe (Nemotron 3.5 Lightning) is recognized but not yet runnable on "
                    + "CUDA: the DeepSeek-V3-style MoE forward is not implemented. Note the smallest "
                    + "published GGUF is ~17.6 GiB — larger than most single consumer GPUs. Tracked "
                    + "in issue #375.");

            case Architecture.NemotronH:
            {
                var nemotronH = Architectures.CudaNemotronHTransformerModel
                    .LoadFromGguf(gguf, config, deviceId, ptxDir);
                return (nemotronH, size => nemotronH.CreateKvCache(size));
            }

            // gpt-oss (llama.cpp LLM_ARCH_OPENAI_MOE) routes to the plain
            // CudaTransformerModel via `default` — it is Llama-shaped (GQA attention +
            // routed-MoE FFN, standard blk.N tensor naming), differing only in three
            // features that are now all implemented on CUDA rather than in model
            // structure, so it needs no dedicated model class. This mirrors the CPU
            // side, where ModelLoader.CreateCpuModelFromGguf's `_` arm sends GptOss to
            // the plain TransformerModel. The three deltas and where each landed:
            //   • Per-expert MoE bias + clamped `swiglu_oai` activation — issue #348
            //     (CudaMoeWeightsLoader.LoadLayerQuant uploads the biases,
            //     CudaMoeFfn.Forward runs LaunchSwiGLUOaiF32).
            //   • Alternating sliding-window/dense attention (window on even layers,
            //     dense on odd; SlidingWindowPattern=2) — issue #366
            //     (CudaSlidingWindowResolver, plumbed into every per-layer attention
            //     dispatch), plus that issue's dense-YaRN RoPE mscale fix, which
            //     gpt-oss's factor=32 scaling depends on.
            //   • Per-head attention sinks — issue #365 (attn_sinks.weight uploaded in
            //     CudaWeights as TransformerLayerWeights.AttnSinksDevice, consumed by
            //     the sink epilogue in the attention_f32 and attention_f16 kernels).
            // Composition of all three is gated by CudaGptOssParitySyntheticTests, which
            // runs a synthetic four-feature gpt-oss GGUF through both this dispatch and
            // the CPU oracle and compares last-token logits.

            default:
            {
                var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId, ptxDir);
                return (model, size => model.CreateKvCache(size));
            }
        }
    }

    /// <summary>
    /// Loads a transformer model from an HF safetensors checkpoint onto the
    /// specified GPU. Delegates to
    /// <see cref="ModelLoader.OpenSafetensorsAndConfig"/> for source+config
    /// resolution, then uploads through
    /// <c>CudaTransformerModel.LoadFromSafetensors</c>. Covers the same
    /// Transformer-family architectures as the CPU safetensors loader; Mamba3
    /// is not loadable via this method (its layer shape is not
    /// <see cref="CudaTransformerModel"/>-compatible) and throws
    /// <see cref="NotSupportedException"/> pointing at the dedicated
    /// <see cref="LoadMamba3FromSafetensors"/> entry point instead.
    /// </summary>
    /// <param name="path">A <c>*.safetensors</c> file, a
    /// <c>model.safetensors.index.json</c>, or a directory containing one.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    public static (CudaTransformerModel Model, ISafetensorsTensorSource Source, ModelConfig Config)
        LoadFromSafetensors(string path, int deviceId = 0, string? ptxDir = null)
    {
        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
        try
        {
            if (config.Architecture == Architecture.Mamba3)
                throw new NotSupportedException(
                    "Mamba3 is not loadable via CudaModelLoader.LoadFromSafetensors — its layer "
                    + "shape (no attention, no standard FFN) is not CudaTransformerModel-compatible. "
                    + "Use CudaModelLoader.LoadMamba3FromSafetensors(path, deviceId, ptxDir) instead.");

            var i2sCache = ModelLoader.TryCreateBitNetI2SCache(path, config);
            var model = CudaTransformerModel.LoadFromSafetensors(source, config, deviceId, ptxDir, i2sCache);
            return (model, source, config);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a Mamba-3 model from an HF safetensors checkpoint onto the specified GPU.
    /// This is the CUDA safetensors entry point for <see cref="Architecture.Mamba3"/> —
    /// a dedicated method rather than a branch inside <see cref="LoadFromSafetensors"/>
    /// because that method's return type is pinned to the concrete
    /// <see cref="CudaTransformerModel"/>, which <see cref="Architectures.CudaMamba3TransformerModel"/>
    /// is not (Mamba-3's layer shape — no attention, no standard FFN — is not
    /// <see cref="CudaTransformerModel"/>-compatible). Mirrors
    /// <see cref="ModelLoader.LoadFromSafetensors(string, ThreadingConfig?)"/> (CPU) and
    /// <c>VulkanMamba3TransformerModel.LoadFromSafetensors</c>'s dedicated-entry-point
    /// pattern.
    /// </summary>
    /// <param name="path">A <c>*.safetensors</c> file or a directory containing one, plus a sibling <c>config.json</c>.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    public static (Architectures.CudaMamba3TransformerModel Model, ISafetensorsTensorSource Source, ModelConfig Config)
        LoadMamba3FromSafetensors(string path, int deviceId = 0, string? ptxDir = null)
    {
        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
        try
        {
            if (config.Architecture != Architecture.Mamba3)
                throw new ArgumentException(
                    $"CudaModelLoader.LoadMamba3FromSafetensors requires a Mamba3 checkpoint, "
                    + $"got Architecture.{config.Architecture}. Use CudaModelLoader.LoadFromSafetensors instead.",
                    nameof(path));

            var model = Architectures.CudaMamba3TransformerModel.LoadFromSafetensors(source, config, deviceId, ptxDir);
            return (model, source, config);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }
}
