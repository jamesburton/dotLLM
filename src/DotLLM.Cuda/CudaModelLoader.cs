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
                    "CUDA has no dedicated loader for Mamba3 yet (tracked separately from the "
                    + "Mamba3 guard on CudaModelLoader.LoadFromSafetensors — that one covers the "
                    + "safetensors path only, this is the GGUF path). Use the CPU or Vulkan "
                    + "backend for Mamba3 checkpoints.");

            case Architecture.NemotronH:
                throw new NotSupportedException(
                    "CUDA has no dedicated loader for NemotronH yet — its Mamba2 SSM layers "
                    + "(A, dt_bias, conv1d, etc.) are not tensor-compatible with the generic "
                    + "CudaTransformerModel this would otherwise silently fall through to. Use "
                    + "the CPU or Vulkan backend for NemotronH checkpoints.");

            // gpt-oss's MoE experts carry a per-expert bias (GateExpsBias/UpExpsBias/DownExpsBias)
            // and use an OAI-clamped-SwiGLU activation (UseSwiGluOai), neither of which
            // CudaMoeWeightsLoader/CudaMoeFfn reference (confirmed: zero call sites outside
            // generated XML docs). Falling through to the generic MoE path would silently drop
            // the bias and run the wrong activation on every layer (gpt-oss is all-MoE) rather
            // than crash or warn — worse than a clean failure. Fail loudly until CudaMoeFfn
            // actually implements both.
            case Architecture.GptOss:
                throw new NotSupportedException(
                    "CUDA does not yet implement gpt-oss's per-expert MoE bias or OAI-clamped-"
                    + "SwiGLU activation (CudaMoeFfn has no support for UseQuantExperts/"
                    + "*ExpsBias/UseSwiGluOai) — falling through to the generic MoE path would "
                    + "silently produce wrong output rather than fail. Use the CPU or Vulkan "
                    + "backend for gpt-oss checkpoints.");

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
    /// is not supported on CUDA and throws <see cref="NotSupportedException"/>.
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
                    "CUDA loader does not yet support Mamba3. Use the CPU safetensors loader "
                    + "(ModelLoader.LoadFromSafetensors) or the GGUF path.");

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
}
