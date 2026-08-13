using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-architecture dispatch point for creating a Vulkan <see cref="IModel"/> from an
/// already-opened GGUF file — the Vulkan mirror of <c>ModelLoader.CreateCpuModelFromGguf</c>
/// and <c>CudaModelLoader.CreateFromGguf</c>.
/// </summary>
/// <remarks>
/// Exists so every Vulkan entry point (<c>bench</c>, <c>perplexity</c>, …) resolves hybrid
/// architectures identically. The plain <see cref="VulkanTransformerModel"/> loader assumes
/// dense-attention tensor naming, so a Gated-DeltaNet layer — which has no
/// <c>attn_output.weight</c> — fails there with "blk.0.attn_output.weight not present"
/// (issue #259). Duplicating the switch per command is how that regression reappears.
/// </remarks>
public static class VulkanModelLoader
{
    /// <summary>
    /// Creates the architecture-appropriate Vulkan model for <paramref name="gguf"/>.
    /// </summary>
    /// <param name="device">An initialized Vulkan device. Not owned; the caller disposes it.</param>
    /// <param name="gguf">An opened GGUF file. Must remain alive for the lifetime of the model.</param>
    /// <param name="config">Model configuration extracted from <paramref name="gguf"/>.</param>
    /// <param name="spvDir">Directory containing compiled SPIR-V blobs.</param>
    /// <param name="nCpuMoeLayers">
    /// MoE expert-bank layers to keep on the CPU (Qwen3MoeHybrid only). <c>-1</c> auto-selects.
    /// Ignored by architectures without a routed expert bank.
    /// </param>
    /// <returns>
    /// The loaded model together with a factory for the KV-cache it expects. The factory is
    /// returned rather than left to the caller because each architecture needs its own concrete
    /// cache type and there is no common <c>CreateKvCache</c> interface.
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// The architecture has no GGUF representation at all (Mamba-3).
    /// </exception>
    public static (IModel Model, Func<int, IKvCache> KvCacheFactory) CreateFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir,
        int nCpuMoeLayers = -1)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(spvDir);

        switch (config.Architecture)
        {
            case Architecture.Qwen3MoeHybrid:
            {
                var moe = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                    device, gguf, config, spvDir, nCpuMoeLayers);
                return (moe, size => moe.CreateKvCache(size));
            }

            case Architecture.NemotronH:
            {
                var nemotron = VulkanNemotronHTransformerModel.BuildFromGguf(
                    device, gguf, config, spvDir);
                return (nemotron, size => nemotron.CreateKvCache(size));
            }

            case Architecture.Qwen3HybridDense:
            {
                var dense = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(
                    device, gguf, config, spvDir);
                return (dense, size => dense.CreateKvCache(size));
            }

            // Explicit rejections. Without these these architectures fall into `default`,
            // where VulkanTransformerModel fails on dense-attention tensor naming — the
            // caller then sees "blk.0.attn_output.weight not present" (or a bare
            // "Hybrid SSM / Mamba architectures are not supported") instead of the actual
            // reason.
            case Architecture.NemotronHMoe:
                throw new NotSupportedException(
                    "nemotron_h_moe (Nemotron 3.5 Lightning) is recognized but not yet runnable on " +
                    "Vulkan: the DeepSeek-V3-style MoE forward is not implemented, and its expert " +
                    "tensors ship in quantizations (Q5_0/IQ4_NL/Q4_0) the expert-indexed MoE kernel " +
                    "family does not cover yet. Tracked in issue #375.");

            case Architecture.Mamba3:
                throw new NotSupportedException(
                    "Mamba-3 has no GGUF representation: no upstream 'mamba3' value for " +
                    "general.architecture and no GGUF tensor-naming convention, so GgufModelConfigExtractor " +
                    "cannot produce Architecture.Mamba3 in the first place. Mamba-3 is safetensors-first on " +
                    "every backend — load it via VulkanMamba3TransformerModel.LoadFromSafetensors.");

            default:
            {
                var model = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);
                return (model, size => model.CreateKvCache(size));
            }
        }
    }
}
