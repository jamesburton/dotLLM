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
    /// The architecture has no Vulkan GGUF loader yet (e.g. Nemotron-H).
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
                throw new NotSupportedException(
                    "Nemotron-H has no Vulkan GGUF loader yet — use --device cpu.");

            default:
            {
                var model = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);
                return (model, size => model.CreateKvCache(size));
            }
        }
    }
}
