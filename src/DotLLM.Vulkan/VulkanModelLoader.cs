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
    private static readonly Lazy<VulkanDevice> SharedDeviceLazy = new(VulkanDevice.Create);

    /// <summary>
    /// Process-wide Vulkan device for long-lived hosts (<c>serve</c>, <c>run</c>). Created on first use and
    /// intentionally never disposed: the model does not own its device, and one device per process is
    /// what the serving paths want across model reloads. Short-lived commands that want deterministic
    /// teardown (<c>bench</c>) create and dispose their own.
    /// </summary>
    public static VulkanDevice SharedDevice => SharedDeviceLazy.Value;

    /// <summary>
    /// Resolves the SPIR-V blob directory: <c>spv/</c> beside the running assembly (the MSBuild
    /// content-copy layout), falling back to the in-repo <c>native/vulkan/spv</c> for <c>dotnet run</c>
    /// from the source tree.
    /// </summary>
    public static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found (looked for spv/ beside the binary and native/vulkan/spv). " +
            "Build them with native/vulkan/build.ps1 (requires the Vulkan SDK).");
    }

    /// <summary>True when a <c>--device</c> string selects the Vulkan backend (<c>vulkan</c>).</summary>
    public static bool IsVulkanDeviceString(string? device) =>
        device is not null && device.StartsWith("vulkan", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// <see cref="CreateFromGguf"/> on the <see cref="SharedDevice"/> with the standard SPIR-V directory —
    /// the one call the serving entry points make.
    /// </summary>
    public static (IModel Model, Func<int, IKvCache> KvCacheFactory) CreateSharedFromGguf(
        GgufFile gguf, ModelConfig config, int nCpuMoeLayers = -1)
        => CreateFromGguf(SharedDevice, gguf, config, ResolveSpvDir(), nCpuMoeLayers);

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
    /// The architecture has no GGUF representation at all (Mamba-3), is recognized but not
    /// runnable on Vulkan yet (nemotron_h_moe), or is only wired on the single-device model
    /// (gpt-oss on the pipeline/hybrid models; #737).
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

            case Architecture.Qwen4Exp:
            {
                // Qwen3.8-Flash-Next (#818 V1): dense-attention QSA fallback, host n-gram branch. The model owns its sequence
                // state until the engine integration of #817, so there is no engine KV cache to hand out.
                var q4 = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
                return (q4, _ => throw new NotSupportedException(
                    "qwen4exp keeps its own sequence state and has no engine KV cache yet; engine/scheduler integration is issue #817."));
            }

            case Architecture.Mamba3:
                throw new NotSupportedException(
                    "Mamba-3 has no GGUF representation: no upstream 'mamba3' value for " +
                    "general.architecture and no GGUF tensor-naming convention, so GgufModelConfigExtractor " +
                    "cannot produce Architecture.Mamba3 in the first place. Mamba-3 is safetensors-first on " +
                    "every backend — load it via VulkanMamba3TransformerModel.LoadFromSafetensors.");

            // gpt-oss (#737) falls through to `default`: VulkanTransformerModel implements its attention
            // sinks, dense YaRN RoPE, OAI SwiGLU and MXFP4 experts. RejectUnsupportedArchitecture still
            // refuses it on the pipeline / hybrid / prebuilt-stage side doors.

            default:
            {
                var model = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);
                // MLA (DeepSeek-V2/V3, GLM-4.7-Flash) layers read K_nope/V/K_pe from a per-layer
                // MlaVulkanKvCache. A plain GQA VulkanKvCache makes the forward fall back to its
                // cacheless branch (attention sees only the current step's rows): prefill is exact
                // but every decode step after it is garbage (#742 real-weights validation).
                if (model.Config.MlaConfig is not null)
                    return (model, size => model.CreateMlaKvCache(size));
                return (model, size => model.CreateKvCache(size));
            }
        }
    }
}
