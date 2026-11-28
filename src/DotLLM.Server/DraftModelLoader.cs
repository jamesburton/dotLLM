using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Server;

/// <summary>A loaded draft model and where it went.</summary>
/// <param name="Model">The draft model.</param>
/// <param name="KvCacheFactory">The KV-cache factory a GPU host expects; null for the CPU model (the engine's own cache).</param>
/// <param name="Device">Canonical device the draft is on (<c>cpu</c>, <c>gpu:N</c>, <c>vulkan</c> or <c>vulkan:N</c>).</param>
public sealed record DraftLoadResult(IModel Model, Func<ModelConfig, int, IKvCache>? KvCacheFactory, string Device);

/// <summary>
/// The one place <c>run</c>, <c>chat</c> and <c>serve</c> load a speculative draft model (#2064). The draft goes on the target's
/// device; a target that cannot verify drafts loads none.
/// </summary>
public static class DraftModelLoader
{
    /// <summary>
    /// Loads the draft for an already-loaded <paramref name="target"/>.
    /// </summary>
    /// <param name="target">The target model.</param>
    /// <param name="targetDevice">The target's resolved device (<c>cpu</c>, <c>gpu:N</c>, <c>vulkan</c> or <c>vulkan:N</c>).</param>
    /// <param name="draftGguf">The opened draft file; must outlive the returned model.</param>
    /// <param name="draftConfig">The draft's configuration.</param>
    /// <param name="speculativeCandidates">Drafts per round (K).</param>
    /// <param name="threading">CPU threading for a CPU draft.</param>
    /// <param name="warn">Receives why the draft was not loaded, or why it is not on the target's device.</param>
    /// <returns>The draft, or null when the target cannot verify drafts.</returns>
    public static DraftLoadResult? Load(
        IModel target, string targetDevice, GgufFile draftGguf, ModelConfig draftConfig, int speculativeCandidates,
        ThreadingConfig threading, Action<string> warn)
        => LoadWith(target, targetDevice, draftConfig, speculativeCandidates, warn, device =>
        {
            var spec = DeviceSpec.Parse(device);
            switch (spec.Kind)
            {
                case DeviceKind.Vulkan:
                {
                    // The same per-index VulkanDevice the target is on: never a second logical device for one GPU.
                    var (model, kv) = DotLLM.Vulkan.VulkanModelLoader.CreateOnDeviceFromGguf(draftGguf, draftConfig, spec.VulkanIndex);
                    return (model, kv);
                }
                case DeviceKind.Cuda:
                {
                    var (model, kv) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(draftGguf, draftConfig, spec.Ordinal);
                    return (model, kv);
                }
                default:
                    return (TransformerModel.LoadFromGguf(draftGguf, draftConfig, threading), null);
            }
        });

    /// <summary>Same, with the per-device loader injected (testable without hardware).</summary>
    internal static DraftLoadResult? LoadWith(
        IModel target, string targetDevice, ModelConfig draftConfig, int speculativeCandidates, Action<string> warn,
        Func<string, (IModel Model, Func<int, IKvCache>? Kv)> loadExact)
    {
        // Verification reads one logit row per drafted token (#2030). Loading a draft the target will never use only costs memory.
        target.TrySetAllRowLogitsLimit(speculativeCandidates + 1);
        if (target.MaxAllRowLogitsLength <= 1)
        {
            warn($"draft model not loaded: {target.GetType().Name} returns logits for the last token only " +
                 "(MaxAllRowLogitsLength = 1), so drafts cannot be verified. Decoding without speculation.");
            return null;
        }

        var spec = DeviceSpec.Parse(targetDevice);
        if (spec.IsGpu)
        {
            if (!UsesDenseHost(draftConfig.Architecture))
            {
                warn($"the draft model runs on the CPU: only dense-host drafts are placed on {targetDevice}, and this one is " +
                     $"{draftConfig.Architecture}.");
            }
            else
            {
                try
                {
                    var (model, kv) = loadExact(spec.Canonical);
                    return new DraftLoadResult(model, kv is null ? null : (_, size) => kv(size), spec.Canonical);
                }
                catch (Exception ex)
                {
                    warn($"the draft model runs on the CPU: loading it on {targetDevice} failed ({ex.Message}). " +
                         "Expect speculation to be much slower.");
                }
            }
        }

        var (cpu, _) = loadExact("cpu");
        return new DraftLoadResult(cpu, null, "cpu");
    }

    /// <summary>
    /// Architectures the GPU loaders send to their shared dense host. The others have recurrent state the draft phase cannot roll
    /// back, or a KV handle the engine does not drive for a second model.
    /// </summary>
    private static bool UsesDenseHost(Architecture architecture)
        => architecture is not (Architecture.Qwen3MoeHybrid or Architecture.Qwen3HybridDense or Architecture.NemotronH
            or Architecture.NemotronHMoe or Architecture.Lfm2Moe or Architecture.Lfm3 or Architecture.Qwen4Exp
            or Architecture.Mamba3 or Architecture.Olala or Architecture.Inkling2 or Architecture.Ricursive);
}
