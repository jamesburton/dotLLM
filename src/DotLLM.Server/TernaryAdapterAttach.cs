using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Vulkan;

namespace DotLLM.Server;

/// <summary>
/// The single attach path for ternary residual adapters (<c>--ternary-adapter</c>) behind <c>run</c>, <c>chat</c> and <c>serve</c> (#1711).
/// A CPU model implementing <see cref="IResidualModifierHost"/> gets a <see cref="TernaryResidualHook"/>; a Vulkan model implementing
/// <see cref="IVulkanResidualModifierHost"/> gets a device-resident <see cref="VulkanTernaryResidualModifier"/>. Anything else (CUDA models,
/// architectures without a hook) is a hard, specific <see cref="NotSupportedException"/>: an adapter that is requested but not applied would
/// silently produce base-model output.
/// </summary>
public static class TernaryAdapterAttach
{
    /// <summary>Owns an attached modifier; disposing detaches it from the model and frees its buffers.</summary>
    public sealed class Handle : IDisposable
    {
        private Action? _detach;
        private IDisposable? _owned;

        internal Handle(Action detach, IDisposable owned, int circuits, string where)
        {
            _detach = detach; _owned = owned; CircuitCount = circuits; Where = where;
        }

        /// <summary>Number of ternary circuits attached.</summary>
        public int CircuitCount { get; }

        /// <summary>"cpu" or "vulkan".</summary>
        public string Where { get; }

        /// <inheritdoc/>
        public void Dispose()
        {
            _detach?.Invoke(); _detach = null;
            _owned?.Dispose(); _owned = null;
        }
    }

    /// <summary>
    /// Why <paramref name="model"/> cannot take a ternary adapter, or <see langword="null"/> when it can. The text names the model class and
    /// architecture so the user sees exactly which (architecture, backend) combination is unwired.
    /// </summary>
    public static string? Unsupported(IModel model, ModelConfig config)
    {
        if (model is IResidualModifierHost || model is IVulkanResidualModifierHost) return null;
        string backend = model.GetType().Namespace switch
        {
            { } n when n.Contains("Vulkan", StringComparison.Ordinal) => "Vulkan",
            { } n when n.Contains("Cuda", StringComparison.Ordinal) => "CUDA",
            _ => "this",
        };
        return $"--ternary-adapter: no residual-modifier hook on {model.GetType().Name} ({config.Architecture}, {backend} backend); " +
               "the adapter would be silently ignored, so the load is refused. See the support table in docs/TERNARY_ADAPTER.md.";
    }

    /// <summary>True for a device string whose backend has no wired model at all (CUDA): lets <c>--device auto</c> skip it up front.</summary>
    public static bool BackendHasNoHook(string canonicalDevice)
        => canonicalDevice.StartsWith("gpu", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Loads <paramref name="paths"/> (<c>.nsadapter</c> or adapter GGUF), validates them against <paramref name="config"/> and attaches them to
    /// <paramref name="model"/>. Does not dispose the model on failure.
    /// </summary>
    /// <exception cref="NotSupportedException">The model has no hook.</exception>
    /// <exception cref="InvalidOperationException">The adapters do not fit the model geometry.</exception>
    public static Handle Attach(IModel model, ModelConfig config, IReadOnlyList<string> paths)
    {
        ArgumentNullException.ThrowIfNull(paths);
        if (Unsupported(model, config) is { } why) throw new NotSupportedException(why);

        if (model is IResidualModifierHost host)
        {
            var hook = TernaryResidualHook.Load(paths);
            try
            {
                hook.ValidateFor(config.HiddenSize, config.NumLayers);
                host.ResidualModifier = hook;
            }
            catch { hook.Dispose(); throw; }
            return new Handle(() => { if (ReferenceEquals(host.ResidualModifier, hook)) host.ResidualModifier = null; }, hook, hook.CircuitCount, "cpu");
        }

        var vk = (IVulkanResidualModifierHost)model;
        var packages = paths.SelectMany(NsAdapterGguf.LoadAny).ToList();
        if (packages.Count == 0) throw new ArgumentException("no adapter paths given");
        var mod = VulkanTernaryResidualModifier.Create(vk.ResidualDevice, VulkanModelLoader.ResolveSpvDir(), packages);
        try
        {
            mod.ValidateFor(config.HiddenSize, config.NumLayers);
            vk.ResidualModifier = mod;
        }
        catch { mod.Dispose(); throw; }
        return new Handle(() => { if (ReferenceEquals(vk.ResidualModifier, mod)) vk.ResidualModifier = null; }, mod, packages.Count, "vulkan");
    }
}
