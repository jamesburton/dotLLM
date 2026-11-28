using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures.DSpark;

/// <summary>
/// An Inkling 2 target (CPU or Vulkan) that can take the DSpark drafter (issues #2039, #2046). The resolver, the CLI and the server test for this marker instead of a
/// concrete model class, so every backend that implements it gets <c>--dspark</c> and the sibling auto-attach.
/// </summary>
public interface IInkling2DSparkHost : IDFlashDraftAttachable
{
    /// <summary>True when an auto-detected sibling drafter is not attached at load but remembered (<see cref="DeferAutoAttach"/>) and attached per request above a context threshold. The Vulkan model: the drafter never won below several thousand tokens (#2046).</summary>
    bool DefersAutoAttach => false;

    /// <summary>Remembers an auto-detected drafter GGUF for the lazy attach; nothing is opened or allocated.</summary>
    /// <param name="path">The drafter GGUF.</param>
    void DeferAutoAttach(string path) => throw new NotSupportedException("This host attaches an auto-detected drafter at load.");

    /// <summary>The path remembered by <see cref="DeferAutoAttach"/> (null when none or already attached) and the context threshold, for the startup log.</summary>
    (string Path, int MinContext)? DeferredDrafter => null;
}
