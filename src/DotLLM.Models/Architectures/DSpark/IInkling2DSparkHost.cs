using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures.DSpark;

/// <summary>
/// An Inkling 2 target (CPU or Vulkan) that can take the DSpark drafter (issues #2039, #2046). The resolver, the CLI and the server test for this marker instead of a
/// concrete model class, so every backend that implements it gets <c>--dspark</c> and the sibling auto-attach.
/// </summary>
public interface IInkling2DSparkHost : IDFlashDraftAttachable
{
}
