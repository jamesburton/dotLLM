namespace DotLLM.Core.Models;

/// <summary>
/// A model whose served context window can be lowered after load (issues #1068, #1101): <c>--ctx-size</c>, or a fit to free
/// device memory. Lowering <see cref="IModel.Config"/>'s <c>MaxSequenceLength</c> bounds every position, KV growth limit and
/// prompt-length check that reads it.
/// </summary>
public interface IContextLimitable
{
    /// <summary>The window the model was loaded with (the checkpoint's, or factor × original with YaRN), before any limit.</summary>
    int NativeMaxSequenceLength { get; }

    /// <summary>Lowers the served context window.</summary>
    /// <param name="maxSequenceLength">The new window, 1..<see cref="NativeMaxSequenceLength"/>.</param>
    void LimitContextLength(int maxSequenceLength);
}
