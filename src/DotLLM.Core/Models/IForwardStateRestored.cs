namespace DotLLM.Core.Models;

/// <summary>
/// Marks an exception thrown by <see cref="IModel.Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, DotLLM.Core.Attention.IKvCache?)"/> after the model has already restored its KV cache and recurrent state to their values at forward entry, so the caller's bookkeeping needs no recovery.
/// </summary>
public interface IForwardStateRestored
{
}
