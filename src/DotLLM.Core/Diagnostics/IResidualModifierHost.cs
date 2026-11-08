namespace DotLLM.Core.Diagnostics;

/// <summary>
/// A CPU model whose forward pass honours an <see cref="IResidualModifier"/> after every layer (<see cref="HookPoint.PostLayer"/>).
/// Entry points (CLI <c>run</c>/<c>chat</c>/<c>serve</c>) attach ternary residual adapters through this seam; a model that does not
/// implement it must be refused with a specific error, never silently run without the adapter (#1711).
/// </summary>
public interface IResidualModifierHost
{
    /// <summary>The attached modifier, or <see langword="null"/> (the default: one branch per layer, nothing else).</summary>
    IResidualModifier? ResidualModifier { get; set; }
}
