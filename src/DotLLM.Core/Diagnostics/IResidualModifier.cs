namespace DotLLM.Core.Diagnostics;

/// <summary>
/// In-place, allocation-free modifier of the residual stream at <see cref="HookPoint.PostLayer"/>.
/// </summary>
/// <remarks>
/// <see cref="IInferenceHook"/> is observe-or-replace: its <c>Replace</c> result carries a freshly
/// allocated <c>float[]</c> and the call goes through a virtual dispatch per (layer, token), so it cannot
/// serve a zero-allocation steering adapter. This interface is the smallest extension that can: the
/// engine holds a nullable reference (<c>if (modifier is not null)</c> - one predictable branch per layer,
/// nothing else when no modifier is attached) and passes the contiguous <c>[tokenCount x hiddenSize]</c>
/// residual block by span so the implementation mutates it in place.
/// Single-threaded per model instance: the engine calls it from the forward thread only.
/// </remarks>
public interface IResidualModifier
{
    /// <summary>Whether this modifier does any work for <paramref name="layerIndex"/> (checked once per layer).</summary>
    bool AppliesTo(int layerIndex);

    /// <summary>
    /// Modifies the residual stream after layer <paramref name="layerIndex"/> in place.
    /// </summary>
    /// <param name="residual">Row-major <c>[tokenCount x hiddenSize]</c> residual stream (F32), mutated in place.</param>
    /// <param name="layerIndex">Zero-based layer whose output this is.</param>
    /// <param name="tokenCount">Number of token rows in <paramref name="residual"/>.</param>
    void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount);
}
