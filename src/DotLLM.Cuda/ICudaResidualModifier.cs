namespace DotLLM.Cuda;

/// <summary>Storage type of the residual stream a <see cref="ICudaResidualModifier"/> is handed.</summary>
public enum CudaResidualDType
{
    /// <summary>IEEE half (FP16) elements: the dense CUDA models' residual stream.</summary>
    F16,
    /// <summary>IEEE single (FP32) elements: the hybrid (GDN) models' and BitNet's residual stream.</summary>
    F32,
}

/// <summary>
/// CUDA counterpart of <c>DotLLM.Core.Diagnostics.IResidualModifier</c> / <c>IVulkanResidualModifier</c> (#1919): one call after each
/// layer's residual update (<c>HookPoint.PostLayer</c>), on the device residual, stream-ordered on the forward's stream. The residual never
/// crosses the host boundary unless the implementation chooses so (a test-only perturber syncs and edits one row from the host).
/// A model holds a nullable reference; with none attached the cost is one predictable branch per layer. A set modifier bypasses the decode
/// CUDA graphs (the same eager fallback the adapter and tap paths use), because a graph replay never reaches the per-layer call.
/// </summary>
public interface ICudaResidualModifier
{
    /// <summary>Whether this modifier does any work after <paramref name="layerIndex"/> (checked once per layer).</summary>
    bool AppliesTo(int layerIndex);

    /// <summary>
    /// Modifies the row-major <c>[tokenCount x hiddenSize]</c> residual after layer <paramref name="layerIndex"/>, in place.
    /// The caller has not synchronised: an implementation that reads or writes through the host must synchronise <paramref name="stream"/>
    /// first; one that launches work must do so on <paramref name="stream"/> so it is ordered between this layer and the next.
    /// </summary>
    /// <param name="residualDevicePtr">Device pointer to the residual stream after the layer.</param>
    /// <param name="dtype">Element type of <paramref name="residualDevicePtr"/>.</param>
    /// <param name="layerIndex">Zero-based layer whose output this is.</param>
    /// <param name="tokenCount">Number of token rows.</param>
    /// <param name="hiddenSize">Elements per row.</param>
    /// <param name="stream">The forward's CUDA stream handle.</param>
    void Apply(nint residualDevicePtr, CudaResidualDType dtype, int layerIndex, int tokenCount, int hiddenSize, nint stream);
}

/// <summary>
/// A CUDA model whose forward pass calls an <see cref="ICudaResidualModifier"/> after every layer. Not wired to
/// <c>TernaryAdapterAttach</c> (the server still refuses ternary adapters on CUDA models); this seam is what would enable it later.
/// </summary>
public interface ICudaResidualModifierHost
{
    /// <summary>The attached modifier, or <see langword="null"/> (the default: one branch per layer, nothing else).</summary>
    ICudaResidualModifier? ResidualModifier { get; set; }
}
