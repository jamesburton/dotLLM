namespace DotLLM.Vulkan;

/// <summary>
/// Device-side counterpart of <c>DotLLM.Core.Diagnostics.IResidualModifier</c> (#1618): records an in-place modification of
/// the device-resident residual stream after a transformer layer (<c>HookPoint.PostLayer</c>) into the forward's command
/// buffer. The residual never crosses the host boundary. The model holds a nullable reference, so with no modifier attached
/// the cost is one predictable branch per layer.
/// </summary>
public interface IVulkanResidualModifier
{
    /// <summary>Whether this modifier does any work after <paramref name="layerIndex"/> (global layer index; checked once per layer).</summary>
    bool AppliesTo(int layerIndex);

    /// <summary>
    /// Records the modification of the row-major <c>[tokenCount x hiddenSize]</c> F32 <paramref name="residual"/> after layer
    /// <paramref name="layerIndex"/>. The caller has already made the layer's writes to the residual visible (compute-to-compute
    /// barrier) and makes this recording's writes visible to the next consumer.
    /// </summary>
    void Record(nint cmdBuf, VulkanDevice.Buffer residual, int layerIndex, int tokenCount, int hiddenSize);

    /// <summary>Drops cached descriptor sets; the model calls this when its scratch buffers were re-allocated.</summary>
    void InvalidateDescriptorCache();
}
