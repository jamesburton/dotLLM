namespace DotLLM.Core.Models;

/// <summary>
/// Per-sequence recurrent state container for the Gated DeltaNet (GDN) layers of a
/// Qwen3MoeHybrid model. One instance covers all GDN layers for a single sequence.
/// </summary>
/// <remarks>
/// <para>
/// GDN layers maintain two pieces of recurrent state per layer, mutated in place as the
/// scan advances token-by-token: a rolling Q/K/V conv-history buffer of shape
/// <c>[(DConv − 1) × convDim]</c> and a full <c>[NVHead × DState × DState]</c>
/// associative-memory matrix. The exact buffer storage (host F32, GPU device-local,
/// quantised) is backend-specific; this marker interface lets the
/// <see cref="SequenceForwardRequest.GdnState"/> field carry the right concrete state
/// container to each backend without leaking backend types into the Core abstraction.
/// </para>
/// <para>
/// Backends (CPU <c>GdnStateCache</c>, Vulkan <c>VulkanGdnStateCache</c>, CUDA
/// <c>CudaGdnStateCache</c>) implement this interface and accept their concrete type
/// via pattern matching in the forward path — mirroring how
/// <see cref="DotLLM.Core.Attention.IKvCache"/> is consumed today.
/// </para>
/// <para>
/// GDN state is conceptually distinct from an attention KV-cache (it has no
/// position indexing — it is opaque per-layer recurrent state), which is why it is
/// modelled as a separate request-side container rather than piggy-backing on the
/// existing <see cref="DotLLM.Core.Attention.IKvCache"/> slot. A multi-seq batched
/// dispatch can therefore carry a per-seq GDN state alongside a per-seq KV-cache
/// without either container conflating responsibilities.
/// </para>
/// </remarks>
public interface IGdnState : IDisposable
{
    /// <summary>
    /// Number of GDN layers this state covers. Must equal the model's GDN-layer
    /// count for the state to be valid for that model.
    /// </summary>
    int NumGdnLayers { get; }

    /// <summary>
    /// Re-zeroes every layer's recurrent state. Call between independent sequences
    /// when reusing a single state container.
    /// </summary>
    void Reset();
}
