namespace DotLLM.Core.Models;

/// <summary>
/// Per-sequence recurrent state container for the SSM layers of a Mamba-3 model.
/// One instance covers all Mamba-3 layers for a single sequence.
/// </summary>
/// <remarks>
/// <para>
/// Mamba-3 layers maintain four pieces of recurrent state per layer, mutated in place
/// as the SSD scan advances token-by-token: the canonical SSM hidden state
/// <c>[n_head, head_dim, d_state]</c>; a cumulative RoPE angle buffer
/// <c>[n_head, num_rope_angles]</c>; and two streaming-chunk boundary buffers
/// (<c>k_state</c> and <c>v_state</c>) that carry the previous chunk's last-token K
/// and V across the chunk boundary so the SSD scan reproduces a one-shot forward
/// bit-for-bit. The exact buffer storage (host F32, GPU device-local, quantised) is
/// backend-specific; this marker interface lets the
/// <see cref="SequenceForwardRequest.MambaState"/> field carry the right concrete
/// state container to each backend without leaking backend types into the Core
/// abstraction.
/// </para>
/// <para>
/// Backends (CPU <c>Mamba3State</c>, Vulkan <c>VulkanMamba3State</c>) implement this
/// interface and accept their concrete type via pattern matching in the forward path
/// — mirroring how <see cref="IGdnState"/> already works for Qwen3MoeHybrid's GDN
/// layers and how <see cref="DotLLM.Core.Attention.IKvCache"/> works for attention.
/// </para>
/// <para>
/// Mamba-3 recurrent state is conceptually distinct from an attention KV-cache (no
/// position indexing — it is opaque per-layer recurrent state), which is why it is
/// modelled as a separate request-side container rather than piggy-backing on the
/// existing <see cref="DotLLM.Core.Attention.IKvCache"/> slot. A multi-seq batched
/// dispatch can therefore carry a per-seq Mamba state alongside a per-seq KV-cache
/// (unused by Mamba-3 itself, but the API shape stays consistent across hosts).
/// </para>
/// </remarks>
public interface IMambaState : IDisposable
{
    /// <summary>
    /// Number of Mamba-3 layers this state covers. Must equal the model's layer
    /// count for the state to be valid for that model.
    /// </summary>
    int NumLayers { get; }

    /// <summary>
    /// Re-zeroes every layer's recurrent state. Call between independent sequences
    /// when reusing a single state container.
    /// </summary>
    void Reset();
}
