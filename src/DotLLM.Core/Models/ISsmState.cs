namespace DotLLM.Core.Models;

/// <summary>
/// Per-sequence recurrent state container for the Mamba-2 SSM layers of a Nemotron-H hybrid model.
/// One instance covers all SSM layers for a single sequence.
/// </summary>
/// <remarks>
/// <para>
/// Each Nemotron-H SSM layer maintains two pieces of recurrent state per layer, mutated in place as
/// the selective scan advances token-by-token: a rolling conv-history buffer
/// <c>[(d_conv − 1) × conv_dim]</c> and the SSM hidden state <c>[d_inner × d_state]</c>. The exact
/// buffer storage (host F32, GPU device-local) is backend-specific; this marker interface lets the
/// <see cref="SequenceForwardRequest.SsmState"/> field carry the right concrete container to each
/// backend without leaking backend types into Core — mirroring <see cref="IMambaState"/> (Mamba-3)
/// and <see cref="IGdnState"/> (Qwen3MoeHybrid GDN).
/// </para>
/// <para>
/// Backends (CPU <c>SsmStateCache</c>, Vulkan <c>VulkanSsmStateCache</c>) implement this interface and
/// accept their concrete type via pattern matching in the forward path. SSM recurrent state is
/// position-less per-layer state, distinct from an attention KV-cache, so a multi-seq batched dispatch
/// carries a per-seq SSM state alongside a per-seq KV-cache (the hybrid model has both attention and
/// SSM layers).
/// </para>
/// </remarks>
public interface ISsmState : IRecurrentSequenceState
{
    /// <summary>
    /// Number of SSM layers this state covers. Must equal the model's SSM-layer count for the state to
    /// be valid for that model.
    /// </summary>
    int NumSsmLayers { get; }

    // Reset() is inherited from IRecurrentSequenceState.
}
