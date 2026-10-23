namespace DotLLM.Core.Models;

/// <summary>
/// Marker base for a model's per-sequence recurrent state container — the opaque,
/// position-less per-layer state that a recurrent architecture (Mamba-3 SSM,
/// Qwen3MoeHybrid Gated DeltaNet) carries alongside the attention KV-cache.
/// </summary>
/// <remarks>
/// <para>One instance covers all of a single sequence's recurrent layers. The concrete
/// container (host F32, GPU device-local, quantised) is backend-specific; this base lets
/// a host-agnostic caller — notably <c>ContinuousBatchScheduler</c> — allocate one per
/// sequence via <see cref="IModel.CreateSequenceState"/>, thread it through that
/// sequence's prefill / decode / resume forwards, and dispose it on completion, without
/// the caller knowing whether it is an <see cref="IMambaState"/> or an
/// <see cref="IGdnState"/>. The forward path routes it back to the right concrete backend
/// via the typed <see cref="SequenceForwardRequest.MambaState"/> /
/// <see cref="SequenceForwardRequest.GdnState"/> slots (an <c>as</c>-cast at request-build
/// time selects the matching slot).</para>
/// <para>Both <see cref="IMambaState"/> and <see cref="IGdnState"/> derive from this base
/// so a single <see cref="IModel.CreateSequenceState"/> factory return type covers either
/// family. Architectures whose recurrent state has no interface yet (Nemotron-H's
/// <c>SsmStateCache</c>) do not participate until they expose one — they report
/// <see cref="IModel.SupportsThreadedSequenceState"/> = <see langword="false"/> and keep
/// the per-sequence model-owned-state forward loop.</para>
/// </remarks>
public interface IRecurrentSequenceState : IDisposable
{
    /// <summary>
    /// Re-zeroes every layer's recurrent state. Call between independent sequences when
    /// reusing a single state container.
    /// </summary>
    void Reset();
}
