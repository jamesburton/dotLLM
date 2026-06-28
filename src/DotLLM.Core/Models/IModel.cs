using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.Models;

/// <summary>
/// A loaded, ready-to-run transformer model.
/// </summary>
public interface IModel : IDisposable
{
    /// <summary>Model configuration.</summary>
    ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference compute scratch buffers.</summary>
    long ComputeMemoryBytes { get; }

    /// <summary>
    /// Runs a forward pass through the model.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this batch.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <returns>Logits tensor of shape [batch, vocab_size].</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId);

    /// <summary>
    /// Runs a forward pass with optional KV-cache for efficient autoregressive decoding.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <returns>Logits tensor of shape [1, vocab_size] for the last token.</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache);

    /// <summary>
    /// Runs a forward pass with optional KV-cache and an optional LoRA adapter.
    /// When <paramref name="adapter"/> is non-null and supplies <c>(layer, proj)</c>
    /// factor pairs that match the current model's projection sites, the runtime
    /// adds the LoRA delta <c>alpha × (x · B) · A</c> to each adapted projection.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <param name="adapter">
    /// Optional LoRA adapter. When null, behaves byte-equivalently to the
    /// adapter-less <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// overload (default implementation forwards to it).
    /// </param>
    /// <returns>Logits tensor of shape [seq, vocab_size] for all input positions.</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                    IKvCache? kvCache, ILoraAdapter? adapter)
        => Forward(tokenIds, positions, deviceId, kvCache);

    /// <summary>
    /// Runs a forward pass with an explicit attention-mask mode (causal / bidirectional / hybrid).
    /// </summary>
    /// <remarks>
    /// <para>The default implementation forwards to
    /// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, ILoraAdapter?)"/>
    /// when <paramref name="maskSpec"/> is the causal default, so EVERY existing implementor keeps the
    /// byte-identical autoregressive behaviour with no code change. Implementations that do not support
    /// non-causal masking throw <see cref="NotSupportedException"/> for the non-causal modes via this
    /// default — only backends that implement bidirectional / hybrid attention (currently the CPU
    /// <c>TransformerModel</c>) override this method.</para>
    /// </remarks>
    /// <param name="tokenIds">Input token IDs for this step.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <param name="adapter">Optional LoRA adapter. When null, behaves like the adapter-less overload.</param>
    /// <param name="maskSpec">Attention-mask mode. Defaults to <see cref="AttentionMaskSpec.Causal"/>.</param>
    /// <returns>Logits tensor of shape [seq, vocab_size] for all input positions.</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                    IKvCache? kvCache, ILoraAdapter? adapter, AttentionMaskSpec maskSpec)
    {
        if (!maskSpec.IsCausal)
            throw new NotSupportedException(
                $"{GetType().Name} does not support non-causal attention (mask mode {maskSpec.Mode}). " +
                "Only the CPU TransformerModel currently implements bidirectional / hybrid attention.");
        return Forward(tokenIds, positions, deviceId, kvCache, adapter);
    }

    /// <summary>
    /// Supplies the DiffusionGemma self-conditioning (SC) state consumed by the NEXT
    /// <c>Forward</c> over a <see cref="AttentionMaskSpec.Hybrid(int)"/> canvas: the
    /// PREVIOUS denoise step's canvas-region logits (post-softcap) plus the SC gate.
    /// </summary>
    /// <remarks>
    /// <para>On a diffusion-gemma model with <c>scUse &gt; 0</c> and a non-empty
    /// <paramref name="prevCanvasLogits"/>, the next forward replaces the canvas region's
    /// plain <c>rms_noscale(scaled_embed)</c> with <c>rms_noscale(scaled_embed + sc_sig)</c>,
    /// where <c>sc_sig</c> is a gated GeGLU MLP over a soft token-embedding of the previous
    /// logits (the trained self-conditioning denoiser feedback). On step 0 of a canvas the
    /// generator passes <c>scUse = 0</c> (and may pass an empty span), reproducing the
    /// zero-SC first-step behaviour byte-for-byte.</para>
    /// <para>The default implementation is a no-op — only the CPU <c>TransformerModel</c>
    /// implements diffusion self-conditioning. Single-threaded per generation (the model's
    /// forward state is instance-scoped), matching the existing mask/adapter state contract.</para>
    /// </remarks>
    /// <param name="prevCanvasLogits">Previous step's canvas-region logits, row-major
    /// <c>[canvasLen, vocabSize]</c> (post-softcap). Empty when <paramref name="scUse"/> is 0.</param>
    /// <param name="canvasLen">Number of canvas rows in <paramref name="prevCanvasLogits"/>.</param>
    /// <param name="scUse">Self-conditioning gate: 0 on the first denoise step (zero-SC), 1 thereafter.</param>
    void SetDiffusionSelfCond(ReadOnlySpan<float> prevCanvasLogits, int canvasLen, float scUse) { }

    /// <summary>
    /// True when this model implements the DiffusionGemma prompt-KV (PKV) prefill/decode
    /// cache — the throughput optimisation that captures the prompt's per-layer K/V once
    /// (<see cref="DiffusionPrefillPromptKv"/>) and reuses them on every denoise step
    /// (<see cref="DiffusionDecodeWithPromptKv"/>) instead of recomputing the prompt prefix.
    /// Default false: callers fall back to the cacheless unified <c>[prompt | canvas]</c> forward.
    /// </summary>
    bool SupportsDiffusionPromptKv => false;

    /// <summary>
    /// PKV <b>prefill</b>: runs a prompt-only causal forward (over the <paramref name="promptTokens"/>)
    /// and captures each transformer layer's post-norm/post-rope prompt <c>K</c> and weight-less-normed
    /// prompt <c>V</c> into <paramref name="store"/> for reuse by <see cref="DiffusionDecodeWithPromptKv"/>.
    /// </summary>
    /// <remarks>
    /// <para>The captured K/V are byte-equivalent to the prompt rows the cacheless unified forward would
    /// compute (the prompt embedding is fixed across denoise steps, and K/V projections do not depend on
    /// the attention mask). On V-less global layers <c>V</c> is the raw <c>K</c> projection, exactly as
    /// the unified path. The default implementation throws — only models reporting
    /// <see cref="SupportsDiffusionPromptKv"/> implement it.</para>
    /// </remarks>
    /// <param name="promptTokens">Prompt token ids (the causal prefix to cache).</param>
    /// <param name="positions">Position indices for the prompt tokens (typically <c>0..P-1</c>).</param>
    /// <param name="store">Destination prompt-KV store; resized and filled for the current prompt.</param>
    void DiffusionPrefillPromptKv(
        ReadOnlySpan<int> promptTokens, ReadOnlySpan<int> positions, DiffusionPromptKvStore store)
        => throw new NotSupportedException(
            $"{GetType().Name} does not support DiffusionGemma prompt-KV prefill.");

    /// <summary>
    /// PKV <b>decode</b>: runs a canvas-only forward (over the <paramref name="canvasTokens"/>, length C)
    /// that, for each layer, computes fresh canvas Q/K/V and attends over the concatenation
    /// <c>[cached prompt K/V | fresh canvas K/V]</c> under a rectangular bidirectional mask
    /// (a canvas query attends all prompt keys + all canvas keys, clipped by the per-layer sliding
    /// window). The canvas region embedding (weight-less rms + self-conditioning) and the canvas
    /// <c>layer_output_scale</c> apply to all C rows exactly as the unified forward.
    /// </summary>
    /// <remarks>
    /// <para>Produces canvas logits byte-equivalent to the cacheless unified forward's canvas rows — a
    /// pure optimisation. The canvas RoPE positions are <c>promptLen + 0 .. promptLen + C - 1</c>
    /// (supplied via <paramref name="positions"/>). Self-conditioning is supplied beforehand via
    /// <see cref="SetDiffusionSelfCond"/>, identical to the unified path. The default implementation
    /// throws — only models reporting <see cref="SupportsDiffusionPromptKv"/> implement it.</para>
    /// </remarks>
    /// <param name="canvasTokens">Canvas token ids for this step (length C).</param>
    /// <param name="positions">Canvas RoPE positions (<c>promptLen .. promptLen+C-1</c>).</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="store">Prompt-KV store filled by a prior <see cref="DiffusionPrefillPromptKv"/>.</param>
    /// <returns>Logits tensor of shape <c>[C, vocabSize]</c> for the canvas positions.</returns>
    ITensor DiffusionDecodeWithPromptKv(
        ReadOnlySpan<int> canvasTokens, ReadOnlySpan<int> positions, int deviceId,
        DiffusionPromptKvStore store)
        => throw new NotSupportedException(
            $"{GetType().Name} does not support DiffusionGemma prompt-KV decode.");

    /// <summary>
    /// True when this model's per-sequence forward carries recurrent state (SSM / linear-attention)
    /// that the caller must supply per request, beyond the KV-cache. Default <c>false</c>.
    /// </summary>
    /// <remarks>
    /// Recurrent architectures (Mamba-3, Qwen3-MoE-Hybrid GDN, Nemotron-H) require a per-sequence
    /// <see cref="SequenceForwardRequest.MambaState"/> / <see cref="SequenceForwardRequest.GdnState"/>
    /// on every request when <see cref="ForwardBatch"/> is called with 2+ entries (a null state
    /// throws or corrupts cross-sequence recurrence). The continuous-batch scheduler does not yet
    /// allocate/thread that state, so it gates batched decode on this flag: <c>false</c> ⇒ a stateless
    /// (dense KV-only) model whose decode can be safely fused via <see cref="ForwardBatch"/>;
    /// <c>true</c> ⇒ keep the per-sequence decode loop. Threading recurrent state to lift this is a
    /// follow-up.
    /// </remarks>
    bool RequiresPerSequenceState => false;

    /// <summary>
    /// True when this model can have its per-sequence recurrent state <em>threaded by the caller</em>:
    /// the caller allocates one container per sequence via <see cref="CreateSequenceState"/> and supplies
    /// it on every <see cref="ForwardBatch"/> request (via <see cref="SequenceForwardRequest.MambaState"/> /
    /// <see cref="SequenceForwardRequest.GdnState"/>). Default <c>false</c>.
    /// </summary>
    /// <remarks>
    /// <para>For a recurrent model (<see cref="RequiresPerSequenceState"/> = <c>true</c>) this is what lets
    /// the continuous-batch scheduler fuse its decode/prefill via <see cref="ForwardBatch"/> instead of the
    /// per-sequence loop: when <c>true</c>, the scheduler allocates and threads a state per sequence and
    /// dispatches everything through <see cref="ForwardBatch"/> (the only entrypoint that carries the state),
    /// which also fixes the latent cross-sequence corruption of running &gt;1 concurrent recurrent sequence
    /// against a shared model-owned default state. When <c>false</c> on a recurrent model (e.g. Nemotron-H,
    /// whose SSM state has no <see cref="IRecurrentSequenceState"/> container yet), the scheduler keeps the
    /// per-sequence forward loop. A non-recurrent (dense) model leaves this <c>false</c> and ignores it.</para>
    /// </remarks>
    bool SupportsThreadedSequenceState => false;

    /// <summary>
    /// Allocates a fresh, zero-initialised per-sequence recurrent-state container for this model, or
    /// <see langword="null"/> when the model carries no caller-threadable recurrent state.
    /// </summary>
    /// <remarks>
    /// Returns non-null exactly when <see cref="SupportsThreadedSequenceState"/> is <c>true</c> — a
    /// recurrent host returns its concrete <see cref="IMambaState"/> / <see cref="IGdnState"/> sized for
    /// the model's recurrent layers. The caller owns the returned container's lifetime (dispose when the
    /// sequence finishes) and supplies it on that sequence's <see cref="ForwardBatch"/> requests. The
    /// default implementation returns <see langword="null"/>.
    /// </remarks>
    IRecurrentSequenceState? CreateSequenceState() => null;

    /// <summary>
    /// Runs a fused forward pass across multiple in-flight sequences.
    /// </summary>
    /// <remarks>
    /// <para>The continuous-batch scheduler calls this once per iteration when 2+ sequences are
    /// active, instead of looping <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, ILoraAdapter?)"/>
    /// per sequence. Each <paramref name="requests"/> entry carries its own tokens, positions,
    /// and KV-cache — sequences are independent at the attention level (no cross-sequence
    /// attention).</para>
    /// <para>The default implementation simply loops over <c>Forward</c> per request and returns
    /// the results in input order. Implementations can override to fuse the per-sequence GEMVs
    /// into batched GEMMs and avoid the per-iteration kernel-dispatch overhead — this is the
    /// principal continuous-batching throughput win.</para>
    /// <para>The returned tensors follow the same shape contract as <c>Forward</c>: each entry
    /// is <c>[N_i, vocab_size]</c> where <c>N_i</c> matches that request's token count (CPU
    /// model) or <c>[1, vocab_size]</c> for the last token only (GPU/hybrid). The caller is
    /// responsible for disposing each returned tensor.</para>
    /// </remarks>
    /// <param name="requests">One entry per active sequence. Order is preserved in the result.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <returns>Logits tensors, one per request, in the same order.</returns>
    IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter);
        }
        return results;
    }
}
