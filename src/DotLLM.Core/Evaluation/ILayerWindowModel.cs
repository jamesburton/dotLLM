using DotLLM.Core.Tensors;

namespace DotLLM.Core.Evaluation;

/// <summary>
/// A model that can execute an arbitrary <em>contiguous window</em> of its transformer layers,
/// entering from either token ids (window starts at layer 0) or a boundary hidden state, and
/// leaving the residual stream at the window's far edge.
/// </summary>
/// <remarks>
/// <para><b>Why this exists (issue #395).</b> Perplexity for a model larger than a device's memory
/// cannot be measured by loading the whole model onto that device. It can be measured by cycling:
/// run corpus pass 1 with layers <c>[0..k)</c> resident, saving the hidden state at the cut for
/// every scored window; then load <c>[k..2k)</c> and replay from those saved activations rather
/// than from token embeddings; repeat to the final window. Every layer is then device-executed
/// exactly once across a single logical corpus pass, instead of <c>N</c> passes each bottlenecked by
/// the half of the model that did not fit.</para>
/// <para><b>Why the factory shape.</b> The whole point is that only one window's weights are
/// resident at a time, so windows are <em>created and destroyed</em> rather than selected on a model
/// that already holds everything. A CPU implementation may legitimately return cheap views over one
/// resident mmap'd copy; a device implementation uploads that window's weights and frees them on
/// <see cref="IDisposable.Dispose"/>. Callers must not hold two windows of a device-backed factory
/// alive at once unless they have accounted for the memory.</para>
/// <para><b>The output head is on the factory, not the last window.</b> Applying the final norm and
/// LM head separately keeps the layer windows uniform (every window is hidden-in/hidden-out) and
/// lets the head produce logits for <em>every</em> row — which sliding-window scoring requires and
/// which the device forward paths, optimised for the last row only, do not give. The head is a
/// single GEMM against a weight that is negligible next to a layer window, so hosting it outside the
/// device window costs little and buys all-rows scoring.</para>
/// </remarks>
public interface ILayerWindowModel : IDisposable
{
    /// <summary>Total transformer layers in the model (the trunk that windows partition).</summary>
    int NumLayers { get; }

    /// <summary>Residual-stream width; the row length of every boundary hidden state.</summary>
    int HiddenSize { get; }

    /// <summary>Vocabulary size; the row length of the logits <see cref="ApplyOutputHead"/> returns.</summary>
    int VocabSize { get; }

    /// <summary>Maximum token window a single forward accepts.</summary>
    int MaxContextLength { get; }

    /// <summary>
    /// Makes layers <c>[firstLayer, firstLayer + layerCount)</c> executable, allocating whatever
    /// device residency that requires.
    /// </summary>
    /// <param name="firstLayer">First global layer index in the window.</param>
    /// <param name="layerCount">Number of layers in the window; at least one.</param>
    /// <returns>An executor for that window. The caller owns and disposes it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The window falls outside <see cref="NumLayers"/>.</exception>
    ILayerWindowExecutor CreateWindow(int firstLayer, int layerCount);

    /// <summary>
    /// Applies the final norm and LM head to a post-trunk hidden state, producing logits for every
    /// row.
    /// </summary>
    /// <param name="hidden">Row-major <c>[seqLen, HiddenSize]</c> hidden state leaving the last layer.</param>
    /// <param name="seqLen">Number of rows in <paramref name="hidden"/>.</param>
    /// <returns>
    /// A caller-owned <c>[seqLen, VocabSize]</c> FP32 logits tensor on the host.
    /// </returns>
    ITensor ApplyOutputHead(ReadOnlySpan<float> hidden, int seqLen);
}

/// <summary>
/// Executes one contiguous layer window of an <see cref="ILayerWindowModel"/>.
/// </summary>
public interface ILayerWindowExecutor : IDisposable
{
    /// <summary>First global layer index this executor covers.</summary>
    int FirstLayer { get; }

    /// <summary>Number of layers this executor covers.</summary>
    int LayerCount { get; }

    /// <summary>
    /// Runs the window over one sequence and writes the residual stream leaving its last layer.
    /// </summary>
    /// <param name="tokenIds">
    /// Token ids, used only when <see cref="FirstLayer"/> is 0 (the window owns the embedding).
    /// Ignored otherwise and may be empty.
    /// </param>
    /// <param name="hiddenIn">
    /// Row-major <c>[seqLen, HiddenSize]</c> boundary hidden state to resume from, required when
    /// <see cref="FirstLayer"/> is greater than 0. Empty for the first window.
    /// </param>
    /// <param name="positions">Absolute position id per row; its length defines <c>seqLen</c>.</param>
    /// <param name="hiddenOut">
    /// Destination for the <c>[seqLen, HiddenSize]</c> residual stream leaving the window.
    /// </param>
    void Run(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> hiddenIn, ReadOnlySpan<int> positions,
             Span<float> hiddenOut);

    /// <summary>
    /// Re-zeroes any recurrent (Mamba/SSM conv + state, Gated-DeltaNet) state this window's layers
    /// carry, so the next <see cref="Run"/> starts a genuinely fresh sequence.
    /// </summary>
    /// <remarks>
    /// <para><b>This is load-bearing, not hygiene.</b> Perplexity treats every corpus window as an
    /// independent sequence. A recurrent layer's uncached forward advances model-owned state that
    /// survives the call, so without this the second corpus window inherits the first's recurrence
    /// and the reported number is wrong with nothing raised — issue #261, found in exactly this
    /// code path. Cycling multiplies the exposure: each layer window replays the whole corpus, so a
    /// single missed reset corrupts every subsequent window of that pass.</para>
    /// <para>Recurrent state is <em>not</em> carried across a layer-window boundary and does not
    /// belong in the boundary checkpoint: state is per layer, and a given layer is owned by exactly
    /// one window which replays the entire corpus in window order. What a layer's state must equal
    /// at the start of corpus window <c>w</c> is therefore "zero", identically in a cycled run and
    /// in a whole-device run — provided this method is called before every corpus window. That
    /// equality is the invariant the cycling driver enforces and the tests discriminate on.</para>
    /// </remarks>
    void ResetState();
}
