using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Models.Evaluation;

/// <summary>
/// Adapts any <see cref="IModel"/> to <see cref="IPerplexityModel"/>, so a single evaluator run can
/// score the CPU, CUDA and Vulkan backends without a per-backend harness.
/// </summary>
/// <remarks>
/// <para>Holds a borrowed reference: the adapter does not own the model and does not dispose it, so
/// the caller keeps a single resident copy of the weights.</para>
/// <para><b>Why <see cref="ReturnsAllRows"/> is supplied rather than assumed.</b> Backends disagree
/// on the logits shape: the CPU transformer returns <c>[seqLen, vocab]</c>, while the CUDA model
/// returns only the final row. That flag selects between O(n) single-pass scoring and O(n²)
/// growing-prefix scoring, and getting it wrong produces a *wrong perplexity number* rather than an
/// error — the evaluator would read one row and attribute it to the wrong target. Because it cannot
/// fail loudly, it is passed in explicitly and is best obtained from
/// <see cref="Probe"/> rather than from an assumption about the concrete type.</para>
/// </remarks>
public sealed class BackendPerplexityModel : IPerplexityModel
{
    private readonly IModel _model;
    private readonly int _deviceId;

    /// <param name="model">An already-loaded model. Not owned; not disposed by this adapter.</param>
    /// <param name="deviceId">Device for the forward pass; <c>-1</c> is CPU.</param>
    /// <param name="returnsAllRows">
    /// Whether <paramref name="model"/>'s forward returns logits for every position. Prefer
    /// <see cref="Probe"/> over hardcoding this per backend.
    /// </param>
    public BackendPerplexityModel(IModel model, int deviceId, bool returnsAllRows)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _deviceId = deviceId;
        ReturnsAllRows = returnsAllRows;
    }

    /// <summary>Vocabulary size; the row length of the returned logits.</summary>
    public int VocabSize => _model.Config.VocabSize;

    /// <summary>Maximum token window a single forward accepts.</summary>
    public int MaxContextLength => _model.Config.MaxSequenceLength;

    /// <inheritdoc/>
    public bool ReturnsAllRows { get; }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        => _model.Forward(tokens, positions, _deviceId);

    /// <inheritdoc/>
    public void ResetState() => _model.ResetSequenceState();

    /// <summary>
    /// Determines empirically whether <paramref name="model"/> returns logits for every position, by
    /// running a two-token forward and measuring the returned element count.
    /// </summary>
    /// <param name="model">Model to probe.</param>
    /// <param name="deviceId">Device for the probe forward; <c>-1</c> is CPU.</param>
    /// <returns><see langword="true"/> when the forward returned at least two rows of logits.</returns>
    /// <remarks>
    /// Measuring beats assuming here. The alternative — branching on the concrete model type — bakes
    /// in a fact that lives in each backend's forward implementation and can change without this
    /// adapter noticing, and the failure mode is a silently wrong perplexity rather than an
    /// exception. A two-token probe costs one trivial forward and cannot be wrong about the backend
    /// it just ran.
    /// <para><b>The probe leaves no trace.</b> On a recurrent architecture the probe's forward
    /// advances model-owned recurrent state, so without the reset below the two throwaway tokens
    /// would have been prepended to the very first scored window (issue #261). The reset runs in a
    /// <c>finally</c>: a probe that throws part-way through a forward has still dirtied the state,
    /// and leaving it dirty would corrupt whatever the caller does next.</para>
    /// </remarks>
    public static bool Probe(IModel model, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(model);

        try
        {
            ReadOnlySpan<int> tokens = stackalloc int[2] { 0, 0 };
            ReadOnlySpan<int> positions = stackalloc int[2] { 0, 1 };
            using ITensor logits = model.Forward(tokens, positions, deviceId);
            return logits.ElementCount >= 2L * model.Config.VocabSize;
        }
        finally
        {
            model.ResetSequenceState();
        }
    }
}
