using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Assigns each layer window of one model to a specific backing <see cref="ILayerWindowModel"/>, so a
/// single evaluation can put an arbitrary contiguous slice of the trunk on the GPU and the rest on
/// the CPU.
/// </summary>
/// <remarks>
/// <para><b>Why a composition and not a new hybrid model class (issue #395).</b> The existing
/// partial-offload models split the trunk exactly once, at a <em>prefix</em> boundary: GPU runs
/// <c>[0..n)</c>, CPU runs the rest. That cannot express "put layers <c>[12..24)</c> on the GPU",
/// which is what full-coverage verification of a model larger than the device needs — you have to be
/// able to slide the device window over every part of the trunk. Because every layer window already
/// exposes the same hidden-in/hidden-out contract regardless of device, an arbitrary window is just
/// a three-entry assignment (<c>CPU [0..k)</c>, <c>GPU [k..k+n)</c>, <c>CPU [k+n..L)</c>) over that
/// one contract, rather than a fourth hand-written CPU/GPU splitter to keep in sync with the other
/// three.</para>
/// <para>The output head is taken from a nominated backing model — normally the CPU one, whose head
/// produces logits for every row.</para>
/// </remarks>
public sealed class CompositeLayerWindowModel : ILayerWindowModel
{
    private readonly IReadOnlyList<LayerAssignment> _assignments;
    private readonly ILayerWindowModel _headModel;
    private readonly bool _ownsBackings;

    /// <summary>Binds a layer slice to the model that executes it.</summary>
    /// <param name="Window">The layer slice.</param>
    /// <param name="Model">Backing model that owns those layers' weights.</param>
    public readonly record struct LayerAssignment(LayerWindow Window, ILayerWindowModel Model);

    /// <summary>
    /// Composes a model from per-window device assignments.
    /// </summary>
    /// <param name="assignments">
    /// Contiguous, ordered, gapless cover of <c>[0, headModel.NumLayers)</c>. Each entry names the
    /// backing model that holds those layers.
    /// </param>
    /// <param name="headModel">Backing model whose output head produces the logits.</param>
    /// <param name="ownsBackings">
    /// When <see langword="true"/>, <see cref="Dispose"/> disposes every distinct backing model.
    /// </param>
    /// <exception cref="ArgumentException">The assignments do not cover the trunk exactly once.</exception>
    public CompositeLayerWindowModel(
        IReadOnlyList<LayerAssignment> assignments, ILayerWindowModel headModel, bool ownsBackings = false)
    {
        ArgumentNullException.ThrowIfNull(assignments);
        ArgumentNullException.ThrowIfNull(headModel);
        if (assignments.Count == 0)
            throw new ArgumentException("At least one layer assignment is required.", nameof(assignments));

        int expected = 0;
        foreach (LayerAssignment a in assignments)
        {
            if (a.Model is null)
                throw new ArgumentException($"Layer window {a.Window} has no backing model.", nameof(assignments));
            if (a.Window.LayerCount <= 0 || a.Window.FirstLayer != expected)
                throw new ArgumentException(
                    $"Layer assignments must tile [0, {headModel.NumLayers}) contiguously and in order; " +
                    $"expected a window starting at layer {expected} but got {a.Window}.", nameof(assignments));
            expected = a.Window.EndLayer;
        }

        if (expected != headModel.NumLayers)
            throw new ArgumentException(
                $"Layer assignments cover [0, {expected}) but the model has {headModel.NumLayers} layers.",
                nameof(assignments));

        _assignments = assignments;
        _headModel = headModel;
        _ownsBackings = ownsBackings;
    }

    /// <inheritdoc/>
    public int NumLayers => _headModel.NumLayers;

    /// <inheritdoc/>
    public int HiddenSize => _headModel.HiddenSize;

    /// <inheritdoc/>
    public int VocabSize => _headModel.VocabSize;

    /// <inheritdoc/>
    public int MaxContextLength => _headModel.MaxContextLength;

    /// <summary>The device assignment, in trunk order. Surfaced so a run can report its own split.</summary>
    public IReadOnlyList<LayerAssignment> Assignments => _assignments;

    /// <summary>The layer windows of this composition, in trunk order.</summary>
    /// <returns>The windows to hand to <see cref="CyclingPerplexityEvaluator.Evaluate"/>.</returns>
    public IReadOnlyList<LayerWindow> Windows()
    {
        var windows = new LayerWindow[_assignments.Count];
        for (int i = 0; i < _assignments.Count; i++) windows[i] = _assignments[i].Window;
        return windows;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The requested window must match an assignment exactly. Splitting across two backing models
    /// would silently run part of the range on the wrong device, so it throws instead.
    /// </remarks>
    public ILayerWindowExecutor CreateWindow(int firstLayer, int layerCount)
    {
        foreach (LayerAssignment a in _assignments)
        {
            if (a.Window.FirstLayer == firstLayer && a.Window.LayerCount == layerCount)
                return a.Model.CreateWindow(firstLayer, layerCount);
        }

        throw new ArgumentOutOfRangeException(nameof(firstLayer),
            $"No backing model is assigned exactly the layer window [{firstLayer}..{firstLayer + layerCount}). " +
            $"Assigned windows are: {string.Join(", ", _assignments.Select(a => a.Window.ToString()))}.");
    }

    /// <inheritdoc/>
    public ITensor ApplyOutputHead(ReadOnlySpan<float> hidden, int seqLen)
        => _headModel.ApplyOutputHead(hidden, seqLen);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_ownsBackings) return;

        // Distinct: several windows normally share one backing model.
        foreach (ILayerWindowModel model in _assignments.Select(a => a.Model).Distinct())
            model.Dispose();
        if (!_assignments.Any(a => ReferenceEquals(a.Model, _headModel)))
            _headModel.Dispose();
    }
}
