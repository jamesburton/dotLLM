using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;

namespace DotLLM.Models.Evaluation;

/// <summary>
/// Exposes an already-loaded CPU model as an <see cref="ILayerWindowModel"/>, so the cycling
/// perplexity driver can be exercised — and its numbers validated against a whole-model run — without
/// a device backend.
/// </summary>
/// <remarks>
/// <para><b>Ownership.</b> Holds a borrowed reference: this class does not own the model and does not
/// dispose it, so the caller keeps a single resident copy of the weights — the same contract as
/// <see cref="BackendPerplexityModel"/>. <see cref="Dispose"/> is therefore a no-op that exists only
/// to satisfy the interface.</para>
/// <para><b>Every layer stays resident.</b> CPU weights are memory-mapped, so there is nothing to
/// allocate when a window opens and nothing to free when it closes: <see cref="CreateWindow"/> hands
/// back a cheap view that records the layer range and forwards to the model. This is precisely the
/// case the <see cref="ILayerWindowModel"/> factory shape was written to allow. A device
/// implementation of the same interface behaves the opposite way — it uploads that window's weights
/// on creation and frees them on disposal, which is the whole reason cycling exists — so callers must
/// not read this class's free disposal as licence to hold several windows of a device-backed factory
/// alive at once.</para>
/// <para><b>The reference oracle for cycling.</b> Because the CPU trunk is windowed by restricting the
/// one shared layer loop, and the output head is the same shared tail the whole-model forward runs,
/// cycling a CPU model reproduces the un-cycled logits to floating-point noise. That equality is what
/// makes this class useful as a test oracle for the device implementations.</para>
/// </remarks>
public sealed class CpuLayerWindowModel : ILayerWindowModel
{
    private readonly IModel _model;
    private readonly ModelConfig _config;

    /// <summary>
    /// Wraps an already-loaded CPU model.
    /// </summary>
    /// <param name="model">
    /// A loaded CPU model of a windowable architecture. Not owned; not disposed by this class.
    /// </param>
    /// <param name="config">That model's configuration — trunk depth, hidden size, vocabulary.</param>
    /// <exception cref="ArgumentNullException">Either argument is <see langword="null"/>.</exception>
    /// <exception cref="NotSupportedException">
    /// <paramref name="model"/> is an architecture with no windowed forward.
    /// </exception>
    public CpuLayerWindowModel(IModel model, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);

        // Fail at construction rather than at the first window run: the caller has typically just
        // planned a multi-hour corpus sweep by this point.
        if (model is not (Qwen3MoeHybridTransformerModel or TransformerModel))
            throw Unsupported(model);

        _model = model;
        _config = config;
    }

    /// <inheritdoc/>
    public int NumLayers => _config.NumLayers;

    /// <inheritdoc/>
    public int HiddenSize => _config.HiddenSize;

    /// <inheritdoc/>
    public int VocabSize => _config.VocabSize;

    /// <inheritdoc/>
    public int MaxContextLength => _config.MaxSequenceLength;

    /// <inheritdoc/>
    public ILayerWindowExecutor CreateWindow(int firstLayer, int layerCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstLayer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        if (firstLayer + layerCount > NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerCount),
                $"Layer window [{firstLayer}..{firstLayer + layerCount}) falls outside the "
                + $"{NumLayers}-layer trunk.");

        return new Window(_model, firstLayer, layerCount);
    }

    /// <inheritdoc/>
    public ITensor ApplyOutputHead(ReadOnlySpan<float> hidden, int seqLen) => _model switch
    {
        Qwen3MoeHybridTransformerModel hybrid => hybrid.ApplyOutputHead(hidden, seqLen, deviceId: -1),
        TransformerModel dense => dense.ApplyOutputHead(hidden, seqLen, deviceId: -1),
        _ => throw Unsupported(_model),
    };

    /// <summary>Releases nothing — the model is borrowed and every layer is permanently resident.</summary>
    public void Dispose()
    {
        // Intentionally empty: see the ownership note in the type remarks.
    }

    /// <summary>
    /// Builds the rejection thrown for an architecture with no windowed CPU forward, naming it so the
    /// message is actionable rather than "unsupported model".
    /// </summary>
    private static NotSupportedException Unsupported(IModel model) => new(
        $"CpuLayerWindowModel does not support {model.GetType().Name}. Layer cycling requires a "
        + "windowed forward (ForwardLayerWindow); today that exists for "
        + $"{nameof(Qwen3MoeHybridTransformerModel)} and {nameof(TransformerModel)}.");

    /// <summary>
    /// One resident layer range. Cheap by construction — the model already holds every layer, so this
    /// carries nothing but the range and the borrowed model reference.
    /// </summary>
    private sealed class Window(IModel model, int firstLayer, int layerCount) : ILayerWindowExecutor
    {
        /// <inheritdoc/>
        public int FirstLayer => firstLayer;

        /// <inheritdoc/>
        public int LayerCount => layerCount;

        /// <inheritdoc/>
        public void Run(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> hiddenIn,
                        ReadOnlySpan<int> positions, Span<float> hiddenOut)
        {
            switch (model)
            {
                case Qwen3MoeHybridTransformerModel hybrid:
                    hybrid.ForwardLayerWindow(tokenIds, hiddenIn, positions, firstLayer, layerCount, hiddenOut);
                    break;
                case TransformerModel dense:
                    dense.ForwardLayerWindow(tokenIds, hiddenIn, positions, firstLayer, layerCount, hiddenOut);
                    break;
                default:
                    throw Unsupported(model);
            }
        }

        /// <inheritdoc/>
        public void ResetState() => model.ResetSequenceState();

        /// <summary>Releases nothing — CPU layers are mmap'd and stay resident between windows.</summary>
        public void Dispose()
        {
            // Intentionally empty: see CpuLayerWindowModel's remarks.
        }
    }
}
