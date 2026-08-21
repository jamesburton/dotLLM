using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// One contiguous slice of a model's transformer trunk, executed as a unit by a cycling run.
/// </summary>
/// <param name="FirstLayer">First global layer index in the slice.</param>
/// <param name="LayerCount">Number of layers in the slice.</param>
public readonly record struct LayerWindow(int FirstLayer, int LayerCount)
{
    /// <summary>One past the last global layer index in the slice.</summary>
    public int EndLayer => FirstLayer + LayerCount;

    /// <summary>Renders the slice as <c>[first..end)</c> for logs and result tables.</summary>
    /// <returns>A human-readable half-open range.</returns>
    public override string ToString() => $"[{FirstLayer}..{EndLayer})";
}

/// <summary>
/// Scores perplexity by <em>cycling</em> a device through the model's layers: each layer window
/// replays the whole corpus once, resuming from the previous window's saved boundary activations
/// instead of from token embeddings, so every layer is executed on the fast device exactly once
/// across a single logical corpus pass.
/// </summary>
/// <remarks>
/// <para><b>The problem it solves (issue #395).</b> A 19 GB model cannot be perplexity-tested on a
/// 12 GB card at all with whole-device loading, and the obvious alternative — a fixed CPU/GPU split
/// re-run once per layer window — pays for the CPU half of the model on every pass, so the total
/// cost is <c>N x</c> a CPU-bottlenecked pass. Cycling pays for one device-speed pass plus the
/// checkpoint traffic. Checkpoint volume is <c>windows x context x hidden x 4 B</c>, which for a
/// wikitext-2 sweep at <c>hidden = 2688</c> is a few GB per boundary.</para>
/// <para><b>Correctness rests on two invariants.</b> First, every layer-window pass must enumerate
/// exactly the corpus windows that the final scoring pass does, in the same order, because boundary
/// activations are indexed by window position — hence the shared
/// <see cref="PerplexityWindowPlan"/> and the token cross-check in the replay adapter rather than a
/// re-derived loop. Second, every pass must call <see cref="ILayerWindowExecutor.ResetState"/>
/// before each corpus window, or a recurrent architecture leaks state between windows exactly as in
/// issue #261 — multiplied here, because each pass replays the entire corpus.</para>
/// <para><b>Recurrent state deliberately does not cross a layer boundary.</b> Recurrent state is per
/// layer, and each layer belongs to exactly one layer window that replays the whole corpus in window
/// order. So the state a layer must start corpus window <c>w</c> with is "zero" in a cycled run and
/// in a whole-device run alike, and the checkpoint that has to cross the layer cut is the hidden
/// state only. Serialising a recurrent snapshot into the boundary record would be dead weight that
/// still would not protect against the failure that actually occurs, which is a missing per-window
/// reset. The tests discriminate on the reset, not on a snapshot field.</para>
/// </remarks>
public static class CyclingPerplexityEvaluator
{
    /// <summary>
    /// Reports progress at the start of each layer-window pass.
    /// </summary>
    /// <param name="phaseIndex">Zero-based index of the layer window about to run.</param>
    /// <param name="phaseCount">Total number of layer windows.</param>
    /// <param name="window">The layer slice about to be made resident.</param>
    public delegate void PhaseObserver(int phaseIndex, int phaseCount, LayerWindow window);

    /// <summary>
    /// Partitions <paramref name="numLayers"/> into consecutive windows of at most
    /// <paramref name="windowSize"/> layers.
    /// </summary>
    /// <param name="numLayers">Total transformer layers.</param>
    /// <param name="windowSize">Maximum layers resident at once; the last window may be smaller.</param>
    /// <returns>The partition, in execution order.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Either argument is not positive.</exception>
    public static IReadOnlyList<LayerWindow> PartitionLayers(int numLayers, int windowSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowSize);

        var windows = new List<LayerWindow>();
        for (int first = 0; first < numLayers; first += windowSize)
            windows.Add(new LayerWindow(first, Math.Min(windowSize, numLayers - first)));
        return windows;
    }

    /// <summary>
    /// Scores <paramref name="tokens"/> by cycling <paramref name="model"/> through
    /// <paramref name="layerWindows"/>.
    /// </summary>
    /// <param name="model">Layer-window-capable model. Not owned; not disposed here.</param>
    /// <param name="tokens">Corpus token ids.</param>
    /// <param name="options">Mode and corpus-window geometry — identical in meaning to a whole-device run.</param>
    /// <param name="layerWindows">
    /// Contiguous, ordered, gapless partition of <c>[0, model.NumLayers)</c>. Build one with
    /// <see cref="PartitionLayers"/>.
    /// </param>
    /// <param name="onWindow">Optional per-corpus-window diagnostic callback (<c>--per-window</c>).</param>
    /// <param name="onPhase">Optional per-layer-window progress callback.</param>
    /// <returns>The same <see cref="PerplexityResult"/> shape a whole-device run produces.</returns>
    /// <exception cref="ArgumentException">
    /// The partition is not a gapless cover of the trunk, or the corpus is too short for one window.
    /// </exception>
    public static PerplexityResult Evaluate(
        ILayerWindowModel model,
        ReadOnlySpan<int> tokens,
        PerplexityOptions options,
        IReadOnlyList<LayerWindow> layerWindows,
        PerplexityEvaluator.WindowObserver? onWindow = null,
        PhaseObserver? onPhase = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(layerWindows);
        ValidatePartition(layerWindows, model.NumLayers);

        // The final scoring pass is delegated to PerplexityEvaluator against the SAME options, so
        // the plan built here has to be the plan that evaluator will walk. For sliding-window that
        // is a direct construction. Teacher-forced single-pass scoring is the special case of one
        // window covering the (possibly truncated) corpus with a one-token unscored prefix — row i
        // predicts token i+1 — so it maps onto the same plan type rather than needing a second
        // enumeration path.
        int effectiveContext = Math.Min(options.ContextLength, model.MaxContextLength);
        PerplexityWindowPlan plan;
        int usableTokens;
        if (options.Mode == PerplexityMode.SlidingWindow)
        {
            plan = PerplexityWindowPlan.Create(options, tokens.Length, model.MaxContextLength);
            usableTokens = tokens.Length;
        }
        else
        {
            usableTokens = Math.Min(tokens.Length, effectiveContext);
            plan = PerplexityWindowPlan.Create(
                new PerplexityOptions(PerplexityMode.SlidingWindow, usableTokens, usableTokens, 0, 1, -1),
                usableTokens, usableTokens);
        }

        if (plan.WindowCount == 0)
            throw new ArgumentException(
                $"Corpus of {tokens.Length} tokens is shorter than one context window of {plan.ContextLength}.",
                nameof(tokens));

        int seqLen = plan.ContextLength;
        int hiddenSize = model.HiddenSize;
        int windowCount = plan.WindowCount;

        var positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        var tokenBuffer = new int[seqLen];

        // Two boundaries alive at once: the pass being read and the pass being written. Holding all
        // P boundaries would multiply the (already multi-GB) checkpoint footprint by the number of
        // layer windows for no benefit — a pass only ever needs its immediate predecessor.
        float[][] previous = AllocateBoundary(windowCount, seqLen, hiddenSize);
        float[][] next = AllocateBoundary(windowCount, seqLen, hiddenSize);

        for (int p = 0; p < layerWindows.Count; p++)
        {
            LayerWindow lw = layerWindows[p];
            onPhase?.Invoke(p, layerWindows.Count, lw);

            using ILayerWindowExecutor executor = model.CreateWindow(lw.FirstLayer, lw.LayerCount);
            bool isFirst = lw.FirstLayer == 0;

            for (int w = 0; w < windowCount; w++)
            {
                plan.CopyWindow(tokens, w, tokenBuffer);

                // Per corpus window, in every pass — see ILayerWindowExecutor.ResetState. Skipping
                // this is issue #261 all over again, and it is what the discriminating hybrid test
                // toggles.
                executor.ResetState();
                executor.Run(
                    isFirst ? tokenBuffer : ReadOnlySpan<int>.Empty,
                    isFirst ? ReadOnlySpan<float>.Empty : previous[w],
                    positions,
                    next[w]);
            }

            (previous, next) = (next, previous);
        }

        // Final scoring runs through the ordinary evaluator so that the reported figure is produced
        // by exactly the code a whole-device run uses — window geometry, BOS handling, Welford
        // accumulation and the standard error included. The adapter turns each of the evaluator's
        // Forward calls into "apply the output head to the boundary saved for that window".
        var replay = new BoundaryReplayModel(model, plan, previous, tokens, seqLen);
        PerplexityResult result = options.Mode == PerplexityMode.SlidingWindow
            ? PerplexityEvaluator.Evaluate(replay, tokens, options, onWindow)
            : PerplexityEvaluator.Evaluate(replay, tokens[..usableTokens], options, onWindow);

        replay.AssertFullyConsumed();
        return result;
    }

    private static float[][] AllocateBoundary(int windowCount, int seqLen, int hiddenSize)
    {
        var boundary = new float[windowCount][];
        for (int i = 0; i < windowCount; i++)
            boundary[i] = new float[(long)seqLen * hiddenSize];
        return boundary;
    }

    private static void ValidatePartition(IReadOnlyList<LayerWindow> windows, int numLayers)
    {
        if (windows.Count == 0)
            throw new ArgumentException("At least one layer window is required.", nameof(windows));

        int expected = 0;
        foreach (LayerWindow w in windows)
        {
            if (w.LayerCount <= 0)
                throw new ArgumentException($"Layer window {w} is empty.", nameof(windows));
            if (w.FirstLayer != expected)
                throw new ArgumentException(
                    $"Layer windows must tile [0, {numLayers}) contiguously and in order; " +
                    $"expected a window starting at layer {expected} but got {w}.", nameof(windows));
            expected = w.EndLayer;
        }

        if (expected != numLayers)
            throw new ArgumentException(
                $"Layer windows cover [0, {expected}) but the model has {numLayers} layers; " +
                "cycling must cover every layer or the perplexity is computed from an incomplete trunk.",
                nameof(windows));
    }

    /// <summary>
    /// Replays saved post-trunk boundary activations as if they were forward passes, so the final
    /// scoring pass can be the ordinary <see cref="PerplexityEvaluator"/>.
    /// </summary>
    /// <remarks>
    /// The token cross-check is the guard against the one failure this design is exposed to: the
    /// cycling passes and the scoring pass disagreeing about which corpus window is which. Both
    /// walk the same <see cref="PerplexityWindowPlan"/>, so they cannot disagree today; the check
    /// makes any future divergence throw instead of silently scoring window <c>i</c>'s activations
    /// against window <c>j</c>'s targets.
    /// </remarks>
    private sealed class BoundaryReplayModel : IPerplexityModel
    {
        private readonly ILayerWindowModel _model;
        private readonly PerplexityWindowPlan _plan;
        private readonly float[][] _boundary;
        private readonly int[] _expectedWindow;
        private readonly int[] _tokens;
        private readonly int _seqLen;
        private int _cursor;

        public BoundaryReplayModel(
            ILayerWindowModel model, PerplexityWindowPlan plan, float[][] boundary,
            ReadOnlySpan<int> tokens, int seqLen)
        {
            _model = model;
            _plan = plan;
            _boundary = boundary;
            _seqLen = seqLen;
            _expectedWindow = new int[seqLen];
            _tokens = tokens.ToArray();
        }

        public int VocabSize => _model.VocabSize;

        public int MaxContextLength => _model.MaxContextLength;

        public bool ReturnsAllRows => true;

        public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        {
            if (_cursor >= _boundary.Length)
                throw new InvalidOperationException(
                    $"Scoring requested corpus window {_cursor} but only {_boundary.Length} boundary " +
                    "checkpoints were produced — the cycling passes and the scoring pass disagree " +
                    "about the corpus-window enumeration.");

            _plan.CopyWindow(_tokens, _cursor, _expectedWindow);
            if (tokens.Length != _seqLen || !tokens.SequenceEqual(_expectedWindow))
                throw new InvalidOperationException(
                    $"Corpus window {_cursor} presented for scoring does not match the window whose " +
                    "boundary activations were checkpointed. The cycling passes and the scoring pass " +
                    "must enumerate identical windows.");

            return _model.ApplyOutputHead(_boundary[_cursor++], _seqLen);
        }

        public void ResetState()
        {
            // Nothing to reset: this adapter runs no layers. The recurrent state that matters was
            // reset per corpus window inside each cycling pass.
        }

        public void AssertFullyConsumed()
        {
            if (_cursor != _boundary.Length)
                throw new InvalidOperationException(
                    $"Scoring consumed {_cursor} of {_boundary.Length} boundary checkpoints; the " +
                    "cycling passes evaluated corpus windows the scoring pass did not.");
        }
    }
}
