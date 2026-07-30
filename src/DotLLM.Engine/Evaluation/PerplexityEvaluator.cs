using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Computes perplexity over a token sequence using an <see cref="IPerplexityModel"/>.
/// </summary>
/// <remarks>
/// The evaluator never loads weights — callers pass an already-constructed model. On unified-memory
/// parts a large VRAM carve-out leaves host RAM scarce, and perplexity (a long run of full-context
/// prefills) is the workload most punished by holding a second host-side copy of the weights.
/// </remarks>
public static class PerplexityEvaluator
{
    /// <summary>Scores <paramref name="tokens"/> and returns the aggregate result.</summary>
    /// <param name="model">An already-constructed model. Not owned; not disposed here.</param>
    /// <param name="tokens">Token ids to score.</param>
    /// <param name="options">Mode and window geometry.</param>
    public static PerplexityResult Evaluate(
        IPerplexityModel model, ReadOnlySpan<int> tokens, PerplexityOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (tokens.Length < 2)
            throw new ArgumentException("At least two tokens are required to score one target.", nameof(tokens));

        int context = Math.Min(options.ContextLength, model.MaxContextLength);
        if (context < 2)
            throw new ArgumentException("Context length must be at least 2.", nameof(options));

        return options.Mode switch
        {
            PerplexityMode.TeacherForced => EvaluateTeacherForced(model, tokens, context),
            PerplexityMode.SlidingWindow => EvaluateSlidingWindow(model, tokens, context, options.Stride),
            _ => throw new NotSupportedException($"Unknown mode {options.Mode}."),
        };
    }

    private static PerplexityResult EvaluateTeacherForced(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context)
        => model.ReturnsAllRows
            ? TeacherForcedSinglePass(model, tokens, context)
            : TeacherForcedGrowingPrefix(model, tokens, context);

    // Backend returns only the final row, so each target needs its own prefill over the growing
    // prefix. O(n^2) in forward passes — unavoidable, and the reason the CUDA harnesses that
    // originated this methodology carry a stride.
    private static unsafe PerplexityResult TeacherForcedGrowingPrefix(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context)
    {
        int length = Math.Min(tokens.Length, context);
        int vocab = model.VocabSize;
        var positions = new int[length];
        for (int i = 0; i < length; i++) positions[i] = i;

        double sumNll = 0;
        int scored = 0;
        for (int prefix = 1; prefix < length; prefix++)
        {
            using ITensor logits = model.Forward(tokens[..prefix], positions.AsSpan(0, prefix));
            var row = new ReadOnlySpan<float>((void*)logits.DataPointer, vocab);
            sumNll += -LogProb.OfTarget(row, tokens[prefix]);
            scored++;
        }

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, WindowCount: scored);
    }

    // llama.cpp `--perplexity` tiling. Window w covers [w*S, w*S + L) and scores absolute targets
    // in [w*S + L - S, w*S + L), so every scored token carries L-S tokens of context and the
    // scored ranges tile the corpus exactly — no gaps, no double-counting. S = L/2 reproduces
    // llama.cpp's default of scoring the second half of each chunk.
    //
    // Targets before the first window's scored range are never scored. llama.cpp skips them too;
    // "fixing" that would break the comparability this mode exists to provide.
    private static unsafe PerplexityResult EvaluateSlidingWindow(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context, int stride)
    {
        if (stride < 1 || stride >= context)
            throw new ArgumentException(
                $"Stride must be in [1, {context - 1}] for a context of {context}; each scored token needs at least one token of context.",
                nameof(stride));

        if (!model.ReturnsAllRows)
            throw new NotSupportedException(
                "Sliding-window mode requires a backend that returns all rows. Use PerplexityMode.TeacherForced " +
                "for last-row-only backends — re-prefilling per target inside a window would be O(n^2) and is " +
                "already what the growing-prefix path does.");

        int vocab = model.VocabSize;
        var positions = new int[context];
        double sumNll = 0;
        int scored = 0, windows = 0;

        for (int start = 0; start + context <= tokens.Length; start += stride)
        {
            for (int i = 0; i < context; i++) positions[i] = start + i;

            using ITensor logits = model.Forward(tokens.Slice(start, context), positions);
            windows++;

            // Absolute targets [start + context - stride, start + context); row for target t is t-start-1.
            for (int t = start + context - stride; t < start + context; t++)
            {
                int row = t - start - 1;
                var span = new ReadOnlySpan<float>(
                    (void*)(logits.DataPointer + (nint)row * vocab * sizeof(float)), vocab);
                sumNll += -LogProb.OfTarget(span, tokens[t]);
                scored++;
            }
        }

        if (scored == 0)
            throw new ArgumentException(
                $"Corpus of {tokens.Length} tokens is shorter than one context window of {context}.", nameof(tokens));

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, windows);
    }

    // Backend returns every row, so one forward pass scores every target: row i predicts token i+1.
    private static unsafe PerplexityResult TeacherForcedSinglePass(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context)
    {
        int length = Math.Min(tokens.Length, context);
        Span<int> positions = length <= 512 ? stackalloc int[length] : new int[length];
        for (int i = 0; i < length; i++) positions[i] = i;

        double sumNll = 0;
        int scored = 0;

        using ITensor logits = model.Forward(tokens[..length], positions);
        int vocab = model.VocabSize;
        // Row i predicts token i+1, so the final row has no target within the window.
        for (int i = 0; i < length - 1; i++)
        {
            var row = new ReadOnlySpan<float>(
                (void*)(logits.DataPointer + (nint)i * vocab * sizeof(float)), vocab);
            sumNll += -LogProb.OfTarget(row, tokens[i + 1]);
            scored++;
        }

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, WindowCount: 1);
    }
}
